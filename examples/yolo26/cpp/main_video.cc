#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include "yolo26.h"            // 依赖你的 YOLOv26 模型封装
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"

#include "utils/mpp_decoder.h"
#include "utils/mpp_encoder.h"
#include "utils/drawing.h"

#if defined(BUILD_VIDEO_RTSP)
#include "mk_mediakit.h"
#endif

#define OUT_VIDEO_PATH "out_yolo26.h264"

// 全局上下文：将 YOLO 的上下文和视频流上下文打包
typedef struct {
    rknn_app_context_t yolo_app_ctx;
    MppDecoder *decoder;
    MppEncoder *encoder;
    FILE *out_fp;
    char *enc_data;        // 【新增】持久化编码 buffer 指针
    int enc_buf_size;      // 【新增】记录分配的 buffer 大小
} video_app_context_t;

/*-------------------------------------------
                  Functions
-------------------------------------------*/
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *read_file_data(const char *filename, int *model_size) {
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(size);
    fread(data, 1, size, fp);
    fclose(fp);
    *model_size = size;
    return data;
}

// 核心回调函数：MPP 解码出一帧后自动触发
void mpp_decoder_frame_callback(void *userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data) {
    video_app_context_t *ctx = (video_app_context_t *)userdata;
    int ret = 0;
    static int frame_index = 0;
    frame_index++;

    void *mpp_frame = NULL;
    int mpp_frame_fd = 0;
    void *mpp_frame_addr = NULL;
    int enc_data_size;

    rga_buffer_t origin;
    rga_buffer_t src;

    // 1. 初始化编码器 (仅在第一帧时)
    if (ctx->encoder == NULL) {
        MppEncoder *mpp_encoder = new MppEncoder();
        MppEncoderParams enc_params;
        memset(&enc_params, 0, sizeof(MppEncoderParams));
        enc_params.width = width;
        enc_params.height = height;
        enc_params.hor_stride = width_stride;
        enc_params.ver_stride = height_stride;
        enc_params.fmt = MPP_FMT_YUV420SP;
        enc_params.type = MPP_VIDEO_CodingAVC; // 默认 H264
        mpp_encoder->Init(enc_params, NULL);
        ctx->encoder = mpp_encoder;
        // 【核心改进】：在这里只分配一次内存
        ctx->enc_buf_size = ctx->encoder->GetFrameSize();
        ctx->enc_data = (char *)malloc(ctx->enc_buf_size);
    }

    // 直接使用上下文中预分配的内存
    char *enc_data = ctx->enc_data; 
    int enc_buf_size = ctx->enc_buf_size;

    // 2. 构造 Image Buffer 丢给 YOLOv26 推理
    // 【核心改进】：直接使用 MPP 硬件解码出来的 DMA fd，不发生 CPU 拷贝
    image_buffer_t img;
    memset(&img, 0, sizeof(image_buffer_t));
    img.width = width;
    img.height = height;
    img.width_stride = width_stride;
    img.height_stride = height_stride;
    img.fd = fd;                          // 直接传递物理内存文件描述符
    img.virt_addr = (unsigned char *)data;
    img.format = IMAGE_FORMAT_YUV420SP;   // 确认 zoo 里的 image_utils.h 支持该枚举
    img.size = width_stride * height_stride * 3 / 2;

    object_detect_result_list detect_result;
    memset(&detect_result, 0, sizeof(object_detect_result_list));

    // 执行推理 (内部应当调用 RGA 进行零拷贝 Resize)
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);
    ret = inference_yolo26_model(&(ctx->yolo_app_ctx), &img, &detect_result);
    gettimeofday(&stop_time, NULL);
    if (frame_index % 30 == 0) { // 每30帧打印一次耗时，避免日志刷屏
        printf("Frame %d inference time: %f ms\n", frame_index, (__get_us(stop_time) - __get_us(start_time)) / 1000);
    }

    if (ret != 0) {
        printf("inference_yolo26_model fail\n");
        goto RET;
    }

    // 3. RGA 内存拷贝 (保护原始解码 Buffer)
    mpp_frame = ctx->encoder->GetInputFrameBuffer();
    mpp_frame_fd = ctx->encoder->GetInputFrameBufferFd(mpp_frame);
    mpp_frame_addr = ctx->encoder->GetInputFrameBufferAddr(mpp_frame);

    origin = wrapbuffer_fd(fd, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
    src = wrapbuffer_fd(mpp_frame_fd, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
    imcopy(origin, src); // RGA 硬件极速拷贝

    // 4. 直接在拷贝后的 YUV 显存上画框
    for (int i = 0; i < detect_result.count; i++) {
        object_detect_result *det_result = &(detect_result.results[i]);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        
        // 使用针对 YUV 优化的绘制函数，比转 RGB 再绘制快非常多
        draw_rectangle_yuv420sp((unsigned char *)mpp_frame_addr, width_stride, height_stride, 
                                x1, y1, x2 - x1 + 1, y2 - y1 + 1, 0x00FF0000, 4);
        
        // 如果需要在 YUV 上写字，可以扩展你的 drawing.h，或者牺牲性能在这里转 RGB
        // printf("Detected: %s %.2f\n", coco_cls_to_name(det_result->cls_id), det_result->prop);
    }

    // 5. 硬件编码保存
    if (frame_index == 1) {
        int header_size = ctx->encoder->GetHeader(enc_data, enc_buf_size);
        fwrite(enc_data, 1, header_size, ctx->out_fp);
    }
    
    memset(enc_data, 0, enc_buf_size);
    int enc_data_size = ctx->encoder->Encode(mpp_frame, enc_data, enc_buf_size);
    fwrite(enc_data, 1, enc_data_size, ctx->out_fp);
RET:
    // 如果有其他局部变量需要 free 再处理，enc_data 已变为持久化
    return;
}

// 文件流读取
int process_video_file(video_app_context_t *ctx, const char *path) {
    int video_size;
    char *video_data = (char *)read_file_data(path, &video_size);
    if (!video_data) return -1;
    char *video_data_end = video_data + video_size;
    
    const int SIZE = 8192;
    char *video_data_ptr = video_data;

    printf("Start playing video file: %s\n", path);
    do {
        int pkt_eos = 0;
        int size = SIZE;
        if (video_data_ptr + size >= video_data_end) {
            pkt_eos = 1;
            size = video_data_end - video_data_ptr;
        }

        ctx->decoder->Decode((uint8_t *)video_data_ptr, size, pkt_eos);
        video_data_ptr += size;

        if (video_data_ptr >= video_data_end) {
            printf("Video EOF, break.\n");
            break;
        }
    } while (1);

    free(video_data);
    return 0;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: %s <yolo26_model_path> <video_path> <video_type 264/265>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *video_path = argv[2];
    int video_type = atoi(argv[3]);

    int ret;
    video_app_context_t app_ctx;
    // 初始化上下文，将所有指针清零，防止后续 free 崩溃
    memset(&app_ctx, 0, sizeof(video_app_context_t));

    // 1. 初始化后处理参数（加载类别标签等）
    init_post_process();

    // 2. 初始化 YOLOv26 模型上下文
    ret = init_yolo26_model(model_path, &(app_ctx.yolo_app_ctx));
    if (ret != 0) {
        printf("init_yolo26_model fail! ret=%d\n", ret);
        return -1;
    }

    // 3. 初始化 MPP 解码器，并设置回调函数
    // 当解码器解出一帧 YUV 图像后，会自动进入 mpp_decoder_frame_callback
    MppDecoder *decoder = new MppDecoder();
    decoder->Init(video_type, 30, &app_ctx);
    decoder->SetCallback(mpp_decoder_frame_callback);
    app_ctx.decoder = decoder;

    // 4. 打开输出文件用于保存 H264 视频流
    app_ctx.out_fp = fopen(OUT_VIDEO_PATH, "w");
    if (!app_ctx.out_fp) {
        printf("Failed to open output file: %s\n", OUT_VIDEO_PATH);
        return -1;
    }

    // 5. 开始处理视频流（文件或实时流）
    if (strncmp(video_path, "rtsp", 4) == 0) {
        printf("RTSP stream is not implemented in this snippet.\n");
    } else {
        // 进入文件处理循环，直到 EOF
        process_video_file(&app_ctx, video_path);
    }

    printf("Processing finished, waiting for cleanup...\n");
    // 稍微等待，确保异步编码器的最后一帧数据写入完毕
    usleep(1 * 1000 * 1000); 

    // =========================================================
    // 【第三个修改的具体实现：最终释放持久化缓存】
    // =========================================================
    // 因为 enc_data 是在回调函数中 malloc 的，程序跑完后必须手动释放
    if (app_ctx.enc_data != nullptr) {
        free(app_ctx.enc_data);
        app_ctx.enc_data = nullptr;
        printf("Successfully released persistent enc_data buffer.\n");
    }

    // 6. 依次释放视频流相关的硬件资源
    if (app_ctx.out_fp) {
        fflush(app_ctx.out_fp);
        fclose(app_ctx.out_fp);
    }

    if (app_ctx.decoder != nullptr) delete app_ctx.decoder;
    if (app_ctx.encoder != nullptr) delete app_ctx.encoder;
    
    // 7. 释放模型和后处理资源
    release_yolo26_model(&(app_ctx.yolo_app_ctx));
    deinit_post_process();

    printf("All resources released. Exit safely.\n");
    return 0;
}