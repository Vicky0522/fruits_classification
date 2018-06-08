/*************************************************************************
    > File Name: roi_interp/src/roi_interp.c
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月09日 星期一 19时43分09秒
 ************************************************************************/
#include <TH/TH.h>
#include <math.h>

int roi_putin_forward(const int front_height, const int front_width,
                      THFloatTensor * input_front,
                      THFloatTensor * rois, THFloatTensor * output) {
  // Grab the input tensor
  float * data_front_flat = THFloatTensor_data(input_front);
  float * roi_flat = THFloatTensor_data(rois);

  float * out_flat = THFloatTensor_data(output);

  int batch_size = THFloatTensor_size(input_front, 0);
  int num_channels = THFloatTensor_size(input_front, 1);
  int back_height = THFloatTensor_size(output, 2);
  int back_width = THFloatTensor_size(output, 3);

  int b, hi, wi, ho, wo, c;
  int data_front_index = 0, out_index = 0, roi_index = 0;
  int data_front_size = num_channels * front_height * front_width; 
  int out_size = num_channels * back_height * back_width;
  for (b=0; b<batch_size; ++b) {
    float * data_front = &data_front_flat[data_front_index];
    float * out = &out_flat[out_index];
    float * roi = &roi_flat[roi_index]; 
    int roi_width_start = roi[0];
    int roi_height_start = roi[1];
    int roi_width_end = roi[2];
    int roi_height_end = roi[3];
    int roi_height = roi_height_end - roi_height_start + 1;
    int roi_width = roi_width_end - roi_width_start + 1;

    //special case : just copy
    if (roi_height == front_height && roi_width == front_width) {
      for (hi=0; hi<front_height; ++hi) {
        ho = hi;
        for (wi=0; wi<front_width; ++wi) {
          wo = wi;
          float * out_roi = out + (roi_height_start + ho) * back_width + (roi_width_start + wo);
          float * front = data_front + hi * front_width + wi;
          for (c=0; c<num_channels; ++c) {
            out_roi[0] = front[0];
            out_roi += back_width * back_height;
            front += front_width * front_height;
          }
        }
      }
      return 1;
    }
    const float rheight = (roi_height > 1) ? (float)(front_height - 1) / (roi_height - 1) : 0.f;
    const float rwidth = (roi_width > 1) ? (float)(front_width - 1) / (roi_width - 1) : 0.f;
    for (ho = 0; ho < roi_height; ++ho) {
      const float h1r = rheight * ho;
      hi = h1r;
      const int h1p = (hi < front_height - 1) ? 1 : 0;
      const float h1lambda = h1r - hi;
      const float h0lambda = (float)(1.) - h1lambda;
      for (wo = 0; wo < roi_width; ++wo) {
        const float w1r = rwidth * wo;
        wi = w1r;
        const int w1p = (wi < front_width - 1) ? 1 : 0;
        const float w1lambda = w1r - wi;
        const float w0lambda = (float)(1.) - w1lambda;

        const float* pos1 = &data_front[hi * front_width + wi];
        float* pos2 = &out[(roi_height_start + ho) * back_width + (roi_width_start + wo)];
        for (c = 0; c < num_channels; ++c) {
          pos2[0] =
            h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) + 
            h1lambda * (w0lambda * pos1[h1p * front_width] + w1lambda * pos1[h1p * front_width + w1p]);
          pos1 += front_width * front_height;
          pos2 += back_width * back_height;
        }
      }
    }

    data_front_index += data_front_size;
    out_index += out_size;
    roi_index += 4;
  }
  return 1;
}


