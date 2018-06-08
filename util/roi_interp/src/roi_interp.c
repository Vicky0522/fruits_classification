/*************************************************************************
    > File Name: roi_interp/src/roi_interp.c
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月09日 星期一 19时43分09秒
 ************************************************************************/
#include <TH/TH.h>
#include <math.h>

int roi_interp_forward(int interp_height, int interp_width, 
                       THFloatTensor * input, THFloatTensor * rois, 
                       THFloatTensor * output) {
  // Grab the input tensor
  float * data_flat = THFloatTensor_data(input);
  float * roi_flat = THFloatTensor_data(rois);

  float * out_flat = THFloatTensor_data(output);

  int batch_size = THFloatTensor_size(input, 0);
  int num_channels = THFloatTensor_size(input, 1);
  int data_height = THFloatTensor_size(input, 2);
  int data_width = THFloatTensor_size(input, 3);

  int b, hi, wi, ho, wo, c;
  int data_index = 0, out_index = 0, roi_index = 0;
  int data_size = num_channels * data_height * data_width; 
  int out_size = num_channels * interp_height * interp_width;
  for (b=0; b<batch_size; ++b) {
    float * data = &data_flat[data_index];
    float * out = &out_flat[out_index];
    float * roi = &roi_flat[roi_index]; 
    int roi_width_start = roi[0];
    int roi_height_start = roi[1];
    int roi_width_end = roi[2];
    int roi_height_end = roi[3];
    int roi_height = roi_height_end - roi_height_start;
    int roi_width = roi_width_end - roi_width_start;

    //special case : just copy
    if (roi_height == interp_height && roi_width == interp_width) {
      for (hi=0; hi<roi_height; ++hi) {
        ho = hi;
        for (wi=0; wi<roi_width; ++wi) {
          wo = wi;
          float * data_roi = data + (roi_height_start + hi) * data_width + (roi_width_start + wi);
          float * data_out = out + ho * interp_width + wo;
          for (c=0; c<num_channels; ++c) {
            data_out[0] = data_roi[0];
            data_roi += data_width * data_height;
            data_out += interp_width * interp_height;
          }
        }
      }
      return 1;
    }
    const float rheight = (interp_height > 1) ? (float)(roi_height - 1) / (interp_height - 1) : 0.f;
    const float rwidth = (interp_width > 1) ? (float)(roi_width - 1) / (interp_width - 1) : 0.f;
    for (ho = 0; ho < interp_height; ++ho) {
      const float h1r = rheight * ho;
      hi = h1r;
      const int h1p = (hi < roi_height - 1) ? 1 : 0;
      const float h1lambda = h1r - hi;
      const float h0lambda = (float)(1.) - h1lambda;
      for (wo = 0; wo < interp_width; ++wo) {
        const float w1r = rwidth * wo;
        wi = w1r;
        const int w1p = (wi < roi_width - 1) ? 1 : 0;
        const float w1lambda = w1r - wi;
        const float w0lambda = (float)(1.) - w1lambda;

        const float* pos1 = &data[(roi_height_start + hi) * data_width + (roi_width_start + wi)];
        float* pos2 = &out[ho * interp_width + wo];
        for (c = 0; c < num_channels; ++c) {
          pos2[0] =
            h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) + 
            h1lambda * (w0lambda * pos1[h1p * data_width] + w1lambda * pos1[h1p * data_width + w1p]);
          pos1 += data_width * data_height;
          pos2 += interp_width * interp_height;
        }
      }
    }

    data_index += data_size;
    out_index += out_size;
    roi_index += 4;
  }
  return 1;
}


