#include <cctype>
#include <cstring>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <dmlc/logging.h>
#include <dmlc/recordio.h>
#include <opencv2/opencv.hpp>
#include "../src/io/image_recordio.h"
#include <random>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: <image.lst> <image_root_dir> <output.rec> [additional parameters in form key=value]\n"\
           "Possible additional parameters:\n"\
           "\tcolor=USE_COLOR[default=1] Force color (1), gray image (0) or keep source unchanged (-1).\n"\
           "\tresize=newsize resize the shorter edge of image to the newsize, original images will be packed by default\n"\
           "\tlabel_width=WIDTH[default=1] specify the label_width in the list, by default set to 1\n"\
           "\tpack_label=PACK_LABEL[default=0] whether to also pack multi dimenional label in the record file\n"\
           "\tnsplit=NSPLIT[default=1] used for part generation, logically split the image.list to NSPLIT parts by position\n"\
           "\tpart=PART[default=0] used for part generation, pack the images from the specific part in image.list\n"\
           "\tcenter_crop=CENTER_CROP[default=0] specify whether to crop the center image to make it square.\n"\
           "\tquality=QUALITY[default=95] JPEG quality for encoding (1-100, default: 95) or PNG compression for encoding (1-9, default: 3).\n"\
           "\tencoding=ENCODING[default='.jpg'] Encoding type. Can be '.jpg' or '.png'\n"\
           "\tinter_method=INTER_METHOD[default=1] NN(0) BILINEAR(1) CUBIC(2) AREA(3) LANCZOS4(4) AUTO(9) RAND(10).\n"\
           "\tunchanged=UNCHANGED[default=0] Keep the original image encoding, size and color. If set to 1, it will ignore the others parameters.\n");
    return 0;
  }
  int label_width = 1;
  int pack_label = 0;
  int new_size = -1;
  int nsplit = 1;
  int partid = 0;
  int center_crop = 0;
  int quality = 95;
  int color_mode = CV_LOAD_IMAGE_COLOR;
  int unchanged = 0;
  int inter_method = CV_INTER_LINEAR;
  std::string encoding(".jpg");
  for (int i = 4; i < argc; ++i) {
    char key[128], val[128];
    int effct_len = 0;

#ifdef _MSC_VER
    effct_len = sscanf_s(argv[i], "%[^=]=%s", key, sizeof(key), val, sizeof(val));
#else
    effct_len = sscanf(argv[i], "%[^=]=%s", key, val);
#endif
    if (effct_len == 2) {
      if (!strcmp(key, "resize")) new_size = atoi(val); // 默认是负一
      if (!strcmp(key, "label_width")) label_width = atoi(val); // 1
      if (!strcmp(key, "pack_label")) pack_label = atoi(val); // 0 
      if (!strcmp(key, "nsplit")) nsplit = atoi(val); // 1
      if (!strcmp(key, "part")) partid = atoi(val); // 0
      if (!strcmp(key, "center_crop")) center_crop = atoi(val); // 0
      if (!strcmp(key, "quality")) quality = atoi(val); // 95
      if (!strcmp(key, "color")) color_mode = atoi(val); // CV_LOAD_IMAGE_COLOR
      if (!strcmp(key, "encoding")) encoding = std::string(val); // .jpg
      if (!strcmp(key, "unchanged")) unchanged = atoi(val); // 0
      if (!strcmp(key, "inter_method")) inter_method = atoi(val); // CV_INTER_LINEAR
    }
  }
  std::random_device rd;
  std::mt19937 prnd(rd());
  using namespace dmlc;
  const static size_t kBufferSize = 1 << 20UL;
  std::string root = argv[2];
  mxnet::io::ImageRecordIO rec;
  size_t imcnt = 0;
  dmlc::InputSplit *flist = dmlc::InputSplit::
      Create(argv[1], partid, nsplit, "text"); 
  dmlc::InputSplit::Blob line;
  flist->NextRecord(&line);
  std::string sline(static_cast<char*>(line.dptr), line.size);
  LOG(INFO) << sline;
}
