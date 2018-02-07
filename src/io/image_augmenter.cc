#include <dmlc/registry.h>
#include "./image_augmenter.h"

#if MXNET_USE_OPENCV
namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::io::ImageAugmenterReg);
}

namespace mxnet {
namespace io {
ImageAugmenter* ImageAugmenter::Create(const std::string& name) {
  return dmlc::Registry<ImageAugmenterReg>::Find(name)->body();
}
} // namespace io
} // namespace mxnet

#endif

