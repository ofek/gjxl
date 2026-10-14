// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/ans_internal.h"
#include "codestream/modular/search.h"
#include "codestream/modular/workflow.h"
#include "codestream/profile_internal.h"
#include "io/pnm.h"
#include "modular_reference.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::codestream_internal;
using Clock = std::chrono::steady_clock;
namespace ref = gjxl::test::modular_reference;
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void Require(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
const char *Name(size_t mode) { return mode ? "estimated" : "exact"; }
AnsSelectionMode Selection(size_t mode) {
  return mode ? AnsSelectionMode::kEstimated : AnsSelectionMode::kExact;
}
void PolicyJson(const ModularCodingPolicy &policy) {
  std::cout << "{\"rct\":" << unsigned(policy.rct) << ",\"transforms\":[";
  for (size_t i = 0; i < policy.transforms.size; ++i) {
    const auto &t = policy.transforms.entries[i];
    std::cout << (i ? "," : "") << '[' << int(t.kind) << ',' << unsigned(t.begin) << ','
              << unsigned(t.count) << ',' << t.colors << ',' << t.horizontal << ',' << t.in_place
              << ']';
  }
  std::cout << "],\"nodes\":[";
  for (size_t i = 0; i < policy.tree.size; ++i) {
    const auto &n = policy.tree.nodes[i];
    std::cout << (i ? "," : "") << '[' << n.property << ',' << n.split << ',' << n.left << ','
              << n.right << ',' << int(n.predictor) << ',' << n.offset << ',' << n.multiplier
              << ']';
  }
  std::cout << "]}";
}
ModularCodingPolicy ReadPolicy(const std::string &path, size_t channels) {
  ModularCodingPolicy policy;
  if (path == "gradient") {
    policy.rct = channels >= 3 ? 6 : 0;
    return policy;
  }
  std::ifstream in(path);
  in.exceptions(std::ios::badbit | std::ios::failbit);
  unsigned rct;
  in >> rct >> policy.transforms.size >> policy.tree.size;
  Require(rct < 42 && policy.transforms.size <= kMaximumTransforms && policy.tree.size &&
              policy.tree.size <= kMaximumTreeNodes,
          "Invalid saved policy dimensions");
  policy.rct = rct;
  for (size_t i = 0; i < policy.transforms.size; ++i) {
    auto &t = policy.transforms.entries[i];
    unsigned kind, begin, count;
    in >> kind >> begin >> count >> t.colors >> t.horizontal >> t.in_place;
    Require(kind <= 1 && begin <= 255 && count <= 255, "Invalid transform in saved policy");
    t.kind = static_cast<TransformKind>(kind);
    t.begin = begin;
    t.count = count;
  }
  policy.tree.nodes.resize(policy.tree.size);
  for (size_t i = 0; i < policy.tree.size; ++i) {
    auto &n = policy.tree.nodes[i];
    unsigned predictor;
    in >> n.property >> n.split >> n.left >> n.right >> predictor >> n.offset >> n.multiplier;
    Require(predictor < 14, "Invalid saved predictor");
    n.predictor = static_cast<Predictor>(predictor);
  }
  Ok(ValidateCodingPolicy(policy));
  return policy;
}

#include "study_profile.h"
void Passes(const std::vector<learning_study::Pass>& passes) {
  std::cout << '[';
  for (size_t i=0;i<passes.size();++i) {
    const auto& p=passes[i];
    std::cout << (i?",":"") << "{\"policy\":"; PolicyJson(p.base);
    std::cout << ",\"sample_cap\":" << p.sample_cap << ",\"leaf_cap\":" << p.leaf_cap
      << ",\"samples\":" << p.samples << ",\"population\":" << p.population
      << ",\"leaves\":" << p.leaves << ",\"sample_bytes\":" << p.sample_bytes
      << ",\"total\":" << p.total << ",\"small_tree\":" << p.small_tree
      << ",\"cpu\":" << p.cpu << ",\"fit_cpu\":" << p.fit_cpu
      << ",\"fit\":" << p.fit << ",\"setup\":" << p.setup << ",\"parent\":" << p.parent
      << ",\"sort\":" << p.sort << ",\"sweep\":" << p.sweep << ",\"partition\":" << p.partition
      << ",\"nodes\":" << p.nodes << ",\"node_samples\":" << p.node_samples
      << ",\"sorts\":" << p.sorts << ",\"sort_items\":" << p.sort_items
      << ",\"constant_properties\":" << p.constant_properties << ",\"constant_items\":" << p.constant_items
      << ",\"thresholds\":" << p.thresholds << ",\"sweep_items\":" << p.sweep_items << '}';
  }
  std::cout << ']';
}
int main(int argc,char** argv) try {
  if(argc!=9) throw std::runtime_error("native_runner INPUT OUT search|fixed|quantized POLICY SAMPLES LEAVES REPEATS WARMUPS");
  const std::filesystem::path out=argv[2];
  const std::string phase=argv[3];
  const size_t samples=std::stoul(argv[5]),leaves=std::stoul(argv[6]),repeats=std::stoul(argv[7]),warmups=std::stoul(argv[8]);
  Require((phase=="search"||phase=="fixed"||phase=="quantized")&&repeats,"Invalid phase");
  io::IntegerImage image;Ok(io::ReadPnm(argv[1],&image));
  PackedModularImageView view{image.bytes,image.extent,image.row_stride,static_cast<PackedModularFormat>(image.format),SampleByteOrder::kBigEndian};
  ModularInputProfile input;Ok(ResolveModularInput(image.extent,view.format,&input));
  const unsigned bits=input.metadata.bits,channels=input.channel_count;
  ref::IntegerImage original{image.extent.width,image.extent.height,channels,bits,{}};
  for(size_t y=0;y<image.extent.height;++y)for(size_t x=0;x<image.extent.width*channels;++x){
    const size_t i=y*image.row_stride+x*(bits/8);
    original.samples.push_back(bits==8?image.bytes[i]:(uint16_t{image.bytes[i]}<<8)|image.bytes[i+1]);
  }
  std::filesystem::create_directories(out);
  gjxl::modular_internal::ModularEncodingOptions options;
  options.entropy=EntropyCodingMode::kAns;options.ans_selection=AnsSelectionMode::kEstimated;
  options.cpu_thread_count=4;options.search=phase=="search";
  options.learning={samples,leaves,phase=="quantized"?TreeLearningMode::kQuantized:TreeLearningMode::kLegacy};
  options.tree_checkpoints=options.search;
  const ModularCodingPolicy fixed=options.search?ModularCodingPolicy{}:ReadPolicy(argv[4],channels);
  std::vector<uint8_t> expected;
  std::cout<<std::setprecision(17);
  for(size_t iteration=0;iteration<repeats+warmups;++iteration){
    std::vector<learning_study::Pass> passes;
    learning_study::sink=&passes;
    ModularProfile profile;ModularCodingPolicy resolved;
    std::vector<uint8_t> encoded;
    double learning_seconds=0;
#if STUDY_CONTROL
    double learning_cpu=0;
#endif
    const auto start=Clock::now();
    {
      ProfileSession session(&profile);
      if(options.search){
        Ok(EncodeModularImage(view,options,&encoded));resolved=profile.resolved;
      }else{
        ModularCodingPolicy single,split,learned;
        TreeLearningCheckpoints checkpoints;
        {
          ModularEncoderFrame frame;Ok(ModularEncoderFrame::Prepare(view,fixed,&frame));
          ChannelShape shape;Ok(DescribeImage(frame.image(),&shape));
          ModularFrameGeometry geometry;Ok(ModularFrameGeometry::Create(image.extent,&geometry));
          ModularStreamPlan layout;Ok(BuildModularStreamPlan(geometry,shape.channels(),shape.metadata,&layout));
          const auto training_start=Clock::now();
#if STUDY_CONTROL
          const auto training_cpu_start=std::clock();
#endif
          Ok(LearnModularPolicies(frame,layout,fixed,&single,&split,&learned,options.learning,nullptr,&checkpoints));
          learning_seconds=std::chrono::duration<double>(Clock::now()-training_start).count();
#if STUDY_CONTROL
          learning_cpu=double(std::clock()-training_cpu_start)/CLOCKS_PER_SEC;
#endif
        }
        // Compare the same final/checkpoint portfolio under this learner. No
        // fallback to the saved winner: retain regressions in this ablation.
        options.coding=learned;
        Ok(EncodeModularImage(view,options,&encoded));resolved=learned;
        for(size_t i=0;i<checkpoints.size;++i){
          options.coding.tree=checkpoints.trees[i];std::vector<uint8_t> candidate;
          Ok(EncodeModularImage(view,options,&candidate));
          if(candidate.size()<encoded.size()){encoded=std::move(candidate);resolved=options.coding;}
        }
      }
    }
    const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
    learning_study::sink=nullptr;
    if(expected.empty()){
      Require(ref::DecodeLossless(encoded)==original,"Decode mismatch");expected=encoded;
      std::ofstream file(out/"output.jxl",std::ios::binary);file.exceptions(std::ios::badbit|std::ios::failbit);
      file.write(reinterpret_cast<const char*>(encoded.data()),encoded.size());
    }
    Require(encoded==expected,"Repeated output changed");
    std::cout<<"{\"variant\":\""<<STUDY_NAME<<"\",\"phase\":\""<<phase<<"\",\"iteration\":"<<iteration
      <<",\"warmup\":"<<(iteration<warmups?"true":"false")<<",\"seconds\":"<<elapsed
      <<",\"learning_seconds\":"<<learning_seconds<<",\"bytes\":"<<encoded.size()<<",\"decoded_exact\":true,\"candidates\":"<<profile.encoded_candidates<<",\"stages\":[";
    for(size_t i=0;i<profile.seconds.size();++i)std::cout<<(i?",":"")<<profile.seconds[i];
    std::cout<<"],\"policy\":";PolicyJson(resolved);std::cout<<",\"passes\":";Passes(passes);std::cout<<"}"<<std::endl;
#if STUDY_CONTROL
    std::cerr<<std::setprecision(17)<<"CONTROL {\"iteration\":"<<iteration
      <<",\"warmup\":"<<(iteration<warmups?"true":"false")
      <<",\"learning_cpu\":"<<learning_cpu<<"}\n";
#endif
  }
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
