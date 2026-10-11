#!/usr/bin/env python3
"""Generate research-only native overrides; never edit production sources."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
def change(s,old,new,n=1):
 assert s.count(old)==n,(old[:100],s.count(old),n)
 return s.replace(old,new)
s=(a.source/'src/codestream/modular/search.cpp').read_text()
s='#include "study_profile.h"\n'+s
s=change(s,'  ProfileScope profile_scope(ProfileStage::kTraining);\n  if (!single', '  ProfileScope profile_scope(ProfileStage::kTraining);\n  learning_study::PassScope study_pass(base, learning);\n  if (!single')
s=change(s,'  if (auto status = LearnDecisionTrees(std::span(samples).first(used), &leaf.tree, &branch.tree);\n      !status.ok())\n    return status;', '''  {
    learning_study::Timer timer(learning_study::current ? &learning_study::current->small_tree : nullptr);
    if (auto status = LearnDecisionTrees(std::span(samples).first(used), &leaf.tree, &branch.tree); !status.ok()) return status;
  }
  if (learning_study::current) {
    learning_study::current->samples = stratified.size();
    learning_study::current->population = total;
  }''')
s=change(s,'    *entropy = std::move(learned);','    if (learning_study::current) learning_study::current->leaves=(learned.tree.size+1)/2;\n    *entropy = std::move(learned);')
(a.output/'search.cpp').write_text(s)
s=(a.source/'src/codec/modular/tree_learning.cpp').read_text()
s='#include "study_profile.h"\n#include <vector>\n'+s
s=change(s,'constexpr size_t kPredictors = 14, kSymbols = 128, kThresholds = 32;', '''#ifndef STUDY_PREDICTORS
#define STUDY_PREDICTORS 14
#endif
#ifndef STUDY_KEYS
#define STUDY_KEYS 0
#endif
#ifndef STUDY_SKIP_CONSTANT
#define STUDY_SKIP_CONSTANT 0
#endif
constexpr size_t kPredictors = STUDY_PREDICTORS, kSymbols = 128, kThresholds = 32;
static_assert(kPredictors == 14 || kPredictors == 2);
constexpr size_t WirePredictor(size_t p) { return kPredictors == 14 ? p : p + 5; }
struct StudyKey { int64_t value; uint32_t index; };''')
s=change(s,'  resource_budget_internal::ManagedVector<Leaf> leaves;','  resource_budget_internal::ManagedVector<Leaf> leaves;\n  std::vector<StudyKey> study_keys;')
s=change(s,'  size_t predictor = 5; // Same deterministic tie order as the retained learner.', '  size_t predictor = kPredictors == 14 ? 5 : 0; // Gradient wins ties.')
s=change(s,'  const size_t count = leaf.end - leaf.begin;\n  w.parent = {};', '''  const size_t count = leaf.end - leaf.begin;
  auto* study = learning_study::current;
  if (study) { ++study->nodes; study->node_samples += count; }
  const auto parent_start=learning_study::Clock::now();
  w.parent = {};''')
s=change(s,'  tree.nodes[leaf.node].predictor = static_cast<Predictor>(predictor);','  tree.nodes[leaf.node].predictor = static_cast<Predictor>(WirePredictor(predictor));\n  if (study) study->parent += std::chrono::duration<double>(learning_study::Clock::now()-parent_start).count();')
s=change(s,'''    std::sort(begin, end, [&](uint32_t a, uint32_t b) {
      const auto x = samples[a].properties[property], y = samples[b].properties[property];
      return x != y ? x < y : a < b;
    });''', '''    const auto sort_start=learning_study::Clock::now();
    if constexpr (STUDY_SKIP_CONSTANT) {
      const auto value=samples[*begin].properties[property];
      if (std::all_of(begin,end,[&](uint32_t i){return samples[i].properties[property]==value;})) {
        if (study) { ++study->constant_properties; study->constant_items+=count;
          study->sort+=std::chrono::duration<double>(learning_study::Clock::now()-sort_start).count(); }
        continue;
      }
    }
    if constexpr (STUDY_KEYS) {
      w.study_keys.resize(count);
      for (size_t i=0;i<count;++i) w.study_keys[i]={samples[begin[i]].properties[property],begin[i]};
      std::sort(w.study_keys.begin(),w.study_keys.end(),[](auto a, auto b){return a.value!=b.value?a.value<b.value:a.index<b.index;});
      for (size_t i=0;i<count;++i) begin[i]=w.study_keys[i].index;
    } else {
      std::sort(begin, end, [&](uint32_t a, uint32_t b) {
        const auto x = samples[a].properties[property], y = samples[b].properties[property];
        return x != y ? x < y : a < b;
      });
    }
    if (study) {
      ++study->sorts; study->sort_items+=count;
      if (samples[*begin].properties[property]==samples[*(end-1)].properties[property]) {
        ++study->constant_properties; study->constant_items+=count;
      }
      study->sort+=std::chrono::duration<double>(learning_study::Clock::now()-sort_start).count();
    }
    learning_study::Timer sweep_timer(study ? &study->sweep : nullptr);
    if (study) study->sweep_items+=count-1;''')
s=change(s,'      const int64_t split_bits = 12 + ResidualWidth(static_cast<int32_t>(threshold));','      if (study) ++study->thresholds;\n      const int64_t split_bits = 12 + ResidualWidth(static_cast<int32_t>(threshold));')
s=change(s,'  if (options.mode == TreeLearningMode::kQuantized)\n    return LearnQuantizedEntropyTree', '  learning_study::Timer fit_timer(learning_study::current ? &learning_study::current->fit : nullptr, learning_study::current ? &learning_study::current->fit_cpu : nullptr);\n  if (options.mode == TreeLearningMode::kQuantized)\n    return LearnQuantizedEntropyTree')
s=change(s,'  resource_budget_internal::ManagedVector<LearningWorkspace> storage(1);','  const auto setup_start=learning_study::Clock::now();\n  resource_budget_internal::ManagedVector<LearningWorkspace> storage(1);')
s=change(s,'      w.codes[i][p] = EncodeResidual(samples[i].residuals[p]);','      w.codes[i][p] = EncodeResidual(samples[i].residuals[WirePredictor(p)]);')
s=change(s,'  FindSplit(samples, w.leaves[0], tree, w);','  if (learning_study::current) learning_study::current->setup += std::chrono::duration<double>(learning_study::Clock::now()-setup_start).count();\n  FindSplit(samples, w.leaves[0], tree, w);')
s=change(s,'    const auto middle =\n        std::partition','    const auto partition_start=learning_study::Clock::now();\n    const auto middle =\n        std::partition')
s=change(s,'    const size_t boundary = static_cast<size_t>(middle - w.order.begin());','    const size_t boundary = static_cast<size_t>(middle - w.order.begin());\n    if (learning_study::current) learning_study::current->partition += std::chrono::duration<double>(learning_study::Clock::now()-partition_start).count();')
(a.output/'tree_learning.cpp').write_text(s)
