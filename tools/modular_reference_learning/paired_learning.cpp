// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "paired_learning.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include <ostream>
#include <stdexcept>
#include <string>

namespace reference_learning {
namespace native = gjxl::modular_internal;
namespace {
void Require(bool good, const char *message) {
  if (!good)
    throw std::runtime_error(message);
}
void Ok(const gjxl::Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
struct Measurement {
  Timing total, preparation, fit;
  size_t samples = 0, distinct = 0, nodes = 0;
};
void Print(std::ostream &out, const char *name, Timing time) {
  out << ",\"" << name << "_cpu\":" << time.cpu << ",\"" << name
      << "_wall\":" << time.wall;
}
} // namespace

void RunPairedLearning(std::span<const jxl::Image> images,
                       const jxl::ModularOptions &option,
                       const Fixture &fixture, size_t repeats,
                       std::ostream &out) {
  Require(!images.empty() && fixture.multipliers.empty() && repeats &&
              repeats <= 20,
          "Invalid paired-learning control");
  // Borrow the reference planes so both paths see identical pixels and strides.
  const auto borrowed = BorrowReferenceImages(images);
  const auto native_options = NativeSamplingOptions(option);
  const std::vector<jxl::ModularOptions> options(images.size(), option);
  gjxl::resource_budget_internal::HostStorageBound preparation_bound, fit_bound;
  Ok(native::ComputeHistogramSamplePreparationStorageBound(
      borrowed.images, native_options, &preparation_bound));
  auto run_native = [&](bool warmup) {
    Measurement result;
    native::DecisionTree tree;
    const Timer total;
    {
      native::PreparedHistogramSamples prepared;
      const Timer preparation;
      Ok(native::PrepareHistogramSamples(borrowed.images, &prepared,
                                         native_options));
      result.preparation = preparation.Elapsed();
      // The warmup additionally verifies the rows before fitting reorders them.
      // Its timings are never reported.
      if (warmup) {
        CheckPreparedSamples(prepared, fixture);
        Ok(native::ComputeHistogramTreeLearningStorageBound(
            prepared.samples(), prepared.tree_options(), &fit_bound));
      }
      result.samples = prepared.samples().total_samples;
      result.distinct = prepared.samples().weights.size();
      const Timer fitting;
      Ok(native::LearnPreparedHistogramTree(prepared, &tree));
      result.fit = fitting.Elapsed();
    } // Include sample destruction in total, as the reference call does.
    result.total = total.Elapsed();
    result.nodes = tree.size;
    CheckNativeTree(tree, fixture.expected);
    Require(result.samples == fixture.samples.NumSamples() &&
                result.distinct == fixture.samples.NumDistinctSamples(),
            "Native paired sample population changed");
    return result;
  };
  auto run_reference = [&] {
    Measurement result;
    Session session;
    session.fits.reserve(1);
    session.learning.reserve(1);
    jxl::Tree tree;
    {
      const ScopedSession scope(&session);
      const Timer total;
      auto learned =
          jxl::LearnTree(images.data(), options.data(), 0, images.size());
      Require(learned.ok(), "Paired reference learning failed");
      tree = std::move(learned).value_();
      result.total = total.Elapsed();
    }
    Require(session.fits.size() == 1 && session.learning.size() == 1 &&
                !session.fits[0].replayed,
            "Unexpected paired reference learning scope");
    const auto &fit = session.fits[0];
    result.fit = fit.fit;
    result.samples = fit.samples;
    result.distinct = fit.distinct;
    result.nodes = tree.size();
    Require(SameTree(tree, fixture.expected) &&
                result.samples == fixture.samples.NumSamples() &&
                result.distinct == fixture.samples.NumDistinctSamples(),
            "Paired reference tree or population changed");
    return result;
  };
  (void)run_reference();
  (void)run_native(true);
  for (size_t i = 0; i < repeats; ++i) {
    Measurement reference, native;
    if (i % 2 == 0)
      reference = run_reference();
    native = run_native(false);
    if (i % 2 != 0)
      reference = run_reference();
    out << "{\"iteration\":" << i
        << ",\"reference_first\":" << (i % 2 == 0 ? "true" : "false")
        << ",\"samples_exact_in_warmup\":true,\"trees_exact\":true,\"samples\":"
        << native.samples << ",\"distinct\":" << native.distinct
        << ",\"nodes\":" << native.nodes
        << ",\"predictors\":" << native_options.predictor_count
        << ",\"properties\":" << native_options.property_count
        << ",\"preparation_storage_bound_bytes\":"
        << preparation_bound.peak_bytes
        << ",\"fitting_storage_bound_bytes\":" << fit_bound.peak_bytes;
    Print(out, "native_total", native.total);
    Print(out, "native_preparation", native.preparation);
    Print(out, "native_fit", native.fit);
    Print(out, "reference_total", reference.total);
    Print(out, "reference_fit", reference.fit);
    // Reference non-fit is a difference, not an isolated preparation timer:
    // includes sample collection, finalization, release and wrapper overhead.
    Print(out, "reference_non_fit",
          {reference.total.wall - reference.fit.wall,
           reference.total.cpu - reference.fit.cpu});
    out << "}\n";
  }
}
} // namespace reference_learning
