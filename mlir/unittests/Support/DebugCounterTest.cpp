//===- DebugCounterTest.cpp - Debug Counter Tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugCounter.h"
#include "mlir/Support/TypeID.h"
#include "gmock/gmock.h"

using namespace mlir;

// DebugActionManager is only enabled in DEBUG mode.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS

namespace {

struct CounterAction : public DebugAction<CounterAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CounterAction)
  static StringRef getTag() { return "counter-action"; }
  static StringRef getDescription() { return "Test action for debug counters"; }
};

ActionResult noOp() { return {IRUnit(), false, success()}; }

TEST(DebugCounterTest, CounterTest) {
  std::unique_ptr<DebugCounter> counter = std::make_unique<DebugCounter>();
  counter->addCounter(CounterAction::getTag(), /*countToSkip=*/1,
                      /*countToStopAfter=*/3);

  DebugActionManager manager;
  manager.registerActionHandler(std::move(counter));

  // The first execution is skipped.
  EXPECT_TRUE(failed(manager.execute<CounterAction>({}, {}, noOp)));

  // The counter stops after 3 successful executions.
  EXPECT_TRUE(succeeded(manager.execute<CounterAction>({}, {}, noOp)));
  EXPECT_TRUE(succeeded(manager.execute<CounterAction>({}, {}, noOp)));
  EXPECT_TRUE(succeeded(manager.execute<CounterAction>({}, {}, noOp)));
  EXPECT_TRUE(failed(manager.execute<CounterAction>({}, {}, noOp)));
}

} // namespace

#endif
