//===- DebugExecutionContextTest.cpp - Debug Execution Context first impl -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugExecutionContext.h"
#include "mlir/Support/BreakpointManagers/SimpleBreakpointManager.h"
#include "llvm/ADT/MapVector.h"
#include "gmock/gmock.h"

using namespace mlir;

// DebugActionManager is only enabled in DEBUG mode.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS

namespace {
struct DebuggerAction : public DebugAction<DebuggerAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DebuggerAction)
  static StringRef getTag() { return "debugger-action"; }
  static StringRef getDescription() { return "Test action for debug client"; }
};
struct OtherAction : public DebugAction<OtherAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OtherAction)
  static StringRef getTag() { return "other-action"; }
  static StringRef getDescription() { return "Test action for debug client"; }
};
struct ThirdAction : public DebugAction<ThirdAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThirdAction)
  static StringRef getTag() { return "third-action"; }
  static StringRef getDescription() { return "Test action for debug client"; }
};

ActionResult noOp() { return {IRUnit(), false, success()}; }

TEST(DebugExecutionContext, DebugActionInformationTest) {
  DebugActionManager manager;

  std::vector<std::vector<StringRef>> expectedStacks = {
      {DebuggerAction::getTag()},
      {OtherAction::getTag(), DebuggerAction::getTag()},
      {ThirdAction::getTag(), OtherAction::getTag(), DebuggerAction::getTag()}};
  std::vector<StringRef> currentStack;

  auto generateStack = [&](const DebugActionInformation *backtrace) {
    currentStack.clear();
    auto *cur = backtrace;
    while (cur != nullptr) {
      currentStack.push_back(cur->action.tag);
      cur = cur->prev;
    }
    return currentStack;
  };

  auto checkStacks = [&](const std::vector<StringRef> &currentStack,
                         const std::vector<StringRef> &expectedStack) {
    if (currentStack.size() != expectedStack.size()) {
      return false;
    }
    bool areEqual = true;
    for (int i = 0; i < (int)currentStack.size(); ++i) {
      if (currentStack[i] != expectedStack[i]) {
        areEqual = false;
        break;
      }
    }
    return areEqual;
  };

  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Step, DebugExecutionControl::Step,
      DebugExecutionControl::Apply};
  int idx = 0;
  StringRef current;
  int currentDepth;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    current = backtrace->action.tag;
    currentDepth = depth;
    generateStack(backtrace);
    return controlSequence[idx++];
  };

  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  std::vector<SimpleBreakpoint *> breakpoints;
  breakpoints.push_back(
      SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
          DebuggerAction::getTag()));
  breakpoints.push_back(
      SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
          OtherAction::getTag()));
  breakpoints.push_back(
      SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
          ThirdAction::getTag()));

  auto third = [&]() {
    EXPECT_EQ(current, ThirdAction::getTag());
    EXPECT_EQ(currentDepth, 3);
    EXPECT_TRUE(checkStacks(currentStack, expectedStacks[2]));
    return noOp();
  };
  auto nested = [&]() {
    EXPECT_EQ(current, OtherAction::getTag());
    EXPECT_EQ(currentDepth, 2);
    EXPECT_TRUE(checkStacks(currentStack, expectedStacks[1]));
    EXPECT_TRUE(succeeded(manager.execute<ThirdAction>({}, {}, third)));
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(current, DebuggerAction::getTag());
    EXPECT_EQ(currentDepth, 1);
    EXPECT_TRUE(checkStacks(currentStack, expectedStacks[0]));
    EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, nested)));
    return noOp();
  };

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, original)));
  for (auto *breakpoint : breakpoints) {
    SimpleBreakpointManager::getGlobalInstance().deleteBreakpoint(breakpoint);
  }
}

TEST(DebugExecutionContext, DebuggerTest) {
  DebugActionManager manager;
  int match = 0;
  auto callback = [&match](ArrayRef<IRUnit> units,
                           ArrayRef<StringRef> instanceTags, StringRef tag,
                           StringRef desc, const int &depth,
                           const DebugActionInformation *backtrace) {
    match++;
    return DebugExecutionControl::Skip;
  };
  auto ptr = std::make_unique<DebugExecutionContext>(callback);
  manager.registerActionHandler(std::move(ptr));

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));

  EXPECT_EQ(match, 0);

  auto dbgBreakpoint =
      SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
          DebuggerAction::getTag());
  EXPECT_TRUE(failed(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 1);

  SimpleBreakpointManager::getGlobalInstance().disableBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 1);

  SimpleBreakpointManager::getGlobalInstance().enableBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(failed(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 2);

  EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, noOp)));
  EXPECT_EQ(match, 2);

  SimpleBreakpointManager::getGlobalInstance().deleteBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 2);
}

TEST(DebugExecutionContext, ApplyTest) {
  DebugActionManager manager;
  std::vector<StringRef> tagSequence = {DebuggerAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
      DebuggerAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, callback)));
  EXPECT_EQ(counter, 1);
}

TEST(DebugExecutionContext, SkipTest) {
  DebugActionManager manager;
  std::vector<StringRef> tagSequence = {DebuggerAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Skip};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    assert(false);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
      DebuggerAction::getTag());

  EXPECT_TRUE(failed(manager.execute<DebuggerAction>({}, {}, callback)));
  EXPECT_EQ(counter, 1);
}

TEST(DebugExecutionContext, StepApplyTest) {
  DebugActionManager manager;
  std::vector<StringRef> tagSequence = {DebuggerAction::getTag(),
                                        OtherAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Step, DebugExecutionControl::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto nested = [&]() {
    EXPECT_EQ(counter, 2);
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(counter, 1);
    EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, nested)));
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
      DebuggerAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, original)));
  EXPECT_EQ(counter, 2);
}

TEST(DebugExecutionContext, StepNothingInsideTest) {
  DebugActionManager manager;
  std::vector<StringRef> tagSequence = {DebuggerAction::getTag(),
                                        DebuggerAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Step, DebugExecutionControl::Step};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
      DebuggerAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, callback)));
  EXPECT_EQ(counter, 2);
}

TEST(DebugExecutionContext, NextTest) {
  DebugActionManager manager;
  std::vector<StringRef> tagSequence = {DebuggerAction::getTag(),
                                        DebuggerAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Next, DebugExecutionControl::Next};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
      DebuggerAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, callback)));
  EXPECT_EQ(counter, 2);
}

TEST(DebugExecutionContext, FinishTest) {
  DebugActionManager manager;
  std::vector<StringRef> tagSequence = {DebuggerAction::getTag(),
                                        OtherAction::getTag(),
                                        DebuggerAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Step, DebugExecutionControl::Finish,
      DebugExecutionControl::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto nested = [&]() {
    EXPECT_EQ(counter, 2);
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(counter, 1);
    EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, nested)));
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  auto dbgBreakpoint =
      SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
          DebuggerAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, original)));
  EXPECT_EQ(counter, 3);
  SimpleBreakpointManager::getGlobalInstance().deleteBreakpoint(dbgBreakpoint);
}

TEST(DebugExecutionContext, FinishBreakpointInNestedTest) {
  DebugActionManager manager;
  std::vector<StringRef> tagSequence = {OtherAction::getTag(),
                                        DebuggerAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Finish, DebugExecutionControl::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto nested = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(counter, 0);
    EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, nested)));
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
      OtherAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, original)));
  EXPECT_EQ(counter, 2);
}

TEST(DebugExecutionContext, FinishNothingBackTest) {
  DebugActionManager manager;
  std::vector<StringRef> tagSequence = {DebuggerAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Finish};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
      DebuggerAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, callback)));
  EXPECT_EQ(counter, 1);
}

TEST(DebugExecutionContext, EnableDisableBreakpointOnCallback) {
  DebugActionManager manager;

  std::vector<StringRef> tagSequence = {
      DebuggerAction::getTag(), ThirdAction::getTag(), OtherAction::getTag(),
      DebuggerAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Apply, DebugExecutionControl::Finish,
      DebugExecutionControl::Finish, DebugExecutionControl::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };

  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  manager.registerActionHandler(std::move(ptr));
  SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
      DebuggerAction::getTag());
  auto toBeDisabled =
      SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
          OtherAction::getTag());

  auto third = [&]() {
    EXPECT_EQ(counter, 2);
    return noOp();
  };
  auto nested = [&]() {
    EXPECT_EQ(counter, 1);
    EXPECT_TRUE(succeeded(manager.execute<ThirdAction>({}, {}, third)));
    EXPECT_EQ(counter, 2);
    return noOp();
  };
  auto original = [&]() {
    EXPECT_EQ(counter, 1);
    SimpleBreakpointManager::getGlobalInstance().disableBreakpoint(
        toBeDisabled);
    SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
        ThirdAction::getTag());
    EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, nested)));
    EXPECT_EQ(counter, 3);
    return noOp();
  };

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, original)));
  EXPECT_EQ(counter, 4);
}
} // namespace

#endif
