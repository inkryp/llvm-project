//===- DebugExecutionContextTest.cpp - Debug Execution Context first impl -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugAction.h"
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

ActionResult noOp() { return {nullptr, false, success()}; }

struct SimpleBreakpoint {
  std::string tag;
  bool enabled;
  SimpleBreakpoint(const std::string &_tag) : tag(_tag), enabled(true) {}
};

struct SimpleBreakpointManager {
  llvm::Optional<SimpleBreakpoint *> match(const StringRef &tag) {
    auto it = breakpoints.find(tag);
    if (it != breakpoints.end() && it->second->enabled) {
      return it->second.get();
    }
    return {};
  }
  SimpleBreakpoint *addBreakpoint(StringRef tag) {
    auto result = breakpoints.insert({tag, nullptr});
    auto &it = result.first;
    if (result.second) {
      it->second = std::make_unique<SimpleBreakpoint>(tag.str());
    }
    return it->second.get();
  }
  void enableBreakpoint(SimpleBreakpoint *breakpoint) {
    breakpoint->enabled = true;
  }
  void disableBreakpoint(SimpleBreakpoint *breakpoint) {
    breakpoint->enabled = false;
  }
  void deleteBreakpoint(SimpleBreakpoint *breakpoint) {
    breakpoints.erase(breakpoint->tag);
  }
  static SimpleBreakpointManager &getGlobalSbm() {
    static SimpleBreakpointManager *sbm = new SimpleBreakpointManager();
    return *sbm;
  }
  llvm::StringMap<std::unique_ptr<SimpleBreakpoint>> breakpoints;
};

enum DebugExecutionControl {
  Apply = 1,
  Skip = 2,
  Step = 3,
  Next = 4,
  Finish = 5
};

class DebugExecutionContext : public DebugActionManager::GenericHandler {
public:
  DebugExecutionContext(
      llvm::function_ref<DebugExecutionControl(
          ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef)>
          callback)
      : OnBreakpoint(callback), sbm(SimpleBreakpointManager::getGlobalSbm()) {}
  FailureOr<bool> execute(ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags,
                          llvm::function_ref<ActionResult()> transform,
                          const DebugActionBase &action) final {
    ++depth;
    auto handleUserInput = [&]() -> bool {
      auto todoNext =
          OnBreakpoint(units, instanceTags, action.tag, action.desc);
      while (depth == 1 && todoNext == DebugExecutionControl::Finish) {
        todoNext = OnBreakpoint(units, instanceTags, action.tag, action.desc);
      }
      switch (todoNext) {
      case DebugExecutionControl::Apply:
        depthToBreak = std::nullopt;
        return true;
      case DebugExecutionControl::Skip:
        depthToBreak = std::nullopt;
        return false;
      case DebugExecutionControl::Step:
        depthToBreak = depth + 1;
        return true;
      case DebugExecutionControl::Next:
        depthToBreak = depth;
        return true;
      case DebugExecutionControl::Finish:
        depthToBreak = depth - 1;
        return true;
      }
    };
    auto breakpoint = sbm.match(action.tag);
    bool apply = true;
    if (breakpoint || (depthToBreak && depth <= depthToBreak)) {
      apply = handleUserInput();
    }

    if (apply) {
      transform();
    }

    if (depthToBreak && depth <= depthToBreak) {
      handleUserInput();
    }
    --depth;
    return apply;
  }
  SimpleBreakpoint *addSimpleBreakpoint(StringRef tag) {
    return sbm.addBreakpoint(tag);
  }
  void disableSimpleBreakpoint(SimpleBreakpoint *breakpoint) {
    sbm.disableBreakpoint(breakpoint);
  }
  void enableSimpleBreakpoint(SimpleBreakpoint *breakpoint) {
    sbm.enableBreakpoint(breakpoint);
  }
  void deleteSimpleBreakpoint(SimpleBreakpoint *breakpoint) {
    sbm.deleteBreakpoint(breakpoint);
  }

private:
  llvm::function_ref<DebugExecutionControl(
      ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef)>
      OnBreakpoint;
  SimpleBreakpointManager &sbm;
  int depth = 0;
  Optional<int> depthToBreak;
};

TEST(DebugExecutionContext, DebuggerTest) {
  DebugActionManager manager;
  int match = 0;
  auto callback = [&match](ArrayRef<IRUnit> units,
                           ArrayRef<StringRef> instanceTags, StringRef tag,
                           StringRef desc) {
    match++;
    return DebugExecutionControl::Skip;
  };
  auto ptr = std::make_unique<DebugExecutionContext>(callback);
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));

  EXPECT_EQ(match, 0);

  auto dbgBreakpoint = dbg->addSimpleBreakpoint(DebuggerAction::getTag());
  EXPECT_TRUE(failed(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 1);

  dbg->disableSimpleBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 1);

  dbg->enableSimpleBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(failed(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 2);

  EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, noOp)));
  EXPECT_EQ(match, 2);

  dbg->deleteSimpleBreakpoint(dbgBreakpoint);
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
                          StringRef desc) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  dbg->addSimpleBreakpoint(DebuggerAction::getTag());

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
                          StringRef desc) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    assert(false);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  dbg->addSimpleBreakpoint(DebuggerAction::getTag());

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
                          StringRef desc) {
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
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  dbg->addSimpleBreakpoint(DebuggerAction::getTag());

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
                          StringRef desc) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  dbg->addSimpleBreakpoint(DebuggerAction::getTag());

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
                          StringRef desc) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 1);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  dbg->addSimpleBreakpoint(DebuggerAction::getTag());

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
                          StringRef desc) {
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
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  auto dbgBreakpoint = dbg->addSimpleBreakpoint(DebuggerAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, original)));
  EXPECT_EQ(counter, 3);
  dbg->deleteSimpleBreakpoint(dbgBreakpoint);
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
                          StringRef desc) {
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
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  dbg->addSimpleBreakpoint(OtherAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, original)));
  EXPECT_EQ(counter, 2);
}

TEST(DebugExecutionContext, FinishNothingBackTest) {
  DebugActionManager manager;
  std::vector<StringRef> tagSequence = {DebuggerAction::getTag(),
                                        DebuggerAction::getTag(),
                                        DebuggerAction::getTag()};
  std::vector<DebugExecutionControl> controlSequence = {
      DebugExecutionControl::Finish, DebugExecutionControl::Finish,
      DebugExecutionControl::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };
  auto callback = [&]() {
    EXPECT_EQ(counter, 3);
    return noOp();
  };
  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  dbg->addSimpleBreakpoint(DebuggerAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, callback)));
  EXPECT_EQ(counter, 3);
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
                          StringRef desc) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], tag);
    return controlSequence[idx++];
  };

  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  dbg->addSimpleBreakpoint(DebuggerAction::getTag());
  auto toBeDisabled = dbg->addSimpleBreakpoint(OtherAction::getTag());

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
    dbg->disableSimpleBreakpoint(toBeDisabled);
    dbg->addSimpleBreakpoint(ThirdAction::getTag());
    EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, nested)));
    EXPECT_EQ(counter, 3);
    return noOp();
  };

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, original)));
  EXPECT_EQ(counter, 4);
}
} // namespace

#endif
