//===- DebugActionTest.cpp - Debug Action Tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugAction.h"
#include "mlir/Support/TypeID.h"
#include "gmock/gmock.h"

// DebugActionManager is only enabled in DEBUG mode.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS

using namespace mlir;

namespace {
struct SimpleAction : DebugAction<SimpleAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimpleAction)
  static StringRef getTag() { return "simple-action"; }
  static StringRef getDescription() { return "simple-action-description"; }
};
struct OtherSimpleAction : DebugAction<OtherSimpleAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OtherSimpleAction)
  static StringRef getTag() { return "other-simple-action"; }
  static StringRef getDescription() {
    return "other-simple-action-description";
  }
};
struct ParametricAction : DebugAction<ParametricAction, bool> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParametricAction)
  ParametricAction(bool executeParam) : executeParam(executeParam) {}
  bool executeParam;
  static StringRef getTag() { return "param-action"; }
  static StringRef getDescription() { return "param-action-description"; }
};

ActionResult noOp() { return {nullptr, false, success()}; }

TEST(DebugActionTest, GenericHandler) {
  DebugActionManager manager;

  // A generic handler that always executes the simple action, but not the
  // parametric action.
  struct GenericHandler : DebugActionManager::GenericHandler {
    FailureOr<bool> execute(ArrayRef<IRUnit> units,
                            ArrayRef<StringRef> instanceTags,
                            llvm::function_ref<ActionResult()> transform,
                            const DebugActionBase &actionBase) final {
      if (llvm::isa<SimpleAction>(actionBase)) {
        const auto &action = llvm::cast<SimpleAction>(actionBase);
        EXPECT_EQ(action.tag, SimpleAction::getTag());
        transform();
        return true;
      }

      const auto &action = llvm::cast<ParametricAction>(actionBase);
      EXPECT_EQ(action.tag, ParametricAction::getTag());
      EXPECT_EQ(action.desc, ParametricAction::getDescription());
      return false;
    }
  };
  manager.registerActionHandler<GenericHandler>();

  EXPECT_TRUE(succeeded(manager.execute<SimpleAction>({}, {}, noOp)));
  EXPECT_TRUE(failed(manager.execute<ParametricAction>({}, {}, noOp, true)));
}

TEST(DebugActionTest, ActionSpecificHandler) {
  DebugActionManager manager;

  // Handler that simply uses the input as the decider.
  struct ActionSpecificHandler : ParametricAction::Handler {
    FailureOr<bool> execute(ArrayRef<IRUnit> units,
                            ArrayRef<StringRef> instanceTags,
                            llvm::function_ref<ActionResult()> transform,
                            const ParametricAction &action) final {
      if (action.executeParam) {
        transform();
      }
      return action.executeParam;
    }
  };
  manager.registerActionHandler<ActionSpecificHandler>();

  EXPECT_TRUE(succeeded(manager.execute<ParametricAction>({}, {}, noOp, true)));
  EXPECT_TRUE(failed(manager.execute<ParametricAction>({}, {}, noOp, false)));

  // There is no handler for the simple action, so it is always executed.
  EXPECT_TRUE(succeeded(manager.execute<SimpleAction>({}, {}, noOp)));
}

TEST(DebugActionTest, DebugCounterHandler) {
  DebugActionManager manager;

  // Handler that uses the number of action executions as the decider.
  struct DebugCounterHandler : SimpleAction::Handler {
    FailureOr<bool> execute(ArrayRef<IRUnit> units,
                            ArrayRef<StringRef> instanceTags,
                            llvm::function_ref<ActionResult()> transform,
                            const SimpleAction &action) final {
      if (numExecutions++ < 3) {
        transform();
      }
      return numExecutions <= 3;
    }
    unsigned numExecutions = 0;
  };
  manager.registerActionHandler<DebugCounterHandler>();

  // Check that the action is executed 3 times, but no more after.
  EXPECT_TRUE(succeeded(manager.execute<SimpleAction>({}, {}, noOp)));
  EXPECT_TRUE(succeeded(manager.execute<SimpleAction>({}, {}, noOp)));
  EXPECT_TRUE(succeeded(manager.execute<SimpleAction>({}, {}, noOp)));
  EXPECT_TRUE(failed(manager.execute<SimpleAction>({}, {}, noOp)));
  EXPECT_TRUE(failed(manager.execute<SimpleAction>({}, {}, noOp)));
}

TEST(DebugActionTest, NonOverlappingActionSpecificHandlers) {
  DebugActionManager manager;

  // One handler returns true and another returns false
  struct SimpleActionHandler : SimpleAction::Handler {
    FailureOr<bool> execute(ArrayRef<IRUnit> units,
                            ArrayRef<StringRef> instanceTags,
                            llvm::function_ref<ActionResult()> transform,
                            const SimpleAction &action) final {
      return true;
    }
  };
  struct OtherSimpleActionHandler : OtherSimpleAction::Handler {
    FailureOr<bool> execute(ArrayRef<IRUnit> units,
                            ArrayRef<StringRef> instanceTags,
                            llvm::function_ref<ActionResult()> transform,
                            const OtherSimpleAction &action) final {
      return false;
    }
  };
  manager.registerActionHandler<SimpleActionHandler>();
  manager.registerActionHandler<OtherSimpleActionHandler>();
  EXPECT_TRUE(succeeded(manager.execute<SimpleAction>({}, {}, noOp)));
  EXPECT_TRUE(failed(manager.execute<OtherSimpleAction>({}, {}, noOp)));
}

} // namespace

#endif
