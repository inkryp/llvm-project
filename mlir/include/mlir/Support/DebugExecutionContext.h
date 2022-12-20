//===- DebugExecutionContext.h - Debug Execution Context Support *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: Write a proper description for the service
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H
#define MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H

#include "mlir/Support/BreakpointManagers/BreakpointManager.h"
#include "mlir/Support/DebugAction.h"

// TODO(inkryp): Should this value be protected?
// TODO(inkryp): Should the user dynamically register `BreakpointManager`?
extern llvm::SmallVector<mlir::BreakpointManager *> &
getGlobalInstancesOfBreakpointManagers();

extern llvm::MapVector<
    unsigned, std::tuple<mlir::Breakpoint *, mlir::BreakpointManager &>> &
getGlobalInstanceOfBreakpoindIdsMap();

namespace mlir {

enum DebugExecutionControl {
  Apply = 1,
  Skip = 2,
  Step = 3,
  Next = 4,
  Finish = 5
};

struct DebugActionInformation {
  const DebugActionInformation *prev;
  const DebugActionBase &action;
};

class DebugExecutionContext : public DebugActionManager::GenericHandler {
public:
  DebugExecutionContext(
      llvm::function_ref<DebugExecutionControl(
          ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef,
          const int &, const DebugActionInformation *)>
          callback);

  struct Observer {
    virtual ~Observer() = default;

    virtual void onCallbackBeforeExecution(ArrayRef<IRUnit>,
                                           ArrayRef<StringRef>,
                                           const DebugActionInformation *,
                                           const int &,
                                           llvm::Optional<Breakpoint *>) {}

    virtual void onCallbackAfterExecution(ActionResult) {}
  };

  void registerObserver(Observer *);

  FailureOr<bool> execute(ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags,
                          llvm::function_ref<ActionResult()> transform,
                          const DebugActionBase &action) final;

private:
  llvm::function_ref<DebugExecutionControl(
      ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef, const int &,
      const DebugActionInformation *)>
      onBreakpointControlExecutionCallback;

  int depth = 0;

  const DebugActionInformation *daiHead;

  Optional<int> depthToBreak;

  SmallVector<Observer *> observers;
};

} // namespace mlir

#endif // MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H
