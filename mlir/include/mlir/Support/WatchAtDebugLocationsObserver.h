//===- WatchAtDebugLocationsObserver.h - TODO(inkryp): Write ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_WATCHATDEBUGLOCATIONSOBSERVER_H
#define MLIR_SUPPORT_WATCHATDEBUGLOCATIONSOBSERVER_H

#include "mlir/Support/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/Support/DebugExecutionContext.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {

/// TODO(inkryp): Write a description of the service.
class WatchAtDebugLocationsObserver : public DebugExecutionContext::Observer {
public:
  WatchAtDebugLocationsObserver();
  ~WatchAtDebugLocationsObserver();

  /// Register the command line options for location breakpoints.
  static void registerCLOptions();

private:
  /// TODO(inkryp): Write a description for this method.
  void applyCLOptions();

  void onCallbackBeforeExecution(ArrayRef<IRUnit>, ArrayRef<StringRef>,
                                 const DebugActionInformation *, const int &,
                                 llvm::Optional<Breakpoint *>) override;

  FileLineColLocBreakpointManager breakpointManager;

  std::unique_ptr<llvm::ToolOutputFile> outputFile = nullptr;
};

} // namespace mlir

#endif // MLIR_SUPPORT_WATCHATDEBUGLOCATIONSOBSERVER_H
