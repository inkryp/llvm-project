//===- WatchAtDebugLocationsClient.h - TODO(inkryp): Write ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_WATCHATDEBUGLOCATIONSCLIENT_H
#define MLIR_SUPPORT_WATCHATDEBUGLOCATIONSCLIENT_H

#include "mlir/Support/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/Support/DebugExecutionContext.h"

namespace mlir {

/// TODO(inkryp): Write a description of the service.
class WatchAtDebugLocationsClient : public DebugExecutionContext::Observer {
public:
  WatchAtDebugLocationsClient();

  /// Register the command line options for location breakpoints.
  static void registerCLOptions();

private:
  /// TODO(inkryp): Write a description for this method.
  void applyCLOptions();

  FileLineColLocBreakpointManager breakpointManager;
};

} // namespace mlir

#endif // MLIR_SUPPORT_WATCHATDEBUGLOCATIONSCLIENT_H
