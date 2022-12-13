//===- WatchAtDebugLocationsClient.h - TODO(inkryp): Write ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_WATCHATDEBUGLOCATIONSCLIENT_H
#define MLIR_SUPPORT_WATCHATDEBUGLOCATIONSCLIENT_H

#include "mlir/Support/DebugExecutionContext.h"

namespace mlir {

/// TODO(inkryp): Write a description of the service.
class WatchAtDebugLocationsClient {
public:
  WatchAtDebugLocationsClient(DebugExecutionContext *);

  /// Register the command line options for location breakpoints.
  static void registerCLOptions();

  void attachToDebugExecutionContext(DebugExecutionContext *);

private:
  /// Apply the registered CL options to this watch at debug locations client
  /// instance.
  void applyCLOptions(DebugExecutionContext *);

  llvm::function_ref<void(ArrayRef<IRUnit>, ArrayRef<StringRef>,
                          const DebugActionInformation *, const int &,
                          llvm::Optional<Breakpoint *>)>
      callback;
};

} // namespace mlir

#endif // MLIR_SUPPORT_WATCHATDEBUGLOCATIONSCLIENT_H
