//===- WatchAtDebugLocationsObserver.cpp - Location breakpoint Observer ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/WatchAtDebugLocationsObserver.h"
#include "mlir/Support/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/Support/DebugExecutionContext.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// WatchAtDebugLocationsObserverOptions CommandLine Options
//===----------------------------------------------------------------------===//

namespace {
/// This struct contains command line options that can be used to initialize
/// various bits of a DebugExecutionContext. This uses a struct wrapper to avoid
/// the need for global command line options.
struct WatchAtDebugLocationsObserverOptions {
  llvm::cl::list<std::string> locations{
      "watch-at-debug-locations",
      llvm::cl::desc("Comma separated list of location arguments"),
      llvm::cl::CommaSeparated};
};
} // namespace

static llvm::ManagedStatic<WatchAtDebugLocationsObserverOptions> clOptions;

//===----------------------------------------------------------------------===//
// WatchAtDebugLocationsObserver
//===----------------------------------------------------------------------===//

WatchAtDebugLocationsObserver::WatchAtDebugLocationsObserver() : Observer() {
  applyCLOptions();
}

/// Register a set of useful command-line options that can be used to configure
/// various flags within the DebugCounter. These flags are used when
/// constructing a DebugCounter for initialization.
void WatchAtDebugLocationsObserver::registerCLOptions() {
#ifndef NDEBUG
  // Make sure that the options struct has been initialized.
  *clOptions;
#endif
}

// This is called by the command line parser when it sees a value for the
// watch-at-debug-locations option defined above.
// TODO(inkryp): Study behavior when this function gets called twice
void WatchAtDebugLocationsObserver::applyCLOptions() {
  if (!clOptions.isConstructed())
    return;

  for (StringRef arg : clOptions->locations) {
    if (arg.empty())
      continue;

    // Watch at debug locations arguments are expected to be in the form:
    // `fileName:line:col`.
    // TODO(inkryp): Clean this up. Is there a way to get the tuple right away?
    auto [locationFile, locationLineCol] = arg.split(':');
    auto [locationLineStr, locationColStr] = locationLineCol.split(':');
    if (locationLineStr.empty()) {
      llvm::errs()
          << "error: expected WatchAtDebugLocationsObserver argument to have "
             "a `:` separating the file name and the line and column, "
             "but provided argument was : "
          << arg << "`\n";
      llvm::report_fatal_error(
          "Invalid WatchAtDebugLocationsObserver command-line configuration");
    }

    // Extract the line and column value
    int64_t locationLineInt, locationColInt;
    if (locationLineStr.getAsInteger(0, locationLineInt)) {
      llvm::errs()
          << "error: expected WatchAtDebugLocationsObserver line value to be "
             "numeric, but got `"
          << locationLineStr << "`\n";
      llvm::report_fatal_error(
          "Invalid WatchAtDebugLocationsObserver command-line configuration");
    }
    if (locationColStr.getAsInteger(0, locationColInt)) {
      llvm::errs()
          << "error: expected WatchAtDebugLocationsObserver column value to be "
             "numeric, but got `"
          << locationColStr << "`\n";
      llvm::report_fatal_error(
          "Invalid WatchAtDebugLocationsObserver command-line configuration");
    }

    // TODO(inkryp): Check for values to be positive
    breakpointManager.addBreakpoint(locationFile, locationLineInt,
                                    locationColInt);
  }
}

void WatchAtDebugLocationsObserver::onCallbackBeforeExecution(
    ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
    const DebugActionInformation *daiHead, const int &depth,
    llvm::Optional<Breakpoint *> breakpoint) {
  auto match = breakpointManager.match(daiHead->action, instanceTags, units);
  if (match) {
    // TODO(inkryp): Enable a way of specifying the output
    daiHead->action.print(llvm::dbgs());
  }
}
