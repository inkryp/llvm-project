//===- DebugExecutionContext.cpp - Debug Execution Context Support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugExecutionContext.h"
#include "mlir/Support/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/Support/BreakpointManagers/RewritePatternBreakpointManager.h"
#include "mlir/Support/BreakpointManagers/SimpleBreakpointManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// DebugExecutionContextOptions CommandLine Options
//===----------------------------------------------------------------------===//

namespace {
/// This struct contains command line options that can be used to initialize
/// various bits of a DebugExecutionContext. This uses a struct wrapper to avoid
/// the need for global command line options.
struct DebugExecutionContextOptions {
  llvm::cl::list<std::string> locations{
      "watch-at-debug-locations",
      llvm::cl::desc("Comma separated list of location arguments"),
      llvm::cl::CommaSeparated};
};
} // namespace

static llvm::ManagedStatic<DebugExecutionContextOptions> clOptions;

//===----------------------------------------------------------------------===//
// DebugExecutionContext
//===----------------------------------------------------------------------===//

DebugExecutionContext::DebugExecutionContext(
    llvm::function_ref<DebugExecutionControl(
        ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef,
        const int &, const DebugActionInformation *)>
        callback)
    : onBreakpointControlExecutionCallback(callback), daiHead(nullptr) {
  breakpointManagers.push_back(&SimpleBreakpointManager::getGlobalInstance());
  breakpointManagers.push_back(
      &RewritePatternBreakpointManager::getGlobalInstance());
  breakpointManagers.push_back(
      &FileLineColLocBreakpointManager::getGlobalInstance());
  applyCLOptions();
}

FailureOr<bool>
DebugExecutionContext::execute(ArrayRef<IRUnit> units,
                               ArrayRef<StringRef> instanceTags,
                               llvm::function_ref<ActionResult()> transform,
                               const DebugActionBase &action) {
  DebugActionInformation info{daiHead, action};
  daiHead = &info;
  ++depth;
  auto handleUserInput = [&]() -> bool {
    auto todoNext = onBreakpointControlExecutionCallback(
        units, instanceTags, action.tag, action.desc, depth, daiHead);
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
  llvm::Optional<Breakpoint *> breakpoint;
  for (auto *breakpointManager : breakpointManagers) {
    auto cur = breakpointManager->match(action, instanceTags, units);
    if (cur) {
      breakpoint = cur;
      // TODO(inkryp): Most of the times we only want to check if there is one.
      // However that is not always the case. How should we handle the existance
      // of multiple breakpoints?
      break;
    }
  }
  // For now implement it with breakpoint
  for (auto &onClientCallback : clientCallbacks) {
    onClientCallback(units, instanceTags, daiHead, depth, breakpoint);
  }
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
  daiHead = info.prev;
  return apply;
}

/// Register a set of useful command-line options that can be used to configure
/// various flags within the DebugCounter. These flags are used when
/// constructing a DebugCounter for initialization.
void DebugExecutionContext::registerCLOptions() {
#ifndef NDEBUG
  // Make sure that the options struct has been initialized.
  *clOptions;
#endif
}

// This is called by the command line parser when it sees a value for the
// watch-at-debug-locations option defined above.
// TODO(inkryp): Study behavior when this function gets called twice
void DebugExecutionContext::applyCLOptions() {
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
      llvm::errs() << "error: expected DebugExecutionContext argument to have "
                      "a `:` separating the file name and the line and column, "
                      "but provided argument was : "
                   << arg << "`\n";
      llvm::report_fatal_error(
          "Invalid DebugExecutionContext command-line configuration");
    }

    // Extract the line and column value
    int64_t locationLineInt, locationColInt;
    if (locationLineStr.getAsInteger(0, locationLineInt)) {
      llvm::errs() << "error: expected DebugExecutionContext line value to be "
                      "numeric, but got `"
                   << locationLineStr << "`\n";
      llvm::report_fatal_error(
          "Invalid DebugExecutionContext command-line configuration");
    }
    if (locationColStr.getAsInteger(0, locationColInt)) {
      llvm::errs()
          << "error: expected DebugExecutionContext column value to be "
             "numeric, but got `"
          << locationColStr << "`\n";
      llvm::report_fatal_error(
          "Invalid DebugExecutionContext command-line configuration");
    }

    // TODO(inkryp): Check for values to be positive
    FileLineColLocBreakpointManager &fileLineColLocBreakpointManager =
        FileLineColLocBreakpointManager::getGlobalInstance();
    fileLineColLocBreakpointManager.addBreakpoint(locationFile, locationLineInt,
                                                  locationColInt);
  }

  if (!clOptions->locations.empty()) {
    // TODO(inkryp): This is still a client that exists on DEC. This should be
    // its own client
    auto callback = [&](ArrayRef<IRUnit> units,
                        ArrayRef<StringRef> instanceTags,
                        const DebugActionInformation *daiHead, const int &depth,
                        llvm::Optional<Breakpoint *> breakpoint) {
      // TODO(inkryp): This is slightly wrong as it is technically possible that
      // two breakpoint of different types matched at the same time and
      // therefore the one that we are receiving might no be a
      // `FileLineColLocBreakpoint`. Find a work around or address this in a
      // different way. Should we match in here as well?
      if (breakpoint && llvm::isa<FileLineColLocBreakpoint>(*breakpoint)) {
        // TODO(inkryp): Enable a way of specifying the output
        daiHead->action.print(llvm::dbgs());
      }
    };
    clientCallbacks.push_back(callback);
  }
}
