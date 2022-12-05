//===- BreakpointManager.h - Breakpoint Manager Support ----*- C++ -*-===//
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

#ifndef MLIR_SUPPORT_BREAKPOINTMANAGERS_BREAKPOINTMANAGER_H
#define MLIR_SUPPORT_BREAKPOINTMANAGERS_BREAKPOINTMANAGER_H

#include "mlir/Support/DebugAction.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

/// This class represents the base class of a breakpoint.
class BreakpointBase {
public:
  virtual ~BreakpointBase() = default;

  /// Return the unique breakpoint id of this breakpoint, use for casting
  /// functionality.
  TypeID getBreakpointID() const { return breakpointID; }

protected:
  BreakpointBase(TypeID breakpointID)
      : breakpointID(breakpointID), enableStatus(true) {}

  /// The type of the derived breakpoint class. This allows for detecting the
  /// specific handler of a given breakpoint type.
  TypeID breakpointID;

  bool getEnableStatus() const { return enableStatus; }

  void setEnableStatusTrue() { enableStatus = true; }

  void setEnableStatusFalse() { enableStatus = false; }

private:
  /// The current state of the breakpoint. A breakpoint can be either enabled
  /// or disabled.
  bool enableStatus;

  /// Allow access to `enableStatus` operations.
  friend class BreakpointManagerBase;
};

/// This class represents the base class of a breakpoint manager.
class BreakpointManagerBase {
public:
  virtual ~BreakpointManagerBase() = default;

  /// Return the unique breakpoint manager id of this breakpoint manager, use
  /// for casting functionality.
  TypeID getBreakpointManagerID() const { return breakpointManagerID; }

  virtual BreakpointBase *addBreakpoint(StringRef tag) = 0;

  virtual void deleteBreakpoint(BreakpointBase *breakpoint) = 0;

  virtual llvm::Optional<BreakpointBase *> match(const StringRef &tag) = 0;

  void enableBreakpoint(BreakpointBase *breakpoint) {
    breakpoint->setEnableStatusTrue();
  }

  void disableBreakpoint(BreakpointBase *breakpoint) {
    breakpoint->setEnableStatusFalse();
  }

protected:
  BreakpointManagerBase(TypeID breakpointManagerID)
      : breakpointManagerID(breakpointManagerID) {}

  /// The type of the derived breakpoint manager class. This allows for
  /// detecting the specific handler of a given breakpoint type.
  TypeID breakpointManagerID;
};

} // namespace mlir

#endif // MLIR_SUPPORT_BREAKPOINTMANAGERS_BREAKPOINTMANAGER_H
