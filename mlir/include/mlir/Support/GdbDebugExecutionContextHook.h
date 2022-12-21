//===- GdbDebugExecutionContextHook.h - GDB for Debugger Support *- C++ -*-===//
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

#ifndef MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H
#define MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H

#include "mlir/Support/DebugExecutionContext.h"
#include "llvm/Support/Compiler.h"

extern "C" {
void mlirDebuggerSetControl(int controlOption);

void mlirDebuggerAddSimpleBreakpoint(const char *tag);

void mlirDebuggerAddRewritePatternBreakpoint(const char *patternNameInfo);

void mlirDebuggerAddFileLineColLocBreakpoint(const char *file, unsigned line,
                                             unsigned col);

/// Deletes a breakpoint based on its ID. True indicates a value was deleted.
/// When deleting is agnostic of the specific type of `Breakpoint`.
bool mlirDebuggerDeleteBreakpoint(unsigned id);

bool mlirDebuggerChangeStatusOfBreakpoint(unsigned breakpointID, bool status);

bool mlirDebuggerListBreakpoints();

bool mlirDebuggerPrintAction();

void mlirDebuggerPrintIRUnit(const void *);

bool mlirDebuggerShowContext();

const void *mlirDebuggerRetrieveIRUnit(unsigned);
}

namespace mlir {

class GdbDebugExecutionContextInformation {
public:
  static GdbDebugExecutionContextInformation &getGlobalInstance();
  void updateContents(ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef,
                      StringRef, const int &, const DebugActionInformation *);
  const DebugActionInformation *getDebugActionInformation() { return daiHead; }
  const ArrayRef<IRUnit> &getArrayOfIRUnits() { return units; }

private:
  ArrayRef<IRUnit> units;
  ArrayRef<StringRef> instanceTags;
  StringRef tag;
  StringRef desc;
  int depth;
  const DebugActionInformation *daiHead;
  int idxActiveUnit;
  IRUnit activeUnit;
};

DebugExecutionControl
GdbCallBackFunction(ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
                    StringRef something, StringRef other, const int &depth,
                    const DebugActionInformation *daiHead);

} // namespace mlir

#endif // MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H
