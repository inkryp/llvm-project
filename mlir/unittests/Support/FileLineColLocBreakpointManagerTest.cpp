//===- FileLineColLocBreakpointManagerTest.cpp - TODO: write message ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/DebugExecutionContext.h"
#include "llvm/ADT/MapVector.h"
#include "gmock/gmock.h"

using namespace mlir;

static Operation *createOp(MLIRContext *context, Location loc,
                           StringRef operationName,
                           unsigned int numRegions = 0) {
  context->allowUnregisteredDialects();
  return Operation::create(loc, OperationName(operationName, context),
                           std::nullopt, std::nullopt, std::nullopt,
                           std::nullopt, numRegions);
}

// DebugActionManager is only enabled in DEBUG mode.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS

namespace {
struct FileLineColLocTestingAction
    : public DebugAction<FileLineColLocTestingAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FileLineColLocTestingAction)
  static StringRef getTag() { return "file-line-col-loc-testing-action"; }
  static StringRef getDescription() { return "Test action for debug client"; }
};

struct FindMatchActionHandler : DebugActionManager::GenericHandler {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FindMatchActionHandler)
  FailureOr<bool> execute(ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags,
                          llvm::function_ref<ActionResult()> transform,
                          const DebugActionBase &actionBase) final {
    if (auto it = FileLineColLocBreakpointManager::getGlobalInstance().match(
            actionBase, instanceTags, units)) {
      return false;
    }
    return true;
  }
};

ActionResult noOp() { return {IRUnit(), false, success()}; }

TEST(FileLineColLocBreakpointManager, OperationMatch) {
  MLIRContext context;
  // Miscellaneous information to define operations
  std::vector<StringRef> fileNames = {
      StringRef("foo.bar"), StringRef("baz.qux"), StringRef("quux.corge")};
  std::vector<std::pair<unsigned, unsigned>> lineColLoc = {{42, 7}, {24, 3}};
  Location callee = UnknownLoc::get(&context),
           caller = UnknownLoc::get(&context), loc = UnknownLoc::get(&context);

  // Set of operations over where we are going to be testing the functionality
  std::vector<Operation *> operations = {
      createOp(&context, CallSiteLoc::get(callee, caller),
               "callSiteLocOperation"),
      createOp(&context,
               FileLineColLoc::get(&context, fileNames[0], lineColLoc[0].first,
                                   lineColLoc[0].second),
               "fileLineColLocOperation"),
      createOp(&context, FusedLoc::get(&context, {}, Attribute()),
               "fusedLocOperation"),
      createOp(&context, NameLoc::get(StringAttr::get(&context, fileNames[2])),
               "nameLocOperation"),
      createOp(&context, OpaqueLoc::get<void *>(nullptr, loc),
               "opaqueLocOperation"),
      createOp(&context, UnknownLoc::get(&context), "unknownLocOperation"),
      createOp(&context,
               FileLineColLoc::get(&context, fileNames[1], lineColLoc[1].first,
                                   lineColLoc[1].second),
               "anotherFileLineColLocOperation"),
  };

  // TODO: Rewrite test to not rely on a `DebugActionManager`.
  // Preferably not rely on
  // `FileLineColLocBreakpointManager::getGlobalInstance()` either.
  DebugActionManager &manager = context.getDebugActionManager();
  manager.registerActionHandler<FindMatchActionHandler>();

  for (auto *op : operations) {
    EXPECT_TRUE(succeeded(
        manager.execute<FileLineColLocTestingAction>({op}, {}, noOp)));
  }

  auto &fileLineColLocBreakpointManager =
      FileLineColLocBreakpointManager::getGlobalInstance();

  auto *breakpoint = fileLineColLocBreakpointManager.addBreakpoint(
      fileNames[0], lineColLoc[0].first, lineColLoc[0].second);
  for (int i = 0; i < (int)operations.size(); ++i) {
    auto result =
        manager.execute<FileLineColLocTestingAction>({operations[i]}, {}, noOp);
    if (i == 1) {
      EXPECT_FALSE(succeeded(result));
    } else {
      EXPECT_TRUE(succeeded(result));
    }
  }

  fileLineColLocBreakpointManager.disableBreakpoint(breakpoint);
  for (auto *op : operations) {
    EXPECT_TRUE(succeeded(
        manager.execute<FileLineColLocTestingAction>({op}, {}, noOp)));
  }

  auto *randomBreakpoint = fileLineColLocBreakpointManager.addBreakpoint(
      StringRef("random.file"), 3, 14);
  for (auto *op : operations) {
    EXPECT_TRUE(succeeded(
        manager.execute<FileLineColLocTestingAction>({op}, {}, noOp)));
  }
  fileLineColLocBreakpointManager.deleteBreakpoint(randomBreakpoint);

  fileLineColLocBreakpointManager.addBreakpoint(
      fileNames[1], lineColLoc[1].first, lineColLoc[1].second);
  for (int i = 0; i < (int)operations.size(); ++i) {
    auto result =
        manager.execute<FileLineColLocTestingAction>({operations[i]}, {}, noOp);
    if (i == 6) {
      EXPECT_FALSE(succeeded(result));
    } else {
      EXPECT_TRUE(succeeded(result));
    }
  }

  fileLineColLocBreakpointManager.enableBreakpoint(breakpoint);
  for (int i = 0; i < (int)operations.size(); ++i) {
    auto result =
        manager.execute<FileLineColLocTestingAction>({operations[i]}, {}, noOp);
    if (i == 1 || i == 6) {
      EXPECT_FALSE(succeeded(result));
    } else {
      EXPECT_TRUE(succeeded(result));
    }
  }

  for (auto *op : operations) {
    op->destroy();
  }
}

TEST(FileLineColLocBreakpointManager, BlockMatch) {
  MLIRContext context;
  std::vector<StringRef> fileNames = {StringRef("grault.garply"),
                                      StringRef("waldo.fred")};
  std::vector<std::pair<unsigned, unsigned>> lineColLoc = {{42, 7}, {24, 3}};
  Operation *frontOp = createOp(&context,
                                FileLineColLoc::get(&context, fileNames.front(),
                                                    lineColLoc.front().first,
                                                    lineColLoc.front().second),
                                "firstOperation");
  Operation *backOp = createOp(&context,
                               FileLineColLoc::get(&context, fileNames.back(),
                                                   lineColLoc.back().first,
                                                   lineColLoc.back().second),
                               "secondOperation");
  Block block;
  block.push_back(frontOp);
  block.push_back(backOp);

  DebugActionManager &manager = context.getDebugActionManager();
  manager.registerActionHandler<FindMatchActionHandler>();
  EXPECT_TRUE(succeeded(
      manager.execute<FileLineColLocTestingAction>({&block}, {}, noOp)));

  auto &fileLineColLocBreakpointManager =
      FileLineColLocBreakpointManager::getGlobalInstance();
  auto *breakpoint = fileLineColLocBreakpointManager.addBreakpoint(
      fileNames.front(), lineColLoc.front().first, lineColLoc.front().second);
  EXPECT_FALSE(succeeded(
      manager.execute<FileLineColLocTestingAction>({&block}, {}, noOp)));
  fileLineColLocBreakpointManager.deleteBreakpoint(breakpoint);
  EXPECT_TRUE(succeeded(
      manager.execute<FileLineColLocTestingAction>({&block}, {}, noOp)));
  breakpoint = fileLineColLocBreakpointManager.addBreakpoint(
      fileNames.back(), lineColLoc.back().first, lineColLoc.back().second);
  EXPECT_FALSE(succeeded(
      manager.execute<FileLineColLocTestingAction>({&block}, {}, noOp)));
  fileLineColLocBreakpointManager.deleteBreakpoint(breakpoint);
  EXPECT_TRUE(succeeded(
      manager.execute<FileLineColLocTestingAction>({&block}, {}, noOp)));
}

TEST(FileLineColLocBreakpointManager, RegionMatch) {
  MLIRContext context;
  StringRef fileName("plugh.xyzzy");
  unsigned line = 42, col = 7;
  Operation *containerOp =
      createOp(&context, FileLineColLoc::get(&context, fileName, line, col),
               "containerOperation", 1);
  Region &region = containerOp->getRegion(0);

  DebugActionManager &manager = context.getDebugActionManager();
  manager.registerActionHandler<FindMatchActionHandler>();
  EXPECT_TRUE(succeeded(
      manager.execute<FileLineColLocTestingAction>({&region}, {}, noOp)));
  auto &fileLineColLocBreakpointManager =
      FileLineColLocBreakpointManager::getGlobalInstance();
  auto *breakpoint =
      fileLineColLocBreakpointManager.addBreakpoint(fileName, line, col);
  EXPECT_FALSE(succeeded(
      manager.execute<FileLineColLocTestingAction>({&region}, {}, noOp)));
  fileLineColLocBreakpointManager.deleteBreakpoint(breakpoint);
  EXPECT_TRUE(succeeded(
      manager.execute<FileLineColLocTestingAction>({&region}, {}, noOp)));

  containerOp->destroy();
}
} // namespace

#endif
