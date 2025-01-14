//===- DebugAction.h - Debug Action Support ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the debug action framework. This framework
// allows for external entities to control certain actions taken by the compiler
// by registering handler functions. A debug action handler provides the
// internal implementation for the various queries on a debug action, such as
// whether it is going to execute or not.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_DEBUGACTION_H
#define MLIR_SUPPORT_DEBUGACTION_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <type_traits>

namespace mlir {

/// Element of the IR to be considered during a transformation
using IRUnit = std::variant<Operation *, Block *, Region *>;

/// Information about what happened on the transformation
struct ActionResult {
  IRUnit unit;
  bool changed;
  LogicalResult status;
};

/// This class represents the base class of a debug action.
class DebugActionBase {
public:
  virtual ~DebugActionBase() = default;

  /// Return the unique action id of this action, use for casting
  /// functionality.
  TypeID getActionID() const { return actionID; }

  StringRef tag;

  StringRef desc;

  virtual void print(raw_ostream &os) const {
    os << "In action: `" << tag << "`\n" << desc << '\n';
  }

protected:
  DebugActionBase(TypeID actionID, StringRef tag, StringRef desc)
      : tag(tag), desc(desc), actionID(actionID) {}

  /// The type of the derived action class. This allows for detecting the
  /// specific handler of a given action type.
  TypeID actionID;
};

//===----------------------------------------------------------------------===//
// DebugActionManager
//===----------------------------------------------------------------------===//

/// This class represents manages debug actions, and orchestrates the
/// communication between action queries and action handlers. An action handler
/// is either an action specific handler, i.e. a derived class of
/// `MyActionType::Handler`, or a generic handler, i.e. a derived class of
/// `DebugActionManager::GenericHandler`. For more details on action specific
/// handlers, see the definition of `DebugAction::Handler` below. For more
/// details on generic handlers, see `DebugActionManager::GenericHandler` below.
class DebugActionManager {
public:
  //===--------------------------------------------------------------------===//
  // Handlers
  //===--------------------------------------------------------------------===//

  /// This class represents the base class of a debug action handler.
  class HandlerBase {
  public:
    virtual ~HandlerBase() = default;

    /// Return the unique handler id of this handler, use for casting
    /// functionality.
    TypeID getHandlerID() const { return handlerID; }

  protected:
    HandlerBase(TypeID handlerID) : handlerID(handlerID) {}

    /// The type of the derived handler class. This allows for detecting if a
    /// handler can handle a given action type.
    TypeID handlerID;
  };

  /// This class represents a generic action handler. A generic handler allows
  /// for handling any action type. Handlers of this type are useful for
  /// implementing general functionality that doesn't necessarily need to
  /// interpret the exact action parameters, or can rely on an external
  /// interpreter (such as the user). Given that these handlers are generic,
  /// they take a set of opaque parameters that try to map the context of the
  /// action type in a generic way.
  class GenericHandler : public HandlerBase {
  public:
    GenericHandler() : HandlerBase(TypeID::get<GenericHandler>()) {}

    /// This hook allows for controlling whether an action should execute or
    /// not. It should return failure if the handler could not process the
    /// action, passing it to the next registered handler.
    virtual FailureOr<bool>
    execute(ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
            llvm::function_ref<ActionResult()> transform,
            const DebugActionBase &action) {
      return failure();
    }

    /// Provide classof to allow casting between handler types.
    static bool classof(const DebugActionManager::HandlerBase *handler) {
      return handler->getHandlerID() == TypeID::get<GenericHandler>();
    }
  };

  /// Register the given action handler with the manager.
  void registerActionHandler(std::unique_ptr<HandlerBase> handler) {
    // The manager is always disabled if built without debug.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    actionHandlers.emplace_back(std::move(handler));
#endif
  }
  template <typename T, typename... Args>
  void registerActionHandler(Args &&...args) {
    registerActionHandler(std::make_unique<T>(std::forward<Args>(args)...));
  }

  //===--------------------------------------------------------------------===//
  // Action Queries
  //===--------------------------------------------------------------------===//

  /// Find a handler to execute an action provided with `transform` and
  /// identified with `instanceTag` on the list of IRUnit `units`. The list of
  /// extra `args` matches the expectation of `ActionType`. If no handler is
  /// found the `transform` callback is invoked directly.
  template <typename ActionType, typename... Args>
  LogicalResult
  execute(ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
          llvm::function_ref<ActionResult()> transform, Args &&...args) {
    // The manager is always disabled if built without debug.
#if !LLVM_ENABLE_ABI_BREAKING_CHECKS
    return true;
#else
    // Invoke the `execute` method on the provided handler.
    auto executeFn = [&](auto *handler, auto &&...handlerParams) {
      return handler->execute(
          units, instanceTags, transform,
          ActionType(std::forward<decltype(handlerParams)>(handlerParams)...));
    };
    FailureOr<bool> result = dispatchToHandler<ActionType, bool>(
        executeFn, std::forward<Args>(args)...);
    // Execute transformation if no handler was defined.
    if (failed(result)) {
      transform();
    }
    // Could only have been executed if was dispatched to handler and handler
    // returned true, or if it was not dispatched to any handler.
    bool executed = (succeeded(result) && *result) || failed(result);

    // Whether the action was executed or not
    return success(executed);
#endif
  }

private:
// The manager is always disabled if built without debug.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  //===--------------------------------------------------------------------===//
  // Query to Handler Dispatch
  //===--------------------------------------------------------------------===//

  /// Dispath a given callback on any handlers that are able to process queries
  /// on the given action type. This method returns failure if no handlers could
  /// process the action, or success(with a result) if a handler processed the
  /// action.
  template <typename ActionType, typename ResultT, typename HandlerCallbackT,
            typename... Args>
  FailureOr<ResultT> dispatchToHandler(HandlerCallbackT &&handlerCallback,
                                       Args &&...args) {
    static_assert(ActionType::template canHandleWith<Args...>(),
                  "cannot execute action with the given set of parameters");

    // Process any generic or action specific handlers.
    // TODO: We currently just pick the first handler that gives us a result,
    // but in the future we may want to employ a reduction over all of the
    // values returned.
    for (std::unique_ptr<HandlerBase> &it : llvm::reverse(actionHandlers)) {
      FailureOr<ResultT> result = failure();
      if (auto *handler = dyn_cast<typename ActionType::Handler>(&*it)) {
        result = handlerCallback(handler, std::forward<Args>(args)...);
      } else if (auto *genericHandler = dyn_cast<GenericHandler>(&*it)) {
        result = handlerCallback(genericHandler, std::forward<Args>(args)...);
      }

      // If the handler succeeded, return the result. Otherwise, try a new
      // handler.
      if (succeeded(result))
        return result;
    }
    return failure();
  }

  /// The set of action handlers that have been registered with the manager.
  SmallVector<std::unique_ptr<HandlerBase>> actionHandlers;
#endif
};

//===----------------------------------------------------------------------===//
// DebugAction
//===----------------------------------------------------------------------===//

/// A debug action is a specific action that is to be taken by the compiler,
/// that can be toggled and controlled by an external user. There are no
/// constraints on the granularity of an action, it could be as simple as
/// "perform this fold" and as complex as "run this pass pipeline". Via template
/// parameters `ParameterTs`, a user may provide the set of argument types that
/// are provided when handling a query on this action. Derived classes are
/// expected to provide the following:
///   * static llvm::StringRef getTag()
///     - This method returns a unique string identifier, similar to a command
///       line flag or DEBUG_TYPE.
///   * static llvm::StringRef getDescription()
///     - This method returns a short description of what the action represents.
///
/// This class provides a handler class that can be derived from to handle
/// instances of this action. The parameters to its query methods map 1-1 to the
/// types on the action type.
template <typename Derived, typename... ParameterTs>
class DebugAction : public DebugActionBase {
public:
  DebugAction()
      : DebugActionBase(TypeID::get<Derived>(), Derived::getTag(),
                        Derived::getDescription()) {}

  /// Provide classof to allow casting between action types.
  static bool classof(const DebugActionBase *action) {
    return action->getActionID() == TypeID::get<Derived>();
  }

  class Handler : public DebugActionManager::HandlerBase {
  public:
    Handler() : HandlerBase(TypeID::get<Derived>()) {}

    /// This hook allows for controlling whether an action should execute or
    /// not. `parameters` correspond to the set of values provided by the
    /// action as context. It should return failure if the handler could not
    /// process the action, passing it to the next registered handler.
    virtual FailureOr<bool>
    execute(ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
            llvm::function_ref<ActionResult()> transform,
            const Derived &action) {
      return failure();
    }

    /// Provide classof to allow casting between handler types.
    static bool classof(const DebugActionManager::HandlerBase *handler) {
      return handler->getHandlerID() == TypeID::get<Derived>();
    }
  };

private:
  /// Returns true if the action can be handled within the given set of
  /// parameter types.
  template <typename... CallerParameterTs>
  static constexpr bool canHandleWith() {
    return std::is_invocable_v<function_ref<void(ParameterTs...)>,
                               CallerParameterTs...>;
  }

  /// Allow access to `canHandleWith`.
  friend class DebugActionManager;
};

} // namespace mlir

#endif // MLIR_SUPPORT_DEBUGACTION_H
