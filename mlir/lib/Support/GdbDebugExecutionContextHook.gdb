define-prefix mlir

define mlir apply
  call (void)mlirDebuggerSetControl(1)
  continue
end

define mlir skip
  call (void)mlirDebuggerSetControl(2)
  continue
end

define mlir step
  call (void)mlirDebuggerSetControl(3)
  continue
end

define mlir next
  call (void)mlirDebuggerSetControl(4)
  continue
end

define mlir finish
  call (void)mlirDebuggerSetControl(5)
  continue
end

define mlir simpleBreakpoint
  set $i = 0
  while $i < $argc
    eval "call (void)mlirDebuggerAddSimpleBreakpoint($arg%d)", $i
    set $i = $i + 1
  end
end

define mlir patternBreakpoint
  set $i = 0
  while $i < $argc
    eval "call (void)mlirDebuggerAddRewritePatternBreakpoint($arg%d)", $i
    set $i = $i + 1
  end
end

# TODO: Find a way to enforce user to have the correct types on the arguments
define mlir locationBreakpoint
  if $argc == 2
    call (void)mlirDebuggerAddFileLineColLocBreakpoint($arg0, $arg1, 0)
  else
    if $argc == 3
      call (void)mlirDebuggerAddFileLineColLocBreakpoint($arg0, $arg1, $arg2)
    end
  end
end

# TODO: Support parsing to a string to represent a range.
define mlir deleteBreakpoint
  if $argc == 1
    # In the original signature of `mlirDebuggerDeleteBreakpoint` it returns a
    # `bool`. However, inside the call to SIGTRAP our program might end on a C
    # context. As `bool` is not a C type, we have to cast it to something
    # supported, in this case `int` and then do the conversion with bitwise
    # operations to have the behavior of a `bool`.
    # TODO: Find a way to retrieve the bool value directly.
    set $mlirDebuggerDeleteBreakpointResult = \
        ((int (*)(unsigned))mlirDebuggerDeleteBreakpoint)($arg0)
    if !($mlirDebuggerDeleteBreakpointResult & 1)
      printf "Could not find Breakpoint with ID %d\n", $arg0
    end
  end
end
