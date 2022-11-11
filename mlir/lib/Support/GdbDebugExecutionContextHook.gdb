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

define mlir breakpoint
  set $i = 0
  while $i < $argc
    eval "call (void)mlirDebuggerAddBreakpoint($arg%d)", $i
    set $i = $i + 1
  end
end
