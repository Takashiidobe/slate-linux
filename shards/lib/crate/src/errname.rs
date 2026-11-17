
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! Operation not permitted
//! Resource deadlock would occur
//! These should never be seen by user programs.  To return one of ERESTART*
//! codes, signal_pending() MUST be set.  Note that ptrace can observe these
//! at syscall exit tracing, but they will never be left for the debugged user
//! process to see.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! check_add_overflow() - Calculate addition with overflow checking
//! @a: first addend
//! @b: second addend
//! @d: pointer to store sum
//!
//! Returns true on wrap-around, false otherwise.
//!
//! *@d holds the results of the attempted addition, regardless of whether
//! wrap-around occurred.
//! See linux/instrumentation.h
//! 64 bit
//! Begin/end of an instrumentation safe region
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
//! label
//! const_*() definitions provide good compile-time optimizations when
//! the passed arguments can be resolved at compile time.
//! Many architecture-specific non-atomic bitops contain inline asm code and due
//! to that the compiler can't optimize them to compile-time expressions or
//! constants. In contrary, generic_*() helpers are defined in pure C and
//! compilers optimize them just well.
//! Therefore, to make `unsigned long foo = 0; __set_bit(BAR, &foo)` effectively
//! equal to `unsigned long foo = BIT(BAR)`, pick the generic C alternative when
//! the arguments can be resolved at compile time. That expression itself is a
//! constant and doesn't bring any functional changes to the rest of cases.
//! The casts to `uintptr_t` are needed to mitigate `-Waddress` warnings when
//! passing a bitmap from .bss or .data (-> `!!addr` is always true).
//! This counts to 15. Any more, it will return 16th argument.
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! "popcnt" POPCNT instruction
//! Compile time versions of __arch_hweightN()
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
// This header exists to force full rebuild when the compiler is upgraded.
//
// When fixdep scans this, it will find this string "CONFIG_CC_VERSION_TEXT"
// and add dependency on include/config/CC_VERSION_TEXT, which is touched
// by Kconfig when the version string from the compiler changes.
// Additional tree-wide dependencies start here.
// If any of the GCC plugins change, we need to rebuild everything that
// was built with them, as they may have changed their behavior and those
// behaviors may need to be synchronized across all translation units.
// If the randstruct seed itself changes (whether for GCC plugins or
// Clang), the entire tree needs to be rebuilt since the randomization of
// structures may change between compilation units if not.
// If any external changes affect Clang's integer wrapping sanitizer
// behavior, a full rebuild is needed as the coverage for wrapping types
// may have changed, which may impact the expected behaviors that should
// not differ between compilation units.
// The use of "&&" / "||" is limited in certain expressions.
// The following enable to calculate "and" / "or" with macro expansion only.
// Helper macros to use CONFIG_ options in C/CPP expressions. Note that
// these only work with boolean and tristate options.
// IS_BUILTIN(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y', 0
// otherwise. For boolean options, this is equivalent to
// IS_ENABLED(CONFIG_FOO).
// IS_MODULE(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'm', 0
// otherwise.  CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1" in
// autoconf.h.
// IS_REACHABLE(CONFIG_FOO) evaluates to 1 if the currently compiled
// code can call a function defined in code compiled based on CONFIG_FOO.
// This is similar to IS_ENABLED(), but returns false when invoked from
// built-in code when CONFIG_FOO is set to 'm'.
// __has_builtin is supported on gcc >= 10, clang >= 3 and icc >= 21.
// In the meantime, to support gcc < 10, we implement __has_builtin
// by hand.
// Indirect macros required for expanded argument pasting, eg. __LINE__.
// C23 introduces "auto" as a standard way to define type-inferred
// variables, but "auto" has been a (useless) keyword even since K&R C,
// so it has always been "namespace reserved."
//
// Until at some future time we require C23 support, we need the gcc
// extension __auto_type, but there is no reason to put that elsewhere
// in the source code.
// nothing
// sparse defines __CHECKER__; see Documentation/dev-tools/sparse.rst
//
// address spaces
// other
// Attributes
// Note: users of __always_inline currently do not write "inline" themselves,
// which seems to be required by gcc to apply the attribute according
// to its docs (and also "warning: always_inline function might not be
// inlinable [-Wattributes]" is emitted).
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-always_005finline-function-attribute
// clang: mentioned
// Note the long name.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-format-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#format
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noreturn-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#noreturn
// clang: https://clang.llvm.org/docs/AttributeReference.html#id1
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-packed-type-attribute
// clang: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-packed-variable-attribute
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-section-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-section-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#section-declspec-allocate
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-warn_005funused_005fresult-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#nodiscard-warn-unused-result
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-cold-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Label-Attributes.html#index-cold-label-attribute
//
// When -falign-functions=N is in use, we must avoid the cold attribute as
// GCC drops the alignment for cold functions. Worse, GCC can implicitly mark
// callees of cold functions as cold themselves, so it's not sufficient to add
// __function_aligned here as that will not ensure that callees are correctly
// aligned.
//
// See:
//
//   https://lore.kernel.org/lkml/Y77%2FqVgvaJidFpYt@FVFF77S0Q05N
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88345#c9
// On x86-64 and arm64 targets, __preserve_most changes the calling convention
// of a function to make the code in the caller as unintrusive as possible. This
// convention behaves identically to the C calling convention on how arguments
// and return values are passed, but uses a different set of caller- and callee-
// saved registers.
//
// The purpose is to alleviates the burden of saving and recovering a large
// register set before and after the call in the caller.  This is beneficial for
// rarely taken slow paths, such as error-reporting functions that may be called
// from hot paths.
//
// Note: This may conflict with instrumentation inserted on function entry which
// does not use __preserve_most or equivalent convention (if in assembly). Since
// function tracing assumes the normal C calling convention, where the attribute
// is supported, __preserve_most implies notrace.  It is recommended to restrict
// use of the attribute to functions that should or already disable tracing.
//
// Optional: not supported by gcc.
//
// clang: https://clang.llvm.org/docs/AttributeReference.html#preserve-most
// Compiler specific macros.
// Some architectures need to provide custom definitions of macros provided
// by linux/compiler-*.h, and can do so using asm/compiler.h. We include that
// conditionally rather than using an asm-generic wrapper in order to avoid
// build failures if any C compilation, which will include this file via an
// -include argument in c_flags, occurs prior to the asm-generic wrappers being
// generated.
// it doesn't make sense on ARM (currently the only user of __naked)
// to trace naked functions because then mcount is called without
// stack and frame pointer being set up and there is no chance to
// restore the lr register to the value before mcount was called.
// Prefer gnu_inline, so that extern inline functions do not emit an
// externally visible function. This makes extern inline behave as per gnu89
// semantics rather than c99. This prevents multiple symbol definition errors
// of extern inline functions at link time.
// A lot of inline functions can cause havoc with function tracing.
// gcc provides both __inline__ and __inline as alternate spellings of
// the inline keyword, though the latter is undocumented. New kernel
// code should only use the inline spelling, but some existing code
// uses __inline__. Since we #define inline above, to ensure
// __inline__ has the same semantics, we need this #define.
//
// However, the spelling __inline is strictly reserved for referring
// to the bare keyword.
// GCC does not warn about unused static inline functions for -Wunused-function.
// Suppress the warning in clang as well by using __maybe_unused, but enable it
// for W=2 build. This will allow clang to find unused functions.
// Rather then using noinline to prevent stack consumption, use
// noinline_for_stack instead.  For documentation reasons.
// Use noinline_for_tracing for functions that should not be inlined.
// For tracing reasons.
// Sanitizer helper attributes: Because using __always_inline and
// __no_sanitize_* conflict, provide helper attributes that will either expand
// to __no_sanitize_* in compilation units where instrumentation is enabled
// (__SANITIZE_*__), or __always_inline in compilation units without
// instrumentation (__SANITIZE_*__ undefined).
//
// The assume attribute is used to indicate that a certain condition is
// assumed to be true. If this condition is violated at runtime, the behavior
// is undefined. Compilers may or may not use this indication to generate
// optimized code.
//
// Note that the clang documentation states that optimizers may react
// differently to this attribute, and this may even have a negative
// performance impact. Therefore this attribute should be used with care.
//
// Optional: only supported since gcc >= 13
// Optional: only supported since clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Statement-Attributes.html#index-assume-statement-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#id13
// Optional: only supported since gcc >= 15
// Optional: only supported since clang >= 18
//
//   gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108896
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
//
// __bdos on clang < 19.1.2 can erroneously return 0:
// https://github.com/llvm/llvm-project/pull/110497
//
// __bdos on clang < 19.1.3 can be off by 4:
// https://github.com/llvm/llvm-project/pull/112636
// Runtime track number of objects pointed to by a pointer member for use by
// CONFIG_FORTIFY_SOURCE and CONFIG_UBSAN_BOUNDS.
//
// Optional: only supported since gcc >= 16
// Optional: only supported since clang >= 22
//
//   gcc: https://gcc.gnu.org/pipermail/gcc-patches/2025-April/681727.html
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
// Optional: only supported since gcc >= 15
// Optional: not supported by Clang
//
// gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=117178
// Apply __counted_by() when the Endianness matches to increase test coverage.
// This designates the minimum number of elements a passed array parameter must
// have. For example:
//
//     void some_function(u8 param[at_least 7]);
//
// If a caller passes an array with fewer than 7 elements, the compiler will
// emit a warning.
// Section for code which can't be instrumented at all
// The __cpuidle section is used twofold:
//
//  1) the original use -- identifying if a CPU is 'stuck' in idle state based
//     on it's instruction pointer. See cpu_in_idle().
//
//  2) supressing instrumentation around where cpuidle disables RCU; where the
//     function isn't strictly required for #1, this is interchangeable with
//     noinstr.
// The below symbols may be defined for one or more, but not ALL, of the above
// compilers. We don't consider that to be an error, so set them to nothing.
// For example, some of them are for compiler specific plugins.
// Any place that could be marked with the "alloc_size" attribute is also
// a place to be marked with the "malloc" attribute, except those that may
// be performing a _reallocation_, as that may alias the existing pointer.
// For these, use __realloc_size().
// When the size of an allocated object is needed, use the best available
// mechanism to find it. (For cases where sizeof() cannot be used.)
//
// Optional: only supported since gcc >= 12
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Object-Size-Checking.html
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#evaluating-object-size
// Determine if an attribute has been applied to a variable.
// Using __annotated needs to check for __annotated being available,
// or negative tests may fail when annotation cannot be checked. For
// example, see the definition of __is_cstr().
// __flex_counter() - Get pointer to counter member for the given
//                    flexible array, if it was annotated with __counted_by()
// @FAM: Pointer to flexible array member of an addressable struct instance
//
// For example, with:
//
// 	struct foo {
// 		int counter;
// 		short array[] __counted_by(counter);
// 	} *p;
//
// __flex_counter(p->array) will resolve to &p->counter.
//
// Note that Clang may not allow this to be assigned to a separate
// variable; it must be used directly.
//
// If p->array is unannotated, this returns (void *)NULL.
// Some versions of gcc do not mark 'asm goto' volatile:
//
//  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103979
//
// We do it here by hand, because it doesn't hurt.
// Clang has trouble with constraints with multiple
// alternative behaviors ("g" , "rm" and "=rm").
// Use __typeof_unqual__() when available.
// Are two types/vars the same type (ignoring qualifiers)?
// __unqual_scalar_typeof(x) - Declare an unqualified scalar type, leaving
// 			       non-scalar types unchanged.
// __signed_scalar_typeof(x) - Declare a signed scalar type, leaving
// 			       non-scalar types unchanged.
// Is this type a native word size -- useful for atomic operations
// #ifdef __OPTIMIZE__ is only a good approximation; for instance "make
// CFLAGS_foo.o=-Og" defines __OPTIMIZE__, does not elide the conditional code
// and can break compilation with wrong error message(s). Combine with
// -U__OPTIMIZE__ when needed.
// 									 * __noreturn is needed to give the compiler enough			 * information to avoid certain possibly-uninitialized			 * warnings (regardless of the build failing).
// compiletime_assert - break build and emit msg if condition is false
// @condition: a compile-time constant condition to check
// @msg:       a message to emit if condition is false
//
// In tradition of POSIX assert, this macro will break the build if the
// supplied condition is *false*, emitting the supplied error message if the
// compiler has support to do so.
// Helpers for emitting diagnostics in pragmas.
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
// &a[0] degrades to a pointer: a different type from an array
// Force the compiler to emit 'sym' as a symbol, so that we can reference
// it from inline assembler. Necessary in case 'sym' could be inlined
// otherwise, or eliminated entirely due to lack of references that are
// visible to the compiler.
// Useful shorthand for "is this condition known at compile-time?"
//
// Note that the condition may involve non-constant values,
// but the compiler may know enough about the details of the
// values to determine that the condition is statically true.
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_read_write - check regular read-write access for races
//
// @ptr: address of access
// @size: size of access
// static_assert - check integer constant expression at build time
// @expr: expression to be checked
//
// static_assert() is a wrapper for the C11 _Static_assert, with a
// little macro magic to make the message optional (defaulting to the
// stringification of the tested expression).
//
// Contrary to BUILD_BUG_ON(), static_assert() can be used at global
// scope, but requires the expression to be an integer constant
// expression (i.e., it is not enough that __builtin_constant_p() is
// true for expr).
//
// Also note that BUILD_BUG_ON() fails the build if the condition is
// true, while static_assert() fails the build if the expression is
// false.
// No such file or directory
// No such process
// Interrupted system call
// I/O error
// No such device or address
// Argument list too long
// Exec format error
// Bad file number
// No child processes
// Try again
// Out of memory
// Permission denied
// Bad address
// Block device required
// Device or resource busy
// File exists
// Cross-device link
// No such device
// Not a directory
// Is a directory
// Invalid argument
// File table overflow
// Too many open files
// Not a typewriter
// Text file busy
// File too large
// No space left on device
// Illegal seek
// Read-only file system
// Too many links
// Broken pipe
// Math argument out of domain of func
// Math result not representable
// File name too long
// No record locks available
// This error code is special: arch syscall entry code will return
// -ENOSYS if users try to call a syscall that doesn't exist.  To keep
// failures of syscalls that really do exist distinguishable from
// failures due to attempts to use a nonexistent syscall, syscall
// implementations should refrain from returning -ENOSYS.
// Invalid system call number
// Directory not empty
// Too many symbolic links encountered
// Operation would block
// No message of desired type
// Identifier removed
// Channel number out of range
// Level 2 not synchronized
// Level 3 halted
// Level 3 reset
// Link number out of range
// Protocol driver not attached
// No CSI structure available
// Level 2 halted
// Invalid exchange
// Invalid request descriptor
// Exchange full
// No anode
// Invalid request code
// Invalid slot
// Bad font file format
// Device not a stream
// No data available
// Timer expired
// Out of streams resources
// Machine is not on the network
// Package not installed
// Object is remote
// Link has been severed
// Advertise error
// Srmount error
// Communication error on send
// Protocol error
// Multihop attempted
// RFS specific error
// Not a data message
// Value too large for defined data type
// Name not unique on network
// File descriptor in bad state
// Remote address changed
// Can not access a needed shared library
// Accessing a corrupted shared library
// .lib section in a.out corrupted
// Attempting to link in too many shared libraries
// Cannot exec a shared library directly
// Illegal byte sequence
// Interrupted system call should be restarted
// Streams pipe error
// Too many users
// Socket operation on non-socket
// Destination address required
// Message too long
// Protocol wrong type for socket
// Protocol not available
// Protocol not supported
// Socket type not supported
// Operation not supported on transport endpoint
// Protocol family not supported
// Address family not supported by protocol
// Address already in use
// Cannot assign requested address
// Network is down
// Network is unreachable
// Network dropped connection because of reset
// Software caused connection abort
// Connection reset by peer
// No buffer space available
// Transport endpoint is already connected
// Transport endpoint is not connected
// Cannot send after transport endpoint shutdown
// Too many references: cannot splice
// Connection timed out
// Connection refused
// Host is down
// No route to host
// Operation already in progress
// Operation now in progress
// Stale file handle
// Structure needs cleaning
// Not a XENIX named type file
// No XENIX semaphores available
// Is a named type file
// Remote I/O error
// Quota exceeded
// No medium found
// Wrong medium type
// Operation Canceled
// Required key not available
// Key has expired
// Key has been revoked
// Key was rejected by service
// for robust mutexes
// Owner died
// State not recoverable
// Operation not possible due to RF-kill
// Memory page has hardware error
// restart if no handler..
// No ioctl command
// restart by calling sys_restart_syscall
// Driver requests probe retry
// open found a stale dentry
// Parameter not supported
// Defined for the NFSv3 protocol
// Illegal NFS file handle
// Update synchronization mismatch
// Cookie is stale
// Operation is not supported
// Buffer or request is too small
// An untranslatable error occurred
// Type not supported by server
// Request initiated, but will not complete before timeout
// iocb queued, will get completion event
// conflict with recalled state
// check_sub_overflow() - Calculate subtraction with overflow checking
// @a: minuend; value to subtract from
// @b: subtrahend; value to subtract from @a
// @d: pointer to store difference
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted subtraction, regardless of whether
// wrap-around occurred.
// check_mul_overflow() - Calculate multiplication with overflow checking
// @a: first factor
// @b: second factor
// @d: pointer to store product
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted multiplication, regardless of whether
// wrap-around occurred.
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Because instrumentation_{begin,end}() can nest, objtool validation considers
// _begin() a +1 and _end() a -1 and computes a sum over the instructions.
// When the value is greater than 0, we consider instrumentation allowed.
//
// There is a problem with code like:
//
// noinstr void foo()
// {
// 	instrumentation_begin();
// 	...
// 	if (cond) {
// 		instrumentation_begin();
// 		...
// 		instrumentation_end();
// 	}
// 	bar();
// 	instrumentation_end();
// }
//
// If instrumentation_end() would be an empty label, like all the other
// annotations, the inner _end(), which is at the end of a conditional block,
// would land on the instruction after the block.
//
// If we then consider the sum of the !cond path, we'll see that the call to
// bar() is with a 0-value, even though, we meant it to happen with a positive
// value.
//
// To avoid this, have _end() be a NOP instruction, this ensures it will be
// part of the condition block and does not escape.
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// new instruction
// feature + flags
// source len
// replacement len
// replacement
// alternative assembly primitive:
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Concatenate two parameters, but allow them to be expanded beforehand.
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// Check that the bitops prototypes are sane
// order_base_2 - calculate the (rounded up) base 2 order of the argument
// @n: parameter
//
// The first few values calculated by this routine:
//  ob2(0) = 0
//  ob2(1) = 0
//  ob2(2) = 1
//  ob2(3) = 2
//  ob2(4) = 2
//  ob2(5) = 3
//  ... and so on.
// Skipped when running bindgen due to a libclang issue;
// see https://github.com/rust-lang/rust-bindgen/issues/2244.
// Annotating a function/variable with __retain tells the compiler to place
// the object in its own section and set the flag SHF_GNU_RETAIN. This flag
// instructs the linker to retain the object during garbage-cleanup or LTO
// phases.
//
// Note that the __used macro is also used to prevent functions or data
// being optimized out, but operates at the compiler/IR-level and may still
// allow unintended removal of objects during linking.
//
// Optional: only supported since gcc >= 11, clang >= 13
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-retain-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#retain
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__2 {
() => {
((core::mem::size_of::<[*const u8; 134]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__15!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__6 {
() => {
((core::mem::size_of::<[*const u8; 19]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__20!() as u64)
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED {
() => {
__is_defined!()
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

/// Getting something that works in C and CPP for an arg that may or may
/// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
/// we match on the placeholder define, insert the "0," for arg1 and generate
/// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
/// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
/// the last step cherry picks the 2nd arg, we get a zero.
macro_rules! __is_defined {
() => {
____is_defined!()
};
}

macro_rules! __must_be_array__15 {
() => {
core::mem::size_of::<__SlateRecord_errname_c_203>() as i32
};
}

macro_rules! __must_be_array__20 {
() => {
core::mem::size_of::<__SlateRecord_errname_c_205>() as i32
};
}

/// abs - return absolute value of an argument
/// @x: the value.
///
/// If it is unsigned type, @x is converted to signed type first.
/// char is treated as if it was signed (regardless of whether it really is)
/// but the macro's return type is preserved as char.
///
/// NOTE, for signed type if @x is the minimum, the returned result is undefined
/// as there is not enough bits to represent it as a positive number.
///
/// Return: an absolute value of @x.
macro_rules! abs__2 {
($x:expr) => {
{
let mut __v: i32;
let __v2: i32 = $x;
__v = if __v2 < 0 { __v2.wrapping_neg() } else { __v2 };
__v
}
};
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_errname_c_203 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_errname_c_205 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

// Ensure these tables do not accidentally become gigantic if some
// huge errno makes it in. On most architectures, the first table will
// only have about 140 entries, but mips and parisc have more sparsely
// allocated errnos (with EHWPOISON = 257 on parisc, and EDQUOT = 1133
// on mips), so this wastes a bit of space on those - though we
// special case the EDQUOT case.
/// EWOULDBLOCK
/// ECANCELLED
/// EREFUSED
/// EDEADLOCK
static mut names_0: __SlateAlign16<[*const u8; 134]> = __SlateAlign16({
let mut __t0: [*const u8; 134] = unsafe { core::mem::zeroed() };
__t0[1] = (b"-EPERM\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"-ENOENT\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"-ESRCH\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"-EINTR\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"-EIO\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"-ENXIO\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"-E2BIG\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"-ENOEXEC\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"-EBADF\0".as_ptr() as *mut u8) as *const u8;
__t0[10] = (b"-ECHILD\0".as_ptr() as *mut u8) as *const u8;
__t0[11] = (b"-EAGAIN\0".as_ptr() as *mut u8) as *const u8;
__t0[12] = (b"-ENOMEM\0".as_ptr() as *mut u8) as *const u8;
__t0[13] = (b"-EACCES\0".as_ptr() as *mut u8) as *const u8;
__t0[14] = (b"-EFAULT\0".as_ptr() as *mut u8) as *const u8;
__t0[15] = (b"-ENOTBLK\0".as_ptr() as *mut u8) as *const u8;
__t0[16] = (b"-EBUSY\0".as_ptr() as *mut u8) as *const u8;
__t0[17] = (b"-EEXIST\0".as_ptr() as *mut u8) as *const u8;
__t0[18] = (b"-EXDEV\0".as_ptr() as *mut u8) as *const u8;
__t0[19] = (b"-ENODEV\0".as_ptr() as *mut u8) as *const u8;
__t0[20] = (b"-ENOTDIR\0".as_ptr() as *mut u8) as *const u8;
__t0[21] = (b"-EISDIR\0".as_ptr() as *mut u8) as *const u8;
__t0[22] = (b"-EINVAL\0".as_ptr() as *mut u8) as *const u8;
__t0[23] = (b"-ENFILE\0".as_ptr() as *mut u8) as *const u8;
__t0[24] = (b"-EMFILE\0".as_ptr() as *mut u8) as *const u8;
__t0[25] = (b"-ENOTTY\0".as_ptr() as *mut u8) as *const u8;
__t0[26] = (b"-ETXTBSY\0".as_ptr() as *mut u8) as *const u8;
__t0[27] = (b"-EFBIG\0".as_ptr() as *mut u8) as *const u8;
__t0[28] = (b"-ENOSPC\0".as_ptr() as *mut u8) as *const u8;
__t0[29] = (b"-ESPIPE\0".as_ptr() as *mut u8) as *const u8;
__t0[30] = (b"-EROFS\0".as_ptr() as *mut u8) as *const u8;
__t0[31] = (b"-EMLINK\0".as_ptr() as *mut u8) as *const u8;
__t0[32] = (b"-EPIPE\0".as_ptr() as *mut u8) as *const u8;
__t0[33] = (b"-EDOM\0".as_ptr() as *mut u8) as *const u8;
__t0[34] = (b"-ERANGE\0".as_ptr() as *mut u8) as *const u8;
__t0[35] = (b"-EDEADLK\0".as_ptr() as *mut u8) as *const u8;
__t0[36] = (b"-ENAMETOOLONG\0".as_ptr() as *mut u8) as *const u8;
__t0[37] = (b"-ENOLCK\0".as_ptr() as *mut u8) as *const u8;
__t0[38] = (b"-ENOSYS\0".as_ptr() as *mut u8) as *const u8;
__t0[39] = (b"-ENOTEMPTY\0".as_ptr() as *mut u8) as *const u8;
__t0[40] = (b"-ELOOP\0".as_ptr() as *mut u8) as *const u8;
__t0[42] = (b"-ENOMSG\0".as_ptr() as *mut u8) as *const u8;
__t0[43] = (b"-EIDRM\0".as_ptr() as *mut u8) as *const u8;
__t0[44] = (b"-ECHRNG\0".as_ptr() as *mut u8) as *const u8;
__t0[45] = (b"-EL2NSYNC\0".as_ptr() as *mut u8) as *const u8;
__t0[46] = (b"-EL3HLT\0".as_ptr() as *mut u8) as *const u8;
__t0[47] = (b"-EL3RST\0".as_ptr() as *mut u8) as *const u8;
__t0[48] = (b"-ELNRNG\0".as_ptr() as *mut u8) as *const u8;
__t0[49] = (b"-EUNATCH\0".as_ptr() as *mut u8) as *const u8;
__t0[50] = (b"-ENOCSI\0".as_ptr() as *mut u8) as *const u8;
__t0[51] = (b"-EL2HLT\0".as_ptr() as *mut u8) as *const u8;
__t0[52] = (b"-EBADE\0".as_ptr() as *mut u8) as *const u8;
__t0[53] = (b"-EBADR\0".as_ptr() as *mut u8) as *const u8;
__t0[54] = (b"-EXFULL\0".as_ptr() as *mut u8) as *const u8;
__t0[55] = (b"-ENOANO\0".as_ptr() as *mut u8) as *const u8;
__t0[56] = (b"-EBADRQC\0".as_ptr() as *mut u8) as *const u8;
__t0[57] = (b"-EBADSLT\0".as_ptr() as *mut u8) as *const u8;
__t0[59] = (b"-EBFONT\0".as_ptr() as *mut u8) as *const u8;
__t0[60] = (b"-ENOSTR\0".as_ptr() as *mut u8) as *const u8;
__t0[61] = (b"-ENODATA\0".as_ptr() as *mut u8) as *const u8;
__t0[62] = (b"-ETIME\0".as_ptr() as *mut u8) as *const u8;
__t0[63] = (b"-ENOSR\0".as_ptr() as *mut u8) as *const u8;
__t0[64] = (b"-ENONET\0".as_ptr() as *mut u8) as *const u8;
__t0[65] = (b"-ENOPKG\0".as_ptr() as *mut u8) as *const u8;
__t0[66] = (b"-EREMOTE\0".as_ptr() as *mut u8) as *const u8;
__t0[67] = (b"-ENOLINK\0".as_ptr() as *mut u8) as *const u8;
__t0[68] = (b"-EADV\0".as_ptr() as *mut u8) as *const u8;
__t0[69] = (b"-ESRMNT\0".as_ptr() as *mut u8) as *const u8;
__t0[70] = (b"-ECOMM\0".as_ptr() as *mut u8) as *const u8;
__t0[71] = (b"-EPROTO\0".as_ptr() as *mut u8) as *const u8;
__t0[72] = (b"-EMULTIHOP\0".as_ptr() as *mut u8) as *const u8;
__t0[73] = (b"-EDOTDOT\0".as_ptr() as *mut u8) as *const u8;
__t0[74] = (b"-EBADMSG\0".as_ptr() as *mut u8) as *const u8;
__t0[75] = (b"-EOVERFLOW\0".as_ptr() as *mut u8) as *const u8;
__t0[76] = (b"-ENOTUNIQ\0".as_ptr() as *mut u8) as *const u8;
__t0[77] = (b"-EBADFD\0".as_ptr() as *mut u8) as *const u8;
__t0[78] = (b"-EREMCHG\0".as_ptr() as *mut u8) as *const u8;
__t0[79] = (b"-ELIBACC\0".as_ptr() as *mut u8) as *const u8;
__t0[80] = (b"-ELIBBAD\0".as_ptr() as *mut u8) as *const u8;
__t0[81] = (b"-ELIBSCN\0".as_ptr() as *mut u8) as *const u8;
__t0[82] = (b"-ELIBMAX\0".as_ptr() as *mut u8) as *const u8;
__t0[83] = (b"-ELIBEXEC\0".as_ptr() as *mut u8) as *const u8;
__t0[84] = (b"-EILSEQ\0".as_ptr() as *mut u8) as *const u8;
__t0[85] = (b"-ERESTART\0".as_ptr() as *mut u8) as *const u8;
__t0[86] = (b"-ESTRPIPE\0".as_ptr() as *mut u8) as *const u8;
__t0[87] = (b"-EUSERS\0".as_ptr() as *mut u8) as *const u8;
__t0[88] = (b"-ENOTSOCK\0".as_ptr() as *mut u8) as *const u8;
__t0[89] = (b"-EDESTADDRREQ\0".as_ptr() as *mut u8) as *const u8;
__t0[90] = (b"-EMSGSIZE\0".as_ptr() as *mut u8) as *const u8;
__t0[91] = (b"-EPROTOTYPE\0".as_ptr() as *mut u8) as *const u8;
__t0[92] = (b"-ENOPROTOOPT\0".as_ptr() as *mut u8) as *const u8;
__t0[93] = (b"-EPROTONOSUPPORT\0".as_ptr() as *mut u8) as *const u8;
__t0[94] = (b"-ESOCKTNOSUPPORT\0".as_ptr() as *mut u8) as *const u8;
__t0[95] = (b"-EOPNOTSUPP\0".as_ptr() as *mut u8) as *const u8;
__t0[96] = (b"-EPFNOSUPPORT\0".as_ptr() as *mut u8) as *const u8;
__t0[97] = (b"-EAFNOSUPPORT\0".as_ptr() as *mut u8) as *const u8;
__t0[98] = (b"-EADDRINUSE\0".as_ptr() as *mut u8) as *const u8;
__t0[99] = (b"-EADDRNOTAVAIL\0".as_ptr() as *mut u8) as *const u8;
__t0[100] = (b"-ENETDOWN\0".as_ptr() as *mut u8) as *const u8;
__t0[101] = (b"-ENETUNREACH\0".as_ptr() as *mut u8) as *const u8;
__t0[102] = (b"-ENETRESET\0".as_ptr() as *mut u8) as *const u8;
__t0[103] = (b"-ECONNABORTED\0".as_ptr() as *mut u8) as *const u8;
__t0[104] = (b"-ECONNRESET\0".as_ptr() as *mut u8) as *const u8;
__t0[105] = (b"-ENOBUFS\0".as_ptr() as *mut u8) as *const u8;
__t0[106] = (b"-EISCONN\0".as_ptr() as *mut u8) as *const u8;
__t0[107] = (b"-ENOTCONN\0".as_ptr() as *mut u8) as *const u8;
__t0[108] = (b"-ESHUTDOWN\0".as_ptr() as *mut u8) as *const u8;
__t0[109] = (b"-ETOOMANYREFS\0".as_ptr() as *mut u8) as *const u8;
__t0[110] = (b"-ETIMEDOUT\0".as_ptr() as *mut u8) as *const u8;
__t0[111] = (b"-ECONNREFUSED\0".as_ptr() as *mut u8) as *const u8;
__t0[112] = (b"-EHOSTDOWN\0".as_ptr() as *mut u8) as *const u8;
__t0[113] = (b"-EHOSTUNREACH\0".as_ptr() as *mut u8) as *const u8;
__t0[114] = (b"-EALREADY\0".as_ptr() as *mut u8) as *const u8;
__t0[115] = (b"-EINPROGRESS\0".as_ptr() as *mut u8) as *const u8;
__t0[116] = (b"-ESTALE\0".as_ptr() as *mut u8) as *const u8;
__t0[117] = (b"-EUCLEAN\0".as_ptr() as *mut u8) as *const u8;
__t0[118] = (b"-ENOTNAM\0".as_ptr() as *mut u8) as *const u8;
__t0[119] = (b"-ENAVAIL\0".as_ptr() as *mut u8) as *const u8;
__t0[120] = (b"-EISNAM\0".as_ptr() as *mut u8) as *const u8;
__t0[121] = (b"-EREMOTEIO\0".as_ptr() as *mut u8) as *const u8;
__t0[122] = (b"-EDQUOT\0".as_ptr() as *mut u8) as *const u8;
__t0[123] = (b"-ENOMEDIUM\0".as_ptr() as *mut u8) as *const u8;
__t0[124] = (b"-EMEDIUMTYPE\0".as_ptr() as *mut u8) as *const u8;
__t0[125] = (b"-ECANCELED\0".as_ptr() as *mut u8) as *const u8;
__t0[126] = (b"-ENOKEY\0".as_ptr() as *mut u8) as *const u8;
__t0[127] = (b"-EKEYEXPIRED\0".as_ptr() as *mut u8) as *const u8;
__t0[128] = (b"-EKEYREVOKED\0".as_ptr() as *mut u8) as *const u8;
__t0[129] = (b"-EKEYREJECTED\0".as_ptr() as *mut u8) as *const u8;
__t0[130] = (b"-EOWNERDEAD\0".as_ptr() as *mut u8) as *const u8;
__t0[131] = (b"-ENOTRECOVERABLE\0".as_ptr() as *mut u8) as *const u8;
__t0[132] = (b"-ERFKILL\0".as_ptr() as *mut u8) as *const u8;
__t0[133] = (b"-EHWPOISON\0".as_ptr() as *mut u8) as *const u8;
__t0
});

// everywhere
static mut names_512: __SlateAlign16<[*const u8; 19]> = __SlateAlign16({
let mut __t0: [*const u8; 19] = unsafe { core::mem::zeroed() };
__t0[0] = (b"-ERESTARTSYS\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"-ERESTARTNOINTR\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"-ERESTARTNOHAND\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"-ENOIOCTLCMD\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"-ERESTART_RESTARTBLOCK\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"-EPROBE_DEFER\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"-EOPENSTALE\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"-ENOPARAM\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"-EBADHANDLE\0".as_ptr() as *mut u8) as *const u8;
__t0[10] = (b"-ENOTSYNC\0".as_ptr() as *mut u8) as *const u8;
__t0[11] = (b"-EBADCOOKIE\0".as_ptr() as *mut u8) as *const u8;
__t0[12] = (b"-ENOTSUPP\0".as_ptr() as *mut u8) as *const u8;
__t0[13] = (b"-ETOOSMALL\0".as_ptr() as *mut u8) as *const u8;
__t0[14] = (b"-ESERVERFAULT\0".as_ptr() as *mut u8) as *const u8;
__t0[15] = (b"-EBADTYPE\0".as_ptr() as *mut u8) as *const u8;
__t0[16] = (b"-EJUKEBOX\0".as_ptr() as *mut u8) as *const u8;
__t0[17] = (b"-EIOCBQUEUED\0".as_ptr() as *mut u8) as *const u8;
__t0[18] = (b"-ERECALLCONFLICT\0".as_ptr() as *mut u8) as *const u8;
__t0
});

fn __errname(err: u32) -> *const u8 {
if (err as u64) < ARRAY_SIZE__2!() {
return unsafe { names_0.0[err as usize] };
}
if err >= 512 && (err.wrapping_sub(512) as u64) < ARRAY_SIZE__6!() {
return unsafe { names_512.0[err.wrapping_sub(512) as usize] };
}
// But why?
if IS_ENABLED!() != 0 && err == 122 {
return c"-EDQUOT".as_ptr() as *const u8;
}
// 1133
core::ptr::null::<u8>()
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_errname: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous", sym errname, options(att_syntax));

/// errname(EIO) -> "EIO"
/// errname(-EIO) -> "-EIO"
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.errname.errname")]
extern "C" fn errname(err: i32) -> *const u8 {
let name: *const u8 = __errname(abs__2!(err) as u32);
if name.is_null() {
return core::ptr::null::<u8>();
}
if err > 0 { name.wrapping_offset(1 as isize) } else { name }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_errname_5: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32) -> *const u8>, *mut ()>(Some(errname)) };
