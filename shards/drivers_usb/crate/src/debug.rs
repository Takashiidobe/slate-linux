
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//!
//! Common USB debugging functions
//!
//! Copyright (C) 2010-2011 Texas Instruments Incorporated - https://www.ti.com
//!
//! Authors: Felipe Balbi <balbi@ti.com>,
//! 	    Sebastian Andrzej Siewior <bigeasy@linutronix.de>
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//!
//! @TYPE: The type of each flexible array element
//! @NAME: The name of the flexible array member
//!
//! In order to have a flexible array member in a union or alone in a
//! struct, it needs to be wrapped in an anonymous struct with at least 1
//! named member, but that member can be empty.
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! static_assert - check integer constant expression at build time
//! @expr: expression to be checked
//!
//! static_assert() is a wrapper for the C11 _Static_assert, with a
//! little macro magic to make the message optional (defaulting to the
//! stringification of the tested expression).
//!
//! Contrary to BUILD_BUG_ON(), static_assert() can be used at global
//! scope, but requires the expression to be an integer constant
//! expression (i.e., it is not enough that __builtin_constant_p() is
//! true for expr).
//!
//! Also note that BUILD_BUG_ON() fails the build if the condition is
//! true, while static_assert() fails the build if the expression is
//! false.
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
//! USB directions
//!
//! This bit flag is used in endpoint descriptors' bEndpointAddress field.
//! It's also one of three fields in control requests bRequestType.
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
// Getting something that works in C and CPP for an arg that may or may
// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
// we match on the placeholder define, insert the "0," for arg1 and generate
// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
// the last step cherry picks the 2nd arg, we get a zero.
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
// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
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
// to device
// to host
// USB types, the second of three bRequestType fields
// USB recipients, the third of three bRequestType fields
// Standard requests, for the bRequest field of a SETUP packet.
//
// These are qualified by the bRequestType field, so that for example
// TYPE_CLASS or TYPE_VENDOR specific feature flags could be retrieved
// by a GET_STATUS request.
// USB feature flags are written using USB_REQ_{CLEAR,SET}_FEATURE, and
// are read as a bit array returned by USB_REQ_GET_STATUS.  (So there
// are at most sixteen features of each type.)  Hubs may also support a
// new USB_REQ_TEST_AND_SET_FEATURE to put ports into L1 suspend.
// (read only)
// dev may initiate wakeup
// (wired high speed only)
// Test Mode Selectors
// See USB 2.0 spec Table 9-7
// New Feature Selectors as added by USB 3.0
// See USB 3.0 spec Table 9-7
// dev may initiate U1 transition
// dev may initiate U2 transition
// dev may send LTM
// function suspend
// IN/OUT will STALL
// Descriptor types ... USB 2.0 spec table 9.5
// these are from a minor usb 2.0 revision (ECN)
// From the T10 UAS specification
// From the USB 3.0 spec
// From the USB 3.1 spec
// Endpoints
// in bEndpointAddress
// in bmAttributes
// The USB 3.0 spec redefines bits 5:4 of bmAttributes as interrupt ep type.
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
// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
unsafe extern "C" {
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
}

fn usb_decode_get_status(bRequestType: u8, wIndex: u16, wLength: u16, str: *mut u8, size: u64) {
match (bRequestType as i32) & 31 {
0 => {
unsafe { snprintf(str, size, c"Get Device Status(Length = %d)".as_ptr() as *const u8, wLength as i32) };
}
1 => {
unsafe { snprintf(str, size, c"Get Interface Status(Intf = %d, Length = %d)".as_ptr() as *const u8, wIndex as i32, wLength as i32) };
}
2 => {
unsafe { snprintf(str, size, c"Get Endpoint Status(ep%d%s)".as_ptr() as *const u8, (wIndex as i32) & !128, if (wIndex as i32) & 128 != 0 { c"in".as_ptr() as *mut u8 } else { c"out".as_ptr() as *mut u8 }) };
}
_ => {
}
}
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_usb_decode_ctrl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous", sym usb_decode_ctrl, options(att_syntax));

fn usb_decode_device_feature(wValue: u16) -> *const u8 {
match wValue as i32 {
0 => {
return c"Self Powered".as_ptr() as *const u8;
}
1 => {
return c"Remote Wakeup".as_ptr() as *const u8;
}
2 => {
return c"Test Mode".as_ptr() as *const u8;
}
48 => {
return c"U1 Enable".as_ptr() as *const u8;
}
49 => {
return c"U2 Enable".as_ptr() as *const u8;
}
50 => {
return c"LTM Enable".as_ptr() as *const u8;
}
_ => {
return c"UNKNOWN".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

fn usb_decode_test_mode(wIndex: u16) -> *const u8 {
match wIndex as i32 {
1 => {
return c": TEST_J".as_ptr() as *const u8;
}
2 => {
return c": TEST_K".as_ptr() as *const u8;
}
3 => {
return c": TEST_SE0_NAK".as_ptr() as *const u8;
}
4 => {
return c": TEST_PACKET".as_ptr() as *const u8;
}
5 => {
return c": TEST_FORCE_EN".as_ptr() as *const u8;
}
_ => {
return c": UNKNOWN".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

fn usb_decode_set_clear_feature(bRequestType: u8, bRequest: u8, wValue: u16, wIndex: u16, str: *mut u8, size: u64) {
match (bRequestType as i32) & 31 {
0 => {
let __v0: *const u8 = if (wValue as i32) == 2 {
usb_decode_test_mode(wIndex)
} else {
c"".as_ptr() as *const u8
};
unsafe { snprintf(str, size, c"%s Device Feature(%s%s)".as_ptr() as *const u8, if (bRequest as i32) == 1 { c"Clear".as_ptr() as *mut u8 } else { c"Set".as_ptr() as *mut u8 }, usb_decode_device_feature(wValue), __v0) };
}
1 => {
unsafe { snprintf(str, size, c"%s Interface Feature(%s)".as_ptr() as *const u8, if (bRequest as i32) == 1 { c"Clear".as_ptr() as *mut u8 } else { c"Set".as_ptr() as *mut u8 }, if (wValue as i32) == 0 { c"Function Suspend".as_ptr() as *mut u8 } else { c"UNKNOWN".as_ptr() as *mut u8 }) };
}
2 => {
unsafe { snprintf(str, size, c"%s Endpoint Feature(%s ep%d%s)".as_ptr() as *const u8, if (bRequest as i32) == 1 { c"Clear".as_ptr() as *mut u8 } else { c"Set".as_ptr() as *mut u8 }, if (wValue as i32) == 0 { c"Halt".as_ptr() as *mut u8 } else { c"UNKNOWN".as_ptr() as *mut u8 }, (wIndex as i32) & !128, if (wIndex as i32) & 128 != 0 { c"in".as_ptr() as *mut u8 } else { c"out".as_ptr() as *mut u8 }) };
}
_ => {
}
}
}

fn usb_decode_set_address(wValue: u16, str: *mut u8, size: u64) {
unsafe { snprintf(str, size, c"Set Address(Addr = %02x)".as_ptr() as *const u8, wValue as i32) };
}

fn usb_decode_get_set_descriptor(bRequestType: u8, bRequest: u8, wValue: u16, wIndex: u16, wLength: u16, str: *mut u8, size: u64) {
let mut s: *mut u8 = unsafe { core::mem::zeroed() };
match (wValue as i32) >> 8i32 {
1 => {
s = c"Device".as_ptr() as *mut u8;
}
2 => {
s = c"Configuration".as_ptr() as *mut u8;
}
3 => {
s = c"String".as_ptr() as *mut u8;
}
4 => {
s = c"Interface".as_ptr() as *mut u8;
}
5 => {
s = c"Endpoint".as_ptr() as *mut u8;
}
6 => {
s = c"Device Qualifier".as_ptr() as *mut u8;
}
7 => {
s = c"Other Speed Config".as_ptr() as *mut u8;
}
8 => {
s = c"Interface Power".as_ptr() as *mut u8;
}
9 => {
s = c"OTG".as_ptr() as *mut u8;
}
10 => {
s = c"Debug".as_ptr() as *mut u8;
}
11 => {
s = c"Interface Association".as_ptr() as *mut u8;
}
15 => {
s = c"BOS".as_ptr() as *mut u8;
}
16 => {
s = c"Device Capability".as_ptr() as *mut u8;
}
36 => {
s = c"Pipe Usage".as_ptr() as *mut u8;
}
48 => {
s = c"SS Endpoint Companion".as_ptr() as *mut u8;
}
49 => {
s = c"SSP Isochronous Endpoint Companion".as_ptr() as *mut u8;
}
_ => {
s = c"UNKNOWN".as_ptr() as *mut u8;
}
}
unsafe { snprintf(str, size, c"%s %s Descriptor(Index = %d, Length = %d)".as_ptr() as *const u8, if (bRequest as i32) == 6 { c"Get".as_ptr() as *mut u8 } else { c"Set".as_ptr() as *mut u8 }, s, (wValue as i32) & 255, wLength as i32) };
}

fn usb_decode_get_configuration(wLength: u16, str: *mut u8, size: u64) {
unsafe { snprintf(str, size, c"Get Configuration(Length = %d)".as_ptr() as *const u8, wLength as i32) };
}

fn usb_decode_set_configuration(wValue: u8, str: *mut u8, size: u64) {
unsafe { snprintf(str, size, c"Set Configuration(Config = %d)".as_ptr() as *const u8, wValue as i32) };
}

fn usb_decode_get_intf(wIndex: u16, wLength: u16, str: *mut u8, size: u64) {
unsafe { snprintf(str, size, c"Get Interface(Intf = %d, Length = %d)".as_ptr() as *const u8, wIndex as i32, wLength as i32) };
}

fn usb_decode_set_intf(wValue: u8, wIndex: u16, str: *mut u8, size: u64) {
unsafe { snprintf(str, size, c"Set Interface(Intf = %d, Alt.Setting = %d)".as_ptr() as *const u8, wIndex as i32, wValue as i32) };
}

fn usb_decode_synch_frame(wIndex: u16, wLength: u16, str: *mut u8, size: u64) {
unsafe { snprintf(str, size, c"Synch Frame(Endpoint = %d, Length = %d)".as_ptr() as *const u8, wIndex as i32, wLength as i32) };
}

fn usb_decode_set_sel(wLength: u16, str: *mut u8, size: u64) {
unsafe { snprintf(str, size, c"Set SEL(Length = %d)".as_ptr() as *const u8, wLength as i32) };
}

fn usb_decode_set_isoch_delay(wValue: u8, str: *mut u8, size: u64) {
unsafe { snprintf(str, size, c"Set Isochronous Delay(Delay = %d ns)".as_ptr() as *const u8, wValue as i32) };
}

fn usb_decode_ctrl_generic(str: *mut u8, size: u64, bRequestType: u8, bRequest: u8, wValue: u16, wIndex: u16, wLength: u16) {
let recip: u8 = ((bRequestType as i32) & 31) as u8;
let r#type: u8 = ((bRequestType as i32) & 96i32) as u8;
unsafe { snprintf(str, size, c"Type=%s Recipient=%s Dir=%s bRequest=%u wValue=%u wIndex=%u wLength=%u".as_ptr() as *const u8, if (r#type as i32) == 0i32 { c"Standard".as_ptr() as *mut u8 } else { if (r#type as i32) == 64i32 { c"Vendor".as_ptr() as *mut u8 } else { if (r#type as i32) == 32i32 { c"Class".as_ptr() as *mut u8 } else { c"Unknown".as_ptr() as *mut u8 } } }, if (recip as i32) == 0 { c"Device".as_ptr() as *mut u8 } else { if (recip as i32) == 1 { c"Interface".as_ptr() as *mut u8 } else { if (recip as i32) == 2 { c"Endpoint".as_ptr() as *mut u8 } else { c"Unknown".as_ptr() as *mut u8 } } }, if (bRequestType as i32) & 128 != 0 { c"IN".as_ptr() as *mut u8 } else { c"OUT".as_ptr() as *mut u8 }, bRequest as i32, wValue as i32, wIndex as i32, wLength as i32) };
}

fn usb_decode_ctrl_standard(str: *mut u8, size: u64, bRequestType: u8, bRequest: u8, wValue: u16, wIndex: u16, wLength: u16) {
match bRequest as i32 {
0 => {
usb_decode_get_status(bRequestType, wIndex, wLength, str, size);
}
1 | 3 => {
usb_decode_set_clear_feature(bRequestType, bRequest, wValue, wIndex, str, size);
}
5 => {
usb_decode_set_address(wValue, str, size);
}
6 | 7 => {
usb_decode_get_set_descriptor(bRequestType, bRequest, wValue, wIndex, wLength, str, size);
}
8 => {
usb_decode_get_configuration(wLength, str, size);
}
9 => {
usb_decode_set_configuration(wValue as u8, str, size);
}
10 => {
usb_decode_get_intf(wIndex, wLength, str, size);
}
11 => {
usb_decode_set_intf(wValue as u8, wIndex, str, size);
}
12 => {
usb_decode_synch_frame(wIndex, wLength, str, size);
}
48 => {
usb_decode_set_sel(wLength, str, size);
}
49 => {
usb_decode_set_isoch_delay(wValue as u8, str, size);
}
_ => {
usb_decode_ctrl_generic(str, size, bRequestType, bRequest, wValue, wIndex, wLength);
}
}
}

/// usb_decode_ctrl - Returns human readable representation of control request.
/// @str: buffer to return a human-readable representation of control request.
///       This buffer should have about 200 bytes.
/// @size: size of str buffer.
/// @bRequestType: matches the USB bmRequestType field
/// @bRequest: matches the USB bRequest field
/// @wValue: matches the USB wValue field (CPU byte order)
/// @wIndex: matches the USB wIndex field (CPU byte order)
/// @wLength: matches the USB wLength field (CPU byte order)
///
/// Function returns decoded, formatted and human-readable description of
/// control request packet.
///
/// The usage scenario for this is for tracepoints, so function as a return
/// use the same value as in parameters. This approach allows to use this
/// function in TP_printk
///
/// Important: wValue, wIndex, wLength parameters before invoking this function
/// should be processed by le16_to_cpu macro.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.debug.usb_decode_ctrl")]
extern "C" fn usb_decode_ctrl(str: *mut u8, size: u64, bRequestType: u8, bRequest: u8, wValue: u16, wIndex: u16, wLength: u16) -> *const u8 {
match (bRequestType as i32) & 96i32 {
0 => {
usb_decode_ctrl_standard(str, size, bRequestType, bRequest, wValue, wIndex, wLength);
}
_ => {
usb_decode_ctrl_generic(str, size, bRequestType, bRequest, wValue, wIndex, wLength);
}
}
str as *const u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_decode_ctrl_5: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, u64, u8, u8, u16, u16, u16) -> *const u8>, *mut ()>(Some(usb_decode_ctrl)) };
