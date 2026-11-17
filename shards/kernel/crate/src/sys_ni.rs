
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! unused, was: #define EX_TYPE_COPY	 4
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! check_add_overflow() - Calculate addition with overflow checking
//! @a: first addend
//! @b: second addend
//! @d: pointer to store sum
//!
//! Returns true on wrap-around, false otherwise.
//!
//! *@d holds the results of the attempted addition, regardless of whether
//! wrap-around occurred.
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
//! label
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! As some compat syscalls may not be implemented, we need to expand
//! COND_SYSCALL_COMPAT in kernel/sys_ni.c to cover this case as well.
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
// Invalid system call number
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
// Like alternative_input, but with a single output argument
// early_idt_handler_array is an array of entry points referenced in the
// early IDT.  For simplicity, it's a real array with one entry point
// every nine bytes.  That leaves room for an optional 'push $0' if the
// vector has no error code (two bytes), a 'push $vector_number' (two
// bytes), and a jump to the common entry code (up to five bytes).
// Load a segment. Fall back on loading the zero segment if something goes
// wrong.  This variant assumes that loading zero fully clears the segment.
// This is always the case on Intel CPUs and, even on 64-bit AMD CPUs, any
// failure to fully clear the cached descriptor is only observable for
// FS and GS.
// Save a segment register away:
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
unsafe extern "C" {
    fn __x64_sys_acct(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_acct(arg0: *const pt_regs) -> i64;
    fn __x64_sys_capget(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_capget(arg0: *const pt_regs) -> i64;
    fn __x64_sys_capset(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_capset(arg0: *const pt_regs) -> i64;
    fn __x64_sys_kexec_load(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_kexec_load(arg0: *const pt_regs) -> i64;
    fn __ia32_compat_sys_kexec_load(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setregid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setregid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setgid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setgid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setreuid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setreuid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setuid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setuid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setresuid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setresuid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_getresuid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_getresuid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setresgid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setresgid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_getresgid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_getresgid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setfsuid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setfsuid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setfsgid(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setfsgid(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setgroups(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setgroups(arg0: *const pt_regs) -> i64;
    fn __x64_sys_getgroups(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_getgroups(arg0: *const pt_regs) -> i64;
    fn __x64_sys_kcmp(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_kcmp(arg0: *const pt_regs) -> i64;
    fn __x64_sys_seccomp(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_seccomp(arg0: *const pt_regs) -> i64;
    fn __x64_sys_rt_sigtimedwait_time32(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_rt_sigtimedwait_time32(arg0: *const pt_regs) -> i64;
    fn __ia32_compat_sys_rt_sigtimedwait_time32(arg0: *const pt_regs) -> i64;
    fn __x64_sys_ssetmask(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_ssetmask(arg0: *const pt_regs) -> i64;
    fn __x64_sys_chown16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_chown16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_fchown16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_fchown16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_getgroups16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_getgroups16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_getresgid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_getresgid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_getresuid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_getresuid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_lchown16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_lchown16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setfsgid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setfsgid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setfsuid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setfsuid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setgid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setgid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setgroups16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setgroups16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setregid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setregid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setresgid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setresgid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setresuid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setresuid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setreuid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setreuid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_setuid16(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_setuid16(arg0: *const pt_regs) -> i64;
    fn __x64_sys_rseq(arg0: *const pt_regs) -> i64;
    fn __ia32_sys_rseq(arg0: *const pt_regs) -> i64;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fred_cs {
    __slate_bits_0: __slate_bits::__SlateBits10U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_ss {
    __slate_bits_0: __slate_bits::__SlateBits11U0,
}


/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
#[repr(C)]
#[derive(Clone, Copy)]
struct pt_regs {
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    bp: u64,
    bx: u64,
    r11: u64,
    r10: u64,
    r9: u64,
    r8: u64,
    ax: u64,
    cx: u64,
    dx: u64,
    si: u64,
    di: u64,
    orig_ax: u64,
    ip: u64,
    __slate_anon_17: __SlateRecord_ptrace_h_144,
    flags: u64,
    sp: u64,
    __slate_anon_20: __SlateRecord_ptrace_h_156,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ptrace_h_144 {
    cs: u16,
    csx: u64,
    fred_cs: fred_cs,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ptrace_h_156 {
    ss: u16,
    ssx: u64,
    fred_ss: fred_ss,
}


mod __slate_bits {
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits10U0 {
        #[bits(16)]
        pub cs: u64,
        #[bits(2)]
        pub sl: u64,
        #[bits(1)]
        pub wfe: u64,
        #[bits(45, access = na)]
        pub __slate_pad_3: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits11U0 {
        #[bits(16)]
        pub ss: u64,
        #[bits(1)]
        pub sti: u64,
        #[bits(1)]
        pub swevent: u64,
        #[bits(1)]
        pub nmi: u64,
        #[bits(13, access = na)]
        pub __slate_pad_4: u16,
        #[bits(8)]
        pub vector: u64,
        #[bits(8, access = na)]
        pub __slate_pad_6: u8,
        #[bits(4)]
        pub r#type: u64,
        #[bits(4, access = na)]
        pub __slate_pad_8: u8,
        #[bits(1)]
        pub enclave: u64,
        #[bits(1)]
        pub l: u64,
        #[bits(1)]
        pub nested: u64,
        #[bits(1, access = na)]
        pub __slate_pad_12: u8,
        #[bits(4)]
        pub insnlen: u64,
    }
}

// Architectures may override COND_SYSCALL and COND_SYSCALL_COMPAT
/// Non-implemented system calls get redirected here.
#[unsafe(no_mangle)]
extern "C" fn sys_ni_syscall() -> i64 {
38i32.wrapping_neg() as i64
}

// This list is kept in the same order as include/uapi/asm-generic/unistd.h.
// Architecture specific entries go below, followed by deprecated or obsolete
// system calls.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_setup(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_setup(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_io_setup(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_destroy(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_destroy(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_submit(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_submit(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_io_submit(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_cancel(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_cancel(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_getevents_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_getevents_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_getevents(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_getevents(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_pgetevents_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_pgetevents_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_pgetevents(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_pgetevents(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_io_pgetevents(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_io_pgetevents_time64(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_uring_setup(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_uring_setup(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_uring_enter(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_uring_enter(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_io_uring_register(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_io_uring_register(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_eventfd2(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_eventfd2(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_epoll_create1(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_epoll_create1(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_epoll_ctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_epoll_ctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_epoll_pwait(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_epoll_pwait(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_epoll_pwait(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_epoll_pwait2(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_epoll_pwait2(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_epoll_pwait2(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_inotify_init1(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_inotify_init1(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_inotify_add_watch(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_inotify_add_watch(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_inotify_rm_watch(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_inotify_rm_watch(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_ioprio_set(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_ioprio_set(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_ioprio_get(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_ioprio_get(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_flock(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_flock(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_quotactl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_quotactl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_quotactl_fd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_quotactl_fd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_signalfd4(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_signalfd4(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_signalfd4(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timerfd_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timerfd_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timerfd_settime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timerfd_settime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timerfd_settime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timerfd_settime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timerfd_gettime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timerfd_gettime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timerfd_gettime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timerfd_gettime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_futex(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_futex(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_futex_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_futex_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_set_robust_list(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_set_robust_list(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_set_robust_list(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_get_robust_list(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_get_robust_list(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_get_robust_list(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_futex_waitv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_futex_waitv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_futex_wake(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_futex_wake(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_futex_wait(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_futex_wait(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_futex_requeue(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_futex_requeue(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_init_module(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_init_module(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_delete_module(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_delete_module(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_syslog(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_syslog(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mq_open(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mq_open(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_mq_open(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mq_unlink(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mq_unlink(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mq_timedsend(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mq_timedsend(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mq_timedsend_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mq_timedsend_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mq_timedreceive(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mq_timedreceive(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mq_timedreceive_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mq_timedreceive_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mq_notify(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mq_notify(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_mq_notify(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mq_getsetattr(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mq_getsetattr(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_mq_getsetattr(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_msgget(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_msgget(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_old_msgctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_old_msgctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_msgctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_msgctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_msgctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_old_msgctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_msgrcv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_msgrcv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_msgrcv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_msgsnd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_msgsnd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_msgsnd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_semget(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_semget(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_old_semctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_old_semctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_semctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_semctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_semctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_old_semctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_semtimedop(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_semtimedop(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_semtimedop_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_semtimedop_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_semop(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_semop(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_shmget(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_shmget(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_old_shmctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_old_shmctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_shmctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_shmctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_shmctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_old_shmctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_shmat(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_shmat(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_shmat(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_shmdt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_shmdt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_socket(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_socket(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_socketpair(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_socketpair(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_bind(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_bind(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_listen(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_listen(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_accept(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_accept(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_connect(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_connect(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_getsockname(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_getsockname(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_getpeername(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_getpeername(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_setsockopt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_setsockopt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_setsockopt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_getsockopt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_getsockopt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_getsockopt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_sendto(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_sendto(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_shutdown(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_shutdown(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_recvfrom(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_recvfrom(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_recvfrom(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_sendmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_sendmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_sendmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_recvmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_recvmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_recvmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mremap(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mremap(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_add_key(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_add_key(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_request_key(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_request_key(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_keyctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_keyctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_keyctl(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_landlock_create_ruleset(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_landlock_create_ruleset(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_landlock_add_rule(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_landlock_add_rule(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_landlock_restrict_self(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_landlock_restrict_self(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_fadvise64_64(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_fadvise64_64(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_fadvise64_64(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_lsm_get_self_attr(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_lsm_get_self_attr(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_lsm_set_self_attr(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_lsm_set_self_attr(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_lsm_list_modules(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_lsm_list_modules(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// CONFIG_MMU only
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_swapon(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_swapon(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_swapoff(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_swapoff(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mprotect(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mprotect(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_msync(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_msync(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mlock(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mlock(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_munlock(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_munlock(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mlockall(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mlockall(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_munlockall(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_munlockall(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mincore(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mincore(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_madvise(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_madvise(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_process_madvise(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_process_madvise(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_process_mrelease(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_process_mrelease(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_remap_file_pages(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_remap_file_pages(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mbind(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mbind(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_get_mempolicy(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_get_mempolicy(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_set_mempolicy(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_set_mempolicy(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_migrate_pages(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_migrate_pages(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_move_pages(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_move_pages(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_set_mempolicy_home_node(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_set_mempolicy_home_node(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_cachestat(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_cachestat(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mseal(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mseal(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_perf_event_open(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_perf_event_open(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_accept4(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_accept4(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_recvmmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_recvmmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_recvmmsg_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_recvmmsg_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_recvmmsg_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_recvmmsg_time64(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// Posix timer syscalls may be configured out
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timer_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timer_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timer_gettime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timer_gettime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timer_getoverrun(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timer_getoverrun(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timer_settime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timer_settime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timer_delete(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timer_delete(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_clock_adjtime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_clock_adjtime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_getitimer(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_getitimer(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_setitimer(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_setitimer(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_alarm(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_alarm(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_timer_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_getitimer(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_setitimer(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

// Architecture specific syscalls: see further below
/// fanotify
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_fanotify_init(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_fanotify_init(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_fanotify_mark(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_fanotify_mark(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// open by handle
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_name_to_handle_at(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_name_to_handle_at(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_open_by_handle_at(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_open_by_handle_at(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_open_by_handle_at(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_sendmmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_sendmmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_sendmmsg(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_process_vm_readv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_process_vm_readv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_process_vm_readv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_process_vm_writev(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_process_vm_writev(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_process_vm_writev(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

// compare kernel pointers
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_finit_module(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_finit_module(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

// operate on Secure Computing state
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_memfd_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_memfd_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// access BPF programs and maps
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_bpf(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_bpf(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// execveat
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_execveat(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_execveat(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_userfaultfd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_userfaultfd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// membarrier
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_membarrier(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_membarrier(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_mlock2(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_mlock2(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_copy_file_range(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_copy_file_range(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// memory protection keys
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_pkey_mprotect(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_pkey_mprotect(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_pkey_alloc(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_pkey_alloc(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_pkey_free(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_pkey_free(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// memfd_secret
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_memfd_secret(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_memfd_secret(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

// Architecture specific weak syscall entries.
/// pciconfig: alpha, arm, arm64, ia64, sparc
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_pciconfig_read(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_pciconfig_read(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_pciconfig_write(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_pciconfig_write(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_pciconfig_iobase(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_pciconfig_iobase(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// sys_socketcall: arm, mips, x86, ...
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_socketcall(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_socketcall(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_socketcall(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// compat syscalls for arm64, x86, ...
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_fanotify_mark(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// x86
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_vm86old(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_vm86old(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_modify_ldt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_modify_ldt(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_vm86(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_vm86(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_kexec_file_load(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_kexec_file_load(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_map_shadow_stack(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_map_shadow_stack(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// s390
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_s390_pci_mmio_read(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_s390_pci_mmio_read(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_s390_pci_mmio_write(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_s390_pci_mmio_write(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_s390_ipc(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_s390_ipc(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_s390_ipc(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// powerpc
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_rtas(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_rtas(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_spu_run(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_spu_run(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_spu_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_spu_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_subpage_prot(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_subpage_prot(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

// Deprecated system calls which are still defined in
// include/uapi/asm-generic/unistd.h and wanted by >= 1 arch
/// __ARCH_WANT_SYSCALL_NO_FLAGS
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_epoll_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_epoll_create(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_inotify_init(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_inotify_init(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_eventfd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_eventfd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_signalfd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_signalfd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_signalfd(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// __ARCH_WANT_SYSCALL_OFF_T
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_fadvise64(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_fadvise64(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// __ARCH_WANT_SYSCALL_DEPRECATED
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_epoll_wait(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_epoll_wait(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_recv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_recv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_recv(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_send(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_send(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_uselib(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_uselib(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// optional: time32
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_stime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_stime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_utime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_utime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_adjtimex_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_adjtimex_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_sched_rr_get_interval_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_sched_rr_get_interval_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_nanosleep_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_nanosleep_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timer_settime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timer_settime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_timer_gettime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_timer_gettime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_clock_settime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_clock_settime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_clock_gettime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_clock_gettime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_clock_getres_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_clock_getres_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_clock_nanosleep_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_clock_nanosleep_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_utimes_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_utimes_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_futimesat_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_futimesat_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_pselect6_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_pselect6_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_pselect6_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_ppoll_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_ppoll_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_ppoll_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_utimensat_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_utimensat_time32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_clock_adjtime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_clock_adjtime32(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_gettimeofday(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_gettimeofday(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_gettimeofday(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_time(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_time(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_stime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_stime(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

// The syscalls below are not found in include/uapi/asm-generic/unistd.h
/// obsolete: SGETMASK_SYSCALL
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_sgetmask(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_sgetmask(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// obsolete: SYSFS_SYSCALL
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_sysfs(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_sysfs(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

/// obsolete: __ARCH_WANT_SYS_IPC
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_ipc(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_ipc(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_compat_sys_ipc(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

// obsolete: UID16
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_getegid16(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_getegid16(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_geteuid16(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_geteuid16(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_getgid16(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_getgid16(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_getuid16(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_getuid16(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

// restartable sequence
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_rseq_slice_yield(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_rseq_slice_yield(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_uretprobe(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_uretprobe(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __x64_sys_uprobe(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn __ia32_sys_uprobe(__unused: *const pt_regs) -> i64 {
sys_ni_syscall()
}
