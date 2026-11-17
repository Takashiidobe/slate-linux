
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! __context_unsafe() - function attribute, disable context checking
//! @comment: comment explaining why opt-out is safe
//!
//! Function attribute denoting that context analysis is disabled for the
//! whole function. Forces adding an inline comment as argument.
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
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
//! list_for_each	-	iterate over a list
//! @pos:	the &struct list_head to use as a loop cursor.
//! @head:	the head for your list.
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
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
// Force the compiler to emit 'sym' as a symbol, so that we can reference
// it from inline assembler. Necessary in case 'sym' could be inlined
// otherwise, or eliminated entirely due to lack of references that are
// visible to the compiler.
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
macro_rules! likely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
}

/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
#[repr(C)]
#[derive(Clone, Copy)]
struct list_head {
    next: *mut list_head,
    prev: *mut list_head,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.list_sort.list_sort")]
extern "C" fn list_sort(r#priv: *mut (), head: *mut list_head, cmp: Option<unsafe extern "C" fn(*mut (), *const list_head, *const list_head) -> i32>) {
let mut list: *mut list_head = unsafe { (*head).next };
let mut pending: *mut list_head = core::ptr::null_mut::<list_head>();
let mut count: u64 = 0; // Count of pending
if list == unsafe { (*head).prev } {
return;
}
// Zero or one elements
// Convert to a null-terminated singly-linked list.
unsafe {
(*(*head).prev).next = core::ptr::null_mut::<list_head>();
}
// Data structure invariants:
// - All lists are singly linked and null-terminated; prev
//   pointers are not maintained.
// - pending is a prev-linked "list of lists" of sorted
//   sublists awaiting further merging.
// - Each of the sorted sublists is power-of-two in size.
// - Sublists are sorted by size and age, smallest & newest at front.
// - There are zero to two sublists of each size.
// - A pair of pending sublists are merged as soon as the number
//   of following pending elements equals their size (i.e.
//   each time count reaches an odd multiple of that size).
//   That ensures each later final merge will be at worst 2:1.
// - Each round consists of:
//   - Merging the two sublists selected by the highest bit
//     which flips when count is incremented, and
//   - Adding an element from the input as a size-1 sublist.
loop {
let mut bits: u64 = 0;
let mut tail: *mut *mut list_head = core::ptr::addr_of_mut!(pending);
// Find the least-significant clear bit in count
bits = count;
while bits & 1 != 0 {
tail = unsafe { core::ptr::addr_of_mut!((*(*tail)).prev) };
bits >>= 1i32;
}
// Do the indicated merge
if likely__2!(bits) != 0 {
let mut a: *mut list_head = unsafe { *tail };
let b: *mut list_head = unsafe { (*a).prev };
a = merge(r#priv, cmp, b, a);
// Install the merged result in place of the inputs
unsafe {
(*a).prev = (*b).prev;
}
unsafe {
*tail = a;
}
}
// Move one element from input list to pending
unsafe {
(*list).prev = pending;
}
pending = list;
list = unsafe { (*list).next };
unsafe {
(*pending).next = core::ptr::null_mut::<list_head>();
}
count = count.wrapping_add(1);
if list.is_null() {
break;
}
}
// End of input; merge together all the pending lists.
list = pending;
pending = unsafe { (*pending).prev };
loop {
let next: *mut list_head = unsafe { (*pending).prev };
if next.is_null() {
break;
}
list = merge(r#priv, cmp, pending, list);
pending = next;
}
// The final merge, rebuilding prev links
merge_final(r#priv, cmp, head, pending, list);
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_list_sort: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous", sym list_sort, options(att_syntax));

/// Returns a list organized in an intermediate format suited
/// to chaining of merge() calls: null-terminated, no reserved or
/// sentinel head node, "prev" links not maintained.
fn merge(r#priv: *mut (), cmp: Option<unsafe extern "C" fn(*mut (), *const list_head, *const list_head) -> i32>, mut a: *mut list_head, mut b: *mut list_head) -> *mut list_head {
let mut head: *mut list_head = unsafe { core::mem::zeroed() };
let mut tail: *mut *mut list_head = core::ptr::addr_of_mut!(head);
loop {
// if equal, take 'a' -- important for sort stability
if (unsafe { cmp.unwrap()(r#priv, a as *const list_head, b as *const list_head) }) <= 0 {
unsafe {
*tail = a;
}
tail = unsafe { core::ptr::addr_of_mut!((*a).next) };
a = unsafe { (*a).next };
if a.is_null() {
unsafe {
*tail = b;
}
break;
}
} else {
unsafe {
*tail = b;
}
tail = unsafe { core::ptr::addr_of_mut!((*b).next) };
b = unsafe { (*b).next };
if b.is_null() {
unsafe {
*tail = a;
}
break;
}
}
}
head
}

/// Combine final list merge with restoration of standard doubly-linked
/// list structure.  This approach duplicates code from merge(), but
/// runs faster than the tidier alternatives of either a separate final
/// prev-link restoration pass, or maintaining the prev links
/// throughout.
fn merge_final(r#priv: *mut (), cmp: Option<unsafe extern "C" fn(*mut (), *const list_head, *const list_head) -> i32>, head: *mut list_head, mut a: *mut list_head, mut b: *mut list_head) {
let mut tail: *mut list_head = head;
loop {
// if equal, take 'a' -- important for sort stability
if (unsafe { cmp.unwrap()(r#priv, a as *const list_head, b as *const list_head) }) <= 0 {
unsafe {
(*tail).next = a;
}
unsafe {
(*a).prev = tail;
}
tail = a;
a = unsafe { (*a).next };
if a.is_null() {
break;
}
} else {
unsafe {
(*tail).next = b;
}
unsafe {
(*b).prev = tail;
}
tail = b;
b = unsafe { (*b).next };
if b.is_null() {
b = a;
break;
}
}
}
// Finish linking remainder of list b on to tail
unsafe {
(*tail).next = b;
}
loop {
unsafe {
(*b).prev = tail;
}
tail = b;
b = unsafe { (*b).next };
if b.is_null() {
break;
}
}
// And the final links to make a circular doubly-linked list
unsafe {
(*tail).next = head;
}
unsafe {
(*head).prev = tail;
}
}

// list_sort - sort a list
// @priv: private data, opaque to list_sort(), passed to @cmp
// @head: the list to sort
// @cmp: the elements comparison function
//
// The comparison function @cmp must return > 0 if @a should sort after
// @b ("@a > @b" if you want an ascending sort), and <= 0 if @a should
// sort before @b *or* their original order should be preserved.  It is
// always called with the element that came first in the input in @a,
// and list_sort is a stable sort, so it is not necessary to distinguish
// the @a < @b and @a == @b cases.
//
// The comparison function must adhere to specific mathematical properties
// to ensure correct and stable sorting:
// - Antisymmetry: cmp(@a, @b) must return the opposite sign of
// cmp(@b, @a).
// - Transitivity: if cmp(@a, @b) <= 0 and cmp(@b, @c) <= 0, then
// cmp(@a, @c) <= 0.
//
// This is compatible with two styles of @cmp function:
// - The traditional style which returns <0 / =0 / >0, or
// - Returning a boolean 0/1.
// The latter offers a chance to save a few cycles in the comparison
// (which is used by e.g. plug_ctx_cmp() in block/blk-mq.c).
//
// A good way to write a multi-word comparison is::
//
// 	if (a->high != b->high)
// 		return a->high > b->high;
// 	if (a->middle != b->middle)
// 		return a->middle > b->middle;
// 	return a->low > b->low;
//
//
// This mergesort is as eager as possible while always performing at least
// 2:1 balanced merges.  Given two pending sublists of size 2^k, they are
// merged to a size-2^(k+1) list as soon as we have 2^k following elements.
//
// Thus, it will avoid cache thrashing as long as 3*2^k elements can
// fit into the cache.  Not quite as good as a fully-eager bottom-up
// mergesort, but it does use 0.2*n fewer comparisons, so is faster in
// the common case that everything fits into L1.
//
//
// The merging is controlled by "count", the number of elements in the
// pending lists.  This is beautifully simple code, but rather subtle.
//
// Each time we increment "count", we set one bit (bit k) and clear
// bits k-1 .. 0.  Each time this happens (except the very first time
// for each bit, when count increments to 2^k), we merge two lists of
// size 2^k into one list of size 2^(k+1).
//
// This merge happens exactly when the count reaches an odd multiple of
// 2^k, which is when we have 2^k elements pending in smaller lists,
// so it's safe to merge away two lists of size 2^k.
//
// After this happens twice, we have created two lists of size 2^(k+1),
// which will be merged into a list of size 2^(k+2) before we create
// a third list of size 2^(k+1), so there are never more than two pending.
//
// The number of pending lists of size 2^k is determined by the
// state of bit k of "count" plus two extra pieces of information:
//
// - The state of bit k-1 (when k == 0, consider bit -1 always set), and
// - Whether the higher-order bits are zero or non-zero (i.e.
//   is count >= 2^(k+1)).
//
// There are six states we distinguish.  "x" represents some arbitrary
// bits, and "y" represents some arbitrary non-zero bits:
// 0:  00x: 0 pending of size 2^k;           x pending of sizes < 2^k
// 1:  01x: 0 pending of size 2^k; 2^(k-1) + x pending of sizes < 2^k
// 2: x10x: 0 pending of size 2^k; 2^k     + x pending of sizes < 2^k
// 3: x11x: 1 pending of size 2^k; 2^(k-1) + x pending of sizes < 2^k
// 4: y00x: 1 pending of size 2^k; 2^k     + x pending of sizes < 2^k
// 5: y01x: 2 pending of size 2^k; 2^(k-1) + x pending of sizes < 2^k
// (merge and loop back to state 2)
//
// We gain lists of size 2^k in the 2->3 and 4->5 transitions (because
// bit k-1 is set while the more significant bits are non-zero) and
// merge them away in the 5->2 transition.  Note in particular that just
// before the 5->2 transition, all lower-order bits are 11 (state 3),
// so there is one list of each smaller size.
//
// When we reach the end of the input, we merge all the pending
// lists, from smallest to largest.  If you work through cases 2 to
// 5 above, you can see that the number of elements we merge with a list
// of size 2^k varies from 2^(k-1) (cases 3 and 5 when x == 0) to
// 2^(k+1) - 1 (second merge of case 5 when x == 2^(k-1) - 1).
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_list_sort_29: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut list_head, Option<unsafe extern "C" fn(*mut (), *const list_head, *const list_head) -> i32>)>, *mut ()>(Some(list_sort)) };
