
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! context_unsafe() - disable context checking for contained code
//!
//! Disables context checking for contained statements or expression.
//!
//! .. code-block:: c
//!
//! 	struct some_data {
//! 		spinlock_t lock;
//! 		int counter __guarded_by(&lock);
//! 	};
//!
//! 	int foo(struct some_data *d)
//! 	{
//! 		// ...
//! 		// other code that is still checked ...
//! 		// ...
//! 		return context_unsafe(d->counter);
//! 	}
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//!
//! Convert integer string representation to an integer.
//! If an integer doesn't fit into specified type, -E is returned.
//!
//! Integer starts with optional sign.
//! kstrtou*() functions do not accept sign "-".
//!
//! Radix 0 means autodetection: leading "0x" implies radix 16,
//! leading "0" implies radix 8, otherwise radix is 10.
//! Autodetection hints work after optional sign, but not before.
//!
//! If -E is returned, result is not touched.
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! The type of an index into the pagecache.
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! No such file or directory
//! Value too large for defined data type
//! restart by calling sys_restart_syscall
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
//! @ll: unsigned 64bit dividend
//! @d: unsigned 32bit divisor
//!
//! Divide unsigned 64bit dividend by unsigned 32bit divisor
//! and round up.
//!
//! Return: dividend / divisor rounded up
//! We need to compute the minimum and maximum values representable in a given
//! type. These macros may also be useful elsewhere. It would seem more obvious
//! to do something like:
//!
//! #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
//! #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
//!
//! Unfortunately, the middle expressions, strictly speaking, have
//! undefined behaviour, and at least some versions of gcc warn about
//! the type_max expression (but not if -fsanitize=undefined is in
//! effect; in that case, the warning is deferred to runtime...).
//!
//! The slightly excessive casting in type_min is to make sure the
//! macros also produce sensible values for the exotic type _Bool. [The
//! overflow checkers only almost work for _Bool, but that's
//! a-feature-not-a-bug, since people shouldn't be doing arithmetic on
//! _Bools. Besides, the gcc builtins don't allow _Bool* as third
//! argument.]
//!
//! Idea stolen from
//! https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
//! credit to Christian Biere.
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
//! This counts to 15. Any more, it will return 16th argument.
//! To support Context Analysis, we need to allow the compiler to see the
//! acquisition and release of the context lock. However, the "cleanup" helpers
//! wrap the lock in a struct passed through separate helper functions, which
//! hides the lock alias from the compiler (no inter-procedural analysis).
//!
//! To make it work, we introduce an explicit alias to the context lock instance
//! that is "cleaned" up with a separate cleanup helper. This helper is a dummy
//! function that does nothing at runtime, but has the "_unlock" attribute to
//! tell the compiler what happens at the end of the scope.
//!
//! To generalize the pattern, the WITH_LOCK_GUARD_1_ATTRS() macro should be used
//! to redefine the constructor, which then also creates the alias variable with
//! the right "cleanup" attribute, *after* DECLARE_LOCK_GUARD_1_ATTRS() has been
//! used.
//!
//! Example usage:
//!
//!   DECLARE_LOCK_GUARD_1_ATTRS(mutex, __acquires(_T), __releases(*(struct mutex **)_T))
//!   #define class_mutex_constructor(_T) WITH_LOCK_GUARD_1_ATTRS(mutex, _T)
//!
//! Note: To support the for-loop based scoped helpers, the auxiliary variable
//! must be a pointer to the "class" type because it is defined in the same
//! statement as the guard variable. However, we initialize it with the lock
//! pointer (despite the type mismatch, the compiler's alias analysis still works
//! as expected). The "_unlock" attribute receives a pointer to the auxiliary
//! variable (a double pointer to the class type), and must be cast and
//! dereferenced appropriately.
//! See linux/instrumentation.h
//! Begin/end of an instrumentation safe region
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! Generate a mask for the specified type @t. Additional checks are made to
//! guarantee the value returned fits in that type, relying on
//! -Wshift-count-overflow compiler check to detect incompatible arguments.
//! For example, all these create build errors or warnings:
//!
//! - GENMASK(15, 20): wrong argument order
//! - GENMASK(72, 15): doesn't fit unsigned long
//! - GENMASK_U32(33, 15): doesn't fit in a u32
//! Internal kernel timer frequency
//! Bitfields in the atomic value:
//!
//! When NR_CPUS < 16K
//!  0- 7: locked byte
//!  8-15: pending byte
//! 16-17: tail index
//! 18-31: tail cpu (+1)
//!
//! When NR_CPUS >= 16K
//!  0- 7: locked byte
//!     8: pending
//!  9-10: tail index
//! 11-31: tail cpu (+1)
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! The generic per-CPU infrastrucutre is not suitable for
//! reading const-qualified variables.
//! Maximum supported processors.  Setting this smaller saves quite a
//! bit of memory.  Use nr_cpu_ids instead of this except for static bitmaps.
//!
//! Places which use this should consider cpumask_var_t.
//! Base implementations of per-CPU variable declarations and definitions, where
//! the section in which the variable is to be placed is provided by the
//! 'sec' argument.  This may be used to affect the parameters governing the
//! variable's storage.
//!
//! NOTE!  The sections for the DECLARE and for the DEFINE must match, lest
//! linkage errors occur due the compiler generating the wrong code to access
//! that section.
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
//! CR3's layout varies depending on several things.
//!
//! If CR4.PCIDE is set (64-bit only), then CR3[11:0] is the address space ID.
//! If PAE is enabled, then CR3[11:5] is part of the PDPT address
//! (i.e. it's 32-byte aligned, not page-aligned) and CR3[4:0] is ignored.
//! Otherwise (non-PAE, non-PCID), CR3[3] is PWT, CR3[4] is PCD, and
//! CR3[2:0] and CR3[11:5] are ignored.
//!
//! In all cases, Linux puts zeros in the low ignored bits and in PWT and PCD.
//!
//! CR3[63] is always read as zero.  If CR4.PCIDE is set, then CR3[63] may be
//! written as 1 to prevent the write to CR3 from flushing the TLB.
//!
//! On systems with SME, one bit (in a variable position!) is stolen to indicate
//! that the top-level paging structure is encrypted.
//!
//! On systemms with LAM, bits 61 and 62 are used to indicate LAM mode.
//!
//! All of the remaining bits indicate the physical address of the top-level
//! paging structure.
//!
//! CR3_ADDR_MASK is the mask used by read_cr3_pa().
//!
//! Mask off the address space ID and SME encryption bits.
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! Defines x86 CPU feature bits
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Compile time versions of __arch_hweightN()
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! writeable
//! 4th level page in 5-level paging case
//! Use alternative patching for paravirt calls:
//! - For replacing an indirect call with a direct one, use the "normal"
//!   ALTERNATIVE() macro with the indirect call as the initial code sequence,
//!   which will be replaced with the related direct call by using the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! - In case the replacement is either a direct call or a short code sequence
//!   depending on a feature bit, the ALTERNATIVE_2() macro is being used.
//!   The indirect call is the initial code sequence again, while the special
//!   code sequence is selected with the specified feature bit. In case the
//!   feature is not active, the direct call is used as above via the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! @a is a power of 2 value
//! cpumask_bits - get the bits in a cpumask
//! @maskp: the struct cpumask *
//!
//! You should only assume nr_cpu_ids bits of this mask are valid.  This is
//! a macro so it's const-correct.
//! ZONE_MOVABLE allowed
//! container_of - cast a member of a structure out to the containing structure
//! @ptr:	the pointer to the member.
//! @type:	the type of the container struct this is embedded in.
//! @member:	the name of the member within the struct.
//!
//! WARNING: any const qualifier of @ptr is lost.
//! Do not use container_of() in new code.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! Parameters used to convert the timespec values:
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! rescheduling necessary
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! These constants are defined in the MSan LLVM instrumentation pass.
//! If !LOCKDEP, the value is meaningless
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! If a tracepoint needs to be called from a header file, it is not
//! recommended to call it directly, as tracepoints in header files
//! may cause side-effects and bloat the kernel. Instead, use
//! tracepoint_enabled() to test if the tracepoint is enabled, then if
//! it is, call a wrapper function defined in a C file that will then
//! call the tracepoint.
//!
//! For "trace_foo_bar()", you would need to create a wrapper function
//! in a C file to call trace_foo_bar():
//!   void do_trace_foo_bar(args) { trace_foo_bar(args); }
//! Then in the header file, declare the tracepoint:
//!   DECLARE_TRACEPOINT(foo_bar);
//! And call your wrapper:
//!   static inline void some_inlined_function() {
//!            [..]
//!            if (tracepoint_enabled(foo_bar))
//!                    do_trace_foo_bar(args);
//!            [..]
//!   }
//!
//! Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//!   but is safe to have in headers, where trace_foo_bar_enabled() is not.
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
//! Convenience macros for the sake of set_current_state:
//! Dependency order vs. p above.
//! Allocated nodes are mutable until they have been inserted into the tree,
//! at which time they cannot change their type until they have been removed
//! from the tree and an RCU grace period has passed.
//!
//! Removed nodes have their ->parent set to point to themselves.  RCU readers
//! check ->parent before relying on the value that they loaded from the
//! slots array.  This lets us reuse the slots array for the RCU head.
//!
//! Nodes in the tree point to their parent unless bit 0 is set.
//!
//! 64bit sizes
//! wait_queue_entry::flags
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! Change timeval to jiffies, trying to avoid the
//! most obvious overflows..
//!
//! And some not so obvious.
//!
//! Note that we don't want to return LONG_MAX, because
//! for various timeout reasons we often end up having
//! to wait "jiffies+1" in order to guarantee that we wait
//! at _least_ "jiffies" - so "jiffies+1" had better still
//! be positive.
//! Subtract two ktime_t variables. rem = lhs -rhs:
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! The seqlock seqcount_t interface does not prescribe a precise sequence of
//! read begin/retry/end. For readers, typically there is a call to
//! read_seqcount_begin() and read_seqcount_retry(), however, there are more
//! esoteric cases which do not follow this pattern.
//!
//! As a consequence, we take the following best-effort approach for raw usage
//! via seqcount_t under KCSAN: upon beginning a seq-reader critical section,
//! pessimistically mark the next KCSAN_SEQLOCK_REGION_MAX memory accesses as
//! atomics; if there is a matching read_seqcount_retry() call, no following
//! memory operations are considered atomic. Usage of the seqlock_t interface
//! is not affected.
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! __MAX_NR_ZONES
//! percpu_counter batch for local add or sub
//! Opaque type representing current VMA (vm_area_struct) flag state. Must be
//! accessed via vma_flags_xxx() helper functions.
//! page_folio - Converts from page to folio.
//! @p: The page.
//!
//! Every page is part of a folio.  This function cannot be called on a
//! NULL pointer.
//!
//! Context: No reference, nor lock is required on @page.  If the caller
//! does not hold a reference, this call may race with a folio split, so
//! it should re-check the folio still contains this page after gaining
//! a reference on the folio.
//! Return: The folio which contains this page.
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! We do arithmetic on the LRU lists in various places in the code,
//! so it is important to keep the active lists LRU_ACTIVE higher in
//! the array than the corresponding inactive lists, and to keep
//! the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//!
//! This has to be kept in sync with the statistics in zone_stat_item
//! above and the descriptions in vmstat_text in mm/vmstat.c
//! GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
//! __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
//! entry starting with bit 0. Bit is set if the combination is not
//! allowed.
//! VMA locks do not behave like most ordinary locks found in the kernel, so we
//! cannot quite have full lockdep tracking in the way we would ideally prefer.
//!
//! Read locks act as shared locks which exclude an exclusive lock being
//! taken. We therefore mark these accordingly on read lock acquire/release.
//!
//! Write locks are acquired exclusively per-VMA, but released in a shared
//! fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
//! that write lock is released.
//!
//! We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
//! is the fact that, of course, we do lockdep-track the mmap lock rwsem which
//! must be held when taking a VMA write lock.
//!
//! We do, however, want to indicate that during either acquisition of a VMA
//! write lock or detachment of a VMA that we require the lock held be exclusive,
//! so we utilise lockdep to do so.
//! MXCSR will be initialized
//! mprotect needs to preserve PAT and encryption bits when updating
//! vm_page_prot
//! 6 because 6 should be plenty and struct tlb_state will fit in two cache
//! lines.
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
// __context_unsafe() - function attribute, disable context checking
// @comment: comment explaining why opt-out is safe
//
// Function attribute denoting that context analysis is disabled for the
// whole function. Forces adding an inline comment as argument.
// token_context_lock() - declare an abstract global context lock instance
// @name: token context lock name
//
// Helper that declares an abstract global context lock instance @name, but not
// backed by a real data structure (linker error if accidentally referenced).
// The type name is `__ctx_lock_@name`.
// token_context_lock_instance() - declare another instance of a global context lock
// @ctx: token context lock previously declared with token_context_lock()
// @name: name of additional global context lock instance
//
// Helper that declares an additional instance @name of the same token context
// lock class @ctx. This is helpful where multiple related token contexts are
// declared, to allow using the same underlying type (`__ctx_lock_@ctx`) as
// function arguments.
// __must_hold() - function attribute, caller must hold exclusive context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) exclusively.
// __acquires() - function attribute, function acquires context lock exclusively
//
// Function attribute declaring that the function acquires the given context
// lock instance(s) exclusively, but does not release them.
// __cond_acquires() - function attribute, function conditionally
//                     acquires a context lock exclusively
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x exclusively, but does not release it. The
// function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases() - function attribute, function releases a context lock exclusively
//
// Function attribute declaring that the function releases the given context
// lock instance(s) exclusively. The associated context(s) must be active on
// entry.
// __acquire() - function to acquire context lock exclusively
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x exclusively.
// __release() - function to release context lock exclusively
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x.
// __must_hold_shared() - function attribute, caller must hold shared context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) with shared access.
// __acquires_shared() - function attribute, function acquires context lock shared
//
// Function attribute declaring that the function acquires the given
// context lock instance(s) with shared access, but does not release them.
// __cond_acquires_shared() - function attribute, function conditionally
//                            acquires a context lock shared
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x with shared access, but does not release it.
// The function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases_shared() - function attribute, function releases a
//                       context lock shared
//
// Function attribute declaring that the function releases the given context
// lock instance(s) with shared access. The associated context(s) must be
// active on entry.
// __acquire_shared() - function to acquire context lock shared
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x with shared
// access.
// __release_shared() - function to release context lock shared
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x with shared
// access.
// sparse defines __CHECKER__; see Documentation/dev-tools/sparse.rst
//
// address spaces
// other
// Attributes
// Note: do not use this directly. Instead, use __alloc_size() since it is conditionally
// available and includes other attributes. For GCC < 9.1, __alloc_size__ gets undefined
// in compiler-gcc.h, due to misbehaviors.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-alloc_005fsize-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#alloc-size
// Note: users of __always_inline currently do not write "inline" themselves,
// which seems to be required by gcc to apply the attribute according
// to its docs (and also "warning: always_inline function might not be
// inlinable [-Wattributes]" is emitted).
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-always_005finline-function-attribute
// clang: mentioned
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-cleanup-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#cleanup
// Note the long name.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-format-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#format
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
// Note the missing underscores.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noinline-function-attribute
// clang: mentioned
// Note: deliberately not named '__nonnull', to avoid clashing with glibc's
// __nonnull() when kernel and userspace headers are combined.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Attributes.html#index-nonnull
// clang: https://clang.llvm.org/docs/AttributeReference.html#nonnull
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
// This version is i.e. to prevent dead stores elimination on @ptr
// where gcc and llvm may behave differently when otherwise using
// normal barrier(): while gcc behavior gets along with a normal
// barrier(), llvm needs an explicit input variable to be assumed
// clobbered. The issue is as follows: while the inline asm might
// access any memory it wants, the compiler could have fit all of
// @ptr into memory registers instead, and since @ptr never escaped
// from that, it proved that the inline asm wasn't touching any of
// it. This version works well with both compilers, i.e. we're telling
// the compiler that the inline asm absolutely may see the contents
// of @ptr. See also: https://llvm.org/bugs/show_bug.cgi?id=15495
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
// data_race - mark an expression as containing intentional data races
//
// This data_race() macro is useful for situations in which data races
// should be forgiven.  One example is diagnostic code that accesses
// shared variables but is not a part of the core synchronization design.
// For example, if accesses to a given variable are protected by a lock,
// except for diagnostic code, then the accesses under the lock should
// be plain C-language accesses and those in the diagnostic code should
// use data_race().  This way, KCSAN will complain if buggy lockless
// accesses to that variable are introduced, even if the buggy accesses
// are protected by READ_ONCE() or WRITE_ONCE().
//
// This macro *does not* affect normal code generation, but is a hint
// to tooling that data races here are to be ignored.  If the access must
// be atomic *and* KCSAN should ignore the access, use both data_race()
// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
// Force the compiler to emit 'sym' as a symbol, so that we can reference
// it from inline assembler. Necessary in case 'sym' could be inlined
// otherwise, or eliminated entirely due to lack of references that are
// visible to the compiler.
// This returns a constant expression while determining if an argument is
// a constant expression, most importantly without evaluating the argument.
// Glory to Martin Uecker <Martin.Uecker@med.uni-goettingen.de>
//
// Details:
// - sizeof() return an integer constant expression, and does not evaluate
//   the value of its operand; it only examines the type of its operand.
// - The results of comparing two integer constant expressions is also
//   an integer constant expression.
// - The first literal "8" isn't important. It could be any literal value.
// - The second literal "8" is to avoid warnings about unaligned pointers;
//   this could otherwise just be "1".
// - (long)(x) is used to avoid warnings about 64-bit types on 32-bit
//   architectures.
// - The C Standard defines "null pointer constant", "(void *)0", as
//   distinct from other void pointers.
// - If (x) is an integer constant expression, then the "* 0l" resolves
//   it into an integer constant expression of value 0. Since it is cast to
//   "void *", this makes the second operand a null pointer constant.
// - If (x) is not an integer constant expression, then the second operand
//   resolves to a void pointer (but not a null pointer constant: the value
//   is not an integer constant 0).
// - The conditional operator's third operand, "(int *)8", is an object
//   pointer (to type "int").
// - The behavior (including the return type) of the conditional operator
//   ("operand1 ? operand2 : operand3") depends on the kind of expressions
//   given for the second and third operands. This is the central mechanism
//   of the macro:
//   - When one operand is a null pointer constant (i.e. when x is an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns the type of the
//     object pointer operand (i.e. "int *"). Here, within the sizeof(), we
//     would then get:
//       sizeof(*((int *)(...))  == sizeof(int)  == 4
//   - When one operand is a void pointer (i.e. when x is not an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns a "void *" type.
//     Here, within the sizeof(), we would then get:
//       sizeof(*((void *)(...)) == sizeof(void) == 1
// - The equality comparison to "sizeof(int)" therefore depends on (x):
//     sizeof(int) == sizeof(int)     (x) was a constant expression
//     sizeof(int) != sizeof(void)    (x) was not a constant expression
// Similar to statically_true() but produces a constant expression
//
// To be used in conjunction with macros, such as BUILD_BUG_ON_ZERO(),
// which require their input to be a constant expression and for which
// statically_true() would otherwise fail.
//
// This is a trade-off: const_true() requires all its operands to be
// compile time constants. Else, it would always returns false even on
// the most trivial cases like:
//
//   true || non_const_var
//
// On the opposite, statically_true() is able to fold more complex
// tautologies and will return true on expressions such as:
//
//   !(non_const_var * 8 % 4)
//
// For the general case, statically_true() is better.
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_read_write - check regular read-write access for races
//
// @ptr: address of access
// @size: size of access
// ASSERT_EXCLUSIVE_WRITER - assert no concurrent writes to @var
//
// Assert that there are no concurrent writes to @var; other readers are
// allowed. This assertion can be used to specify properties of concurrent code,
// where violation cannot be detected as a normal data race.
//
// For example, if we only have a single writer, but multiple concurrent
// readers, to avoid data races, all these accesses must be marked; even
// concurrent marked writes racing with the single writer are bugs.
// Unfortunately, due to being marked, they are no longer data races. For cases
// like these, we can use the macro as follows:
//
// .. code-block:: c
//
// 	void writer(void) {
// 		spin_lock(&update_foo_lock);
// 		ASSERT_EXCLUSIVE_WRITER(shared_foo);
// 		WRITE_ONCE(shared_foo, ...);
// 		spin_unlock(&update_foo_lock);
// 	}
// 	void reader(void) {
// 		// update_foo_lock does not need to be held!
// 		... = READ_ONCE(shared_foo);
// 	}
//
// Note: ASSERT_EXCLUSIVE_WRITER_SCOPED(), if applicable, performs more thorough
// checking if a clear scope where no concurrent writes are expected exists.
//
// @var: variable to assert on
// ASSERT_EXCLUSIVE_BITS - assert no concurrent writes to subset of bits in @var
//
// Bit-granular variant of ASSERT_EXCLUSIVE_WRITER().
//
// Assert that there are no concurrent writes to a subset of bits in @var;
// concurrent readers are permitted. This assertion captures more detailed
// bit-level properties, compared to the other (word granularity) assertions.
// Only the bits set in @mask are checked for concurrent modifications, while
// ignoring the remaining bits, i.e. concurrent writes (or reads) to ~mask bits
// are ignored.
//
// Use this for variables, where some bits must not be modified concurrently,
// yet other bits are expected to be modified concurrently.
//
// For example, variables where, after initialization, some bits are read-only,
// but other bits may still be modified concurrently. A reader may wish to
// assert that this is true as follows:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (READ_ONCE(flags) & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Note: The access that immediately follows ASSERT_EXCLUSIVE_BITS() is assumed
// to access the masked bits only, and KCSAN optimistically assumes it is
// therefore safe, even in the presence of data races, and marking it with
// READ_ONCE() is optional from KCSAN's point-of-view. We caution, however, that
// it may still be advisable to do so, since we cannot reason about all compiler
// optimizations when it comes to bit manipulations (on the reader and writer
// side). If you are sure nothing can go wrong, we can write the above simply
// as:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (flags & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Another example, where this may be used, is when certain bits of @var may
// only be modified when holding the appropriate lock, but other bits may still
// be modified concurrently. Writers, where other bits may change concurrently,
// could use the assertion as follows:
//
// .. code-block:: c
//
// 	spin_lock(&foo_lock);
// 	ASSERT_EXCLUSIVE_BITS(flags, FOO_MASK);
// 	old_flags = flags;
// 	new_flags = (old_flags & ~FOO_MASK) | (new_foo << FOO_SHIFT);
// 	if (cmpxchg(&flags, old_flags, new_flags) != old_flags) { ... }
// 	spin_unlock(&foo_lock);
//
// @var: variable to assert on
// @mask: only check for modifications to bits set in @mask
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// Invalid argument
// Math result not representable
// Operation is not supported
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
// abs - return absolute value of an argument
// @x: the value.
//
// If it is unsigned type, @x is converted to signed type first.
// char is treated as if it was signed (regardless of whether it really is)
// but the macro's return type is preserved as char.
//
// NOTE, for signed type if @x is the minimum, the returned result is undefined
// as there is not enough bits to represent it as a positive number.
//
// Return: an absolute value of @x.
// Concatenate two parameters, but allow them to be expanded beforehand.
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
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
// reg := -EIO
// reg := -EIO
// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
// expression but avoids the generation of any code, even if that expression
// has side-effects.
// BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
// 		      error message.
// @cond: the condition which the compiler should know is false.
// @msg: build-time error message
//
// See BUILD_BUG_ON for description.
// BUILD_BUG_ON - break compile if a condition is true.
// @condition: the condition which the compiler should know is false.
//
// If you have some code which relies on certain constants being equal, or
// some other compile-time-evaluated condition, you should use BUILD_BUG_ON to
// detect if someone changes it.
// BUILD_BUG - break compile if used.
//
// If you have some code that you expect the compiler to eliminate at
// build time, you should use BUILD_BUG to detect if it is
// unexpectedly used.
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// printk - print a kernel message
// @fmt: format string
//
// This is printk(). It can be called from any context. We want it to work.
//
// If printk indexing is enabled, _printk() is called from printk_index_wrap.
// Otherwise, printk is simply #defined to _printk.
//
// We try to grab the console_lock. If we succeed, it's easy - we log the
// output and call the console drivers.  If we fail to get the semaphore, we
// place the output into the log buffer and return. The current holder of
// the console_sem will notice the new output in console_unlock(); and will
// send it to the consoles before releasing the lock.
//
// One effect of this deferred printing is that code which calls printk() and
// then changes console_loglevel may break. This is because console_loglevel
// is inspected when the actual printing occurs.
//
// See also:
// printf(3)
//
// See the vsnprintf() documentation for format string extensions over C99.
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// new instruction
// feature + flags
// source len
// replacement len
// replacement
// alternative assembly primitive:
// If @feature is set, patch in @newinstr_yes, otherwise @newinstr_no.
// Alternative instructions for different CPU types or capabilities.
//
// This allows to use optimized instructions even on generic binary
// kernels.
//
// length of oldinstr must be longer or equal the length of newinstr
// It can be padded with nops as needed.
//
// For non barrier like inlines please define new variants
// without volatile and memory clobber.
// Alternative inline assembly with input.
//
// Peculiarities:
// No memory clobber here.
// Argument numbers start with 1.
// Leaving an unused argument 0 to keep API compatibility.
// Like alternative_input, but with a single output argument
// Atomic operations are already serializing on x86
// Generate a per-CPU add to memory instruction and optimize code
// if one is added or subtracted.
// Add return operation
// this_cpu_read() makes the compiler load the per-CPU variable every time
// it is accessed while this_cpu_read_stable() allows the value to be cached.
// this_cpu_read_stable() is more efficient and can be used if its value
// is guaranteed to be valid across CPUs.  The current users include
// current_task and cpu_current_top_of_stack, both of which are
// actually per-thread variables implemented as per-CPU variables and
// thus stable for the duration of the respective task.
// alpha modules require percpu variables to be defined as
// weak to force the compiler to generate GOT based external
// references for them.  This is necessary because percpu sections
// will be located outside of the usually addressable area.
//
// This definition puts the following two extra restrictions when
// defining percpu variables.
//
// 1. The symbol must be globally unique, even the static ones.
// 2. Static percpu variables cannot be defined inside a function.
//
// Archs which need weak percpu definitions should set
// CONFIG_ARCH_MODULE_NEEDS_WEAK_PER_CPU when necessary.
//
// Normal declaration and definition macros.
// Variant on the per-CPU variable declaration/definition theme used for
// ordinary per-CPU variables.
// Declaration/definition used for per-CPU variables that are frequently
// accessed and should be in a single cacheline.
//
// For use only by architecture and core code.  Only use scalar or pointer
// types to maximize density.
// Declaration/definition used for per-CPU variables that must be cacheline
// aligned under SMP conditions so that, whilst a particular instance of the
// data corresponds to a particular CPU, inefficiencies due to direct access by
// other CPUs are reduced by preventing the data from unnecessarily spanning
// cachelines.
//
// An example of this would be statistical data, where each CPU's set of data
// is updated by that CPU alone, but the data from across all CPUs is collated
// by a CPU processing a read from a proc file.
// Declaration/definition used for per-CPU variables that must be page aligned.
// Declaration/definition used for per-CPU variables that must be read mostly.
// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
// @ptr and is invoked once before a percpu area is accessed by all
// accessors and operations.  This is performed in the generic part of
// percpu and arch overrides don't need to worry about it; however, if an
// arch wants to implement an arch-specific percpu accessor or operation,
// it may use __verify_pcpu_ptr() to verify the parameters.
//
// + 0 is required in order to convert the pointer type from a
// potential array type to a pointer to a single item of the array.
// Add an offset to a pointer.  Use RELOC_HIDE() to prevent the compiler
// from making incorrect assumptions about the pointer value.
// Operations for contexts where we do not want to do any checks for
// preemptions.  Unless strictly necessary, always use [__]this_cpu_*()
// instead.
//
// If there is no other protection through preempt disable and/or disabling
// interrupts then one of these RMW operations can show unexpected behavior
// because the execution thread was rescheduled on another processor or an
// interrupt occurred and the same percpu variable was modified from the
// interrupt context.
// Operations for contexts that are safe from preemption/interrupts.  These
// operations verify that preemption is disabled.
// Operations with implied preemption/interrupt protection.  These
// operations can be used without worrying about preemption or interrupt.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// "erms" Enhanced REP MOVSB/STOSB instructions
// "smap" Supervisor Mode Access Prevention
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "lass" Linear Address Space Separation
// Fast short REP STOSB
// Non-serializing WRMSR
// INVLPGB and TLBSYNC instructions supported
// "wbnoinvd" WBNOINVD instruction
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
// __CPUID_LEAF() - Define a CPUID output and parse info entry
// @_name:	Struct type name of the CPUID leaf/subleaf (e.g. 'leaf_0x7_0'). Such
// 		types are defined at <cpuid/leaf_types.h> and follow the leaf_0xM_N
// 		format, where 0xM is the leaf and N is the subleaf.
// @_count:	Number of storage entries to allocate for this leaf/subleaf.
//
// For a given leaf/subleaf, define an array of CPUID storage entries and an associated
// query info structure.
//
// Use an array of storage entries to accommodate CPUID leaves with multiple subleaves
// having the same output format.  This is common for hierarchical enumeration; e.g.,
// CPUID(0x4), CPUID(0x12), and CPUID(0x8000001d).
// CPUID_LEAF() - Define a 'struct cpuid_leaves' storage entry
// @_leaf:	Leaf number, in compile-time 0xN format
// @_subleaf:	Subleaf number, in compile-time decimal format
//
// Convenience wrapper around __CPUID_LEAF().
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
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
// was written to (raised by CPU)
// on 4KB pages
// On 2MB or 1GB pages
// Protection Keys, bit 1/4
// Saved Dirty bit (leaf)
// The hardware requires shadow stack to be Write=0,Dirty=1. However,
// there are valid cases where the kernel might create read-only PTEs that
// are dirty (e.g., fork(), mprotect(), userfaultfd, soft-dirty tracking). In
// this case, the _PAGE_SAVED_DIRTY bit is used instead of the HW-dirty bit,
// to avoid creating a wrong "shadow stack" PTEs. Such PTEs have
// (Write=0,SavedDirty=1,Dirty=0) set.
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
// No need to mask any bits for !PAE
// Note: no "lock" prefix even on SMP: xchg always implies lock anyway.
// Since this is generally used to protect other memory information, we
// use "asm volatile" and "memory" clobbers to prevent gcc from moving
// information around.
// Atomic compare and exchange.  Compare OLD with MEM, if identical,
// store NEW in MEM.  Return the initial value in MEM.  Success is
// indicated by comparing RETURN with OLD.
// xadd() adds "inc" to "*ptr" and atomically returns the previous
// value of "*ptr".
//
// xadd() is locked when multiple CPUs are online
// DOC: Reclaim modifiers
//
// Reclaim modifiers
// Please note that all the following flags are only applicable to sleepable
// allocations (e.g. %GFP_NOWAIT and %GFP_ATOMIC will ignore them).
//
// %__GFP_IO can start physical IO.
//
// %__GFP_FS can call down to the low-level FS. Clearing the flag avoids the
// allocator recursing into the filesystem which might already be holding
// locks.
//
// %__GFP_DIRECT_RECLAIM indicates that the caller may enter direct reclaim.
// This flag can be cleared to avoid unnecessary delays when a fallback
// option is available.
//
// %__GFP_KSWAPD_RECLAIM indicates that the caller wants to wake kswapd when
// the low watermark is reached and have it reclaim pages until the high
// watermark is reached. A caller may wish to clear this flag when fallback
// options are available and the reclaim is likely to disrupt the system. The
// canonical example is THP allocation where a fallback is cheap but
// reclaim/compaction may cause indirect stalls.
//
// %__GFP_RECLAIM is shorthand to allow/forbid both direct and kswapd reclaim.
//
// The default allocator behavior depends on the request size. We have a concept
// of so-called costly allocations (with order > %PAGE_ALLOC_COSTLY_ORDER).
// !costly allocations are too essential to fail so they are implicitly
// non-failing by default (with some exceptions like OOM victims might fail so
// the caller still has to check for failures) while costly requests try to be
// not disruptive and back off even without invoking the OOM killer.
// The following three modifiers might be used to override some of these
// implicit rules. Please note that all of them must be used along with
// %__GFP_DIRECT_RECLAIM flag.
//
// %__GFP_NORETRY: The VM implementation will try only very lightweight
// memory direct reclaim to get some memory under memory pressure (thus
// it can sleep). It will avoid disruptive actions like OOM killer. The
// caller must handle the failure which is quite likely to happen under
// heavy memory pressure. The flag is suitable when failure can easily be
// handled at small cost, such as reduced throughput.
//
// %__GFP_RETRY_MAYFAIL: The VM implementation will retry memory reclaim
// procedures that have previously failed if there is some indication
// that progress has been made elsewhere.  It can wait for other
// tasks to attempt high-level approaches to freeing memory such as
// compaction (which removes fragmentation) and page-out.
// There is still a definite limit to the number of retries, but it is
// a larger limit than with %__GFP_NORETRY.
// Allocations with this flag may fail, but only when there is
// genuinely little unused memory. While these allocations do not
// directly trigger the OOM killer, their failure indicates that
// the system is likely to need to use the OOM killer soon.  The
// caller must handle failure, but can reasonably do so by failing
// a higher-level request, or completing it only in a much less
// efficient manner.
// If the allocation does fail, and the caller is in a position to
// free some non-essential memory, doing so could benefit the system
// as a whole.
//
// %__GFP_NOFAIL: The VM implementation _must_ retry infinitely: the caller
// cannot handle allocation failures. The allocation could block
// indefinitely but will never return with failure. Testing for
// failure is pointless.
// It _must_ be blockable and used together with __GFP_DIRECT_RECLAIM.
// It should _never_ be used in non-sleepable contexts.
// New users should be evaluated carefully (and the flag should be
// used only when there is no reasonable failure policy) but it is
// definitely preferable to use the flag rather than opencode endless
// loop around allocator.
// Allocating pages from the buddy with __GFP_NOFAIL and order > 1 is
// not supported. Please consider using kvmalloc() instead.
// Caller can reclaim
// DOC: Action modifiers
//
// Action modifiers
//
// %__GFP_NOWARN suppresses allocation failure reports.
//
// %__GFP_COMP address compound page metadata.
//
// %__GFP_ZERO returns a zeroed page on success.
//
// %__GFP_ZEROTAGS zeroes memory tags at allocation time. Setting memory tags at
// the same time as zeroing memory (e.g., with __GFP_ZERO) has minimal
// additional performance impact. However, __GFP_ZEROTAGS also zeroes the tags
// even if memory is not getting zeroed at allocation time (e.g.,
// with init_on_free).
//
// %__GFP_SKIP_KASAN makes KASAN skip unpoisoning on page allocation.
// Used for userspace and vmalloc pages; the latter are unpoisoned by
// kasan_unpoison_vmalloc instead. If passed to vmalloc, kasan_unpoison_vmalloc
// is skipped too. For userspace pages, results in poisoning being skipped as
// well, see should_skip_kasan_poison for details. Only effective in HW_TAGS mode.
// Disable lockdep for GFP context tracking
// DOC: Useful GFP flag combinations
//
// Useful GFP flag combinations
//
// Useful GFP flag combinations that are commonly used. It is recommended
// that subsystems start with one of these combinations and then set/clear
// %__GFP_FOO flags as necessary.
//
// %GFP_ATOMIC users can not sleep and need the allocation to succeed. A lower
// watermark is applied to allow access to "atomic reserves".
// The current implementation doesn't support NMI, nor contexts that disable
// preemption under PREEMPT_RT. This includes raw_spin_lock() and plain
// preempt_disable() - see "Memory allocation" in
// Documentation/core-api/real-time/differences.rst for more info.
//
// %GFP_KERNEL is typical for kernel-internal allocations. The caller requires
// %ZONE_NORMAL or a lower zone for direct access but can direct reclaim.
//
// %GFP_KERNEL_ACCOUNT is the same as GFP_KERNEL, except the allocation is
// accounted to the active cgroup context.
//
// %GFP_NOWAIT is for kernel allocations that should not stall for direct
// reclaim, start physical IO or use any filesystem callback.  It is very
// likely to fail to allocate memory, even for very small allocations.
// The same restrictions on calling contexts apply as for %GFP_ATOMIC.
//
// %GFP_NOIO will use direct reclaim to discard clean pages or slab pages
// that do not require the starting of any physical IO.
// Please try to avoid using this flag directly and instead use
// memalloc_noio_{save,restore} to mark the whole scope which cannot
// perform any IO with a short explanation why. All allocation requests
// will inherit GFP_NOIO implicitly.
//
// %GFP_NOFS will use direct reclaim but will not use any filesystem interfaces.
// Please try to avoid using this flag directly and instead use
// memalloc_nofs_{save,restore} to mark the whole scope which cannot/shouldn't
// recurse into the FS layer with a short explanation why. All allocation
// requests will inherit GFP_NOFS implicitly.
//
// %GFP_USER is for userspace allocations that also need to be directly
// accessibly by the kernel or hardware. It is typically used by hardware
// for buffers that are mapped to userspace (e.g. graphics) that hardware
// still must DMA to. cpuset limits are enforced for these allocations.
//
// %GFP_DMA exists for historical reasons and should be avoided where possible.
// The flags indicates that the caller requires that the lowest zone be
// used (%ZONE_DMA or 16M on x86-64). Ideally, this would be removed but
// it would require careful auditing as some users really require it and
// others use the flag to avoid lowmem reserves in %ZONE_DMA and treat the
// lowest zone as a type of emergency reserve.
//
// %GFP_DMA32 is similar to %GFP_DMA except that the caller requires a 32-bit
// address. Note that kmalloc(..., GFP_DMA32) does not return DMA32 memory
// because the DMA32 kmalloc cache array is not implemented.
// (Reason: there is no such user in kernel).
//
// %GFP_HIGHUSER is for userspace allocations that may be mapped to userspace,
// do not need to be directly accessible by the kernel but that cannot
// move once in use. An example may be a hardware allocation that maps
// data directly into userspace but has no addressing limitations.
//
// %GFP_HIGHUSER_MOVABLE is for userspace allocations that the kernel does not
// need direct access to but can use kmap() when access is required. They
// are expected to be movable via page reclaim or page migration. Typically,
// pages on the LRU would also be allocated with %GFP_HIGHUSER_MOVABLE.
//
// %GFP_TRANSHUGE and %GFP_TRANSHUGE_LIGHT are used for THP allocations. They
// are compound allocations that will generally fail quickly if memory is not
// available and will not wake kswapd/kcompactd on failure. The _LIGHT
// version does not attempt reclaim/compaction at all and is by default used
// in page fault path, while the non-light is used by khugepaged.
// list_first_entry - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note, that list is expected to be not empty.
// list_next_entry - get the next element in list
// @pos:	the type * to cursor
// @member:	the name of the list_head within the struct.
// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_entry_is_head - test if the entry points to the head of the list
// @pos:	the type * to cursor
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
// N 32-bit words worth of info
// The preempt_count offset needed for things like:
//
//  spin_lock_bh()
//
// Which need to disable both preemption (CONFIG_PREEMPT_COUNT) and
// softirqs, such that unlock sequences of:
//
//  spin_unlock();
//  local_bh_enable();
//
// Work as expected.
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
// get_task_state():
// set_current_state() includes a barrier so that the write of current->__state
// is correctly serialised wrt the caller's subsequent test of whether to
// actually sleep:
//
//   for (;;) {
// 	set_current_state(TASK_UNINTERRUPTIBLE);
// 	if (CONDITION)
// 	   break;
//
// 	schedule();
//   }
//   __set_current_state(TASK_RUNNING);
//
// If the caller does not need such serialisation (because, for instance, the
// CONDITION test and condition change and wakeup are under the same lock) then
// use __set_current_state().
//
// The above is typically ordered against the wakeup, which does:
//
//   CONDITION = 1;
//   wake_up_state(p, TASK_UNINTERRUPTIBLE);
//
// where wake_up_state()/try_to_wake_up() executes a full memory barrier before
// accessing p->__state.
//
// Wakeup will do: if (@state & p->__state) p->__state = TASK_RUNNING, that is,
// once it observes the TASK_UNINTERRUPTIBLE store the waking CPU can issue a
// TASK_RUNNING store which can collide with __set_current_state(TASK_RUNNING).
//
// However, with slightly different timing the wakeup TASK_RUNNING store can
// also collide with the TASK_UNINTERRUPTIBLE store. Losing that store is not
// a problem either because that will result in one extra go around the loop
// and our @cond test will save the day.
//
// Also see the comments of try_to_wake_up().
// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
// All allocations inherit GFP_NOFS. See memalloc_nfs_save()
// All allocations inherit GFP_NOIO. See memalloc_noio_save()
// I am a kernel thread
// Userland is not allowed to meddle with cpus_mask
// Allocations constrained to zones which allow long term pinning.
// See memalloc_pin_save()
// Per-process atomic flags.
// May not gain new privileges.
// Spread page cache over cpuset
// Speculative Store Bypass disabled
// Speculative Store Bypass force disabled
// Indirect branch speculation restricted
// Indirect branch speculation permanently restricted
// Speculative Store Bypass clear on execve()
// Dependency order vs. p above.
// RCU_INITIALIZER() - statically initialize an RCU-protected global variable
// @v: The value to statically initialize with.
// rcu_assign_pointer() - assign to RCU-protected pointer
// @p: pointer to assign to
// @v: value to assign (publish)
//
// Assigns the specified value to the specified RCU-protected
// pointer, ensuring that any concurrent RCU readers will see
// any prior initialization.
//
// Inserts memory barriers on architectures that require them
// (which is most of them), and also prevents the compiler from
// reordering the code that initializes the structure after the pointer
// assignment.  More importantly, this call documents which pointers
// will be dereferenced by RCU read-side code.
//
// In some special cases, you may use RCU_INIT_POINTER() instead
// of rcu_assign_pointer().  RCU_INIT_POINTER() is a bit faster due
// to the fact that it does not constrain either the CPU or the compiler.
// That said, using RCU_INIT_POINTER() when you should have used
// rcu_assign_pointer() is a very bad thing that results in
// impossible-to-diagnose memory corruption.  So please be careful.
// See the RCU_INIT_POINTER() comment header for details.
//
// Note that rcu_assign_pointer() evaluates each of its arguments only
// once, appearances notwithstanding.  One of the "extra" evaluations
// is in typeof() and the other visible only to sparse (__CHECKER__),
// neither of which actually execute the argument.  As with most cpp
// macros, this execute-arguments-only-once property is important, so
// please be careful when making changes to rcu_assign_pointer() and the
// other macros that it invokes.
// rcu_dereference_check() - rcu_dereference with debug checking
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Do an rcu_dereference(), but check that the conditions under which the
// dereference will take place are correct.  Typically the conditions
// indicate the various locking conditions that should be held at that
// point.  The check should return true if the conditions are satisfied.
// An implicit check for being in an RCU read-side critical section
// (rcu_read_lock()) is included.
//
// For example:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock));
//
// could be used to indicate to lockdep that foo->bar may only be dereferenced
// if either rcu_read_lock() is held, or that the lock required to replace
// the bar struct at foo->bar is held.
//
// Note that the list of conditions may also include indications of when a lock
// need not be held, for example during initialisation or destruction of the
// target struct:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock) ||
// 					      atomic_read(&foo->usage) == 0);
//
// Inserts memory barriers on architectures that require them
// (currently only the Alpha), prevents the compiler from refetching
// (and from merging fetches), and, more importantly, documents exactly
// which pointers are protected by RCU and checks that the pointer is
// annotated as __rcu.
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
// 256 bytes including ->parent
// 256 bytes
// 240 bytes
// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
// SEQCOUNT_LOCKNAME()	- Instantiate seqcount_LOCKNAME_t and helpers
// seqprop_LOCKNAME_*()	- Property accessors for seqcount_LOCKNAME_t
//
// @lockname:		"LOCKNAME" part of seqcount_LOCKNAME_t
// @locktype:		LOCKNAME canonical C data type
// @preemptible:	preemptibility of above locktype
// @lockbase:		prefix for associated lock/unlock
// 									 * Re-read the sequence counter since the (possibly			 * preempted) writer made progress.
// PREEMPT_RT relies on the above LOCK+UNLOCK
// __read_seqcount_begin() - begin a seqcount_t read section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount_begin() - begin a seqcount_t read section w/o lockdep
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// read_seqcount_begin() - begin a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount() - read the raw seqcount_t counter value
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// raw_read_seqcount opens a read critical section of the given
// seqcount_t, without any lockdep checking, and without checking or
// masking the sequence counter LSB. Calling code is responsible for
// handling that.
//
// Return: count to be passed to read_seqcount_retry()
// raw_seqcount_try_begin() - begin a seqcount_t read critical section
//                            w/o lockdep and w/o counter stabilization
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count to be passed to read_seqcount_retry()
//
// Similar to raw_seqcount_begin(), except it enables eliding the critical
// section entirely if odd, instead of doing the speculation knowing it will
// fail.
//
// Useful when counter stabilization is more or less equivalent to taking
// the lock and there is a slowpath that does that.
//
// If true, start will be set to the (even) sequence count read.
//
// Return: true when a read critical section is started.
// read_seqcount_retry() - end a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count, from read_seqcount_begin()
//
// read_seqcount_retry closes the read critical section of given
// seqcount_t.  If the critical section was invalid, it must be ignored
// (and typically retried).
//
// Return: true if a read section retry is required, else false
// seqlock_init() - dynamic initializer for seqlock_t
// @sl: Pointer to the seqlock_t instance
// read_seqlock_excl_irqsave() - start a non-interruptible seqlock_t
// 				 locking reader section
// @lock:  Pointer to seqlock_t
// @flags: Stack-allocated storage for saving caller's local interrupt
//         state, to be passed to read_sequnlock_excl_irqrestore().
//
// _irqsave variant of read_seqlock_excl(). Use this only if the seqlock_t
// write side section, *or other read sections*, can be invoked from a
// hardirq context.
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
// Page flags policies wrt compound pages
//
// PF_POISONED_CHECK
//     check if this struct page poisoned/uninitialized
//
// PF_ANY:
//     the page flag is relevant for small, head and tail pages.
//
// PF_HEAD:
//     for compound page all operations related to the page flag applied to
//     head page.
//
// PF_NO_TAIL:
//     modifications of the page flag must be done on small or head pages,
//     checks can be done on tail pages too.
//
// PF_NO_COMPOUND:
//     the page flag is not relevant for compound pages.
//
// PF_SECOND:
//     the page flag is stored in the first tail page.
// Which page is the flag stored in
// Macros to create function definitions for page flags
// On an anonymous folio mapped into a user virtual memory area,
// folio->mapping points to its anon_vma, not to a struct address_space;
// with the FOLIO_MAPPING_ANON bit set to distinguish it.  See rmap.h.
//
// On an anonymous folio in a VM_MERGEABLE area, if CONFIG_KSM is enabled,
// the FOLIO_MAPPING_ANON_KSM bit may be set along with the FOLIO_MAPPING_ANON
// bit; and then folio->mapping points, not to an anon_vma, but to a private
// structure which KSM associates with that merged folio.  See ksm.h.
//
// Please note that, confusingly, "folio_mapping" refers to the inode
// address_space which maps the folio from disk; whereas "folio_mapped"
// refers to user virtual address space into which the folio is mapped.
//
// For slab pages, since slab reuses the bits in struct page to store its
// internal states, the folio->mapping does not exist as such, nor do
// these flags below.  So in order to avoid testing non-existent bits,
// please make sure that folio_test_slab(folio) actually evaluates to
// false before calling the following functions (e.g., folio_test_anon).
// See mm/slab.h.
// local_lock - Acquire a per CPU local lock
// @lock:	The lock variable
// local_lock_irq - Acquire a per CPU local lock and disable interrupts
// @lock:	The lock variable
// local_lock_irqsave - Acquire a per CPU local lock, save and disable
// 			 interrupts
// @lock:	The lock variable
// @flags:	Storage for interrupt flags
// local_unlock - Release a per CPU local lock
// @lock:	The lock variable
// local_unlock_irq - Release a per CPU local lock and enable interrupts
// @lock:	The lock variable
// local_unlock_irqrestore - Release a per CPU local lock and restore
// 			      interrupt flags
// @lock:	The lock variable
// @flags:      Interrupt flags to restore
// local_trylock_init - Runtime initialize a lock instance
// @lock:	The lock variable
// Maximum number of zones on a zonelist
// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// Please ignore the name of this function.  It should be called
// switch_to_kernel_thread().
//
// enter_lazy_tlb() is a hint from the scheduler that we are entering a
// kernel thread or other context without an mm.  Acceptable implementations
// include doing nothing whatsoever, switching to init_mm, or various clever
// lazy tricks to try to minimize TLB flushes.
//
// The scheduler reserves the right to call enter_lazy_tlb() several times
// in a row.  It will notify us that we're going back to a real mm by
// calling switch_mm_irqs_off().
// Keep stack-allocated flush_tlb_info cacheline aligned, but cap the
// alignment to avoid excessive stack usage on large-cacheline systems.
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
macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED__2 {
() => {
__or!()
};
}

macro_rules! ____or {
() => {
1i32
};
}

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __ismask {
($x:expr) => {
unsafe { *(core::ptr::addr_of!(_ctype) as *const u8).wrapping_offset(((($x) as u8) as i32) as isize) }
};
}

macro_rules! __or {
() => {
____or!()
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use__2 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u64>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
};
}

/// check_add_overflow() - Calculate addition with overflow checking
/// @a: first addend
/// @b: second addend
/// @d: pointer to store sum
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted addition, regardless of whether
/// wrap-around occurred.
macro_rules! check_add_overflow {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_add(__t2);
unsafe {
*$d = __t3 as u64;
}
__t.checked_add(__t2).is_none() || ((__t3 as u64) as i128) != __t3
})
};
}

/// check_mul_overflow() - Calculate multiplication with overflow checking
/// @a: first factor
/// @b: second factor
/// @d: pointer to store product
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted multiplication, regardless of whether
/// wrap-around occurred.
macro_rules! check_mul_overflow {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_mul(__t2);
unsafe {
*$d = __t3 as u64;
}
__t.checked_mul(__t2).is_none() || ((__t3 as u64) as i128) != __t3
})
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
};
}

macro_rules! isxdigit {
($c:expr) => {
(__ismask!($c) as i32) & (4 | 64) != 0
};
}

/// Useful shorthand for "is this condition known at compile-time?"
///
/// Note that the condition may involve non-constant values,
/// but the compiler may know enough about the details of the
/// values to determine that the condition is statically true.
macro_rules! statically_true {
($x:expr) => {
(unsafe { __slate_intrinsic_1cfe0293d0a2af8d($x) }) && $x
};
}

macro_rules! unlikely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    static mut _ctype: [u8; 0];
    fn int_pow(arg0: u64, arg1: u32) -> u64;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn _copy_from_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn __compiletime_assert_282();
    fn __compiletime_assert_286();
    fn __compiletime_assert_290();
    fn __compiletime_assert_294();
    fn __compiletime_assert_298();
    fn __compiletime_assert_302();
    fn __compiletime_assert_306();
    fn __compiletime_assert_310();
    fn __compiletime_assert_314();
    fn __compiletime_assert_318();
    fn __compiletime_assert_322();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

fn isdigit(c: i32) -> i32 {
(48 <= c && c <= 57) as i32
}

fn _tolower(c: u8) -> u8 {
((c as i32) | 32) as u8
}

fn kstrtoul(s: *const u8, base: u32, res: *mut u64) -> i32 {
// We want to shortcut function call, but
// __builtin_types_compatible_p(unsigned long, unsigned long long) = 0.
if ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u64>() }) as u64) && ((const { core::mem::align_of::<u64>() }) as u64) == ((const { core::mem::align_of::<u64>() }) as u64) {
return kstrtoull(s, base, res as *mut u64);
} else {
return _kstrtoul(s, base, res);
}
}

fn kstrtol(s: *const u8, base: u32, res: *mut i64) -> i32 {
// We want to shortcut function call, but
// __builtin_types_compatible_p(long, long long) = 0.
if ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) && ((const { core::mem::align_of::<i64>() }) as u64) == ((const { core::mem::align_of::<i64>() }) as u64) {
return kstrtoll(s, base, res as *mut i64);
} else {
return _kstrtol(s, base, res);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtoull_from_user")]
extern "C" fn kstrtoull_from_user(s: *const u8, mut count: u64, base: u32, res: *mut u64) -> i32 {
let mut buf: __SlateAlign16<[u8; 67]> = __SlateAlign16([0u8; 67]);
let mut __v0: u64;
let __UNIQUE_ID_x__284: u64 = count;
let __UNIQUE_ID_y__285: u64 = (core::mem::size_of::<[u8; 67]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__284, __UNIQUE_ID_y__285) == 0 {
unsafe { __compiletime_assert_286() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__284 < __UNIQUE_ID_y__285 { __UNIQUE_ID_x__284 } else { __UNIQUE_ID_y__285 };
count = __v0;
if copy_from_user(buf.0.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf.0[count as usize] = 0;
kstrtoull(buf.0.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtoll_from_user")]
extern "C" fn kstrtoll_from_user(s: *const u8, mut count: u64, base: u32, res: *mut i64) -> i32 {
let mut buf: __SlateAlign16<[u8; 67]> = __SlateAlign16([0u8; 67]);
let mut __v0: u64;
let __UNIQUE_ID_x__288: u64 = count;
let __UNIQUE_ID_y__289: u64 = (core::mem::size_of::<[u8; 67]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__288, __UNIQUE_ID_y__289) == 0 {
unsafe { __compiletime_assert_290() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__288 < __UNIQUE_ID_y__289 { __UNIQUE_ID_x__288 } else { __UNIQUE_ID_y__289 };
count = __v0;
if copy_from_user(buf.0.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf.0[count as usize] = 0;
kstrtoll(buf.0.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtoul_from_user")]
extern "C" fn kstrtoul_from_user(s: *const u8, mut count: u64, base: u32, res: *mut u64) -> i32 {
let mut buf: __SlateAlign16<[u8; 67]> = __SlateAlign16([0u8; 67]);
let mut __v0: u64;
let __UNIQUE_ID_x__292: u64 = count;
let __UNIQUE_ID_y__293: u64 = (core::mem::size_of::<[u8; 67]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__292, __UNIQUE_ID_y__293) == 0 {
unsafe { __compiletime_assert_294() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__292 < __UNIQUE_ID_y__293 { __UNIQUE_ID_x__292 } else { __UNIQUE_ID_y__293 };
count = __v0;
if copy_from_user(buf.0.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf.0[count as usize] = 0;
kstrtoul(buf.0.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtol_from_user")]
extern "C" fn kstrtol_from_user(s: *const u8, mut count: u64, base: u32, res: *mut i64) -> i32 {
let mut buf: __SlateAlign16<[u8; 67]> = __SlateAlign16([0u8; 67]);
let mut __v0: u64;
let __UNIQUE_ID_x__296: u64 = count;
let __UNIQUE_ID_y__297: u64 = (core::mem::size_of::<[u8; 67]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__296, __UNIQUE_ID_y__297) == 0 {
unsafe { __compiletime_assert_298() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__296 < __UNIQUE_ID_y__297 { __UNIQUE_ID_x__296 } else { __UNIQUE_ID_y__297 };
count = __v0;
if copy_from_user(buf.0.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf.0[count as usize] = 0;
kstrtol(buf.0.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtouint_from_user")]
extern "C" fn kstrtouint_from_user(s: *const u8, mut count: u64, base: u32, res: *mut u32) -> i32 {
let mut buf: __SlateAlign16<[u8; 35]> = __SlateAlign16([0u8; 35]);
let mut __v0: u64;
let __UNIQUE_ID_x__300: u64 = count;
let __UNIQUE_ID_y__301: u64 = (core::mem::size_of::<[u8; 35]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__300, __UNIQUE_ID_y__301) == 0 {
unsafe { __compiletime_assert_302() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__300 < __UNIQUE_ID_y__301 { __UNIQUE_ID_x__300 } else { __UNIQUE_ID_y__301 };
count = __v0;
if copy_from_user(buf.0.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf.0[count as usize] = 0;
kstrtouint(buf.0.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtoint_from_user")]
extern "C" fn kstrtoint_from_user(s: *const u8, mut count: u64, base: u32, res: *mut i32) -> i32 {
let mut buf: __SlateAlign16<[u8; 35]> = __SlateAlign16([0u8; 35]);
let mut __v0: u64;
let __UNIQUE_ID_x__304: u64 = count;
let __UNIQUE_ID_y__305: u64 = (core::mem::size_of::<[u8; 35]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__304, __UNIQUE_ID_y__305) == 0 {
unsafe { __compiletime_assert_306() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__304 < __UNIQUE_ID_y__305 { __UNIQUE_ID_x__304 } else { __UNIQUE_ID_y__305 };
count = __v0;
if copy_from_user(buf.0.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf.0[count as usize] = 0;
kstrtoint(buf.0.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtou16_from_user")]
extern "C" fn kstrtou16_from_user(s: *const u8, mut count: u64, base: u32, res: *mut u16) -> i32 {
let mut buf: __SlateAlign16<[u8; 19]> = __SlateAlign16([0u8; 19]);
let mut __v0: u64;
let __UNIQUE_ID_x__308: u64 = count;
let __UNIQUE_ID_y__309: u64 = (core::mem::size_of::<[u8; 19]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__308, __UNIQUE_ID_y__309) == 0 {
unsafe { __compiletime_assert_310() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__308 < __UNIQUE_ID_y__309 { __UNIQUE_ID_x__308 } else { __UNIQUE_ID_y__309 };
count = __v0;
if copy_from_user(buf.0.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf.0[count as usize] = 0;
kstrtou16(buf.0.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtos16_from_user")]
extern "C" fn kstrtos16_from_user(s: *const u8, mut count: u64, base: u32, res: *mut i16) -> i32 {
let mut buf: __SlateAlign16<[u8; 19]> = __SlateAlign16([0u8; 19]);
let mut __v0: u64;
let __UNIQUE_ID_x__312: u64 = count;
let __UNIQUE_ID_y__313: u64 = (core::mem::size_of::<[u8; 19]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__312, __UNIQUE_ID_y__313) == 0 {
unsafe { __compiletime_assert_314() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__312 < __UNIQUE_ID_y__313 { __UNIQUE_ID_x__312 } else { __UNIQUE_ID_y__313 };
count = __v0;
if copy_from_user(buf.0.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf.0[count as usize] = 0;
kstrtos16(buf.0.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtou8_from_user")]
extern "C" fn kstrtou8_from_user(s: *const u8, mut count: u64, base: u32, res: *mut u8) -> i32 {
let mut buf: [u8; 11] = [0u8; 11];
let mut __v0: u64;
let __UNIQUE_ID_x__316: u64 = count;
let __UNIQUE_ID_y__317: u64 = (core::mem::size_of::<[u8; 11]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__316, __UNIQUE_ID_y__317) == 0 {
unsafe { __compiletime_assert_318() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__316 < __UNIQUE_ID_y__317 { __UNIQUE_ID_x__316 } else { __UNIQUE_ID_y__317 };
count = __v0;
if copy_from_user(buf.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf[count as usize] = 0;
kstrtou8(buf.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtos8_from_user")]
extern "C" fn kstrtos8_from_user(s: *const u8, mut count: u64, base: u32, res: *mut i8) -> i32 {
let mut buf: [u8; 11] = [0u8; 11];
let mut __v0: u64;
let __UNIQUE_ID_x__320: u64 = count;
let __UNIQUE_ID_y__321: u64 = (core::mem::size_of::<[u8; 11]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__320, __UNIQUE_ID_y__321) == 0 {
unsafe { __compiletime_assert_322() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__320 < __UNIQUE_ID_y__321 { __UNIQUE_ID_x__320 } else { __UNIQUE_ID_y__321 };
count = __v0;
if copy_from_user(buf.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf[count as usize] = 0;
kstrtos8(buf.as_mut_ptr() as *const u8, base, res)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtobool_from_user")]
extern "C" fn kstrtobool_from_user(s: *const u8, mut count: u64, res: *mut bool) -> i32 {
// Longest string needed to differentiate, newline, terminator
let mut buf: [u8; 4] = [0u8; 4];
let mut __v0: u64;
let __UNIQUE_ID_x__280: u64 = count;
let __UNIQUE_ID_y__281: u64 = (core::mem::size_of::<[u8; 4]>() as u64).wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__280, __UNIQUE_ID_y__281) == 0 {
unsafe { __compiletime_assert_282() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__280 < __UNIQUE_ID_y__281 { __UNIQUE_ID_x__280 } else { __UNIQUE_ID_y__281 };
count = __v0;
if copy_from_user(buf.as_mut_ptr() as *mut (), s as *const (), count) != 0 {
return 14i32.wrapping_neg();
}
buf[count as usize] = 0;
kstrtobool(buf.as_mut_ptr() as *const u8, res)
}

#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

// check_sub_overflow() - Calculate subtraction with overflow checking
// @a: minuend; value to subtract from
// @b: subtrahend; value to subtract from @a
// @d: pointer to store difference
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted subtraction, regardless of whether
// wrap-around occurred.
fn check_object_size(ptr: *const (), n: u64, to_user: bool) {
}

fn copy_overflow(size: i32, count: u64) {
if IS_ENABLED__2!() != 0 {
unsafe { __copy_overflow(size, count) };
}
}

#[inline(always)]
fn check_copy_size(addr: *const (), bytes: u64, is_source: bool) -> bool {
let sz: i32 = (unsafe { __slate_intrinsic_2913c0303de876b3(addr, const { false as bool }, const { false as bool }, const { false as bool }) }) as i32;
if unlikely!(sz >= 0 && (sz as u64) < bytes) != 0 {
if !unsafe { __slate_intrinsic_f217c4720f04f6db(bytes) } {
copy_overflow(sz, bytes);
} else {
if is_source {
unsafe { __bad_copy_from() };
} else {
unsafe { __bad_copy_to() };
}
}
return false;
}
let mut __v0: i64;
let __ret_warn_on: i32 = (bytes > 2147483647u64) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("103: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 103b - ., 3; .popsection\n# {0}", const 103 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_check_copy_size_1, sym __slate_asm_string_check_copy_size_2, const 57 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("104: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 104b - ., 4; .popsection\n# {0}", const 104 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return false;
}
check_object_size(addr, bytes, is_source);
true
}

#[inline(always)]
fn copy_from_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(to as *const (), n, false) {
return n;
}
unsafe { _copy_from_user(to, from, n) }
}

#[inline(never)]
#[unsafe(no_mangle)]
extern "C" fn _parse_integer_fixup_radix(mut s: *const u8, base: *mut u32) -> *const u8 {
if (unsafe { *base }) == 0 {
if ((unsafe { *s.wrapping_offset(0 as isize) }) as i32) == 48 {
if (_tolower(unsafe { *s.wrapping_offset(1 as isize) }) as i32) == 120 && isxdigit!(unsafe { *s.wrapping_offset(2 as isize) }) {
unsafe {
*base = 16;
}
} else {
unsafe {
*base = 8;
}
}
} else {
unsafe {
*base = 10;
}
}
}
let __v0: bool = (unsafe { *base }) == 16 && ((unsafe { *s.wrapping_offset(0 as isize) }) as i32) == 48 && (_tolower(unsafe { *s.wrapping_offset(1 as isize) }) as i32) == 120;
if __v0 {
let __v1: *const u8 = s;
s = __v1.wrapping_offset(2 as isize);
}
s
}

/// _parse_integer_limit - Convert integer string representation to an integer
/// @s: Integer string representation
/// @base: Radix
/// @p: Where to store result
/// @max_chars: Maximum amount of characters to convert
/// @init: Initial value of the multiply-accumulate result
///
/// Convert non-negative integer string representation in explicitly given
/// radix to an integer. If overflow occurs, value at @p is set to ULLONG_MAX.
///
/// This function is the workhorse of other string conversion functions and it
/// is discouraged to use it explicitly. Consider kstrto*() family instead.
///
/// Return: Number of characters consumed, maybe ORed with overflow bit
#[inline(never)]
#[unsafe(no_mangle)]
extern "C" fn _parse_integer_limit(mut s: *const u8, base: u32, p: *mut u64, max_chars: u64, mut res: u64) -> u32 {
let mut rv: u32 = 0;
let mut overflow: u32 = 0;
while (rv as u64) < max_chars {
let c: u32 = (unsafe { *s }) as u32;
let lc: u32 = _tolower(c as u8) as u32;
let mut val: u32 = 0;
if 48 <= c && c <= 57 {
val = c.wrapping_sub(48);
} else {
if 97 <= lc && lc <= 102 {
val = lc.wrapping_sub(97).wrapping_add(10);
} else {
break;
}
}
if val >= base {
break;
}
// Check for overflow only if we are within range of
// it in the max base we support (16)
if unlikely__3!(res & 17293822569102704640u64) != 0 {
let __v1: bool = check_mul_overflow!(res, base, core::ptr::addr_of_mut!(res)) || check_add_overflow!(res, val, core::ptr::addr_of_mut!(res));
if __v1 {
res = !0;
overflow = 2147483648u32;
}
} else {
res = res.wrapping_mul(base as u64).wrapping_add(val as u64);
}
rv = rv.wrapping_add(1);
let __v4: *const u8 = s;
s = __v4.wrapping_offset(1 as isize);
}
unsafe {
*p = res;
}
rv | overflow
}

fn _kstrtoull(mut s: *const u8, mut base: u32, res: *mut u64) -> i32 {
let mut _res: u64 = 0;
let mut rv: u32 = 0;
s = _parse_integer_fixup_radix(s, core::ptr::addr_of_mut!(base));
rv = _parse_integer_limit(s, base, core::ptr::addr_of_mut!(_res), 2147483647u64, 0u64);
if rv & 2147483648u32 != 0 {
return 34i32.wrapping_neg();
}
if rv == 0 {
return 22i32.wrapping_neg();
}
let __v0: *const u8 = s;
s = __v0.wrapping_offset(rv as isize);
if ((unsafe { *s }) as i32) == 10 {
let __v2: *const u8 = s;
s = __v2.wrapping_offset(1 as isize);
}
if (unsafe { *s }) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
*res = _res;
}
0
}

/// kstrtoull - convert a string to an unsigned long long
/// @s: The start of the string. The string must be null-terminated, and may also
///  include a single newline before its terminating null. The first character
///  may also be a plus sign, but not a minus sign.
/// @base: The number base to use. The maximum supported base is 16. If base is
///  given as 0, then the base of the string is automatically detected with the
///  conventional semantics - If it begins with 0x the number will be parsed as a
///  hexadecimal (case insensitive), if it otherwise begins with 0, it will be
///  parsed as an octal number. Otherwise it will be parsed as a decimal.
/// @res: Where to write the result of the conversion on success.
///
/// Returns 0 on success, -ERANGE on overflow and -EINVAL on parsing error.
/// Preferred over simple_strtoull(). Return code must be checked.
#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtoull")]
extern "C" fn kstrtoull(mut s: *const u8, base: u32, res: *mut u64) -> i32 {
if ((unsafe { *s.wrapping_offset(0 as isize) }) as i32) == 43 {
let __v0: *const u8 = s;
s = __v0.wrapping_offset(1 as isize);
}
_kstrtoull(s, base, res)
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_kstrtoull: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtoll: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol__kstrtoul: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol__kstrtol: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtouint: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtoint: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtou16: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtos16: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtou8: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtos8: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtobool: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtoudec64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtodec64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtobool_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtoull_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtoll_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtoul_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtol_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtouint_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtoint_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtou16_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtos16_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtou8_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kstrtos8_from_user: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous", sym kstrtoull, sym kstrtoll, sym _kstrtoul, sym _kstrtol, sym kstrtouint, sym kstrtoint, sym kstrtou16, sym kstrtos16, sym kstrtou8, sym kstrtos8, sym kstrtobool, sym kstrtoudec64, sym kstrtodec64, sym kstrtobool_from_user, sym kstrtoull_from_user, sym kstrtoll_from_user, sym kstrtoul_from_user, sym kstrtol_from_user, sym kstrtouint_from_user, sym kstrtoint_from_user, sym kstrtou16_from_user, sym kstrtos16_from_user, sym kstrtou8_from_user, sym kstrtos8_from_user, options(att_syntax));

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtoull_267: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut u64) -> i32>, *mut ()>(Some(kstrtoull)) };

/// kstrtoll - convert a string to a long long
/// @s: The start of the string. The string must be null-terminated, and may also
///  include a single newline before its terminating null. The first character
///  may also be a plus sign or a minus sign.
/// @base: The number base to use. The maximum supported base is 16. If base is
///  given as 0, then the base of the string is automatically detected with the
///  conventional semantics - If it begins with 0x the number will be parsed as a
///  hexadecimal (case insensitive), if it otherwise begins with 0, it will be
///  parsed as an octal number. Otherwise it will be parsed as a decimal.
/// @res: Where to write the result of the conversion on success.
///
/// Returns 0 on success, -ERANGE on overflow and -EINVAL on parsing error.
/// Preferred over simple_strtoll(). Return code must be checked.
#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtoll")]
extern "C" fn kstrtoll(s: *const u8, base: u32, res: *mut i64) -> i32 {
let mut tmp: u64 = 0;
let mut rv: i32 = 0;
if ((unsafe { *s.wrapping_offset(0 as isize) }) as i32) == 45 {
rv = _kstrtoull(s.wrapping_offset(1 as isize), base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if (tmp.wrapping_neg() as i64) > 0 {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp.wrapping_neg() as i64;
}
} else {
rv = kstrtoull(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if (tmp as i64) < 0 {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp as i64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtoll_268: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut i64) -> i32>, *mut ()>(Some(kstrtoll)) };

/// Internal, do not use.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox._kstrtoul")]
extern "C" fn _kstrtoul(s: *const u8, base: u32, res: *mut u64) -> i32 {
let mut tmp: u64 = 0;
let rv: i32 = kstrtoull(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if tmp != tmp {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable__kstrtoul_269: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut u64) -> i32>, *mut ()>(Some(_kstrtoul)) };

/// Internal, do not use.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox._kstrtol")]
extern "C" fn _kstrtol(s: *const u8, base: u32, res: *mut i64) -> i32 {
let mut tmp: i64 = 0;
let rv: i32 = kstrtoll(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if tmp != tmp {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable__kstrtol_270: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut i64) -> i32>, *mut ()>(Some(_kstrtol)) };

/// kstrtouint - convert a string to an unsigned int
/// @s: The start of the string. The string must be null-terminated, and may also
///  include a single newline before its terminating null. The first character
///  may also be a plus sign, but not a minus sign.
/// @base: The number base to use. The maximum supported base is 16. If base is
///  given as 0, then the base of the string is automatically detected with the
///  conventional semantics - If it begins with 0x the number will be parsed as a
///  hexadecimal (case insensitive), if it otherwise begins with 0, it will be
///  parsed as an octal number. Otherwise it will be parsed as a decimal.
/// @res: Where to write the result of the conversion on success.
///
/// Returns 0 on success, -ERANGE on overflow and -EINVAL on parsing error.
/// Preferred over simple_strtoul(). Return code must be checked.
#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtouint")]
extern "C" fn kstrtouint(s: *const u8, base: u32, res: *mut u32) -> i32 {
let mut tmp: u64 = 0;
let rv: i32 = kstrtoull(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if tmp != ((tmp as u32) as u64) {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp as u32;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtouint_271: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut u32) -> i32>, *mut ()>(Some(kstrtouint)) };

/// kstrtoint - convert a string to an int
/// @s: The start of the string. The string must be null-terminated, and may also
///  include a single newline before its terminating null. The first character
///  may also be a plus sign or a minus sign.
/// @base: The number base to use. The maximum supported base is 16. If base is
///  given as 0, then the base of the string is automatically detected with the
///  conventional semantics - If it begins with 0x the number will be parsed as a
///  hexadecimal (case insensitive), if it otherwise begins with 0, it will be
///  parsed as an octal number. Otherwise it will be parsed as a decimal.
/// @res: Where to write the result of the conversion on success.
///
/// Returns 0 on success, -ERANGE on overflow and -EINVAL on parsing error.
/// Preferred over simple_strtol(). Return code must be checked.
#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtoint")]
extern "C" fn kstrtoint(s: *const u8, base: u32, res: *mut i32) -> i32 {
let mut tmp: i64 = 0;
let rv: i32 = kstrtoll(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if tmp != ((tmp as i32) as i64) {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp as i32;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtoint_272: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut i32) -> i32>, *mut ()>(Some(kstrtoint)) };

#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtou16")]
extern "C" fn kstrtou16(s: *const u8, base: u32, res: *mut u16) -> i32 {
let mut tmp: u64 = 0;
let rv: i32 = kstrtoull(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if tmp != (((tmp as u16) as i32) as u64) {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp as u16;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtou16_273: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut u16) -> i32>, *mut ()>(Some(kstrtou16)) };

#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtos16")]
extern "C" fn kstrtos16(s: *const u8, base: u32, res: *mut i16) -> i32 {
let mut tmp: i64 = 0;
let rv: i32 = kstrtoll(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if tmp != (((tmp as i16) as i32) as i64) {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp as i16;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtos16_274: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut i16) -> i32>, *mut ()>(Some(kstrtos16)) };

#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtou8")]
extern "C" fn kstrtou8(s: *const u8, base: u32, res: *mut u8) -> i32 {
let mut tmp: u64 = 0;
let rv: i32 = kstrtoull(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if tmp != (((tmp as u8) as i32) as u64) {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp as u8;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtou8_275: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut u8) -> i32>, *mut ()>(Some(kstrtou8)) };

#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtos8")]
extern "C" fn kstrtos8(s: *const u8, base: u32, res: *mut i8) -> i32 {
let mut tmp: i64 = 0;
let rv: i32 = kstrtoll(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if tmp != (((tmp as i8) as i32) as i64) {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp as i8;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtos8_276: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut i8) -> i32>, *mut ()>(Some(kstrtos8)) };

/// kstrtobool - convert common user inputs into boolean values
/// @s: input string
/// @res: result
///
/// This routine returns 0 iff the first character is one of 'EeYyTt1DdNnFf0',
/// or [oO][NnFf] for "on" and "off". Otherwise it will return -EINVAL.  Value
/// pointed to by res is updated upon finding a match.
#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtobool")]
extern "C" fn kstrtobool(s: *const u8, res: *mut bool) -> i32 {
if s.is_null() {
return 22i32.wrapping_neg();
}
match (unsafe { *s.wrapping_offset(0 as isize) }) as i32 {
49 | 69 | 84 | 89 | 101 | 116 | 121 => {
unsafe {
*res = true;
}
return 0;
}
48 | 68 | 70 | 78 | 100 | 102 | 110 => {
unsafe {
*res = false;
}
return 0;
}
79 | 111 => {
match (unsafe { *s.wrapping_offset(1 as isize) }) as i32 {
78 | 110 => {
unsafe {
*res = true;
}
return 0;
}
70 | 102 => {
unsafe {
*res = false;
}
return 0;
}
_ => {
}
}
}
_ => {
}
}
22i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtobool_277: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *mut bool) -> i32>, *mut ()>(Some(kstrtobool)) };

fn _kstrtoudec64(mut s: *const u8, scale: u32, res: *mut u64) -> i32 {
let mut rv_int: u32 = 0;
let mut rv_frac: u32 = 0;
let mut _res: u64 = 0;
rv_int = _parse_integer_limit(s, 10u32, core::ptr::addr_of_mut!(_res), 2147483647u64, 0u64);
if rv_int & 2147483648u32 != 0 {
return 34i32.wrapping_neg();
}
let __v0: *const u8 = s;
s = __v0.wrapping_offset(rv_int as isize);
if ((unsafe { *s }) as i32) == 46 {
let __v2: *const u8 = s;
s = __v2.wrapping_offset(1 as isize);
}
// skip decimal point
rv_frac = _parse_integer_limit(s, 10u32, core::ptr::addr_of_mut!(_res), scale as u64, _res);
if rv_frac & 2147483648u32 != 0 {
return 34i32.wrapping_neg();
}
let __v4: *const u8 = s;
s = __v4.wrapping_offset(rv_frac as isize);
// Check input beyond rv_int and rv_frac to cover cases like ".5" with
// scale 0, which is considered a valid input, being parsed as 0.
let __v6: bool = rv_int == 0 && rv_frac == 0 && isdigit((unsafe { *s }) as i32) == 0;
if __v6 {
return 22i32.wrapping_neg();
}
while isdigit((unsafe { *s }) as i32) != 0 {
let __v7: *const u8 = s;
s = __v7.wrapping_offset(1 as isize);
}
// truncate digits
if ((unsafe { *s }) as i32) == 10 {
let __v10: *const u8 = s;
s = __v10.wrapping_offset(1 as isize);
}
if (unsafe { *s }) != 0 {
return 22i32.wrapping_neg();
}
let __v12: bool = _res != 0 && (scale.wrapping_sub(rv_frac) > 19 || check_mul_overflow!(_res, unsafe { int_pow(10u64, scale.wrapping_sub(rv_frac)) }, core::ptr::addr_of_mut!(_res)));
if __v12 {
return 34i32.wrapping_neg();
}
// log10(2^64) = 19.26
unsafe {
*res = _res;
}
0
}

/// kstrtoudec64() - Convert a string to an unsigned 64-bit scaled decimal value.
/// @s: The start of the string. The string must be null-terminated, and may also
///  include a single newline before its terminating null. The first character
///  may also be a plus sign, but not a minus sign.
/// @scale: The number of digits to the right of the decimal point.
/// @res: Where to write the result of the conversion on success.
///
/// For example, a scale of 3 with input "123.45" results in 123450. Note that
/// trailing zeros in the fractional part input to match the scale are not
/// required. Also, digits beyond the specified scale are ignored.
///
/// Return: 0 on success, -ERANGE on overflow and -EINVAL on parsing error.
#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtoudec64")]
extern "C" fn kstrtoudec64(mut s: *const u8, scale: u32, res: *mut u64) -> i32 {
if ((unsafe { *s.wrapping_offset(0 as isize) }) as i32) == 43 {
let __v0: *const u8 = s;
s = __v0.wrapping_offset(1 as isize);
}
_kstrtoudec64(s, scale, res)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtoudec64_278: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut u64) -> i32>, *mut ()>(Some(kstrtoudec64)) };

/// kstrtodec64() - Convert a string to a signed 64-bit scaled decimal value.
/// @s: The start of the string. The string must be null-terminated, and may also
///  include a single newline before its terminating null. The first character
///  may also be a plus sign or a minus sign.
/// @scale: The number of digits to the right of the decimal point.
/// @res: Where to write the result of the conversion on success.
///
/// For example, a scale of 4 with input "-3.141592" results in -31415. Note
/// that digits beyond the specified scale are ignored. Also, trailing zeros in
/// the fractional part input to match the scale are not required.
///
/// Return: 0 on success, -ERANGE on overflow and -EINVAL on parsing error.
#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.kstrtox.kstrtodec64")]
extern "C" fn kstrtodec64(s: *const u8, scale: u32, res: *mut i64) -> i32 {
let mut tmp: u64 = 0;
let mut rv: i32 = 0;
if ((unsafe { *s.wrapping_offset(0 as isize) }) as i32) == 45 {
rv = _kstrtoudec64(s.wrapping_offset(1 as isize), scale, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if (tmp.wrapping_neg() as i64) > 0 {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp.wrapping_neg() as i64;
}
} else {
rv = kstrtoudec64(s, scale, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if (tmp as i64) < 0 {
return 34i32.wrapping_neg();
}
unsafe {
*res = tmp as i64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtodec64_279: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u32, *mut i64) -> i32>, *mut ()>(Some(kstrtodec64)) };

// Since "base" would be a nonsense argument, this open-codes the
// _from_user helper instead of using the helper macro below.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtobool_from_user_283: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, *mut bool) -> i32>, *mut ()>(Some(kstrtobool_from_user)) };

// sign, base 2 representation, newline, terminator
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtoull_from_user_287: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut u64) -> i32>, *mut ()>(Some(kstrtoull_from_user)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtoll_from_user_291: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut i64) -> i32>, *mut ()>(Some(kstrtoll_from_user)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtoul_from_user_295: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut u64) -> i32>, *mut ()>(Some(kstrtoul_from_user)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtol_from_user_299: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut i64) -> i32>, *mut ()>(Some(kstrtol_from_user)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtouint_from_user_303: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut u32) -> i32>, *mut ()>(Some(kstrtouint_from_user)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtoint_from_user_307: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut i32) -> i32>, *mut ()>(Some(kstrtoint_from_user)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtou16_from_user_311: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut u16) -> i32>, *mut ()>(Some(kstrtou16_from_user)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtos16_from_user_315: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut i16) -> i32>, *mut ()>(Some(kstrtos16_from_user)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtou8_from_user_319: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut u8) -> i32>, *mut ()>(Some(kstrtou8_from_user)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kstrtos8_from_user_323: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, u32, *mut i8) -> i32>, *mut ()>(Some(kstrtos8_from_user)) };

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";
