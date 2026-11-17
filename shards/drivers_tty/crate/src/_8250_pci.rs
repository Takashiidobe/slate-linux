
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
//! Probe module for 8250/16550-type PCI serial ports.
//!
//! Based on drivers/char/serial.c, by Linus Torvalds, Theodore Ts'o.
//!
//! Copyright (C) 2001 Russell King, All Rights Reserved.
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
//! __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//!
//! @TYPE: The type of each flexible array element
//! @NAME: The name of the flexible array member
//!
//! In order to have a flexible array member in a union or alone in a
//! struct, it needs to be wrapped in an anonymous struct with at least 1
//! named member, but that member can be empty.
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
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
//! Generate a mask for the specified type @t. Additional checks are made to
//! guarantee the value returned fits in that type, relying on
//! -Wshift-count-overflow compiler check to detect incompatible arguments.
//! For example, all these create build errors or warnings:
//!
//! - GENMASK(15, 20): wrong argument order
//! - GENMASK(72, 15): doesn't fit unsigned long
//! - GENMASK_U32(33, 15): doesn't fit in a u32
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
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
//! This counts to 15. Any more, it will return 16th argument.
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
//! Defines x86 CPU feature bits
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! writeable
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! restart by calling sys_restart_syscall
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
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
//! 64bit FS base
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
//! strscpy - Copy a C-string into a sized buffer
//! @dst: Where to copy the string to
//! @src: Where to copy the string from
//! @...: Size of destination buffer (optional)
//!
//! Copy the source string @src, or as much of it as fits, into the
//! destination @dst buffer. The behavior is undefined if the string
//! buffers overlap. The destination @dst buffer is always NUL terminated,
//! unless it's zero-sized.
//!
//! The size argument @... is only required when @dst is not an array, or
//! when the copy needs to be smaller than sizeof(@dst).
//!
//! Preferred to strncpy() since it always returns a valid string, and
//! doesn't unnecessarily force the tail of the destination buffer to be
//! zero padded. If padding is desired please use strscpy_pad().
//!
//! Returns the number of characters copied in @dst (not including the
//! trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
//! truncated.
//! __sign_use for integer expressions:
//!   bit #0 set if ok for unsigned comparisons
//!   bit #1 set if ok for signed comparisons
//!
//! In particular, statically non-negative signed integer expressions
//! are ok for both.
//!
//! NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
//! in expressions, and are accepted for signed conversions for now.
//! This is debatable.
//!
//! Note that 'x' is the original expression, and 'ux' is the unique variable
//! that contains the value.
//!
//! We use 'ux' for pure type checking, and 'x' for when we need to look at the
//! value (but without evaluating it for side effects!
//! Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
//!
//! Pointers end up being checked by the normal C type rules at the actual
//! comparison, and these expressions only need to be careful to not cause
//! warnings for pointer use.
//! PTR_IF - evaluate to @ptr if @cond is true, or to NULL otherwise.
//! @cond: A conditional, usually in a form of IS_ENABLED(CONFIG_FOO)
//! @ptr: A pointer to assign if @cond is true.
//!
//! PTR_IF(IS_ENABLED(CONFIG_FOO), ptr) evaluates to @ptr if CONFIG_FOO is set
//! to 'y' or 'm', or to NULL otherwise. The @ptr argument must be a pointer.
//!
//! The macro can be very useful to help compiler dropping dead code.
//!
//! For instance, consider the following::
//!
//!     #ifdef CONFIG_FOO_SUSPEND
//!     static int foo_suspend(struct device *dev)
//!     {
//!        ...
//!     }
//!     #endif
//!
//!     static struct pm_ops foo_ops = {
//!     #ifdef CONFIG_FOO_SUSPEND
//!         .suspend = foo_suspend,
//!     #endif
//!     };
//!
//! While this works, the foo_suspend() macro is compiled conditionally,
//! only when CONFIG_FOO_SUSPEND is set. This is problematic, as there could
//! be a build bug in this function, we wouldn't have a way to know unless
//! the configuration option is set.
//!
//! An alternative is to declare foo_suspend() always, but mark it
//! as __maybe_unused. This works, but the __maybe_unused attribute
//! is required to instruct the compiler that the function may not
//! be referenced anywhere, and is safe to remove without making
//! a fuss about it. This makes the programmer responsible for tagging
//! the functions that can be garbage-collected.
//!
//! With the macro it is possible to write the following::
//!
//!     static int foo_suspend(struct device *dev)
//!     {
//!        ...
//!     }
//!
//!     static struct pm_ops foo_ops = {
//!         .suspend = PTR_IF(IS_ENABLED(CONFIG_FOO_SUSPEND), foo_suspend),
//!     };
//!
//! The foo_suspend() function will now be automatically dropped by the
//! compiler, and it does not require any specific attribute.
//! used to decode ioctl numbers..
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
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! signal pending
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
//! SCHED_ISO: reserved but not implemented yet
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
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
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! wait_queue_entry::flags
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
//! __MAX_NR_ZONES
//! 'empty' nodes are nodes that are known not to be inserted in an rbtree
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
//! time_after - returns true if the time a is after time b.
//! @a: first comparable as unsigned long
//! @b: second comparable as unsigned long
//!
//! Do this with "<0" and ">=0" to only test the sign of the result. A
//! good compiler would generate better code (and a really good compiler
//! wouldn't care). Gcc is currently neither.
//!
//! Return: %true is time a is after time b, otherwise %false.
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
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
//! Helper macro to avoid gfp flags if they are the default one
//! Desired design of maximum size and alignment (see RFC2553)
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
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
//! This is the smallest unsigned integer type that can fit a value
//! (up to 'long long')
//! CAP_BPF allows the following BPF operations:
//! - Creating all types of BPF maps
//! - Advanced verifier features
//!   - Indirect variable access
//!   - Bounded loops
//!   - BPF to BPF function calls
//!   - Scalar precision tracking
//!   - Larger complexity limits
//!   - Dead code elimination
//!   - And potentially other features
//! - Loading BPF Type Format (BTF) data
//! - Retrieve xlated and JITed code of BPF programs
//! - Use bpf_spin_lock() helper
//!
//! CAP_PERFMON relaxes the verifier checks further:
//! - BPF progs can use of pointer-to-integer conversions
//! - speculation attack hardening measures are bypassed
//! - bpf_probe_read to read arbitrary kernel memory is allowed
//! - bpf_trace_printk to print kernel memory is allowed
//!
//! CAP_SYS_ADMIN is required to use bpf_probe_write_user.
//!
//! CAP_SYS_ADMIN is required to iterate system wide loaded
//! programs, maps, links, BTFs and convert their IDs to file descriptors.
//!
//! CAP_PERFMON and CAP_BPF are required to load tracing programs.
//! CAP_NET_ADMIN and CAP_BPF are required to load networking programs.
//! return the first ptr or next element in an RCU protected list
//! This hash multiplies the input by a large odd number and takes the
//! high bits.  Since multiplication propagates changes to the most
//! significant end only, it is essential that the high bits of the
//! product be used for the hash value.
//!
//! Chuck Lever verified the effectiveness of this technique:
//! http://www.citi.umich.edu/techreports/reports/citi-tr-00-1.pdf
//!
//! Although a random odd number will do, it turns out that the golden
//! ratio phi = (sqrt(5)-1)/2, or its negative, has particularly nice
//! properties.  (See Knuth vol 3, section 6.4, exercise 9.)
//!
//! These are the negative, (1 - phi) = phi**2 = (3 - sqrt(5))/2,
//! which is very slightly easier to multiply by and makes no
//! difference to the hash distribution.
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! The structure layout of user_xstateregs, used for exporting the
//! extended register state through ptrace and core-dump (NT_X86_XSTATE note)
//! interfaces will be same as the memory layout of xsave used by the processor
//! (except for the bytes 464..511, which can be used by the software) and hence
//! the size of this structure varies depending on the features supported by the
//! processor and OS. The size of the structure that users need to use can be
//! obtained by doing:
//!     cpuid_count(0xd, 0, &eax, &ptrace_xstateregs_struct_size, &ecx, &edx);
//! i.e., cpuid.(eax=0xd,ecx=0).ebx will be the size that user (debuggers, etc.)
//! need to use.
//!
//! For now, only the first 8 bytes of the software usable bytes[464..471] will
//! be used and will be set to OS enabled xstate mask (which is same as the
//! 64bit mask returned by the xgetbv's xCR0).  Users (analyzing core dump
//! remotely, etc.) can use this mask as well as the mask saved in the
//! xstate_hdr bytes and interpret what states the processor/OS supports
//! and what states are in modified/initialized conditions for the
//! particular process/thread.
//!
//! Also when the user modifies certain state FP/SSE/etc through the
//! ptrace interface, they must ensure that the header.xfeatures
//! bytes[512..519] of the memory layout are updated correspondingly.
//! i.e., for example when FP state is modified to a non-init state,
//! header.xfeatures's bit 0 must be set to '1', when SSE is modified to
//! non-init state, header.xfeatures's bit 1 must to be set to '1', etc.
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generic info of form tag = "info"
//! Generate SD flag indexes
//! Number of standard BARs
//! 2x 16550
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! Out: Interrupt Enable Register
//! FIFO and shifting register empty
//! Sunix UART
//! Use successive BARs (PCI base address registers),
//! else use offset into some specified BAR
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
// The second argument is optional (default 0), so we use a variadic macro
// to make the shorthand.
//
// Beware: Do not apply this to functions which may return
// ERR_PTRs. Also, it is probably unwise to apply it to functions
// returning extra information in the low bits (but in that case the
// compiler should see some alignment anyway, when the return value is
// massaged by 'flags = ptr & 3; ptr &= ~3;').
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-assume_005faligned-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#assume-aligned
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-cleanup-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#cleanup
// Note the long name.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
// Don't. Just don't. See commit 771c035372a0 ("deprecate the '__deprecated'
// attribute warnings entirely and for good") for more information.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-deprecated-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-deprecated-type-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-deprecated-variable-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Enumerator-Attributes.html#index-deprecated-enumerator-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#deprecated
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-format-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#format
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
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
// &a[0] degrades to a pointer: a different type from an array
// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
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
// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
// bool and also pointer types.
// Useful shorthand for "is this condition known at compile-time?"
//
// Note that the condition may involve non-constant values,
// but the compiler may know enough about the details of the
// values to determine that the condition is statically true.
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// ASSERT_EXCLUSIVE_ACCESS - assert no concurrent accesses to @var
//
// Assert that there are no concurrent accesses to @var (no readers nor
// writers). This assertion can be used to specify properties of concurrent
// code, where violation cannot be detected as a normal data race.
//
// For example, where exclusive access is expected after determining no other
// users of an object are left, but the object is not actually freed. We can
// check that this property actually holds as follows:
//
// .. code-block:: c
//
// 	if (refcount_dec_and_test(&obj->refcnt)) {
// 		ASSERT_EXCLUSIVE_ACCESS(*obj);
// 		do_some_cleanup(obj);
// 		release_for_reuse(obj);
// 	}
//
// Note:
//
// 1. ASSERT_EXCLUSIVE_ACCESS_SCOPED(), if applicable, performs more thorough
//    checking if a clear scope where no concurrent accesses are expected exists.
//
// 2. For cases where the object is freed, `KASAN <kasan.html>`_ is a better
//    fit to detect use-after-free bugs.
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Some subsystems have their own custom printk that applies a va_format to a
// generic format, for example, to include a device number or other metadata
// alongside the format supplied by the caller.
//
// In order to store these in the way they would be emitted by the printk
// infrastructure, the subsystem provides us with the start, fixed string, and
// any subsequent text in the format string.
//
// We take a variable argument list as pr_fmt/dev_fmt/etc are sometimes passed
// as multiple arguments (eg: `"%s: ", "blah"`), and we must only take the
// first one.
//
// subsys_fmt_prefix must be known at compile time, or compilation will fail
// (since this is a mistake). If fmt or level is not known at compile time, no
// index entry will be made (since this can legitimately happen).
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
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
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
// bits_per - calculate the number of bits required for the argument
// @n: parameter
//
// This is constant-capable and can be used for compile time
// initializations, e.g bitfields.
//
// The first few values calculated by this routine:
// bf(0) = 1
// bf(1) = 1
// bf(2) = 2
// bf(3) = 2
// bf(4) = 3
// ... and so on.
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Not a typewriter
// Text file busy
// No space left on device
// Math result not representable
// Invalid system call number
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
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
// Physical address zone modifiers (see linux/mmzone.h - low four bits)
//
// Do not put any conditional on these. If necessary modify the definitions
// without the underscores and use them consistently. The definitions here may
// be used in bit comparisons.
// ZONE_MOVABLE allowed
// DOC: Page mobility and placement hints
//
// Page mobility and placement hints
//
// These flags provide hints about how mobile the page is. Pages with similar
// mobility are placed within the same pageblocks to minimise problems due
// to external fragmentation.
//
// %__GFP_MOVABLE (also a zone modifier) indicates that the page can be
// moved by page migration during memory compaction or can be reclaimed.
//
// %__GFP_RECLAIMABLE is used for slab allocations that specify
// SLAB_RECLAIM_ACCOUNT and whose pages can be freed via shrinkers.
//
// %__GFP_WRITE indicates the caller intends to dirty the page. Where possible,
// these pages will be spread between local zones to avoid all the dirty
// pages being in one zone (fair zone allocation policy).
//
// %__GFP_HARDWALL enforces the cpuset memory allocation policy.
//
// %__GFP_THISNODE forces the allocation to be satisfied from the requested
// node with no fallbacks or placement policy enforcements.
//
// %__GFP_ACCOUNT causes the allocation to be accounted to the active
// cgroup context.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// disable sas during sighandling
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
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
// Same as rb_first(), but O(1)
// time_before - returns true if the time a is before time b.
// @a: first comparable as unsigned long
// @b: second comparable as unsigned long
//
// Return: %true is time a is before time b, otherwise %false.
// Change timeval to jiffies, trying to avoid the
// most obvious overflows..
//
// And some not so obvious.
//
// Note that we don't want to return LONG_MAX, because
// for various timeout reasons we often end up having
// to wait "jiffies+1" in order to guarantee that we wait
// at _least_ "jiffies" - so "jiffies+1" had better still
// be positive.
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
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
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
// Implementation specific max size
// list_entry_rcu - get the struct for this entry
// @ptr:        the &struct list_head pointer.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// list_next_or_null_rcu - get the next element from a list
// @head:	the head for the list.
// @ptr:        the list head to take the next element from.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// Note that if the ptr is at the end of the list, NULL is returned.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// return the first or the next element in an RCU protected hlist
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
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
// This is used for both get_user() and __get_user() to expand to
// the proper special function call that has odd calling conventions
// due to returning both a value and an error, and that depends on
// the size of the pointer passed in.
//
// Careful: we have to cast the result to the type of the pointer
// for sign reasons.
//
// The use of _ASM_DX as the register specifier is a bit of a
// simplification, as gcc only cares about it as the starting point
// and not size: for a 64-bit value it will use %ecx:%edx on 32 bits
// (%ecx being the next register in gcc's x86 register sequence), and
// %rdx on 64 bits.
//
// Clang/LLVM cares about the size of the register, but still wants
// the base register for something that ends up being a pair.
// get_user - Get a simple variable from user space.
// @x:   Variable to store result.
// @ptr: Source address, in user space.
//
// Context: User context only. This function may sleep if pagefaults are
//          enabled.
//
// This macro copies a single simple variable from user space to kernel
// space.  It supports simple types like char and int, but not larger
// data types like structures or arrays.
//
// @ptr must have pointer-to-simple-variable type, and the result of
// dereferencing @ptr must be assignable to @x without a cast.
//
// Return: zero on success, or -EFAULT on error.
// On error, the variable @x is set to zero.
// The bottom two bits of the slot determine how the remaining bits in the
// slot are interpreted:
//
// 00 - data pointer
// 10 - internal entry
// x1 - value entry
//
// The internal entry may be a pointer to the next level in the tree, a
// sibling entry, or an indicator that the entry in this slot has been moved
// to another location in the tree and the lookup should be restarted.  While
// NULL fits the 'data pointer' pattern, it means that there is no entry in
// the tree for this index (no matter what level of the tree it is found at).
// This means that storing a NULL entry in the tree is the same as deleting
// the entry from the tree.
// direct disk access hint
// Reserved kernel ranges [-100], [-10000, -40000].
// Special value for dirfd used to
// indicate openat should use the
// current working directory.
// Reserved for per-syscall flags	0xff.
// Do not follow symbolic
// links.
// Allow empty relative
// pathname to operate on dirfd
// directly.
// The 10 bits between the priority class and the priority level are used to
// optionally define I/O hints for any combination of I/O priority class and
// level. Depending on the kernel configuration, I/O scheduler being used and
// the target I/O device being used, hints can influence how I/Os are processed
// without affecting the I/O scheduling ordering defined by the I/O priority
// class and level.
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// See the comment in asm-generic/io.h about ioremap_np().
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// 2x 16550, half of 4 port
// 2x 16550, half of 4 port
// 4x 16550A, half of 8 port
// 4x 16550A, half of 8 port
// 2x 16650
// 2x 16650, half of 4 port
// 2x 16650, half of 4 port
// 1x 16550
// 1x 16650
// ADDI-DATA GmbH communication cards <info@addi-data.com>
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// A dma_addr_t can hold any valid DMA or bus address for the platform.  It can
// be given to a device to use as a DMA source or target.  It is specific to a
// given device and there may be a translation between the CPU physical address
// space and the bus address space.
//
// DMA_MAPPING_ERROR is the magic error code if a mapping failed.  It should not
// be used directly in drivers, but checked for using dma_mapping_error()
// instead.
// Use the high bit to mark if we used swiotlb for one or more ranges.
// Enable Transmitter holding register int.
// Out: Line Control Register
// Out: Modem Control Register
// Divide clock by 4 (TI16C752, EFR[4]=1)
// Enable loopback test mode
// Out2 complement
// Out1 complement
// RTS complement
// DTR complement
// In:  Line Status Register
// Data Carrier Detect
// Ring Indicator
// Data Set Ready
// Clear to Send
// I/O: Scratch Register
// Max divisor value
// Index Control Register
// The 16950 ICR registers
// Additional Control Register
// Clock Prescalar Register
// Times Clock Register
// Clock Select Register
// ICR Read enable
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__2 {
() => {
((core::mem::size_of::<[i16; 7]>() as u64) / (core::mem::size_of::<i16>() as u64)).wrapping_add(__must_be_array__22!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__6 {
() => {
((core::mem::size_of::<[pciserial_board; 114]>() as u64) / (core::mem::size_of::<pciserial_board>() as u64)).wrapping_add(__must_be_array__17!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__12 {
() => {
((core::mem::size_of::<[[u8; 2]; 134]>() as u64) / (core::mem::size_of::<[u8; 2]>() as u64)).wrapping_add(__must_be_array__12!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__22 {
() => {
((core::mem::size_of::<[timedia_struct; 4]>() as u64) / (core::mem::size_of::<timedia_struct>() as u64)).wrapping_add(__must_be_array__24!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__36 {
() => {
((core::mem::size_of::<[i16; 7]>() as u64) / (core::mem::size_of::<i16>() as u64)).wrapping_add(__must_be_array__26!() as u64)
};
}

macro_rules! BIT_ULL {
($nr:expr) => {
ULL!() << $nr
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! FL_GET_BASE {
($x:expr) => {
($x) & 7
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

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED {
() => {
__is_defined!()
};
}

/// IS_ERR_VALUE - Detect an error pointer.
/// @x: The pointer to check.
///
/// Like IS_ERR(), but does not generate a compiler warning if result is unused.
macro_rules! IS_ERR_VALUE {
($x:expr) => {
unlikely!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! ULL {
() => {
_ULL!()
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _BITUL {
($x:expr) => {
_UL!() << $x
};
}

macro_rules! _ULL {
() => {
_AC!()
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

macro_rules! ____or {
() => {
1i32
};
}

/// __acquire() - function to acquire context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x exclusively.
macro_rules! __acquire {
() => {
__acquire_ctx_lock!();
};
}

macro_rules! __acquire_ctx_lock {
() => {
while false {
}
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

macro_rules! __must_be_array__12 {
() => {
core::mem::size_of::<__SlateRecord_8250_pci_c_1196>() as i32
};
}

macro_rules! __must_be_array__17 {
() => {
core::mem::size_of::<__SlateRecord_8250_pci_c_4171>() as i32
};
}

macro_rules! __must_be_array__22 {
() => {
core::mem::size_of::<__SlateRecord_8250_pci_c_965>() as i32
};
}

macro_rules! __must_be_array__24 {
() => {
core::mem::size_of::<__SlateRecord_8250_pci_c_658>() as i32
};
}

macro_rules! __must_be_array__26 {
() => {
core::mem::size_of::<__SlateRecord_8250_pci_c_985>() as i32
};
}

macro_rules! __must_be_array__30 {
() => {
core::mem::size_of::<__SlateRecord_8250_pci_c_4031_4>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release {
() => {
__release_ctx_lock!();
};
}

macro_rules! __release_ctx_lock {
() => {
while false {
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<*mut task_struct>() as *const ();
if true {
break;
}
}
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
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

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__2 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size__3 {
($count:expr) => {
size_mul($count, (core::mem::size_of::<i32>() as u64).wrapping_add(__must_be_array__30!() as u64))
};
}

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! offsetof__2 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, port) as u64
};
}

macro_rules! pci_resource_end {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).end }
};
}

macro_rules! pci_resource_flags {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).flags }
};
}

macro_rules! pci_resource_flags__2 {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n__2!($dev, $bar)).flags }
};
}

macro_rules! pci_resource_len {
($dev:expr, $bar:expr) => {
{
let __v: u64 = if pci_resource_end!($dev, $bar) != 0 {
resource_size(pci_resource_n!($dev, $bar) as *const resource)
} else {
0
};
__v
}
};
}

/// These helpers provide future and backwards compatibility
/// for accessing popular PCI BAR info
macro_rules! pci_resource_n {
($dev:expr, $bar:expr) => {
unsafe { (*$dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(($bar) as isize)
};
}

/// These helpers provide future and backwards compatibility
/// for accessing popular PCI BAR info
macro_rules! pci_resource_n__2 {
($dev:expr, $bar:expr) => {
unsafe { (*$dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset((($bar) as i32) as isize)
};
}

macro_rules! pci_resource_start {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).start }
};
}

macro_rules! raw_spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
$flags = unsafe { _raw_spin_lock_irqsave($lock) };
if true {
break;
}
}
};
}

macro_rules! raw_spin_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe { _raw_spin_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

/// Compatibility cruft
macro_rules! release_region__2 {
($start:expr, $n:expr) => {
unsafe { __release_region(core::ptr::addr_of_mut!(ioport_resource), ($start) as u64, ($n) as u64) };
};
}

/// Compatibility cruft
macro_rules! release_region {
($start:expr, $n:expr) => {
unsafe { __release_region(core::ptr::addr_of_mut!(ioport_resource), $start, ($n) as u64) };
};
}

/// Convenience shorthand with allocation
macro_rules! request_region__2 {
($start:expr, $n:expr, $name:expr) => {
unsafe { __request_region(core::ptr::addr_of_mut!(ioport_resource), ($start) as u64, ($n) as u64, ($name) as *const u8, 0i32) }
};
}

macro_rules! spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave!(spinlock_check($lock), $flags);
__release!();
__acquire!();
if true {
break;
}
}
};
}

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size__2 {
($count:expr) => {
size_add(32u64, flex_array_size__3!($count))
};
}

/// struct_size_t() - Calculate size of structure with trailing flexible array
/// @type: structure type name.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure @type followed by an
/// array of @count number of @member elements. Prefer using struct_size()
/// when possible instead, to keep calculations associated with a specific
/// instance variable of type @type.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size_t__2 {
($count:expr) => {
struct_size__2!($count)
};
}

macro_rules! to_pci_dev {
($n:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!(pci_dev) as isize)) }) as *mut ()) as *mut pci_dev;
__v
}
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck {
($type:ty) => {
let mut __v: i32;
let mut __dummy: $type = 0;
let mut __dummy2: $type = 0;
core::ptr::addr_of_mut!(__dummy) == core::ptr::addr_of_mut!(__dummy2);
__v = 1;
};
}

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut ioport_resource: resource;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_29();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn up(arg0: *mut semaphore);
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __request_region(arg0: *mut resource, arg1: u64, arg2: u64, arg3: *const u8, arg4: i32) -> *mut resource;
    fn __release_region(arg0: *mut resource, arg1: u64, arg2: u64);
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn devm_kmalloc(arg0: *mut device, arg1: u64, arg2: u32) -> *mut ();
    fn ioremap(arg0: u64, arg1: u64) -> *mut ();
    fn iounmap(arg0: *mut ());
    fn ioread8(arg0: *const ()) -> u32;
    fn ioread32(arg0: *const ()) -> u32;
    fn pci_iomap(arg0: *mut pci_dev, arg1: i32, arg2: u64) -> *mut ();
    fn pci_iounmap(arg0: *mut pci_dev, arg1: *mut ());
    fn pcibios_resource_to_bus(arg0: *mut pci_bus, arg1: *mut pci_bus_region, arg2: *mut resource);
    fn pci_read_config_byte(arg0: *const pci_dev, arg1: i32, arg2: *mut u8) -> i32;
    fn pci_read_config_word(arg0: *const pci_dev, arg1: i32, arg2: *mut u16) -> i32;
    fn pci_read_config_dword(arg0: *const pci_dev, arg1: i32, arg2: *mut u32) -> i32;
    fn pci_write_config_byte(arg0: *const pci_dev, arg1: i32, arg2: u8) -> i32;
    fn pci_write_config_dword(arg0: *const pci_dev, arg1: i32, arg2: u32) -> i32;
    fn pci_enable_device(arg0: *mut pci_dev) -> i32;
    fn pcim_enable_device(arg0: *mut pci_dev) -> i32;
    fn pci_disable_device(arg0: *mut pci_dev);
    fn pci_set_master(arg0: *mut pci_dev);
    fn pci_save_state(arg0: *mut pci_dev) -> i32;
    fn pci_restore_state(arg0: *mut pci_dev);
    fn __pci_register_driver(arg0: *mut pci_driver, arg1: *mut module, arg2: *const u8) -> i32;
    fn pci_unregister_driver(arg0: *mut pci_driver);
    fn pci_match_id(arg0: *const pci_device_id, arg1: *mut pci_dev) -> *const pci_device_id;
    fn pci_alloc_irq_vectors(arg0: *mut pci_dev, arg1: u32, arg2: u32, arg3: u32) -> i32;
    fn pci_irq_vector(arg0: *mut pci_dev, arg1: u32) -> i32;
    fn pcim_iomap(arg0: *mut pci_dev, arg1: i32, arg2: u64) -> *mut ();
    fn pci_ioremap_bar(arg0: *mut pci_dev, arg1: i32) -> *mut ();
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn msleep(arg0: u32);
    fn serial8250_register_8250_port(arg0: *const uart_8250_port) -> i32;
    fn serial8250_unregister_port(arg0: i32);
    fn serial8250_suspend_port(arg0: i32);
    fn serial8250_resume_port(arg0: i32);
    fn serial8250_do_set_mctrl(arg0: *mut uart_port, arg1: u32);
    fn serial8250_do_set_divisor(arg0: *mut uart_port, arg1: u32, arg2: u32);
    fn serial8250_clear_and_reinit_fifos(arg0: *mut uart_8250_port);
    fn serial8250_pci_setup_port(arg0: *mut pci_dev, arg1: *mut uart_8250_port, arg2: u8, arg3: u32, arg4: i32, arg5: *mut ()) -> i32;
    fn serial_8250_warn_need_ioport(arg0: *mut pci_dev) -> i32;
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

// generic data direction definitions
// The type of an index into the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_190 {
    counter: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_197 {
    counter: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_head {
    next: *mut list_head,
    prev: *mut list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_head {
    first: *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_node {
    next: *mut hlist_node,
    pprev: *mut *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct callback_head {
    next: *mut callback_head,
    func: Option<unsafe extern "C" fn(*mut callback_head)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcuwait {
    task: *mut task_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
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
#[inline(always)]
fn size_mul(factor1: u64, factor2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_mul_overflow!(factor1, factor2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

#[inline(always)]
fn size_add(addend1: u64, addend2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_add_overflow!(addend1, addend2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

// __set_flex_counter() - Set the counter associated with the given flexible
//                        array member that has been annoated by __counted_by().
// @FAM: Instance of flexible array member within a given struct.
// @COUNT: Value to store to the __counted_by annotated @FAM_PTR's counter.
//
// This is a no-op if no annotation exists. Count needs to be checked with
// overflows_flex_counter_type() before using this function.
#[repr(C)]
#[derive(Clone, Copy)]
struct qspinlock {
    __slate_anon_0: __SlateRecord_qspinlock_types_h_44,
}


// Bitfields in the atomic value:
//
// When NR_CPUS < 16K
//  0- 7: locked byte
//  8-15: pending byte
// 16-17: tail index
// 18-31: tail cpu (+1)
//
// When NR_CPUS >= 16K
//  0- 7: locked byte
//     8: pending
//  9-10: tail index
// 11-31: tail cpu (+1)
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qspinlock_types_h_44 {
    val: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qspinlock_types_h_44_2,
    __slate_anon_2: __SlateRecord_qspinlock_types_h_44_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_2 {
    locked: u8,
    pending: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_3 {
    locked_pending: u16,
    tail: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qrwlock {
    __slate_anon_0: __SlateRecord_qrwlock_types_h_27,
    wait_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qrwlock_types_h_27 {
    cnts: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qrwlock_types_h_27_2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qrwlock_types_h_27_2 {
    wlocked: u8,
    __lstate: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lock_class_key {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockdep_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ratelimit_state {
    lock: raw_spinlock,
    interval: i32,
    burst: i32,
    rs_n_left: __SlateRecord_types_h_190,
    missed: __SlateRecord_types_h_190,
    flags: u32,
    begin: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


fn div_s64_rem(dividend: i64, divisor: i32, remainder: &mut i32) -> i64 {
unsafe {
*(remainder as *mut i32) = (dividend % (divisor as i64)) as i32;
}
dividend / (divisor as i64)
}

// DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
// @ll: unsigned 64bit dividend
// @d: unsigned 32bit divisor
//
// Divide unsigned 64bit dividend by unsigned 32bit divisor
// and round up.
//
// Return: dividend / divisor rounded up
#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_29() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = core::ptr::null_mut::<task_struct>();
return __v0;
}
let mut __v2: *mut task_struct;
let mut pscr_ret__: *mut task_struct = unsafe { core::mem::zeroed() };
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<*mut task_struct>() }) as u64 {
1 => {
let mut __v4: *mut task_struct;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v4 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v4;
}
2 => {
let mut __v6: *mut task_struct;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v6 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v6;
}
4 => {
let mut __v7: *mut task_struct;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v7 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v7;
}
8 => {
let mut __v8: *mut task_struct;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v8 = pfo_val__ as *mut task_struct;
pscr_ret__ = __v8;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v2 = pscr_ret__;
__v2
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_22 {
    pmd: u64,
}


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
// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
#[repr(C)]
#[derive(Clone, Copy)]
struct pgprot {
    pgprot: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


// No need to mask any bits for !PAE
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits116U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits121U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
}


// cpumask_bits - get the bits in a cpumask
// @maskp: the struct cpumask *
//
// You should only assume nr_cpu_ids bits of this mask are valid.  This is
// a macro so it's const-correct.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_struct {
    tls_array: [desc_struct; 3],
    sp: u64,
    es: u16,
    ds: u16,
    fsindex: u16,
    gsindex: u16,
    fsbase: u64,
    gsbase: u64,
    ptrace_bps: [*mut perf_event; 4],
    virtual_dr6: u64,
    ptrace_dr7: u64,
    cr2: u64,
    trap_nr: u64,
    error_code: u64,
    io_bitmap: *mut io_bitmap,
    iopl_emul: u64,
    __slate_bits_0: __slate_bits::__SlateBits122U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct restart_block {
    arch_data: u64,
    r#fn: Option<unsafe extern "C" fn(*mut restart_block) -> i64>,
    __slate_anon_2: __SlateRecord_restart_block_h_29,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_29 {
    futex: __SlateRecord_restart_block_h_31,
    nanosleep: __SlateRecord_restart_block_h_40,
    poll: __SlateRecord_restart_block_h_50,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_31 {
    uaddr: *mut u32,
    val: u32,
    flags: u32,
    bitset: u32,
    time: i64,
    uaddr2: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_40 {
    clockid: i32,
    r#type: u32,
    __slate_anon_2: __SlateRecord_restart_block_h_43,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_43 {
    rmtp: *mut __kernel_timespec,
    compat_rmtp: *mut old_timespec32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_50 {
    ufds: *mut pollfd,
    nfds: i32,
    has_timeout: i32,
    end_time: timespec64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_sched_data {
    robust_list: *mut robust_list_head,
    compat_robust_list: *mut compat_robust_list_head,
    pi_state_list: list_head,
    pi_state_cache: *mut futex_pi_state,
    exit_mutex: mutex,
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_phash {
    lock: mutex,
    hash: *mut futex_private_hash,
    hash_new: *mut futex_private_hash,
    batches: u64,
    rcu: callback_head,
    atomic: __SlateRecord_types_h_197,
    r#ref: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_range {
    start_ip: u64,
    len: u32,
    pop_size32: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_ranges {
    cs_ranges: [futex_unlock_cs_range; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_sem {
    undo_list: *mut sem_undo_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node_linked {
    node: rb_node,
    prev: *mut rb_node_linked,
    next: *mut rb_node_linked,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root {
    rb_node: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer {
    node: timerqueue_linked_node,
    base: *mut hrtimer_clock_base,
    is_queued: bool,
    is_rel: bool,
    is_soft: bool,
    is_hard: bool,
    is_lazy: bool,
    _softexpires: i64,
    function: Option<unsafe extern "C" fn(*mut hrtimer) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timer_list {
    entry: hlist_node,
    expires: u64,
    function: Option<unsafe extern "C" fn(*mut timer_list)>,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __sifields {
    _kill: __SlateRecord_siginfo_h_39,
    _timer: __SlateRecord_siginfo_h_45,
    _rt: __SlateRecord_siginfo_h_53,
    _sigchld: __SlateRecord_siginfo_h_60,
    _sigfault: __SlateRecord_siginfo_h_69,
    _sigpoll: __SlateRecord_siginfo_h_103,
    _sigsys: __SlateRecord_siginfo_h_109,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigaction {
    sa_handler: Option<unsafe extern "C" fn(i32)>,
    sa_flags: u64,
    sa_restorer: Option<unsafe extern "C" fn()>,
    sa_mask: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_39 {
    _pid: i32,
    _uid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_45 {
    _tid: i32,
    _overrun: i32,
    _sigval: sigval,
    _sys_private: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_53 {
    _pid: i32,
    _uid: u32,
    _sigval: sigval,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_60 {
    _pid: i32,
    _uid: u32,
    _status: i32,
    _utime: i64,
    _stime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_69 {
    _addr: *mut (),
    __slate_anon_1: __SlateRecord_siginfo_h_74,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_siginfo_h_74 {
    _trapno: i32,
    _addr_lsb: i16,
    _addr_bnd: __SlateRecord_siginfo_h_83,
    _addr_pkey: __SlateRecord_siginfo_h_89,
    _perf: __SlateRecord_siginfo_h_94,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_83 {
    _dummy_bnd: [u8; 8],
    _lower: *mut (),
    _upper: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_89 {
    _dummy_pkey: [u8; 8],
    _pkey: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_94 {
    _data: u64,
    _type: u32,
    _flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_103 {
    _band: i64,
    _fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_109 {
    _call_addr: *mut (),
    _syscall: i32,
    _arch: u32,
}


#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_frag {
    page: *mut page,
    offset: u32,
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tlbflush_unmap_batch {
    arch: arch_tlbflush_unmap_batch,
    flush_required: bool,
    writable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_io_accounting {
    rchar: u64,
    wchar: u64,
    syscr: u64,
    syscw: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers {
    bases: [posix_cputimer_base; 3],
    timers_active: u32,
    expiry_active: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers_work {
    work: callback_head,
    mutex: mutex,
    scheduled: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct work_struct {
    data: __SlateRecord_types_h_197,
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut work_struct)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_38 {
    all: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_40,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_40 {
    __slate_anon_0: __SlateRecord_rseq_types_h_41,
    has_rseq: u8,
    __pad: u8,
    __slate_anon_3: __SlateRecord_rseq_types_h_52,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_41 {
    events: u32,
    __slate_anon_1: __SlateRecord_rseq_types_h_43,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_43 {
    sched_switch: u8,
    ids_changed: u8,
    user_irq: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_52 {
    error: u16,
    __slate_anon_1: __SlateRecord_rseq_types_h_54,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_54 {
    fatal: u8,
    slowpath: u8,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_74 {
    cpu_cid: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_76,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_76 {
    cpu_id: u32,
    mm_cid: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_mm_cid {
    pcpu: *mut mm_cid_pcpu,
    mode: u32,
    max_cids: u32,
    irq_work: irq_work,
    work: work_struct,
    lock: raw_spinlock,
    mutex: mutex,
    user_list: hlist_head,
    nr_cpus_allowed: u32,
    users: u32,
    pcpu_thrs: u32,
    update_deferred: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_cache {
    unwind_completed: u64,
    nr_entries: u32,
    entries: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


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
// set_special_state() should be used for those states when the blocking task
// can not use the regular condition based wait-loop. In that case we must
// serialize against wakeups such that any possible in-flight TASK_RUNNING
// stores will not collide with our state change.
// may shadow
#[repr(C)]
#[derive(Clone, Copy)]
struct prev_cputime {
    utime: u64,
    stime: u64,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_info {
    pcount: u64,
    run_delay: u64,
    max_run_delay: u64,
    min_run_delay: u64,
    last_arrival: u64,
    last_queued: u64,
    max_run_delay_ts: timespec64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct load_weight {
    weight: u64,
    inv_weight: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_plug {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_net_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_avg {
    last_update_time: u64,
    load_sum: u64,
    runnable_sum: u64,
    util_sum: u32,
    period_contrib: u32,
    load_avg: u64,
    runnable_avg: u64,
    util_avg: u64,
    util_est: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mempolicy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_ctx_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pipe_inode_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_statistics {
    wait_start: u64,
    wait_max: u64,
    wait_count: u64,
    wait_sum: u64,
    iowait_count: u64,
    iowait_sum: u64,
    sleep_start: u64,
    sleep_max: u64,
    sum_sleep_runtime: i64,
    block_start: u64,
    block_max: u64,
    sum_block_runtime: i64,
    exec_max: i64,
    slice_max: u64,
    nr_failed_migrations_affine: u64,
    nr_failed_migrations_running: u64,
    nr_failed_migrations_hot: u64,
    nr_forced_migrations: u64,
    nr_wakeups: u64,
    nr_wakeups_sync: u64,
    nr_wakeups_migrate: u64,
    nr_wakeups_local: u64,
    nr_wakeups_remote: u64,
    nr_wakeups_affine: u64,
    nr_wakeups_affine_attempts: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_entity {
    load: load_weight,
    h_load: load_weight,
    run_node: rb_node,
    deadline: u64,
    min_vruntime: u64,
    min_slice: u64,
    max_slice: u64,
    group_node: list_head,
    on_rq: u8,
    sched_delayed: u8,
    rel_deadline: u8,
    custom_slice: u8,
    exec_start: u64,
    sum_exec_runtime: u64,
    prev_sum_exec_runtime: u64,
    vruntime: u64,
    vlag: i64,
    vprot: u64,
    slice: u64,
    nr_migrations: u64,
    depth: i32,
    parent: *mut sched_entity,
    cfs_rq: *mut cfs_rq,
    my_q: *mut cfs_rq,
    runnable_weight: u64,
    avg: sched_avg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_rt_entity {
    run_list: list_head,
    timeout: u64,
    watchdog_stamp: u64,
    time_slice: u32,
    on_rq: u16,
    on_list: u16,
    back: *mut sched_rt_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_dl_entity {
    rb_node: rb_node,
    dl_runtime: u64,
    dl_deadline: u64,
    dl_period: u64,
    dl_bw: u64,
    dl_density: u64,
    runtime: i64,
    deadline: u64,
    flags: u32,
    __slate_bits_0: __slate_bits::__SlateBits276U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_delay_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_exec_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_struct {
    thread_info: thread_info,
    __state: u32,
    saved_state: u32,
    stack: *mut (),
    usage: refcount_struct,
    flags: u32,
    ptrace: u32,
    on_cpu: u8,
    on_rq: u8,
    is_blocked: u8,
    __pad: u8,
    wake_entry: __call_single_node,
    wakee_flips: u32,
    wakee_flip_decay_ts: u64,
    last_wakee: *mut task_struct,
    recent_used_cpu: i32,
    wake_cpu: i32,
    prio: i32,
    static_prio: i32,
    normal_prio: i32,
    rt_priority: u32,
    se: sched_entity,
    rt: sched_rt_entity,
    dl: sched_dl_entity,
    dl_server: *mut sched_dl_entity,
    sched_class: *const sched_class,
    sched_task_group: *mut task_group,
    stats: sched_statistics,
    btrace_seq: u32,
    policy: u32,
    max_allowed_capacity: u64,
    nr_cpus_allowed: i32,
    cpus_ptr: *const cpumask,
    user_cpus_ptr: *mut cpumask,
    cpus_mask: cpumask,
    migration_pending: *mut (),
    migration_disabled: u16,
    migration_flags: u16,
    rcu_read_lock_nesting: i32,
    rcu_read_unlock_special: rcu_special,
    rcu_node_entry: list_head,
    rcu_blocked_node: *mut rcu_node,
    rcu_tasks_nvcsw: u64,
    rcu_tasks_holdout: u8,
    rcu_tasks_idx: u8,
    rcu_tasks_idle_cpu: i32,
    rcu_tasks_holdout_list: list_head,
    rcu_tasks_exit_cpu: i32,
    rcu_tasks_exit_list: list_head,
    trc_reader_nesting: i32,
    trc_reader_scp: *mut srcu_ctr,
    sched_info: sched_info,
    tasks: list_head,
    pushable_tasks: plist_node,
    pushable_dl_tasks: rb_node,
    mm: *mut mm_struct,
    active_mm: *mut mm_struct,
    exec_state: *mut task_exec_state,
    exit_state: i32,
    exit_code: i32,
    exit_signal: i32,
    pdeath_signal: i32,
    jobctl: u64,
    personality: u32,
    __slate_bits_0: __slate_bits::__SlateBits68U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits68U1,
    atomic_flags: u64,
    restart_block: restart_block,
    pid: i32,
    tgid: i32,
    stack_canary: u64,
    real_parent: *mut task_struct,
    parent: *mut task_struct,
    children: list_head,
    sibling: list_head,
    group_leader: *mut task_struct,
    ptraced: list_head,
    ptrace_entry: list_head,
    thread_pid: *mut pid,
    pid_links: [hlist_node; 4],
    thread_node: list_head,
    vfork_done: *mut completion,
    set_child_tid: *mut i32,
    clear_child_tid: *mut i32,
    worker_private: *mut (),
    utime: u64,
    stime: u64,
    gtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    start_time: u64,
    start_boottime: u64,
    min_flt: u64,
    maj_flt: u64,
    posix_cputimers: posix_cputimers,
    posix_cputimers_work: posix_cputimers_work,
    ptracer_cred: *const cred,
    real_cred: *const cred,
    cred: *const cred,
    cached_requested_key: *mut key,
    comm: [u8; 16],
    nameidata: *mut nameidata,
    sysvsem: sysv_sem,
    sysvshm: sysv_shm,
    real_fs: *mut fs_struct,
    fs: *mut fs_struct,
    files: *mut files_struct,
    io_uring: *mut io_uring_task,
    io_uring_restrict: *mut io_restriction,
    nsproxy: *mut nsproxy,
    signal: *mut signal_struct,
    sighand: *mut sighand_struct,
    blocked: __SlateRecord_signal_h_25,
    real_blocked: __SlateRecord_signal_h_25,
    saved_sigmask: __SlateRecord_signal_h_25,
    pending: sigpending,
    sas_ss_sp: u64,
    sas_ss_size: u64,
    sas_ss_flags: u32,
    task_works: *mut callback_head,
    audit_context: *mut audit_context,
    loginuid: __SlateRecord_uidgid_types_h_9,
    sessionid: u32,
    seccomp: seccomp,
    syscall_dispatch: syscall_user_dispatch,
    parent_exec_id: u64,
    self_exec_id: u64,
    alloc_lock: spinlock,
    pi_lock: raw_spinlock,
    wake_q: wake_q_node,
    pi_waiters: rb_root_cached,
    pi_top_task: *mut task_struct,
    pi_blocked_on: *mut rt_mutex_waiter,
    blocked_on: *mut mutex,
    blocked_lock: raw_spinlock,
    blocked_donor: *mut task_struct,
    journal_info: *mut (),
    bio_list: *mut bio_list,
    plug: *mut blk_plug,
    reclaim_state: *mut reclaim_state,
    io_context: *mut io_context,
    capture_control: *mut capture_control,
    ptrace_message: u64,
    last_siginfo: *mut kernel_siginfo,
    ioac: task_io_accounting,
    acct_rss_mem1: u64,
    acct_vm_mem1: u64,
    acct_timexpd: u64,
    mems_allowed: __SlateRecord_nodemask_types_h_17,
    mems_allowed_seq: seqcount_spinlock,
    cpuset_mem_spread_rotor: i32,
    cgroups: *mut css_set,
    cg_list: list_head,
    futex: futex_sched_data,
    perf_recursion: [u8; 4],
    perf_event_ctxp: *mut perf_event_context,
    perf_event_mutex: mutex,
    perf_event_list: list_head,
    perf_ctx_data: *mut perf_ctx_data,
    ipi_mask: task_ipi_mask,
    mempolicy: *mut mempolicy,
    il_prev: i16,
    il_weight: u8,
    pref_node_fork: i16,
    cache_work: callback_head,
    sched_cache_grp: *mut sched_cache_group,
    preferred_llc: i32,
    pref_llc_queued: i32,
    rseq: rseq_data,
    mm_cid: sched_mm_cid,
    tlb_ubc: tlbflush_unmap_batch,
    splice_pipe: *mut pipe_inode_info,
    task_frag: page_frag,
    delays: *mut task_delay_info,
    nr_dirtied: i32,
    nr_dirtied_pause: i32,
    dirty_paused_when: u64,
    timer_slack_ns: u64,
    default_timer_slack_ns: u64,
    trace_recursion: u64,
    throttle_disk: *mut gendisk,
    utask: *mut uprobe_task,
    kmap_ctrl: kmap_ctrl,
    rcu: callback_head,
    rcu_users: refcount_struct,
    pagefault_disabled: i32,
    oom_reaper_list: *mut task_struct,
    oom_reaper_timer: timer_list,
    stack_vm_area: *mut vm_struct,
    stack_refcount: refcount_struct,
    security: *mut (),
    bpf_net_context: *mut bpf_net_context,
    mce_vaddr: *mut (),
    mce_kflags: u64,
    mce_addr: u64,
    __slate_bits_2: __slate_bits::__SlateBits68U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


// Per process flags
// I'm a virtual CPU
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
#[repr(C)]
#[derive(Clone, Copy)]
struct sched_cache_time {
    runtime: u64,
    epoch: u64,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_cache_group {
    pcpu_sched: *mut sched_cache_time,
    lock: raw_spinlock,
    epoch: u64,
    nr_running_avg: u64,
    next_scan: u64,
    footprint: u64,
    cpu: i32,
    refcnt: refcount_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstat {
    result_mask: u32,
    mode: u16,
    nlink: u32,
    blksize: u32,
    attributes: u64,
    attributes_mask: u64,
    ino: u64,
    dev: u32,
    rdev: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    size: i64,
    atime: timespec64,
    mtime: timespec64,
    ctime: timespec64,
    btime: timespec64,
    blocks: u64,
    mnt_id: u64,
    change_cookie: u64,
    subvol: u64,
    dio_mem_align: u32,
    dio_offset_align: u32,
    dio_read_offset_align: u32,
    atomic_write_unit_min: u32,
    atomic_write_unit_max: u32,
    atomic_write_unit_max_opt: u32,
    atomic_write_segments_max: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


// Dependency order vs. p above.
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
// rcu_replace_pointer() - replace an RCU pointer, returning its old value
// @rcu_ptr: RCU pointer, whose old value is returned
// @ptr: regular pointer
// @c: the lockdep conditions under which the dereference will take place
//
// Perform a replacement, where @rcu_ptr is an RCU-annotated
// pointer and @c is the lockdep argument that is passed to the
// rcu_dereference_protected() call used to read that pointer.  The old
// value of @rcu_ptr is returned, and @rcu_ptr is set to @ptr.
// rcu_access_pointer() - fetch RCU pointer with no dereferencing
// @p: The pointer to read
//
// Return the value of the specified RCU-protected pointer, but omit the
// lockdep checks for being in an RCU read-side critical section.  This is
// useful when the value of this pointer is accessed, but the pointer is
// not dereferenced, for example, when testing an RCU-protected pointer
// against NULL.  Within an RCU read-side critical section, there is little
// reason to use rcu_access_pointer().  Although rcu_access_pointer() may
// also be used in cases where update-side locks prevent the value of the
// pointer from changing, you should instead use rcu_dereference_protected()
// for this use case.  It is also permissible to use rcu_access_pointer()
// within lockless updaters to obtain the old value for an atomic operation,
// for example, for cmpxchg().
//
// It is usually best to test the rcu_access_pointer() return value
// directly in order to avoid accidental dereferences being introduced
// by later inattentive changes.  In other words, assigning the
// rcu_access_pointer() return value to a local variable results in an
// accident waiting to happen.
//
// It is also permissible to use rcu_access_pointer() when read-side
// access to the pointer was removed at least one grace period ago, as is
// the case in the context of the RCU callback that is freeing up the data,
// or after a synchronize_rcu() returns.  This can be useful when tearing
// down multi-linked structures after a grace period has elapsed.  However,
// rcu_dereference_protected() is normally preferred for this use case.
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
// rcu_dereference_protected() - fetch RCU pointer when updates prevented
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Return the value of the specified RCU-protected pointer, but omit
// the READ_ONCE().  This is useful in cases where update-side locks
// prevent the value of the pointer from changing.  Please note that this
// primitive does *not* prevent the compiler from repeating this reference
// or combining it with other references, so it should not be used without
// protection of appropriate locks.
//
// This function is only for update-side use.  Using this function
// when protected only by rcu_read_lock() will result in infrequent
// but very ugly failures.
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
// 256 bytes including ->parent
// 256 bytes
// 240 bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct maple_tree {
    __slate_anon_0: __SlateRecord_maple_tree_h_261,
    ma_flags: u32,
    ma_root: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct files_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_restriction {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_segcblist {
    head: *mut callback_head,
    tails: [*mut *mut callback_head; 4],
    gp_seq: [rcu_gp_seq; 4],
    len: i64,
    seglen: [i64; 4],
    flags: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct srcu_data {
    srcu_ctrs: [srcu_ctr; 2],
    srcu_reader_flavor: i32,
    __slate_pad_2: [u8; 28],
    lock: raw_spinlock,
    srcu_cblist: rcu_segcblist,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_cblist_invoking: bool,
    delay_work: timer_list,
    work: work_struct,
    srcu_barrier_head: callback_head,
    srcu_ec_head: callback_head,
    srcu_ec_state: i32,
    mynode: *mut srcu_node,
    grpmask: u64,
    cpu: i32,
    ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_node {
    lock: raw_spinlock,
    srcu_have_cbs: [u64; 4],
    srcu_data_have_cbs: [u64; 4],
    srcu_gp_seq_needed_exp: u64,
    srcu_parent: *mut srcu_node,
    grplo: i32,
    grphi: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_usage {
    node: *mut srcu_node,
    level: [*mut srcu_node; 3],
    srcu_size_state: i32,
    srcu_cb_mutex: mutex,
    lock: raw_spinlock,
    srcu_gp_mutex: mutex,
    srcu_gp_seq: u64,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_gp_start: u64,
    srcu_last_gp_end: u64,
    srcu_size_jiffies: u64,
    srcu_n_lock_retries: u64,
    srcu_n_exp_nodelay: u64,
    sda_is_static: bool,
    srcu_barrier_seq: u64,
    srcu_barrier_mutex: mutex,
    srcu_barrier_completion: completion,
    srcu_barrier_cpu_cnt: __SlateRecord_types_h_190,
    reschedule_jiffies: u64,
    reschedule_count: u64,
    work: delayed_work,
    irq_work: irq_work,
    srcu_ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_task {
    state: u32,
    depth: u32,
    return_instances: *mut return_instance,
    ri_pool: *mut return_instance,
    ri_timer: timer_list,
    ri_seqcount: seqcount,
    __slate_anon_6: __SlateRecord_uprobes_h_136,
    active_uprobe: *mut uprobe,
    xol_vaddr: u64,
    signal_denied: bool,
    auprobe: *mut arch_uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct return_consumer {
    cookie: u64,
    id: u64,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct return_instance {
    hprobe: hprobe,
    func: u64,
    stack: u64,
    orig_ret_vaddr: u64,
    chained: bool,
    cons_cnt: i32,
    next: *mut return_instance,
    rcu: callback_head,
    consumer: return_consumer,
    extra_consumers: *mut return_consumer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mmu_h_84 {
    ctx_id: u64,
    tlb_gen: __SlateRecord_types_h_197,
    next_trim_cpumask: u64,
    ldt_usr_sem: rw_semaphore,
    ldt: *mut ldt_struct,
    flags: u64,
    lock: mutex,
    vdso: *mut (),
    vdso_image: *const vdso_image,
    perf_rdpmc_allowed: __SlateRecord_types_h_190,
    pkey_allocation_map: u16,
    execute_only_pkey: i16,
    global_asid: u16,
    asid_transition: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_41 {
    f: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_31 {
    insn: [u8; 16],
    ixol: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_38 {
    branch: __SlateRecord_uprobes_h_39,
    defparam: __SlateRecord_uprobes_h_44,
    push: __SlateRecord_uprobes_h_48,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_39 {
    offs: i32,
    ilen: u8,
    opc1: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_44 {
    fixups: u8,
    ilen: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_48 {
    reg_offset: u8,
    ilen: u8,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct page {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_89,
    __slate_anon_2: __SlateRecord_mm_types_h_155,
    _refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio {
    __slate_anon_0: __SlateRecord_mm_types_h_406,
    __slate_anon_1: __SlateRecord_mm_types_h_451,
    __slate_anon_2: __SlateRecord_mm_types_h_483,
    __slate_anon_3: __SlateRecord_mm_types_h_497,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_136 {
    __slate_anon_0: __SlateRecord_uprobes_h_137,
    __slate_anon_1: __SlateRecord_uprobes_h_142,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_137 {
    autask: arch_uprobe_task,
    vaddr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_142 {
    dup_xol_work: callback_head,
    dup_xol_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits400U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
}


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_area_desc {
    mm: *mut mm_struct,
    file: *mut file,
    start: u64,
    end: u64,
    pgoff: u64,
    vm_file: *mut file,
    vma_flags: __SlateRecord_mm_types_h_871,
    page_prot: pgprot,
    vm_ops: *const vm_operations_struct,
    private_data: *mut (),
    action: mmap_action,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct vm_area_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_926,
    vm_mm: *mut mm_struct,
    vm_page_prot: pgprot,
    __slate_anon_3: __SlateRecord_mm_types_h_947,
    vm_lock_seq: u32,
    __vm_anon_pgoff_lo: u32,
    anon_vma_chain: list_head,
    anon_vma: *mut anon_vma,
    vm_ops: *const vm_operations_struct,
    vm_pgoff: u64,
    vm_file: *mut file,
    vm_private_data: *mut (),
    swap_readahead_info: __SlateRecord_types_h_197,
    vm_policy: *mut mempolicy,
    __slate_pad_14: [u8; 8],
    vm_refcnt: refcount_struct,
    __vm_anon_pgoff_hi: u32,
    shared: __SlateRecord_mm_types_h_1062,
    vm_userfaultfd_ctx: vm_userfaultfd_ctx,
    pfnmap_track_ctx: *mut pfnmap_track_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_89 {
    __slate_anon_0: __SlateRecord_mm_types_h_90,
    __slate_anon_1: __SlateRecord_mm_types_h_118,
    __slate_anon_2: __SlateRecord_mm_types_h_129,
    __slate_anon_3: __SlateRecord_mm_types_h_132,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_90 {
    __slate_anon_0: __SlateRecord_mm_types_h_96,
    mapping: *mut address_space,
    __slate_anon_2: __SlateRecord_mm_types_h_105,
    private: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_96 {
    lru: list_head,
    buddy_list: list_head,
    pcp_list: list_head,
    pcp_llist: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_105 {
    __folio_index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_118 {
    pp_magic: u64,
    pp: *mut page_pool,
    _pp_mapping_pad: u64,
    dma_addr: u64,
    pp_ref_count: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_129 {
    compound_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_132 {
    _unused_pgmap_compound_info: *mut (),
    zone_device_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_155 {
    page_type: u32,
    _mapcount: __SlateRecord_types_h_190,
}


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_406 {
    __slate_anon_0: __SlateRecord_mm_types_h_407,
    page: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_407 {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_410,
    mapping: *mut address_space,
    __slate_anon_3: __SlateRecord_mm_types_h_425,
    __slate_anon_4: __SlateRecord_mm_types_h_429,
    _mapcount: __SlateRecord_types_h_190,
    _refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_410 {
    lru: list_head,
    __slate_anon_1: __SlateRecord_mm_types_h_414,
    pgmap: *mut dev_pagemap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_414 {
    __filler: *mut (),
    mlock_count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mm_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_1176,
    flexible_array: [u8; 0],
}


// Clear the access bit
// Clear the dirty bit
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_425 {
    index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_429 {
    private: *mut (),
    swap: __SlateRecord_mm_types_h_288,
    migrate_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_451 {
    __slate_anon_0: __SlateRecord_mm_types_h_452,
    __page_1: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_452 {
    _flags_1: u64,
    _head_1: u64,
    __slate_anon_2: __SlateRecord_mm_types_h_455,
    _mapcount_1: __SlateRecord_types_h_190,
    _refcount_1: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_455 {
    __slate_anon_0: __SlateRecord_mm_types_h_456,
    _usable_1: [u64; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_456 {
    _large_mapcount: __SlateRecord_types_h_190,
    _nr_pages_mapped: __SlateRecord_types_h_190,
    _entire_mapcount: __SlateRecord_types_h_190,
    _pincount: __SlateRecord_types_h_190,
    _mm_id_mapcount: [i32; 2],
    __slate_anon_5: __SlateRecord_mm_types_h_465,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_465 {
    _mm_id: [u32; 2],
    _mm_ids: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_483 {
    __slate_anon_0: __SlateRecord_mm_types_h_484,
    __page_2: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_484 {
    _flags_2: u64,
    _head_2: u64,
    _deferred_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_497 {
    __slate_anon_0: __SlateRecord_mm_types_h_498,
    __page_3: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_498 {
    _flags_3: u64,
    _head_3: u64,
    _hugetlb_subpool: *mut (),
    _hugetlb_cgroup: *mut (),
    _hugetlb_cgroup_rsvd: *mut (),
    _hugetlb_hwpoison: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kvec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iov_iter {
    iter_type: u8,
    nofault: bool,
    data_source: bool,
    iov_offset: u64,
    __slate_anon_4: __SlateRecord_uio_h_58,
    __slate_anon_5: __SlateRecord_uio_h_78,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_830 {
    remap: __SlateRecord_mm_types_h_831,
    simple_ioremap: __SlateRecord_mm_types_h_837,
    map_kernel: __SlateRecord_mm_types_h_841,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_831 {
    start: u64,
    start_pfn: u64,
    size: u64,
    pgprot: pgprot,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_837 {
    start_phys_addr: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_841 {
    start: u64,
    pages: *mut *mut page,
    nr_pages: u64,
    pgoff: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrinker {
    count_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    scan_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    batch: i64,
    seeks: i32,
    flags: u32,
    refcount: refcount_struct,
    done: completion,
    rcu: callback_head,
    private_data: *mut (),
    list: list_head,
    nr_deferred: *mut __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_926 {
    __slate_anon_0: __SlateRecord_mm_types_h_927,
    vm_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_927 {
    vm_start: u64,
    vm_end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_947 {
    vm_flags: u64,
    flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_mm_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1176 {
    __slate_anon_0: __SlateRecord_mm_types_h_1181,
    mm_mt: maple_tree,
    mmap_base: u64,
    mmap_legacy_base: u64,
    mmap_compat_base: u64,
    mmap_compat_legacy_base: u64,
    task_size: u64,
    pgd: *mut __SlateRecord_pgtable_types_h_295,
    membarrier_state: __SlateRecord_types_h_190,
    mm_users: __SlateRecord_types_h_190,
    mm_cid: mm_mm_cid,
    sched_cache_grp: *mut sched_cache_group,
    pgtables_bytes: __SlateRecord_types_h_197,
    map_count: i32,
    page_table_lock: spinlock,
    mmap_lock: rw_semaphore,
    mmlist: list_head,
    vma_writer_wait: rcuwait,
    mm_lock_seq: seqcount,
    futex: futex_mm_data,
    hiwater_rss: u64,
    hiwater_vm: u64,
    total_vm: u64,
    locked_vm: u64,
    pinned_vm: __SlateRecord_types_h_197,
    data_vm: u64,
    exec_vm: u64,
    stack_vm: u64,
    __slate_anon_28: __SlateRecord_mm_types_h_1289,
    write_protect_seq: seqcount,
    arg_lock: spinlock,
    start_code: u64,
    end_code: u64,
    start_data: u64,
    end_data: u64,
    start_brk: u64,
    brk: u64,
    start_stack: u64,
    arg_start: u64,
    arg_end: u64,
    env_start: u64,
    env_end: u64,
    saved_auxv: [u64; 56],
    rss_stat: [percpu_counter; 4],
    binfmt: *mut linux_binfmt,
    context: __SlateRecord_mmu_h_84,
    flags: __SlateRecord_mm_types_h_1171,
    ioctx_lock: spinlock,
    ioctx_table: *mut kioctx_table,
    exe_file: *mut file,
    notifier_subscriptions: *mut mmu_notifier_subscriptions,
    tlb_flush_pending: __SlateRecord_types_h_190,
    tlb_flush_batched: __SlateRecord_types_h_190,
    uprobes_state: uprobes_state,
    hugetlb_usage: __SlateRecord_types_h_197,
    async_put_work: work_struct,
    iommu_mm: *mut iommu_mm_data,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1181 {
    mm_count: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_1289 {
    def_flags: u64,
    def_vma_flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct linux_binfmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_notifier_subscriptions {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_58 {
    __ubuf_iovec: iovec,
    __slate_anon_1: __SlateRecord_uio_h_65,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uio_h_65 {
    __slate_anon_0: __SlateRecord_uio_h_66,
    count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_66 {
    __iov: *const iovec,
    kvec: *const kvec,
    bvec: *const bio_vec,
    folioq: *const folio_queue,
    xarray: *mut xarray,
    ubuf: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct keyring_index_key {
    hash: u64,
    __slate_anon_1: __SlateRecord_key_h_117,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
union key_payload {
    rcu_data0: *mut (),
    data: [*mut (); 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_restriction {
    check: Option<unsafe extern "C" fn(*mut key, *const key_type, *const key_payload, *mut key) -> i32>,
    key: *mut key,
    keytype: *mut key_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key {
    usage: refcount_struct,
    serial: i32,
    __slate_anon_2: __SlateRecord_key_h_198,
    sem: rw_semaphore,
    user: *mut key_user,
    security: *mut (),
    __slate_anon_6: __SlateRecord_key_h_208,
    last_used_at: i64,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    perm: u32,
    quotalen: u16,
    datalen: u16,
    state: i16,
    flags: u64,
    __slate_anon_15: __SlateRecord_key_h_246,
    __slate_anon_16: __SlateRecord_key_h_261,
    restrict_link: *mut key_restriction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_struct {
    __count: refcount_struct,
    epoll_watches: percpu_counter,
    unix_inflight: u64,
    pipe_bufs: __SlateRecord_types_h_197,
    uidhash_node: hlist_node,
    uid: __SlateRecord_uidgid_types_h_9,
    locked_vm: __SlateRecord_types_h_197,
    ratelimit: ratelimit_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct group_info {
    usage: refcount_struct,
    ngroups: i32,
    gid: [__SlateRecord_uidgid_types_h_13; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cred {
    usage: __SlateRecord_types_h_197,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    suid: __SlateRecord_uidgid_types_h_9,
    sgid: __SlateRecord_uidgid_types_h_13,
    euid: __SlateRecord_uidgid_types_h_9,
    egid: __SlateRecord_uidgid_types_h_13,
    fsuid: __SlateRecord_uidgid_types_h_9,
    fsgid: __SlateRecord_uidgid_types_h_13,
    securebits: u32,
    cap_inheritable: __SlateRecord_capability_h_24,
    cap_permitted: __SlateRecord_capability_h_24,
    cap_effective: __SlateRecord_capability_h_24,
    cap_bset: __SlateRecord_capability_h_24,
    cap_ambient: __SlateRecord_capability_h_24,
    jit_keyring: u8,
    session_keyring: *mut key,
    process_keyring: *mut key,
    thread_keyring: *mut key,
    request_key_auth: *mut key,
    security: *mut (),
    user: *mut user_struct,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    group_info: *mut group_info,
    __slate_anon_25: __SlateRecord_cred_h_147,
}


// current_cred - Access the current task's subjective credentials
//
// Access the subjective credentials of the current task.  RCU-safe,
// since nobody else can modify it.
#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid {
    count: refcount_struct,
    level: u32,
    lock: spinlock,
    __slate_anon_3: __SlateRecord_pid_h_62,
    tasks: [hlist_head; 4],
    inodes: hlist_head,
    wait_pidfd: wait_queue_head,
    rcu: callback_head,
    numbers: [upid; 0],
}


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pacct_struct {
    ac_flag: i32,
    ac_exitcode: i64,
    ac_mem: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_117 {
    __slate_anon_0: __SlateRecord_key_h_118,
    x: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_118 {
    desc_len: u16,
    desc: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_cputime_atomic {
    utime: __SlateRecord_types_h_197,
    stime: __SlateRecord_types_h_197,
    sum_exec_runtime: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_198 {
    graveyard_link: list_head,
    serial_node: rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_user {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_208 {
    expiry: i64,
    revoked_at: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_246 {
    index_key: keyring_index_key,
    __slate_anon_1: __SlateRecord_key_h_248,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_248 {
    hash: u64,
    len_desc: u64,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_261 {
    payload: key_payload,
    __slate_anon_1: __SlateRecord_key_h_263,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_263 {
    name_link: list_head,
    keys: assoc_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_state {
    nr_threads: __SlateRecord_types_h_190,
    dumper: core_thread,
    startup: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct signal_struct {
    sigcnt: refcount_struct,
    live: __SlateRecord_types_h_190,
    nr_threads: i32,
    quick_threads: i32,
    thread_head: list_head,
    wait_chldexit: wait_queue_head,
    curr_target: *mut task_struct,
    shared_pending: sigpending,
    multiprocess: hlist_head,
    group_exit_code: i32,
    notify_count: i32,
    group_exec_task: *mut task_struct,
    group_stop_count: i32,
    flags: u32,
    core_state: *mut core_state,
    __slate_bits_0: __slate_bits::__SlateBits279U0,
    next_posix_timer_id: __SlateRecord_types_h_190,
    posix_timers: hlist_head,
    ignored_posix_timers: hlist_head,
    real_timer: hrtimer,
    it_real_incr: i64,
    it: [cpu_itimer; 2],
    cputimer: thread_group_cputimer,
    posix_cputimers: posix_cputimers,
    pids: [*mut pid; 4],
    tty_old_pgrp: *mut pid,
    leader: i32,
    tty: *mut tty_struct,
    stats_lock: seqlock,
    utime: u64,
    stime: u64,
    cutime: u64,
    cstime: u64,
    gtime: u64,
    cgtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    cnvcsw: u64,
    cnivcsw: u64,
    min_flt: u64,
    maj_flt: u64,
    cmin_flt: u64,
    cmaj_flt: u64,
    inblock: u64,
    oublock: u64,
    cinblock: u64,
    coublock: u64,
    maxrss: u64,
    cmaxrss: u64,
    ioac: task_io_accounting,
    sum_sched_runtime: u64,
    rlim: [rlimit; 16],
    pacct: pacct_struct,
    stats: *mut taskstats,
    audit_tty: u32,
    tty_audit_buf: *mut tty_audit_buf,
    cgroup_threadgroup_rwsem: rw_semaphore,
    oom_flag_origin: bool,
    oom_score_adj: i16,
    oom_score_adj_min: i16,
    oom_mm: *mut mm_struct,
    cred_guard_mutex: mutex,
    exec_update_lock: rw_semaphore,
}


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_361() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*((*(*task).signal).rlim.as_mut_ptr() as *mut rlimit).wrapping_offset(limit as isize)).rlim_cur) as *const u64) };
__v0
}

fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_sync {
    gp_state: i32,
    gp_count: i32,
    gp_wait: wait_queue_head,
    cb_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_projid_h_24 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kqid {
    __slate_anon_0: __SlateRecord_quota_h_69,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqblk {
    dqb_bhardlimit: i64,
    dqb_bsoftlimit: i64,
    dqb_curspace: i64,
    dqb_rsvspace: i64,
    dqb_ihardlimit: i64,
    dqb_isoftlimit: i64,
    dqb_curinodes: i64,
    dqb_btime: i64,
    dqb_itime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqinfo {
    dqi_format: *mut quota_format_type,
    dqi_fmt_id: i32,
    dqi_dirty_list: list_head,
    dqi_flags: u64,
    dqi_bgrace: u32,
    dqi_igrace: u32,
    dqi_max_spc_limit: i64,
    dqi_max_ino_limit: i64,
    dqi_priv: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot {
    dq_hash: hlist_node,
    dq_inuse: list_head,
    dq_free: list_head,
    dq_dirty: list_head,
    dq_lock: mutex,
    dq_dqb_lock: spinlock,
    dq_count: __SlateRecord_types_h_190,
    dq_sb: *mut super_block,
    dq_id: kqid,
    dq_off: i64,
    dq_flags: u64,
    dq_dqb: mem_dqblk,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_ops {
    check_quota_file: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    write_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    free_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    commit_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot_operations {
    write_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    alloc_dquot: Option<unsafe extern "C" fn(*mut super_block, i32) -> *mut dquot>,
    destroy_dquot: Option<unsafe extern "C" fn(*mut dquot)>,
    acquire_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    mark_dirty: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    write_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    get_reserved_space: Option<unsafe extern "C" fn(*mut inode) -> *mut i64>,
    get_projid: Option<unsafe extern "C" fn(*mut inode, *mut __SlateRecord_projid_h_24) -> i32>,
    get_inode_usage: Option<unsafe extern "C" fn(*mut inode, *mut i64) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_dqblk {
    d_fieldmask: i32,
    d_spc_hardlimit: u64,
    d_spc_softlimit: u64,
    d_ino_hardlimit: u64,
    d_ino_softlimit: u64,
    d_space: u64,
    d_ino_count: u64,
    d_ino_timer: i64,
    d_spc_timer: i64,
    d_ino_warns: i32,
    d_spc_warns: i32,
    d_rt_spc_hardlimit: u64,
    d_rt_spc_softlimit: u64,
    d_rt_space: u64,
    d_rt_spc_timer: i64,
    d_rt_spc_warns: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_type_state {
    flags: u32,
    spc_timelimit: u32,
    ino_timelimit: u32,
    rt_spc_timelimit: u32,
    spc_warnlimit: u32,
    ino_warnlimit: u32,
    rt_spc_warnlimit: u32,
    ino: u64,
    blocks: u64,
    nextents: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_info {
    i_fieldmask: i32,
    i_flags: u32,
    i_spc_timelimit: u32,
    i_ino_timelimit: u32,
    i_rt_spc_timelimit: u32,
    i_spc_warnlimit: u32,
    i_ino_warnlimit: u32,
    i_rt_spc_warnlimit: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quotactl_ops {
    quota_on: Option<unsafe extern "C" fn(*mut super_block, i32, i32, *const path) -> i32>,
    quota_off: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    quota_enable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_disable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_sync: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    set_info: Option<unsafe extern "C" fn(*mut super_block, i32, *mut qc_info) -> i32>,
    get_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_nextdqblk: Option<unsafe extern "C" fn(*mut super_block, *mut kqid, *mut qc_dqblk) -> i32>,
    set_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_state: Option<unsafe extern "C" fn(*mut super_block, *mut qc_state) -> i32>,
    rm_xquota: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_type {
    qf_fmt_id: i32,
    qf_ops: *const quota_format_ops,
    qf_owner: *mut module,
    qf_next: *mut quota_format_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_info {
    flags: u32,
    dqio_sem: rw_semaphore,
    files: [*mut inode; 3],
    info: [mem_dqinfo; 3],
    ops: [*const quota_format_ops; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_operations {
    alloc_inode: Option<unsafe extern "C" fn(*mut super_block) -> *mut inode>,
    destroy_inode: Option<unsafe extern "C" fn(*mut inode)>,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
    dirty_inode: Option<unsafe extern "C" fn(*mut inode, i32)>,
    write_inode: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    sync_inode_metadata: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    drop_inode: Option<unsafe extern "C" fn(*mut inode) -> i32>,
    evict_inode: Option<unsafe extern "C" fn(*mut inode)>,
    put_super: Option<unsafe extern "C" fn(*mut super_block)>,
    sync_fs: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    freeze_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    freeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    thaw_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    unfreeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    statfs: Option<unsafe extern "C" fn(*mut dentry, *mut kstatfs) -> i32>,
    umount_begin: Option<unsafe extern "C" fn(*mut super_block)>,
    show_options: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_devname: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_path: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_stats: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    quota_read: Option<unsafe extern "C" fn(*mut super_block, i32, *mut u8, u64, i64) -> i64>,
    quota_write: Option<unsafe extern "C" fn(*mut super_block, i32, *const u8, u64, i64) -> i64>,
    get_dquots: Option<unsafe extern "C" fn(*mut inode) -> *mut *mut dquot>,
    nr_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    free_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    remove_bdev: Option<unsafe extern "C" fn(*mut super_block, *mut block_device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut super_block)>,
    report_error: Option<unsafe extern "C" fn(*const fserror_event)>,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct super_block {
    s_list: list_head,
    s_dev: u32,
    s_super_dev: *mut super_dev,
    s_blocksize_bits: u8,
    s_blocksize: u64,
    s_maxbytes: i64,
    s_type: *mut file_system_type,
    s_op: *const super_operations,
    dq_op: *const dquot_operations,
    s_qcop: *const quotactl_ops,
    s_export_op: *const export_operations,
    s_flags: u64,
    s_iflags: u64,
    s_magic: u64,
    s_root: *mut dentry,
    s_umount: rw_semaphore,
    s_passive: refcount_struct,
    s_active: __SlateRecord_types_h_190,
    s_security: *mut (),
    s_xattr: *const *const xattr_handler,
    s_roots: hlist_head,
    s_roots_lock: spinlock,
    s_mounts: *mut mount,
    s_bdev: *mut block_device,
    s_bdev_file: *mut file,
    s_bdi: *mut backing_dev_info,
    s_mtd: *mut mtd_info,
    s_instances: hlist_node,
    s_quota_types: u32,
    s_dquot: quota_info,
    s_writers: sb_writers,
    s_fs_info: *mut (),
    s_time_gran: u32,
    s_time_min: i64,
    s_time_max: i64,
    s_fsnotify_mask: u32,
    s_fsnotify_info: *mut fsnotify_sb_info,
    s_id: [u8; 32],
    s_uuid: __SlateRecord_uuid_h_21,
    s_uuid_len: u8,
    s_sysfs_name: [u8; 37],
    s_max_links: u32,
    s_d_flags: u32,
    s_vfs_rename_mutex: mutex,
    s_subtype: *const u8,
    __s_d_op: *const dentry_operations,
    s_shrink: *mut shrinker,
    s_remove_count: __SlateRecord_types_h_197,
    s_readonly_remount: i32,
    s_wb_err: u32,
    s_dio_done_wq: *mut workqueue_struct,
    s_pins: hlist_head,
    s_user_ns: *mut user_namespace,
    s_dentry_lru: list_lru,
    s_inode_lru: list_lru,
    rcu: callback_head,
    destroy_work: work_struct,
    s_sync_lock: mutex,
    s_stack_depth: i32,
    __slate_pad_59: [u8; 12],
    s_inode_list_lock: spinlock,
    s_inodes: list_head,
    s_inode_wblist_lock: spinlock,
    s_inodes_wb: list_head,
    s_min_writeback_pages: i64,
    s_pending_errors: refcount_struct,
}


// sb->s_flags.  Note that these mirror the equivalent MS_* flags where
// represented in both.
// Mount read-only
// Update the on-disk [acm]times lazily
#[repr(C)]
#[derive(Clone, Copy)]
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qstr {
    __slate_anon_0: __SlateRecord_dcache_h_50,
    name: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
}


// 192 bytes
#[repr(C)]
#[derive(Clone, Copy)]
union shortname_store {
    string: [u8; 40],
    words: [u64; 5],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct export_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_sb_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstatfs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mount {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mtd_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_dev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dentry {
    d_flags: u32,
    d_seq: seqcount_spinlock,
    d_hash: hlist_bl_node,
    d_parent: *mut dentry,
    __slate_anon_4: __SlateRecord_dcache_h_99,
    d_inode: *mut inode,
    d_shortname: shortname_store,
    d_op: *const dentry_operations,
    d_sb: *mut super_block,
    d_time: u64,
    d_fsdata: *mut (),
    d_lockref: lockref,
    d_lru: list_head,
    d_sib: hlist_node,
    d_children: hlist_head,
    __slate_anon_15: __SlateRecord_dcache_h_126,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct dentry_operations {
    d_revalidate: Option<unsafe extern "C" fn(*mut inode, *const qstr, *mut dentry, u32) -> i32>,
    d_weak_revalidate: Option<unsafe extern "C" fn(*mut dentry, u32) -> i32>,
    d_hash: Option<unsafe extern "C" fn(*const dentry, *mut qstr) -> i32>,
    d_compare: Option<unsafe extern "C" fn(*const dentry, u32, *const u8, *const qstr) -> i32>,
    d_delete: Option<unsafe extern "C" fn(*const dentry) -> i32>,
    d_init: Option<unsafe extern "C" fn(*mut dentry) -> i32>,
    d_release: Option<unsafe extern "C" fn(*mut dentry)>,
    d_prune: Option<unsafe extern "C" fn(*mut dentry)>,
    d_iput: Option<unsafe extern "C" fn(*mut dentry, *mut inode)>,
    d_dname: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> *mut u8>,
    d_automount: Option<unsafe extern "C" fn(*mut path) -> *mut vfsmount>,
    d_manage: Option<unsafe extern "C" fn(*const path, bool) -> i32>,
    d_real: Option<unsafe extern "C" fn(*mut dentry, u32) -> *mut dentry>,
    d_unalias_trylock: Option<unsafe extern "C" fn(*const dentry) -> bool>,
    d_unalias_unlock: Option<unsafe extern "C" fn(*const dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_lockref_h_26 {
    lock_count: u64,
    __slate_anon_1: __SlateRecord_lockref_h_30,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_lockref_h_30 {
    lock: spinlock,
    count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct semaphore {
    lock: raw_spinlock,
    count: u32,
    first_waiter: *mut semaphore_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_50 {
    __slate_anon_0: __SlateRecord_dcache_h_51,
    hash_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dcache_h_51 {
    hash: u32,
    len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_99 {
    __d_name: qstr,
    d_name: qstr,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_126 {
    d_alias: hlist_node,
    d_in_lookup_hash: hlist_bl_node,
    d_rcu: callback_head,
    waiters: *mut completion_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct semaphore_waiter {
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_386() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(mnt as *const vfsmount)).mnt_idmap) as *const *mut mnt_idmap) };
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits558U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
}


// no-op
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
#[inline(always)]
fn kmalloc_type(flags: u32, token: __SlateRecord_slab_h_532) -> u32 {
// The most common case is KMALLOC_NORMAL, so test for it
// with a single branch for all the relevant flags.
if likely!(flags & ((BIT!(4i32) as u32) | if IS_ENABLED__2!() != 0 { BIT!(0i32) as u32 } else { 0 } | if IS_ENABLED!() != 0 { BIT!(22i32) as u32 } else { 0 }) == 0) != 0 {
return 0;
}
// At least one of the flags has to be set. Their priorities in
// decreasing order are:
//  1) __GFP_DMA
//  2) __GFP_RECLAIMABLE
//  3) __GFP_ACCOUNT
if IS_ENABLED__2!() != 0 && flags & (BIT!(0i32) as u32) != 0 {
return 2;
}
if IS_ENABLED!() == 0 || flags & (BIT!(4i32) as u32) != 0 {
return 1;
} else {
return 0;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


#[inline(always)]
fn __kmalloc_index(size: u64, size_is_constant: bool) -> u32 {
if size == 0 {
return 0;
}
if size <= 8u64 {
return 3;
}
if 8i32 <= 32 && size > 64 && size <= 96 {
return 1;
}
if 8i32 <= 64 && size > 128 && size <= 192 {
return 2;
}
if size <= 8 {
return 3;
}
if size <= 16 {
return 4;
}
if size <= 32 {
return 5;
}
if size <= 64 {
return 6;
}
if size <= 128 {
return 7;
}
if size <= 256 {
return 8;
}
if size <= 512 {
return 9;
}
if size <= 1024 {
return 10;
}
if size <= 2048u64 {
return 11;
}
if size <= 4096u64 {
return 12;
}
if size <= 8192u64 {
return 13;
}
if size <= 16384u64 {
return 14;
}
if size <= 32768u64 {
return 15;
}
if size <= 65536u64 {
return 16;
}
if size <= 131072u64 {
return 17;
}
if size <= 262144u64 {
return 18;
}
if size <= 524288u64 {
return 19;
}
if size <= 1048576u64 {
return 20;
}
if size <= 2097152u64 {
return 21;
}
if IS_ENABLED!() == 0 && size_is_constant {
loop {
if true {
unsafe { __compiletime_assert_392() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("393: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 393b - ., 3; .popsection\n# {0}", const 393 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___kmalloc_index_1, sym __slate_asm_string___kmalloc_index_2, const 834 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
// Will never be reached. Needed because the compiler may complain
1i32.wrapping_neg() as u32
}

#[inline(always)]
fn _kmalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
if (unsafe { __slate_intrinsic_f217c4720f04f6db(size) }) && size != 0 {
let mut index: u32 = 0;
if size > 8192u64 {
return unsafe { __kmalloc_large_noprof(size, flags) };
}
index = kmalloc_index!(size);
return unsafe { __kmalloc_cache_noprof(kmalloc_caches[(kmalloc_type(flags, token) as u32) as usize][index as usize], flags, size) };
}
unsafe { __kmalloc_noprof(PASS_TOKEN_PARAMS!(size), flags) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_mark_connector {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parameter_spec {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_kattr {
}


// __alloc_flex - Allocate an object that has a trailing flexible array
// @KMALLOC: kmalloc wrapper function to use for allocation.
// @GFP: GFP flags for the allocation.
// @TYPE: type of structure to allocate space for.
// @FAM: The name of the flexible array member of @TYPE structure.
// @COUNT: how many @FAM elements to allocate space for.
//
// Returns: Newly allocated pointer to @TYPE with @COUNT-many trailing
// @FAM elements, or NULL on failure or if @COUNT cannot be represented
// by the member of @TYPE that counts the @FAM elements (annotated via
// __counted_by()).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_272 {
    ia_uid: __SlateRecord_uidgid_types_h_9,
    ia_vfsuid: __SlateRecord_mnt_idmapping_h_17,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_276 {
    ia_gid: __SlateRecord_uidgid_types_h_13,
    ia_vfsgid: __SlateRecord_mnt_idmapping_h_21,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


// File is embedded in backing_file object
// File is capable of returning -EAGAIN if I/O will block
// The two FMODE_NONOTIFY* define which fsnotify events should not be generated
// for an open file. These are the possible values of
// (f->f_mode & FMODE_FSNOTIFY_MASK) and their meaning:
//
// FMODE_NONOTIFY - suppress all (incl. non-permission) events.
// FMODE_NONOTIFY_PERM - suppress permission (incl. pre-content) events.
// FMODE_NONOTIFY | FMODE_NONOTIFY_PERM - suppress only FAN_ACCESS_PERM.
// Whiteout is represented by a char device.  The following constants define the
// mode and device number to use.
#[repr(C)]
#[derive(Clone, Copy)]
struct iattr {
    ia_valid: u32,
    ia_mode: u16,
    __slate_anon_2: __SlateRecord_fs_h_272,
    __slate_anon_3: __SlateRecord_fs_h_276,
    ia_size: i64,
    ia_atime: timespec64,
    ia_mtime: timespec64,
    ia_ctime: timespec64,
    ia_file: *mut file,
}


// non-RWF related bits - start at 16
#[repr(C)]
#[derive(Clone, Copy)]
struct kiocb {
    ki_filp: *mut file,
    ki_pos: i64,
    ki_complete: Option<unsafe extern "C" fn(*mut kiocb, i64)>,
    private: *mut (),
    ki_flags: i32,
    ki_ioprio: u16,
    ki_write_stream: u8,
    ki_waitq: *mut wait_page_queue,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_790 {
    i_nlink: u32,
    __i_nlink: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_834 {
    i_dentry: hlist_head,
    i_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_846 {
    i_fop: *const file_operations,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_852 {
    i_devices: list_head,
    i_linklen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_856 {
    i_pipe: *mut pipe_inode_info,
    i_cdev: *mut cdev,
    i_link: *mut u8,
    i_dir_seq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space_operations {
    read_folio: Option<unsafe extern "C" fn(*mut file, *mut folio) -> i32>,
    writepages: Option<unsafe extern "C" fn(*mut address_space, *mut writeback_control) -> i32>,
    dirty_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> bool>,
    readahead: Option<unsafe extern "C" fn(*mut readahead_control)>,
    write_begin: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, *mut *mut folio, *mut *mut ()) -> i32>,
    write_end: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, u32, *mut folio, *mut ()) -> i32>,
    bmap: Option<unsafe extern "C" fn(*mut address_space, u64) -> u64>,
    invalidate_folio: Option<unsafe extern "C" fn(*mut folio, u64, u64)>,
    release_folio: Option<unsafe extern "C" fn(*mut folio, u32) -> bool>,
    free_folio: Option<unsafe extern "C" fn(*mut folio)>,
    direct_IO: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    migrate_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio, *mut folio, u32) -> i32>,
    launder_folio: Option<unsafe extern "C" fn(*mut folio) -> i32>,
    is_partially_uptodate: Option<unsafe extern "C" fn(*mut folio, u64, u64) -> bool>,
    is_dirty_writeback: Option<unsafe extern "C" fn(*mut folio, *mut bool, *mut bool)>,
    error_remove_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> i32>,
    swap_activate: Option<unsafe extern "C" fn(*mut swap_info_struct, *mut file, *mut u64) -> i32>,
    swap_deactivate: Option<unsafe extern "C" fn(*mut file)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space {
    host: *mut inode,
    i_pages: xarray,
    invalidate_lock: rw_semaphore,
    gfp_mask: u32,
    i_mmap_writable: __SlateRecord_types_h_190,
    i_mmap: rb_root_cached,
    nrpages: u64,
    writeback_index: u64,
    a_ops: *const address_space_operations,
    flags: u64,
    wb_err: u32,
    i_private_lock: spinlock,
    i_mmap_rwsem: rw_semaphore,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1267 {
    f_path: path,
    __f_path: path,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1271 {
    f_pos_lock: mutex,
    f_pipe: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1287 {
    f_task_work: callback_head,
    f_llist: llist_node,
    f_ra: file_ra_state,
    f_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lease {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode {
    i_mode: u16,
    i_opflags: u16,
    i_flags: u32,
    i_acl: *mut posix_acl,
    i_default_acl: *mut posix_acl,
    i_uid: __SlateRecord_uidgid_types_h_9,
    i_gid: __SlateRecord_uidgid_types_h_13,
    i_op: *const inode_operations,
    i_sb: *mut super_block,
    i_mapping: *mut address_space,
    i_security: *mut (),
    i_ino: u64,
    __slate_anon_12: __SlateRecord_fs_h_790,
    i_rdev: u32,
    i_size: i64,
    i_atime_sec: i64,
    i_mtime_sec: i64,
    i_ctime_sec: i64,
    i_atime_nsec: u32,
    i_mtime_nsec: u32,
    i_ctime_nsec: u32,
    i_generation: u32,
    i_lock: spinlock,
    i_bytes: u16,
    i_blkbits: u8,
    i_write_hint: u8,
    i_blocks: u64,
    i_state: inode_state_flags,
    i_rwsem: rw_semaphore,
    dirtied_when: u64,
    dirtied_time_when: u64,
    i_hash: hlist_node,
    i_io_list: list_head,
    i_lru: list_head,
    i_sb_list: list_head,
    i_wb_list: list_head,
    __slate_anon_36: __SlateRecord_fs_h_834,
    i_version: __SlateRecord_types_h_197,
    i_sequence: __SlateRecord_types_h_197,
    i_count: __SlateRecord_types_h_190,
    i_dio_count: __SlateRecord_types_h_190,
    i_writecount: __SlateRecord_types_h_190,
    i_readcount: __SlateRecord_types_h_190,
    __slate_anon_43: __SlateRecord_fs_h_846,
    i_flctx: *mut file_lock_context,
    i_data: address_space,
    __slate_anon_46: __SlateRecord_fs_h_852,
    __slate_anon_47: __SlateRecord_fs_h_856,
    i_fsnotify_mask: u32,
    i_fsnotify_marks: *mut fsnotify_mark_connector,
    i_private: *mut (),
}


// Get bit address from inode->i_state to use with wait_var_event()
// infrastructre.
#[repr(C)]
#[derive(Clone, Copy)]
struct fown_struct {
    file: *mut file,
    lock: rwlock,
    pid: *mut pid,
    pid_type: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    euid: __SlateRecord_uidgid_types_h_9,
    signum: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_ra_state {
    start: u64,
    size: u32,
    async_size: u32,
    ra_pages: u32,
    order: u16,
    mmap_miss: u16,
    prev_pos: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file {
    f_lock: spinlock,
    f_mode: u32,
    f_op: *const file_operations,
    f_mapping: *mut address_space,
    private_data: *mut (),
    f_inode: *mut inode,
    f_flags: u32,
    f_iocb_flags: u32,
    f_cred: *const cred,
    f_owner: *mut fown_struct,
    __slate_anon_10: __SlateRecord_fs_h_1267,
    __slate_anon_11: __SlateRecord_fs_h_1271,
    f_pos: i64,
    f_security: *mut (),
    f_wb_err: u32,
    f_sb_err: u32,
    f_ep: *mut hlist_head,
    __slate_anon_17: __SlateRecord_fs_h_1287,
    f_ref: __SlateRecord_file_ref_h_52,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fasync_struct {
    fa_lock: rwlock,
    magic: i32,
    fa_fd: i32,
    fa_next: *mut fasync_struct,
    fa_file: *mut file,
    fa_rcu: callback_head,
}


// Multigrain timestamps
//
// Conditionally use fine-grained ctime and mtime timestamps when there
// are users actively observing them via getattr. The primary use-case
// for this is NFS clients that use the ctime to distinguish between
// different states of the file, and that are often fooled by multiple
// operations that occur in the same coarse-grained timer tick.
#[repr(C)]
#[derive(Clone, Copy)]
struct dir_context {
    actor: Option<unsafe extern "C" fn(*mut dir_context, *const u8, i32, i64, u64, u32) -> bool>,
    pos: i64,
    count: i32,
    dt_flags_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_operations {
    owner: *mut module,
    fop_flags: u32,
    llseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
    read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    read_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    write_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    iopoll: Option<unsafe extern "C" fn(*mut kiocb, *mut io_comp_batch, u32) -> i32>,
    iterate_shared: Option<unsafe extern "C" fn(*mut file, *mut dir_context) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
    unlocked_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut vm_area_struct) -> i32>,
    open: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    flush: Option<unsafe extern "C" fn(*mut file, *mut ()) -> i32>,
    release: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    fsync: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    fasync: Option<unsafe extern "C" fn(i32, *mut file, i32) -> i32>,
    lock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
    check_flags: Option<unsafe extern "C" fn(i32) -> i32>,
    flock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    splice_write: Option<unsafe extern "C" fn(*mut pipe_inode_info, *mut file, *mut i64, u64, u32) -> i64>,
    splice_read: Option<unsafe extern "C" fn(*mut file, *mut i64, *mut pipe_inode_info, u64, u32) -> i64>,
    splice_eof: Option<unsafe extern "C" fn(*mut file)>,
    setlease: Option<unsafe extern "C" fn(*mut file, i32, *mut *mut file_lease, *mut *mut ()) -> i32>,
    fallocate: Option<unsafe extern "C" fn(*mut file, i32, i64, i64) -> i64>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut seq_file, *mut file)>,
    copy_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, u64, u32) -> i64>,
    remap_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, i64, u32) -> i64>,
    fadvise: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    uring_cmd: Option<unsafe extern "C" fn(*mut io_uring_cmd, u32) -> i32>,
    uring_cmd_iopoll: Option<unsafe extern "C" fn(*mut io_uring_cmd, *mut io_comp_batch, u32) -> i32>,
    mmap_prepare: Option<unsafe extern "C" fn(*mut vm_area_desc) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


// File system supports uncached read/write buffered IO
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct inode_operations {
    lookup: Option<unsafe extern "C" fn(*mut inode, *mut dentry, u32) -> *mut dentry>,
    get_link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut delayed_call) -> *const u8>,
    permission: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, i32) -> i32>,
    get_inode_acl: Option<unsafe extern "C" fn(*mut inode, i32, bool) -> *mut posix_acl>,
    readlink: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> i32>,
    create: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> i32>,
    link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut dentry) -> i32>,
    unlink: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    symlink: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *const u8) -> i32>,
    mkdir: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> *mut dentry>,
    rmdir: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    mknod: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16, u32) -> i32>,
    rename: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *mut inode, *mut dentry, u32) -> i32>,
    setattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut iattr) -> i32>,
    getattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *const path, *mut kstat, u32, u32) -> i32>,
    listxattr: Option<unsafe extern "C" fn(*mut dentry, *mut u8, u64) -> i64>,
    fiemap: Option<unsafe extern "C" fn(*mut inode, *mut fiemap_extent_info, u64, u64) -> i32>,
    update_time: Option<unsafe extern "C" fn(*mut inode, u32, u32) -> i32>,
    sync_lazytime: Option<unsafe extern "C" fn(*mut inode)>,
    atomic_open: Option<unsafe extern "C" fn(*mut inode, *mut dentry, *mut file, u32, u16) -> i32>,
    tmpfile: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut file, u16) -> i32>,
    get_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, i32) -> *mut posix_acl>,
    set_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut posix_acl, i32) -> i32>,
    fileattr_set: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut file_kattr) -> i32>,
    fileattr_get: Option<unsafe extern "C" fn(*mut dentry, *mut file_kattr) -> i32>,
    get_offset_ctx: Option<unsafe extern "C" fn(*mut inode) -> *mut offset_ctx>,
}


// no suid or xattr security attributes
// Note that nosuid etc flags are inode-specific: setting some file-system
// flags just means all the inodes inherit those flags by default. It might be
// possible to override it selectively if you really wanted to with some
// ioctl() that is not currently implemented.
//
// Exception: SB_RDONLY is always applied to the entire file system.
//
// Unfortunately, it is possible to change a filesystems flags with it mounted
// with files in use.  This means that all of the inodes will not have their
// i_flags updated.  Hence, i_flags no longer inherit the superblock mount
// flags, so these have to be checked separately. -- rmk@arm.uk.linux.org
#[repr(C)]
#[derive(Clone, Copy)]
struct file_system_type {
    name: *const u8,
    fs_flags: i32,
    init_fs_context: Option<unsafe extern "C" fn(*mut fs_context) -> i32>,
    parameters: *const fs_parameter_spec,
    kill_sb: Option<unsafe extern "C" fn(*mut super_block)>,
    owner: *mut module,
    list: hlist_node,
    fs_supers: hlist_head,
    s_lock_key: lock_class_key,
    s_umount_key: lock_class_key,
    s_vfs_rename_key: lock_class_key,
    s_writers_key: [lock_class_key; 3],
    i_lock_key: lock_class_key,
    i_mutex_key: lock_class_key,
    invalidate_lock_key: lock_class_key,
    i_mutex_dir_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list {
    next: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vdso_image {
    data: *mut (),
    size: u64,
    alt: u64,
    alt_len: u64,
    extable_base: u64,
    extable_len: u64,
    extable: *const (),
    sym___kernel_sigreturn: i64,
    sym___kernel_rt_sigreturn: i64,
    sym___kernel_vsyscall: i64,
    sym_int80_landing_pad: i64,
    sym_vdso32_sigreturn_landing_pad: i64,
    sym_vdso32_rt_sigreturn_landing_pad: i64,
    sym___futex_list64_try_unlock_cs_start: i64,
    sym___futex_list64_try_unlock_cs_end: i64,
    sym___futex_list32_try_unlock_cs_start: i64,
    sym___futex_list32_try_unlock_cs_end: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elf64_sym {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
}


// Set the IDR flag and the IDR_FREE tag
#[repr(C)]
#[derive(Clone, Copy)]
struct ida {
    xa: xarray,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_107 {
    is_visible: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, i32) -> u16>,
    is_visible_const: Option<unsafe extern "C" fn(*mut kobject, *const attribute, i32) -> u16>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_118 {
    attrs: *mut *mut attribute,
    attrs_const: *const *const attribute,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_attr {
    ops: *const kernfs_ops,
    open: *mut kernfs_open_node,
    size: i64,
    notify_next: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_node {
    count: __SlateRecord_types_h_190,
    active: __SlateRecord_types_h_190,
    __parent: *mut kernfs_node,
    name: *const u8,
    rb: rb_node,
    ns: *const ns_common,
    hash: u32,
    flags: u16,
    mode: u16,
    __slate_anon_9: __SlateRecord_kernfs_h_221,
    id: u64,
    r#priv: *mut (),
    iattr: *mut kernfs_iattrs,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_file {
    kn: *mut kernfs_node,
    file: *mut file,
    seq_file: *mut seq_file,
    r#priv: *mut (),
    mutex: mutex,
    prealloc_mutex: mutex,
    event: i32,
    list: list_head,
    prealloc_buf: *mut u8,
    atomic_write_len: u64,
    __slate_bits_0: __slate_bits::__SlateBits627U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_ops {
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    read: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    atomic_write_len: u64,
    prealloc: bool,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    mmap: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut vm_area_struct) -> i32>,
    llseek: Option<unsafe extern "C" fn(*mut kernfs_open_file, i64, i32) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_ns_type_operations {
    r#type: u32,
    current_may_mount: Option<unsafe extern "C" fn() -> bool>,
    grab_current_ns: Option<unsafe extern "C" fn() -> *mut ns_common>,
    netlink_ns: Option<unsafe extern "C" fn(*mut sock) -> *const ns_common>,
    initial_ns: Option<unsafe extern "C" fn() -> *const ns_common>,
    drop_ns: Option<unsafe extern "C" fn(*mut ns_common)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute {
    name: *const u8,
    mode: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute_group {
    name: *const u8,
    __slate_anon_1: __SlateRecord_sysfs_h_107,
    is_bin_visible: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u16>,
    bin_size: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u64>,
    __slate_anon_4: __SlateRecord_sysfs_h_118,
    bin_attrs: *const *const bin_attribute,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bin_attribute {
    attr: attribute,
    size: u64,
    private: *mut (),
    f_mapping: Option<unsafe extern "C" fn() -> *mut address_space>,
    read: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, i64, i32) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut vm_area_struct) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
}


// buffer for the variables
#[repr(C)]
#[derive(Clone, Copy)]
struct kobject {
    name: *const u8,
    entry: list_head,
    parent: *mut kobject,
    kset: *mut kset,
    ktype: *const kobj_type,
    sd: *mut kernfs_node,
    kref: kref,
    __slate_bits_0: __slate_bits::__SlateBits569U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_type {
    release: Option<unsafe extern "C" fn(*mut kobject)>,
    sysfs_ops: *const sysfs_ops,
    default_groups: *mut *const attribute_group,
    child_ns_type: Option<unsafe extern "C" fn(*const kobject) -> *const kobj_ns_type_operations>,
    namespace: Option<unsafe extern "C" fn(*const kobject) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const kobject, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_uevent_env {
    argv: [*mut u8; 3],
    envp: [*mut u8; 64],
    envp_idx: i32,
    buf: [u8; 2048],
    buflen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset_uevent_ops {
    filter: Option<unsafe extern "C" fn(*const kobject) -> i32>,
    name: Option<unsafe extern "C" fn(*const kobject) -> *const u8>,
    uevent: Option<unsafe extern "C" fn(*const kobject, *mut kobj_uevent_env) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param_ops {
    flags: u32,
    set: Option<unsafe extern "C" fn(*const u8, *const kernel_param) -> i32>,
    get: Option<unsafe extern "C" fn(*mut u8, *const kernel_param) -> i32>,
    free: Option<unsafe extern "C" fn(*mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param {
    name: *const u8,
    r#mod: *mut module,
    ops: *const kernel_param_ops,
    perm: u16,
    level: i8,
    flags: u8,
    __slate_anon_6: __SlateRecord_moduleparam_h_89,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_symbol {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_sect_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_notes_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_call {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_string {
    maxlen: u32,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_array {
    max: u32,
    elemsize: u32,
    num: *mut u32,
    ops: *const kernel_param_ops,
    elem: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct latch_tree_node {
    node: [rb_node; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct its_array {
    pages: *mut *mut (),
    num: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mod_arch_specific {
    num_orcs: u32,
    orc_unwind_ip: *mut i32,
    orc_unwind: *mut orc_entry,
    its_pages: its_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_kobject {
    kobj: kobject,
    r#mod: *mut module,
    drivers_dir: *mut kobject,
    mp: *mut module_param_attrs,
    kobj_completion: *mut completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_attribute {
    attr: attribute,
    show: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *const u8, u64) -> i64>,
    setup: Option<unsafe extern "C" fn(*mut module, *const u8)>,
    test: Option<unsafe extern "C" fn(*mut module) -> i32>,
    free: Option<unsafe extern "C" fn(*mut module)>,
}


// module_init() - driver initialization entry point
// @x: function to be run at kernel boot time or module insertion
//
// module_init() will either be called during do_initcalls() (if
// builtin) or at module insertion time (if a module).  There can only
// be one per module.
// module_exit() - driver exit entry point
// @x: function to be run when driver is removed
//
// module_exit() will wrap the driver clean-up code
// with cleanup_module() when used with rmmod when
// the driver is a module.  If the driver is statically
// compiled into the kernel, module_exit() has no effect.
// There can only be one per module.
// The following license idents are currently accepted as indicating free
// software modules
//
// 	"GPL"				[GNU Public License v2]
// 	"GPL v2"			[GNU Public License v2]
// 	"GPL and additional rights"	[GNU Public License v2 rights and more]
// 	"Dual BSD/GPL"			[GNU Public License v2
// 					 or BSD license choice]
// 	"Dual MIT/GPL"			[GNU Public License v2
// 					 or MIT license choice]
// 	"Dual MPL/GPL"			[GNU Public License v2
// 					 or Mozilla license choice]
//
// The following other idents are available
//
// 	"Proprietary"			[Non free products]
//
// Both "GPL v2" and "GPL" (the latter also in dual licensed strings) are
// merely stating that the module is licensed under the GPL v2, but are not
// telling whether "GPL v2 only" or "GPL v2 or later". The reason why there
// are two variants is a historic and failed attempt to convey more
// information in the MODULE_LICENSE string. For module loading the
// "only/or later" distinction is completely irrelevant and does neither
// replace the proper license identifiers in the corresponding source file
// nor amends them in any way. The sole purpose is to make the
// 'Proprietary' flagging work and to refuse to bind symbols which are
// exported with EXPORT_SYMBOL_GPL when a non free module is loaded.
//
// In the same way "BSD" is not a clear license information. It merely
// states, that the module is licensed under one of the compatible BSD
// license variants. The detailed and correct license information is again
// to be found in the corresponding source files.
//
// There are dual licensed components, but when running with Linux it is the
// GPL that is relevant so this is a non issue. Similarly LGPL linked with GPL
// is a GPL combined work.
//
// This exists for several reasons
// 1.	So modinfo can show license info for users wanting to vet their setup
// 	is free
// 2.	So the community can ignore bug reports including proprietary modules
// 3.	So vendors can do likewise based on their own policies
// What your module does.
// Format: __mod_device_table__kmod_<modname>__<type>__<name>
// Parts of the string `__kmod_` and `__` are used as delimiters when parsing
// a symbol in file2alias.c
// Creates an alias so file2alias.c can find device table.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_tree_node {
    r#mod: *mut module,
    node: latch_tree_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_memory {
    base: *mut (),
    is_rox: bool,
    size: u32,
    mtn: mod_tree_node,
}


// Only touch one cacheline for common rbtree-for-core-layout case.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_kallsyms {
    symtab: *mut elf64_sym,
    num_symtab: u32,
    strtab: *mut u8,
    typetab: *mut u8,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct module {
    state: u32,
    list: list_head,
    name: [u8; 56],
    mkobj: module_kobject,
    modinfo_attrs: *mut module_attribute,
    version: *const u8,
    srcversion: *const u8,
    imported_namespaces: *const u8,
    holders_dir: *mut kobject,
    syms: *const kernel_symbol,
    crcs: *const u32,
    flagstab: *const u8,
    num_syms: u32,
    param_lock: mutex,
    kp: *mut kernel_param,
    num_kp: u32,
    using_gplonly_symbols: bool,
    async_probe_requested: bool,
    num_exentries: u32,
    extable: *mut exception_table_entry,
    init: Option<unsafe extern "C" fn() -> i32>,
    __slate_pad_21: [u8; 8],
    mem: [module_memory; 7],
    arch: mod_arch_specific,
    taints: u64,
    num_bugs: u32,
    bug_list: list_head,
    bug_table: *mut bug_entry,
    kallsyms: *mut mod_kallsyms,
    core_kallsyms: mod_kallsyms,
    sect_attrs: *mut module_sect_attrs,
    notes_attrs: *mut module_notes_attrs,
    percpu: *mut (),
    percpu_size: u32,
    noinstr_text_start: *mut (),
    noinstr_text_size: u32,
    num_tracepoints: u32,
    tracepoints_ptrs: *const i32,
    num_srcu_structs: u32,
    srcu_struct_ptrs: *mut *mut srcu_struct,
    jump_entries: *mut jump_entry,
    num_jump_entries: u32,
    num_trace_bprintk_fmt: u32,
    trace_bprintk_fmt_start: *mut *const u8,
    trace_events: *mut *mut trace_event_call,
    num_trace_events: u32,
    trace_evals: *mut *mut trace_eval_map,
    num_trace_evals: u32,
    kprobes_text_start: *mut (),
    kprobes_text_size: u32,
    kprobe_blacklist: *mut u64,
    num_kprobe_blacklist: u32,
    num_static_call_sites: i32,
    static_call_sites: *mut static_call_site,
    source_list: list_head,
    target_list: list_head,
    exit: Option<unsafe extern "C" fn()>,
    refcnt: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_device_id {
    vendor: u32,
    device: u32,
    subvendor: u32,
    subdevice: u32,
    class: u32,
    class_mask: u32,
    driver_data: u64,
    override_only: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct resource {
    start: u64,
    end: u64,
    name: *const u8,
    flags: u64,
    desc: u64,
    parent: *mut resource,
    sibling: *mut resource,
    child: *mut resource,
}


// Resource type
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
fn resource_size(res: *const resource) -> u64 {
unsafe { (*res).end }.wrapping_sub(unsafe { (*res).start }).wrapping_add(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_iommu {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_ops {
    prepare: Option<unsafe extern "C" fn(*mut device) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_idle: Option<unsafe extern "C" fn(*mut device) -> i32>,
}


// Deprecated. Use DEFINE_SIMPLE_DEV_PM_OPS() instead.
#[repr(C)]
#[derive(Clone, Copy)]
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits674U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits674U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits674U2,
    links_count: u32,
    request: u32,
    runtime_status: i32,
    last_status: i32,
    runtime_error: i32,
    autosuspend_delay: i32,
    last_busy: u64,
    active_time: u64,
    suspended_time: u64,
    accounting_timestamp: u64,
    subsys_data: *mut pm_subsys_data,
    set_latency_tolerance: Option<unsafe extern "C" fn(*mut device, i32)>,
    qos: *mut dev_pm_qos,
    __slate_bits_3: __slate_bits::__SlateBits674U3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_domain {
    ops: dev_pm_ops,
    start: Option<unsafe extern "C" fn(*mut device) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut device, bool)>,
    activate: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync: Option<unsafe extern "C" fn(*mut device)>,
    dismiss: Option<unsafe extern "C" fn(*mut device)>,
    set_performance_state: Option<unsafe extern "C" fn(*mut device, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_device_h_719 {
    name: *const u8,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_dma_region {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_tlb_mem {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_type {
    name: *const u8,
    dev_name: *const u8,
    bus_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    drv_groups: *const *const attribute_group,
    r#match: Option<unsafe extern "C" fn(*mut device, *const device_driver) -> i32>,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device)>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    irq_get_affinity: Option<unsafe extern "C" fn(*mut device, u32) -> *const cpumask>,
    online: Option<unsafe extern "C" fn(*mut device) -> i32>,
    offline: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    num_vf: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_configure: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_cleanup: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
    driver_override: bool,
    need_parent_lock: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct class {
    name: *const u8,
    class_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    dev_uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16) -> *mut u8>,
    class_release: Option<unsafe extern "C" fn(*const class)>,
    dev_release: Option<unsafe extern "C" fn(*mut device)>,
    shutdown_pre: Option<unsafe extern "C" fn(*mut device) -> i32>,
    ns_type: *const kobj_ns_type_operations,
    namespace: Option<unsafe extern "C" fn(*const device) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const device, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    pm: *const dev_pm_ops,
}


fn devm_kzalloc(dev: *mut device, size: u64, gfp: u32) -> *mut () {
unsafe { devm_kmalloc(dev, size, gfp | (BIT!(8i32) as u32)) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_id {
    id: [u8; 16],
    driver_data: u64,
    cls: u32,
    cls_msk: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct of_device_id {
    name: [u8; 32],
    r#type: [u8; 32],
    compatible: [u8; 128],
    data: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hotplug_slot {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_driver {
    name: *const u8,
    bus: *const bus_type,
    owner: *mut module,
    mod_name: *const u8,
    suppress_bind_attrs: bool,
    probe_type: u32,
    of_match_table: *const of_device_id,
    acpi_match_table: *const acpi_device_id,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    pm: *const dev_pm_ops,
    coredump: Option<unsafe extern "C" fn(*mut device)>,
    p: *mut driver_private,
    p_cb: __SlateRecord_driver_h_126,
}


// module_driver() - Helper macro for drivers that don't do anything
// special in module init/exit. This eliminates a lot of boilerplate.
// Each module may only use this macro once, and calling it replaces
// module_init() and module_exit().
//
// @__driver: driver name
// @__register: register function for this driver type
// @__unregister: unregister function for this driver type
// @...: Additional arguments to be passed to __register and __unregister.
//
// Use this macro to construct bus specific macros for registering
// drivers, and do not use it on its own.
#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_bwctrl_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_link_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sriov {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcec_ea {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_archdata {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_type {
    name: *const u8,
    groups: *mut *const attribute_group,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13) -> *mut u8>,
    release: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_links_info {
    suppliers: list_head,
    consumers: list_head,
    defer_sync: list_head,
    status: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_msi_info {
    domain: *mut irq_domain,
    data: *mut msi_device_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_physical_location {
    panel: u32,
    vertical_position: u32,
    horizontal_position: u32,
    dock: bool,
    lid: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_memremap_h_143 {
    range: range,
    __slate_anon_1: __SlateRecord_memremap_h_145,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145 {
    __empty_ranges: __SlateRecord_memremap_h_145_2,
    ranges: [range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_h_754 {
    vma: *mut vm_area_struct,
    gfp_mask: u32,
    pgoff: u64,
    address: u64,
    real_address: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_h_768 {
    orig_pte: __SlateRecord_pgtable_64_types_h_21,
    orig_pmd: __SlateRecord_pgtable_64_types_h_22,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device {
    kobj: kobject,
    parent: *mut device,
    p: *mut device_private,
    init_name: *const u8,
    r#type: *const device_type,
    bus: *const bus_type,
    driver: *mut device_driver,
    platform_data: *mut (),
    driver_data: *mut (),
    driver_override: __SlateRecord_device_h_719,
    mutex: mutex,
    links: dev_links_info,
    power: dev_pm_info,
    pm_domain: *mut dev_pm_domain,
    msi: dev_msi_info,
    dma_mask: *mut u64,
    coherent_dma_mask: u64,
    bus_dma_limit: u64,
    dma_range_map: *const bus_dma_region,
    dma_parms: *mut device_dma_parameters,
    dma_pools: list_head,
    dma_io_tlb_mem: *mut io_tlb_mem,
    archdata: dev_archdata,
    of_node: *mut device_node,
    fwnode: *mut fwnode_handle,
    numa_node: i32,
    devt: u32,
    id: u32,
    devres_lock: spinlock,
    devres_head: list_head,
    class: *const class,
    groups: *mut *const attribute_group,
    release: Option<unsafe extern "C" fn(*mut device)>,
    iommu_group: *mut iommu_group,
    iommu: *mut dev_iommu,
    physical_location: *mut device_physical_location,
    removable: u32,
    flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wakeup_source {
    name: *const u8,
    id: i32,
    entry: list_head,
    lock: spinlock,
    wakeirq: *mut wake_irq,
    timer: timer_list,
    timer_expires: u64,
    total_time: i64,
    max_time: i64,
    last_time: i64,
    start_prevent_time: i64,
    prevent_sleep_time: i64,
    event_count: u64,
    active_count: u64,
    relax_count: u64,
    expire_count: u64,
    wakeup_count: u64,
    dev: *mut device,
    __slate_bits_0: __slate_bits::__SlateBits671U0,
}


fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
}

fn dev_set_drvdata(dev: *mut device, data: *mut ()) {
unsafe {
(*dev).driver_data = data;
}
}

#[inline(always)]
fn __outb(mut value: u8, mut port: u16) {
unsafe {
core::arch::asm!("outb %al, %dx", in("ax") (value as u8) as u32, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[inline(always)]
fn __inb(mut port: u16) -> u8 {
let mut value: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("inb %dx, %al", lateout("ax") __t0, in("dx") port as u16, options(nostack, att_syntax));
value = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
value
}

#[inline(always)]
fn __outl(mut value: u32, mut port: u16) {
unsafe {
core::arch::asm!("outl %eax, %dx", in("ax") value as u32, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[inline(always)]
fn __inl(mut port: u16) -> u32 {
let mut value: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("inl %dx, %eax", lateout("ax") __t0, in("dx") port as u16, options(nostack, att_syntax));
value = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
value
}

fn readb(mut addr: *const ()) -> u8 {
let mut ret: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb ({1:r}),{0:l}", lateout(reg) __t0, in(reg) addr as *mut u8, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
ret
}

fn readw(mut addr: *const ()) -> u16 {
let mut ret: u16 = 0;
unsafe {
let __t0: u16;
core::arch::asm!("movw ({1:r}),{0:x}", lateout(reg) __t0, in(reg) addr as *mut u16, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u16, u16>(__t0 as u16) };
}
ret
}

fn readl(mut addr: *const ()) -> u32 {
let mut ret: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movl ({1:r}),{0:e}", lateout(reg) __t0, in(reg) addr as *mut u32, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
ret
}

fn writeb(mut val: u8, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movb {0:l},({1:r})", in(reg) (val as u8) as u32, in(reg) addr as *mut u8, options(nostack, att_syntax));
}
}

fn writew(mut val: u16, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movw {0:x},({1:r})", in(reg) val as u16, in(reg) addr as *mut u16, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_tty_buffer_h_11 {
    next: *mut tty_buffer,
    free: llist_node,
}


fn writel(mut val: u32, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movl {0:e},({1:r})", in(reg) val as u32, in(reg) addr as *mut u32, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
    secondary: *mut fwnode_handle,
    ops: *const fwnode_operations,
    dev: *mut device,
    suppliers: list_head,
    consumers: list_head,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_endpoint {
    port: u32,
    id: u32,
    local_fwnode: *const fwnode_handle,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_port_h_118 {
    __slate_anon_0: __SlateRecord_tty_port_h_118_2,
    buf: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_tty_port_h_118_2 {
    kfifo: __kfifo,
    r#type: *mut u8,
    const_type: *const u8,
    rectype: *mut [u8; 0],
    ptr: *mut u8,
    ptr_const: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_h_212 {
    lock: spinlock,
    stopped: bool,
    tco_stopped: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_h_218 {
    pgrp: *mut pid,
    session: *mut pid,
    lock: spinlock,
    pktstatus: u8,
    packet: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct printk_buffers {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_reference_args {
    fwnode: *mut fwnode_handle,
    nargs: u32,
    args: [u64; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_operations {
    get: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    put: Option<unsafe extern "C" fn(*mut fwnode_handle)>,
    device_is_available: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_match_data: Option<unsafe extern "C" fn(*const fwnode_handle, *const device) -> *const ()>,
    device_dma_supported: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_dma_attr: Option<unsafe extern "C" fn(*const fwnode_handle) -> u32>,
    property_present: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_bool: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_int_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, u32, *mut (), u64) -> i32>,
    property_read_string_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *mut *const u8, u64) -> i32>,
    get_name: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_name_prefix: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_parent: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    get_next_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    get_named_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> *mut fwnode_handle>,
    get_reference_args: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *const u8, u32, u32, *mut fwnode_reference_args) -> i32>,
    graph_get_next_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    graph_get_remote_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    graph_get_port_parent: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    graph_parse_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_endpoint) -> i32>,
    iomap: Option<unsafe extern "C" fn(*mut fwnode_handle, i32) -> *mut ()>,
    irq_get: Option<unsafe extern "C" fn(*const fwnode_handle, u32) -> i32>,
    add_links: Option<unsafe extern "C" fn(*mut fwnode_handle) -> i32>,
}


// The PCI interface treats multi-function devices as independent
// devices.  The slot/function address of each device is encoded
// in a single byte as follows:
//
// 	7:3 = slot
// 	2:0 = function
//
// PCI_DEVFN(), PCI_SLOT(), and PCI_FUNC() are defined in uapi/linux/pci.h.
// In the interest of not exposing interfaces to user-space unnecessarily,
// the following kernel-only defines are being added here.
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_slot {
    bus: *mut pci_bus,
    list: list_head,
    hotplug: *mut hotplug_slot,
    number: u16,
    __slate_bits_0: __slate_bits::__SlateBits719U0,
    kobj: kobject,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_serial_h_161 {
    padding: [u32; 5],
    __slate_anon_1: __SlateRecord_serial_h_164,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_serial_h_164 {
    addr_recv: u8,
    addr_dest: u8,
    padding0: [u8; 2],
    padding1: [u32; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_vpd {
    lock: mutex,
    len: u32,
    cap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev {
    bus_list: list_head,
    bus: *mut pci_bus,
    subordinate: *mut pci_bus,
    sysdata: *mut (),
    procent: *mut proc_dir_entry,
    slot: *mut pci_slot,
    devfn: u32,
    vendor: u16,
    device: u16,
    subsystem_vendor: u16,
    subsystem_device: u16,
    class: u32,
    revision: u8,
    hdr_type: u8,
    rcec_ea: *mut rcec_ea,
    rcec: *mut pci_dev,
    devcap: u32,
    rebar_cap: u16,
    pcie_cap: u8,
    msi_cap: u8,
    msix_cap: u8,
    __slate_bits_0: __slate_bits::__SlateBits714U0,
    rom_base_reg: u8,
    pin: u8,
    pcie_flags_reg: u16,
    dma_alias_mask: *mut u64,
    driver: *mut pci_driver,
    dma_mask: u64,
    msi_addr_mask: u64,
    dma_parms: device_dma_parameters,
    current_state: i32,
    pm_cap: u8,
    __slate_bits_1: __slate_bits::__SlateBits714U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits714U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits714U3,
    dev_flags: u16,
    enable_cnt: __SlateRecord_types_h_190,
    pcie_cap_lock: spinlock,
    saved_config_space: [u32; 16],
    saved_cap_space: hlist_head,
    msix_base: *mut (),
    msi_lock: raw_spinlock,
    vpd: pci_vpd,
    link_bwctrl: *mut pcie_bwctrl_data,
    __slate_anon_116: __SlateRecord_pci_h_547,
    ats_cap: u16,
    ats_stu: u8,
    pri_cap: u16,
    pri_reqs_alloc: u32,
    __slate_bits_4: __slate_bits::__SlateBits714U4,
    pasid_cap: u16,
    pasid_features: u16,
    acs_cap: u16,
    acs_capabilities: u16,
    supported_speeds: u8,
    rom: u64,
    romlen: u64,
    priv_flags: u64,
    reset_methods: [u8; 8],
    wake: *mut gpio_desc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_port_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_bus {
    node: list_head,
    parent: *mut pci_bus,
    children: list_head,
    devices: list_head,
    self_: *mut pci_dev,
    slots: list_head,
    resource: [*mut resource; 4],
    resources: list_head,
    busn_res: resource,
    ops: *mut pci_ops,
    sysdata: *mut (),
    procdir: *mut proc_dir_entry,
    number: u8,
    primary: u8,
    max_bus_speed: u8,
    cur_bus_speed: u8,
    name: [u8; 48],
    bridge_ctl: u16,
    bus_flags: u16,
    bridge: *mut device,
    dev: device,
    __slate_bits_0: __slate_bits::__SlateBits430U0,
}


// Error values that may be returned by PCI functions
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_ops {
    add_bus: Option<unsafe extern "C" fn(*mut pci_bus) -> i32>,
    remove_bus: Option<unsafe extern "C" fn(*mut pci_bus)>,
    map_bus: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32) -> *mut ()>,
    read: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, *mut u32) -> i32>,
    write: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_bus_region {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dynids {
    lock: spinlock,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_error_handlers {
    error_detected: Option<unsafe extern "C" fn(*mut pci_dev, u32) -> u32>,
    mmio_enabled: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    slot_reset: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    reset_prepare: Option<unsafe extern "C" fn(*mut pci_dev)>,
    reset_done: Option<unsafe extern "C" fn(*mut pci_dev)>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev)>,
    cor_error_detected: Option<unsafe extern "C" fn(*mut pci_dev)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_driver {
    name: *const u8,
    id_table: *const pci_device_id,
    probe: Option<unsafe extern "C" fn(*mut pci_dev, *const pci_device_id) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut pci_dev)>,
    suspend: Option<unsafe extern "C" fn(*mut pci_dev, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut pci_dev)>,
    sriov_configure: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_set_msix_vec_count: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_get_vf_total_msix: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    err_handler: *const pci_error_handlers,
    groups: *mut *const attribute_group,
    dev_groups: *mut *const attribute_group,
    driver: device_driver,
    dynids: pci_dynids,
    driver_managed_dma: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vmem_altmap {
    base_pfn: u64,
    end_pfn: u64,
    reserve: u64,
    free: u64,
    align: u64,
    alloc: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap_ops {
    folio_free: Option<unsafe extern "C" fn(*mut folio)>,
    migrate_to_ram: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    memory_failure: Option<unsafe extern "C" fn(*mut dev_pagemap, u64, u64, i32) -> i32>,
    folio_split: Option<unsafe extern "C" fn(*mut folio, *mut folio)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap {
    altmap: vmem_altmap,
    r#ref: percpu_ref,
    done: completion,
    r#type: u32,
    flags: u32,
    vmemmap_shift: u64,
    ops: *const dev_pagemap_ops,
    owner: *mut (),
    nr_range: i32,
    __slate_anon_9: __SlateRecord_memremap_h_143,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mctrl_gpios {
}


// VMA basic access permission flags
// These flags can be updated atomically via VMA/mmap read lock.
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_fault {
    __slate_anon_0: __SlateRecord_mm_h_754,
    flags: u32,
    pmd: *mut __SlateRecord_pgtable_64_types_h_22,
    pud: *mut __SlateRecord_pgtable_types_h_368,
    __slate_anon_4: __SlateRecord_mm_h_768,
    cow_page: *mut page,
    page: *mut page,
    pte: *mut __SlateRecord_pgtable_64_types_h_21,
    ptl: *mut spinlock,
    prealloc_pte: *mut page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
    open: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    close: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    mapped: Option<unsafe extern "C" fn(u64, u64, u64, *const file, *mut *mut ()) -> i32>,
    may_split: Option<unsafe extern "C" fn(*mut vm_area_struct, u64) -> i32>,
    mremap: Option<unsafe extern "C" fn(*mut vm_area_struct) -> i32>,
    mprotect: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, u64, u64) -> i32>,
    fault: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    huge_fault: Option<unsafe extern "C" fn(*mut vm_fault, u32) -> u32>,
    map_pages: Option<unsafe extern "C" fn(*mut vm_fault, u64, u64) -> u32>,
    pagesize: Option<unsafe extern "C" fn(*mut vm_area_struct) -> u64>,
    page_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    pfn_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    access: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut (), i32, i32) -> i32>,
    name: Option<unsafe extern "C" fn(*mut vm_area_struct) -> *const u8>,
    set_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, *mut mempolicy) -> i32>,
    get_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut u64) -> *mut mempolicy>,
}


// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
// 127: arbitrary random number, small enough to assemble well
// GUP_PIN_COUNTING_BIAS, and the associated functions that use it, overload
// the page's refcount so that two separate items are tracked: the original page
// reference count, and also a new count of how many pin_user_pages() calls were
// made against the page. ("gup-pinned" is another term for the latter).
//
// With this scheme, pin_user_pages() becomes special: such pages are marked as
// distinct from normal pages. As such, the unpin_user_page() call (and its
// variants) must be used in order to release gup-pinned pages.
//
// Choice of value:
//
// By making GUP_PIN_COUNTING_BIAS a power of two, debugging of page reference
// counts with respect to pin_user_pages() and unpin_user_page() becomes
// simpler, due to the fact that adding an even power of two to the page
// refcount has the effect of using only the upper N bits, for the code that
// counts up using the bias value. This means that the lower bits are left for
// the exclusive use of the original code that increments and decrements by one
// (or at least, by much smaller values than the bias value).
//
// Of course, once the lower bits overflow into the upper bits (and this is
// OK, because subtraction recovers the original values), then visual inspection
// no longer suffices to directly view the separate counts. However, for normal
// applications that don't have huge page reference counts, this won't be an
// issue.
//
// Locking: the lockless algorithm described in folio_try_get_rcu()
// provides safe operation for get_user_pages(), folio_mkclean() and
// other calls that race to set up page table entries.
// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
#[repr(C)]
#[derive(Clone, Copy)]
struct scatterlist {
    page_link: u64,
    offset: u32,
    length: u32,
    dma_address: u64,
    dma_length: u32,
    dma_flags: u32,
}


// PCI_DEVICE - macro used to describe a specific PCI device
// @vend: the 16 bit PCI Vendor ID
// @dev: the 16 bit PCI Device ID
//
// This macro is used to create a struct pci_device_id that matches a
// specific device.  The subvendor and subdevice fields will be set to
// PCI_ANY_ID.
// PCI_DEVICE_SUB - macro used to describe a specific PCI device with subsystem
// @vend: the 16 bit PCI Vendor ID
// @dev: the 16 bit PCI Device ID
// @subvend: the 16 bit PCI Subvendor ID
// @subdev: the 16 bit PCI Subdevice ID
//
// This macro is used to create a struct pci_device_id that matches a
// specific device with subsystem information.
// PCI_DEVICE_CLASS - macro used to describe a specific PCI device class
// @dev_class: the class, subclass, prog-if triple for this device
// @dev_class_mask: the class mask for this device
//
// This macro is used to create a struct pci_device_id that matches a
// specific PCI class.  The vendor, device, subvendor, and subdevice
// fields will be set to PCI_ANY_ID.
// PCI_VDEVICE - macro used to describe a specific PCI device in short form
// @vend: the vendor name
// @dev: the 16 bit PCI Device ID
//
// This macro is used to create a struct pci_device_id that matches a
// specific PCI device.  The subvendor, and subdevice fields will be set
// to PCI_ANY_ID. The macro allows the next field to follow as the device
// private data.
// PCI_VDEVICE_SUB - describe a specific PCI device/subdevice in a short form
// @vend: the vendor name
// @dev: the 16 bit PCI Device ID
// @subvend: the 16 bit PCI Subvendor ID
// @subdev: the 16 bit PCI Subdevice ID
//
// Generate the pci_device_id struct layout for the specific PCI
// device/subdevice. Private data may follow the output.
// PCI_DEVICE_DATA - macro used to describe a specific PCI device in very short form
// @vend: the vendor name (without PCI_VENDOR_ID_ prefix)
// @dev: the device name (without PCI_DEVICE_ID_<vend>_ prefix)
// @data: the driver data to be filled
//
// This macro is used to create a struct pci_device_id that matches a
// specific PCI device.  The subvendor, and subdevice fields will be set
// to PCI_ANY_ID.
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// module_pci_driver() - Helper macro for registering a PCI driver
// @__pci_driver: pci_driver struct
//
// Helper macro for PCI drivers which do not do anything special in module
// init/exit. This eliminates a lot of boilerplate. Each module may only
// use this macro once, and calling it replaces module_init() and module_exit()
// These macros should be used after a dma_map_sg call has been done
// to get bus addresses of each of the SG entries and their lengths.
// You should only work with the number of sg entries dma_map_sg
// returns, or alternatively stop on the first sg_dma_len(sg) which
// is 0.
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
fn pci_get_drvdata(pdev: *mut pci_dev) -> *mut () {
dev_get_drvdata((unsafe { core::ptr::addr_of_mut!((*pdev).dev) }) as *const device)
}

fn pci_set_drvdata(pdev: *mut pci_dev, data: *mut ()) {
dev_set_drvdata(unsafe { core::ptr::addr_of_mut!((*pdev).dev) }, data);
}

// The microseconds/nanosecond delay multiplicators are used to convert a
// constant microseconds/nanoseconds value to a value which can be used by the
// architectures specific implementation to transform it into loops.
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
#[inline(always)]
fn udelay(usec: u64) {
if unsafe { __slate_intrinsic_f217c4720f04f6db(usec) } {
if usec >= 20000 {
unsafe { __bad_udelay() };
} else {
unsafe { __const_udelay(usec.wrapping_mul(4295u64)) };
}
} else {
unsafe { __udelay(usec) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ktermios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 19],
    c_ispeed: u32,
    c_ospeed: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct winsize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
    kobj: kobject,
    owner: *mut module,
    ops: *const file_operations,
    list: list_head,
    dev: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
    buf: *mut u8,
    size: u64,
    from: u64,
    count: u64,
    pad_until: u64,
    index: i64,
    read_pos: i64,
    lock: mutex,
    op: *const seq_operations,
    poll_event: i32,
    file: *const file,
    private: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_operations {
    start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_operations {
    lookup: Option<unsafe extern "C" fn(*mut tty_driver, *mut file, i32) -> *mut tty_struct>,
    install: Option<unsafe extern "C" fn(*mut tty_driver, *mut tty_struct) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut tty_driver, *mut tty_struct)>,
    open: Option<unsafe extern "C" fn(*mut tty_struct, *mut file) -> i32>,
    close: Option<unsafe extern "C" fn(*mut tty_struct, *mut file)>,
    shutdown: Option<unsafe extern "C" fn(*mut tty_struct)>,
    cleanup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    write: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, u64) -> i64>,
    put_char: Option<unsafe extern "C" fn(*mut tty_struct, u8) -> i32>,
    flush_chars: Option<unsafe extern "C" fn(*mut tty_struct)>,
    write_room: Option<unsafe extern "C" fn(*mut tty_struct) -> u32>,
    chars_in_buffer: Option<unsafe extern "C" fn(*mut tty_struct) -> u32>,
    ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i64>,
    set_termios: Option<unsafe extern "C" fn(*mut tty_struct, *const ktermios)>,
    throttle: Option<unsafe extern "C" fn(*mut tty_struct)>,
    unthrottle: Option<unsafe extern "C" fn(*mut tty_struct)>,
    stop: Option<unsafe extern "C" fn(*mut tty_struct)>,
    start: Option<unsafe extern "C" fn(*mut tty_struct)>,
    hangup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    break_ctl: Option<unsafe extern "C" fn(*mut tty_struct, i32) -> i32>,
    flush_buffer: Option<unsafe extern "C" fn(*mut tty_struct)>,
    ldisc_ok: Option<unsafe extern "C" fn(*mut tty_struct, i32) -> i32>,
    set_ldisc: Option<unsafe extern "C" fn(*mut tty_struct)>,
    wait_until_sent: Option<unsafe extern "C" fn(*mut tty_struct, i32)>,
    send_xchar: Option<unsafe extern "C" fn(*mut tty_struct, u8)>,
    tiocmget: Option<unsafe extern "C" fn(*mut tty_struct) -> i32>,
    tiocmset: Option<unsafe extern "C" fn(*mut tty_struct, u32, u32) -> i32>,
    resize: Option<unsafe extern "C" fn(*mut tty_struct, *mut winsize) -> i32>,
    get_icount: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_icounter_struct) -> i32>,
    get_serial: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_struct) -> i32>,
    set_serial: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_struct) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut tty_struct, *mut seq_file)>,
    proc_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_driver {
    kref: kref,
    cdevs: *mut *mut cdev,
    owner: *mut module,
    driver_name: *const u8,
    name: *const u8,
    name_base: i32,
    major: i32,
    minor_start: i32,
    num: u32,
    r#type: u32,
    subtype: u32,
    init_termios: ktermios,
    flags: u64,
    proc_entry: *mut proc_dir_entry,
    other: *mut tty_driver,
    flip_wq: *mut workqueue_struct,
    ttys: *mut *mut tty_struct,
    ports: *mut *mut tty_port,
    termios: *mut *mut ktermios,
    driver_state: *mut (),
    ops: *const tty_operations,
    tty_drivers: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ld_semaphore {
    count: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    wait_readers: u32,
    read_wait: list_head,
    write_wait: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_ldisc_ops {
    name: *mut u8,
    num: i32,
    open: Option<unsafe extern "C" fn(*mut tty_struct) -> i32>,
    close: Option<unsafe extern "C" fn(*mut tty_struct)>,
    flush_buffer: Option<unsafe extern "C" fn(*mut tty_struct)>,
    read: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *mut u8, u64, *mut *mut (), u64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *const u8, u64) -> i64>,
    ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    set_termios: Option<unsafe extern "C" fn(*mut tty_struct, *const ktermios)>,
    poll: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *mut poll_table_struct) -> u32>,
    hangup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    receive_buf: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64)>,
    write_wakeup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    dcd_change: Option<unsafe extern "C" fn(*mut tty_struct, bool)>,
    receive_buf2: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64) -> u64>,
    lookahead_buf: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64)>,
    owner: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_ldisc {
    ops: *const tty_ldisc_ops,
    tty: *mut tty_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kfifo {
    r#in: u32,
    out: u32,
    mask: u32,
    esize: u32,
    data: *mut (),
}


// helper macro to distinguish between real in place fifo where the fifo
// array is a part of the structure and the fifo type where the array is
// outside of the fifo structure.
// DECLARE_KFIFO_PTR - macro to declare a fifo pointer object
// @fifo: name of the declared fifo
// @type: type of the fifo elements
// kfifo_is_empty - returns true if the fifo is empty
// @fifo: address of the fifo to be used
// kfifo_skip_count - skip output data
// @fifo: address of the fifo to be used
// @count: count of data to skip
// kfifo_get - get data from the fifo
// @fifo: address of the fifo to be used
// @val: address where to store the data
//
// This macro reads the data from the fifo.
// It returns 0 if the fifo was empty. Otherwise it returns the number
// processed elements.
//
// Note that with only one concurrent reader and one concurrent
// writer, you don't need extra locking to use these macro.
// kfifo_out - get data from the fifo
// @fifo: address of the fifo to be used
// @buf: pointer to the storage buffer
// @n: max. number of elements to get
//
// This macro gets some data from the fifo and returns the numbers of elements
// copied.
//
// Note that with only one concurrent reader and one concurrent
// writer, you don't need extra locking to use these macro.
#[repr(C)]
#[derive(Clone, Copy)]
struct tty_buffer {
    __slate_anon_0: __SlateRecord_tty_buffer_h_11,
    used: u32,
    size: u32,
    commit: u32,
    lookahead: u32,
    read: u32,
    flags: bool,
    __slate_pad_7: [u8; 3],
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_bufhead {
    head: *mut tty_buffer,
    flip_wq: *mut workqueue_struct,
    work: work_struct,
    lock: mutex,
    priority: __SlateRecord_types_h_190,
    sentinel: tty_buffer,
    free: llist_head,
    mem_used: __SlateRecord_types_h_190,
    mem_limit: i32,
    tail: *mut tty_buffer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port_operations {
    carrier_raised: Option<unsafe extern "C" fn(*mut tty_port) -> bool>,
    dtr_rts: Option<unsafe extern "C" fn(*mut tty_port, bool)>,
    shutdown: Option<unsafe extern "C" fn(*mut tty_port)>,
    activate: Option<unsafe extern "C" fn(*mut tty_port, *mut tty_struct) -> i32>,
    destruct: Option<unsafe extern "C" fn(*mut tty_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port_client_operations {
    receive_buf: Option<unsafe extern "C" fn(*mut tty_port, *const u8, *const u8, u64) -> u64>,
    lookahead_buf: Option<unsafe extern "C" fn(*mut tty_port, *const u8, *const u8, u64)>,
    write_wakeup: Option<unsafe extern "C" fn(*mut tty_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_pci_c_658 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_pci_c_965 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_pci_c_985 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_pci_c_1196 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_pci_c_4031_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_pci_c_4171 {
}


type __SlateImage_blacklist = __SlateAlign16<[pci_device_id; 31]>;

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits276U0 {
        #[bits(1)]
        pub dl_throttled: u32,
        #[bits(1)]
        pub dl_yielded: u32,
        #[bits(1)]
        pub dl_non_contending: u32,
        #[bits(1)]
        pub dl_overrun: u32,
        #[bits(1)]
        pub dl_server: u32,
        #[bits(1)]
        pub dl_server_active: u32,
        #[bits(1)]
        pub dl_defer: u32,
        #[bits(1)]
        pub dl_defer_armed: u32,
        #[bits(1)]
        pub dl_defer_running: u32,
        #[bits(1)]
        pub dl_defer_idle: u32,
        #[bits(1)]
        pub dl_bw_attached: u32,
        #[bits(5, access = na)]
        pub __slate_pad_11: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits68U0 {
        #[bits(1)]
        pub sched_reset_on_fork: u32,
        #[bits(1)]
        pub sched_contributes_to_load: u32,
        #[bits(1)]
        pub sched_migrated: u32,
        #[bits(1)]
        pub sched_task_hot: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits68U1 {
        #[bits(1)]
        pub sched_remote_wakeup: u32,
        #[bits(1)]
        pub sched_rt_mutex: u32,
        #[bits(1)]
        pub in_execve: u32,
        #[bits(1)]
        pub in_iowait: u32,
        #[bits(1)]
        pub restore_sigmask: u32,
        #[bits(1)]
        pub no_cgroup_migration: u32,
        #[bits(1)]
        pub frozen: u32,
        #[bits(1)]
        pub use_memdelay: u32,
        #[bits(1)]
        pub in_eventfd: u32,
        #[bits(1)]
        pub pasid_activated: u32,
        #[bits(1)]
        pub reported_split_lock: u32,
        #[bits(1)]
        pub in_thrashing: u32,
        #[bits(1)]
        pub in_nf_duplicate: u32,
        #[bits(3, access = na)]
        pub __slate_pad_13: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits279U0 {
        #[bits(1)]
        pub is_child_subreaper: u32,
        #[bits(1)]
        pub has_child_subreaper: u32,
        #[bits(1)]
        pub autoreap: u32,
        #[bits(1)]
        pub timer_create_restore_ids: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits674U0 {
        #[bits(1)]
        pub can_wakeup: bool,
        #[bits(1)]
        pub async_suspend: bool,
        #[bits(1)]
        pub in_dpm_list: bool,
        #[bits(1)]
        pub is_prepared: bool,
        #[bits(1)]
        pub is_suspended: bool,
        #[bits(1)]
        pub is_noirq_suspended: bool,
        #[bits(1)]
        pub is_late_suspended: bool,
        #[bits(1)]
        pub no_pm: bool,
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub direct_complete: bool,
        #[bits(6, access = na)]
        pub __slate_pad_10: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits671U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits674U1 {
        #[bits(1)]
        pub wakeup_path: bool,
        #[bits(1)]
        pub syscore: bool,
        #[bits(1)]
        pub no_pm_callbacks: bool,
        #[bits(1)]
        pub smart_suspend: bool,
        #[bits(1)]
        pub must_resume: bool,
        #[bits(1)]
        pub may_skip_resume: bool,
        #[bits(1)]
        pub out_band_wakeup: bool,
        #[bits(1)]
        pub strict_midlayer: bool,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits674U2 {
        #[bits(3)]
        pub disable_depth: u32,
        #[bits(1)]
        pub idle_notification: bool,
        #[bits(1)]
        pub request_pending: bool,
        #[bits(1)]
        pub deferred_resume: bool,
        #[bits(1)]
        pub needs_force_resume: bool,
        #[bits(1)]
        pub runtime_auto: bool,
        #[bits(1)]
        pub ignore_children: bool,
        #[bits(1)]
        pub no_callbacks: bool,
        #[bits(1)]
        pub irq_safe: bool,
        #[bits(1)]
        pub use_autosuspend: bool,
        #[bits(1)]
        pub timer_autosuspends: bool,
        #[bits(1)]
        pub memalloc_noio: bool,
        #[bits(2, access = na)]
        pub __slate_pad_12: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits674U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits762U0 {
        #[bits(1)]
        pub console: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits68U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits116U0 {
        #[bits(8)]
        pub base1: u16,
        #[bits(4)]
        pub r#type: u16,
        #[bits(1)]
        pub s: u16,
        #[bits(2)]
        pub dpl: u16,
        #[bits(1)]
        pub p: u16,
        #[bits(4)]
        pub limit1: u16,
        #[bits(1)]
        pub avl: u16,
        #[bits(1)]
        pub l: u16,
        #[bits(1)]
        pub d: u16,
        #[bits(1)]
        pub g: u16,
        #[bits(8)]
        pub base2: u16,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits122U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits558U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits627U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits569U0 {
        #[bits(1)]
        pub state_initialized: u32,
        #[bits(1)]
        pub state_in_sysfs: u32,
        #[bits(1)]
        pub state_add_uevent_sent: u32,
        #[bits(1)]
        pub state_remove_uevent_sent: u32,
        #[bits(1)]
        pub uevent_suppress: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits121U0 {
        #[bits(4)]
        pub sp_reg: u32,
        #[bits(4)]
        pub bp_reg: u32,
        #[bits(3)]
        pub r#type: u32,
        #[bits(1)]
        pub signal: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits400U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits430U0 {
        #[bits(1)]
        pub is_added: u32,
        #[bits(1)]
        pub unsafe_warn: u32,
        #[bits(1)]
        pub flit_mode: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits719U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits714U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits714U1 {
        #[bits(5)]
        pub pme_support: u32,
        #[bits(1)]
        pub pme_poll: u32,
        #[bits(1)]
        pub pinned: u32,
        #[bits(1)]
        pub config_rrs_sv: u32,
        #[bits(1)]
        pub imm_ready: u32,
        #[bits(1)]
        pub d1_support: u32,
        #[bits(1)]
        pub d2_support: u32,
        #[bits(1)]
        pub no_d1d2: u32,
        #[bits(1)]
        pub no_d3cold: u32,
        #[bits(1)]
        pub bridge_d3: u32,
        #[bits(1)]
        pub d3cold_allowed: u32,
        #[bits(1)]
        pub mmio_always_on: u32,
        #[bits(1)]
        pub wakeup_prepared: u32,
        #[bits(1)]
        pub skip_bus_pm: u32,
        #[bits(1)]
        pub ignore_hotplug: u32,
        #[bits(1)]
        pub hotplug_user_indicators: u32,
        #[bits(1)]
        pub clear_retrain_link: u32,
        #[bits(1)]
        pub no_bw_notif: u32,
        #[bits(2, access = na)]
        pub __slate_pad_18: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits714U2 {
        #[bits(1)]
        pub aspm_l0s_support: u32,
        #[bits(1)]
        pub aspm_l1_support: u32,
        #[bits(1)]
        pub ltr_path: u32,
        #[bits(1)]
        pub pasid_no_tlp: u32,
        #[bits(3)]
        pub eetlp_prefix_max: u32,
        #[bits(1, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 6], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits714U3 {
        #[bits(1)]
        pub transparent: u32,
        #[bits(1)]
        pub io_window: u32,
        #[bits(1)]
        pub pref_window: u32,
        #[bits(1)]
        pub pref_64_window: u32,
        #[bits(1)]
        pub multifunction: u32,
        #[bits(1)]
        pub is_busmaster: u32,
        #[bits(1)]
        pub no_msi: u32,
        #[bits(1)]
        pub block_cfg_access: u32,
        #[bits(1)]
        pub broken_parity_status: u32,
        #[bits(2)]
        pub irq_reroute_variant: u32,
        #[bits(1)]
        pub msi_enabled: u32,
        #[bits(1)]
        pub msix_enabled: u32,
        #[bits(1)]
        pub ari_enabled: u32,
        #[bits(1)]
        pub ats_enabled: u32,
        #[bits(1)]
        pub pasid_enabled: u32,
        #[bits(1)]
        pub pri_enabled: u32,
        #[bits(1)]
        pub tph_enabled: u32,
        #[bits(1)]
        pub fm_enabled: u32,
        #[bits(1)]
        pub is_managed: u32,
        #[bits(1)]
        pub is_msi_managed: u32,
        #[bits(1)]
        pub needs_freset: u32,
        #[bits(1)]
        pub state_saved: u32,
        #[bits(1)]
        pub is_physfn: u32,
        #[bits(1)]
        pub is_virtfn: u32,
        #[bits(1)]
        pub is_hotplug_bridge: u32,
        #[bits(1)]
        pub is_pciehp: u32,
        #[bits(1)]
        pub shpc_managed: u32,
        #[bits(1)]
        pub is_thunderbolt: u32,
        #[bits(1)]
        pub is_cxl: u32,
        #[bits(1)]
        pub untrusted: u32,
        #[bits(1)]
        pub external_facing: u32,
        #[bits(1)]
        pub broken_intx_masking: u32,
        #[bits(1)]
        pub io_window_1k: u32,
        #[bits(1)]
        pub irq_managed: u32,
        #[bits(1)]
        pub non_compliant_bars: u32,
        #[bits(1)]
        pub is_probed: u32,
        #[bits(1)]
        pub link_active_reporting: u32,
        #[bits(1)]
        pub no_vf_scan: u32,
        #[bits(1)]
        pub no_command_memory: u32,
        #[bits(1)]
        pub rom_bar_overlap: u32,
        #[bits(1)]
        pub rom_attr_enabled: u32,
        #[bits(1)]
        pub non_mappable_bars: u32,
        #[bits(5, access = na)]
        pub __slate_pad_42: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits714U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits778U0 {
        #[bits(1)]
        pub allow_unsafe_takeover: u32,
        #[bits(1)]
        pub backlog: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits799U0 {
        #[bits(1)]
        pub tx_stopped: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port {
    buf: tty_bufhead,
    tty: *mut tty_struct,
    itty: *mut tty_struct,
    ops: *const tty_port_operations,
    client_ops: *const tty_port_client_operations,
    lock: spinlock,
    blocked_open: i32,
    count: i32,
    open_wait: wait_queue_head,
    delta_msr_wait: wait_queue_head,
    flags: u64,
    iflags: u64,
    __slate_bits_0: __slate_bits::__SlateBits762U0,
    mutex: mutex,
    buf_mutex: mutex,
    xmit_buf: *mut u8,
    xmit_fifo: __SlateRecord_tty_port_h_118,
    close_delay: u32,
    closing_wait: u32,
    drain_delay: i32,
    kref: kref,
    client_data: *mut (),
}


// tty_port::iflags bits -- use atomic bit ops
// device is initialized
// device is suspended
// device is open
// uart drivers: use the uart_port::status field and the UPSTAT_* defines
// for s/w-based flow control steering and carrier detection status
// h/w flow control enabled
// carrier detect enabled
// device exclusively opened by
// kernel
#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
    kref: kref,
    index: i32,
    dev: *mut device,
    driver: *mut tty_driver,
    port: *mut tty_port,
    ops: *const tty_operations,
    ldisc: *mut tty_ldisc,
    ldisc_sem: ld_semaphore,
    atomic_write_lock: mutex,
    legacy_mutex: mutex,
    throttle_mutex: mutex,
    termios_rwsem: rw_semaphore,
    winsize_mutex: mutex,
    termios: ktermios,
    termios_locked: ktermios,
    name: [u8; 64],
    flags: u64,
    count: i32,
    receive_room: u32,
    winsize: winsize,
    flow: __SlateRecord_tty_h_212,
    ctrl: __SlateRecord_tty_h_218,
    hw_stopped: bool,
    closing: bool,
    flow_change: i32,
    link: *mut tty_struct,
    fasync: *mut fasync_struct,
    write_wait: wait_queue_head,
    read_wait: wait_queue_head,
    break_wait: wait_queue_head,
    hangup_work: work_struct,
    disc_data: *mut (),
    driver_data: *mut (),
    files_lock: spinlock,
    write_cnt: i32,
    write_buf: *mut u8,
    tty_files: list_head,
    SAK_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nbcon_context {
    console: *mut console,
    spinwait_max_us: u32,
    prio: u32,
    __slate_bits_0: __slate_bits::__SlateBits778U0,
    pbufs: *mut printk_buffers,
    seq: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nbcon_write_context {
    ctxt: nbcon_context,
    outbuf: *mut u8,
    len: u32,
    unsafe_takeover: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct console {
    name: [u8; 16],
    write: Option<unsafe extern "C" fn(*mut console, *const u8, u32)>,
    read: Option<unsafe extern "C" fn(*mut console, *mut u8, u32) -> i32>,
    device: Option<unsafe extern "C" fn(*mut console, *mut i32) -> *mut tty_driver>,
    unblank: Option<unsafe extern "C" fn()>,
    setup: Option<unsafe extern "C" fn(*mut console, *mut u8) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut console) -> i32>,
    r#match: Option<unsafe extern "C" fn(*mut console, *mut u8, i32, *mut u8) -> i32>,
    flags: i16,
    index: i16,
    cflag: i32,
    ispeed: u32,
    ospeed: u32,
    seq: u64,
    dropped: u64,
    data: *mut (),
    node: hlist_node,
    write_atomic: Option<unsafe extern "C" fn(*mut console, *mut nbcon_write_context)>,
    write_thread: Option<unsafe extern "C" fn(*mut console, *mut nbcon_write_context)>,
    device_lock: Option<unsafe extern "C" fn(*mut console, *mut u64)>,
    device_unlock: Option<unsafe extern "C" fn(*mut console, u64)>,
    nbcon_state: __SlateRecord_types_h_190,
    nbcon_seq: __SlateRecord_types_h_197,
    nbcon_device_ctxt: nbcon_context,
    nbcon_prev_seq: __SlateRecord_types_h_197,
    pbufs: *mut printk_buffers,
    kthread: *mut task_struct,
    rcuwait: rcuwait,
    irq_work: irq_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_struct {
    r#type: i32,
    line: i32,
    port: u32,
    irq: i32,
    flags: i32,
    xmit_fifo_size: i32,
    custom_divisor: i32,
    baud_base: i32,
    close_delay: u16,
    io_type: u8,
    reserved_char: [u8; 1],
    hub6: i32,
    closing_wait: u16,
    closing_wait2: u16,
    iomem_base: *mut u8,
    iomem_reg_shift: u16,
    port_high: u32,
    iomap_base: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_icounter_struct {
    cts: i32,
    dsr: i32,
    rng: i32,
    dcd: i32,
    rx: i32,
    tx: i32,
    frame: i32,
    overrun: i32,
    parity: i32,
    brk: i32,
    buf_overrun: i32,
    reserved: [i32; 9],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_rs485 {
    flags: u32,
    delay_rts_before_send: u32,
    delay_rts_after_send: u32,
    __slate_anon_3: __SlateRecord_serial_h_161,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_iso7816 {
    flags: u32,
    tg: u32,
    sc_fi: u32,
    sc_di: u32,
    clk: u32,
    reserved: [u32; 5],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_ops {
    tx_empty: Option<unsafe extern "C" fn(*mut uart_port) -> u32>,
    set_mctrl: Option<unsafe extern "C" fn(*mut uart_port, u32)>,
    get_mctrl: Option<unsafe extern "C" fn(*mut uart_port) -> u32>,
    stop_tx: Option<unsafe extern "C" fn(*mut uart_port)>,
    start_tx: Option<unsafe extern "C" fn(*mut uart_port)>,
    throttle: Option<unsafe extern "C" fn(*mut uart_port)>,
    unthrottle: Option<unsafe extern "C" fn(*mut uart_port)>,
    send_xchar: Option<unsafe extern "C" fn(*mut uart_port, u8)>,
    stop_rx: Option<unsafe extern "C" fn(*mut uart_port)>,
    start_rx: Option<unsafe extern "C" fn(*mut uart_port)>,
    enable_ms: Option<unsafe extern "C" fn(*mut uart_port)>,
    break_ctl: Option<unsafe extern "C" fn(*mut uart_port, i32)>,
    startup: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut uart_port)>,
    flush_buffer: Option<unsafe extern "C" fn(*mut uart_port)>,
    set_termios: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios, *const ktermios)>,
    set_ldisc: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios)>,
    pm: Option<unsafe extern "C" fn(*mut uart_port, u32, u32)>,
    r#type: Option<unsafe extern "C" fn(*mut uart_port) -> *const u8>,
    release_port: Option<unsafe extern "C" fn(*mut uart_port)>,
    request_port: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    config_port: Option<unsafe extern "C" fn(*mut uart_port, i32)>,
    verify_port: Option<unsafe extern "C" fn(*mut uart_port, *mut serial_struct) -> i32>,
    ioctl: Option<unsafe extern "C" fn(*mut uart_port, u32, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_icount {
    cts: u32,
    dsr: u32,
    rng: u32,
    dcd: u32,
    rx: u32,
    tx: u32,
    frame: u32,
    overrun: u32,
    parity: u32,
    brk: u32,
    buf_overrun: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_port {
    lock: spinlock,
    iobase: u64,
    membase: *mut u8,
    serial_in: Option<unsafe extern "C" fn(*mut uart_port, u32) -> u32>,
    serial_out: Option<unsafe extern "C" fn(*mut uart_port, u32, u32)>,
    set_termios: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios, *const ktermios)>,
    set_ldisc: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios)>,
    get_mctrl: Option<unsafe extern "C" fn(*mut uart_port) -> u32>,
    set_mctrl: Option<unsafe extern "C" fn(*mut uart_port, u32)>,
    get_divisor: Option<unsafe extern "C" fn(*mut uart_port, u32, *mut u32) -> u32>,
    set_divisor: Option<unsafe extern "C" fn(*mut uart_port, u32, u32, u32)>,
    get_rxtrig: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    set_rxtrig: Option<unsafe extern "C" fn(*mut uart_port, u8) -> i32>,
    startup: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut uart_port)>,
    throttle: Option<unsafe extern "C" fn(*mut uart_port)>,
    unthrottle: Option<unsafe extern "C" fn(*mut uart_port)>,
    break_ctl: Option<unsafe extern "C" fn(*mut uart_port, i32)>,
    handle_irq: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    pm: Option<unsafe extern "C" fn(*mut uart_port, u32, u32)>,
    handle_break: Option<unsafe extern "C" fn(*mut uart_port)>,
    rs485_config: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios, *mut serial_rs485) -> i32>,
    iso7816_config: Option<unsafe extern "C" fn(*mut uart_port, *mut serial_iso7816) -> i32>,
    ctrl_id: u32,
    port_id: u32,
    irq: u32,
    irqflags: u64,
    uartclk: u32,
    fifosize: u32,
    x_char: u8,
    regshift: u8,
    quirks: u8,
    iotype: i32,
    read_status_mask: u32,
    ignore_status_mask: u32,
    state: *mut uart_state,
    icount: uart_icount,
    cons: *mut console,
    flags: u64,
    status: u32,
    hw_stopped: bool,
    cons_flow: bool,
    mctrl: u32,
    frame_time: u32,
    r#type: u32,
    ops: *const uart_ops,
    custom_divisor: u32,
    line: u32,
    minor: u32,
    mapbase: u64,
    mapsize: u64,
    dev: *mut device,
    port_dev: *mut serial_port_device,
    sysrq: u64,
    sysrq_ch: u8,
    has_sysrq: u8,
    sysrq_seq: u8,
    hub6: u8,
    suspended: u8,
    console_reinit: u8,
    name: *const u8,
    attr_group: *mut attribute_group,
    tty_groups: *mut *const attribute_group,
    rs485: serial_rs485,
    rs485_supported: serial_rs485,
    rs485_term_gpio: *mut gpio_desc,
    rs485_rx_during_tx_gpio: *mut gpio_desc,
    iso7816: serial_iso7816,
    private_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_state {
    port: tty_port,
    pm_state: u32,
    refcount: __SlateRecord_types_h_190,
    remove_wait: wait_queue_head,
    uart_port: *mut uart_port,
}


// do not assign an irq
// Use the Base address register size to cap number of ports
#[repr(C)]
#[derive(Clone, Copy)]
struct pciserial_board {
    flags: u32,
    num_ports: u32,
    base_baud: u32,
    uart_offset: u32,
    reg_shift: u32,
    first_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_8250_ops {
    setup_irq: Option<unsafe extern "C" fn(*mut uart_8250_port) -> i32>,
    release_irq: Option<unsafe extern "C" fn(*mut uart_8250_port)>,
    setup_timer: Option<unsafe extern "C" fn(*mut uart_8250_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_8250_em485 {
    start_tx_timer: hrtimer,
    stop_tx_timer: hrtimer,
    active_timer: *mut hrtimer,
    port: *mut uart_8250_port,
    __slate_bits_0: __slate_bits::__SlateBits799U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_8250_port {
    port: uart_port,
    timer: timer_list,
    list: list_head,
    capabilities: u32,
    bugs: u16,
    tx_loadsz: u32,
    acr: u8,
    fcr: u8,
    ier: u8,
    lcr: u8,
    mcr: u8,
    cur_iotype: u8,
    rpm_tx_active: u32,
    canary: u8,
    probe: u8,
    gpios: *mut mctrl_gpios,
    lsr_saved_flags: u16,
    lsr_save_mask: u16,
    console_line_ended: bool,
    console_msr_work_allow: bool,
    msr_saved_flags: u8,
    console_msr_work: irq_work,
    dma: *mut uart_8250_dma,
    ops: *const uart_8250_ops,
    dl_read: Option<unsafe extern "C" fn(*mut uart_8250_port) -> u32>,
    dl_write: Option<unsafe extern "C" fn(*mut uart_8250_port, u32)>,
    em485: *mut uart_8250_em485,
    rs485_start_tx: Option<unsafe extern "C" fn(*mut uart_8250_port, bool)>,
    rs485_stop_tx: Option<unsafe extern "C" fn(*mut uart_8250_port, bool)>,
    overrun_backoff: delayed_work,
    overrun_backoff_time_ms: u32,
}


fn up_to_u8250p(up: *mut uart_port) -> *mut uart_8250_port {
container_of__2!(up, uart_8250_port)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct data_chunk {
    size: u64,
    icg: u64,
    dst_icg: u64,
    src_icg: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_interleaved_template {
    src_start: u64,
    dst_start: u64,
    dir: u32,
    src_inc: bool,
    dst_inc: bool,
    src_sgl: bool,
    dst_sgl: bool,
    numf: u64,
    frame_size: u64,
    sgl: [data_chunk; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_vec {
    addr: u64,
    len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dmaengine_h_240 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_chan_percpu {
    memcpy_count: u64,
    bytes_transferred: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_router {
    dev: *mut device,
    route_free: Option<unsafe extern "C" fn(*mut device, *mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_chan {
    device: *mut dma_device,
    slave: *mut device,
    cookie: i32,
    completed_cookie: i32,
    lock: spinlock,
    chan_id: i32,
    dev: *mut dma_chan_dev,
    name: *const u8,
    dbg_client_name: *mut u8,
    device_node: list_head,
    local: *mut dma_chan_percpu,
    client_count: i32,
    table_count: i32,
    router: *mut dma_router,
    route_data: *mut (),
    private: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_chan_dev {
    chan: *mut dma_chan,
    device: device,
    dev_id: i32,
    chan_dma_dev: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_slave_config {
    direction: u32,
    src_addr: u64,
    dst_addr: u64,
    src_addr_width: u32,
    dst_addr_width: u32,
    src_maxburst: u32,
    dst_maxburst: u32,
    src_port_window_size: u32,
    dst_port_window_size: u32,
    device_fc: bool,
    peripheral_config: *mut (),
    peripheral_size: u64,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_pciserial_init_ports: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pciserial_remove_ports: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pciserial_suspend_ports: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pciserial_resume_ports: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_8250_pci__602_6086_serial_pci_driver_init6:\t\t\t\n.long\t{4} - .\t\n.previous\t\t\t\t\t\n", sym pciserial_init_ports, sym pciserial_remove_ports, sym pciserial_suspend_ports, sym pciserial_resume_ports, sym serial_pci_driver_init, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct dma_slave_caps {
    src_addr_widths: u32,
    dst_addr_widths: u32,
    directions: u32,
    min_burst: u32,
    max_burst: u32,
    max_sg_burst: u32,
    cmd_pause: bool,
    cmd_resume: bool,
    cmd_terminate: bool,
    residue_granularity: u32,
    descriptor_reuse: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmaengine_result {
    result: u32,
    residue: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmaengine_unmap_data {
    map_cnt: u8,
    to_cnt: u8,
    from_cnt: u8,
    bidi_cnt: u8,
    dev: *mut device,
    kref: kref,
    len: u64,
    addr: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_descriptor_metadata_ops {
    attach: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor, *mut (), u64) -> i32>,
    get_ptr: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor, *mut u64, *mut u64) -> *mut ()>,
    set_len: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_async_tx_descriptor {
    cookie: i32,
    flags: u32,
    phys: u64,
    chan: *mut dma_chan,
    tx_submit: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor) -> i32>,
    desc_free: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor) -> i32>,
    callback: Option<unsafe extern "C" fn(*mut ())>,
    callback_result: Option<unsafe extern "C" fn(*mut (), *const dmaengine_result)>,
    callback_param: *mut (),
    unmap: *mut dmaengine_unmap_data,
    desc_metadata_mode: u32,
    metadata_ops: *const dma_descriptor_metadata_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_tx_state {
    last: i32,
    used: i32,
    residue: u32,
    in_flight_bytes: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_slave_map {
    devname: *const u8,
    slave: *const u8,
    param: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_filter {
    r#fn: Option<unsafe extern "C" fn(*mut dma_chan, *mut ()) -> bool>,
    mapcnt: i32,
    map: *const dma_slave_map,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_device {
    r#ref: kref,
    chancnt: u32,
    privatecnt: u32,
    channels: list_head,
    global_node: list_head,
    filter: dma_filter,
    cap_mask: __SlateRecord_dmaengine_h_240,
    desc_metadata_modes: u32,
    max_xor: u16,
    max_pq: u16,
    copy_align: u32,
    xor_align: u32,
    pq_align: u32,
    fill_align: u32,
    dev_id: i32,
    dev: *mut device,
    owner: *mut module,
    chan_ida: ida,
    src_addr_widths: u32,
    dst_addr_widths: u32,
    directions: u32,
    min_burst: u32,
    max_burst: u32,
    max_sg_burst: u32,
    descriptor_reuse: bool,
    residue_granularity: u32,
    device_alloc_chan_resources: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_router_config: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_free_chan_resources: Option<unsafe extern "C" fn(*mut dma_chan)>,
    device_prep_dma_memcpy: Option<unsafe extern "C" fn(*mut dma_chan, u64, u64, u64, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_xor: Option<unsafe extern "C" fn(*mut dma_chan, u64, *mut u64, u32, u64, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_xor_val: Option<unsafe extern "C" fn(*mut dma_chan, *mut u64, u32, u64, *mut u32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_pq: Option<unsafe extern "C" fn(*mut dma_chan, *mut u64, *mut u64, u32, *const u8, u64, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_pq_val: Option<unsafe extern "C" fn(*mut dma_chan, *mut u64, *mut u64, u32, *const u8, u64, *mut u32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_memset: Option<unsafe extern "C" fn(*mut dma_chan, u64, i32, u64, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_memset_sg: Option<unsafe extern "C" fn(*mut dma_chan, *mut scatterlist, u32, i32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_interrupt: Option<unsafe extern "C" fn(*mut dma_chan, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_peripheral_dma_vec: Option<unsafe extern "C" fn(*mut dma_chan, *const dma_vec, u64, u32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_slave_sg: Option<unsafe extern "C" fn(*mut dma_chan, *mut scatterlist, u32, u32, u64, *mut ()) -> *mut dma_async_tx_descriptor>,
    device_prep_config_sg: Option<unsafe extern "C" fn(*mut dma_chan, *mut scatterlist, u32, u32, u64, *mut dma_slave_config) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_cyclic: Option<unsafe extern "C" fn(*mut dma_chan, u64, u64, u64, u32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_interleaved_dma: Option<unsafe extern "C" fn(*mut dma_chan, *mut dma_interleaved_template, u64) -> *mut dma_async_tx_descriptor>,
    device_caps: Option<unsafe extern "C" fn(*mut dma_chan, *mut dma_slave_caps)>,
    device_config: Option<unsafe extern "C" fn(*mut dma_chan, *mut dma_slave_config) -> i32>,
    device_pause: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_resume: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_terminate_all: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_synchronize: Option<unsafe extern "C" fn(*mut dma_chan)>,
    device_tx_status: Option<unsafe extern "C" fn(*mut dma_chan, i32, *mut dma_tx_state) -> u32>,
    device_issue_pending: Option<unsafe extern "C" fn(*mut dma_chan)>,
    device_release: Option<unsafe extern "C" fn(*mut dma_device)>,
    dbg_summary_show: Option<unsafe extern "C" fn(*mut seq_file, *mut dma_device)>,
    dbg_dev_root: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_8250_dma {
    tx_dma: Option<unsafe extern "C" fn(*mut uart_8250_port) -> i32>,
    rx_dma: Option<unsafe extern "C" fn(*mut uart_8250_port) -> i32>,
    prepare_tx_dma: Option<unsafe extern "C" fn(*mut uart_8250_port)>,
    prepare_rx_dma: Option<unsafe extern "C" fn(*mut uart_8250_port)>,
    r#fn: Option<unsafe extern "C" fn(*mut dma_chan, *mut ()) -> bool>,
    rx_param: *mut (),
    tx_param: *mut (),
    rxconf: dma_slave_config,
    txconf: dma_slave_config,
    rxchan: *mut dma_chan,
    txchan: *mut dma_chan,
    rx_dma_addr: u64,
    tx_dma_addr: u64,
    rx_addr: u64,
    tx_addr: u64,
    rx_cookie: i32,
    tx_cookie: i32,
    rx_buf: *mut (),
    rx_size: u64,
    tx_size: u64,
    tx_running: u8,
    tx_err: u8,
    rx_running: u8,
}


fn serial_out(up: *mut uart_8250_port, offset: i32, value: i32) {
unsafe { unsafe { (*up).port.serial_out }.unwrap()(core::ptr::addr_of_mut!((*up).port), offset as u32, value as u32) };
}

fn serial_icr_write(up: *mut uart_8250_port, offset: i32, value: i32) {
serial_out(up, 7i32, offset);
serial_out(up, 5i32, value);
}

// Unknown vendors/cards - this should not be in linux/pci_ids.h
/// init function returns:
///  > 0 - number of ports
///  = 0 - use board->num_ports
///  < 0 - error
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_serial_quirk {
    vendor: u32,
    device: u32,
    subvendor: u32,
    subdevice: u32,
    probe: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
    init: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
    setup: Option<unsafe extern "C" fn(*mut serial_private, *const pciserial_board, *mut uart_8250_port, i32) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut pci_dev)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct f815xxa_data {
    lock: spinlock,
    idx: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_private {
    dev: *mut pci_dev,
    nr: u32,
    quirk: *mut pci_serial_quirk,
    board: *const pciserial_board,
    line: [i32; 0],
}


static pci_use_msi: __SlateAlign16<[pci_device_id; 7]> = __SlateAlign16([{
let mut __t0: pci_device_id = unsafe { core::mem::zeroed() };
__t0.vendor = 38672u32;
__t0.device = 39168u32;
__t0.subvendor = 40960u32;
__t0.subdevice = 4096u32;
__t0.class = 0u32;
__t0.class_mask = 0u32;
__t0
}, {
let mut __t1: pci_device_id = unsafe { core::mem::zeroed() };
__t1.vendor = 38672u32;
__t1.device = 39186u32;
__t1.subvendor = 40960u32;
__t1.subdevice = 4096u32;
__t1.class = 0u32;
__t1.class_mask = 0u32;
__t1
}, {
let mut __t2: pci_device_id = unsafe { core::mem::zeroed() };
__t2.vendor = 38672u32;
__t2.device = 39202u32;
__t2.subvendor = 40960u32;
__t2.subdevice = 4096u32;
__t2.class = 0u32;
__t2.class_mask = 0u32;
__t2
}, {
let mut __t3: pci_device_id = unsafe { core::mem::zeroed() };
__t3.vendor = 4699u32;
__t3.device = 37120u32;
__t3.subvendor = 40960u32;
__t3.subdevice = 4096u32;
__t3.class = 0u32;
__t3.class_mask = 0u32;
__t3
}, {
let mut __t4: pci_device_id = unsafe { core::mem::zeroed() };
__t4.vendor = 5520u32;
__t4.device = 894u32;
__t4.subvendor = 4294967295u32;
__t4.subdevice = 4294967295u32;
__t4.class = 0u32;
__t4.class_mask = 0u32;
__t4
}, {
let mut __t5: pci_device_id = unsafe { core::mem::zeroed() };
__t5.vendor = 4699u32;
__t5.device = 37120u32;
__t5.subvendor = 40960u32;
__t5.subdevice = 4096u32;
__t5
}, {
let mut __t6: pci_device_id = unsafe { core::mem::zeroed() };
__t6
}]);

fn moan_device(str: *const u8, dev: *mut pci_dev) {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s\nPlease send the output of lspci -vv, this\nmessage (0x%04x,0x%04x,0x%04x,0x%04x), the\nmanufacturer and name of serial board or\nmodem board to <linux-serial@vger.kernel.org>.\n".as_ptr() as *const u8, str, (*dev).vendor as i32, (*dev).device as i32, (*dev).subsystem_vendor as i32, (*dev).subsystem_device as i32) };
}

fn setup_port(r#priv: *mut serial_private, port: *mut uart_8250_port, bar: u8, offset: u32, regshift: i32) -> i32 {
let mut iomem: *mut () = core::ptr::null_mut::<()>();
if pci_resource_flags__2!(unsafe { (*r#priv).dev }, bar) & 512 != 0 {
iomem = unsafe { pcim_iomap((*r#priv).dev, bar as i32, 0u64) };
if iomem.is_null() {
return 12i32.wrapping_neg();
}
}
unsafe { serial8250_pci_setup_port((*r#priv).dev, port, bar, offset, regshift, iomem) }
}

/// ADDI-DATA GmbH communication cards <info@addi-data.com>
#[unsafe(link_section = ".text.slate_distinct._8250_pci.addidata_apci7800_setup")]
extern "C" fn addidata_apci7800_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let mut bar: u32 = 0;
let mut offset: u32 = unsafe { (*board).first_offset };
bar = FL_GET_BASE!(unsafe { (*board).flags });
if idx < 2 {
offset = offset.wrapping_add((idx as u32).wrapping_mul(unsafe { (*board).uart_offset }));
} else {
if idx >= 2 && idx < 4 {
bar = bar.wrapping_add(1);
offset = offset.wrapping_add((idx.wrapping_sub(2) as u32).wrapping_mul(unsafe { (*board).uart_offset }));
} else {
if idx >= 4 && idx < 6 {
bar = bar.wrapping_add(2);
offset = offset.wrapping_add((idx.wrapping_sub(4) as u32).wrapping_mul(unsafe { (*board).uart_offset }));
} else {
if idx >= 6 {
bar = bar.wrapping_add(3);
offset = offset.wrapping_add((idx.wrapping_sub(6) as u32).wrapping_mul(unsafe { (*board).uart_offset }));
}
}
}
}
setup_port(r#priv, port, bar as u8, offset, (unsafe { (*board).reg_shift }) as i32)
}

/// AFAVLAB uses a different mixture of BARs and offsets
/// Not that ugly ;) -- HW
#[unsafe(link_section = ".text.slate_distinct._8250_pci.afavlab_setup")]
extern "C" fn afavlab_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let mut bar: u32 = 0;
let mut offset: u32 = unsafe { (*board).first_offset };
bar = FL_GET_BASE!(unsafe { (*board).flags });
if idx < 4 {
bar = bar.wrapping_add(idx as u32);
} else {
bar = 4;
offset = offset.wrapping_add((idx.wrapping_sub(4) as u32).wrapping_mul(unsafe { (*board).uart_offset }));
}
setup_port(r#priv, port, bar as u8, offset, (unsafe { (*board).reg_shift }) as i32)
}

/// HP's Remote Management Console.  The Diva chip came in several
/// different versions.  N-class, L2000 and A500 have two Diva chips, each
/// with 3 UARTs (the third UART on the second chip is unused).  Superdome
/// and Keystone have one Diva chip with 3 UARTs.  Some later machines have
/// one Diva chip, but it has been expanded to 5 UARTs.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_hp_diva_init")]
extern "C" fn pci_hp_diva_init(dev: *mut pci_dev) -> i32 {
let mut rc: i32 = 0;
match (unsafe { (*dev).subsystem_device }) as i32 {
4169 | 4643 | 4646 | 4738 => {
rc = 3;
}
4170 => {
rc = 2;
}
4171 => {
rc = 4;
}
4647 | 4906 => {
rc = 1;
}
_ => {
}
}
rc
}

/// HP's Diva chip puts the 4th/5th serial port further out, and
/// some serial ports are supposed to be hidden on certain models.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_hp_diva_setup")]
extern "C" fn pci_hp_diva_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, mut idx: i32) -> i32 {
let mut offset: u32 = unsafe { (*board).first_offset };
let bar: u32 = FL_GET_BASE!(unsafe { (*board).flags });
match (unsafe { (*(*r#priv).dev).subsystem_device }) as i32 {
4171 => {
if idx == 3 {
idx = idx.wrapping_add(1);
}
}
4738 => {
if idx > 0 {
idx = idx.wrapping_add(1);
}
if idx > 2 {
idx = idx.wrapping_add(1);
}
}
_ => {
}
}
if idx > 2 {
offset = 24;
}
offset = offset.wrapping_add((idx as u32).wrapping_mul(unsafe { (*board).uart_offset }));
setup_port(r#priv, port, bar as u8, offset, (unsafe { (*board).reg_shift }) as i32)
}

/// Added for EKF Intel i960 serial boards
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_inteli960ni_init")]
extern "C" fn pci_inteli960ni_init(dev: *mut pci_dev) -> i32 {
let mut oldval: u32 = 0;
if ((unsafe { (*dev).subsystem_device }) as i32) & 4096 == 0 {
return 19i32.wrapping_neg();
}
// is firmware started?
unsafe { pci_read_config_dword(dev as *const pci_dev, 68i32, core::ptr::addr_of_mut!(oldval)) };
if (oldval as i64) == 4096 {
// RESET value
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Local i960 firmware missing\n".as_ptr() as *const u8) };
}
return 19i32.wrapping_neg();
}
0
}

/// Some PCI serial cards using the PLX 9050 PCI interface chip require
/// that the card interrupt be explicitly enabled or disabled.  This
/// seems to be mainly needed on card using the PLX which also use I/O
/// mapped memory.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_plx9050_init")]
extern "C" fn pci_plx9050_init(dev: *mut pci_dev) -> i32 {
let mut irq_config: u8 = 65;
if pci_resource_flags!(dev, 0i32) & 512 == 0 {
moan_device(c"no memory in bar 0".as_ptr() as *const u8, dev);
return 0;
}
if ((unsafe { (*dev).vendor }) as i32) == 5332 || ((unsafe { (*dev).subsystem_vendor }) as i32) == 55373 {
irq_config = 67;
}
if ((unsafe { (*dev).vendor }) as i32) == 4277 && ((unsafe { (*dev).device }) as i32) == 4202 {
irq_config = 91;
}
// As the megawolf cards have the int pins active
// high, and have 2 UART chips, both ints must be
// enabled on the 9050. Also, the UARTS are set in
// 16450 mode by default, so we have to enable the
// 16C950 'enhanced' mode so that we can use the
// deep FIFOs
// enable/disable interrupts
let p__0: *mut () = unsafe { ioremap(pci_resource_start!(dev, 0i32), 128u64) };
if p__0.is_null() {
return 12i32.wrapping_neg();
}
writel(irq_config as u32, (unsafe { (p__0 as *const u8).wrapping_offset(76 as isize) }) as *mut ());
// Read the register back to ensure that it took effect.
readl(((unsafe { (p__0 as *const u8).wrapping_offset(76 as isize) }) as *mut ()) as *const ());
unsafe { iounmap(p__0 as *mut ()) };
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_plx9050_exit")]
extern "C" fn pci_plx9050_exit(dev: *mut pci_dev) {
if pci_resource_flags!(dev, 0i32) & 512 == 0 {
return;
}
// disable interrupts
let p__0: *mut u8 = (unsafe { ioremap(pci_resource_start!(dev, 0i32), 128u64) }) as *mut u8;
if !p__0.is_null() {
writel(0u32, p__0.wrapping_offset(76 as isize) as *mut ());
// Read the register back to ensure that it took effect.
readl(p__0.wrapping_offset(76 as isize) as *const ());
unsafe { iounmap(p__0 as *mut ()) };
}
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_ni8420_exit")]
extern "C" fn pci_ni8420_exit(dev: *mut pci_dev) {
let bar: u32 = 0;
if pci_resource_flags!(dev, bar) & 512 == 0 {
moan_device(c"no memory in bar".as_ptr() as *const u8, dev);
return;
}
let p__0: *mut () = unsafe { pci_ioremap_bar(dev, bar as i32) };
if p__0.is_null() {
return;
}
// Disable the CPU Interrupt
writel(readl(((unsafe { (p__0 as *const u8).wrapping_offset(56 as isize) }) as *mut ()) as *const ()) & 4294959103u32, (unsafe { (p__0 as *const u8).wrapping_offset(56 as isize) }) as *mut ());
unsafe { iounmap(p__0 as *mut ()) };
}

// MITE registers
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_ni8430_exit")]
extern "C" fn pci_ni8430_exit(dev: *mut pci_dev) {
let bar: u32 = 0;
if pci_resource_flags!(dev, bar) & 512 == 0 {
moan_device(c"no memory in bar".as_ptr() as *const u8, dev);
return;
}
let p__0: *mut () = unsafe { pci_ioremap_bar(dev, bar as i32) };
if p__0.is_null() {
return;
}
// Disable the CPU Interrupt
writel(1073741824u32, (unsafe { (p__0 as *const u8).wrapping_offset(16 as isize) }) as *mut ());
unsafe { iounmap(p__0 as *mut ()) };
}

/// SBS Technologies Inc. PMC-OCTPRO and P-OCTAL cards
#[unsafe(link_section = ".text.slate_distinct._8250_pci.sbs_setup")]
extern "C" fn sbs_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let bar: u32 = 0;
let mut offset: u32 = unsafe { (*board).first_offset };
if idx < 4 {
// first four channels map to 0, 0x100, 0x200, 0x300
offset = offset.wrapping_add((idx as u32).wrapping_mul(unsafe { (*board).uart_offset }));
} else {
if idx < 8 {
// last four channels map to 0x1000, 0x1100, 0x1200, 0x1300
offset = offset.wrapping_add((idx as u32).wrapping_mul(unsafe { (*board).uart_offset }).wrapping_add(3072));
} else {
return 1;
}
}
// we have only 8 ports on PMC-OCTALPRO
setup_port(r#priv, port, bar as u8, offset, (unsafe { (*board).reg_shift }) as i32)
}

// This does initialization for PMC OCTALPRO cards:
// maps the device memory, resets the UARTs (needed, bc
// if the module is removed and inserted again, the card
// is in the sleep mode) and enables global interrupt.
// global control register offset for SBS PMC-OctalPro
#[unsafe(link_section = ".text.slate_distinct._8250_pci.sbs_init")]
extern "C" fn sbs_init(dev: *mut pci_dev) -> i32 {
let p__0: *mut u8 = (unsafe { pci_ioremap_bar(dev, 0i32) }) as *mut u8;
if p__0.is_null() {
return 12i32.wrapping_neg();
}
// Set bit-4 Control Register (UART RESET) in to reset the uarts
writeb(16u8, p__0.wrapping_offset(1280 as isize) as *mut ());
udelay(50u64);
writeb(0u8, p__0.wrapping_offset(1280 as isize) as *mut ());
// Set bit-2 (INTENABLE) of Control Register
writeb(4u8, p__0.wrapping_offset(1280 as isize) as *mut ());
unsafe { iounmap(p__0 as *mut ()) };
0
}

// Disables the global interrupt of PMC-OctalPro
#[unsafe(link_section = ".text.slate_distinct._8250_pci.sbs_exit")]
extern "C" fn sbs_exit(dev: *mut pci_dev) {
let p__0: *mut u8 = (unsafe { pci_ioremap_bar(dev, 0i32) }) as *mut u8;
// FIXME: What if resource_len < OCT_REG_CR_OFF
if !p__0.is_null() {
writeb(0u8, p__0.wrapping_offset(1280 as isize) as *mut ());
}
unsafe { iounmap(p__0 as *mut ()) };
}

// SIIG serial cards have an PCI interface chip which also controls
// the UART clocking frequency. Each UART can be clocked independently
// (except cards equipped with 4 UARTs) and initial clocking settings
// are stored in the EEPROM chip. It can cause problems because this
// version of serial driver doesn't support differently clocked UART's
// on single PCI card. To prevent this, initialization functions set
// high frequency clocking for all UART's on given card. It is safe (I
// hope) because it doesn't touch EEPROM settings to prevent conflicts
// with other OSes (like M$ DOS).
//
//  SIIG support added by Andrey Panin <pazke@donpac.ru>, 10/1999
//
// There is two family of SIIG serial cards with different PCI
// interface chip and different configuration methods:
//     - 10x cards have control registers in IO and/or memory space;
//     - 20x cards have control registers in standard PCI configuration space.
//
// Note: all 10x cards have PCI device ids 0x10..
//       all 20x cards have PCI device ids 0x20..
//
// There are also Quartet Serial cards which use Oxford Semiconductor
// 16954 quad UART PCI chip clocked by 18.432 MHz quartz.
//
// Note: some SIIG cards are probed by the parport_serial object.
fn pci_siig10x_init(dev: *mut pci_dev) -> i32 {
let mut data: u16 = 0;
match ((unsafe { (*dev).device }) as i32) & 65528 {
4096 => {
data = 65503; // 1S
}
4144 => {
data = 63487; // 2S, 2S1P
}
_ => {
data = 65531; // 1S1P, 4S
}
}
let p__0: *mut () = unsafe { ioremap(pci_resource_start!(dev, 0i32), 128u64) };
if p__0.is_null() {
return 12i32.wrapping_neg();
}
writew(((readw(((unsafe { (p__0 as *const u8).wrapping_offset(40 as isize) }) as *mut ()) as *const ()) as i32) & (data as i32)) as u16, (unsafe { (p__0 as *const u8).wrapping_offset(40 as isize) }) as *mut ());
readw(((unsafe { (p__0 as *const u8).wrapping_offset(40 as isize) }) as *mut ()) as *const ());
unsafe { iounmap(p__0 as *mut ()) };
0
}

fn pci_siig20x_init(dev: *mut pci_dev) -> i32 {
let mut data: u8 = 0;
// Change clock frequency for the first UART.
unsafe { pci_read_config_byte(dev as *const pci_dev, 111i32, core::ptr::addr_of_mut!(data)) };
unsafe { pci_write_config_byte(dev as *const pci_dev, 111i32, ((data as i32) & 239) as u8) };
// If this card has 2 UART, we have to do the same with second UART.
if ((unsafe { (*dev).device }) as i32) & 65532 == 8240 & 65532 || ((unsafe { (*dev).device }) as i32) & 65532 == 8288 & 65532 {
unsafe { pci_read_config_byte(dev as *const pci_dev, 115i32, core::ptr::addr_of_mut!(data)) };
unsafe { pci_write_config_byte(dev as *const pci_dev, 115i32, ((data as i32) & 239) as u8) };
}
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_siig_init")]
extern "C" fn pci_siig_init(dev: *mut pci_dev) -> i32 {
let r#type: u32 = (((unsafe { (*dev).device }) as i32) & 65280) as u32;
if r#type == 4096 {
return pci_siig10x_init(dev);
}
if r#type == 8192 {
return pci_siig20x_init(dev);
}
moan_device(c"Unknown SIIG card".as_ptr() as *const u8, dev);
19i32.wrapping_neg()
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_siig_setup")]
extern "C" fn pci_siig_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let mut bar: u32 = FL_GET_BASE!(unsafe { (*board).flags }).wrapping_add(idx as u32);
let mut offset: u32 = 0;
if idx > 3 {
bar = 4;
offset = idx.wrapping_sub(4).wrapping_mul(8) as u32;
}
setup_port(r#priv, port, bar as u8, offset, 0i32)
}

/// Timedia has an explosion of boards, and to avoid the PCI table from
/// growing *huge*, we use this function to collapse some 70 entries
/// in the PCI table into one, for sanity's and compactness's sake.
static timedia_single_port: [u16; 6] = [16421, 16423, 16424, 20517, 20519, 0];

static timedia_dual_port: __SlateAlign16<[u16; 30]> = __SlateAlign16([2u16, 16438u16, 16439u16, 16440u16, 16504u16, 16505u16, 16517u16, 16520u16, 16521u16, 20535u16, 20600u16, 20601u16, 20613u16, 24697u16, 28793u16, 32889u16, 33079u16, 33080u16, 33335u16, 33336u16, 36985u16, 37175u16, 37176u16, 37431u16, 37432u16, 41081u16, 45177u16, 49273u16, 53369u16, 0u16]);

static timedia_quad_port: __SlateAlign16<[u16; 23]> = __SlateAlign16([16469u16, 16470u16, 16533u16, 16534u16, 20566u16, 33110u16, 33111u16, 33366u16, 33367u16, 36950u16, 37206u16, 37207u16, 37208u16, 37209u16, 37462u16, 37463u16, 41046u16, 41303u16, 41304u16, 41305u16, 45142u16, 45399u16, 0u16]);

static timedia_eight_port: __SlateAlign16<[u16; 13]> = __SlateAlign16([16485u16, 16486u16, 20581u16, 20582u16, 33126u16, 36966u16, 37222u16, 37223u16, 37224u16, 41062u16, 41319u16, 41320u16, 0u16]);

#[repr(C)]
#[derive(Clone, Copy)]
struct timedia_struct {
    num: i32,
    ids: *const u16,
}


static timedia_data: __SlateAlign16<[timedia_struct; 4]> = __SlateAlign16([{
let mut __t0: timedia_struct = unsafe { core::mem::zeroed() };
__t0.num = 1i32;
__t0.ids = unsafe { core::ptr::addr_of!(timedia_single_port) as *const u16 };
__t0
}, {
let mut __t1: timedia_struct = unsafe { core::mem::zeroed() };
__t1.num = 2i32;
__t1.ids = unsafe { core::ptr::addr_of!(timedia_dual_port.0) as *const u16 };
__t1
}, {
let mut __t2: timedia_struct = unsafe { core::mem::zeroed() };
__t2.num = 4i32;
__t2.ids = unsafe { core::ptr::addr_of!(timedia_quad_port.0) as *const u16 };
__t2
}, {
let mut __t3: timedia_struct = unsafe { core::mem::zeroed() };
__t3.num = 8i32;
__t3.ids = unsafe { core::ptr::addr_of!(timedia_eight_port.0) as *const u16 };
__t3
}]);

/// There are nearly 70 different Timedia/SUNIX PCI serial devices.  Instead of
/// listing them individually, this driver merely grabs them all with
/// PCI_ANY_ID.  Some of these devices, however, also feature a parallel port,
/// and should be left free to be claimed by parport_serial instead.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_timedia_probe")]
extern "C" fn pci_timedia_probe(dev: *mut pci_dev) -> i32 {
// Check the third digit of the subdevice ID
// (0,2,3,5,6: serial only -- 7,8,9: serial + parallel)
if ((unsafe { (*dev).subsystem_device }) as i32) & 240 >= 112 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ignoring Timedia subdevice %04x for parport_serial\n".as_ptr() as *const u8, (*dev).subsystem_device as i32) };
return 19i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_timedia_init")]
extern "C" fn pci_timedia_init(dev: *mut pci_dev) -> i32 {
let mut i: i32 = 0;
let mut j: i32 = 0;
while (i as u64) < ARRAY_SIZE__22!() {
let ids: *const u16 = unsafe { timedia_data.0[i as usize].ids };
j = 0;
while (unsafe { *ids.wrapping_offset(j as isize) }) != 0 {
if ((unsafe { (*dev).subsystem_device }) as i32) == ((unsafe { *ids.wrapping_offset(j as isize) }) as i32) {
return unsafe { timedia_data.0[i as usize].num };
}
j = j.wrapping_add(1);
}
i = i.wrapping_add(1);
}
0
}

/// Timedia/SUNIX uses a mixture of BARs and offsets
/// Ugh, this is ugly as all hell --- TYT
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_timedia_setup")]
extern "C" fn pci_timedia_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let mut bar: u32 = 0;
let mut offset: u32 = unsafe { (*board).first_offset };
// BAR 2
// BAR 3
// BAR 4
// BAR 5
match idx {
0 => {
bar = 0;
}
1 => {
offset = unsafe { (*board).uart_offset };
bar = 0;
}
2 => {
bar = 1;
}
3 => {
offset = unsafe { (*board).uart_offset };
bar = idx.wrapping_sub(2) as u32; // BAR 2
// BAR 3
// BAR 4
// BAR 5
}
4..=7 => {
bar = idx.wrapping_sub(2) as u32; // BAR 2
// BAR 3
// BAR 4
// BAR 5
}
_ => {
}
}
setup_port(r#priv, port, bar as u8, offset, (unsafe { (*board).reg_shift }) as i32)
}

/// Some Titan cards are also a little weird
#[unsafe(link_section = ".text.slate_distinct._8250_pci.titan_400l_800l_setup")]
extern "C" fn titan_400l_800l_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let mut bar: u32 = 0;
let mut offset: u32 = unsafe { (*board).first_offset };
match idx {
0 => {
bar = 1;
}
1 => {
bar = 2;
}
_ => {
bar = 4;
offset = (idx.wrapping_sub(2) as u32).wrapping_mul(unsafe { (*board).uart_offset });
}
}
setup_port(r#priv, port, bar as u8, offset, (unsafe { (*board).reg_shift }) as i32)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_xircom_init")]
extern "C" fn pci_xircom_init(dev: *mut pci_dev) -> i32 {
unsafe { msleep(100u32) };
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_ni8420_init")]
extern "C" fn pci_ni8420_init(dev: *mut pci_dev) -> i32 {
let bar: u32 = 0;
if pci_resource_flags!(dev, bar) & 512 == 0 {
moan_device(c"no memory in bar".as_ptr() as *const u8, dev);
return 0;
}
let p__0: *mut () = unsafe { pci_ioremap_bar(dev, bar as i32) };
if p__0.is_null() {
return 12i32.wrapping_neg();
}
// Enable CPU Interrupt
writel(readl(((unsafe { (p__0 as *const u8).wrapping_offset(56 as isize) }) as *mut ()) as *const ()) | 8192, (unsafe { (p__0 as *const u8).wrapping_offset(56 as isize) }) as *mut ());
unsafe { iounmap(p__0 as *mut ()) };
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_ni8430_init")]
extern "C" fn pci_ni8430_init(dev: *mut pci_dev) -> i32 {
let mut region: pci_bus_region = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(region) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut device_window: u32 = 0;
let bar: u32 = 0;
if pci_resource_flags!(dev, bar) & 512 == 0 {
moan_device(c"no memory in bar".as_ptr() as *const u8, dev);
return 0;
}
let p__0: *mut () = unsafe { pci_ioremap_bar(dev, bar as i32) };
if p__0.is_null() {
return 12i32.wrapping_neg();
}
// Set device window address and size in BAR0, while acknowledging that
// the resource structure may contain a translated address that differs
// from the address the device responds to.
unsafe { pcibios_resource_to_bus((*dev).bus, core::ptr::addr_of_mut!(region), ((*dev).resource.as_mut_ptr() as *mut resource).wrapping_offset(bar as isize)) };
device_window = (region.start.wrapping_add(2048) & 4294967040 | 128u64 | 10) as u32;
writel(device_window, (unsafe { (p__0 as *const u8).wrapping_offset(196 as isize) }) as *mut ());
// Set window access to go to RAMSEL IO address space
writel(readl(((unsafe { (p__0 as *const u8).wrapping_offset(244 as isize) }) as *mut ()) as *const ()) & 4294967294, (unsafe { (p__0 as *const u8).wrapping_offset(244 as isize) }) as *mut ());
// Enable IO Bus Interrupt 0
writel(16777216u32, (unsafe { (p__0 as *const u8).wrapping_offset(8 as isize) }) as *mut ());
// Enable CPU Interrupt
writel(2147483648u32, (unsafe { (p__0 as *const u8).wrapping_offset(16 as isize) }) as *mut ());
unsafe { iounmap(p__0 as *mut ()) };
0
}

// UART Port Control Register
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_ni8430_setup")]
extern "C" fn pci_ni8430_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let dev: *mut pci_dev = unsafe { (*r#priv).dev };
let mut bar: u32 = 0;
let mut offset: u32 = unsafe { (*board).first_offset };
if (idx as u32) >= unsafe { (*board).num_ports } {
return 1;
}
bar = FL_GET_BASE!(unsafe { (*board).flags });
offset = offset.wrapping_add((idx as u32).wrapping_mul(unsafe { (*board).uart_offset }));
let p__0: *mut () = unsafe { pci_ioremap_bar(dev, bar as i32) };
if p__0.is_null() {
return 12i32.wrapping_neg();
}
// enable the transceiver
writeb(((readb(((unsafe { (((p__0 as *const u8).wrapping_offset(offset as isize) as *mut ()) as *const u8).wrapping_offset(15 as isize) }) as *mut ()) as *const ()) as i32) | 8i32) as u8, (unsafe { (((p__0 as *const u8).wrapping_offset(offset as isize) as *mut ()) as *const u8).wrapping_offset(15 as isize) }) as *mut ());
unsafe { iounmap(p__0 as *mut ()) };
setup_port(r#priv, port, bar as u8, offset, (unsafe { (*board).reg_shift }) as i32)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_netmos_9900_setup")]
extern "C" fn pci_netmos_9900_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let mut bar: u32 = 0;
if ((unsafe { (*(*r#priv).dev).device }) as i32) != 39013 && ((unsafe { (*(*r#priv).dev).subsystem_device }) as i32) & 65280 == 12288 {
// netmos apparently orders BARs by datasheet layout, so serial
// ports get BARs 0 and 3 (or 1 and 4 for memmapped)
bar = 3i32.wrapping_mul(idx) as u32;
return setup_port(r#priv, port, bar as u8, 0u32, (unsafe { (*board).reg_shift }) as i32);
}
pci_default_setup(r#priv, board, port, idx)
}

/// the 99xx series comes with a range of device IDs and a variety
/// of capabilities:
///
/// 9900 has varying capabilities and can cascade to sub-controllers
///   (cascading should be purely internal)
/// 9904 is hardwired with 4 serial ports
/// 9912 and 9922 are hardwired with 2 serial ports
fn pci_netmos_9900_numports(dev: *mut pci_dev) -> i32 {
let c: u32 = unsafe { (*dev).class };
let pi: u32 = c & 255;
let mut sub_serports: u16 = 0;
if pi == 2 {
return 1;
}
if pi == 0 && ((unsafe { (*dev).device }) as i32) == 39168 {
// two possibilities: 0x30ps encodes number of parallel and
// serial ports, or 0x1000 indicates *something*. This is not
// immediately obvious, since the 2s1p+4s configuration seems
// to offer all functionality on functions 0..2, while still
// advertising the same function 3 as the 4s+2s1p config.
sub_serports = (((unsafe { (*dev).subsystem_device }) as i32) & 15) as u16;
if (sub_serports as i32) > 0 {
return sub_serports as i32;
}
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"NetMos/Mostech serial driver ignoring port on ambiguous config.\n".as_ptr() as *const u8) };
return 0;
}
moan_device(c"unknown NetMos/Mostech program interface".as_ptr() as *const u8, dev);
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_netmos_init")]
extern "C" fn pci_netmos_init(dev: *mut pci_dev) -> i32 {
// subdevice 0x00PS means <P> parallel, <S> serial
let mut num_serial: u32 = (((unsafe { (*dev).subsystem_device }) as i32) & 15) as u32;
if ((unsafe { (*dev).device }) as i32) == 39169 || ((unsafe { (*dev).device }) as i32) == 39013 {
return 0;
}
if ((unsafe { (*dev).subsystem_vendor }) as i32) == 4116 && ((unsafe { (*dev).subsystem_device }) as i32) == 665 {
return 0;
}
match (unsafe { (*dev).device }) as i32 {
37120 | 39168 | 39172 | 39186 | 39202 => {
// FALLTHROUGH on all
num_serial = pci_netmos_9900_numports(dev) as u32;
}
_ => {
}
}
if num_serial == 0 {
moan_device(c"unknown NetMos/Mostech device".as_ptr() as *const u8, dev);
return 19i32.wrapping_neg();
}
num_serial as i32
}

// These chips are available with optionally one parallel port and up to
// two serial ports. Unfortunately they all have the same product id.
//
// Basic configuration is done over a region of 32 I/O ports. The base
// ioport is called INTA or INTC, depending on docs/other drivers.
//
// The region of the 32 I/O ports is configured in POSIO0R...
// registers
// I/O space size
// I/O space size (bits 26-24; 8 bytes = 011b)
// I/O space size (bits 26-24; 32 bytes = 101b)
// Decoding speed (1 = slow, 2 = medium, 3 = fast)
// enable IO_Space bit
/// inta_addr are the configuration addresses of the ITE
static inta_addr: [i16; 7] = [672, 704, 544, 576, 480, 512, 640];

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_ite887x_init")]
extern "C" fn pci_ite887x_init(dev: *mut pci_dev) -> i32 {
let mut ret: i32 = 0;
// search for the base-ioport
let mut i: i32 = 0;
let mut r#type: i32 = 0;
let mut iobase: *mut resource = core::ptr::null_mut::<resource>();
let mut miscr: u32 = 0;
let mut uartbar: u32 = 0;
let mut ioport: u32 = 0;
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport(dev) };
}
while (i as u64) < ARRAY_SIZE__2!() {
iobase = request_region__2!(unsafe { inta_addr[i as usize] }, 32i32, c"ite887x".as_ptr() as *mut u8);
if !iobase.is_null() {
// write POSIO0R - speed | size | ioport
unsafe { pci_write_config_dword(dev as *const pci_dev, 96i32, (-452984832i32 | (inta_addr[i as usize] as i32)) as u32) };
// write INTCBAR - ioport
unsafe { pci_write_config_dword(dev as *const pci_dev, 120i32, inta_addr[i as usize] as u32) };
ret = __inb((unsafe { inta_addr[i as usize] }) as u16) as i32;
if ret != 255 {
// ioport connected
break;
}
release_region!(unsafe { (*iobase).start }, 32i32);
}
i = i.wrapping_add(1);
}
if (i as u64) == ARRAY_SIZE__36!() {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"could not find iobase\n".as_ptr() as *const u8) };
return 19i32.wrapping_neg();
}
// start of undocumented type checking (see parport_pc.c)
r#type = (__inb(unsafe { (*iobase).start }.wrapping_add(24) as u16) as i32) & 15;
match r#type {
2 | 10 => {
ret = 0; // ITE8871 (1P)
// ITE8875 (1P)
}
14 => {
ret = 2; // ITE8872 (2S1P)
}
6 => {
ret = 1; // ITE8873 (1S)
}
8 => {
ret = 2; // ITE8874 (2S)
}
_ => {
moan_device(c"Unknown ITE887x".as_ptr() as *const u8, dev);
ret = 19i32.wrapping_neg();
}
}
// configure all serial ports
i = 0;
while i < ret {
// read the I/O port from the device
unsafe { pci_read_config_dword(dev as *const pci_dev, 16i32.wrapping_add(4i32.wrapping_mul(i.wrapping_add(1))), core::ptr::addr_of_mut!(ioport)) };
ioport &= 65280; // the actual base address
unsafe { pci_write_config_dword(dev as *const pci_dev, 96i32.wrapping_add(4i32.wrapping_mul(i.wrapping_add(1))), 3808428032u32 | ioport) };
// write the ioport to the UARTBAR
unsafe { pci_read_config_dword(dev as *const pci_dev, 124i32, core::ptr::addr_of_mut!(uartbar)) };
let __v8: u32 = uartbar & (!(65535i32 << 16i32.wrapping_mul(i)) as u32);
uartbar = __v8; // clear half the reg
uartbar |= ioport << 16i32.wrapping_mul(i); // set the ioport
unsafe { pci_write_config_dword(dev as *const pci_dev, 124i32, uartbar) };
// get current config
unsafe { pci_read_config_dword(dev as *const pci_dev, 156i32, core::ptr::addr_of_mut!(miscr)) };
// disable interrupts (UARTx_Routing[3:0])
let __v12: u32 = miscr & (!(15i32 << 12i32.wrapping_sub(4i32.wrapping_mul(i))) as u32);
miscr = __v12;
// activate the UART (UARTx_En)
miscr |= (1i32 << 23i32.wrapping_sub(i)) as u32;
// write new config with activated UART
unsafe { pci_write_config_dword(dev as *const pci_dev, 156i32, miscr) };
i = i.wrapping_add(1);
}
if ret <= 0 {
// the device has no UARTs if we get here
release_region!(unsafe { (*iobase).start }, 32i32);
}
ret
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_ite887x_exit")]
extern "C" fn pci_ite887x_exit(dev: *mut pci_dev) {
let mut ioport: u32 = 0;
// the ioport is bit 0-15 in POSIO0R
unsafe { pci_read_config_dword(dev as *const pci_dev, 96i32, core::ptr::addr_of_mut!(ioport)) };
ioport &= 65535;
release_region__2!(ioport, 32i32);
}

// Oxford Semiconductor Inc.
// Check if an OxSemi device is part of the Tornado range of devices.
fn pci_oxsemi_tornado_p(dev: *mut pci_dev) -> bool {
// OxSemi Tornado devices are all 0xCxxx
if ((unsafe { (*dev).vendor }) as i32) == 5141 && ((unsafe { (*dev).device }) as i32) & 61440 != 49152 {
return false;
}
// EndRun devices are all 0xExxx
if ((unsafe { (*dev).vendor }) as i32) == 29697 && ((unsafe { (*dev).device }) as i32) & 61440 != 57344 {
return false;
}
true
}

/// Determine the number of ports available on a Tornado device.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_oxsemi_tornado_init")]
extern "C" fn pci_oxsemi_tornado_init(dev: *mut pci_dev) -> i32 {
let mut deviceID: u64 = 0;
let mut number_uarts: u32 = 0;
if !pci_oxsemi_tornado_p(dev) {
return 0;
}
let p__0: *mut u8 = (unsafe { pci_iomap(dev, 0i32, 5u64) }) as *mut u8;
if p__0.is_null() {
return 12i32.wrapping_neg();
}
deviceID = (unsafe { ioread32(p__0 as *const ()) }) as u64;
// Tornado device
if deviceID == 117441024 {
number_uarts = unsafe { ioread8(p__0.wrapping_offset(4 as isize) as *const ()) };
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%d ports detected on %s PCI Express device\n".as_ptr() as *const u8, number_uarts, if ((*dev).vendor as i32) == 29697 { c"EndRun".as_ptr() as *mut u8 } else { c"Oxford".as_ptr() as *mut u8 }) };
}
}
unsafe { pci_iounmap(dev, p__0 as *mut ()) };
number_uarts as i32
}

// Tornado-specific constants for the TCR and CPR registers; see below.
/// Determine the oversampling rate, the clock prescaler, and the clock
/// divisor for the requested baud rate.  The clock rate is 62.5 MHz,
/// which is four times the baud base, and the prescaler increments in
/// steps of 1/8.  Therefore to make calculations on integers we need
/// to use a scaled clock rate, which is the baud base multiplied by 32
/// (or our assumed UART clock rate multiplied by 2).
///
/// The allowed oversampling rates are from 4 up to 16 inclusive (values
/// from 0 to 3 inclusive map to 16).  Likewise the clock prescaler allows
/// values between 1.000 and 63.875 inclusive (operation for values from
/// 0.000 to 0.875 has not been specified).  The clock divisor is the usual
/// unsigned 16-bit integer.
///
/// For the most accurate baud rate we use a table of predetermined
/// oversampling rates and clock prescalers that records all possible
/// products of the two parameters in the range from 4 up to 255 inclusive,
/// and additionally 335 for the 1500000bps rate, with the prescaler scaled
/// by 8.  The table is sorted by the decreasing value of the oversampling
/// rate and ties are resolved by sorting by the decreasing value of the
/// product.  This way preference is given to higher oversampling rates.
///
/// We iterate over the table and choose the product of an oversampling
/// rate and a clock prescaler that gives the lowest integer division
/// result deviation, or if an exact integer divider is found we stop
/// looking for it right away.  We do some fixup if the resulting clock
/// divisor required would be out of its unsigned 16-bit integer range.
///
/// Finally we abuse the supposed fractional part returned to encode the
/// 4-bit value of the oversampling rate and the 9-bit value of the clock
/// prescaler which will end up in the TCR and CPR/CPR2 registers.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_oxsemi_tornado_get_divisor")]
extern "C" fn pci_oxsemi_tornado_get_divisor(port: *mut uart_port, baud: u32, frac: *mut u32) -> u32 {
// Scale the quotient for comparison to get the fractional part.
let quot_scale: u32 = 65536;
let sclk: u32 = unsafe { (*port).uartclk }.wrapping_mul(2);
let mut sdiv: u32 = 0;
let mut __v0: u32;
{
let __x: u32 = sclk;
let __d: u32 = baud;
__v0 = if (1i32.wrapping_neg() as u32) > 0 || (1i32.wrapping_neg() as u32) > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add(__d / 2) / __d } else { __x.wrapping_sub(__d / 2) / __d };
}
sdiv = __v0;
let mut best_squot: u32 = quot_scale;
let mut squot: u32 = 0;
let mut quot: u32 = 0;
let mut cpr: u16 = 0;
let mut tcr: u8 = 0;
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__12!() {
let mut spre: u32 = 0;
let mut srem: u32 = 0;
let mut cp: u8 = 0;
let tc: u8 = unsafe { p.0[i as usize][0] };
cp = unsafe { p.0[i as usize][1] };
spre = (tc as i32).wrapping_mul(cp as i32) as u32;
srem = sdiv % spre;
if srem > spre / 2 {
srem = spre.wrapping_sub(srem);
}
let mut __v1: u32;
{
let __x: u32 = srem.wrapping_mul(quot_scale);
let __d: u32 = spre;
__v1 = if (1i32.wrapping_neg() as u32) > 0 || (1i32.wrapping_neg() as u32) > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add(__d / 2) / __d } else { __x.wrapping_sub(__d / 2) / __d };
}
squot = __v1;
if srem == 0 {
tcr = tc;
cpr = cp as u16;
quot = sdiv / spre;
break;
} else {
if squot < best_squot {
best_squot = squot;
tcr = tc;
cpr = cp as u16;
let mut __v3: u32;
let __x: u32 = sdiv;
let __d: u32 = spre;
__v3 = if (1i32.wrapping_neg() as u32) > 0 || (1i32.wrapping_neg() as u32) > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add(__d / 2) / __d } else { __x.wrapping_sub(__d / 2) / __d };
quot = __v3;
}
}
i = i.wrapping_add(1);
}
while (tcr as i32) <= 8i32 && quot % 2 == 0 {
quot >>= 1i32;
tcr = ((tcr as i32) << 1i32) as u8;
}
while quot > 65535 {
if (tcr as i32) <= 8i32 {
quot >>= 1i32;
tcr = ((tcr as i32) << 1i32) as u8;
} else {
if (cpr as i32) <= 255i32 {
quot >>= 1i32;
cpr = ((cpr as i32) << 1i32) as u16;
} else {
quot = quot.wrapping_mul((cpr as i32) as u32) / 511;
cpr = 511;
}
}
}
unsafe {
*frac = ((cpr as i32) << 8i32 | (tcr as i32) & 15) as u32;
}
quot
}

static mut p: __SlateAlign16<[[u8; 2]; 134]> = __SlateAlign16([[16u8, 14u8], [16u8, 13u8], [16u8, 12u8], [16u8, 11u8], [16u8, 10u8], [16u8, 9u8], [16u8, 8u8], [15u8, 17u8], [15u8, 16u8], [15u8, 15u8], [15u8, 14u8], [15u8, 13u8], [15u8, 12u8], [15u8, 11u8], [15u8, 10u8], [15u8, 9u8], [15u8, 8u8], [14u8, 18u8], [14u8, 17u8], [14u8, 14u8], [14u8, 13u8], [14u8, 12u8], [14u8, 11u8], [14u8, 10u8], [14u8, 9u8], [14u8, 8u8], [13u8, 19u8], [13u8, 18u8], [13u8, 17u8], [13u8, 13u8], [13u8, 12u8], [13u8, 11u8], [13u8, 10u8], [13u8, 9u8], [13u8, 8u8], [12u8, 19u8], [12u8, 18u8], [12u8, 17u8], [12u8, 11u8], [12u8, 9u8], [12u8, 8u8], [11u8, 23u8], [11u8, 22u8], [11u8, 21u8], [11u8, 20u8], [11u8, 19u8], [11u8, 18u8], [11u8, 17u8], [11u8, 11u8], [11u8, 10u8], [11u8, 9u8], [11u8, 8u8], [10u8, 25u8], [10u8, 23u8], [10u8, 20u8], [10u8, 19u8], [10u8, 17u8], [10u8, 10u8], [10u8, 9u8], [10u8, 8u8], [9u8, 27u8], [9u8, 23u8], [9u8, 21u8], [9u8, 19u8], [9u8, 18u8], [9u8, 17u8], [9u8, 9u8], [9u8, 8u8], [8u8, 31u8], [8u8, 29u8], [8u8, 23u8], [8u8, 19u8], [8u8, 17u8], [8u8, 8u8], [7u8, 35u8], [7u8, 31u8], [7u8, 29u8], [7u8, 25u8], [7u8, 23u8], [7u8, 21u8], [7u8, 19u8], [7u8, 17u8], [7u8, 15u8], [7u8, 14u8], [7u8, 13u8], [7u8, 12u8], [7u8, 11u8], [7u8, 10u8], [7u8, 9u8], [7u8, 8u8], [6u8, 41u8], [6u8, 37u8], [6u8, 31u8], [6u8, 29u8], [6u8, 23u8], [6u8, 19u8], [6u8, 17u8], [6u8, 13u8], [6u8, 11u8], [6u8, 10u8], [6u8, 9u8], [6u8, 8u8], [5u8, 67u8], [5u8, 47u8], [5u8, 43u8], [5u8, 41u8], [5u8, 37u8], [5u8, 31u8], [5u8, 29u8], [5u8, 25u8], [5u8, 23u8], [5u8, 19u8], [5u8, 17u8], [5u8, 15u8], [5u8, 13u8], [5u8, 11u8], [5u8, 10u8], [5u8, 9u8], [5u8, 8u8], [4u8, 61u8], [4u8, 59u8], [4u8, 53u8], [4u8, 47u8], [4u8, 43u8], [4u8, 41u8], [4u8, 37u8], [4u8, 31u8], [4u8, 29u8], [4u8, 23u8], [4u8, 19u8], [4u8, 17u8], [4u8, 13u8], [4u8, 9u8], [4u8, 8u8]]);

/// Set the oversampling rate in the transmitter clock cycle register (TCR),
/// the clock prescaler in the clock prescaler register (CPR and CPR2), and
/// the clock divisor in the divisor latch (DLL and DLM).  Note that for
/// backwards compatibility any write to CPR clears CPR2 and therefore CPR
/// has to be written first, followed by CPR2, which occupies the location
/// of CKS used with earlier UART designs.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_oxsemi_tornado_set_divisor")]
extern "C" fn pci_oxsemi_tornado_set_divisor(port: *mut uart_port, baud: u32, quot: u32, quot_frac: u32) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let cpr2: u8 = (quot_frac >> 16i32) as u8;
let cpr: u8 = (quot_frac >> 8i32) as u8;
let tcr: u8 = quot_frac as u8;
serial_icr_write(up, 2i32, tcr as i32);
serial_icr_write(up, 1i32, cpr as i32);
serial_icr_write(up, 3i32, cpr2 as i32);
unsafe { serial8250_do_set_divisor(port, baud, quot) };
}

/// For Tornado devices we force MCR[7] set for the Divide-by-M N/8 baud rate
/// generator prescaler (CPR and CPR2).  Otherwise no prescaler would be used.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_oxsemi_tornado_set_mctrl")]
extern "C" fn pci_oxsemi_tornado_set_mctrl(port: *mut uart_port, mctrl: u32) {
let up: *mut uart_8250_port = up_to_u8250p(port);
unsafe {
(*up).mcr = (((*up).mcr as i32) | 128) as u8;
}
unsafe { serial8250_do_set_mctrl(port, mctrl) };
}

/// We require EFR features for clock programming, so set UPF_FULL_PROBE
/// for full probing regardless of CONFIG_SERIAL_8250_16550A_VARIANTS setting.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_oxsemi_tornado_setup")]
extern "C" fn pci_oxsemi_tornado_setup(r#priv: *mut serial_private, board: *const pciserial_board, up: *mut uart_8250_port, idx: i32) -> i32 {
let dev: *mut pci_dev = unsafe { (*r#priv).dev };
if pci_oxsemi_tornado_p(dev) {
let __v1: u64 = unsafe { (*up).port.flags };
let __v2: u64 = __v1 | BIT_ULL!(32i32);
unsafe {
(*up).port.flags = __v2;
}
unsafe {
(*up).port.get_divisor = Some(pci_oxsemi_tornado_get_divisor);
}
unsafe {
(*up).port.set_divisor = Some(pci_oxsemi_tornado_set_divisor);
}
unsafe {
(*up).port.set_mctrl = Some(pci_oxsemi_tornado_set_mctrl);
}
}
pci_default_setup(r#priv, board, up, idx)
}

/// Quatech devices have their own extra interface features
static mut quatech_cards: __SlateAlign16<[pci_device_id; 20]> = __SlateAlign16([{
let mut __t0: pci_device_id = unsafe { core::mem::zeroed() };
__t0.vendor = 4956u32;
__t0.device = 16u32;
__t0.subvendor = 4294967295u32;
__t0.subdevice = 4294967295u32;
__t0.class = 0u32;
__t0.class_mask = 0u32;
__t0.driver_data = 1u64;
__t0
}, {
let mut __t1: pci_device_id = unsafe { core::mem::zeroed() };
__t1.vendor = 4956u32;
__t1.device = 32u32;
__t1.subvendor = 4294967295u32;
__t1.subdevice = 4294967295u32;
__t1.class = 0u32;
__t1.class_mask = 0u32;
__t1.driver_data = 1u64;
__t1
}, {
let mut __t2: pci_device_id = unsafe { core::mem::zeroed() };
__t2.vendor = 4956u32;
__t2.device = 385u32;
__t2.subvendor = 4294967295u32;
__t2.subdevice = 4294967295u32;
__t2.class = 0u32;
__t2.class_mask = 0u32;
__t2.driver_data = 0u64;
__t2
}, {
let mut __t3: pci_device_id = unsafe { core::mem::zeroed() };
__t3.vendor = 4956u32;
__t3.device = 48u32;
__t3.subvendor = 4294967295u32;
__t3.subdevice = 4294967295u32;
__t3.class = 0u32;
__t3.class_mask = 0u32;
__t3.driver_data = 1u64;
__t3
}, {
let mut __t4: pci_device_id = unsafe { core::mem::zeroed() };
__t4.vendor = 4956u32;
__t4.device = 433u32;
__t4.subvendor = 4294967295u32;
__t4.subdevice = 4294967295u32;
__t4.class = 0u32;
__t4.class_mask = 0u32;
__t4.driver_data = 0u64;
__t4
}, {
let mut __t5: pci_device_id = unsafe { core::mem::zeroed() };
__t5.vendor = 4956u32;
__t5.device = 80u32;
__t5.subvendor = 4294967295u32;
__t5.subdevice = 4294967295u32;
__t5.class = 0u32;
__t5.class_mask = 0u32;
__t5.driver_data = 1u64;
__t5
}, {
let mut __t6: pci_device_id = unsafe { core::mem::zeroed() };
__t6.vendor = 4956u32;
__t6.device = 96u32;
__t6.subvendor = 4294967295u32;
__t6.subdevice = 4294967295u32;
__t6.class = 0u32;
__t6.class_mask = 0u32;
__t6.driver_data = 1u64;
__t6
}, {
let mut __t7: pci_device_id = unsafe { core::mem::zeroed() };
__t7.vendor = 4956u32;
__t7.device = 288u32;
__t7.subvendor = 4294967295u32;
__t7.subdevice = 4294967295u32;
__t7.class = 0u32;
__t7.class_mask = 0u32;
__t7.driver_data = 1u64;
__t7
}, {
let mut __t8: pci_device_id = unsafe { core::mem::zeroed() };
__t8.vendor = 4956u32;
__t8.device = 304u32;
__t8.subvendor = 4294967295u32;
__t8.subdevice = 4294967295u32;
__t8.class = 0u32;
__t8.class_mask = 0u32;
__t8.driver_data = 1u64;
__t8
}, {
let mut __t9: pci_device_id = unsafe { core::mem::zeroed() };
__t9.vendor = 4956u32;
__t9.device = 320u32;
__t9.subvendor = 4294967295u32;
__t9.subdevice = 4294967295u32;
__t9.class = 0u32;
__t9.class_mask = 0u32;
__t9.driver_data = 1u64;
__t9
}, {
let mut __t10: pci_device_id = unsafe { core::mem::zeroed() };
__t10.vendor = 4956u32;
__t10.device = 336u32;
__t10.subvendor = 4294967295u32;
__t10.subdevice = 4294967295u32;
__t10.class = 0u32;
__t10.class_mask = 0u32;
__t10.driver_data = 1u64;
__t10
}, {
let mut __t11: pci_device_id = unsafe { core::mem::zeroed() };
__t11.vendor = 4956u32;
__t11.device = 480u32;
__t11.subvendor = 4294967295u32;
__t11.subdevice = 4294967295u32;
__t11.class = 0u32;
__t11.class_mask = 0u32;
__t11.driver_data = 0u64;
__t11
}, {
let mut __t12: pci_device_id = unsafe { core::mem::zeroed() };
__t12.vendor = 4956u32;
__t12.device = 368u32;
__t12.subvendor = 4294967295u32;
__t12.subdevice = 4294967295u32;
__t12.class = 0u32;
__t12.class_mask = 0u32;
__t12.driver_data = 0u64;
__t12
}, {
let mut __t13: pci_device_id = unsafe { core::mem::zeroed() };
__t13.vendor = 4956u32;
__t13.device = 384u32;
__t13.subvendor = 4294967295u32;
__t13.subdevice = 4294967295u32;
__t13.class = 0u32;
__t13.class_mask = 0u32;
__t13.driver_data = 0u64;
__t13
}, {
let mut __t14: pci_device_id = unsafe { core::mem::zeroed() };
__t14.vendor = 4956u32;
__t14.device = 400u32;
__t14.subvendor = 4294967295u32;
__t14.subdevice = 4294967295u32;
__t14.class = 0u32;
__t14.class_mask = 0u32;
__t14.driver_data = 0u64;
__t14
}, {
let mut __t15: pci_device_id = unsafe { core::mem::zeroed() };
__t15.vendor = 4956u32;
__t15.device = 416u32;
__t15.subvendor = 4294967295u32;
__t15.subdevice = 4294967295u32;
__t15.class = 0u32;
__t15.class_mask = 0u32;
__t15.driver_data = 0u64;
__t15
}, {
let mut __t16: pci_device_id = unsafe { core::mem::zeroed() };
__t16.vendor = 4956u32;
__t16.device = 432u32;
__t16.subvendor = 4294967295u32;
__t16.subdevice = 4294967295u32;
__t16.class = 0u32;
__t16.class_mask = 0u32;
__t16.driver_data = 0u64;
__t16
}, {
let mut __t17: pci_device_id = unsafe { core::mem::zeroed() };
__t17.vendor = 4956u32;
__t17.device = 448u32;
__t17.subvendor = 4294967295u32;
__t17.subdevice = 4294967295u32;
__t17.class = 0u32;
__t17.class_mask = 0u32;
__t17.driver_data = 0u64;
__t17
}, {
let mut __t18: pci_device_id = unsafe { core::mem::zeroed() };
__t18.vendor = 4956u32;
__t18.device = 632u32;
__t18.subvendor = 4294967295u32;
__t18.subdevice = 4294967295u32;
__t18.class = 0u32;
__t18.class_mask = 0u32;
__t18.driver_data = 0u64;
__t18
}, {
let mut __t19: pci_device_id = unsafe { core::mem::zeroed() };
__t19.vendor = 0u32;
__t19
}]);

fn pci_quatech_rqopr(port: *mut uart_8250_port) -> i32 {
let base: u64 = unsafe { (*port).port.iobase };
let LCR: u8 = __inb(base.wrapping_add(3) as u16);
let mut val: u8 = 0;
__outb(191u8, base.wrapping_add(3) as u16);
val = __inb(base.wrapping_add(7) as u16);
__outb(LCR, base.wrapping_add(3) as u16);
val as i32
}

fn pci_quatech_wqopr(port: *mut uart_8250_port, qopr: u8) {
let base: u64 = unsafe { (*port).port.iobase };
let LCR: u8 = __inb(base.wrapping_add(3) as u16);
__outb(191u8, base.wrapping_add(3) as u16);
__inb(base.wrapping_add(7) as u16);
__outb(qopr, base.wrapping_add(7) as u16);
__outb(LCR, base.wrapping_add(3) as u16);
}

fn pci_quatech_rqmcr(port: *mut uart_8250_port) -> i32 {
let base: u64 = unsafe { (*port).port.iobase };
let LCR: u8 = __inb(base.wrapping_add(3) as u16);
let mut val: u8 = 0;
let mut qmcr: u8 = 0;
__outb(191u8, base.wrapping_add(3) as u16);
val = __inb(base.wrapping_add(7) as u16);
__outb(((val as i32) | 16) as u8, base.wrapping_add(7) as u16);
qmcr = __inb(base.wrapping_add(4) as u16);
__outb(val, base.wrapping_add(7) as u16);
__outb(LCR, base.wrapping_add(3) as u16);
qmcr as i32
}

fn pci_quatech_wqmcr(port: *mut uart_8250_port, qmcr: u8) {
let base: u64 = unsafe { (*port).port.iobase };
let LCR: u8 = __inb(base.wrapping_add(3) as u16);
let mut val: u8 = 0;
__outb(191u8, base.wrapping_add(3) as u16);
val = __inb(base.wrapping_add(7) as u16);
__outb(((val as i32) | 16) as u8, base.wrapping_add(7) as u16);
__outb(qmcr, base.wrapping_add(4) as u16);
__outb(val, base.wrapping_add(7) as u16);
__outb(LCR, base.wrapping_add(3) as u16);
}

fn pci_quatech_has_qmcr(port: *mut uart_8250_port) -> i32 {
let base: u64 = unsafe { (*port).port.iobase };
let LCR: u8 = __inb(base.wrapping_add(3) as u16);
let mut val: u8 = 0;
__outb(191u8, base.wrapping_add(3) as u16);
val = __inb(base.wrapping_add(7) as u16);
if (val as i32) & 32 != 0 {
__outb(128u8, 3u16);
if (__inb(7u16) as i32) & 32 == 0 {
__outb(LCR, base.wrapping_add(3) as u16);
return 1;
}
}
0
}

fn pci_quatech_test(port: *mut uart_8250_port) -> i32 {
let mut reg: u8 = 0;
let qopr: u8 = pci_quatech_rqopr(port) as u8;
pci_quatech_wqopr(port, ((qopr as i32) & 63) as u8);
reg = (pci_quatech_rqopr(port) & 192) as u8;
if (reg as i32) != 0 {
return 22i32.wrapping_neg();
}
pci_quatech_wqopr(port, ((qopr as i32) & 63 | 64) as u8);
reg = (pci_quatech_rqopr(port) & 192) as u8;
if (reg as i32) != 64 {
return 22i32.wrapping_neg();
}
pci_quatech_wqopr(port, ((qopr as i32) & 63 | 128) as u8);
reg = (pci_quatech_rqopr(port) & 192) as u8;
if (reg as i32) != 64 {
return 22i32.wrapping_neg();
}
pci_quatech_wqopr(port, ((qopr as i32) & 63 | 192) as u8);
reg = (pci_quatech_rqopr(port) & 192) as u8;
if (reg as i32) != 128 {
return 22i32.wrapping_neg();
}
pci_quatech_wqopr(port, qopr);
0
}

fn pci_quatech_clock(port: *mut uart_8250_port) -> i32 {
let mut clock: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(clock) as *mut u64;
let mut set: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u8 = core::ptr::addr_of_mut!(set) as *mut u8;
let mut reg: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u8 = core::ptr::addr_of_mut!(reg) as *mut u8;
let mut qopr: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u8 = core::ptr::addr_of_mut!(qopr) as *mut u8;
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if pci_quatech_test(port) < 0 {
return 1843200;
} else {
*__slate_slot_9 = pci_quatech_rqopr(port) as u8;
pci_quatech_wqopr(port, ((*__slate_slot_9 as i32) & !3) as u8);
*__slate_slot_8 = pci_quatech_rqopr(port) as u8;
if (*__slate_slot_8 as i32) & 3 != 0 {
*__slate_slot_6 = 1843200;
} else {
pci_quatech_wqopr(port, ((*__slate_slot_9 as i32) | 3) as u8);
*__slate_slot_8 = pci_quatech_rqopr(port) as u8;
if (*__slate_slot_8 as i32) & 3 == 0 {
*__slate_slot_6 = 1843200;
} else {
*__slate_slot_8 = ((*__slate_slot_8 as i32) & 3) as u8;
if (*__slate_slot_8 as i32) == 1 {
*__slate_slot_6 = 3685400;
*__slate_slot_7 = 1;
} else {
if (*__slate_slot_8 as i32) == 2 {
*__slate_slot_6 = 7372800;
*__slate_slot_7 = 2;
} else {
if (*__slate_slot_8 as i32) == 3 {
*__slate_slot_6 = 14745600;
*__slate_slot_7 = 3;
} else {
*__slate_slot_6 = 1843200;
*__slate_slot_7 = 0;
}
}
}
let __v2: u8 = ((*__slate_slot_9 as i32) & !3) as u8;
*__slate_slot_9 = __v2;
*__slate_slot_9 = ((*__slate_slot_9 as i32) | (*__slate_slot_7 as i32)) as u8;
}
}
pci_quatech_wqopr(port, *__slate_slot_9);
return *__slate_slot_6 as i32;
}
}
}

fn pci_quatech_rs422(port: *mut uart_8250_port) -> i32 {
let mut qmcr: u8 = 0;
let mut rs422: i32 = 0;
if pci_quatech_has_qmcr(port) == 0 {
return 0;
}
qmcr = pci_quatech_rqmcr(port) as u8;
pci_quatech_wqmcr(port, 255u8);
if pci_quatech_rqmcr(port) != 0 {
rs422 = 1;
}
pci_quatech_wqmcr(port, qmcr);
rs422
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_quatech_init")]
extern "C" fn pci_quatech_init(dev: *mut pci_dev) -> i32 {
let mut amcc: bool = false;
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport(dev) };
}
let r#match: *const pci_device_id = unsafe { pci_match_id((core::ptr::addr_of_mut!(quatech_cards.0) as *mut pci_device_id) as *const pci_device_id, dev) };
if !r#match.is_null() {
amcc = (unsafe { (*r#match).driver_data }) != 0;
} else {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"unknown port type '0x%04X'.\n".as_ptr() as *const u8, (*dev).device as i32) };
}
if amcc {
let base: u64 = pci_resource_start!(dev, 0i32);
if base != 0 {
let mut tmp: u32 = 0;
__outl(__inl(base.wrapping_add(56) as u16) | 8192, base.wrapping_add(56) as u16);
tmp = __inl(base.wrapping_add(60) as u16);
__outl(tmp | 16777216, base.wrapping_add(60) as u16);
__outl(tmp & 4278190079u32, base.wrapping_add(60) as u16);
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_quatech_setup")]
extern "C" fn pci_quatech_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport((*r#priv).dev) };
}
// Needed by pci_quatech calls below
unsafe {
(*port).port.iobase = pci_resource_start!(unsafe { (*r#priv).dev }, FL_GET_BASE!(unsafe { (*board).flags }));
}
// Set up the clocking
unsafe {
(*port).port.uartclk = pci_quatech_clock(port) as u32;
}
// For now just warn about RS422
if pci_quatech_rs422(port) != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*(*r#priv).dev).dev) as *const device, c"software control of RS422 features not currently supported.\n".as_ptr() as *const u8) };
}
pci_default_setup(r#priv, board, port, idx)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_default_setup")]
extern "C" fn pci_default_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let mut bar: u32 = 0;
let mut offset: u32 = unsafe { (*board).first_offset };
let mut maxnr: u32 = 0;
bar = FL_GET_BASE!(unsafe { (*board).flags });
if (unsafe { (*board).flags }) & 8 != 0 {
bar = bar.wrapping_add(idx as u32);
} else {
offset = offset.wrapping_add((idx as u32).wrapping_mul(unsafe { (*board).uart_offset }));
}
maxnr = (pci_resource_len!(unsafe { (*r#priv).dev }, bar).wrapping_sub((unsafe { (*board).first_offset }) as u64) >> unsafe { (*board).reg_shift }.wrapping_add(3)) as u32;
if (unsafe { (*board).flags }) & 256 != 0 && (idx as u32) >= maxnr {
return 1;
}
setup_port(r#priv, port, bar as u8, offset, (unsafe { (*board).reg_shift }) as i32)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.ce4100_serial_setup")]
extern "C" fn ce4100_serial_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let ret: i32 = setup_port(r#priv, port, idx as u8, 0u32, (unsafe { (*board).reg_shift }) as i32);
unsafe {
(*port).port.iotype = 3;
}
unsafe {
(*port).port.r#type = 15;
}
unsafe {
(*port).port.flags = (*port).port.flags | BIT_ULL!(29i32) | BIT_ULL!(27i32);
}
unsafe {
(*port).port.regshift = 2;
}
ret
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_omegapci_setup")]
extern "C" fn pci_omegapci_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
setup_port(r#priv, port, 2u8, idx.wrapping_mul(8) as u32, 0i32)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_brcm_trumanage_setup")]
extern "C" fn pci_brcm_trumanage_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let ret: i32 = pci_default_setup(r#priv, board, port, idx);
unsafe {
(*port).port.r#type = 25;
}
unsafe {
(*port).port.flags = (*port).port.flags | BIT_ULL!(29i32) | BIT_ULL!(27i32);
}
ret
}

// RTS will control by MCR if this bit is 0
// only worked with FINTEK_RTS_CONTROL_BY_HW on
/// We should do proper H/W transceiver setting before change to RS485 mode
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_fintek_rs485_config")]
extern "C" fn pci_fintek_rs485_config(port: *mut uart_port, termios: *mut ktermios, rs485: *mut serial_rs485) -> i32 {
let pci_dev: *mut pci_dev = to_pci_dev!(unsafe { (*port).dev });
let mut setting: u8 = 0;
let index: *mut u8 = (unsafe { (*port).private_data }) as *mut u8;
unsafe { pci_read_config_byte(pci_dev as *const pci_dev, 64i32.wrapping_add(8i32.wrapping_mul(*index as i32)).wrapping_add(7), core::ptr::addr_of_mut!(setting)) };
if ((unsafe { (*rs485).flags }) as u64) & _BITUL!(0i32) != 0 {
// Enable RTS H/W control mode
let __v1: u8 = setting;
setting = (((__v1 as i32) as u64) | BIT!(4i32)) as u8;
if ((unsafe { (*rs485).flags }) as u64) & _BITUL!(1i32) != 0 {
// RTS driving high on TX
let __v3: u8 = setting;
setting = (((__v3 as i32) as u64) & !BIT!(5i32)) as u8;
} else {
// RTS driving low on TX
let __v5: u8 = setting;
setting = (((__v5 as i32) as u64) | BIT!(5i32)) as u8;
}
} else {
// Disable RTS H/W control mode
let __v7: u8 = setting;
setting = (((__v7 as i32) as u64) & !(BIT!(4i32) | BIT!(5i32))) as u8;
}
unsafe { pci_write_config_byte(pci_dev as *const pci_dev, 64i32.wrapping_add(8i32.wrapping_mul(*index as i32)).wrapping_add(7), setting) };
0
}

/// F81504/508/512 does not support RTS delay before or after send
static pci_fintek_rs485_supported: serial_rs485 = {
let mut __t0: serial_rs485 = unsafe { core::mem::zeroed() };
__t0.flags = 7u32;
__t0
};

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_fintek_setup")]
extern "C" fn pci_fintek_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let pdev: *mut pci_dev = unsafe { (*r#priv).dev };
let mut config_base: u8 = 0;
let mut iobase: u16 = 0;
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport(pdev) };
}
config_base = 64i32.wrapping_add(8i32.wrapping_mul(idx)) as u8;
// Get the io address from configuration space
unsafe { pci_read_config_word(pdev as *const pci_dev, (config_base as i32).wrapping_add(4), core::ptr::addr_of_mut!(iobase)) };
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"idx=%d iobase=0x%x".as_ptr() as *const u8, idx, iobase as i32) };
}
unsafe {
(*port).port.iotype = 0;
}
unsafe {
(*port).port.iobase = iobase as u64;
}
unsafe {
(*port).port.rs485_config = Some(pci_fintek_rs485_config);
}
unsafe {
(*port).port.rs485_supported = pci_fintek_rs485_supported;
}
let data: *mut u8 = devm_kzalloc(unsafe { core::ptr::addr_of_mut!((*pdev).dev) }, core::mem::size_of::<u8>() as u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *mut u8;
if data.is_null() {
return 12i32.wrapping_neg();
}
// preserve index in PCI configuration space
unsafe {
*data = idx as u8;
}
unsafe {
(*port).port.private_data = data as *mut ();
}
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_fintek_init")]
extern "C" fn pci_fintek_init(dev: *mut pci_dev) -> i32 {
let mut iobase: u64 = 0;
let mut max_port: u32 = 0;
let mut bar_data: __SlateAlign16<[u64; 3]> = __SlateAlign16([0u64; 3]);
let mut config_base: u8 = 0;
let r#priv: *mut serial_private = pci_get_drvdata(dev) as *mut serial_private;
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport(dev) };
}
if pci_resource_flags!(dev, 5i32) & 256 == 0 || pci_resource_flags!(dev, 4i32) & 256 == 0 || pci_resource_flags!(dev, 3i32) & 256 == 0 {
return 19i32.wrapping_neg();
}
match (unsafe { (*dev).device }) as i32 {
4356 | 4360 => {
max_port = (((unsafe { (*dev).device }) as i32) & 255) as u32; // 4 ports
// 8 ports
}
4370 => {
max_port = 12; // 12 ports
}
_ => {
return 22i32.wrapping_neg();
}
}
// Get the io address dispatch from the BIOS
bar_data.0[0] = pci_resource_start!(dev, 5i32);
bar_data.0[1] = pci_resource_start!(dev, 4i32);
bar_data.0[2] = pci_resource_start!(dev, 3i32);
for i in 0..max_port {
// UART0 configuration offset start from 0x40
config_base = 64u32.wrapping_add(8u32.wrapping_mul(i)) as u8;
// Calculate Real IO Port
iobase = (bar_data.0[(i / 4) as usize] & 4294967264).wrapping_add((i % 4).wrapping_mul(8) as u64);
// Enable UART I/O port
unsafe { pci_write_config_byte(dev as *const pci_dev, (config_base as i32).wrapping_add(0), 1u8) };
// Select 128-byte FIFO and 8x FIFO threshold
unsafe { pci_write_config_byte(dev as *const pci_dev, (config_base as i32).wrapping_add(1), 51u8) };
// LSB UART
unsafe { pci_write_config_byte(dev as *const pci_dev, (config_base as i32).wrapping_add(4), (iobase & 255) as u8) };
// MSB UART
unsafe { pci_write_config_byte(dev as *const pci_dev, (config_base as i32).wrapping_add(5), ((iobase & 65280) >> 8i32) as u8) };
unsafe { pci_write_config_byte(dev as *const pci_dev, (config_base as i32).wrapping_add(6), (*dev).irq as u8) };
if r#priv.is_null() {
// First init without port data
// force init to RS232 Mode
unsafe { pci_write_config_byte(dev as *const pci_dev, (config_base as i32).wrapping_add(7), 1u8) };
}
}
max_port as i32
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.f815xxa_mem_serial_out")]
extern "C" fn f815xxa_mem_serial_out(p__0: *mut uart_port, offset: u32, value: u32) {
let data: *mut f815xxa_data = (unsafe { (*p__0).private_data }) as *mut f815xxa_data;
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*data).lock) }, flags);
writeb(value as u8, unsafe { (*p__0).membase }.wrapping_offset(offset as isize) as *mut ());
readb(unsafe { (*p__0).membase }.wrapping_offset(7 as isize) as *const ()); // Dummy read for flush pcie tx queue
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*data).lock) }, flags);
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_fintek_f815xxa_setup")]
extern "C" fn pci_fintek_f815xxa_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
let pdev: *mut pci_dev = unsafe { (*r#priv).dev };
let mut data: *mut f815xxa_data = unsafe { core::mem::zeroed() };
data = devm_kzalloc(unsafe { core::ptr::addr_of_mut!((*pdev).dev) }, core::mem::size_of::<f815xxa_data>() as u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *mut f815xxa_data;
if data.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*data).idx = idx;
}
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*data).lock) });
unsafe {
*core::ptr::addr_of_mut!((*data).lock) = *{
__slate_compound_0 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
if true {
break;
}
}
unsafe {
(*port).port.private_data = data as *mut ();
}
unsafe {
(*port).port.iotype = 2;
}
let __v3: u64 = unsafe { (*port).port.flags };
let __v4: u64 = __v3 | BIT_ULL!(31i32);
unsafe {
(*port).port.flags = __v4;
}
unsafe {
(*port).port.mapbase = pci_resource_start!(pdev, 0i32).wrapping_add(8i32.wrapping_mul(idx) as u64);
}
unsafe {
(*port).port.serial_out = Some(f815xxa_mem_serial_out);
}
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_fintek_f815xxa_init")]
extern "C" fn pci_fintek_f815xxa_init(dev: *mut pci_dev) -> i32 {
let mut max_port: u32 = 0;
let mut config_base: i32 = 0;
if pci_resource_flags!(dev, 0i32) & 512 == 0 {
return 19i32.wrapping_neg();
}
match (unsafe { (*dev).device }) as i32 {
4612 | 4616 => {
max_port = (((unsafe { (*dev).device }) as i32) & 255) as u32; // 4 ports
// 8 ports
}
4626 => {
max_port = 12; // 12 ports
}
_ => {
return 22i32.wrapping_neg();
}
}
// Set to mmio decode
unsafe { pci_write_config_byte(dev as *const pci_dev, 521i32, 64u8) };
for i in 0..max_port {
// UART0 configuration offset start from 0x2A0
config_base = 672u32.wrapping_add(8u32.wrapping_mul(i)) as i32;
// Select 128-byte FIFO and 8x FIFO threshold
unsafe { pci_write_config_byte(dev as *const pci_dev, config_base.wrapping_add(1), 51u8) };
// Enable UART I/O port
unsafe { pci_write_config_byte(dev as *const pci_dev, config_base.wrapping_add(0), 1u8) };
}
max_port as i32
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.skip_tx_en_setup")]
extern "C" fn skip_tx_en_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
let __v1: u8 = unsafe { (*port).port.quirks };
let __v2: u8 = (((__v1 as i32) as u64) | BIT!(0i32)) as u8;
unsafe {
(*port).port.quirks = __v2;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*r#priv).dev).dev) as *const device, c"serial8250: skipping TxEn test for device [%04x:%04x] subsystem [%04x:%04x]\n".as_ptr() as *const u8, (*(*r#priv).dev).vendor as i32, (*(*r#priv).dev).device as i32, (*(*r#priv).dev).subsystem_vendor as i32, (*(*r#priv).dev).subsystem_device as i32) };
}
pci_default_setup(r#priv, board, port, idx)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.kt_handle_break")]
extern "C" fn kt_handle_break(p__0: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(p__0);
// On receipt of a BI, serial device in Intel ME (Intel
// management engine) needs to have its fifos cleared for sane
// SOL (Serial Over Lan) output.
unsafe { serial8250_clear_and_reinit_fifos(up) };
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.kt_serial_in")]
extern "C" fn kt_serial_in(p__0: *mut uart_port, offset: u32) -> u32 {
let up: *mut uart_8250_port = up_to_u8250p(p__0);
let mut val: u32 = 0;
// When the Intel ME (management engine) gets reset its serial
// port registers could return 0 momentarily.  Functions like
// serial8250_console_write, read and save the IER, perform
// some operation and then restore it.  In order to avoid
// setting IER register inadvertently to 0, if the value read
// is 0, double check with ier value in uart_8250_port and use
// that instead.  up->ier should be the same value as what is
// currently configured.
val = __inb(unsafe { (*p__0).iobase }.wrapping_add(offset as u64) as u16) as u32;
if offset == 1 {
if val == 0 {
val = (unsafe { (*up).ier }) as u32;
}
}
val
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.kt_serial_setup")]
extern "C" fn kt_serial_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport((*r#priv).dev) };
}
let __v1: u64 = unsafe { (*port).port.flags };
let __v2: u64 = __v1 | BIT_ULL!(26i32);
unsafe {
(*port).port.flags = __v2;
}
unsafe {
(*port).port.serial_in = Some(kt_serial_in);
}
unsafe {
(*port).port.handle_break = Some(kt_handle_break);
}
skip_tx_en_setup(r#priv, board, port, idx)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_eg20t_init")]
extern "C" fn pci_eg20t_init(dev: *mut pci_dev) -> i32 {
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_wch_ch353_setup")]
extern "C" fn pci_wch_ch353_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport((*r#priv).dev) };
}
let __v1: u64 = unsafe { (*port).port.flags };
let __v2: u64 = __v1 | BIT_ULL!(27i32);
unsafe {
(*port).port.flags = __v2;
}
unsafe {
(*port).port.r#type = 4;
}
pci_default_setup(r#priv, board, port, idx)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_wch_ch355_setup")]
extern "C" fn pci_wch_ch355_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport((*r#priv).dev) };
}
let __v1: u64 = unsafe { (*port).port.flags };
let __v2: u64 = __v1 | BIT_ULL!(27i32);
unsafe {
(*port).port.flags = __v2;
}
unsafe {
(*port).port.r#type = 4;
}
pci_default_setup(r#priv, board, port, idx)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_wch_ch38x_setup")]
extern "C" fn pci_wch_ch38x_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, idx: i32) -> i32 {
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport((*r#priv).dev) };
}
let __v1: u64 = unsafe { (*port).port.flags };
let __v2: u64 = __v1 | BIT_ULL!(27i32);
unsafe {
(*port).port.flags = __v2;
}
unsafe {
(*port).port.r#type = 12;
}
pci_default_setup(r#priv, board, port, idx)
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_wch_ch38x_init")]
extern "C" fn pci_wch_ch38x_init(dev: *mut pci_dev) -> i32 {
let mut max_port: i32 = 0;
let mut iobase: u64 = 0;
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport(dev) };
}
match (unsafe { (*dev).device }) as i32 {
14419 => {
max_port = 8; // 8 ports
}
_ => {
return 22i32.wrapping_neg();
}
}
iobase = pci_resource_start!(dev, 0i32);
__outb(2u8, iobase.wrapping_add(235) as u16);
max_port
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_wch_ch38x_exit")]
extern "C" fn pci_wch_ch38x_exit(dev: *mut pci_dev) {
let mut iobase: u64 = 0;
if IS_ENABLED__2!() == 0 {
unsafe { serial_8250_warn_need_ioport(dev) };
return;
}
iobase = pci_resource_start!(dev, 0i32);
__outb(0u8, iobase.wrapping_add(235) as u16);
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_sunix_setup")]
extern "C" fn pci_sunix_setup(r#priv: *mut serial_private, board: *const pciserial_board, port: *mut uart_8250_port, mut idx: i32) -> i32 {
let mut bar: i32 = 0;
let mut offset: i32 = 0;
let __v1: u64 = unsafe { (*port).port.flags };
let __v2: u64 = __v1 | BIT_ULL!(27i32);
unsafe {
(*port).port.flags = __v2;
}
unsafe {
(*port).port.r#type = 121;
}
if idx < 4 {
bar = 0;
offset = (idx as u32).wrapping_mul(unsafe { (*board).uart_offset }) as i32;
} else {
bar = 1;
idx = idx.wrapping_sub(4);
idx = div_s64_rem(idx as i64, 4i32, unsafe { &mut (*core::ptr::addr_of_mut!(offset)) }) as i32;
offset = (idx.wrapping_mul(64) as u32).wrapping_add((offset as u32).wrapping_mul(unsafe { (*board).uart_offset })) as i32;
}
setup_port(r#priv, port, bar as u8, offset as u32, 0i32)
}

// Interrupt mask register, p0 to p7
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_systembase_init")]
extern "C" fn pci_systembase_init(dev: *mut pci_dev) -> i32 {
let mut iobase: u64 = 0;
if IS_ENABLED__2!() == 0 {
return unsafe { serial_8250_warn_need_ioport(dev) };
}
iobase = pci_resource_start!(dev, 1i32);
// This will support up to 8 ports
__outb(255u8, iobase.wrapping_add(12) as u16);
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pci_systembase_exit")]
extern "C" fn pci_systembase_exit(dev: *mut pci_dev) {
let mut iobase: u64 = 0;
if IS_ENABLED__2!() == 0 {
unsafe { serial_8250_warn_need_ioport(dev) };
return;
}
iobase = pci_resource_start!(dev, 0i32);
__outb(0u8, iobase.wrapping_add(12) as u16);
}

/// Master list of serial port init/setup/exit quirks.
/// This does not describe the general nature of the port.
/// (ie, baud base, number and location of ports, etc)
///
/// This list is ordered alphabetically by vendor then device.
/// Specific entries must come before more generic entries.
/// ADDI-DATA GmbH communication cards <info@addi-data.com>
/// AFAVLAB cards - these may be called via parport_serial
///  It is not clear whether this applies to all products.
/// HP Diva
/// HPE PCI serial device
/// Intel
/// ITE
/// National Instruments
/// Quatech
/// Panacom
/// PLX
/// SBS Technologies, Inc., PMC-OCTALPRO 232
/// SBS Technologies, Inc., PMC-OCTALPRO 422
/// SBS Technologies, Inc., P-Octal 232
/// SBS Technologies, Inc., P-Octal 422
/// SIIG cards - these may be called via parport_serial
/// Systembase
/// Titan cards
/// Timedia cards
/// Sunix PCI serial boards
/// Xircom cards
/// Netmos cards - these may be called via parport_serial
/// EndRun Technologies
/// For Oxford Semiconductor Tornado based devices
/// Brainboxes devices - all Oxsemi based
/// Cronyx Omega PCI (PLX-chip based)
/// WCH CH353 1S1P card (16550 clone)
/// WCH CH353 2S1P card (16550 clone)
/// WCH CH353 4S card (16550 clone)
/// WCH CH353 2S1PF card (16550 clone)
/// WCH CH352 2S card (16550 clone)
/// WCH CH355 4S card (16550 clone)
/// WCH CH382 2S card (16850 clone)
/// WCH CH382 2S1P card (16850 clone)
/// WCH CH384 4S card (16850 clone)
/// WCH CH384 8S card (16850 clone)
/// Broadcom TruManage (NetXtreme)
/// Default "match everything" terminator entry
static mut pci_serial_quirks: __SlateAlign16<[pci_serial_quirk; 96]> = __SlateAlign16([{
let mut __t0: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t0.vendor = 4328u32;
__t0.device = 33166u32;
__t0.subvendor = 4294967295u32;
__t0.subdevice = 4294967295u32;
__t0.setup = Some(addidata_apci7800_setup);
__t0
}, {
let mut __t1: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t1.vendor = 5339u32;
__t1.device = 4294967295u32;
__t1.subvendor = 4294967295u32;
__t1.subdevice = 4294967295u32;
__t1.setup = Some(afavlab_setup);
__t1
}, {
let mut __t2: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t2.vendor = 4156u32;
__t2.device = 4168u32;
__t2.subvendor = 4294967295u32;
__t2.subdevice = 4294967295u32;
__t2.init = Some(pci_hp_diva_init);
__t2.setup = Some(pci_hp_diva_setup);
__t2
}, {
let mut __t3: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t3.vendor = 5520u32;
__t3.device = 894u32;
__t3.subvendor = 4294967295u32;
__t3.subdevice = 4294967295u32;
__t3.setup = Some(pci_hp_diva_setup);
__t3
}, {
let mut __t4: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t4.vendor = 32902u32;
__t4.device = 6496u32;
__t4.subvendor = 58559u32;
__t4.subdevice = 4294967295u32;
__t4.init = Some(pci_inteli960ni_init);
__t4.setup = Some(pci_default_setup);
__t4
}, {
let mut __t5: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t5.vendor = 32902u32;
__t5.device = 4194u32;
__t5.subvendor = 4294967295u32;
__t5.subdevice = 4294967295u32;
__t5.setup = Some(skip_tx_en_setup);
__t5
}, {
let mut __t6: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t6.vendor = 32902u32;
__t6.device = 4239u32;
__t6.subvendor = 4294967295u32;
__t6.subdevice = 4294967295u32;
__t6.setup = Some(skip_tx_en_setup);
__t6
}, {
let mut __t7: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t7.vendor = 32902u32;
__t7.device = 4229u32;
__t7.subvendor = 4294967295u32;
__t7.subdevice = 4294967295u32;
__t7.setup = Some(skip_tx_en_setup);
__t7
}, {
let mut __t8: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t8.vendor = 32902u32;
__t8.device = 11878u32;
__t8.subvendor = 4294967295u32;
__t8.subdevice = 4294967295u32;
__t8.setup = Some(ce4100_serial_setup);
__t8
}, {
let mut __t9: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t9.vendor = 32902u32;
__t9.device = 7485u32;
__t9.subvendor = 4294967295u32;
__t9.subdevice = 4294967295u32;
__t9.setup = Some(kt_serial_setup);
__t9
}, {
let mut __t10: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t10.vendor = 4739u32;
__t10.device = 34930u32;
__t10.subvendor = 4294967295u32;
__t10.subdevice = 4294967295u32;
__t10.init = Some(pci_ite887x_init);
__t10.setup = Some(pci_default_setup);
__t10.exit = Some(pci_ite887x_exit);
__t10
}, {
let mut __t11: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t11.vendor = 4243u32;
__t11.device = 53936u32;
__t11.subvendor = 4294967295u32;
__t11.subdevice = 4294967295u32;
__t11.init = Some(pci_ni8420_init);
__t11.setup = Some(pci_default_setup);
__t11.exit = Some(pci_ni8420_exit);
__t11
}, {
let mut __t12: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t12.vendor = 4243u32;
__t12.device = 53584u32;
__t12.subvendor = 4294967295u32;
__t12.subdevice = 4294967295u32;
__t12.init = Some(pci_ni8420_init);
__t12.setup = Some(pci_default_setup);
__t12.exit = Some(pci_ni8420_exit);
__t12
}, {
let mut __t13: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t13.vendor = 4243u32;
__t13.device = 53568u32;
__t13.subvendor = 4294967295u32;
__t13.subdevice = 4294967295u32;
__t13.init = Some(pci_ni8420_init);
__t13.setup = Some(pci_default_setup);
__t13.exit = Some(pci_ni8420_exit);
__t13
}, {
let mut __t14: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t14.vendor = 4243u32;
__t14.device = 53552u32;
__t14.subvendor = 4294967295u32;
__t14.subdevice = 4294967295u32;
__t14.init = Some(pci_ni8420_init);
__t14.setup = Some(pci_default_setup);
__t14.exit = Some(pci_ni8420_exit);
__t14
}, {
let mut __t15: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t15.vendor = 4243u32;
__t15.device = 53872u32;
__t15.subvendor = 4294967295u32;
__t15.subdevice = 4294967295u32;
__t15.init = Some(pci_ni8420_init);
__t15.setup = Some(pci_default_setup);
__t15.exit = Some(pci_ni8420_exit);
__t15
}, {
let mut __t16: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t16.vendor = 4243u32;
__t16.device = 53840u32;
__t16.subvendor = 4294967295u32;
__t16.subdevice = 4294967295u32;
__t16.init = Some(pci_ni8420_init);
__t16.setup = Some(pci_default_setup);
__t16.exit = Some(pci_ni8420_exit);
__t16
}, {
let mut __t17: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t17.vendor = 4243u32;
__t17.device = 53745u32;
__t17.subvendor = 4294967295u32;
__t17.subdevice = 4294967295u32;
__t17.init = Some(pci_ni8420_init);
__t17.setup = Some(pci_default_setup);
__t17.exit = Some(pci_ni8420_exit);
__t17
}, {
let mut __t18: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t18.vendor = 4243u32;
__t18.device = 53744u32;
__t18.subvendor = 4294967295u32;
__t18.subdevice = 4294967295u32;
__t18.init = Some(pci_ni8420_init);
__t18.setup = Some(pci_default_setup);
__t18.exit = Some(pci_ni8420_exit);
__t18
}, {
let mut __t19: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t19.vendor = 4243u32;
__t19.device = 53728u32;
__t19.subvendor = 4294967295u32;
__t19.subdevice = 4294967295u32;
__t19.init = Some(pci_ni8420_init);
__t19.setup = Some(pci_default_setup);
__t19.exit = Some(pci_ni8420_exit);
__t19
}, {
let mut __t20: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t20.vendor = 4243u32;
__t20.device = 53712u32;
__t20.subvendor = 4294967295u32;
__t20.subdevice = 4294967295u32;
__t20.init = Some(pci_ni8420_init);
__t20.setup = Some(pci_default_setup);
__t20.exit = Some(pci_ni8420_exit);
__t20
}, {
let mut __t21: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t21.vendor = 4243u32;
__t21.device = 53664u32;
__t21.subvendor = 4294967295u32;
__t21.subdevice = 4294967295u32;
__t21.init = Some(pci_ni8420_init);
__t21.setup = Some(pci_default_setup);
__t21.exit = Some(pci_ni8420_exit);
__t21
}, {
let mut __t22: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t22.vendor = 4243u32;
__t22.device = 53648u32;
__t22.subvendor = 4294967295u32;
__t22.subdevice = 4294967295u32;
__t22.init = Some(pci_ni8420_init);
__t22.setup = Some(pci_default_setup);
__t22.exit = Some(pci_ni8420_exit);
__t22
}, {
let mut __t23: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t23.vendor = 4243u32;
__t23.device = 4294967295u32;
__t23.subvendor = 4294967295u32;
__t23.subdevice = 4294967295u32;
__t23.init = Some(pci_ni8430_init);
__t23.setup = Some(pci_ni8430_setup);
__t23.exit = Some(pci_ni8430_exit);
__t23
}, {
let mut __t24: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t24.vendor = 4956u32;
__t24.device = 4294967295u32;
__t24.subvendor = 4294967295u32;
__t24.subdevice = 4294967295u32;
__t24.init = Some(pci_quatech_init);
__t24.setup = Some(pci_quatech_setup);
__t24
}, {
let mut __t25: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t25.vendor = 5332u32;
__t25.device = 1024u32;
__t25.subvendor = 4294967295u32;
__t25.subdevice = 4294967295u32;
__t25.init = Some(pci_plx9050_init);
__t25.setup = Some(pci_default_setup);
__t25.exit = Some(pci_plx9050_exit);
__t25
}, {
let mut __t26: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t26.vendor = 5332u32;
__t26.device = 1026u32;
__t26.subvendor = 4294967295u32;
__t26.subdevice = 4294967295u32;
__t26.init = Some(pci_plx9050_init);
__t26.setup = Some(pci_default_setup);
__t26.exit = Some(pci_plx9050_exit);
__t26
}, {
let mut __t27: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t27.vendor = 4277u32;
__t27.device = 36944u32;
__t27.subvendor = 55373u32;
__t27.subdevice = 16469u32;
__t27.init = Some(pci_plx9050_init);
__t27.setup = Some(pci_default_setup);
__t27.exit = Some(pci_plx9050_exit);
__t27
}, {
let mut __t28: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t28.vendor = 4277u32;
__t28.device = 36944u32;
__t28.subvendor = 4521u32;
__t28.subdevice = 21300u32;
__t28.init = Some(pci_plx9050_init);
__t28.setup = Some(pci_default_setup);
__t28.exit = Some(pci_plx9050_exit);
__t28
}, {
let mut __t29: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t29.vendor = 4277u32;
__t29.device = 4202u32;
__t29.subvendor = 4277u32;
__t29.subdevice = 4202u32;
__t29.init = Some(pci_plx9050_init);
__t29.setup = Some(pci_default_setup);
__t29.exit = Some(pci_plx9050_exit);
__t29
}, {
let mut __t30: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t30.vendor = 4683u32;
__t30.device = 1u32;
__t30.subvendor = 4683u32;
__t30.subdevice = 264u32;
__t30.init = Some(sbs_init);
__t30.setup = Some(sbs_setup);
__t30.exit = Some(sbs_exit);
__t30
}, {
let mut __t31: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t31.vendor = 4683u32;
__t31.device = 1u32;
__t31.subvendor = 4683u32;
__t31.subdevice = 520u32;
__t31.init = Some(sbs_init);
__t31.setup = Some(sbs_setup);
__t31.exit = Some(sbs_exit);
__t31
}, {
let mut __t32: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t32.vendor = 4683u32;
__t32.device = 1u32;
__t32.subvendor = 4683u32;
__t32.subdevice = 776u32;
__t32.init = Some(sbs_init);
__t32.setup = Some(sbs_setup);
__t32.exit = Some(sbs_exit);
__t32
}, {
let mut __t33: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t33.vendor = 4683u32;
__t33.device = 1u32;
__t33.subvendor = 4683u32;
__t33.subdevice = 1032u32;
__t33.init = Some(sbs_init);
__t33.setup = Some(sbs_setup);
__t33.exit = Some(sbs_exit);
__t33
}, {
let mut __t34: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t34.vendor = 4895u32;
__t34.device = 4294967295u32;
__t34.subvendor = 4294967295u32;
__t34.subdevice = 4294967295u32;
__t34.init = Some(pci_siig_init);
__t34.setup = Some(pci_siig_setup);
__t34
}, {
let mut __t35: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t35.vendor = 5281u32;
__t35.device = 8u32;
__t35.subvendor = 4294967295u32;
__t35.subdevice = 4294967295u32;
__t35.init = Some(pci_systembase_init);
__t35.setup = Some(pci_default_setup);
__t35.exit = Some(pci_systembase_exit);
__t35
}, {
let mut __t36: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t36.vendor = 5330u32;
__t36.device = 32832u32;
__t36.subvendor = 4294967295u32;
__t36.subdevice = 4294967295u32;
__t36.setup = Some(titan_400l_800l_setup);
__t36
}, {
let mut __t37: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t37.vendor = 5330u32;
__t37.device = 32896u32;
__t37.subvendor = 4294967295u32;
__t37.subdevice = 4294967295u32;
__t37.setup = Some(titan_400l_800l_setup);
__t37
}, {
let mut __t38: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t38.vendor = 5129u32;
__t38.device = 29032u32;
__t38.subvendor = 5129u32;
__t38.subdevice = 4294967295u32;
__t38.probe = Some(pci_timedia_probe);
__t38.init = Some(pci_timedia_init);
__t38.setup = Some(pci_timedia_setup);
__t38
}, {
let mut __t39: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t39.vendor = 5129u32;
__t39.device = 4294967295u32;
__t39.subvendor = 4294967295u32;
__t39.subdevice = 4294967295u32;
__t39.setup = Some(pci_timedia_setup);
__t39
}, {
let mut __t40: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t40.vendor = 8148u32;
__t40.device = 6553u32;
__t40.subvendor = 8148u32;
__t40.subdevice = 4294967295u32;
__t40.setup = Some(pci_sunix_setup);
__t40
}, {
let mut __t41: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t41.vendor = 4445u32;
__t41.device = 259u32;
__t41.subvendor = 4294967295u32;
__t41.subdevice = 4294967295u32;
__t41.init = Some(pci_xircom_init);
__t41.setup = Some(pci_default_setup);
__t41
}, {
let mut __t42: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t42.vendor = 38672u32;
__t42.device = 4294967295u32;
__t42.subvendor = 4294967295u32;
__t42.subdevice = 4294967295u32;
__t42.init = Some(pci_netmos_init);
__t42.setup = Some(pci_netmos_9900_setup);
__t42
}, {
let mut __t43: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t43.vendor = 4699u32;
__t43.device = 4294967295u32;
__t43.subvendor = 4294967295u32;
__t43.subdevice = 4294967295u32;
__t43.init = Some(pci_netmos_init);
__t43.setup = Some(pci_netmos_9900_setup);
__t43
}, {
let mut __t44: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t44.vendor = 29697u32;
__t44.device = 4294967295u32;
__t44.subvendor = 4294967295u32;
__t44.subdevice = 4294967295u32;
__t44.init = Some(pci_oxsemi_tornado_init);
__t44.setup = Some(pci_default_setup);
__t44
}, {
let mut __t45: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t45.vendor = 5141u32;
__t45.device = 4294967295u32;
__t45.subvendor = 4294967295u32;
__t45.subdevice = 4294967295u32;
__t45.init = Some(pci_oxsemi_tornado_init);
__t45.setup = Some(pci_oxsemi_tornado_setup);
__t45
}, {
let mut __t46: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t46.vendor = 5410u32;
__t46.device = 4294967295u32;
__t46.subvendor = 4294967295u32;
__t46.subdevice = 4294967295u32;
__t46.init = Some(pci_oxsemi_tornado_init);
__t46.setup = Some(pci_oxsemi_tornado_setup);
__t46
}, {
let mut __t47: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t47.vendor = 4431u32;
__t47.device = 246u32;
__t47.subvendor = 4116u32;
__t47.subdevice = 4294967295u32;
__t47.init = Some(pci_oxsemi_tornado_init);
__t47.setup = Some(pci_oxsemi_tornado_setup);
__t47
}, {
let mut __t48: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t48.vendor = 4954u32;
__t48.device = 16423u32;
__t48.subvendor = 4294967295u32;
__t48.subdevice = 4294967295u32;
__t48.init = Some(pci_oxsemi_tornado_init);
__t48.setup = Some(pci_oxsemi_tornado_setup);
__t48
}, {
let mut __t49: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t49.vendor = 4954u32;
__t49.device = 16424u32;
__t49.subvendor = 4294967295u32;
__t49.subdevice = 4294967295u32;
__t49.init = Some(pci_oxsemi_tornado_init);
__t49.setup = Some(pci_oxsemi_tornado_setup);
__t49
}, {
let mut __t50: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t50.vendor = 4954u32;
__t50.device = 16425u32;
__t50.subvendor = 4294967295u32;
__t50.subdevice = 4294967295u32;
__t50.init = Some(pci_oxsemi_tornado_init);
__t50.setup = Some(pci_oxsemi_tornado_setup);
__t50
}, {
let mut __t51: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t51.vendor = 4954u32;
__t51.device = 16409u32;
__t51.subvendor = 4294967295u32;
__t51.subdevice = 4294967295u32;
__t51.init = Some(pci_oxsemi_tornado_init);
__t51.setup = Some(pci_oxsemi_tornado_setup);
__t51
}, {
let mut __t52: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t52.vendor = 4954u32;
__t52.device = 16406u32;
__t52.subvendor = 4294967295u32;
__t52.subdevice = 4294967295u32;
__t52.init = Some(pci_oxsemi_tornado_init);
__t52.setup = Some(pci_oxsemi_tornado_setup);
__t52
}, {
let mut __t53: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t53.vendor = 4954u32;
__t53.device = 16405u32;
__t53.subvendor = 4294967295u32;
__t53.subdevice = 4294967295u32;
__t53.init = Some(pci_oxsemi_tornado_init);
__t53.setup = Some(pci_oxsemi_tornado_setup);
__t53
}, {
let mut __t54: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t54.vendor = 4954u32;
__t54.device = 16394u32;
__t54.subvendor = 4294967295u32;
__t54.subdevice = 4294967295u32;
__t54.init = Some(pci_oxsemi_tornado_init);
__t54.setup = Some(pci_oxsemi_tornado_setup);
__t54
}, {
let mut __t55: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t55.vendor = 4954u32;
__t55.device = 16398u32;
__t55.subvendor = 4294967295u32;
__t55.subdevice = 4294967295u32;
__t55.init = Some(pci_oxsemi_tornado_init);
__t55.setup = Some(pci_oxsemi_tornado_setup);
__t55
}, {
let mut __t56: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t56.vendor = 4954u32;
__t56.device = 16396u32;
__t56.subvendor = 4294967295u32;
__t56.subdevice = 4294967295u32;
__t56.init = Some(pci_oxsemi_tornado_init);
__t56.setup = Some(pci_oxsemi_tornado_setup);
__t56
}, {
let mut __t57: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t57.vendor = 4954u32;
__t57.device = 16395u32;
__t57.subvendor = 4294967295u32;
__t57.subdevice = 4294967295u32;
__t57.init = Some(pci_oxsemi_tornado_init);
__t57.setup = Some(pci_oxsemi_tornado_setup);
__t57
}, {
let mut __t58: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t58.vendor = 4954u32;
__t58.device = 16399u32;
__t58.subvendor = 4294967295u32;
__t58.subdevice = 4294967295u32;
__t58.init = Some(pci_oxsemi_tornado_init);
__t58.setup = Some(pci_oxsemi_tornado_setup);
__t58
}, {
let mut __t59: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t59.vendor = 4954u32;
__t59.device = 16400u32;
__t59.subvendor = 4294967295u32;
__t59.subdevice = 4294967295u32;
__t59.init = Some(pci_oxsemi_tornado_init);
__t59.setup = Some(pci_oxsemi_tornado_setup);
__t59
}, {
let mut __t60: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t60.vendor = 4954u32;
__t60.device = 16401u32;
__t60.subvendor = 4294967295u32;
__t60.subdevice = 4294967295u32;
__t60.init = Some(pci_oxsemi_tornado_init);
__t60.setup = Some(pci_oxsemi_tornado_setup);
__t60
}, {
let mut __t61: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t61.vendor = 4954u32;
__t61.device = 16413u32;
__t61.subvendor = 4294967295u32;
__t61.subdevice = 4294967295u32;
__t61.init = Some(pci_oxsemi_tornado_init);
__t61.setup = Some(pci_oxsemi_tornado_setup);
__t61
}, {
let mut __t62: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t62.vendor = 4954u32;
__t62.device = 16414u32;
__t62.subvendor = 4294967295u32;
__t62.subdevice = 4294967295u32;
__t62.init = Some(pci_oxsemi_tornado_init);
__t62.setup = Some(pci_oxsemi_tornado_setup);
__t62
}, {
let mut __t63: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t63.vendor = 4954u32;
__t63.device = 16403u32;
__t63.subvendor = 4294967295u32;
__t63.subdevice = 4294967295u32;
__t63.init = Some(pci_oxsemi_tornado_init);
__t63.setup = Some(pci_oxsemi_tornado_setup);
__t63
}, {
let mut __t64: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t64.vendor = 4954u32;
__t64.device = 16407u32;
__t64.subvendor = 4294967295u32;
__t64.subdevice = 4294967295u32;
__t64.init = Some(pci_oxsemi_tornado_init);
__t64.setup = Some(pci_oxsemi_tornado_setup);
__t64
}, {
let mut __t65: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t65.vendor = 4954u32;
__t65.device = 16408u32;
__t65.subvendor = 4294967295u32;
__t65.subdevice = 4294967295u32;
__t65.init = Some(pci_oxsemi_tornado_init);
__t65.setup = Some(pci_oxsemi_tornado_setup);
__t65
}, {
let mut __t66: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t66.vendor = 4954u32;
__t66.device = 16422u32;
__t66.subvendor = 4294967295u32;
__t66.subdevice = 4294967295u32;
__t66.init = Some(pci_oxsemi_tornado_init);
__t66.setup = Some(pci_oxsemi_tornado_setup);
__t66
}, {
let mut __t67: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t67.vendor = 4954u32;
__t67.device = 16417u32;
__t67.subvendor = 4294967295u32;
__t67.subdevice = 4294967295u32;
__t67.init = Some(pci_oxsemi_tornado_init);
__t67.setup = Some(pci_oxsemi_tornado_setup);
__t67
}, {
let mut __t68: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t68.vendor = 32902u32;
__t68.device = 34833u32;
__t68.subvendor = 4294967295u32;
__t68.subdevice = 4294967295u32;
__t68.init = Some(pci_eg20t_init);
__t68.setup = Some(pci_default_setup);
__t68
}, {
let mut __t69: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t69.vendor = 32902u32;
__t69.device = 34834u32;
__t69.subvendor = 4294967295u32;
__t69.subdevice = 4294967295u32;
__t69.init = Some(pci_eg20t_init);
__t69.setup = Some(pci_default_setup);
__t69
}, {
let mut __t70: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t70.vendor = 32902u32;
__t70.device = 34835u32;
__t70.subvendor = 4294967295u32;
__t70.subdevice = 4294967295u32;
__t70.init = Some(pci_eg20t_init);
__t70.setup = Some(pci_default_setup);
__t70
}, {
let mut __t71: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t71.vendor = 32902u32;
__t71.device = 34836u32;
__t71.subvendor = 4294967295u32;
__t71.subdevice = 4294967295u32;
__t71.init = Some(pci_eg20t_init);
__t71.setup = Some(pci_default_setup);
__t71
}, {
let mut __t72: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t72.vendor = 4315u32;
__t72.device = 32807u32;
__t72.subvendor = 4294967295u32;
__t72.subdevice = 4294967295u32;
__t72.init = Some(pci_eg20t_init);
__t72.setup = Some(pci_default_setup);
__t72
}, {
let mut __t73: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t73.vendor = 4315u32;
__t73.device = 32808u32;
__t73.subvendor = 4294967295u32;
__t73.subdevice = 4294967295u32;
__t73.init = Some(pci_eg20t_init);
__t73.setup = Some(pci_default_setup);
__t73
}, {
let mut __t74: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t74.vendor = 4315u32;
__t74.device = 32809u32;
__t74.subvendor = 4294967295u32;
__t74.subdevice = 4294967295u32;
__t74.init = Some(pci_eg20t_init);
__t74.setup = Some(pci_default_setup);
__t74
}, {
let mut __t75: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t75.vendor = 4315u32;
__t75.device = 32780u32;
__t75.subvendor = 4294967295u32;
__t75.subdevice = 4294967295u32;
__t75.init = Some(pci_eg20t_init);
__t75.setup = Some(pci_default_setup);
__t75
}, {
let mut __t76: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t76.vendor = 4315u32;
__t76.device = 32781u32;
__t76.subvendor = 4294967295u32;
__t76.subdevice = 4294967295u32;
__t76.init = Some(pci_eg20t_init);
__t76.setup = Some(pci_default_setup);
__t76
}, {
let mut __t77: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t77.vendor = 4277u32;
__t77.device = 49153u32;
__t77.subvendor = 4294967295u32;
__t77.subdevice = 4294967295u32;
__t77.setup = Some(pci_omegapci_setup);
__t77
}, {
let mut __t78: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t78.vendor = 17224u32;
__t78.device = 20563u32;
__t78.subvendor = 4294967295u32;
__t78.subdevice = 4294967295u32;
__t78.setup = Some(pci_wch_ch353_setup);
__t78
}, {
let mut __t79: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t79.vendor = 17224u32;
__t79.device = 28755u32;
__t79.subvendor = 4294967295u32;
__t79.subdevice = 4294967295u32;
__t79.setup = Some(pci_wch_ch353_setup);
__t79
}, {
let mut __t80: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t80.vendor = 17224u32;
__t80.device = 13395u32;
__t80.subvendor = 4294967295u32;
__t80.subdevice = 4294967295u32;
__t80.setup = Some(pci_wch_ch353_setup);
__t80
}, {
let mut __t81: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t81.vendor = 17224u32;
__t81.device = 20550u32;
__t81.subvendor = 4294967295u32;
__t81.subdevice = 4294967295u32;
__t81.setup = Some(pci_wch_ch353_setup);
__t81
}, {
let mut __t82: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t82.vendor = 17224u32;
__t82.device = 12883u32;
__t82.subvendor = 4294967295u32;
__t82.subdevice = 4294967295u32;
__t82.setup = Some(pci_wch_ch353_setup);
__t82
}, {
let mut __t83: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t83.vendor = 17224u32;
__t83.device = 29043u32;
__t83.subvendor = 4294967295u32;
__t83.subdevice = 4294967295u32;
__t83.setup = Some(pci_wch_ch355_setup);
__t83
}, {
let mut __t84: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t84.vendor = 7168u32;
__t84.device = 12883u32;
__t84.subvendor = 4294967295u32;
__t84.subdevice = 4294967295u32;
__t84.setup = Some(pci_wch_ch38x_setup);
__t84
}, {
let mut __t85: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t85.vendor = 7168u32;
__t85.device = 12880u32;
__t85.subvendor = 4294967295u32;
__t85.subdevice = 4294967295u32;
__t85.setup = Some(pci_wch_ch38x_setup);
__t85
}, {
let mut __t86: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t86.vendor = 7168u32;
__t86.device = 13424u32;
__t86.subvendor = 4294967295u32;
__t86.subdevice = 4294967295u32;
__t86.setup = Some(pci_wch_ch38x_setup);
__t86
}, {
let mut __t87: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t87.vendor = 7168u32;
__t87.device = 14419u32;
__t87.subvendor = 4294967295u32;
__t87.subdevice = 4294967295u32;
__t87.init = Some(pci_wch_ch38x_init);
__t87.setup = Some(pci_wch_ch38x_setup);
__t87.exit = Some(pci_wch_ch38x_exit);
__t87
}, {
let mut __t88: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t88.vendor = 5348u32;
__t88.device = 5642u32;
__t88.subvendor = 4294967295u32;
__t88.subdevice = 4294967295u32;
__t88.setup = Some(pci_brcm_trumanage_setup);
__t88
}, {
let mut __t89: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t89.vendor = 7209u32;
__t89.device = 4356u32;
__t89.subvendor = 4294967295u32;
__t89.subdevice = 4294967295u32;
__t89.init = Some(pci_fintek_init);
__t89.setup = Some(pci_fintek_setup);
__t89
}, {
let mut __t90: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t90.vendor = 7209u32;
__t90.device = 4360u32;
__t90.subvendor = 4294967295u32;
__t90.subdevice = 4294967295u32;
__t90.init = Some(pci_fintek_init);
__t90.setup = Some(pci_fintek_setup);
__t90
}, {
let mut __t91: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t91.vendor = 7209u32;
__t91.device = 4370u32;
__t91.subvendor = 4294967295u32;
__t91.subdevice = 4294967295u32;
__t91.init = Some(pci_fintek_init);
__t91.setup = Some(pci_fintek_setup);
__t91
}, {
let mut __t92: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t92.vendor = 7209u32;
__t92.device = 4612u32;
__t92.subvendor = 4294967295u32;
__t92.subdevice = 4294967295u32;
__t92.init = Some(pci_fintek_f815xxa_init);
__t92.setup = Some(pci_fintek_f815xxa_setup);
__t92
}, {
let mut __t93: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t93.vendor = 7209u32;
__t93.device = 4616u32;
__t93.subvendor = 4294967295u32;
__t93.subdevice = 4294967295u32;
__t93.init = Some(pci_fintek_f815xxa_init);
__t93.setup = Some(pci_fintek_f815xxa_setup);
__t93
}, {
let mut __t94: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t94.vendor = 7209u32;
__t94.device = 4626u32;
__t94.subvendor = 4294967295u32;
__t94.subdevice = 4294967295u32;
__t94.init = Some(pci_fintek_f815xxa_init);
__t94.setup = Some(pci_fintek_f815xxa_setup);
__t94
}, {
let mut __t95: pci_serial_quirk = unsafe { core::mem::zeroed() };
__t95.vendor = 4294967295u32;
__t95.device = 4294967295u32;
__t95.subvendor = 4294967295u32;
__t95.subdevice = 4294967295u32;
__t95.setup = Some(pci_default_setup);
__t95
}]);

fn quirk_id_matches(quirk_id: u32, dev_id: u32) -> i32 {
(quirk_id == 4294967295u32 || quirk_id == dev_id) as i32
}

fn find_quirk(dev: *mut pci_dev) -> *mut pci_serial_quirk {
let mut quirk: *mut pci_serial_quirk = unsafe { core::mem::zeroed() };
quirk = unsafe { core::ptr::addr_of_mut!(pci_serial_quirks.0) as *mut pci_serial_quirk };
loop {
let __v0: bool = quirk_id_matches(unsafe { (*quirk).vendor }, (unsafe { (*dev).vendor }) as u32) != 0 && quirk_id_matches(unsafe { (*quirk).device }, (unsafe { (*dev).device }) as u32) != 0;
let __v1: bool = __v0 && quirk_id_matches(unsafe { (*quirk).subvendor }, (unsafe { (*dev).subsystem_vendor }) as u32) != 0;
let __v2: bool = __v1 && quirk_id_matches(unsafe { (*quirk).subdevice }, (unsafe { (*dev).subsystem_device }) as u32) != 0;
if __v2 {
break;
}
let __v4: *mut pci_serial_quirk = quirk;
quirk = __v4.wrapping_offset(1 as isize);
}
quirk
}

// uart_offset - the space between channels
// reg_shift   - describes how the UART registers are mapped
//               to PCI memory by the card.
// For example IER register on SBS, Inc. PMC-OctPro is located at
// offset 0x10 from the UART base, while UART_IER is defined as 1
// in include/linux/serial_reg.h,
// see first lines of serial_in() and serial_out() in 8250.c
// This is the configuration table for all of the PCI serial boards
// which we support.  It is directly indexed by the pci_board_num_t enum
// value, which is encoded in the pci_device_id PCI probe table's
// driver_data member.
//
// The makeup of these names are:
//  pbn_bn{_bt}_n_baud{_offsetinhex}
//
//  bn		= PCI BAR number
//  bt		= Index using PCI BARs
//  n		= number of serial ports
//  baud	= baud rate
//  offsetinhex	= offset for each sequential port (in hex)
//
// This table is sorted by (in order): bn, bt, baud, offsetindex, n.
//
// Please note: in theory if n = 1, _bt infix should make no difference.
// ie, pbn_b0_1_115200 is the same as pbn_b0_bt_1_115200
/// Entries following this are board-specific.
/// Panacom - IOMEM
/// I think this entry is broken - the first_offset looks wrong --rmk
/// This board uses the size of PCI Base region 0 to
/// signal now many ports are available
/// EKF addition for i960 Boards form EKF with serial port.
/// Max 256 ports.
/// Computone - uses IOMEM.
/// PA Semi PWRficient PA6T-1682M on-chip UART
/// National Instruments 843x
/// ADDI-DATA GmbH PCI-Express communication cards <info@addi-data.com>
static mut pci_boards: __SlateAlign16<[pciserial_board; 114]> = __SlateAlign16([{
let mut __t0: pciserial_board = unsafe { core::mem::zeroed() };
__t0.flags = 0u32;
__t0.num_ports = 1u32;
__t0.base_baud = 115200u32;
__t0.uart_offset = 8u32;
__t0
}, {
let mut __t1: pciserial_board = unsafe { core::mem::zeroed() };
__t1.flags = 0u32;
__t1.num_ports = 1u32;
__t1.base_baud = 115200u32;
__t1.uart_offset = 8u32;
__t1
}, {
let mut __t2: pciserial_board = unsafe { core::mem::zeroed() };
__t2.flags = 0u32;
__t2.num_ports = 2u32;
__t2.base_baud = 115200u32;
__t2.uart_offset = 8u32;
__t2
}, {
let mut __t3: pciserial_board = unsafe { core::mem::zeroed() };
__t3.flags = 0u32;
__t3.num_ports = 4u32;
__t3.base_baud = 115200u32;
__t3.uart_offset = 8u32;
__t3
}, {
let mut __t4: pciserial_board = unsafe { core::mem::zeroed() };
__t4.flags = 0u32;
__t4.num_ports = 5u32;
__t4.base_baud = 115200u32;
__t4.uart_offset = 8u32;
__t4
}, {
let mut __t5: pciserial_board = unsafe { core::mem::zeroed() };
__t5.flags = 0u32;
__t5.num_ports = 8u32;
__t5.base_baud = 115200u32;
__t5.uart_offset = 8u32;
__t5
}, {
let mut __t6: pciserial_board = unsafe { core::mem::zeroed() };
__t6.flags = 0u32;
__t6.num_ports = 1u32;
__t6.base_baud = 921600u32;
__t6.uart_offset = 8u32;
__t6
}, {
let mut __t7: pciserial_board = unsafe { core::mem::zeroed() };
__t7.flags = 0u32;
__t7.num_ports = 2u32;
__t7.base_baud = 921600u32;
__t7.uart_offset = 8u32;
__t7
}, {
let mut __t8: pciserial_board = unsafe { core::mem::zeroed() };
__t8.flags = 0u32;
__t8.num_ports = 4u32;
__t8.base_baud = 921600u32;
__t8.uart_offset = 8u32;
__t8
}, {
let mut __t9: pciserial_board = unsafe { core::mem::zeroed() };
__t9.flags = 0u32;
__t9.num_ports = 8u32;
__t9.base_baud = 921600u32;
__t9.uart_offset = 8u32;
__t9
}, {
let mut __t10: pciserial_board = unsafe { core::mem::zeroed() };
__t10.flags = 0u32;
__t10.num_ports = 2u32;
__t10.base_baud = 1130000u32;
__t10.uart_offset = 8u32;
__t10
}, {
let mut __t11: pciserial_board = unsafe { core::mem::zeroed() };
__t11.flags = 0u32;
__t11.num_ports = 4u32;
__t11.base_baud = 1152000u32;
__t11.uart_offset = 8u32;
__t11
}, {
let mut __t12: pciserial_board = unsafe { core::mem::zeroed() };
__t12.flags = 0u32;
__t12.num_ports = 4u32;
__t12.base_baud = 1250000u32;
__t12.uart_offset = 8u32;
__t12
}, {
let mut __t13: pciserial_board = unsafe { core::mem::zeroed() };
__t13.flags = 0u32;
__t13.num_ports = 2u32;
__t13.base_baud = 1843200u32;
__t13.uart_offset = 8u32;
__t13
}, {
let mut __t14: pciserial_board = unsafe { core::mem::zeroed() };
__t14.flags = 0u32;
__t14.num_ports = 4u32;
__t14.base_baud = 1843200u32;
__t14.uart_offset = 8u32;
__t14
}, {
let mut __t15: pciserial_board = unsafe { core::mem::zeroed() };
__t15.flags = 0u32;
__t15.num_ports = 1u32;
__t15.base_baud = 15625000u32;
__t15.uart_offset = 8u32;
__t15
}, {
let mut __t16: pciserial_board = unsafe { core::mem::zeroed() };
__t16.flags = 8u32;
__t16.num_ports = 1u32;
__t16.base_baud = 115200u32;
__t16.uart_offset = 8u32;
__t16
}, {
let mut __t17: pciserial_board = unsafe { core::mem::zeroed() };
__t17.flags = 8u32;
__t17.num_ports = 2u32;
__t17.base_baud = 115200u32;
__t17.uart_offset = 8u32;
__t17
}, {
let mut __t18: pciserial_board = unsafe { core::mem::zeroed() };
__t18.flags = 8u32;
__t18.num_ports = 4u32;
__t18.base_baud = 115200u32;
__t18.uart_offset = 8u32;
__t18
}, {
let mut __t19: pciserial_board = unsafe { core::mem::zeroed() };
__t19.flags = 8u32;
__t19.num_ports = 8u32;
__t19.base_baud = 115200u32;
__t19.uart_offset = 8u32;
__t19
}, {
let mut __t20: pciserial_board = unsafe { core::mem::zeroed() };
__t20.flags = 8u32;
__t20.num_ports = 1u32;
__t20.base_baud = 460800u32;
__t20.uart_offset = 8u32;
__t20
}, {
let mut __t21: pciserial_board = unsafe { core::mem::zeroed() };
__t21.flags = 8u32;
__t21.num_ports = 2u32;
__t21.base_baud = 460800u32;
__t21.uart_offset = 8u32;
__t21
}, {
let mut __t22: pciserial_board = unsafe { core::mem::zeroed() };
__t22.flags = 8u32;
__t22.num_ports = 4u32;
__t22.base_baud = 460800u32;
__t22.uart_offset = 8u32;
__t22
}, {
let mut __t23: pciserial_board = unsafe { core::mem::zeroed() };
__t23.flags = 8u32;
__t23.num_ports = 1u32;
__t23.base_baud = 921600u32;
__t23.uart_offset = 8u32;
__t23
}, {
let mut __t24: pciserial_board = unsafe { core::mem::zeroed() };
__t24.flags = 8u32;
__t24.num_ports = 2u32;
__t24.base_baud = 921600u32;
__t24.uart_offset = 8u32;
__t24
}, {
let mut __t25: pciserial_board = unsafe { core::mem::zeroed() };
__t25.flags = 8u32;
__t25.num_ports = 4u32;
__t25.base_baud = 921600u32;
__t25.uart_offset = 8u32;
__t25
}, {
let mut __t26: pciserial_board = unsafe { core::mem::zeroed() };
__t26.flags = 8u32;
__t26.num_ports = 8u32;
__t26.base_baud = 921600u32;
__t26.uart_offset = 8u32;
__t26
}, {
let mut __t27: pciserial_board = unsafe { core::mem::zeroed() };
__t27.flags = 1u32;
__t27.num_ports = 1u32;
__t27.base_baud = 115200u32;
__t27.uart_offset = 8u32;
__t27
}, {
let mut __t28: pciserial_board = unsafe { core::mem::zeroed() };
__t28.flags = 1u32;
__t28.num_ports = 2u32;
__t28.base_baud = 115200u32;
__t28.uart_offset = 8u32;
__t28
}, {
let mut __t29: pciserial_board = unsafe { core::mem::zeroed() };
__t29.flags = 1u32;
__t29.num_ports = 4u32;
__t29.base_baud = 115200u32;
__t29.uart_offset = 8u32;
__t29
}, {
let mut __t30: pciserial_board = unsafe { core::mem::zeroed() };
__t30.flags = 1u32;
__t30.num_ports = 8u32;
__t30.base_baud = 115200u32;
__t30.uart_offset = 8u32;
__t30
}, {
let mut __t31: pciserial_board = unsafe { core::mem::zeroed() };
__t31.flags = 1u32;
__t31.num_ports = 16u32;
__t31.base_baud = 115200u32;
__t31.uart_offset = 8u32;
__t31
}, {
let mut __t32: pciserial_board = unsafe { core::mem::zeroed() };
__t32.flags = 1u32;
__t32.num_ports = 1u32;
__t32.base_baud = 921600u32;
__t32.uart_offset = 8u32;
__t32
}, {
let mut __t33: pciserial_board = unsafe { core::mem::zeroed() };
__t33.flags = 1u32;
__t33.num_ports = 2u32;
__t33.base_baud = 921600u32;
__t33.uart_offset = 8u32;
__t33
}, {
let mut __t34: pciserial_board = unsafe { core::mem::zeroed() };
__t34.flags = 1u32;
__t34.num_ports = 4u32;
__t34.base_baud = 921600u32;
__t34.uart_offset = 8u32;
__t34
}, {
let mut __t35: pciserial_board = unsafe { core::mem::zeroed() };
__t35.flags = 1u32;
__t35.num_ports = 8u32;
__t35.base_baud = 921600u32;
__t35.uart_offset = 8u32;
__t35
}, {
let mut __t36: pciserial_board = unsafe { core::mem::zeroed() };
__t36.flags = 1u32;
__t36.num_ports = 2u32;
__t36.base_baud = 1250000u32;
__t36.uart_offset = 8u32;
__t36
}, {
let mut __t37: pciserial_board = unsafe { core::mem::zeroed() };
__t37.flags = 9u32;
__t37.num_ports = 1u32;
__t37.base_baud = 115200u32;
__t37.uart_offset = 8u32;
__t37
}, {
let mut __t38: pciserial_board = unsafe { core::mem::zeroed() };
__t38.flags = 9u32;
__t38.num_ports = 2u32;
__t38.base_baud = 115200u32;
__t38.uart_offset = 8u32;
__t38
}, {
let mut __t39: pciserial_board = unsafe { core::mem::zeroed() };
__t39.flags = 9u32;
__t39.num_ports = 4u32;
__t39.base_baud = 115200u32;
__t39.uart_offset = 8u32;
__t39
}, {
let mut __t40: pciserial_board = unsafe { core::mem::zeroed() };
__t40.flags = 9u32;
__t40.num_ports = 2u32;
__t40.base_baud = 921600u32;
__t40.uart_offset = 8u32;
__t40
}, {
let mut __t41: pciserial_board = unsafe { core::mem::zeroed() };
__t41.flags = 1u32;
__t41.num_ports = 1u32;
__t41.base_baud = 1382400u32;
__t41.uart_offset = 8u32;
__t41
}, {
let mut __t42: pciserial_board = unsafe { core::mem::zeroed() };
__t42.flags = 1u32;
__t42.num_ports = 2u32;
__t42.base_baud = 1382400u32;
__t42.uart_offset = 8u32;
__t42
}, {
let mut __t43: pciserial_board = unsafe { core::mem::zeroed() };
__t43.flags = 1u32;
__t43.num_ports = 4u32;
__t43.base_baud = 1382400u32;
__t43.uart_offset = 8u32;
__t43
}, {
let mut __t44: pciserial_board = unsafe { core::mem::zeroed() };
__t44.flags = 1u32;
__t44.num_ports = 8u32;
__t44.base_baud = 1382400u32;
__t44.uart_offset = 8u32;
__t44
}, {
let mut __t45: pciserial_board = unsafe { core::mem::zeroed() };
__t45.flags = 2u32;
__t45.num_ports = 1u32;
__t45.base_baud = 115200u32;
__t45.uart_offset = 8u32;
__t45
}, {
let mut __t46: pciserial_board = unsafe { core::mem::zeroed() };
__t46.flags = 2u32;
__t46.num_ports = 2u32;
__t46.base_baud = 115200u32;
__t46.uart_offset = 8u32;
__t46
}, {
let mut __t47: pciserial_board = unsafe { core::mem::zeroed() };
__t47.flags = 2u32;
__t47.num_ports = 4u32;
__t47.base_baud = 115200u32;
__t47.uart_offset = 8u32;
__t47
}, {
let mut __t48: pciserial_board = unsafe { core::mem::zeroed() };
__t48.flags = 2u32;
__t48.num_ports = 8u32;
__t48.base_baud = 115200u32;
__t48.uart_offset = 8u32;
__t48
}, {
let mut __t49: pciserial_board = unsafe { core::mem::zeroed() };
__t49.flags = 2u32;
__t49.num_ports = 1u32;
__t49.base_baud = 460800u32;
__t49.uart_offset = 8u32;
__t49
}, {
let mut __t50: pciserial_board = unsafe { core::mem::zeroed() };
__t50.flags = 2u32;
__t50.num_ports = 4u32;
__t50.base_baud = 460800u32;
__t50.uart_offset = 8u32;
__t50
}, {
let mut __t51: pciserial_board = unsafe { core::mem::zeroed() };
__t51.flags = 2u32;
__t51.num_ports = 8u32;
__t51.base_baud = 460800u32;
__t51.uart_offset = 8u32;
__t51
}, {
let mut __t52: pciserial_board = unsafe { core::mem::zeroed() };
__t52.flags = 2u32;
__t52.num_ports = 16u32;
__t52.base_baud = 460800u32;
__t52.uart_offset = 8u32;
__t52
}, {
let mut __t53: pciserial_board = unsafe { core::mem::zeroed() };
__t53.flags = 2u32;
__t53.num_ports = 1u32;
__t53.base_baud = 921600u32;
__t53.uart_offset = 8u32;
__t53
}, {
let mut __t54: pciserial_board = unsafe { core::mem::zeroed() };
__t54.flags = 2u32;
__t54.num_ports = 4u32;
__t54.base_baud = 921600u32;
__t54.uart_offset = 8u32;
__t54
}, {
let mut __t55: pciserial_board = unsafe { core::mem::zeroed() };
__t55.flags = 2u32;
__t55.num_ports = 8u32;
__t55.base_baud = 921600u32;
__t55.uart_offset = 8u32;
__t55
}, {
let mut __t56: pciserial_board = unsafe { core::mem::zeroed() };
__t56.flags = 2u32;
__t56.num_ports = 8u32;
__t56.base_baud = 1152000u32;
__t56.uart_offset = 8u32;
__t56
}, {
let mut __t57: pciserial_board = unsafe { core::mem::zeroed() };
__t57.flags = 10u32;
__t57.num_ports = 1u32;
__t57.base_baud = 115200u32;
__t57.uart_offset = 8u32;
__t57
}, {
let mut __t58: pciserial_board = unsafe { core::mem::zeroed() };
__t58.flags = 10u32;
__t58.num_ports = 2u32;
__t58.base_baud = 115200u32;
__t58.uart_offset = 8u32;
__t58
}, {
let mut __t59: pciserial_board = unsafe { core::mem::zeroed() };
__t59.flags = 10u32;
__t59.num_ports = 4u32;
__t59.base_baud = 115200u32;
__t59.uart_offset = 8u32;
__t59
}, {
let mut __t60: pciserial_board = unsafe { core::mem::zeroed() };
__t60.flags = 10u32;
__t60.num_ports = 2u32;
__t60.base_baud = 921600u32;
__t60.uart_offset = 8u32;
__t60
}, {
let mut __t61: pciserial_board = unsafe { core::mem::zeroed() };
__t61.flags = 10u32;
__t61.num_ports = 4u32;
__t61.base_baud = 921600u32;
__t61.uart_offset = 8u32;
__t61
}, {
let mut __t62: pciserial_board = unsafe { core::mem::zeroed() };
__t62.flags = 3u32;
__t62.num_ports = 2u32;
__t62.base_baud = 115200u32;
__t62.uart_offset = 8u32;
__t62
}, {
let mut __t63: pciserial_board = unsafe { core::mem::zeroed() };
__t63.flags = 3u32;
__t63.num_ports = 4u32;
__t63.base_baud = 115200u32;
__t63.uart_offset = 8u32;
__t63
}, {
let mut __t64: pciserial_board = unsafe { core::mem::zeroed() };
__t64.flags = 3u32;
__t64.num_ports = 8u32;
__t64.base_baud = 115200u32;
__t64.uart_offset = 8u32;
__t64
}, {
let mut __t65: pciserial_board = unsafe { core::mem::zeroed() };
__t65.flags = 4u32;
__t65.num_ports = 2u32;
__t65.base_baud = 921600u32;
__t65.uart_offset = 8u32;
__t65
}, {
let mut __t66: pciserial_board = unsafe { core::mem::zeroed() };
__t66.flags = 4u32;
__t66.num_ports = 4u32;
__t66.base_baud = 921600u32;
__t66.uart_offset = 8u32;
__t66
}, {
let mut __t67: pciserial_board = unsafe { core::mem::zeroed() };
__t67.flags = 4u32;
__t67.num_ports = 8u32;
__t67.base_baud = 921600u32;
__t67.uart_offset = 8u32;
__t67
}, {
let mut __t68: pciserial_board = unsafe { core::mem::zeroed() };
__t68.flags = 2u32;
__t68.num_ports = 2u32;
__t68.base_baud = 921600u32;
__t68.uart_offset = 1024u32;
__t68.reg_shift = 7u32;
__t68
}, {
let mut __t69: pciserial_board = unsafe { core::mem::zeroed() };
__t69.flags = 10u32;
__t69.num_ports = 2u32;
__t69.base_baud = 921600u32;
__t69.uart_offset = 1024u32;
__t69.reg_shift = 7u32;
__t69
}, {
let mut __t70: pciserial_board = unsafe { core::mem::zeroed() };
__t70.flags = 10u32;
__t70.num_ports = 4u32;
__t70.base_baud = 921600u32;
__t70.uart_offset = 1024u32;
__t70.reg_shift = 7u32;
__t70
}, pciserial_board { flags: 2u32, num_ports: 4u32, base_baud: 921600u32, uart_offset: 32u32, reg_shift: 2u32, first_offset: 3u32 }, {
let mut __t71: pciserial_board = unsafe { core::mem::zeroed() };
__t71.flags = 256u32;
__t71.num_ports = 32u32;
__t71.base_baud = 115200u32;
__t71.uart_offset = 8u32;
__t71
}, {
let mut __t72: pciserial_board = unsafe { core::mem::zeroed() };
__t72.flags = 0u32;
__t72.num_ports = 1u32;
__t72.base_baud = 15625000u32;
__t72.uart_offset = 512u32;
__t72.first_offset = 4096u32;
__t72
}, {
let mut __t73: pciserial_board = unsafe { core::mem::zeroed() };
__t73.flags = 0u32;
__t73.num_ports = 2u32;
__t73.base_baud = 15625000u32;
__t73.uart_offset = 512u32;
__t73.first_offset = 4096u32;
__t73
}, {
let mut __t74: pciserial_board = unsafe { core::mem::zeroed() };
__t74.flags = 0u32;
__t74.num_ports = 4u32;
__t74.base_baud = 15625000u32;
__t74.uart_offset = 512u32;
__t74.first_offset = 4096u32;
__t74
}, {
let mut __t75: pciserial_board = unsafe { core::mem::zeroed() };
__t75.flags = 0u32;
__t75.num_ports = 8u32;
__t75.base_baud = 15625000u32;
__t75.uart_offset = 512u32;
__t75.first_offset = 4096u32;
__t75
}, pciserial_board { flags: 0u32, num_ports: 32u32, base_baud: 921600u32, uart_offset: 32u32, reg_shift: 2u32, first_offset: 65536u32 }, pciserial_board { flags: 128u32, num_ports: 1u32, base_baud: 458333u32, uart_offset: 8u32, reg_shift: 0u32, first_offset: 131448u32 }, pciserial_board { flags: 0u32, num_ports: 4u32, base_baud: 921600u32, uart_offset: 64u32, reg_shift: 2u32, first_offset: 512u32 }, pciserial_board { flags: 0u32, num_ports: 6u32, base_baud: 921600u32, uart_offset: 64u32, reg_shift: 2u32, first_offset: 512u32 }, pciserial_board { flags: 0u32, num_ports: 8u32, base_baud: 921600u32, uart_offset: 64u32, reg_shift: 2u32, first_offset: 512u32 }, {
let mut __t76: pciserial_board = unsafe { core::mem::zeroed() };
__t76.flags = 0u32;
__t76.num_ports = 8u32;
__t76.base_baud = 460800u32;
__t76.uart_offset = 256u32;
__t76.reg_shift = 4u32;
__t76
}, {
let mut __t77: pciserial_board = unsafe { core::mem::zeroed() };
__t77.flags = 0u32;
__t77.num_ports = 1u32;
__t77.base_baud = 8333333u32;
__t77
}, {
let mut __t78: pciserial_board = unsafe { core::mem::zeroed() };
__t78.flags = 0u32;
__t78.num_ports = 2u32;
__t78.base_baud = 3686400u32;
__t78.uart_offset = 16u32;
__t78.first_offset = 2048u32;
__t78
}, {
let mut __t79: pciserial_board = unsafe { core::mem::zeroed() };
__t79.flags = 0u32;
__t79.num_ports = 4u32;
__t79.base_baud = 3686400u32;
__t79.uart_offset = 16u32;
__t79.first_offset = 2048u32;
__t79
}, {
let mut __t80: pciserial_board = unsafe { core::mem::zeroed() };
__t80.flags = 0u32;
__t80.num_ports = 8u32;
__t80.base_baud = 3686400u32;
__t80.uart_offset = 16u32;
__t80.first_offset = 2048u32;
__t80
}, {
let mut __t81: pciserial_board = unsafe { core::mem::zeroed() };
__t81.flags = 0u32;
__t81.num_ports = 16u32;
__t81.base_baud = 3686400u32;
__t81.uart_offset = 16u32;
__t81.first_offset = 2048u32;
__t81
}, {
let mut __t82: pciserial_board = unsafe { core::mem::zeroed() };
__t82.flags = 0u32;
__t82.num_ports = 1u32;
__t82.base_baud = 3906250u32;
__t82.uart_offset = 512u32;
__t82.first_offset = 4096u32;
__t82
}, {
let mut __t83: pciserial_board = unsafe { core::mem::zeroed() };
__t83.flags = 0u32;
__t83.num_ports = 2u32;
__t83.base_baud = 3906250u32;
__t83.uart_offset = 512u32;
__t83.first_offset = 4096u32;
__t83
}, {
let mut __t84: pciserial_board = unsafe { core::mem::zeroed() };
__t84.flags = 0u32;
__t84.num_ports = 4u32;
__t84.base_baud = 3906250u32;
__t84.uart_offset = 512u32;
__t84.first_offset = 4096u32;
__t84
}, {
let mut __t85: pciserial_board = unsafe { core::mem::zeroed() };
__t85.flags = 0u32;
__t85.num_ports = 8u32;
__t85.base_baud = 3906250u32;
__t85.uart_offset = 512u32;
__t85.first_offset = 4096u32;
__t85
}, {
let mut __t86: pciserial_board = unsafe { core::mem::zeroed() };
__t86.flags = 8u32;
__t86.num_ports = 2u32;
__t86.base_baud = 921600u32;
__t86.reg_shift = 2u32;
__t86
}, {
let mut __t87: pciserial_board = unsafe { core::mem::zeroed() };
__t87.flags = 0u32;
__t87.num_ports = 8u32;
__t87.base_baud = 115200u32;
__t87.uart_offset = 512u32;
__t87
}, {
let mut __t88: pciserial_board = unsafe { core::mem::zeroed() };
__t88.flags = 0u32;
__t88.num_ports = 2u32;
__t88.base_baud = 115200u32;
__t88
}, {
let mut __t89: pciserial_board = unsafe { core::mem::zeroed() };
__t89.flags = 0u32;
__t89.num_ports = 1u32;
__t89.base_baud = 115200u32;
__t89.reg_shift = 2u32;
__t89
}, {
let mut __t90: pciserial_board = unsafe { core::mem::zeroed() };
__t90.num_ports = 4u32;
__t90.base_baud = 115200u32;
__t90.uart_offset = 8u32;
__t90.first_offset = 64u32;
__t90
}, {
let mut __t91: pciserial_board = unsafe { core::mem::zeroed() };
__t91.num_ports = 8u32;
__t91.base_baud = 115200u32;
__t91.uart_offset = 8u32;
__t91.first_offset = 64u32;
__t91
}, {
let mut __t92: pciserial_board = unsafe { core::mem::zeroed() };
__t92.num_ports = 12u32;
__t92.base_baud = 115200u32;
__t92.uart_offset = 8u32;
__t92.first_offset = 64u32;
__t92
}, {
let mut __t93: pciserial_board = unsafe { core::mem::zeroed() };
__t93.num_ports = 4u32;
__t93.base_baud = 115200u32;
__t93.uart_offset = 8u32;
__t93
}, {
let mut __t94: pciserial_board = unsafe { core::mem::zeroed() };
__t94.num_ports = 8u32;
__t94.base_baud = 115200u32;
__t94.uart_offset = 8u32;
__t94
}, {
let mut __t95: pciserial_board = unsafe { core::mem::zeroed() };
__t95.num_ports = 12u32;
__t95.base_baud = 115200u32;
__t95.uart_offset = 8u32;
__t95
}, {
let mut __t96: pciserial_board = unsafe { core::mem::zeroed() };
__t96.flags = 0u32;
__t96.num_ports = 2u32;
__t96.base_baud = 115200u32;
__t96.uart_offset = 8u32;
__t96.first_offset = 192u32;
__t96
}, {
let mut __t97: pciserial_board = unsafe { core::mem::zeroed() };
__t97.flags = 0u32;
__t97.num_ports = 4u32;
__t97.base_baud = 115200u32;
__t97.uart_offset = 8u32;
__t97.first_offset = 192u32;
__t97
}, {
let mut __t98: pciserial_board = unsafe { core::mem::zeroed() };
__t98.flags = 0u32;
__t98.num_ports = 8u32;
__t98.base_baud = 115200u32;
__t98.uart_offset = 8u32;
__t98.first_offset = 0u32;
__t98
}, {
let mut __t99: pciserial_board = unsafe { core::mem::zeroed() };
__t99.num_ports = 1u32;
__t99.base_baud = 921600u32;
__t99.uart_offset = 8u32;
__t99
}, {
let mut __t100: pciserial_board = unsafe { core::mem::zeroed() };
__t100.num_ports = 2u32;
__t100.base_baud = 921600u32;
__t100.uart_offset = 8u32;
__t100
}, {
let mut __t101: pciserial_board = unsafe { core::mem::zeroed() };
__t101.num_ports = 4u32;
__t101.base_baud = 921600u32;
__t101.uart_offset = 8u32;
__t101
}, {
let mut __t102: pciserial_board = unsafe { core::mem::zeroed() };
__t102.num_ports = 8u32;
__t102.base_baud = 921600u32;
__t102.uart_offset = 8u32;
__t102
}, {
let mut __t103: pciserial_board = unsafe { core::mem::zeroed() };
__t103.num_ports = 16u32;
__t103.base_baud = 921600u32;
__t103.uart_offset = 8u32;
__t103
}, {
let mut __t104: pciserial_board = unsafe { core::mem::zeroed() };
__t104.flags = 0u32;
__t104.num_ports = 1u32;
__t104.base_baud = 4000000u32;
__t104.uart_offset = 512u32;
__t104.first_offset = 4096u32;
__t104
}, {
let mut __t105: pciserial_board = unsafe { core::mem::zeroed() };
__t105.flags = 0u32;
__t105.num_ports = 2u32;
__t105.base_baud = 4000000u32;
__t105.uart_offset = 512u32;
__t105.first_offset = 4096u32;
__t105
}, {
let mut __t106: pciserial_board = unsafe { core::mem::zeroed() };
__t106.flags = 0u32;
__t106.num_ports = 4u32;
__t106.base_baud = 4000000u32;
__t106.uart_offset = 512u32;
__t106.first_offset = 4096u32;
__t106
}, {
let mut __t107: pciserial_board = unsafe { core::mem::zeroed() };
__t107.flags = 0u32;
__t107.num_ports = 8u32;
__t107.base_baud = 4000000u32;
__t107.uart_offset = 512u32;
__t107.first_offset = 4096u32;
__t107
}]);

static mut __slate_str2970: [u8; 15] = [80, 65, 82, 80, 79, 82, 84, 95, 83, 69, 82, 73, 65, 76, 0];

static mut __slate_str2971: [u8; 15] = [80, 65, 82, 80, 79, 82, 84, 95, 83, 69, 82, 73, 65, 76, 0];

static mut __slate_str2972: [u8; 15] = [80, 65, 82, 80, 79, 82, 84, 95, 83, 69, 82, 73, 65, 76, 0];

static mut __slate_str2973: [u8; 17] = [83, 69, 82, 73, 65, 76, 95, 56, 50, 53, 48, 95, 77, 79, 88, 65, 0];

/// softmodems
/// ALi Corporation M5457 AC'97 Modem
/// Motorola Si3052-based modem
/// Si3052-based modem, default IDs
/// multi-io cards handled by parport_serial
///
/// WCH CH353 2S1P
/// WCH CH353 1S1P
/// WCH CH382 2S1P
/// Intel platforms with MID UART
/// Intel platforms with DesignWare UART
/// Exar devices
/// Pericom devices
/// Moxa devices
/// End of the black list
static mut blacklist: core::mem::MaybeUninit<__SlateImage_blacklist> = unsafe {
let mut __t0: core::mem::MaybeUninit<__SlateImage_blacklist> = core::mem::MaybeUninit::zeroed();
let __t1 = __t0.as_mut_ptr();
(*__t1).0[0].vendor = 4281u32;
(*__t1).0[0].device = 21591u32;
(*__t1).0[0].subvendor = 4294967295u32;
(*__t1).0[0].subdevice = 4294967295u32;
(*__t1).0[0].class = 0u32;
(*__t1).0[0].class_mask = 0u32;
(*__t1).0[1].vendor = 4183u32;
(*__t1).0[1].device = 12370u32;
(*__t1).0[1].subvendor = 4294967295u32;
(*__t1).0[1].subdevice = 4294967295u32;
(*__t1).0[1].class = 0u32;
(*__t1).0[1].class_mask = 0u32;
(*__t1).0[2].vendor = 5443u32;
(*__t1).0[2].device = 12370u32;
(*__t1).0[2].subvendor = 4294967295u32;
(*__t1).0[2].subdevice = 4294967295u32;
(*__t1).0[3].vendor = 17224u32;
(*__t1).0[3].device = 28755u32;
(*__t1).0[3].subvendor = 4294967295u32;
(*__t1).0[3].subdevice = 4294967295u32;
(*__t1).0[3].class = 0u32;
(*__t1).0[3].class_mask = 0u32;
(core::ptr::addr_of_mut!((*__t1).0[3].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*mut [u8; 15], *const ()>(core::ptr::addr_of_mut!(__slate_str2970)) });
(*__t1).0[4].vendor = 17224u32;
(*__t1).0[4].device = 20563u32;
(*__t1).0[4].subvendor = 4294967295u32;
(*__t1).0[4].subdevice = 4294967295u32;
(*__t1).0[4].class = 0u32;
(*__t1).0[4].class_mask = 0u32;
(core::ptr::addr_of_mut!((*__t1).0[4].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*mut [u8; 15], *const ()>(core::ptr::addr_of_mut!(__slate_str2971)) });
(*__t1).0[5].vendor = 7168u32;
(*__t1).0[5].device = 12880u32;
(*__t1).0[5].subvendor = 4294967295u32;
(*__t1).0[5].subdevice = 4294967295u32;
(*__t1).0[5].class = 0u32;
(*__t1).0[5].class_mask = 0u32;
(core::ptr::addr_of_mut!((*__t1).0[5].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*mut [u8; 15], *const ()>(core::ptr::addr_of_mut!(__slate_str2972)) });
(*__t1).0[6].vendor = 32902u32;
(*__t1).0[6].device = 2075u32;
(*__t1).0[6].subvendor = 4294967295u32;
(*__t1).0[6].subdevice = 4294967295u32;
(*__t1).0[6].class = 0u32;
(*__t1).0[6].class_mask = 0u32;
(*__t1).0[6].driver_data = 0u64;
(*__t1).0[7].vendor = 32902u32;
(*__t1).0[7].device = 2076u32;
(*__t1).0[7].subvendor = 4294967295u32;
(*__t1).0[7].subdevice = 4294967295u32;
(*__t1).0[7].class = 0u32;
(*__t1).0[7].class_mask = 0u32;
(*__t1).0[7].driver_data = 0u64;
(*__t1).0[8].vendor = 32902u32;
(*__t1).0[8].device = 2077u32;
(*__t1).0[8].subvendor = 4294967295u32;
(*__t1).0[8].subdevice = 4294967295u32;
(*__t1).0[8].class = 0u32;
(*__t1).0[8].class_mask = 0u32;
(*__t1).0[8].driver_data = 0u64;
(*__t1).0[9].vendor = 32902u32;
(*__t1).0[9].device = 4497u32;
(*__t1).0[9].subvendor = 4294967295u32;
(*__t1).0[9].subdevice = 4294967295u32;
(*__t1).0[9].class = 0u32;
(*__t1).0[9].class_mask = 0u32;
(*__t1).0[9].driver_data = 0u64;
(*__t1).0[10].vendor = 32902u32;
(*__t1).0[10].device = 6360u32;
(*__t1).0[10].subvendor = 4294967295u32;
(*__t1).0[10].subdevice = 4294967295u32;
(*__t1).0[10].class = 0u32;
(*__t1).0[10].class_mask = 0u32;
(*__t1).0[10].driver_data = 0u64;
(*__t1).0[11].vendor = 32902u32;
(*__t1).0[11].device = 6616u32;
(*__t1).0[11].subvendor = 4294967295u32;
(*__t1).0[11].subdevice = 4294967295u32;
(*__t1).0[11].class = 0u32;
(*__t1).0[11].class_mask = 0u32;
(*__t1).0[11].driver_data = 0u64;
(*__t1).0[12].vendor = 32902u32;
(*__t1).0[12].device = 2358u32;
(*__t1).0[12].subvendor = 4294967295u32;
(*__t1).0[12].subdevice = 4294967295u32;
(*__t1).0[12].class = 0u32;
(*__t1).0[12].class_mask = 0u32;
(*__t1).0[12].driver_data = 0u64;
(*__t1).0[13].vendor = 32902u32;
(*__t1).0[13].device = 3850u32;
(*__t1).0[13].subvendor = 4294967295u32;
(*__t1).0[13].subdevice = 4294967295u32;
(*__t1).0[13].class = 0u32;
(*__t1).0[13].class_mask = 0u32;
(*__t1).0[13].driver_data = 0u64;
(*__t1).0[14].vendor = 32902u32;
(*__t1).0[14].device = 3852u32;
(*__t1).0[14].subvendor = 4294967295u32;
(*__t1).0[14].subdevice = 4294967295u32;
(*__t1).0[14].class = 0u32;
(*__t1).0[14].class_mask = 0u32;
(*__t1).0[14].driver_data = 0u64;
(*__t1).0[15].vendor = 32902u32;
(*__t1).0[15].device = 8842u32;
(*__t1).0[15].subvendor = 4294967295u32;
(*__t1).0[15].subdevice = 4294967295u32;
(*__t1).0[15].class = 0u32;
(*__t1).0[15].class_mask = 0u32;
(*__t1).0[15].driver_data = 0u64;
(*__t1).0[16].vendor = 32902u32;
(*__t1).0[16].device = 8844u32;
(*__t1).0[16].subvendor = 4294967295u32;
(*__t1).0[16].subdevice = 4294967295u32;
(*__t1).0[16].class = 0u32;
(*__t1).0[16].class_mask = 0u32;
(*__t1).0[16].driver_data = 0u64;
(*__t1).0[17].vendor = 32902u32;
(*__t1).0[17].device = 19350u32;
(*__t1).0[17].subvendor = 4294967295u32;
(*__t1).0[17].subdevice = 4294967295u32;
(*__t1).0[17].class = 0u32;
(*__t1).0[17].class_mask = 0u32;
(*__t1).0[17].driver_data = 0u64;
(*__t1).0[18].vendor = 32902u32;
(*__t1).0[18].device = 19351u32;
(*__t1).0[18].subvendor = 4294967295u32;
(*__t1).0[18].subdevice = 4294967295u32;
(*__t1).0[18].class = 0u32;
(*__t1).0[18].class_mask = 0u32;
(*__t1).0[18].driver_data = 0u64;
(*__t1).0[19].vendor = 32902u32;
(*__t1).0[19].device = 19352u32;
(*__t1).0[19].subvendor = 4294967295u32;
(*__t1).0[19].subdevice = 4294967295u32;
(*__t1).0[19].class = 0u32;
(*__t1).0[19].class_mask = 0u32;
(*__t1).0[19].driver_data = 0u64;
(*__t1).0[20].vendor = 32902u32;
(*__t1).0[20].device = 19353u32;
(*__t1).0[20].subvendor = 4294967295u32;
(*__t1).0[20].subdevice = 4294967295u32;
(*__t1).0[20].class = 0u32;
(*__t1).0[20].class_mask = 0u32;
(*__t1).0[20].driver_data = 0u64;
(*__t1).0[21].vendor = 32902u32;
(*__t1).0[21].device = 19354u32;
(*__t1).0[21].subvendor = 4294967295u32;
(*__t1).0[21].subdevice = 4294967295u32;
(*__t1).0[21].class = 0u32;
(*__t1).0[21].class_mask = 0u32;
(*__t1).0[21].driver_data = 0u64;
(*__t1).0[22].vendor = 32902u32;
(*__t1).0[22].device = 19355u32;
(*__t1).0[22].subvendor = 4294967295u32;
(*__t1).0[22].subdevice = 4294967295u32;
(*__t1).0[22].class = 0u32;
(*__t1).0[22].class_mask = 0u32;
(*__t1).0[22].driver_data = 0u64;
(*__t1).0[23].vendor = 32902u32;
(*__t1).0[23].device = 40163u32;
(*__t1).0[23].subvendor = 4294967295u32;
(*__t1).0[23].subdevice = 4294967295u32;
(*__t1).0[23].class = 0u32;
(*__t1).0[23].class_mask = 0u32;
(*__t1).0[23].driver_data = 0u64;
(*__t1).0[24].vendor = 32902u32;
(*__t1).0[24].device = 40164u32;
(*__t1).0[24].subvendor = 4294967295u32;
(*__t1).0[24].subdevice = 4294967295u32;
(*__t1).0[24].class = 0u32;
(*__t1).0[24].class_mask = 0u32;
(*__t1).0[24].driver_data = 0u64;
(*__t1).0[25].vendor = 5032u32;
(*__t1).0[25].device = 4294967295u32;
(*__t1).0[25].subvendor = 4294967295u32;
(*__t1).0[25].subdevice = 4294967295u32;
(*__t1).0[25].class = 0u32;
(*__t1).0[25].class_mask = 0u32;
(*__t1).0[25].driver_data = 0u64;
(*__t1).0[26].vendor = 6391u32;
(*__t1).0[26].device = 4294967295u32;
(*__t1).0[26].subvendor = 4294967295u32;
(*__t1).0[26].subdevice = 4294967295u32;
(*__t1).0[26].class = 0u32;
(*__t1).0[26].class_mask = 0u32;
(*__t1).0[26].driver_data = 0u64;
(*__t1).0[27].vendor = 4824u32;
(*__t1).0[27].device = 4294967295u32;
(*__t1).0[27].subvendor = 4294967295u32;
(*__t1).0[27].subdevice = 4294967295u32;
(*__t1).0[27].class = 0u32;
(*__t1).0[27].class_mask = 0u32;
(*__t1).0[27].driver_data = 0u64;
(*__t1).0[28].vendor = 18767u32;
(*__t1).0[28].device = 4294967295u32;
(*__t1).0[28].subvendor = 4294967295u32;
(*__t1).0[28].subdevice = 4294967295u32;
(*__t1).0[28].class = 0u32;
(*__t1).0[28].class_mask = 0u32;
(*__t1).0[28].driver_data = 0u64;
(*__t1).0[29].vendor = 5011u32;
(*__t1).0[29].device = 4294967295u32;
(*__t1).0[29].subvendor = 4294967295u32;
(*__t1).0[29].subdevice = 4294967295u32;
(*__t1).0[29].class = 0u32;
(*__t1).0[29].class_mask = 0u32;
(core::ptr::addr_of_mut!((*__t1).0[29].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*mut [u8; 17], *const ()>(core::ptr::addr_of_mut!(__slate_str2973)) });
__t0
};

fn serial_pci_is_class_communication(dev: *mut pci_dev) -> i32 {
// If it is not a communications device or the programming
// interface is greater than 6, give up.
if (unsafe { (*dev).class }) >> 8i32 != 1792 && (unsafe { (*dev).class }) >> 8i32 != 1794 && (unsafe { (*dev).class }) >> 8i32 != 1795 || (unsafe { (*dev).class }) & 255 > 6 {
return 19i32.wrapping_neg();
}
0
}

/// Given a complete unknown PCI device, try to use some heuristics to
/// guess what the configuration might be, based on the pitiful PCI
/// serial specs.  Returns 0 on success, -ENODEV on failure.
fn serial_pci_guess_board(dev: *mut pci_dev, board: &mut pciserial_board) -> i32 {
let mut num_iomem: i32 = 0;
let mut num_port: i32 = 0;
let mut first_port: i32 = 1i32.wrapping_neg();
let mut i: i32 = 0;
let rc: i32 = serial_pci_is_class_communication(dev);
if rc != 0 {
return rc;
}
// Should we try to make guesses for multiport serial devices later?
if (unsafe { (*dev).class }) >> 8i32 == 1794 {
return 19i32.wrapping_neg();
}
while i < 6 {
if pci_resource_flags!(dev, i) & 256 != 0 {
num_port = num_port.wrapping_add(1);
if first_port == 1i32.wrapping_neg() {
first_port = i;
}
}
if pci_resource_flags!(dev, i) & 512 != 0 {
num_iomem = num_iomem.wrapping_add(1);
}
i = i.wrapping_add(1);
}
// If there is 1 or 0 iomem regions, and exactly one port,
// use it.  We guess the number of ports based on the IO
// region size.
if num_iomem <= 1 && num_port == 1 {
unsafe {
(*(board as *mut pciserial_board)).flags = first_port as u32;
}
unsafe {
(*(board as *mut pciserial_board)).num_ports = (pci_resource_len!(dev, first_port) / 8) as u32;
}
return 0;
}
// Now guess if we've got a board which indexes by BARs.
// Each IO BAR should be 8 bytes, and they should follow
// consecutively.
first_port = 1i32.wrapping_neg();
num_port = 0;
i = 0;
while i < 6 {
let __v8: bool = pci_resource_flags!(dev, i) & 256 != 0 && pci_resource_len!(dev, i) == 8;
if __v8 && (first_port == 1i32.wrapping_neg() || first_port.wrapping_add(num_port) == i) {
num_port = num_port.wrapping_add(1);
if first_port == 1i32.wrapping_neg() {
first_port = i;
}
}
i = i.wrapping_add(1);
}
if num_port > 1 {
unsafe {
(*(board as *mut pciserial_board)).flags = (first_port | 8) as u32;
}
unsafe {
(*(board as *mut pciserial_board)).num_ports = num_port as u32;
}
return 0;
}
19i32.wrapping_neg()
}

fn serial_pci_matches(board: *const pciserial_board, guessed: &pciserial_board) -> i32 {
((unsafe { (*board).num_ports }) == unsafe { (*(guessed as *const pciserial_board)).num_ports } && (unsafe { (*board).base_baud }) == unsafe { (*(guessed as *const pciserial_board)).base_baud } && (unsafe { (*board).uart_offset }) == unsafe { (*(guessed as *const pciserial_board)).uart_offset } && (unsafe { (*board).reg_shift }) == unsafe { (*(guessed as *const pciserial_board)).reg_shift } && (unsafe { (*board).first_offset }) == unsafe { (*(guessed as *const pciserial_board)).first_offset }) as i32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pciserial_init_ports")]
extern "C" fn pciserial_init_ports(dev: *mut pci_dev, board: *const pciserial_board) -> *mut serial_private {
let mut __slate_compound_20: u64 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_ptr: *mut serial_private = unsafe { core::mem::zeroed() };
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut __count: u64 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut nr_ports: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(nr_ports) as *mut i32;
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut quirk: core::mem::MaybeUninit<*mut pci_serial_quirk> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut pci_serial_quirk = core::ptr::addr_of_mut!(quirk) as *mut *mut pci_serial_quirk;
let mut r#priv: core::mem::MaybeUninit<*mut serial_private> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut serial_private = core::ptr::addr_of_mut!(r#priv) as *mut *mut serial_private;
let mut uart: core::mem::MaybeUninit<uart_8250_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut uart_8250_port = core::ptr::addr_of_mut!(uart) as *mut uart_8250_port;
unsafe {
'__join_0: {
{
let __slate_auto_bytes_2 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 840usize);
let __slate_auto_bytes_3 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_15 = (*board).num_ports as i32;
// Find an init and setup quirks.
*__slate_slot_17 = find_quirk(dev);
// Run the new-style initialization function.
// The initialization function returns:
//  <0  - error
//   0  - use board->num_ports
//  >0  - number of ports
if (*(*__slate_slot_17)).init != None {
*__slate_slot_16 = unsafe { (*(*__slate_slot_17)).init }.unwrap()(dev);
if *__slate_slot_16 < 0 {
*__slate_slot_18 = ERR_PTR(*__slate_slot_16 as i64) as *mut serial_private;
break '__join_0;
} else {
if *__slate_slot_16 != 0 {
*__slate_slot_15 = *__slate_slot_16;
}
}
}
}
__count = *__slate_slot_15 as u64;
__obj_size = struct_size_t__2!(__count);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_6 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_7 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
alloc_tag_restore!();
} else {
*__slate_slot_7 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_23 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_23)
});
}
__obj_ptr = *__slate_slot_7 as *mut serial_private;
if !__obj_ptr.is_null() {
let __v4: u64 = __count;
*core::ptr::addr_of_mut!(*{
__slate_compound_20 = 0u64;
core::ptr::addr_of_mut!(__slate_compound_20)
}) = __v4;
}
'__join_2: {
*__slate_slot_18 = __obj_ptr;
if (*__slate_slot_18).is_null() {
*__slate_slot_18 = ERR_PTR(12i32.wrapping_neg() as i64) as *mut serial_private;
} else {
(*(*__slate_slot_18)).dev = dev;
(*(*__slate_slot_18)).quirk = *__slate_slot_17;
memset(core::ptr::addr_of_mut!(*__slate_slot_19) as *mut (), 0i32, core::mem::size_of::<uart_8250_port>() as u64);
(*__slate_slot_19).port.flags = 64u64 | BIT_ULL!(28i32) | BIT_ULL!(24i32);
(*__slate_slot_19).port.uartclk = unsafe { (*board).base_baud }.wrapping_mul(16);
if (*board).flags & 128 != 0 {
(*__slate_slot_19).port.irq = 0;
} else {
if !pci_match_id(core::ptr::addr_of!(pci_use_msi.0) as *const pci_device_id, dev).is_null() {
pci_set_master(dev);
let __v3: u64 = (*__slate_slot_19).port.flags;
let __v2: u64 = __v3 & !BIT_ULL!(24i32);
(*__slate_slot_19).port.flags = __v2;
*__slate_slot_16 = pci_alloc_irq_vectors(dev, 1u32, 1u32, 7u32);
} else {
*__slate_slot_16 = pci_alloc_irq_vectors(dev, 1u32, 1u32, 1u32);
}
if *__slate_slot_16 < 0 {
kfree(*__slate_slot_18 as *const ());
*__slate_slot_18 = ERR_PTR(*__slate_slot_16 as i64) as *mut serial_private;
break '__join_2;
} else {
(*__slate_slot_19).port.irq = pci_irq_vector(dev, 0u32) as u32;
}
}
(*__slate_slot_19).port.dev = core::ptr::addr_of_mut!((*dev).dev);
*__slate_slot_14 = 0;
'__join_3: {
loop {
if *__slate_slot_14 < *__slate_slot_15 {
if unsafe { (*(*__slate_slot_17)).setup }.unwrap()(*__slate_slot_18, board, core::ptr::addr_of_mut!(*__slate_slot_19), *__slate_slot_14) != 0 {
break '__join_3;
} else {
*(core::ptr::addr_of_mut!((*(*__slate_slot_18)).line) as *mut i32).wrapping_offset(*__slate_slot_14 as isize) = serial8250_register_8250_port(core::ptr::addr_of_mut!(*__slate_slot_19) as *const uart_8250_port);
if *(core::ptr::addr_of_mut!((*(*__slate_slot_18)).line) as *mut i32).wrapping_offset(*__slate_slot_14 as isize) < 0 {
break;
} else {
let __v1: i32 = *__slate_slot_14;
*__slate_slot_14 = __v1.wrapping_add(1);
}
}
} else {
break '__join_3;
}
}
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Couldn't register serial port %lx, irq %d, type %d, error %d\n".as_ptr() as *const u8, (*__slate_slot_19).port.iobase, (*__slate_slot_19).port.irq, (*__slate_slot_19).port.iotype as i32, *(core::ptr::addr_of_mut!((*(*__slate_slot_18)).line) as *mut i32).wrapping_offset(*__slate_slot_14 as isize));
}
(*(*__slate_slot_18)).nr = *__slate_slot_14 as u32;
(*(*__slate_slot_18)).board = board;
return *__slate_slot_18;
}
}
if (*(*__slate_slot_17)).exit != None {
unsafe { (*(*__slate_slot_17)).exit }.unwrap()(dev);
}
}
return *__slate_slot_18;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pciserial_init_ports_598: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *const pciserial_board) -> *mut serial_private>, *mut ()>(Some(pciserial_init_ports)) };

fn pciserial_detach_ports(r#priv: *mut serial_private) {
let mut i: i32 = 0;
while (i as u32) < unsafe { (*r#priv).nr } {
unsafe { serial8250_unregister_port(*(core::ptr::addr_of_mut!((*r#priv).line) as *mut i32).wrapping_offset(i as isize)) };
i = i.wrapping_add(1);
}
// Find the exit quirks.
let quirk: *mut pci_serial_quirk = find_quirk(unsafe { (*r#priv).dev });
if (unsafe { (*quirk).exit }) != None {
unsafe { unsafe { (*quirk).exit }.unwrap()((*r#priv).dev) };
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pciserial_remove_ports")]
extern "C" fn pciserial_remove_ports(r#priv: *mut serial_private) {
pciserial_detach_ports(r#priv);
unsafe { kfree(r#priv as *const ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pciserial_remove_ports_599: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut serial_private)>, *mut ()>(Some(pciserial_remove_ports)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pciserial_suspend_ports")]
extern "C" fn pciserial_suspend_ports(r#priv: *mut serial_private) {
let mut i: i32 = 0;
while (i as u32) < unsafe { (*r#priv).nr } {
if (unsafe { *(core::ptr::addr_of_mut!((*r#priv).line) as *mut i32).wrapping_offset(i as isize) }) >= 0 {
unsafe { serial8250_suspend_port(*(core::ptr::addr_of_mut!((*r#priv).line) as *mut i32).wrapping_offset(i as isize)) };
}
i = i.wrapping_add(1);
}
// Ensure that every init quirk is properly torn down
if (unsafe { (*(*r#priv).quirk).exit }) != None {
unsafe { unsafe { (*(*r#priv).quirk).exit }.unwrap()((*r#priv).dev) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pciserial_suspend_ports_600: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut serial_private)>, *mut ()>(Some(pciserial_suspend_ports)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pciserial_resume_ports")]
extern "C" fn pciserial_resume_ports(r#priv: *mut serial_private) {
let mut i: i32 = 0;
// Ensure that the board is correctly configured.
if (unsafe { (*(*r#priv).quirk).init }) != None {
unsafe { unsafe { (*(*r#priv).quirk).init }.unwrap()((*r#priv).dev) };
}
while (i as u32) < unsafe { (*r#priv).nr } {
if (unsafe { *(core::ptr::addr_of_mut!((*r#priv).line) as *mut i32).wrapping_offset(i as isize) }) >= 0 {
unsafe { serial8250_resume_port(*(core::ptr::addr_of_mut!((*r#priv).line) as *mut i32).wrapping_offset(i as isize)) };
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pciserial_resume_ports_601: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut serial_private)>, *mut ()>(Some(pciserial_resume_ports)) };

/// Probe one serial board.  Unfortunately, there is no rhyme nor reason
/// to the arrangement of serial ports on a PCI card.
#[unsafe(link_section = ".text.slate_distinct._8250_pci.pciserial_init_one")]
extern "C" fn pciserial_init_one(dev: *mut pci_dev, ent: *const pci_device_id) -> i32 {
let mut board: *const pciserial_board = unsafe { core::mem::zeroed() };
let mut tmp: pciserial_board = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut rc: i32 = 0;
let quirk: *mut pci_serial_quirk = find_quirk(dev);
if (unsafe { (*quirk).probe }) != None {
rc = unsafe { unsafe { (*quirk).probe }.unwrap()(dev) };
if rc != 0 {
return rc;
}
}
if (unsafe { (*ent).driver_data }) >= ARRAY_SIZE__6!() {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"invalid driver_data: %ld\n".as_ptr() as *const u8, (*ent).driver_data) };
return 22i32.wrapping_neg();
}
board = unsafe { core::ptr::addr_of_mut!(pci_boards.0) as *mut pciserial_board }.wrapping_offset((unsafe { (*ent).driver_data }) as isize) as *const pciserial_board;
let exclude: *const pci_device_id = unsafe { pci_match_id(core::ptr::addr_of!((*(core::ptr::addr_of_mut!(blacklist) as *mut __SlateImage_blacklist)).0) as *const pci_device_id, dev) };
if !exclude.is_null() {
if (unsafe { (*exclude).driver_data }) != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ignoring port, enable %s to handle\n".as_ptr() as *const u8, (*exclude).driver_data as *const u8) };
}
return 19i32.wrapping_neg();
}
rc = unsafe { pcim_enable_device(dev) };
unsafe { pci_save_state(dev) };
if rc != 0 {
return rc;
}
if (unsafe { (*ent).driver_data }) == 0 {
// Use a copy of the pci_board entry for this;
// avoid changing entries in the table.
unsafe { memcpy(core::ptr::addr_of_mut!(tmp) as *mut (), board as *const (), core::mem::size_of::<pciserial_board>() as u64) };
board = core::ptr::addr_of_mut!(tmp) as *const pciserial_board;
// We matched one of our class entries.  Try to
// determine the parameters of this board.
rc = serial_pci_guess_board(dev, unsafe { &mut (*core::ptr::addr_of_mut!(tmp)) });
if rc != 0 {
return rc;
}
} else {
// We matched an explicit entry.  If we are able to
// detect this boards settings with our heuristic,
// then we no longer need this entry.
unsafe { memcpy(core::ptr::addr_of_mut!(tmp) as *mut (), (core::ptr::addr_of_mut!(pci_boards.0) as *mut pciserial_board).wrapping_offset(0 as isize) as *const (), core::mem::size_of::<pciserial_board>() as u64) };
rc = serial_pci_guess_board(dev, unsafe { &mut (*core::ptr::addr_of_mut!(tmp)) });
let __v2: bool = rc == 0 && serial_pci_matches(board, unsafe { &(*(core::ptr::addr_of_mut!(tmp) as *const pciserial_board)) }) != 0;
if __v2 {
moan_device(c"Redundant entry in serial pci_table.".as_ptr() as *const u8, dev);
}
}
let r#priv: *mut serial_private = pciserial_init_ports(dev, board);
if IS_ERR(r#priv as *const ()) {
return PTR_ERR(r#priv as *const ()) as i32;
}
pci_set_drvdata(dev, r#priv as *mut ());
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pciserial_remove_one")]
extern "C" fn pciserial_remove_one(dev: *mut pci_dev) {
let r#priv: *mut serial_private = pci_get_drvdata(dev) as *mut serial_private;
pciserial_remove_ports(r#priv);
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pciserial_suspend_one")]
extern "C" fn pciserial_suspend_one(dev: *mut device) -> i32 {
let r#priv: *mut serial_private = dev_get_drvdata(dev as *const device) as *mut serial_private;
if !r#priv.is_null() {
pciserial_suspend_ports(r#priv);
}
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.pciserial_resume_one")]
extern "C" fn pciserial_resume_one(dev: *mut device) -> i32 {
let pdev: *mut pci_dev = to_pci_dev!(dev);
let r#priv: *mut serial_private = pci_get_drvdata(pdev) as *mut serial_private;
let mut err: i32 = 0;
if !r#priv.is_null() {
// The device may have been disabled.  Re-enable it.
err = unsafe { pci_enable_device(pdev) };
// FIXME: We cannot simply error out here
if err != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Unable to re-enable ports, trying to continue.\n".as_ptr() as *const u8) };
}
pciserial_resume_ports(r#priv);
}
0
}

static pciserial_pm_ops: __SlateConst<dev_pm_ops> = __SlateConst({
let mut __t0: dev_pm_ops = unsafe { core::mem::zeroed() };
__t0.suspend = Some(pciserial_suspend_one);
__t0.resume = Some(pciserial_resume_one);
__t0.freeze = Some(pciserial_suspend_one);
__t0.thaw = Some(pciserial_resume_one);
__t0.poweroff = Some(pciserial_suspend_one);
__t0.restore = Some(pciserial_resume_one);
__t0
});

/// Advantech use PCI_DEVICE_ID_ADVANTECH_PCI3620 (0x3620) as 'PCI_SUBVENDOR_ID'
/// Advantech also use 0x3618 and 0xf618
/// VScom SPCOM800, from sl@s.pl
/// Unknown card - subdevice 0x1584
/// Unknown card - subdevice 0x1588
/// Megawolf Romulus PCI Serial Card, from Mike Hudson
/// (Exoray@isys.ca)
/// Quatech cards. These actually have configurable clocks but for
/// now we just use the default.
///
/// 100 series are RS232, 200 series RS422,
/// The below card is a little controversial since it is the
/// subject of a PCI vendor/device ID clash.  (See
/// www.ussg.iu.edu/hypermail/linux/kernel/0303.1/0516.html).
/// For now just used the hex ID 0x950a.
/// Oxford Semiconductor Inc. Tornado PCI express device range.
/// OXPCIe952 1 Legacy UART
/// OXPCIe952 1 Legacy UART
/// OXPCIe952 1 Native UART
/// OXPCIe952 1 Native UART
/// OXPCIe952 1 Legacy UART
/// OXPCIe952 1 Legacy UART
/// OXPCIe952 1 Native UART
/// OXPCIe952 1 Native UART
/// OXPCIe952 1 Legacy UART
/// OXPCIe952 1 Legacy UART
/// OXPCIe952 1 Legacy UART
/// OXPCIe952 1 Legacy UART
/// OXPCIe952 2 Native UART
/// OXPCIe952 2 Native UART
/// OXPCIe954 4 Native UART
/// OXPCIe954 4 Native UART
/// OXPCIe958 8 Native UART
/// OXPCIe958 8 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// OXPCIe200 1 Native UART
/// Mainpine Inc. IQ Express "Rev3" utilizing OxSemi Tornado
/// IQ Express 1 Port V.34 Super-G3 Fax
/// IQ Express 2 Port V.34 Super-G3 Fax
/// IQ Express 4 Port V.34 Super-G3 Fax
/// IQ Express 8 Port V.34 Super-G3 Fax
/// Digi/IBM PCIe 2-port Async EIA-232 Adapter utilizing OxSemi Tornado
/// EndRun Technologies. PCI express device range.
/// EndRun PTP/1588 has 2 Native UARTs utilizing OxSemi 952.
/// SBS Technologies, Inc. P-Octal and PMC-OCTPRO cards,
/// from skokodyn@yahoo.com
/// Digitan DS560-558, from jimd@esoft.com
/// Titan Electronic cards
///  The 400L and 800L have a custom setup quirk.
/// Computone devices submitted by Doug McNash dmcnash@computone.com
/// Sunix PCI serial boards
/// AFAVLAB serial card, from Harald Welte <laforge@gnumonks.org>
/// Korenix Jetcard F0/F1 cards (JC1204, JC1208, JC1404, JC1408).
/// Cards are identified by their subsystem vendor IDs, which
/// (in hex) match the model number.
///
/// Note that JC140x are RS422/485 cards which require ox950
/// ACR = 0x10, and as such are not currently fully supported.
/// 	{
/// 	PCI_VDEVICE_SUB(KORENIX, PCI_DEVICE_ID_KORENIX_JETCARDF0,
/// 			0x1402, 0x0002),
/// 	.driver_data = pbn_b0_2_921600,
/// },
///
/// 	{
/// 	PCI_VDEVICE_SUB(KORENIX, PCI_DEVICE_ID_KORENIX_JETCARDF0,
/// 			0x1404, 0x0004),
/// 	.driver_data = pbn_b0_4_921600,
/// },
/// Dell Remote Access Card 4 - Tim_T_Murphy@Dell.com
/// Dell Remote Access Card III - Tim_T_Murphy@Dell.com
/// RAStel 2 port modem, gerg@moreton.com.au
/// EKF addition for i960 Boards form EKF with serial port
/// Xircom Cardbus/Ethernet combos
/// Xircom RBM56G cardbus modem - Dirk Arnold (temp entry)
/// Untested PCI modems, sent in from various folks...
/// Elsa Model 56K PCI Modem, from Andreas Rath <arh@01019freenet.de>
/// HP Diva card
/// HPE PCI serial device
/// Topic TP560 Data/Fax/Voice 56k modem (reported by Evan Clarke)
/// ITE
/// IntaShield IS-100
/// IntaShield IS-200; 135a.0d80
/// IntaShield IS-400; 135a.0dc0
/// IntaShield IX-100
/// IntaShield IX-200
/// IntaShield IX-400
/// Brainboxes Devices
///
/// Brainboxes UC-101
/// Brainboxes UC-235/246
/// Brainboxes UC-253/UC-734
/// Brainboxes UC-260/271/701/756
/// Brainboxes UC-268
/// Brainboxes UC-275/279
/// Brainboxes UC-302
/// Brainboxes UC-310
/// Brainboxes UC-313
/// Brainboxes UC-320/324
/// Brainboxes UC-346
/// Brainboxes UC-357
/// Brainboxes UC-368
/// Brainboxes UC-420
/// Brainboxes UC-607
/// Brainboxes UC-836
/// Brainboxes UP-189
/// Brainboxes UP-200
/// Brainboxes UP-869
/// Brainboxes UP-880
/// Brainboxes PX-101
/// Brainboxes PX-235/246
/// Brainboxes PX-203/PX-257
/// Brainboxes PX-260/PX-701
/// Brainboxes PX-275/279
/// Brainboxes PX-310
/// Brainboxes PX-313
/// Brainboxes PX-320/324/PX-376/PX-387
/// Brainboxes PX-335/346
/// Brainboxes PX-368
/// Brainboxes PX-420
/// Brainboxes PX-475
/// Brainboxes PX-803/PX-857
/// Brainboxes PX-820
/// Brainboxes PX-835/PX-846
/// Brainboxes XC-235
/// Brainboxes XC-475
/// Perle PCI-RAS cards
/// Mainpine series cards: Fairly standard layout but fools
/// parts of the autodetect in some cases and uses otherwise
/// unmatched communications subclasses in the PCI Express case
/// RockForceDUO
/// RockForceQUATRO
/// RockForceDUO+
/// RockForceQUATRO+
/// RockForce+
/// RockForce+
/// RockForceOCTO+
/// RockForceDUO+
/// RockForceQUARTRO+
/// RockForceOCTO+
/// RockForceD1
/// RockForceF1
/// RockForceD2
/// RockForceF2
/// RockForceD4
/// RockForceF4
/// RockForceD8
/// RockForceF8
/// IQ Express D1
/// IQ Express F1
/// IQ Express D2
/// IQ Express F2
/// IQ Express D4
/// IQ Express F4
/// IQ Express D8
/// IQ Express F8
/// PA Semi PA6T-1682M on-chip UART
/// National Instruments
/// ADDI-DATA GmbH communication cards <info@addi-data.com>
/// other NetMos 9835 devices are most likely handled by the
/// parport_serial driver, check drivers/parport/parport_serial.c
/// before adding them here.
/// the 9901 is a rebranded 9912
/// Best Connectivity and Rosewill PCI Multi I/O cards
/// ASIX AX99100 PCIe to Multi I/O Controller
/// Intel CE4100
/// Cronyx Omega PCI
/// Broadcom TruManage
/// AgeStar as-prs2-009
/// WCH CH353 series devices: The 2S1P is handled by parport_serial
/// so not listed here.
/// Realtek RealManage
/// Systembase Multi I/O cards
/// Fintek PCI serial cards
/// MKS Tenta SCOM-080x serial cards
/// Amazon PCI serial device
/// These entries match devices with class COMMUNICATION_SERIAL,
/// COMMUNICATION_MODEM or COMMUNICATION_MULTISERIAL
static serial_pci_tbl: __SlateAlign16<[pci_device_id; 409]> = __SlateAlign16([{
let mut __t0: pci_device_id = unsafe { core::mem::zeroed() };
__t0.vendor = 5118u32;
__t0.device = 5632u32;
__t0.subvendor = 5649u32;
__t0.subdevice = 4294967295u32;
__t0.class = 0u32;
__t0.class_mask = 0u32;
__t0.driver_data = 8u64;
__t0
}, {
let mut __t1: pci_device_id = unsafe { core::mem::zeroed() };
__t1.vendor = 5118u32;
__t1.device = 13856u32;
__t1.subvendor = 13856u32;
__t1.subdevice = 1u32;
__t1.class = 0u32;
__t1.class_mask = 0u32;
__t1.driver_data = 55u64;
__t1
}, {
let mut __t2: pci_device_id = unsafe { core::mem::zeroed() };
__t2.vendor = 5118u32;
__t2.device = 13848u32;
__t2.subvendor = 13848u32;
__t2.subdevice = 4294967295u32;
__t2.class = 0u32;
__t2.class_mask = 0u32;
__t2.driver_data = 8u64;
__t2
}, {
let mut __t3: pci_device_id = unsafe { core::mem::zeroed() };
__t3.vendor = 5118u32;
__t3.device = 63000u32;
__t3.subvendor = 13848u32;
__t3.subdevice = 4294967295u32;
__t3.class = 0u32;
__t3.class_mask = 0u32;
__t3.driver_data = 8u64;
__t3
}, {
let mut __t4: pci_device_id = unsafe { core::mem::zeroed() };
__t4.vendor = 4528u32;
__t4.device = 1u32;
__t4.subvendor = 4804u32;
__t4.subdevice = 1u32;
__t4.class = 0u32;
__t4.class_mask = 0u32;
__t4.driver_data = 44u64;
__t4
}, {
let mut __t5: pci_device_id = unsafe { core::mem::zeroed() };
__t5.vendor = 4528u32;
__t5.device = 1u32;
__t5.subvendor = 4804u32;
__t5.subdevice = 2u32;
__t5.class = 0u32;
__t5.class_mask = 0u32;
__t5.driver_data = 43u64;
__t5
}, {
let mut __t6: pci_device_id = unsafe { core::mem::zeroed() };
__t6.vendor = 4528u32;
__t6.device = 1u32;
__t6.subvendor = 4804u32;
__t6.subdevice = 3u32;
__t6.class = 0u32;
__t6.class_mask = 0u32;
__t6.driver_data = 42u64;
__t6
}, {
let mut __t7: pci_device_id = unsafe { core::mem::zeroed() };
__t7.vendor = 4528u32;
__t7.device = 2u32;
__t7.subvendor = 4804u32;
__t7.subdevice = 1u32;
__t7.class = 0u32;
__t7.class_mask = 0u32;
__t7.driver_data = 44u64;
__t7
}, {
let mut __t8: pci_device_id = unsafe { core::mem::zeroed() };
__t8.vendor = 4528u32;
__t8.device = 2u32;
__t8.subvendor = 4804u32;
__t8.subdevice = 2u32;
__t8.class = 0u32;
__t8.class_mask = 0u32;
__t8.driver_data = 43u64;
__t8
}, {
let mut __t9: pci_device_id = unsafe { core::mem::zeroed() };
__t9.vendor = 4528u32;
__t9.device = 2u32;
__t9.subvendor = 4804u32;
__t9.subdevice = 3u32;
__t9.class = 0u32;
__t9.class_mask = 0u32;
__t9.driver_data = 42u64;
__t9
}, {
let mut __t10: pci_device_id = unsafe { core::mem::zeroed() };
__t10.vendor = 4528u32;
__t10.device = 2u32;
__t10.subvendor = 4804u32;
__t10.subdevice = 4u32;
__t10.class = 0u32;
__t10.class_mask = 0u32;
__t10.driver_data = 35u64;
__t10
}, {
let mut __t11: pci_device_id = unsafe { core::mem::zeroed() };
__t11.vendor = 4528u32;
__t11.device = 2u32;
__t11.subvendor = 4804u32;
__t11.subdevice = 5u32;
__t11.class = 0u32;
__t11.class_mask = 0u32;
__t11.driver_data = 35u64;
__t11
}, {
let mut __t12: pci_device_id = unsafe { core::mem::zeroed() };
__t12.vendor = 4528u32;
__t12.device = 2u32;
__t12.subvendor = 4804u32;
__t12.subdevice = 6u32;
__t12.class = 0u32;
__t12.class_mask = 0u32;
__t12.driver_data = 34u64;
__t12
}, {
let mut __t13: pci_device_id = unsafe { core::mem::zeroed() };
__t13.vendor = 4528u32;
__t13.device = 2u32;
__t13.subvendor = 4804u32;
__t13.subdevice = 7u32;
__t13.class = 0u32;
__t13.class_mask = 0u32;
__t13.driver_data = 34u64;
__t13
}, {
let mut __t14: pci_device_id = unsafe { core::mem::zeroed() };
__t14.vendor = 4528u32;
__t14.device = 2u32;
__t14.subvendor = 4804u32;
__t14.subdevice = 8u32;
__t14.class = 0u32;
__t14.class_mask = 0u32;
__t14.driver_data = 33u64;
__t14
}, {
let mut __t15: pci_device_id = unsafe { core::mem::zeroed() };
__t15.vendor = 4528u32;
__t15.device = 2u32;
__t15.subvendor = 4804u32;
__t15.subdevice = 9u32;
__t15.class = 0u32;
__t15.class_mask = 0u32;
__t15.driver_data = 35u64;
__t15
}, {
let mut __t16: pci_device_id = unsafe { core::mem::zeroed() };
__t16.vendor = 4528u32;
__t16.device = 2u32;
__t16.subvendor = 4804u32;
__t16.subdevice = 10u32;
__t16.class = 0u32;
__t16.class_mask = 0u32;
__t16.driver_data = 35u64;
__t16
}, {
let mut __t17: pci_device_id = unsafe { core::mem::zeroed() };
__t17.vendor = 4528u32;
__t17.device = 2u32;
__t17.subvendor = 4804u32;
__t17.subdevice = 11u32;
__t17.class = 0u32;
__t17.class_mask = 0u32;
__t17.driver_data = 34u64;
__t17
}, {
let mut __t18: pci_device_id = unsafe { core::mem::zeroed() };
__t18.vendor = 4528u32;
__t18.device = 2u32;
__t18.subvendor = 4804u32;
__t18.subdevice = 12u32;
__t18.class = 0u32;
__t18.class_mask = 0u32;
__t18.driver_data = 36u64;
__t18
}, {
let mut __t19: pci_device_id = unsafe { core::mem::zeroed() };
__t19.vendor = 5141u32;
__t19.device = 38145u32;
__t19.subvendor = 4804u32;
__t19.subdevice = 513u32;
__t19.class = 0u32;
__t19.class_mask = 0u32;
__t19.driver_data = 13u64;
__t19
}, {
let mut __t20: pci_device_id = unsafe { core::mem::zeroed() };
__t20.vendor = 5141u32;
__t20.device = 38145u32;
__t20.subvendor = 4804u32;
__t20.subdevice = 514u32;
__t20.class = 0u32;
__t20.class_mask = 0u32;
__t20.driver_data = 14u64;
__t20
}, {
let mut __t21: pci_device_id = unsafe { core::mem::zeroed() };
__t21.vendor = 5141u32;
__t21.device = 38145u32;
__t21.subvendor = 5339u32;
__t21.subdevice = 8528u32;
__t21.class = 0u32;
__t21.class_mask = 0u32;
__t21.driver_data = 11u64;
__t21
}, {
let mut __t22: pci_device_id = unsafe { core::mem::zeroed() };
__t22.vendor = 4958u32;
__t22.device = 28929u32;
__t22.subvendor = 4294967295u32;
__t22.subdevice = 4294967295u32;
__t22.class = 0u32;
__t22.class_mask = 0u32;
__t22.driver_data = 57u64;
__t22
}, {
let mut __t23: pci_device_id = unsafe { core::mem::zeroed() };
__t23.vendor = 4958u32;
__t23.device = 29185u32;
__t23.subvendor = 4294967295u32;
__t23.subdevice = 4294967295u32;
__t23.class = 0u32;
__t23.class_mask = 0u32;
__t23.driver_data = 58u64;
__t23
}, {
let mut __t24: pci_device_id = unsafe { core::mem::zeroed() };
__t24.vendor = 4958u32;
__t24.device = 29698u32;
__t24.subvendor = 4294967295u32;
__t24.subdevice = 4294967295u32;
__t24.class = 0u32;
__t24.class_mask = 0u32;
__t24.driver_data = 59u64;
__t24
}, {
let mut __t25: pci_device_id = unsafe { core::mem::zeroed() };
__t25.vendor = 4958u32;
__t25.device = 29186u32;
__t25.subvendor = 4294967295u32;
__t25.subdevice = 4294967295u32;
__t25.class = 0u32;
__t25.class_mask = 0u32;
__t25.driver_data = 58u64;
__t25
}, {
let mut __t26: pci_device_id = unsafe { core::mem::zeroed() };
__t26.vendor = 4958u32;
__t26.device = 29697u32;
__t26.subvendor = 4294967295u32;
__t26.subdevice = 4294967295u32;
__t26.class = 0u32;
__t26.class_mask = 0u32;
__t26.driver_data = 59u64;
__t26
}, {
let mut __t27: pci_device_id = unsafe { core::mem::zeroed() };
__t27.vendor = 4958u32;
__t27.device = 30721u32;
__t27.subvendor = 4294967295u32;
__t27.subdevice = 4294967295u32;
__t27.class = 0u32;
__t27.class_mask = 0u32;
__t27.driver_data = 48u64;
__t27
}, {
let mut __t28: pci_device_id = unsafe { core::mem::zeroed() };
__t28.vendor = 4958u32;
__t28.device = 30723u32;
__t28.subvendor = 4294967295u32;
__t28.subdevice = 4294967295u32;
__t28.class = 0u32;
__t28.class_mask = 0u32;
__t28.driver_data = 51u64;
__t28
}, {
let mut __t29: pci_device_id = unsafe { core::mem::zeroed() };
__t29.vendor = 4958u32;
__t29.device = 30724u32;
__t29.subvendor = 4294967295u32;
__t29.subdevice = 4294967295u32;
__t29.class = 0u32;
__t29.class_mask = 0u32;
__t29.driver_data = 48u64;
__t29
}, {
let mut __t30: pci_device_id = unsafe { core::mem::zeroed() };
__t30.vendor = 4277u32;
__t30.device = 40961u32;
__t30.subvendor = 4294967295u32;
__t30.subdevice = 4294967295u32;
__t30.class = 0u32;
__t30.class_mask = 0u32;
__t30.driver_data = 58u64;
__t30
}, {
let mut __t31: pci_device_id = unsafe { core::mem::zeroed() };
__t31.vendor = 4277u32;
__t31.device = 4355u32;
__t31.subvendor = 4294967295u32;
__t31.subdevice = 4294967295u32;
__t31.class = 0u32;
__t31.class_mask = 0u32;
__t31.driver_data = 60u64;
__t31
}, {
let mut __t32: pci_device_id = unsafe { core::mem::zeroed() };
__t32.vendor = 4277u32;
__t32.device = 4214u32;
__t32.subvendor = 4294967295u32;
__t32.subdevice = 4294967295u32;
__t32.class = 0u32;
__t32.class_mask = 0u32;
__t32.driver_data = 55u64;
__t32
}, {
let mut __t33: pci_device_id = unsafe { core::mem::zeroed() };
__t33.vendor = 4277u32;
__t33.device = 4215u32;
__t33.subvendor = 4294967295u32;
__t33.subdevice = 4294967295u32;
__t33.class = 0u32;
__t33.class_mask = 0u32;
__t33.driver_data = 54u64;
__t33
}, {
let mut __t34: pci_device_id = unsafe { core::mem::zeroed() };
__t34.vendor = 4277u32;
__t34.device = 36944u32;
__t34.subvendor = 4277u32;
__t34.subdevice = 5508u32;
__t34.class = 0u32;
__t34.class_mask = 0u32;
__t34.driver_data = 47u64;
__t34
}, {
let mut __t35: pci_device_id = unsafe { core::mem::zeroed() };
__t35.vendor = 4277u32;
__t35.device = 36944u32;
__t35.subvendor = 4277u32;
__t35.subdevice = 5512u32;
__t35.class = 0u32;
__t35.class_mask = 0u32;
__t35.driver_data = 48u64;
__t35
}, {
let mut __t36: pci_device_id = unsafe { core::mem::zeroed() };
__t36.vendor = 4277u32;
__t36.device = 36944u32;
__t36.subvendor = 4521u32;
__t36.subdevice = 21300u32;
__t36.class = 0u32;
__t36.class_mask = 0u32;
__t36.driver_data = 68u64;
__t36
}, {
let mut __t37: pci_device_id = unsafe { core::mem::zeroed() };
__t37.vendor = 5332u32;
__t37.device = 1024u32;
__t37.subvendor = 4294967295u32;
__t37.subdevice = 4294967295u32;
__t37.class = 0u32;
__t37.class_mask = 0u32;
__t37.driver_data = 70u64;
__t37
}, {
let mut __t38: pci_device_id = unsafe { core::mem::zeroed() };
__t38.vendor = 5332u32;
__t38.device = 1026u32;
__t38.subvendor = 4294967295u32;
__t38.subdevice = 4294967295u32;
__t38.class = 0u32;
__t38.class_mask = 0u32;
__t38.driver_data = 69u64;
__t38
}, {
let mut __t39: pci_device_id = unsafe { core::mem::zeroed() };
__t39.vendor = 4277u32;
__t39.device = 36912u32;
__t39.subvendor = 4862u32;
__t39.subdevice = 273u32;
__t39.class = 0u32;
__t39.class_mask = 0u32;
__t39.driver_data = 47u64;
__t39
}, {
let mut __t40: pci_device_id = unsafe { core::mem::zeroed() };
__t40.vendor = 4277u32;
__t40.device = 36944u32;
__t40.subvendor = 4832u32;
__t40.subdevice = 49u32;
__t40.class = 0u32;
__t40.class_mask = 0u32;
__t40.driver_data = 50u64;
__t40
}, {
let mut __t41: pci_device_id = unsafe { core::mem::zeroed() };
__t41.vendor = 4277u32;
__t41.device = 36944u32;
__t41.subvendor = 4832u32;
__t41.subdevice = 33u32;
__t41.class = 0u32;
__t41.class_mask = 0u32;
__t41.driver_data = 51u64;
__t41
}, {
let mut __t42: pci_device_id = unsafe { core::mem::zeroed() };
__t42.vendor = 4277u32;
__t42.device = 36944u32;
__t42.subvendor = 4832u32;
__t42.subdevice = 17u32;
__t42.class = 0u32;
__t42.class_mask = 0u32;
__t42.driver_data = 52u64;
__t42
}, {
let mut __t43: pci_device_id = unsafe { core::mem::zeroed() };
__t43.vendor = 4277u32;
__t43.device = 36944u32;
__t43.subvendor = 4832u32;
__t43.subdevice = 65u32;
__t43.class = 0u32;
__t43.class_mask = 0u32;
__t43.driver_data = 52u64;
__t43
}, {
let mut __t44: pci_device_id = unsafe { core::mem::zeroed() };
__t44.vendor = 4277u32;
__t44.device = 36944u32;
__t44.subvendor = 4685u32;
__t44.subdevice = 61441u32;
__t44.class = 0u32;
__t44.class_mask = 0u32;
__t44.driver_data = 50u64;
__t44
}, {
let mut __t45: pci_device_id = unsafe { core::mem::zeroed() };
__t45.vendor = 4277u32;
__t45.device = 36944u32;
__t45.subvendor = 4685u32;
__t45.subdevice = 61456u32;
__t45.class = 0u32;
__t45.class_mask = 0u32;
__t45.driver_data = 51u64;
__t45
}, {
let mut __t46: pci_device_id = unsafe { core::mem::zeroed() };
__t46.vendor = 4277u32;
__t46.device = 36944u32;
__t46.subvendor = 55373u32;
__t46.subdevice = 16469u32;
__t46.class = 0u32;
__t46.class_mask = 0u32;
__t46.driver_data = 47u64;
__t46
}, {
let mut __t47: pci_device_id = unsafe { core::mem::zeroed() };
__t47.vendor = 4277u32;
__t47.device = 4202u32;
__t47.subvendor = 4277u32;
__t47.subdevice = 4202u32;
__t47.class = 0u32;
__t47.class_mask = 0u32;
__t47.driver_data = 71u64;
__t47
}, {
let mut __t48: pci_device_id = unsafe { core::mem::zeroed() };
__t48.vendor = 4956u32;
__t48.device = 16u32;
__t48.subvendor = 4294967295u32;
__t48.subdevice = 4294967295u32;
__t48.class = 0u32;
__t48.class_mask = 0u32;
__t48.driver_data = 29u64;
__t48
}, {
let mut __t49: pci_device_id = unsafe { core::mem::zeroed() };
__t49.vendor = 4956u32;
__t49.device = 32u32;
__t49.subvendor = 4294967295u32;
__t49.subdevice = 4294967295u32;
__t49.class = 0u32;
__t49.class_mask = 0u32;
__t49.driver_data = 28u64;
__t49
}, {
let mut __t50: pci_device_id = unsafe { core::mem::zeroed() };
__t50.vendor = 4956u32;
__t50.device = 385u32;
__t50.subvendor = 4294967295u32;
__t50.subdevice = 4294967295u32;
__t50.class = 0u32;
__t50.class_mask = 0u32;
__t50.driver_data = 46u64;
__t50
}, {
let mut __t51: pci_device_id = unsafe { core::mem::zeroed() };
__t51.vendor = 4956u32;
__t51.device = 48u32;
__t51.subvendor = 4294967295u32;
__t51.subdevice = 4294967295u32;
__t51.class = 0u32;
__t51.class_mask = 0u32;
__t51.driver_data = 28u64;
__t51
}, {
let mut __t52: pci_device_id = unsafe { core::mem::zeroed() };
__t52.vendor = 4956u32;
__t52.device = 433u32;
__t52.subvendor = 4294967295u32;
__t52.subdevice = 4294967295u32;
__t52.class = 0u32;
__t52.class_mask = 0u32;
__t52.driver_data = 46u64;
__t52
}, {
let mut __t53: pci_device_id = unsafe { core::mem::zeroed() };
__t53.vendor = 4956u32;
__t53.device = 64u32;
__t53.subvendor = 4294967295u32;
__t53.subdevice = 4294967295u32;
__t53.class = 0u32;
__t53.class_mask = 0u32;
__t53.driver_data = 29u64;
__t53
}, {
let mut __t54: pci_device_id = unsafe { core::mem::zeroed() };
__t54.vendor = 4956u32;
__t54.device = 80u32;
__t54.subvendor = 4294967295u32;
__t54.subdevice = 4294967295u32;
__t54.class = 0u32;
__t54.class_mask = 0u32;
__t54.driver_data = 30u64;
__t54
}, {
let mut __t55: pci_device_id = unsafe { core::mem::zeroed() };
__t55.vendor = 4956u32;
__t55.device = 96u32;
__t55.subvendor = 4294967295u32;
__t55.subdevice = 4294967295u32;
__t55.class = 0u32;
__t55.class_mask = 0u32;
__t55.driver_data = 30u64;
__t55
}, {
let mut __t56: pci_device_id = unsafe { core::mem::zeroed() };
__t56.vendor = 4956u32;
__t56.device = 288u32;
__t56.subvendor = 4294967295u32;
__t56.subdevice = 4294967295u32;
__t56.class = 0u32;
__t56.class_mask = 0u32;
__t56.driver_data = 29u64;
__t56
}, {
let mut __t57: pci_device_id = unsafe { core::mem::zeroed() };
__t57.vendor = 4956u32;
__t57.device = 304u32;
__t57.subvendor = 4294967295u32;
__t57.subdevice = 4294967295u32;
__t57.class = 0u32;
__t57.class_mask = 0u32;
__t57.driver_data = 28u64;
__t57
}, {
let mut __t58: pci_device_id = unsafe { core::mem::zeroed() };
__t58.vendor = 4956u32;
__t58.device = 320u32;
__t58.subvendor = 4294967295u32;
__t58.subdevice = 4294967295u32;
__t58.class = 0u32;
__t58.class_mask = 0u32;
__t58.driver_data = 29u64;
__t58
}, {
let mut __t59: pci_device_id = unsafe { core::mem::zeroed() };
__t59.vendor = 4956u32;
__t59.device = 336u32;
__t59.subvendor = 4294967295u32;
__t59.subdevice = 4294967295u32;
__t59.class = 0u32;
__t59.class_mask = 0u32;
__t59.driver_data = 28u64;
__t59
}, {
let mut __t60: pci_device_id = unsafe { core::mem::zeroed() };
__t60.vendor = 4956u32;
__t60.device = 368u32;
__t60.subvendor = 4294967295u32;
__t60.subdevice = 4294967295u32;
__t60.class = 0u32;
__t60.class_mask = 0u32;
__t60.driver_data = 47u64;
__t60
}, {
let mut __t61: pci_device_id = unsafe { core::mem::zeroed() };
__t61.vendor = 4956u32;
__t61.device = 384u32;
__t61.subvendor = 4294967295u32;
__t61.subdevice = 4294967295u32;
__t61.class = 0u32;
__t61.class_mask = 0u32;
__t61.driver_data = 46u64;
__t61
}, {
let mut __t62: pci_device_id = unsafe { core::mem::zeroed() };
__t62.vendor = 4956u32;
__t62.device = 400u32;
__t62.subvendor = 4294967295u32;
__t62.subdevice = 4294967295u32;
__t62.class = 0u32;
__t62.class_mask = 0u32;
__t62.driver_data = 45u64;
__t62
}, {
let mut __t63: pci_device_id = unsafe { core::mem::zeroed() };
__t63.vendor = 4956u32;
__t63.device = 416u32;
__t63.subvendor = 4294967295u32;
__t63.subdevice = 4294967295u32;
__t63.class = 0u32;
__t63.class_mask = 0u32;
__t63.driver_data = 47u64;
__t63
}, {
let mut __t64: pci_device_id = unsafe { core::mem::zeroed() };
__t64.vendor = 4956u32;
__t64.device = 432u32;
__t64.subvendor = 4294967295u32;
__t64.subdevice = 4294967295u32;
__t64.class = 0u32;
__t64.class_mask = 0u32;
__t64.driver_data = 46u64;
__t64
}, {
let mut __t65: pci_device_id = unsafe { core::mem::zeroed() };
__t65.vendor = 4956u32;
__t65.device = 448u32;
__t65.subvendor = 4294967295u32;
__t65.subdevice = 4294967295u32;
__t65.class = 0u32;
__t65.class_mask = 0u32;
__t65.driver_data = 45u64;
__t65
}, {
let mut __t66: pci_device_id = unsafe { core::mem::zeroed() };
__t66.vendor = 4956u32;
__t66.device = 480u32;
__t66.subvendor = 4294967295u32;
__t66.subdevice = 4294967295u32;
__t66.class = 0u32;
__t66.class_mask = 0u32;
__t66.driver_data = 5u64;
__t66
}, {
let mut __t67: pci_device_id = unsafe { core::mem::zeroed() };
__t67.vendor = 4555u32;
__t67.device = 38145u32;
__t67.subvendor = 4555u32;
__t67.subdevice = 40964u32;
__t67.class = 0u32;
__t67.class_mask = 0u32;
__t67.driver_data = 8u64;
__t67
}, {
let mut __t68: pci_device_id = unsafe { core::mem::zeroed() };
__t68.vendor = 5141u32;
__t68.device = 38145u32;
__t68.subvendor = 4895u32;
__t68.subdevice = 8272u32;
__t68.class = 0u32;
__t68.class_mask = 0u32;
__t68.driver_data = 11u64;
__t68
}, {
let mut __t69: pci_device_id = unsafe { core::mem::zeroed() };
__t69.vendor = 5141u32;
__t69.device = 38149u32;
__t69.subvendor = 4294967295u32;
__t69.subdevice = 4294967295u32;
__t69.class = 0u32;
__t69.class_mask = 0u32;
__t69.driver_data = 24u64;
__t69
}, {
let mut __t70: pci_device_id = unsafe { core::mem::zeroed() };
__t70.vendor = 5141u32;
__t70.device = 38154u32;
__t70.subvendor = 4895u32;
__t70.subdevice = 9472u32;
__t70.class = 0u32;
__t70.class_mask = 0u32;
__t70.driver_data = 2u64;
__t70
}, {
let mut __t71: pci_device_id = unsafe { core::mem::zeroed() };
__t71.vendor = 5141u32;
__t71.device = 38154u32;
__t71.subvendor = 4895u32;
__t71.subdevice = 9520u32;
__t71.class = 0u32;
__t71.class_mask = 0u32;
__t71.driver_data = 2u64;
__t71
}, {
let mut __t72: pci_device_id = unsafe { core::mem::zeroed() };
__t72.vendor = 5141u32;
__t72.device = 38154u32;
__t72.subvendor = 4294967295u32;
__t72.subdevice = 4294967295u32;
__t72.class = 0u32;
__t72.class_mask = 0u32;
__t72.driver_data = 10u64;
__t72
}, {
let mut __t73: pci_device_id = unsafe { core::mem::zeroed() };
__t73.vendor = 5141u32;
__t73.device = 38155u32;
__t73.subvendor = 5141u32;
__t73.subdevice = 1u32;
__t73.class = 0u32;
__t73.class_mask = 0u32;
__t73.driver_data = 6u64;
__t73
}, {
let mut __t74: pci_device_id = unsafe { core::mem::zeroed() };
__t74.vendor = 5141u32;
__t74.device = 38145u32;
__t74.subvendor = 4294967295u32;
__t74.subdevice = 4294967295u32;
__t74.class = 0u32;
__t74.class_mask = 0u32;
__t74.driver_data = 3u64;
__t74
}, {
let mut __t75: pci_device_id = unsafe { core::mem::zeroed() };
__t75.vendor = 5141u32;
__t75.device = 38177u32;
__t75.subvendor = 4294967295u32;
__t75.subdevice = 4294967295u32;
__t75.class = 0u32;
__t75.class_mask = 0u32;
__t75.driver_data = 24u64;
__t75
}, {
let mut __t76: pci_device_id = unsafe { core::mem::zeroed() };
__t76.vendor = 5141u32;
__t76.device = 38200u32;
__t76.subvendor = 4294967295u32;
__t76.subdevice = 4294967295u32;
__t76.class = 0u32;
__t76.class_mask = 0u32;
__t76.driver_data = 56u64;
__t76
}, {
let mut __t77: pci_device_id = unsafe { core::mem::zeroed() };
__t77.vendor = 5141u32;
__t77.device = 49409u32;
__t77.subvendor = 4294967295u32;
__t77.subdevice = 4294967295u32;
__t77.class = 0u32;
__t77.class_mask = 0u32;
__t77.driver_data = 15u64;
__t77
}, {
let mut __t78: pci_device_id = unsafe { core::mem::zeroed() };
__t78.vendor = 5141u32;
__t78.device = 49413u32;
__t78.subvendor = 4294967295u32;
__t78.subdevice = 4294967295u32;
__t78.class = 0u32;
__t78.class_mask = 0u32;
__t78.driver_data = 15u64;
__t78
}, {
let mut __t79: pci_device_id = unsafe { core::mem::zeroed() };
__t79.vendor = 5141u32;
__t79.device = 49435u32;
__t79.subvendor = 4294967295u32;
__t79.subdevice = 4294967295u32;
__t79.class = 0u32;
__t79.class_mask = 0u32;
__t79.driver_data = 73u64;
__t79
}, {
let mut __t80: pci_device_id = unsafe { core::mem::zeroed() };
__t80.vendor = 5141u32;
__t80.device = 49439u32;
__t80.subvendor = 4294967295u32;
__t80.subdevice = 4294967295u32;
__t80.class = 0u32;
__t80.class_mask = 0u32;
__t80.driver_data = 73u64;
__t80
}, {
let mut __t81: pci_device_id = unsafe { core::mem::zeroed() };
__t81.vendor = 5141u32;
__t81.device = 49440u32;
__t81.subvendor = 4294967295u32;
__t81.subdevice = 4294967295u32;
__t81.class = 0u32;
__t81.class_mask = 0u32;
__t81.driver_data = 15u64;
__t81
}, {
let mut __t82: pci_device_id = unsafe { core::mem::zeroed() };
__t82.vendor = 5141u32;
__t82.device = 49444u32;
__t82.subvendor = 4294967295u32;
__t82.subdevice = 4294967295u32;
__t82.class = 0u32;
__t82.class_mask = 0u32;
__t82.driver_data = 15u64;
__t82
}, {
let mut __t83: pci_device_id = unsafe { core::mem::zeroed() };
__t83.vendor = 5141u32;
__t83.device = 49464u32;
__t83.subvendor = 4294967295u32;
__t83.subdevice = 4294967295u32;
__t83.class = 0u32;
__t83.class_mask = 0u32;
__t83.driver_data = 73u64;
__t83
}, {
let mut __t84: pci_device_id = unsafe { core::mem::zeroed() };
__t84.vendor = 5141u32;
__t84.device = 49469u32;
__t84.subvendor = 4294967295u32;
__t84.subdevice = 4294967295u32;
__t84.class = 0u32;
__t84.class_mask = 0u32;
__t84.driver_data = 73u64;
__t84
}, {
let mut __t85: pci_device_id = unsafe { core::mem::zeroed() };
__t85.vendor = 5141u32;
__t85.device = 49472u32;
__t85.subvendor = 4294967295u32;
__t85.subdevice = 4294967295u32;
__t85.class = 0u32;
__t85.class_mask = 0u32;
__t85.driver_data = 15u64;
__t85
}, {
let mut __t86: pci_device_id = unsafe { core::mem::zeroed() };
__t86.vendor = 5141u32;
__t86.device = 49473u32;
__t86.subvendor = 4294967295u32;
__t86.subdevice = 4294967295u32;
__t86.class = 0u32;
__t86.class_mask = 0u32;
__t86.driver_data = 15u64;
__t86
}, {
let mut __t87: pci_device_id = unsafe { core::mem::zeroed() };
__t87.vendor = 5141u32;
__t87.device = 49476u32;
__t87.subvendor = 4294967295u32;
__t87.subdevice = 4294967295u32;
__t87.class = 0u32;
__t87.class_mask = 0u32;
__t87.driver_data = 15u64;
__t87
}, {
let mut __t88: pci_device_id = unsafe { core::mem::zeroed() };
__t88.vendor = 5141u32;
__t88.device = 49477u32;
__t88.subvendor = 4294967295u32;
__t88.subdevice = 4294967295u32;
__t88.class = 0u32;
__t88.class_mask = 0u32;
__t88.driver_data = 15u64;
__t88
}, {
let mut __t89: pci_device_id = unsafe { core::mem::zeroed() };
__t89.vendor = 5141u32;
__t89.device = 49496u32;
__t89.subvendor = 4294967295u32;
__t89.subdevice = 4294967295u32;
__t89.class = 0u32;
__t89.class_mask = 0u32;
__t89.driver_data = 74u64;
__t89
}, {
let mut __t90: pci_device_id = unsafe { core::mem::zeroed() };
__t90.vendor = 5141u32;
__t90.device = 49501u32;
__t90.subvendor = 4294967295u32;
__t90.subdevice = 4294967295u32;
__t90.class = 0u32;
__t90.class_mask = 0u32;
__t90.driver_data = 74u64;
__t90
}, {
let mut __t91: pci_device_id = unsafe { core::mem::zeroed() };
__t91.vendor = 5141u32;
__t91.device = 49672u32;
__t91.subvendor = 4294967295u32;
__t91.subdevice = 4294967295u32;
__t91.class = 0u32;
__t91.class_mask = 0u32;
__t91.driver_data = 75u64;
__t91
}, {
let mut __t92: pci_device_id = unsafe { core::mem::zeroed() };
__t92.vendor = 5141u32;
__t92.device = 49677u32;
__t92.subvendor = 4294967295u32;
__t92.subdevice = 4294967295u32;
__t92.class = 0u32;
__t92.class_mask = 0u32;
__t92.driver_data = 75u64;
__t92
}, {
let mut __t93: pci_device_id = unsafe { core::mem::zeroed() };
__t93.vendor = 5141u32;
__t93.device = 49928u32;
__t93.subvendor = 4294967295u32;
__t93.subdevice = 4294967295u32;
__t93.class = 0u32;
__t93.class_mask = 0u32;
__t93.driver_data = 76u64;
__t93
}, {
let mut __t94: pci_device_id = unsafe { core::mem::zeroed() };
__t94.vendor = 5141u32;
__t94.device = 49933u32;
__t94.subvendor = 4294967295u32;
__t94.subdevice = 4294967295u32;
__t94.class = 0u32;
__t94.class_mask = 0u32;
__t94.driver_data = 76u64;
__t94
}, {
let mut __t95: pci_device_id = unsafe { core::mem::zeroed() };
__t95.vendor = 5141u32;
__t95.device = 50187u32;
__t95.subvendor = 4294967295u32;
__t95.subdevice = 4294967295u32;
__t95.class = 0u32;
__t95.class_mask = 0u32;
__t95.driver_data = 73u64;
__t95
}, {
let mut __t96: pci_device_id = unsafe { core::mem::zeroed() };
__t96.vendor = 5141u32;
__t96.device = 50191u32;
__t96.subvendor = 4294967295u32;
__t96.subdevice = 4294967295u32;
__t96.class = 0u32;
__t96.class_mask = 0u32;
__t96.driver_data = 73u64;
__t96
}, {
let mut __t97: pci_device_id = unsafe { core::mem::zeroed() };
__t97.vendor = 5141u32;
__t97.device = 50203u32;
__t97.subvendor = 4294967295u32;
__t97.subdevice = 4294967295u32;
__t97.class = 0u32;
__t97.class_mask = 0u32;
__t97.driver_data = 73u64;
__t97
}, {
let mut __t98: pci_device_id = unsafe { core::mem::zeroed() };
__t98.vendor = 5141u32;
__t98.device = 50207u32;
__t98.subvendor = 4294967295u32;
__t98.subdevice = 4294967295u32;
__t98.class = 0u32;
__t98.class_mask = 0u32;
__t98.driver_data = 73u64;
__t98
}, {
let mut __t99: pci_device_id = unsafe { core::mem::zeroed() };
__t99.vendor = 5141u32;
__t99.device = 50219u32;
__t99.subvendor = 4294967295u32;
__t99.subdevice = 4294967295u32;
__t99.class = 0u32;
__t99.class_mask = 0u32;
__t99.driver_data = 73u64;
__t99
}, {
let mut __t100: pci_device_id = unsafe { core::mem::zeroed() };
__t100.vendor = 5141u32;
__t100.device = 50223u32;
__t100.subvendor = 4294967295u32;
__t100.subdevice = 4294967295u32;
__t100.class = 0u32;
__t100.class_mask = 0u32;
__t100.driver_data = 73u64;
__t100
}, {
let mut __t101: pci_device_id = unsafe { core::mem::zeroed() };
__t101.vendor = 5141u32;
__t101.device = 50235u32;
__t101.subvendor = 4294967295u32;
__t101.subdevice = 4294967295u32;
__t101.class = 0u32;
__t101.class_mask = 0u32;
__t101.driver_data = 73u64;
__t101
}, {
let mut __t102: pci_device_id = unsafe { core::mem::zeroed() };
__t102.vendor = 5141u32;
__t102.device = 50239u32;
__t102.subvendor = 4294967295u32;
__t102.subdevice = 4294967295u32;
__t102.class = 0u32;
__t102.class_mask = 0u32;
__t102.driver_data = 73u64;
__t102
}, {
let mut __t103: pci_device_id = unsafe { core::mem::zeroed() };
__t103.vendor = 5141u32;
__t103.device = 50251u32;
__t103.subvendor = 4294967295u32;
__t103.subdevice = 4294967295u32;
__t103.class = 0u32;
__t103.class_mask = 0u32;
__t103.driver_data = 73u64;
__t103
}, {
let mut __t104: pci_device_id = unsafe { core::mem::zeroed() };
__t104.vendor = 5141u32;
__t104.device = 50255u32;
__t104.subvendor = 4294967295u32;
__t104.subdevice = 4294967295u32;
__t104.class = 0u32;
__t104.class_mask = 0u32;
__t104.driver_data = 73u64;
__t104
}, {
let mut __t105: pci_device_id = unsafe { core::mem::zeroed() };
__t105.vendor = 5141u32;
__t105.device = 50267u32;
__t105.subvendor = 4294967295u32;
__t105.subdevice = 4294967295u32;
__t105.class = 0u32;
__t105.class_mask = 0u32;
__t105.driver_data = 73u64;
__t105
}, {
let mut __t106: pci_device_id = unsafe { core::mem::zeroed() };
__t106.vendor = 5141u32;
__t106.device = 50271u32;
__t106.subvendor = 4294967295u32;
__t106.subdevice = 4294967295u32;
__t106.class = 0u32;
__t106.class_mask = 0u32;
__t106.driver_data = 73u64;
__t106
}, {
let mut __t107: pci_device_id = unsafe { core::mem::zeroed() };
__t107.vendor = 5141u32;
__t107.device = 50283u32;
__t107.subvendor = 4294967295u32;
__t107.subdevice = 4294967295u32;
__t107.class = 0u32;
__t107.class_mask = 0u32;
__t107.driver_data = 73u64;
__t107
}, {
let mut __t108: pci_device_id = unsafe { core::mem::zeroed() };
__t108.vendor = 5141u32;
__t108.device = 50287u32;
__t108.subvendor = 4294967295u32;
__t108.subdevice = 4294967295u32;
__t108.class = 0u32;
__t108.class_mask = 0u32;
__t108.driver_data = 73u64;
__t108
}, {
let mut __t109: pci_device_id = unsafe { core::mem::zeroed() };
__t109.vendor = 5141u32;
__t109.device = 50299u32;
__t109.subvendor = 4294967295u32;
__t109.subdevice = 4294967295u32;
__t109.class = 0u32;
__t109.class_mask = 0u32;
__t109.driver_data = 73u64;
__t109
}, {
let mut __t110: pci_device_id = unsafe { core::mem::zeroed() };
__t110.vendor = 5141u32;
__t110.device = 50303u32;
__t110.subvendor = 4294967295u32;
__t110.subdevice = 4294967295u32;
__t110.class = 0u32;
__t110.class_mask = 0u32;
__t110.driver_data = 73u64;
__t110
}, {
let mut __t111: pci_device_id = unsafe { core::mem::zeroed() };
__t111.vendor = 5141u32;
__t111.device = 50315u32;
__t111.subvendor = 4294967295u32;
__t111.subdevice = 4294967295u32;
__t111.class = 0u32;
__t111.class_mask = 0u32;
__t111.driver_data = 73u64;
__t111
}, {
let mut __t112: pci_device_id = unsafe { core::mem::zeroed() };
__t112.vendor = 5141u32;
__t112.device = 50319u32;
__t112.subvendor = 4294967295u32;
__t112.subdevice = 4294967295u32;
__t112.class = 0u32;
__t112.class_mask = 0u32;
__t112.driver_data = 73u64;
__t112
}, {
let mut __t113: pci_device_id = unsafe { core::mem::zeroed() };
__t113.vendor = 5141u32;
__t113.device = 50331u32;
__t113.subvendor = 4294967295u32;
__t113.subdevice = 4294967295u32;
__t113.class = 0u32;
__t113.class_mask = 0u32;
__t113.driver_data = 73u64;
__t113
}, {
let mut __t114: pci_device_id = unsafe { core::mem::zeroed() };
__t114.vendor = 5141u32;
__t114.device = 50335u32;
__t114.subvendor = 4294967295u32;
__t114.subdevice = 4294967295u32;
__t114.class = 0u32;
__t114.class_mask = 0u32;
__t114.driver_data = 73u64;
__t114
}, {
let mut __t115: pci_device_id = unsafe { core::mem::zeroed() };
__t115.vendor = 5141u32;
__t115.device = 50347u32;
__t115.subvendor = 4294967295u32;
__t115.subdevice = 4294967295u32;
__t115.class = 0u32;
__t115.class_mask = 0u32;
__t115.driver_data = 73u64;
__t115
}, {
let mut __t116: pci_device_id = unsafe { core::mem::zeroed() };
__t116.vendor = 5141u32;
__t116.device = 50351u32;
__t116.subvendor = 4294967295u32;
__t116.subdevice = 4294967295u32;
__t116.class = 0u32;
__t116.class_mask = 0u32;
__t116.driver_data = 73u64;
__t116
}, {
let mut __t117: pci_device_id = unsafe { core::mem::zeroed() };
__t117.vendor = 5141u32;
__t117.device = 50363u32;
__t117.subvendor = 4294967295u32;
__t117.subdevice = 4294967295u32;
__t117.class = 0u32;
__t117.class_mask = 0u32;
__t117.driver_data = 73u64;
__t117
}, {
let mut __t118: pci_device_id = unsafe { core::mem::zeroed() };
__t118.vendor = 5141u32;
__t118.device = 50367u32;
__t118.subvendor = 4294967295u32;
__t118.subdevice = 4294967295u32;
__t118.class = 0u32;
__t118.class_mask = 0u32;
__t118.driver_data = 73u64;
__t118
}, {
let mut __t119: pci_device_id = unsafe { core::mem::zeroed() };
__t119.vendor = 5141u32;
__t119.device = 50379u32;
__t119.subvendor = 4294967295u32;
__t119.subdevice = 4294967295u32;
__t119.class = 0u32;
__t119.class_mask = 0u32;
__t119.driver_data = 73u64;
__t119
}, {
let mut __t120: pci_device_id = unsafe { core::mem::zeroed() };
__t120.vendor = 5141u32;
__t120.device = 50383u32;
__t120.subvendor = 4294967295u32;
__t120.subdevice = 4294967295u32;
__t120.class = 0u32;
__t120.class_mask = 0u32;
__t120.driver_data = 73u64;
__t120
}, {
let mut __t121: pci_device_id = unsafe { core::mem::zeroed() };
__t121.vendor = 5410u32;
__t121.device = 16384u32;
__t121.subvendor = 5410u32;
__t121.subdevice = 16385u32;
__t121.class = 0u32;
__t121.class_mask = 0u32;
__t121.driver_data = 73u64;
__t121
}, {
let mut __t122: pci_device_id = unsafe { core::mem::zeroed() };
__t122.vendor = 5410u32;
__t122.device = 16384u32;
__t122.subvendor = 5410u32;
__t122.subdevice = 16386u32;
__t122.class = 0u32;
__t122.class_mask = 0u32;
__t122.driver_data = 74u64;
__t122
}, {
let mut __t123: pci_device_id = unsafe { core::mem::zeroed() };
__t123.vendor = 5410u32;
__t123.device = 16384u32;
__t123.subvendor = 5410u32;
__t123.subdevice = 16388u32;
__t123.class = 0u32;
__t123.class_mask = 0u32;
__t123.driver_data = 75u64;
__t123
}, {
let mut __t124: pci_device_id = unsafe { core::mem::zeroed() };
__t124.vendor = 5410u32;
__t124.device = 16384u32;
__t124.subvendor = 5410u32;
__t124.subdevice = 16392u32;
__t124.class = 0u32;
__t124.class_mask = 0u32;
__t124.driver_data = 76u64;
__t124
}, {
let mut __t125: pci_device_id = unsafe { core::mem::zeroed() };
__t125.vendor = 4431u32;
__t125.device = 246u32;
__t125.subvendor = 4116u32;
__t125.subdevice = 4294967295u32;
__t125.class = 0u32;
__t125.class_mask = 0u32;
__t125.driver_data = 74u64;
__t125
}, {
let mut __t126: pci_device_id = unsafe { core::mem::zeroed() };
__t126.vendor = 29697u32;
__t126.device = 57600u32;
__t126.subvendor = 4294967295u32;
__t126.subdevice = 4294967295u32;
__t126.class = 0u32;
__t126.class_mask = 0u32;
__t126.driver_data = 74u64;
__t126
}, {
let mut __t127: pci_device_id = unsafe { core::mem::zeroed() };
__t127.vendor = 4683u32;
__t127.device = 1u32;
__t127.subvendor = 4683u32;
__t127.subdevice = 264u32;
__t127.class = 0u32;
__t127.class_mask = 0u32;
__t127.driver_data = 82u64;
__t127
}, {
let mut __t128: pci_device_id = unsafe { core::mem::zeroed() };
__t128.vendor = 4683u32;
__t128.device = 1u32;
__t128.subvendor = 4683u32;
__t128.subdevice = 520u32;
__t128.class = 0u32;
__t128.class_mask = 0u32;
__t128.driver_data = 82u64;
__t128
}, {
let mut __t129: pci_device_id = unsafe { core::mem::zeroed() };
__t129.vendor = 4683u32;
__t129.device = 1u32;
__t129.subvendor = 4683u32;
__t129.subdevice = 776u32;
__t129.class = 0u32;
__t129.class_mask = 0u32;
__t129.driver_data = 82u64;
__t129
}, {
let mut __t130: pci_device_id = unsafe { core::mem::zeroed() };
__t130.vendor = 4683u32;
__t130.device = 1u32;
__t130.subvendor = 4683u32;
__t130.subdevice = 1032u32;
__t130.class = 0u32;
__t130.class_mask = 0u32;
__t130.driver_data = 82u64;
__t130
}, {
let mut __t131: pci_device_id = unsafe { core::mem::zeroed() };
__t131.vendor = 4545u32;
__t131.device = 1152u32;
__t131.subvendor = 4294967295u32;
__t131.subdevice = 4294967295u32;
__t131.class = 0u32;
__t131.class_mask = 0u32;
__t131.driver_data = 27u64;
__t131
}, {
let mut __t132: pci_device_id = unsafe { core::mem::zeroed() };
__t132.vendor = 5330u32;
__t132.device = 40961u32;
__t132.subvendor = 4294967295u32;
__t132.subdevice = 4294967295u32;
__t132.class = 0u32;
__t132.class_mask = 0u32;
__t132.driver_data = 6u64;
__t132
}, {
let mut __t133: pci_device_id = unsafe { core::mem::zeroed() };
__t133.vendor = 5330u32;
__t133.device = 40965u32;
__t133.subvendor = 4294967295u32;
__t133.subdevice = 4294967295u32;
__t133.class = 0u32;
__t133.class_mask = 0u32;
__t133.driver_data = 7u64;
__t133
}, {
let mut __t134: pci_device_id = unsafe { core::mem::zeroed() };
__t134.vendor = 5330u32;
__t134.device = 40963u32;
__t134.subvendor = 4294967295u32;
__t134.subdevice = 4294967295u32;
__t134.class = 0u32;
__t134.class_mask = 0u32;
__t134.driver_data = 8u64;
__t134
}, {
let mut __t135: pci_device_id = unsafe { core::mem::zeroed() };
__t135.vendor = 5330u32;
__t135.device = 40964u32;
__t135.subvendor = 4294967295u32;
__t135.subdevice = 4294967295u32;
__t135.class = 0u32;
__t135.class_mask = 0u32;
__t135.driver_data = 8u64;
__t135
}, {
let mut __t136: pci_device_id = unsafe { core::mem::zeroed() };
__t136.vendor = 5330u32;
__t136.device = 32784u32;
__t136.subvendor = 4294967295u32;
__t136.subdevice = 4294967295u32;
__t136.class = 0u32;
__t136.class_mask = 0u32;
__t136.driver_data = 32u64;
__t136
}, {
let mut __t137: pci_device_id = unsafe { core::mem::zeroed() };
__t137.vendor = 5330u32;
__t137.device = 32800u32;
__t137.subvendor = 4294967295u32;
__t137.subdevice = 4294967295u32;
__t137.class = 0u32;
__t137.class_mask = 0u32;
__t137.driver_data = 40u64;
__t137
}, {
let mut __t138: pci_device_id = unsafe { core::mem::zeroed() };
__t138.vendor = 5330u32;
__t138.device = 32832u32;
__t138.subvendor = 4294967295u32;
__t138.subdevice = 4294967295u32;
__t138.class = 0u32;
__t138.class_mask = 0u32;
__t138.driver_data = 25u64;
__t138
}, {
let mut __t139: pci_device_id = unsafe { core::mem::zeroed() };
__t139.vendor = 5330u32;
__t139.device = 32896u32;
__t139.subvendor = 4294967295u32;
__t139.subdevice = 4294967295u32;
__t139.class = 0u32;
__t139.class_mask = 0u32;
__t139.driver_data = 26u64;
__t139
}, {
let mut __t140: pci_device_id = unsafe { core::mem::zeroed() };
__t140.vendor = 5330u32;
__t140.device = 32808u32;
__t140.subvendor = 4294967295u32;
__t140.subdevice = 4294967295u32;
__t140.class = 0u32;
__t140.class_mask = 0u32;
__t140.driver_data = 65u64;
__t140
}, {
let mut __t141: pci_device_id = unsafe { core::mem::zeroed() };
__t141.vendor = 5330u32;
__t141.device = 32840u32;
__t141.subvendor = 4294967295u32;
__t141.subdevice = 4294967295u32;
__t141.class = 0u32;
__t141.class_mask = 0u32;
__t141.driver_data = 66u64;
__t141
}, {
let mut __t142: pci_device_id = unsafe { core::mem::zeroed() };
__t142.vendor = 5330u32;
__t142.device = 32904u32;
__t142.subvendor = 4294967295u32;
__t142.subdevice = 4294967295u32;
__t142.class = 0u32;
__t142.class_mask = 0u32;
__t142.driver_data = 67u64;
__t142
}, {
let mut __t143: pci_device_id = unsafe { core::mem::zeroed() };
__t143.vendor = 5330u32;
__t143.device = 40969u32;
__t143.subvendor = 4294967295u32;
__t143.subdevice = 4294967295u32;
__t143.class = 0u32;
__t143.class_mask = 0u32;
__t143.driver_data = 8u64;
__t143
}, {
let mut __t144: pci_device_id = unsafe { core::mem::zeroed() };
__t144.vendor = 5330u32;
__t144.device = 40967u32;
__t144.subvendor = 4294967295u32;
__t144.subdevice = 4294967295u32;
__t144.class = 0u32;
__t144.class_mask = 0u32;
__t144.driver_data = 8u64;
__t144
}, {
let mut __t145: pci_device_id = unsafe { core::mem::zeroed() };
__t145.vendor = 5330u32;
__t145.device = 40968u32;
__t145.subvendor = 4294967295u32;
__t145.subdevice = 4294967295u32;
__t145.class = 0u32;
__t145.class_mask = 0u32;
__t145.driver_data = 8u64;
__t145
}, {
let mut __t146: pci_device_id = unsafe { core::mem::zeroed() };
__t146.vendor = 5330u32;
__t146.device = 40976u32;
__t146.subvendor = 4294967295u32;
__t146.subdevice = 4294967295u32;
__t146.class = 0u32;
__t146.class_mask = 0u32;
__t146.driver_data = 110u64;
__t146
}, {
let mut __t147: pci_device_id = unsafe { core::mem::zeroed() };
__t147.vendor = 5330u32;
__t147.device = 40978u32;
__t147.subvendor = 4294967295u32;
__t147.subdevice = 4294967295u32;
__t147.class = 0u32;
__t147.class_mask = 0u32;
__t147.driver_data = 111u64;
__t147
}, {
let mut __t148: pci_device_id = unsafe { core::mem::zeroed() };
__t148.vendor = 5330u32;
__t148.device = 40979u32;
__t148.subvendor = 4294967295u32;
__t148.subdevice = 4294967295u32;
__t148.class = 0u32;
__t148.class_mask = 0u32;
__t148.driver_data = 112u64;
__t148
}, {
let mut __t149: pci_device_id = unsafe { core::mem::zeroed() };
__t149.vendor = 5330u32;
__t149.device = 40980u32;
__t149.subvendor = 4294967295u32;
__t149.subdevice = 4294967295u32;
__t149.class = 0u32;
__t149.class_mask = 0u32;
__t149.driver_data = 113u64;
__t149
}, {
let mut __t150: pci_device_id = unsafe { core::mem::zeroed() };
__t150.vendor = 5330u32;
__t150.device = 40982u32;
__t150.subvendor = 4294967295u32;
__t150.subdevice = 4294967295u32;
__t150.class = 0u32;
__t150.class_mask = 0u32;
__t150.driver_data = 111u64;
__t150
}, {
let mut __t151: pci_device_id = unsafe { core::mem::zeroed() };
__t151.vendor = 5330u32;
__t151.device = 40983u32;
__t151.subvendor = 4294967295u32;
__t151.subdevice = 4294967295u32;
__t151.class = 0u32;
__t151.class_mask = 0u32;
__t151.driver_data = 111u64;
__t151
}, {
let mut __t152: pci_device_id = unsafe { core::mem::zeroed() };
__t152.vendor = 5330u32;
__t152.device = 41734u32;
__t152.subvendor = 4294967295u32;
__t152.subdevice = 4294967295u32;
__t152.class = 0u32;
__t152.class_mask = 0u32;
__t152.driver_data = 24u64;
__t152
}, {
let mut __t153: pci_device_id = unsafe { core::mem::zeroed() };
__t153.vendor = 5330u32;
__t153.device = 41744u32;
__t153.subvendor = 4294967295u32;
__t153.subdevice = 4294967295u32;
__t153.class = 0u32;
__t153.class_mask = 0u32;
__t153.driver_data = 8u64;
__t153
}, {
let mut __t154: pci_device_id = unsafe { core::mem::zeroed() };
__t154.vendor = 5330u32;
__t154.device = 41746u32;
__t154.subvendor = 4294967295u32;
__t154.subdevice = 4294967295u32;
__t154.class = 0u32;
__t154.class_mask = 0u32;
__t154.driver_data = 8u64;
__t154
}, {
let mut __t155: pci_device_id = unsafe { core::mem::zeroed() };
__t155.vendor = 5330u32;
__t155.device = 41748u32;
__t155.subvendor = 4294967295u32;
__t155.subdevice = 4294967295u32;
__t155.class = 0u32;
__t155.class_mask = 0u32;
__t155.driver_data = 8u64;
__t155
}, {
let mut __t156: pci_device_id = unsafe { core::mem::zeroed() };
__t156.vendor = 5330u32;
__t156.device = 41749u32;
__t156.subvendor = 4294967295u32;
__t156.subdevice = 4294967295u32;
__t156.class = 0u32;
__t156.class_mask = 0u32;
__t156.driver_data = 8u64;
__t156
}, {
let mut __t157: pci_device_id = unsafe { core::mem::zeroed() };
__t157.vendor = 4895u32;
__t157.device = 4096u32;
__t157.subvendor = 4294967295u32;
__t157.subdevice = 4294967295u32;
__t157.class = 0u32;
__t157.class_mask = 0u32;
__t157.driver_data = 49u64;
__t157
}, {
let mut __t158: pci_device_id = unsafe { core::mem::zeroed() };
__t158.vendor = 4895u32;
__t158.device = 4097u32;
__t158.subvendor = 4294967295u32;
__t158.subdevice = 4294967295u32;
__t158.class = 0u32;
__t158.class_mask = 0u32;
__t158.driver_data = 49u64;
__t158
}, {
let mut __t159: pci_device_id = unsafe { core::mem::zeroed() };
__t159.vendor = 4895u32;
__t159.device = 4098u32;
__t159.subvendor = 4294967295u32;
__t159.subdevice = 4294967295u32;
__t159.class = 0u32;
__t159.class_mask = 0u32;
__t159.driver_data = 49u64;
__t159
}, {
let mut __t160: pci_device_id = unsafe { core::mem::zeroed() };
__t160.vendor = 4895u32;
__t160.device = 4144u32;
__t160.subvendor = 4294967295u32;
__t160.subdevice = 4294967295u32;
__t160.class = 0u32;
__t160.class_mask = 0u32;
__t160.driver_data = 60u64;
__t160
}, {
let mut __t161: pci_device_id = unsafe { core::mem::zeroed() };
__t161.vendor = 4895u32;
__t161.device = 4145u32;
__t161.subvendor = 4294967295u32;
__t161.subdevice = 4294967295u32;
__t161.class = 0u32;
__t161.class_mask = 0u32;
__t161.driver_data = 60u64;
__t161
}, {
let mut __t162: pci_device_id = unsafe { core::mem::zeroed() };
__t162.vendor = 4895u32;
__t162.device = 4146u32;
__t162.subvendor = 4294967295u32;
__t162.subdevice = 4294967295u32;
__t162.class = 0u32;
__t162.class_mask = 0u32;
__t162.driver_data = 60u64;
__t162
}, {
let mut __t163: pci_device_id = unsafe { core::mem::zeroed() };
__t163.vendor = 4895u32;
__t163.device = 4176u32;
__t163.subvendor = 4294967295u32;
__t163.subdevice = 4294967295u32;
__t163.class = 0u32;
__t163.class_mask = 0u32;
__t163.driver_data = 61u64;
__t163
}, {
let mut __t164: pci_device_id = unsafe { core::mem::zeroed() };
__t164.vendor = 4895u32;
__t164.device = 4177u32;
__t164.subvendor = 4294967295u32;
__t164.subdevice = 4294967295u32;
__t164.class = 0u32;
__t164.class_mask = 0u32;
__t164.driver_data = 61u64;
__t164
}, {
let mut __t165: pci_device_id = unsafe { core::mem::zeroed() };
__t165.vendor = 4895u32;
__t165.device = 4178u32;
__t165.subvendor = 4294967295u32;
__t165.subdevice = 4294967295u32;
__t165.class = 0u32;
__t165.class_mask = 0u32;
__t165.driver_data = 61u64;
__t165
}, {
let mut __t166: pci_device_id = unsafe { core::mem::zeroed() };
__t166.vendor = 4895u32;
__t166.device = 8192u32;
__t166.subvendor = 4294967295u32;
__t166.subdevice = 4294967295u32;
__t166.class = 0u32;
__t166.class_mask = 0u32;
__t166.driver_data = 6u64;
__t166
}, {
let mut __t167: pci_device_id = unsafe { core::mem::zeroed() };
__t167.vendor = 4895u32;
__t167.device = 8193u32;
__t167.subvendor = 4294967295u32;
__t167.subdevice = 4294967295u32;
__t167.class = 0u32;
__t167.class_mask = 0u32;
__t167.driver_data = 6u64;
__t167
}, {
let mut __t168: pci_device_id = unsafe { core::mem::zeroed() };
__t168.vendor = 4895u32;
__t168.device = 8194u32;
__t168.subvendor = 4294967295u32;
__t168.subdevice = 4294967295u32;
__t168.class = 0u32;
__t168.class_mask = 0u32;
__t168.driver_data = 6u64;
__t168
}, {
let mut __t169: pci_device_id = unsafe { core::mem::zeroed() };
__t169.vendor = 4895u32;
__t169.device = 8240u32;
__t169.subvendor = 4294967295u32;
__t169.subdevice = 4294967295u32;
__t169.class = 0u32;
__t169.class_mask = 0u32;
__t169.driver_data = 24u64;
__t169
}, {
let mut __t170: pci_device_id = unsafe { core::mem::zeroed() };
__t170.vendor = 4895u32;
__t170.device = 8241u32;
__t170.subvendor = 4294967295u32;
__t170.subdevice = 4294967295u32;
__t170.class = 0u32;
__t170.class_mask = 0u32;
__t170.driver_data = 24u64;
__t170
}, {
let mut __t171: pci_device_id = unsafe { core::mem::zeroed() };
__t171.vendor = 4895u32;
__t171.device = 8242u32;
__t171.subvendor = 4294967295u32;
__t171.subdevice = 4294967295u32;
__t171.class = 0u32;
__t171.class_mask = 0u32;
__t171.driver_data = 24u64;
__t171
}, {
let mut __t172: pci_device_id = unsafe { core::mem::zeroed() };
__t172.vendor = 4895u32;
__t172.device = 8272u32;
__t172.subvendor = 4294967295u32;
__t172.subdevice = 4294967295u32;
__t172.class = 0u32;
__t172.class_mask = 0u32;
__t172.driver_data = 25u64;
__t172
}, {
let mut __t173: pci_device_id = unsafe { core::mem::zeroed() };
__t173.vendor = 4895u32;
__t173.device = 8273u32;
__t173.subvendor = 4294967295u32;
__t173.subdevice = 4294967295u32;
__t173.class = 0u32;
__t173.class_mask = 0u32;
__t173.driver_data = 25u64;
__t173
}, {
let mut __t174: pci_device_id = unsafe { core::mem::zeroed() };
__t174.vendor = 4895u32;
__t174.device = 8274u32;
__t174.subvendor = 4294967295u32;
__t174.subdevice = 4294967295u32;
__t174.class = 0u32;
__t174.class_mask = 0u32;
__t174.driver_data = 25u64;
__t174
}, {
let mut __t175: pci_device_id = unsafe { core::mem::zeroed() };
__t175.vendor = 4895u32;
__t175.device = 8320u32;
__t175.subvendor = 4294967295u32;
__t175.subdevice = 4294967295u32;
__t175.class = 0u32;
__t175.class_mask = 0u32;
__t175.driver_data = 26u64;
__t175
}, {
let mut __t176: pci_device_id = unsafe { core::mem::zeroed() };
__t176.vendor = 4895u32;
__t176.device = 8321u32;
__t176.subvendor = 4294967295u32;
__t176.subdevice = 4294967295u32;
__t176.class = 0u32;
__t176.class_mask = 0u32;
__t176.driver_data = 26u64;
__t176
}, {
let mut __t177: pci_device_id = unsafe { core::mem::zeroed() };
__t177.vendor = 4895u32;
__t177.device = 8322u32;
__t177.subvendor = 4294967295u32;
__t177.subdevice = 4294967295u32;
__t177.class = 0u32;
__t177.class_mask = 0u32;
__t177.driver_data = 26u64;
__t177
}, {
let mut __t178: pci_device_id = unsafe { core::mem::zeroed() };
__t178.vendor = 36366u32;
__t178.device = 770u32;
__t178.subvendor = 36366u32;
__t178.subdevice = 1u32;
__t178.class = 0u32;
__t178.class_mask = 0u32;
__t178.driver_data = 79u64;
__t178
}, {
let mut __t179: pci_device_id = unsafe { core::mem::zeroed() };
__t179.vendor = 36366u32;
__t179.device = 770u32;
__t179.subvendor = 36366u32;
__t179.subdevice = 2u32;
__t179.class = 0u32;
__t179.class_mask = 0u32;
__t179.driver_data = 81u64;
__t179
}, {
let mut __t180: pci_device_id = unsafe { core::mem::zeroed() };
__t180.vendor = 36366u32;
__t180.device = 770u32;
__t180.subvendor = 36366u32;
__t180.subdevice = 3u32;
__t180.class = 0u32;
__t180.class_mask = 0u32;
__t180.driver_data = 80u64;
__t180
}, {
let mut __t181: pci_device_id = unsafe { core::mem::zeroed() };
__t181.vendor = 5141u32;
__t181.device = 38161u32;
__t181.subvendor = 4294967295u32;
__t181.subdevice = 4294967295u32;
__t181.class = 0u32;
__t181.class_mask = 0u32;
__t181.driver_data = 72u64;
__t181
}, {
let mut __t182: pci_device_id = unsafe { core::mem::zeroed() };
__t182.vendor = 5129u32;
__t182.device = 29032u32;
__t182.subvendor = 5129u32;
__t182.subdevice = 4294967295u32;
__t182.class = 0u32;
__t182.class_mask = 0u32;
__t182.driver_data = 23u64;
__t182
}, {
let mut __t183: pci_device_id = unsafe { core::mem::zeroed() };
__t183.vendor = 8148u32;
__t183.device = 6553u32;
__t183.subvendor = 8148u32;
__t183.subdevice = 1u32;
__t183.class = 0u32;
__t183.class_mask = 0u32;
__t183.driver_data = 105u64;
__t183
}, {
let mut __t184: pci_device_id = unsafe { core::mem::zeroed() };
__t184.vendor = 8148u32;
__t184.device = 6553u32;
__t184.subvendor = 8148u32;
__t184.subdevice = 2u32;
__t184.class = 0u32;
__t184.class_mask = 0u32;
__t184.driver_data = 106u64;
__t184
}, {
let mut __t185: pci_device_id = unsafe { core::mem::zeroed() };
__t185.vendor = 8148u32;
__t185.device = 6553u32;
__t185.subvendor = 8148u32;
__t185.subdevice = 4u32;
__t185.class = 0u32;
__t185.class_mask = 0u32;
__t185.driver_data = 107u64;
__t185
}, {
let mut __t186: pci_device_id = unsafe { core::mem::zeroed() };
__t186.vendor = 8148u32;
__t186.device = 6553u32;
__t186.subvendor = 8148u32;
__t186.subdevice = 132u32;
__t186.class = 0u32;
__t186.class_mask = 0u32;
__t186.driver_data = 107u64;
__t186
}, {
let mut __t187: pci_device_id = unsafe { core::mem::zeroed() };
__t187.vendor = 8148u32;
__t187.device = 6553u32;
__t187.subvendor = 8148u32;
__t187.subdevice = 8u32;
__t187.class = 0u32;
__t187.class_mask = 0u32;
__t187.driver_data = 108u64;
__t187
}, {
let mut __t188: pci_device_id = unsafe { core::mem::zeroed() };
__t188.vendor = 8148u32;
__t188.device = 6553u32;
__t188.subvendor = 8148u32;
__t188.subdevice = 136u32;
__t188.class = 0u32;
__t188.class_mask = 0u32;
__t188.driver_data = 108u64;
__t188
}, {
let mut __t189: pci_device_id = unsafe { core::mem::zeroed() };
__t189.vendor = 8148u32;
__t189.device = 6553u32;
__t189.subvendor = 8148u32;
__t189.subdevice = 16u32;
__t189.class = 0u32;
__t189.class_mask = 0u32;
__t189.driver_data = 109u64;
__t189
}, {
let mut __t190: pci_device_id = unsafe { core::mem::zeroed() };
__t190.vendor = 5339u32;
__t190.device = 8576u32;
__t190.subvendor = 4294967295u32;
__t190.subdevice = 4294967295u32;
__t190.class = 0u32;
__t190.class_mask = 0u32;
__t190.driver_data = 19u64;
__t190
}, {
let mut __t191: pci_device_id = unsafe { core::mem::zeroed() };
__t191.vendor = 5339u32;
__t191.device = 8578u32;
__t191.subvendor = 4294967295u32;
__t191.subdevice = 4294967295u32;
__t191.class = 0u32;
__t191.class_mask = 0u32;
__t191.driver_data = 19u64;
__t191
}, {
let mut __t192: pci_device_id = unsafe { core::mem::zeroed() };
__t192.vendor = 5127u32;
__t192.device = 256u32;
__t192.subvendor = 4294967295u32;
__t192.subdevice = 4294967295u32;
__t192.class = 0u32;
__t192.class_mask = 0u32;
__t192.driver_data = 17u64;
__t192
}, {
let mut __t193: pci_device_id = unsafe { core::mem::zeroed() };
__t193.vendor = 5127u32;
__t193.device = 257u32;
__t193.subvendor = 4294967295u32;
__t193.subdevice = 4294967295u32;
__t193.class = 0u32;
__t193.class_mask = 0u32;
__t193.driver_data = 17u64;
__t193
}, {
let mut __t194: pci_device_id = unsafe { core::mem::zeroed() };
__t194.vendor = 5127u32;
__t194.device = 258u32;
__t194.subvendor = 4294967295u32;
__t194.subdevice = 4294967295u32;
__t194.class = 0u32;
__t194.class_mask = 0u32;
__t194.driver_data = 17u64;
__t194
}, {
let mut __t195: pci_device_id = unsafe { core::mem::zeroed() };
__t195.vendor = 5127u32;
__t195.device = 384u32;
__t195.subvendor = 4294967295u32;
__t195.subdevice = 4294967295u32;
__t195.class = 0u32;
__t195.class_mask = 0u32;
__t195.driver_data = 22u64;
__t195
}, {
let mut __t196: pci_device_id = unsafe { core::mem::zeroed() };
__t196.vendor = 5127u32;
__t196.device = 385u32;
__t196.subvendor = 4294967295u32;
__t196.subdevice = 4294967295u32;
__t196.class = 0u32;
__t196.class_mask = 0u32;
__t196.driver_data = 22u64;
__t196
}, {
let mut __t197: pci_device_id = unsafe { core::mem::zeroed() };
__t197.vendor = 5127u32;
__t197.device = 512u32;
__t197.subvendor = 4294967295u32;
__t197.subdevice = 4294967295u32;
__t197.class = 0u32;
__t197.class_mask = 0u32;
__t197.driver_data = 21u64;
__t197
}, {
let mut __t198: pci_device_id = unsafe { core::mem::zeroed() };
__t198.vendor = 5127u32;
__t198.device = 513u32;
__t198.subvendor = 4294967295u32;
__t198.subdevice = 4294967295u32;
__t198.class = 0u32;
__t198.class_mask = 0u32;
__t198.driver_data = 21u64;
__t198
}, {
let mut __t199: pci_device_id = unsafe { core::mem::zeroed() };
__t199.vendor = 5127u32;
__t199.device = 514u32;
__t199.subvendor = 4294967295u32;
__t199.subdevice = 4294967295u32;
__t199.class = 0u32;
__t199.class_mask = 0u32;
__t199.driver_data = 21u64;
__t199
}, {
let mut __t200: pci_device_id = unsafe { core::mem::zeroed() };
__t200.vendor = 5127u32;
__t200.device = 1280u32;
__t200.subvendor = 4294967295u32;
__t200.subdevice = 4294967295u32;
__t200.class = 0u32;
__t200.class_mask = 0u32;
__t200.driver_data = 16u64;
__t200
}, {
let mut __t201: pci_device_id = unsafe { core::mem::zeroed() };
__t201.vendor = 5127u32;
__t201.device = 1536u32;
__t201.subvendor = 4294967295u32;
__t201.subdevice = 4294967295u32;
__t201.class = 0u32;
__t201.class_mask = 0u32;
__t201.driver_data = 20u64;
__t201
}, {
let mut __t202: pci_device_id = unsafe { core::mem::zeroed() };
__t202.vendor = 6530u32;
__t202.device = 5632u32;
__t202.subvendor = 4612u32;
__t202.subdevice = 4u32;
__t202.class = 0u32;
__t202.class_mask = 0u32;
__t202.driver_data = 8u64;
__t202
}, {
let mut __t203: pci_device_id = unsafe { core::mem::zeroed() };
__t203.vendor = 6530u32;
__t203.device = 5632u32;
__t203.subvendor = 4616u32;
__t203.subdevice = 4u32;
__t203.class = 0u32;
__t203.class_mask = 0u32;
__t203.driver_data = 8u64;
__t203
}, {
let mut __t204: pci_device_id = unsafe { core::mem::zeroed() };
__t204.vendor = 6530u32;
__t204.device = 5887u32;
__t204.subvendor = 4616u32;
__t204.subdevice = 4u32;
__t204.class = 0u32;
__t204.class_mask = 0u32;
__t204.driver_data = 8u64;
__t204
}, {
let mut __t205: pci_device_id = unsafe { core::mem::zeroed() };
__t205.vendor = 6530u32;
__t205.device = 5888u32;
__t205.subvendor = 4612u32;
__t205.subdevice = 4u32;
__t205.class = 0u32;
__t205.class_mask = 0u32;
__t205.driver_data = 8u64;
__t205
}, {
let mut __t206: pci_device_id = unsafe { core::mem::zeroed() };
__t206.vendor = 6530u32;
__t206.device = 5888u32;
__t206.subvendor = 4616u32;
__t206.subdevice = 4u32;
__t206.class = 0u32;
__t206.class_mask = 0u32;
__t206.driver_data = 8u64;
__t206
}, {
let mut __t207: pci_device_id = unsafe { core::mem::zeroed() };
__t207.vendor = 6530u32;
__t207.device = 6143u32;
__t207.subvendor = 4616u32;
__t207.subdevice = 4u32;
__t207.class = 0u32;
__t207.class_mask = 0u32;
__t207.driver_data = 8u64;
__t207
}, {
let mut __t208: pci_device_id = unsafe { core::mem::zeroed() };
__t208.vendor = 4136u32;
__t208.device = 18u32;
__t208.subvendor = 4294967295u32;
__t208.subdevice = 4294967295u32;
__t208.class = 0u32;
__t208.class_mask = 0u32;
__t208.driver_data = 41u64;
__t208
}, {
let mut __t209: pci_device_id = unsafe { core::mem::zeroed() };
__t209.vendor = 4136u32;
__t209.device = 8u32;
__t209.subvendor = 4294967295u32;
__t209.subdevice = 4294967295u32;
__t209.class = 0u32;
__t209.class_mask = 0u32;
__t209.driver_data = 41u64;
__t209
}, {
let mut __t210: pci_device_id = unsafe { core::mem::zeroed() };
__t210.vendor = 5546u32;
__t210.device = 8192u32;
__t210.subvendor = 4294967295u32;
__t210.subdevice = 4294967295u32;
__t210.class = 0u32;
__t210.class_mask = 0u32;
__t210.driver_data = 58u64;
__t210
}, {
let mut __t211: pci_device_id = unsafe { core::mem::zeroed() };
__t211.vendor = 32902u32;
__t211.device = 6496u32;
__t211.subvendor = 58559u32;
__t211.subdevice = 4294967295u32;
__t211.class = 0u32;
__t211.class_mask = 0u32;
__t211.driver_data = 77u64;
__t211
}, {
let mut __t212: pci_device_id = unsafe { core::mem::zeroed() };
__t212.vendor = 4445u32;
__t212.device = 259u32;
__t212.subvendor = 4294967295u32;
__t212.subdevice = 4294967295u32;
__t212.class = 0u32;
__t212.class_mask = 0u32;
__t212.driver_data = 1u64;
__t212
}, {
let mut __t213: pci_device_id = unsafe { core::mem::zeroed() };
__t213.vendor = 4445u32;
__t213.device = 257u32;
__t213.subvendor = 4294967295u32;
__t213.subdevice = 4294967295u32;
__t213.class = 0u32;
__t213.class_mask = 0u32;
__t213.driver_data = 1u64;
__t213
}, {
let mut __t214: pci_device_id = unsafe { core::mem::zeroed() };
__t214.vendor = 4730u32;
__t214.device = 4100u32;
__t214.subvendor = 4168u32;
__t214.subdevice = 5376u32;
__t214.class = 0u32;
__t214.class_mask = 0u32;
__t214.driver_data = 27u64;
__t214
}, {
let mut __t215: pci_device_id = unsafe { core::mem::zeroed() };
__t215.vendor = 4265u32;
__t215.device = 3u32;
__t215.subvendor = 65280u32;
__t215.subdevice = 0u32;
__t215.class = 0u32;
__t215.class_mask = 0u32;
__t215.driver_data = 78u64;
__t215
}, {
let mut __t216: pci_device_id = unsafe { core::mem::zeroed() };
__t216.vendor = 4156u32;
__t216.device = 4168u32;
__t216.subvendor = 4156u32;
__t216.subdevice = 4865u32;
__t216.class = 0u32;
__t216.class_mask = 0u32;
__t216.driver_data = 27u64;
__t216
}, {
let mut __t217: pci_device_id = unsafe { core::mem::zeroed() };
__t217.vendor = 4156u32;
__t217.device = 4168u32;
__t217.subvendor = 4294967295u32;
__t217.subdevice = 4294967295u32;
__t217.class = 0u32;
__t217.class_mask = 0u32;
__t217.driver_data = 4u64;
__t217
}, {
let mut __t218: pci_device_id = unsafe { core::mem::zeroed() };
__t218.vendor = 4156u32;
__t218.device = 4752u32;
__t218.subvendor = 4294967295u32;
__t218.subdevice = 4294967295u32;
__t218.class = 0u32;
__t218.class_mask = 0u32;
__t218.driver_data = 45u64;
__t218
}, {
let mut __t219: pci_device_id = unsafe { core::mem::zeroed() };
__t219.vendor = 5520u32;
__t219.device = 894u32;
__t219.subvendor = 4294967295u32;
__t219.subdevice = 4294967295u32;
__t219.class = 0u32;
__t219.class_mask = 0u32;
__t219.driver_data = 27u64;
__t219
}, {
let mut __t220: pci_device_id = unsafe { core::mem::zeroed() };
__t220.vendor = 26214u32;
__t220.device = 4u32;
__t220.subvendor = 4294967295u32;
__t220.subdevice = 4294967295u32;
__t220.class = 0u32;
__t220.class_mask = 0u32;
__t220.driver_data = 62u64;
__t220
}, {
let mut __t221: pci_device_id = unsafe { core::mem::zeroed() };
__t221.vendor = 26214u32;
__t221.device = 1u32;
__t221.subvendor = 4294967295u32;
__t221.subdevice = 4294967295u32;
__t221.class = 0u32;
__t221.class_mask = 0u32;
__t221.driver_data = 63u64;
__t221
}, {
let mut __t222: pci_device_id = unsafe { core::mem::zeroed() };
__t222.vendor = 26214u32;
__t222.device = 2u32;
__t222.subvendor = 4294967295u32;
__t222.subdevice = 4294967295u32;
__t222.class = 0u32;
__t222.class_mask = 0u32;
__t222.driver_data = 64u64;
__t222
}, {
let mut __t223: pci_device_id = unsafe { core::mem::zeroed() };
__t223.vendor = 5407u32;
__t223.device = 0u32;
__t223.subvendor = 4294967295u32;
__t223.subdevice = 4294967295u32;
__t223.class = 0u32;
__t223.class_mask = 0u32;
__t223.driver_data = 1u64;
__t223
}, {
let mut __t224: pci_device_id = unsafe { core::mem::zeroed() };
__t224.vendor = 4739u32;
__t224.device = 34930u32;
__t224.subvendor = 4294967295u32;
__t224.subdevice = 4294967295u32;
__t224.class = 0u32;
__t224.class_mask = 0u32;
__t224.driver_data = 37u64;
__t224
}, {
let mut __t225: pci_device_id = unsafe { core::mem::zeroed() };
__t225.vendor = 4954u32;
__t225.device = 3424u32;
__t225.subvendor = 4294967295u32;
__t225.subdevice = 4294967295u32;
__t225.class = 0u32;
__t225.class_mask = 0u32;
__t225.driver_data = 45u64;
__t225
}, {
let mut __t226: pci_device_id = unsafe { core::mem::zeroed() };
__t226.vendor = 4954u32;
__t226.device = 3456u32;
__t226.subvendor = 4294967295u32;
__t226.subdevice = 4294967295u32;
__t226.class = 0u32;
__t226.class_mask = 0u32;
__t226.driver_data = 46u64;
__t226
}, {
let mut __t227: pci_device_id = unsafe { core::mem::zeroed() };
__t227.vendor = 4954u32;
__t227.device = 3520u32;
__t227.subvendor = 4294967295u32;
__t227.subdevice = 4294967295u32;
__t227.class = 0u32;
__t227.class_mask = 0u32;
__t227.driver_data = 47u64;
__t227
}, {
let mut __t228: pci_device_id = unsafe { core::mem::zeroed() };
__t228.vendor = 4954u32;
__t228.device = 16423u32;
__t228.subvendor = 4294967295u32;
__t228.subdevice = 4294967295u32;
__t228.class = 0u32;
__t228.class_mask = 0u32;
__t228.driver_data = 73u64;
__t228
}, {
let mut __t229: pci_device_id = unsafe { core::mem::zeroed() };
__t229.vendor = 4954u32;
__t229.device = 16424u32;
__t229.subvendor = 4294967295u32;
__t229.subdevice = 4294967295u32;
__t229.class = 0u32;
__t229.class_mask = 0u32;
__t229.driver_data = 74u64;
__t229
}, {
let mut __t230: pci_device_id = unsafe { core::mem::zeroed() };
__t230.vendor = 4954u32;
__t230.device = 16425u32;
__t230.subvendor = 4294967295u32;
__t230.subdevice = 4294967295u32;
__t230.class = 0u32;
__t230.class_mask = 0u32;
__t230.driver_data = 75u64;
__t230
}, {
let mut __t231: pci_device_id = unsafe { core::mem::zeroed() };
__t231.vendor = 4954u32;
__t231.device = 2977u32;
__t231.subvendor = 4294967295u32;
__t231.subdevice = 4294967295u32;
__t231.class = 0u32;
__t231.class_mask = 0u32;
__t231.driver_data = 46u64;
__t231
}, {
let mut __t232: pci_device_id = unsafe { core::mem::zeroed() };
__t232.vendor = 4954u32;
__t232.device = 2978u32;
__t232.subvendor = 4294967295u32;
__t232.subdevice = 4294967295u32;
__t232.class = 0u32;
__t232.class_mask = 0u32;
__t232.driver_data = 46u64;
__t232
}, {
let mut __t233: pci_device_id = unsafe { core::mem::zeroed() };
__t233.vendor = 4954u32;
__t233.device = 2979u32;
__t233.subvendor = 4294967295u32;
__t233.subdevice = 4294967295u32;
__t233.class = 0u32;
__t233.class_mask = 0u32;
__t233.driver_data = 46u64;
__t233
}, {
let mut __t234: pci_device_id = unsafe { core::mem::zeroed() };
__t234.vendor = 4954u32;
__t234.device = 2721u32;
__t234.subvendor = 4294967295u32;
__t234.subdevice = 4294967295u32;
__t234.class = 0u32;
__t234.class_mask = 0u32;
__t234.driver_data = 45u64;
__t234
}, {
let mut __t235: pci_device_id = unsafe { core::mem::zeroed() };
__t235.vendor = 4954u32;
__t235.device = 2722u32;
__t235.subvendor = 4294967295u32;
__t235.subdevice = 4294967295u32;
__t235.class = 0u32;
__t235.class_mask = 0u32;
__t235.driver_data = 45u64;
__t235
}, {
let mut __t236: pci_device_id = unsafe { core::mem::zeroed() };
__t236.vendor = 4954u32;
__t236.device = 3233u32;
__t236.subvendor = 4294967295u32;
__t236.subdevice = 4294967295u32;
__t236.class = 0u32;
__t236.class_mask = 0u32;
__t236.driver_data = 46u64;
__t236
}, {
let mut __t237: pci_device_id = unsafe { core::mem::zeroed() };
__t237.vendor = 4954u32;
__t237.device = 3361u32;
__t237.subvendor = 4294967295u32;
__t237.subdevice = 4294967295u32;
__t237.class = 459264u32;
__t237.class_mask = 16776960u32;
__t237.driver_data = 47u64;
__t237
}, {
let mut __t238: pci_device_id = unsafe { core::mem::zeroed() };
__t238.vendor = 4954u32;
__t238.device = 3636u32;
__t238.subvendor = 4294967295u32;
__t238.subdevice = 4294967295u32;
__t238.class = 459264u32;
__t238.class_mask = 16776960u32;
__t238.driver_data = 47u64;
__t238
}, {
let mut __t239: pci_device_id = unsafe { core::mem::zeroed() };
__t239.vendor = 4954u32;
__t239.device = 2113u32;
__t239.subvendor = 4294967295u32;
__t239.subdevice = 4294967295u32;
__t239.class = 0u32;
__t239.class_mask = 0u32;
__t239.driver_data = 47u64;
__t239
}, {
let mut __t240: pci_device_id = unsafe { core::mem::zeroed() };
__t240.vendor = 4954u32;
__t240.device = 2177u32;
__t240.subvendor = 4294967295u32;
__t240.subdevice = 4294967295u32;
__t240.class = 0u32;
__t240.class_mask = 0u32;
__t240.driver_data = 48u64;
__t240
}, {
let mut __t241: pci_device_id = unsafe { core::mem::zeroed() };
__t241.vendor = 4954u32;
__t241.device = 2273u32;
__t241.subvendor = 4294967295u32;
__t241.subdevice = 4294967295u32;
__t241.class = 0u32;
__t241.class_mask = 0u32;
__t241.driver_data = 46u64;
__t241
}, {
let mut __t242: pci_device_id = unsafe { core::mem::zeroed() };
__t242.vendor = 4954u32;
__t242.device = 2274u32;
__t242.subvendor = 4294967295u32;
__t242.subdevice = 4294967295u32;
__t242.class = 0u32;
__t242.class_mask = 0u32;
__t242.driver_data = 46u64;
__t242
}, {
let mut __t243: pci_device_id = unsafe { core::mem::zeroed() };
__t243.vendor = 4954u32;
__t243.device = 2275u32;
__t243.subvendor = 4294967295u32;
__t243.subdevice = 4294967295u32;
__t243.class = 0u32;
__t243.class_mask = 0u32;
__t243.driver_data = 46u64;
__t243
}, {
let mut __t244: pci_device_id = unsafe { core::mem::zeroed() };
__t244.vendor = 4954u32;
__t244.device = 2241u32;
__t244.subvendor = 4294967295u32;
__t244.subdevice = 4294967295u32;
__t244.class = 0u32;
__t244.class_mask = 0u32;
__t244.driver_data = 46u64;
__t244
}, {
let mut __t245: pci_device_id = unsafe { core::mem::zeroed() };
__t245.vendor = 4954u32;
__t245.device = 2209u32;
__t245.subvendor = 4294967295u32;
__t245.subdevice = 4294967295u32;
__t245.class = 0u32;
__t245.class_mask = 0u32;
__t245.driver_data = 46u64;
__t245
}, {
let mut __t246: pci_device_id = unsafe { core::mem::zeroed() };
__t246.vendor = 4954u32;
__t246.device = 2210u32;
__t246.subvendor = 4294967295u32;
__t246.subdevice = 4294967295u32;
__t246.class = 0u32;
__t246.class_mask = 0u32;
__t246.driver_data = 46u64;
__t246
}, {
let mut __t247: pci_device_id = unsafe { core::mem::zeroed() };
__t247.vendor = 4954u32;
__t247.device = 2211u32;
__t247.subvendor = 4294967295u32;
__t247.subdevice = 4294967295u32;
__t247.class = 0u32;
__t247.class_mask = 0u32;
__t247.driver_data = 46u64;
__t247
}, {
let mut __t248: pci_device_id = unsafe { core::mem::zeroed() };
__t248.vendor = 4954u32;
__t248.device = 2657u32;
__t248.subvendor = 4294967295u32;
__t248.subdevice = 4294967295u32;
__t248.class = 0u32;
__t248.class_mask = 0u32;
__t248.driver_data = 45u64;
__t248
}, {
let mut __t249: pci_device_id = unsafe { core::mem::zeroed() };
__t249.vendor = 4954u32;
__t249.device = 2817u32;
__t249.subvendor = 4294967295u32;
__t249.subdevice = 4294967295u32;
__t249.class = 0u32;
__t249.class_mask = 0u32;
__t249.driver_data = 47u64;
__t249
}, {
let mut __t250: pci_device_id = unsafe { core::mem::zeroed() };
__t250.vendor = 4954u32;
__t250.device = 2818u32;
__t250.subvendor = 4294967295u32;
__t250.subdevice = 4294967295u32;
__t250.class = 0u32;
__t250.class_mask = 0u32;
__t250.driver_data = 47u64;
__t250
}, {
let mut __t251: pci_device_id = unsafe { core::mem::zeroed() };
__t251.vendor = 4954u32;
__t251.device = 2689u32;
__t251.subvendor = 4294967295u32;
__t251.subdevice = 4294967295u32;
__t251.class = 0u32;
__t251.class_mask = 0u32;
__t251.driver_data = 46u64;
__t251
}, {
let mut __t252: pci_device_id = unsafe { core::mem::zeroed() };
__t252.vendor = 4954u32;
__t252.device = 2690u32;
__t252.subvendor = 4294967295u32;
__t252.subdevice = 4294967295u32;
__t252.class = 0u32;
__t252.class_mask = 0u32;
__t252.driver_data = 46u64;
__t252
}, {
let mut __t253: pci_device_id = unsafe { core::mem::zeroed() };
__t253.vendor = 4954u32;
__t253.device = 2691u32;
__t253.subvendor = 4294967295u32;
__t253.subdevice = 4294967295u32;
__t253.class = 0u32;
__t253.class_mask = 0u32;
__t253.driver_data = 46u64;
__t253
}, {
let mut __t254: pci_device_id = unsafe { core::mem::zeroed() };
__t254.vendor = 4954u32;
__t254.device = 3137u32;
__t254.subvendor = 4294967295u32;
__t254.subdevice = 4294967295u32;
__t254.class = 0u32;
__t254.class_mask = 0u32;
__t254.driver_data = 47u64;
__t254
}, {
let mut __t255: pci_device_id = unsafe { core::mem::zeroed() };
__t255.vendor = 4954u32;
__t255.device = 3138u32;
__t255.subvendor = 4294967295u32;
__t255.subdevice = 4294967295u32;
__t255.class = 0u32;
__t255.class_mask = 0u32;
__t255.driver_data = 47u64;
__t255
}, {
let mut __t256: pci_device_id = unsafe { core::mem::zeroed() };
__t256.vendor = 4954u32;
__t256.device = 3139u32;
__t256.subvendor = 4294967295u32;
__t256.subdevice = 4294967295u32;
__t256.class = 0u32;
__t256.class_mask = 0u32;
__t256.driver_data = 47u64;
__t256
}, {
let mut __t257: pci_device_id = unsafe { core::mem::zeroed() };
__t257.vendor = 4954u32;
__t257.device = 2337u32;
__t257.subvendor = 4294967295u32;
__t257.subdevice = 4294967295u32;
__t257.class = 0u32;
__t257.class_mask = 0u32;
__t257.driver_data = 47u64;
__t257
}, {
let mut __t258: pci_device_id = unsafe { core::mem::zeroed() };
__t258.vendor = 4954u32;
__t258.device = 2465u32;
__t258.subvendor = 4294967295u32;
__t258.subdevice = 4294967295u32;
__t258.class = 0u32;
__t258.class_mask = 0u32;
__t258.driver_data = 46u64;
__t258
}, {
let mut __t259: pci_device_id = unsafe { core::mem::zeroed() };
__t259.vendor = 4954u32;
__t259.device = 2466u32;
__t259.subvendor = 4294967295u32;
__t259.subdevice = 4294967295u32;
__t259.class = 0u32;
__t259.class_mask = 0u32;
__t259.driver_data = 46u64;
__t259
}, {
let mut __t260: pci_device_id = unsafe { core::mem::zeroed() };
__t260.vendor = 4954u32;
__t260.device = 2467u32;
__t260.subvendor = 4294967295u32;
__t260.subdevice = 4294967295u32;
__t260.class = 0u32;
__t260.class_mask = 0u32;
__t260.driver_data = 46u64;
__t260
}, {
let mut __t261: pci_device_id = unsafe { core::mem::zeroed() };
__t261.vendor = 4954u32;
__t261.device = 3393u32;
__t261.subvendor = 4294967295u32;
__t261.subdevice = 4294967295u32;
__t261.class = 0u32;
__t261.class_mask = 0u32;
__t261.driver_data = 47u64;
__t261
}, {
let mut __t262: pci_device_id = unsafe { core::mem::zeroed() };
__t262.vendor = 4954u32;
__t262.device = 2753u32;
__t262.subvendor = 4294967295u32;
__t262.subdevice = 4294967295u32;
__t262.class = 0u32;
__t262.class_mask = 0u32;
__t262.driver_data = 46u64;
__t262
}, {
let mut __t263: pci_device_id = unsafe { core::mem::zeroed() };
__t263.vendor = 4954u32;
__t263.device = 2754u32;
__t263.subvendor = 4294967295u32;
__t263.subdevice = 4294967295u32;
__t263.class = 0u32;
__t263.class_mask = 0u32;
__t263.driver_data = 46u64;
__t263
}, {
let mut __t264: pci_device_id = unsafe { core::mem::zeroed() };
__t264.vendor = 4954u32;
__t264.device = 2755u32;
__t264.subvendor = 4294967295u32;
__t264.subdevice = 4294967295u32;
__t264.class = 0u32;
__t264.class_mask = 0u32;
__t264.driver_data = 46u64;
__t264
}, {
let mut __t265: pci_device_id = unsafe { core::mem::zeroed() };
__t265.vendor = 4954u32;
__t265.device = 2849u32;
__t265.subvendor = 4294967295u32;
__t265.subdevice = 4294967295u32;
__t265.class = 0u32;
__t265.class_mask = 0u32;
__t265.driver_data = 46u64;
__t265
}, {
let mut __t266: pci_device_id = unsafe { core::mem::zeroed() };
__t266.vendor = 4954u32;
__t266.device = 2850u32;
__t266.subvendor = 4294967295u32;
__t266.subdevice = 4294967295u32;
__t266.class = 0u32;
__t266.class_mask = 0u32;
__t266.driver_data = 46u64;
__t266
}, {
let mut __t267: pci_device_id = unsafe { core::mem::zeroed() };
__t267.vendor = 4954u32;
__t267.device = 2851u32;
__t267.subvendor = 4294967295u32;
__t267.subdevice = 4294967295u32;
__t267.class = 0u32;
__t267.class_mask = 0u32;
__t267.driver_data = 46u64;
__t267
}, {
let mut __t268: pci_device_id = unsafe { core::mem::zeroed() };
__t268.vendor = 4954u32;
__t268.device = 3073u32;
__t268.subvendor = 4294967295u32;
__t268.subdevice = 4294967295u32;
__t268.class = 0u32;
__t268.class_mask = 0u32;
__t268.driver_data = 46u64;
__t268
}, {
let mut __t269: pci_device_id = unsafe { core::mem::zeroed() };
__t269.vendor = 4954u32;
__t269.device = 3074u32;
__t269.subvendor = 4294967295u32;
__t269.subdevice = 4294967295u32;
__t269.class = 0u32;
__t269.class_mask = 0u32;
__t269.driver_data = 46u64;
__t269
}, {
let mut __t270: pci_device_id = unsafe { core::mem::zeroed() };
__t270.vendor = 4954u32;
__t270.device = 3075u32;
__t270.subvendor = 4294967295u32;
__t270.subdevice = 4294967295u32;
__t270.class = 0u32;
__t270.class_mask = 0u32;
__t270.driver_data = 46u64;
__t270
}, {
let mut __t271: pci_device_id = unsafe { core::mem::zeroed() };
__t271.vendor = 4954u32;
__t271.device = 3105u32;
__t271.subvendor = 4294967295u32;
__t271.subdevice = 4294967295u32;
__t271.class = 0u32;
__t271.class_mask = 0u32;
__t271.driver_data = 46u64;
__t271
}, {
let mut __t272: pci_device_id = unsafe { core::mem::zeroed() };
__t272.vendor = 4954u32;
__t272.device = 3106u32;
__t272.subvendor = 4294967295u32;
__t272.subdevice = 4294967295u32;
__t272.class = 0u32;
__t272.class_mask = 0u32;
__t272.driver_data = 46u64;
__t272
}, {
let mut __t273: pci_device_id = unsafe { core::mem::zeroed() };
__t273.vendor = 4954u32;
__t273.device = 3107u32;
__t273.subvendor = 4294967295u32;
__t273.subdevice = 4294967295u32;
__t273.class = 0u32;
__t273.class_mask = 0u32;
__t273.driver_data = 46u64;
__t273
}, {
let mut __t274: pci_device_id = unsafe { core::mem::zeroed() };
__t274.vendor = 4954u32;
__t274.device = 16389u32;
__t274.subvendor = 4294967295u32;
__t274.subdevice = 4294967295u32;
__t274.class = 0u32;
__t274.class_mask = 0u32;
__t274.driver_data = 2u64;
__t274
}, {
let mut __t275: pci_device_id = unsafe { core::mem::zeroed() };
__t275.vendor = 4954u32;
__t275.device = 16409u32;
__t275.subvendor = 4294967295u32;
__t275.subdevice = 4294967295u32;
__t275.class = 0u32;
__t275.class_mask = 0u32;
__t275.driver_data = 74u64;
__t275
}, {
let mut __t276: pci_device_id = unsafe { core::mem::zeroed() };
__t276.vendor = 4954u32;
__t276.device = 16388u32;
__t276.subvendor = 4294967295u32;
__t276.subdevice = 4294967295u32;
__t276.class = 0u32;
__t276.class_mask = 0u32;
__t276.driver_data = 1u64;
__t276
}, {
let mut __t277: pci_device_id = unsafe { core::mem::zeroed() };
__t277.vendor = 4954u32;
__t277.device = 16406u32;
__t277.subvendor = 4294967295u32;
__t277.subdevice = 4294967295u32;
__t277.class = 0u32;
__t277.class_mask = 0u32;
__t277.driver_data = 73u64;
__t277
}, {
let mut __t278: pci_device_id = unsafe { core::mem::zeroed() };
__t278.vendor = 4954u32;
__t278.device = 16390u32;
__t278.subvendor = 4294967295u32;
__t278.subdevice = 4294967295u32;
__t278.class = 0u32;
__t278.class_mask = 0u32;
__t278.driver_data = 2u64;
__t278
}, {
let mut __t279: pci_device_id = unsafe { core::mem::zeroed() };
__t279.vendor = 4954u32;
__t279.device = 16405u32;
__t279.subvendor = 4294967295u32;
__t279.subdevice = 4294967295u32;
__t279.class = 0u32;
__t279.class_mask = 0u32;
__t279.driver_data = 74u64;
__t279
}, {
let mut __t280: pci_device_id = unsafe { core::mem::zeroed() };
__t280.vendor = 4954u32;
__t280.device = 16394u32;
__t280.subvendor = 4294967295u32;
__t280.subdevice = 4294967295u32;
__t280.class = 0u32;
__t280.class_mask = 0u32;
__t280.driver_data = 75u64;
__t280
}, {
let mut __t281: pci_device_id = unsafe { core::mem::zeroed() };
__t281.vendor = 4954u32;
__t281.device = 3649u32;
__t281.subvendor = 4294967295u32;
__t281.subdevice = 4294967295u32;
__t281.class = 0u32;
__t281.class_mask = 0u32;
__t281.driver_data = 48u64;
__t281
}, {
let mut __t282: pci_device_id = unsafe { core::mem::zeroed() };
__t282.vendor = 4954u32;
__t282.device = 16398u32;
__t282.subvendor = 4294967295u32;
__t282.subdevice = 4294967295u32;
__t282.class = 0u32;
__t282.class_mask = 0u32;
__t282.driver_data = 74u64;
__t282
}, {
let mut __t283: pci_device_id = unsafe { core::mem::zeroed() };
__t283.vendor = 4954u32;
__t283.device = 16396u32;
__t283.subvendor = 4294967295u32;
__t283.subdevice = 4294967295u32;
__t283.class = 0u32;
__t283.class_mask = 0u32;
__t283.driver_data = 74u64;
__t283
}, {
let mut __t284: pci_device_id = unsafe { core::mem::zeroed() };
__t284.vendor = 4954u32;
__t284.device = 16395u32;
__t284.subvendor = 4294967295u32;
__t284.subdevice = 4294967295u32;
__t284.class = 0u32;
__t284.class_mask = 0u32;
__t284.driver_data = 73u64;
__t284
}, {
let mut __t285: pci_device_id = unsafe { core::mem::zeroed() };
__t285.vendor = 4954u32;
__t285.device = 16399u32;
__t285.subvendor = 4294967295u32;
__t285.subdevice = 4294967295u32;
__t285.class = 0u32;
__t285.class_mask = 0u32;
__t285.driver_data = 75u64;
__t285
}, {
let mut __t286: pci_device_id = unsafe { core::mem::zeroed() };
__t286.vendor = 4954u32;
__t286.device = 16400u32;
__t286.subvendor = 4294967295u32;
__t286.subdevice = 4294967295u32;
__t286.class = 0u32;
__t286.class_mask = 0u32;
__t286.driver_data = 75u64;
__t286
}, {
let mut __t287: pci_device_id = unsafe { core::mem::zeroed() };
__t287.vendor = 4954u32;
__t287.device = 16384u32;
__t287.subvendor = 4294967295u32;
__t287.subdevice = 4294967295u32;
__t287.class = 0u32;
__t287.class_mask = 0u32;
__t287.driver_data = 3u64;
__t287
}, {
let mut __t288: pci_device_id = unsafe { core::mem::zeroed() };
__t288.vendor = 4954u32;
__t288.device = 16401u32;
__t288.subvendor = 4294967295u32;
__t288.subdevice = 4294967295u32;
__t288.class = 0u32;
__t288.class_mask = 0u32;
__t288.driver_data = 75u64;
__t288
}, {
let mut __t289: pci_device_id = unsafe { core::mem::zeroed() };
__t289.vendor = 4954u32;
__t289.device = 16413u32;
__t289.subvendor = 4294967295u32;
__t289.subdevice = 4294967295u32;
__t289.class = 0u32;
__t289.class_mask = 0u32;
__t289.driver_data = 73u64;
__t289
}, {
let mut __t290: pci_device_id = unsafe { core::mem::zeroed() };
__t290.vendor = 4954u32;
__t290.device = 16393u32;
__t290.subvendor = 4294967295u32;
__t290.subdevice = 4294967295u32;
__t290.class = 0u32;
__t290.class_mask = 0u32;
__t290.driver_data = 2u64;
__t290
}, {
let mut __t291: pci_device_id = unsafe { core::mem::zeroed() };
__t291.vendor = 4954u32;
__t291.device = 16408u32;
__t291.subvendor = 4294967295u32;
__t291.subdevice = 4294967295u32;
__t291.class = 0u32;
__t291.class_mask = 0u32;
__t291.driver_data = 74u64;
__t291
}, {
let mut __t292: pci_device_id = unsafe { core::mem::zeroed() };
__t292.vendor = 4954u32;
__t292.device = 16414u32;
__t292.subvendor = 4294967295u32;
__t292.subdevice = 4294967295u32;
__t292.class = 0u32;
__t292.class_mask = 0u32;
__t292.driver_data = 74u64;
__t292
}, {
let mut __t293: pci_device_id = unsafe { core::mem::zeroed() };
__t293.vendor = 4954u32;
__t293.device = 16386u32;
__t293.subvendor = 4294967295u32;
__t293.subdevice = 4294967295u32;
__t293.class = 0u32;
__t293.class_mask = 0u32;
__t293.driver_data = 3u64;
__t293
}, {
let mut __t294: pci_device_id = unsafe { core::mem::zeroed() };
__t294.vendor = 4954u32;
__t294.device = 16403u32;
__t294.subvendor = 4294967295u32;
__t294.subdevice = 4294967295u32;
__t294.class = 0u32;
__t294.class_mask = 0u32;
__t294.driver_data = 75u64;
__t294
}, {
let mut __t295: pci_device_id = unsafe { core::mem::zeroed() };
__t295.vendor = 4954u32;
__t295.device = 16392u32;
__t295.subvendor = 4294967295u32;
__t295.subdevice = 4294967295u32;
__t295.class = 0u32;
__t295.class_mask = 0u32;
__t295.driver_data = 1u64;
__t295
}, {
let mut __t296: pci_device_id = unsafe { core::mem::zeroed() };
__t296.vendor = 4954u32;
__t296.device = 16407u32;
__t296.subvendor = 4294967295u32;
__t296.subdevice = 4294967295u32;
__t296.class = 0u32;
__t296.class_mask = 0u32;
__t296.driver_data = 73u64;
__t296
}, {
let mut __t297: pci_device_id = unsafe { core::mem::zeroed() };
__t297.vendor = 4954u32;
__t297.device = 16422u32;
__t297.subvendor = 4294967295u32;
__t297.subdevice = 4294967295u32;
__t297.class = 0u32;
__t297.class_mask = 0u32;
__t297.driver_data = 73u64;
__t297
}, {
let mut __t298: pci_device_id = unsafe { core::mem::zeroed() };
__t298.vendor = 4954u32;
__t298.device = 16417u32;
__t298.subvendor = 4294967295u32;
__t298.subdevice = 4294967295u32;
__t298.class = 0u32;
__t298.class_mask = 0u32;
__t298.driver_data = 73u64;
__t298
}, {
let mut __t299: pci_device_id = unsafe { core::mem::zeroed() };
__t299.vendor = 4277u32;
__t299.device = 36912u32;
__t299.subvendor = 5471u32;
__t299.subdevice = 61441u32;
__t299.class = 0u32;
__t299.class_mask = 0u32;
__t299.driver_data = 54u64;
__t299
}, {
let mut __t300: pci_device_id = unsafe { core::mem::zeroed() };
__t300.vendor = 4277u32;
__t300.device = 36912u32;
__t300.subvendor = 5471u32;
__t300.subdevice = 61456u32;
__t300.class = 0u32;
__t300.class_mask = 0u32;
__t300.driver_data = 55u64;
__t300
}, {
let mut __t301: pci_device_id = unsafe { core::mem::zeroed() };
__t301.vendor = 5410u32;
__t301.device = 256u32;
__t301.subvendor = 5410u32;
__t301.subdevice = 512u32;
__t301.class = 0u32;
__t301.class_mask = 0u32;
__t301.driver_data = 2u64;
__t301
}, {
let mut __t302: pci_device_id = unsafe { core::mem::zeroed() };
__t302.vendor = 5410u32;
__t302.device = 256u32;
__t302.subvendor = 5410u32;
__t302.subdevice = 768u32;
__t302.class = 0u32;
__t302.class_mask = 0u32;
__t302.driver_data = 3u64;
__t302
}, {
let mut __t303: pci_device_id = unsafe { core::mem::zeroed() };
__t303.vendor = 5410u32;
__t303.device = 256u32;
__t303.subvendor = 5410u32;
__t303.subdevice = 1024u32;
__t303.class = 0u32;
__t303.class_mask = 0u32;
__t303.driver_data = 2u64;
__t303
}, {
let mut __t304: pci_device_id = unsafe { core::mem::zeroed() };
__t304.vendor = 5410u32;
__t304.device = 256u32;
__t304.subvendor = 5410u32;
__t304.subdevice = 1280u32;
__t304.class = 0u32;
__t304.class_mask = 0u32;
__t304.driver_data = 3u64;
__t304
}, {
let mut __t305: pci_device_id = unsafe { core::mem::zeroed() };
__t305.vendor = 5410u32;
__t305.device = 256u32;
__t305.subvendor = 5410u32;
__t305.subdevice = 1536u32;
__t305.class = 0u32;
__t305.class_mask = 0u32;
__t305.driver_data = 2u64;
__t305
}, {
let mut __t306: pci_device_id = unsafe { core::mem::zeroed() };
__t306.vendor = 5410u32;
__t306.device = 256u32;
__t306.subvendor = 5410u32;
__t306.subdevice = 1792u32;
__t306.class = 0u32;
__t306.class_mask = 0u32;
__t306.driver_data = 3u64;
__t306
}, {
let mut __t307: pci_device_id = unsafe { core::mem::zeroed() };
__t307.vendor = 5410u32;
__t307.device = 256u32;
__t307.subvendor = 5410u32;
__t307.subdevice = 2048u32;
__t307.class = 0u32;
__t307.class_mask = 0u32;
__t307.driver_data = 5u64;
__t307
}, {
let mut __t308: pci_device_id = unsafe { core::mem::zeroed() };
__t308.vendor = 5410u32;
__t308.device = 256u32;
__t308.subvendor = 5410u32;
__t308.subdevice = 3072u32;
__t308.class = 0u32;
__t308.class_mask = 0u32;
__t308.driver_data = 2u64;
__t308
}, {
let mut __t309: pci_device_id = unsafe { core::mem::zeroed() };
__t309.vendor = 5410u32;
__t309.device = 256u32;
__t309.subvendor = 5410u32;
__t309.subdevice = 3328u32;
__t309.class = 0u32;
__t309.class_mask = 0u32;
__t309.driver_data = 3u64;
__t309
}, {
let mut __t310: pci_device_id = unsafe { core::mem::zeroed() };
__t310.vendor = 5410u32;
__t310.device = 256u32;
__t310.subvendor = 5410u32;
__t310.subdevice = 7424u32;
__t310.class = 0u32;
__t310.class_mask = 0u32;
__t310.driver_data = 5u64;
__t310
}, {
let mut __t311: pci_device_id = unsafe { core::mem::zeroed() };
__t311.vendor = 5410u32;
__t311.device = 256u32;
__t311.subvendor = 5410u32;
__t311.subdevice = 8192u32;
__t311.class = 0u32;
__t311.class_mask = 0u32;
__t311.driver_data = 1u64;
__t311
}, {
let mut __t312: pci_device_id = unsafe { core::mem::zeroed() };
__t312.vendor = 5410u32;
__t312.device = 256u32;
__t312.subvendor = 5410u32;
__t312.subdevice = 8448u32;
__t312.class = 0u32;
__t312.class_mask = 0u32;
__t312.driver_data = 1u64;
__t312
}, {
let mut __t313: pci_device_id = unsafe { core::mem::zeroed() };
__t313.vendor = 5410u32;
__t313.device = 256u32;
__t313.subvendor = 5410u32;
__t313.subdevice = 8704u32;
__t313.class = 0u32;
__t313.class_mask = 0u32;
__t313.driver_data = 2u64;
__t313
}, {
let mut __t314: pci_device_id = unsafe { core::mem::zeroed() };
__t314.vendor = 5410u32;
__t314.device = 256u32;
__t314.subvendor = 5410u32;
__t314.subdevice = 8960u32;
__t314.class = 0u32;
__t314.class_mask = 0u32;
__t314.driver_data = 2u64;
__t314
}, {
let mut __t315: pci_device_id = unsafe { core::mem::zeroed() };
__t315.vendor = 5410u32;
__t315.device = 256u32;
__t315.subvendor = 5410u32;
__t315.subdevice = 9216u32;
__t315.class = 0u32;
__t315.class_mask = 0u32;
__t315.driver_data = 3u64;
__t315
}, {
let mut __t316: pci_device_id = unsafe { core::mem::zeroed() };
__t316.vendor = 5410u32;
__t316.device = 256u32;
__t316.subvendor = 5410u32;
__t316.subdevice = 9472u32;
__t316.class = 0u32;
__t316.class_mask = 0u32;
__t316.driver_data = 3u64;
__t316
}, {
let mut __t317: pci_device_id = unsafe { core::mem::zeroed() };
__t317.vendor = 5410u32;
__t317.device = 256u32;
__t317.subvendor = 5410u32;
__t317.subdevice = 9728u32;
__t317.class = 0u32;
__t317.class_mask = 0u32;
__t317.driver_data = 5u64;
__t317
}, {
let mut __t318: pci_device_id = unsafe { core::mem::zeroed() };
__t318.vendor = 5410u32;
__t318.device = 256u32;
__t318.subvendor = 5410u32;
__t318.subdevice = 9984u32;
__t318.class = 0u32;
__t318.class_mask = 0u32;
__t318.driver_data = 5u64;
__t318
}, {
let mut __t319: pci_device_id = unsafe { core::mem::zeroed() };
__t319.vendor = 5410u32;
__t319.device = 256u32;
__t319.subvendor = 5410u32;
__t319.subdevice = 12288u32;
__t319.class = 0u32;
__t319.class_mask = 0u32;
__t319.driver_data = 1u64;
__t319
}, {
let mut __t320: pci_device_id = unsafe { core::mem::zeroed() };
__t320.vendor = 5410u32;
__t320.device = 256u32;
__t320.subvendor = 5410u32;
__t320.subdevice = 12544u32;
__t320.class = 0u32;
__t320.class_mask = 0u32;
__t320.driver_data = 1u64;
__t320
}, {
let mut __t321: pci_device_id = unsafe { core::mem::zeroed() };
__t321.vendor = 5410u32;
__t321.device = 256u32;
__t321.subvendor = 5410u32;
__t321.subdevice = 12800u32;
__t321.class = 0u32;
__t321.class_mask = 0u32;
__t321.driver_data = 2u64;
__t321
}, {
let mut __t322: pci_device_id = unsafe { core::mem::zeroed() };
__t322.vendor = 5410u32;
__t322.device = 256u32;
__t322.subvendor = 5410u32;
__t322.subdevice = 13056u32;
__t322.class = 0u32;
__t322.class_mask = 0u32;
__t322.driver_data = 2u64;
__t322
}, {
let mut __t323: pci_device_id = unsafe { core::mem::zeroed() };
__t323.vendor = 5410u32;
__t323.device = 256u32;
__t323.subvendor = 5410u32;
__t323.subdevice = 13312u32;
__t323.class = 0u32;
__t323.class_mask = 0u32;
__t323.driver_data = 3u64;
__t323
}, {
let mut __t324: pci_device_id = unsafe { core::mem::zeroed() };
__t324.vendor = 5410u32;
__t324.device = 256u32;
__t324.subvendor = 5410u32;
__t324.subdevice = 13568u32;
__t324.class = 0u32;
__t324.class_mask = 0u32;
__t324.driver_data = 3u64;
__t324
}, {
let mut __t325: pci_device_id = unsafe { core::mem::zeroed() };
__t325.vendor = 5410u32;
__t325.device = 256u32;
__t325.subvendor = 5410u32;
__t325.subdevice = 15360u32;
__t325.class = 0u32;
__t325.class_mask = 0u32;
__t325.driver_data = 5u64;
__t325
}, {
let mut __t326: pci_device_id = unsafe { core::mem::zeroed() };
__t326.vendor = 5410u32;
__t326.device = 256u32;
__t326.subvendor = 5410u32;
__t326.subdevice = 15616u32;
__t326.class = 0u32;
__t326.class_mask = 0u32;
__t326.driver_data = 5u64;
__t326
}, {
let mut __t327: pci_device_id = unsafe { core::mem::zeroed() };
__t327.vendor = 6489u32;
__t327.device = 40964u32;
__t327.subvendor = 4294967295u32;
__t327.subdevice = 4294967295u32;
__t327.class = 0u32;
__t327.class_mask = 0u32;
__t327.driver_data = 83u64;
__t327
}, {
let mut __t328: pci_device_id = unsafe { core::mem::zeroed() };
__t328.vendor = 4243u32;
__t328.device = 53936u32;
__t328.subvendor = 4294967295u32;
__t328.subdevice = 4294967295u32;
__t328.class = 0u32;
__t328.class_mask = 0u32;
__t328.driver_data = 31u64;
__t328
}, {
let mut __t329: pci_device_id = unsafe { core::mem::zeroed() };
__t329.vendor = 4243u32;
__t329.device = 53584u32;
__t329.subvendor = 4294967295u32;
__t329.subdevice = 4294967295u32;
__t329.class = 0u32;
__t329.class_mask = 0u32;
__t329.driver_data = 30u64;
__t329
}, {
let mut __t330: pci_device_id = unsafe { core::mem::zeroed() };
__t330.vendor = 4243u32;
__t330.device = 53568u32;
__t330.subvendor = 4294967295u32;
__t330.subdevice = 4294967295u32;
__t330.class = 0u32;
__t330.class_mask = 0u32;
__t330.driver_data = 39u64;
__t330
}, {
let mut __t331: pci_device_id = unsafe { core::mem::zeroed() };
__t331.vendor = 4243u32;
__t331.device = 53552u32;
__t331.subvendor = 4294967295u32;
__t331.subdevice = 4294967295u32;
__t331.class = 0u32;
__t331.class_mask = 0u32;
__t331.driver_data = 38u64;
__t331
}, {
let mut __t332: pci_device_id = unsafe { core::mem::zeroed() };
__t332.vendor = 4243u32;
__t332.device = 53872u32;
__t332.subvendor = 4294967295u32;
__t332.subdevice = 4294967295u32;
__t332.class = 0u32;
__t332.class_mask = 0u32;
__t332.driver_data = 39u64;
__t332
}, {
let mut __t333: pci_device_id = unsafe { core::mem::zeroed() };
__t333.vendor = 4243u32;
__t333.device = 53840u32;
__t333.subvendor = 4294967295u32;
__t333.subdevice = 4294967295u32;
__t333.class = 0u32;
__t333.class_mask = 0u32;
__t333.driver_data = 38u64;
__t333
}, {
let mut __t334: pci_device_id = unsafe { core::mem::zeroed() };
__t334.vendor = 4243u32;
__t334.device = 53745u32;
__t334.subvendor = 4294967295u32;
__t334.subdevice = 4294967295u32;
__t334.class = 0u32;
__t334.class_mask = 0u32;
__t334.driver_data = 31u64;
__t334
}, {
let mut __t335: pci_device_id = unsafe { core::mem::zeroed() };
__t335.vendor = 4243u32;
__t335.device = 53744u32;
__t335.subvendor = 4294967295u32;
__t335.subdevice = 4294967295u32;
__t335.class = 0u32;
__t335.class_mask = 0u32;
__t335.driver_data = 30u64;
__t335
}, {
let mut __t336: pci_device_id = unsafe { core::mem::zeroed() };
__t336.vendor = 4243u32;
__t336.device = 53728u32;
__t336.subvendor = 4294967295u32;
__t336.subdevice = 4294967295u32;
__t336.class = 0u32;
__t336.class_mask = 0u32;
__t336.driver_data = 39u64;
__t336
}, {
let mut __t337: pci_device_id = unsafe { core::mem::zeroed() };
__t337.vendor = 4243u32;
__t337.device = 53712u32;
__t337.subvendor = 4294967295u32;
__t337.subdevice = 4294967295u32;
__t337.class = 0u32;
__t337.class_mask = 0u32;
__t337.driver_data = 38u64;
__t337
}, {
let mut __t338: pci_device_id = unsafe { core::mem::zeroed() };
__t338.vendor = 4243u32;
__t338.device = 53664u32;
__t338.subvendor = 4294967295u32;
__t338.subdevice = 4294967295u32;
__t338.class = 0u32;
__t338.class_mask = 0u32;
__t338.driver_data = 39u64;
__t338
}, {
let mut __t339: pci_device_id = unsafe { core::mem::zeroed() };
__t339.vendor = 4243u32;
__t339.device = 53648u32;
__t339.subvendor = 4294967295u32;
__t339.subdevice = 4294967295u32;
__t339.class = 0u32;
__t339.class_mask = 0u32;
__t339.driver_data = 38u64;
__t339
}, {
let mut __t340: pci_device_id = unsafe { core::mem::zeroed() };
__t340.vendor = 4243u32;
__t340.device = 28800u32;
__t340.subvendor = 4294967295u32;
__t340.subdevice = 4294967295u32;
__t340.class = 0u32;
__t340.class_mask = 0u32;
__t340.driver_data = 84u64;
__t340
}, {
let mut __t341: pci_device_id = unsafe { core::mem::zeroed() };
__t341.vendor = 4243u32;
__t341.device = 28891u32;
__t341.subvendor = 4294967295u32;
__t341.subdevice = 4294967295u32;
__t341.class = 0u32;
__t341.class_mask = 0u32;
__t341.driver_data = 84u64;
__t341
}, {
let mut __t342: pci_device_id = unsafe { core::mem::zeroed() };
__t342.vendor = 4243u32;
__t342.device = 28893u32;
__t342.subvendor = 4294967295u32;
__t342.subdevice = 4294967295u32;
__t342.class = 0u32;
__t342.class_mask = 0u32;
__t342.driver_data = 85u64;
__t342
}, {
let mut __t343: pci_device_id = unsafe { core::mem::zeroed() };
__t343.vendor = 4243u32;
__t343.device = 28895u32;
__t343.subvendor = 4294967295u32;
__t343.subdevice = 4294967295u32;
__t343.class = 0u32;
__t343.class_mask = 0u32;
__t343.driver_data = 85u64;
__t343
}, {
let mut __t344: pci_device_id = unsafe { core::mem::zeroed() };
__t344.vendor = 4243u32;
__t344.device = 28898u32;
__t344.subvendor = 4294967295u32;
__t344.subdevice = 4294967295u32;
__t344.class = 0u32;
__t344.class_mask = 0u32;
__t344.driver_data = 86u64;
__t344
}, {
let mut __t345: pci_device_id = unsafe { core::mem::zeroed() };
__t345.vendor = 4243u32;
__t345.device = 28900u32;
__t345.subvendor = 4294967295u32;
__t345.subdevice = 4294967295u32;
__t345.class = 0u32;
__t345.class_mask = 0u32;
__t345.driver_data = 86u64;
__t345
}, {
let mut __t346: pci_device_id = unsafe { core::mem::zeroed() };
__t346.vendor = 4243u32;
__t346.device = 28902u32;
__t346.subvendor = 4294967295u32;
__t346.subdevice = 4294967295u32;
__t346.class = 0u32;
__t346.class_mask = 0u32;
__t346.driver_data = 87u64;
__t346
}, {
let mut __t347: pci_device_id = unsafe { core::mem::zeroed() };
__t347.vendor = 4243u32;
__t347.device = 28903u32;
__t347.subvendor = 4294967295u32;
__t347.subdevice = 4294967295u32;
__t347.class = 0u32;
__t347.class_mask = 0u32;
__t347.driver_data = 87u64;
__t347
}, {
let mut __t348: pci_device_id = unsafe { core::mem::zeroed() };
__t348.vendor = 4243u32;
__t348.device = 28904u32;
__t348.subvendor = 4294967295u32;
__t348.subdevice = 4294967295u32;
__t348.class = 0u32;
__t348.class_mask = 0u32;
__t348.driver_data = 84u64;
__t348
}, {
let mut __t349: pci_device_id = unsafe { core::mem::zeroed() };
__t349.vendor = 4243u32;
__t349.device = 28906u32;
__t349.subvendor = 4294967295u32;
__t349.subdevice = 4294967295u32;
__t349.class = 0u32;
__t349.class_mask = 0u32;
__t349.driver_data = 84u64;
__t349
}, {
let mut __t350: pci_device_id = unsafe { core::mem::zeroed() };
__t350.vendor = 4243u32;
__t350.device = 28908u32;
__t350.subvendor = 4294967295u32;
__t350.subdevice = 4294967295u32;
__t350.class = 0u32;
__t350.class_mask = 0u32;
__t350.driver_data = 85u64;
__t350
}, {
let mut __t351: pci_device_id = unsafe { core::mem::zeroed() };
__t351.vendor = 4243u32;
__t351.device = 28910u32;
__t351.subvendor = 4294967295u32;
__t351.subdevice = 4294967295u32;
__t351.class = 0u32;
__t351.class_mask = 0u32;
__t351.driver_data = 85u64;
__t351
}, {
let mut __t352: pci_device_id = unsafe { core::mem::zeroed() };
__t352.vendor = 5560u32;
__t352.device = 28672u32;
__t352.subvendor = 4294967295u32;
__t352.subdevice = 4294967295u32;
__t352.class = 0u32;
__t352.class_mask = 0u32;
__t352.driver_data = 3u64;
__t352
}, {
let mut __t353: pci_device_id = unsafe { core::mem::zeroed() };
__t353.vendor = 5560u32;
__t353.device = 28673u32;
__t353.subvendor = 4294967295u32;
__t353.subdevice = 4294967295u32;
__t353.class = 0u32;
__t353.class_mask = 0u32;
__t353.driver_data = 2u64;
__t353
}, {
let mut __t354: pci_device_id = unsafe { core::mem::zeroed() };
__t354.vendor = 5560u32;
__t354.device = 28674u32;
__t354.subvendor = 4294967295u32;
__t354.subdevice = 4294967295u32;
__t354.class = 0u32;
__t354.class_mask = 0u32;
__t354.driver_data = 1u64;
__t354
}, {
let mut __t355: pci_device_id = unsafe { core::mem::zeroed() };
__t355.vendor = 4328u32;
__t355.device = 33166u32;
__t355.subvendor = 4294967295u32;
__t355.subdevice = 4294967295u32;
__t355.class = 0u32;
__t355.class_mask = 0u32;
__t355.driver_data = 30u64;
__t355
}, {
let mut __t356: pci_device_id = unsafe { core::mem::zeroed() };
__t356.vendor = 5560u32;
__t356.device = 28681u32;
__t356.subvendor = 4294967295u32;
__t356.subdevice = 4294967295u32;
__t356.class = 0u32;
__t356.class_mask = 0u32;
__t356.driver_data = 3u64;
__t356
}, {
let mut __t357: pci_device_id = unsafe { core::mem::zeroed() };
__t357.vendor = 5560u32;
__t357.device = 28682u32;
__t357.subvendor = 4294967295u32;
__t357.subdevice = 4294967295u32;
__t357.class = 0u32;
__t357.class_mask = 0u32;
__t357.driver_data = 2u64;
__t357
}, {
let mut __t358: pci_device_id = unsafe { core::mem::zeroed() };
__t358.vendor = 5560u32;
__t358.device = 28683u32;
__t358.subvendor = 4294967295u32;
__t358.subdevice = 4294967295u32;
__t358.class = 0u32;
__t358.class_mask = 0u32;
__t358.driver_data = 1u64;
__t358
}, {
let mut __t359: pci_device_id = unsafe { core::mem::zeroed() };
__t359.vendor = 5560u32;
__t359.device = 28684u32;
__t359.subvendor = 4294967295u32;
__t359.subdevice = 4294967295u32;
__t359.class = 0u32;
__t359.class_mask = 0u32;
__t359.driver_data = 3u64;
__t359
}, {
let mut __t360: pci_device_id = unsafe { core::mem::zeroed() };
__t360.vendor = 5560u32;
__t360.device = 28685u32;
__t360.subvendor = 4294967295u32;
__t360.subdevice = 4294967295u32;
__t360.class = 0u32;
__t360.class_mask = 0u32;
__t360.driver_data = 2u64;
__t360
}, {
let mut __t361: pci_device_id = unsafe { core::mem::zeroed() };
__t361.vendor = 5560u32;
__t361.device = 28686u32;
__t361.subvendor = 4294967295u32;
__t361.subdevice = 4294967295u32;
__t361.class = 0u32;
__t361.class_mask = 0u32;
__t361.driver_data = 1u64;
__t361
}, {
let mut __t362: pci_device_id = unsafe { core::mem::zeroed() };
__t362.vendor = 5560u32;
__t362.device = 28687u32;
__t362.subvendor = 4294967295u32;
__t362.subdevice = 4294967295u32;
__t362.class = 0u32;
__t362.class_mask = 0u32;
__t362.driver_data = 5u64;
__t362
}, {
let mut __t363: pci_device_id = unsafe { core::mem::zeroed() };
__t363.vendor = 5560u32;
__t363.device = 28690u32;
__t363.subvendor = 4294967295u32;
__t363.subdevice = 4294967295u32;
__t363.class = 0u32;
__t363.class_mask = 0u32;
__t363.driver_data = 90u64;
__t363
}, {
let mut __t364: pci_device_id = unsafe { core::mem::zeroed() };
__t364.vendor = 5560u32;
__t364.device = 28689u32;
__t364.subvendor = 4294967295u32;
__t364.subdevice = 4294967295u32;
__t364.class = 0u32;
__t364.class_mask = 0u32;
__t364.driver_data = 89u64;
__t364
}, {
let mut __t365: pci_device_id = unsafe { core::mem::zeroed() };
__t365.vendor = 5560u32;
__t365.device = 28688u32;
__t365.subvendor = 4294967295u32;
__t365.subdevice = 4294967295u32;
__t365.class = 0u32;
__t365.class_mask = 0u32;
__t365.driver_data = 88u64;
__t365
}, {
let mut __t366: pci_device_id = unsafe { core::mem::zeroed() };
__t366.vendor = 5560u32;
__t366.device = 28691u32;
__t366.subvendor = 4294967295u32;
__t366.subdevice = 4294967295u32;
__t366.class = 0u32;
__t366.class_mask = 0u32;
__t366.driver_data = 91u64;
__t366
}, {
let mut __t367: pci_device_id = unsafe { core::mem::zeroed() };
__t367.vendor = 5560u32;
__t367.device = 28675u32;
__t367.subvendor = 4294967295u32;
__t367.subdevice = 4294967295u32;
__t367.class = 0u32;
__t367.class_mask = 0u32;
__t367.driver_data = 3u64;
__t367
}, {
let mut __t368: pci_device_id = unsafe { core::mem::zeroed() };
__t368.vendor = 5560u32;
__t368.device = 28708u32;
__t368.subvendor = 4294967295u32;
__t368.subdevice = 4294967295u32;
__t368.class = 0u32;
__t368.class_mask = 0u32;
__t368.driver_data = 3u64;
__t368
}, {
let mut __t369: pci_device_id = unsafe { core::mem::zeroed() };
__t369.vendor = 5560u32;
__t369.device = 28709u32;
__t369.subvendor = 4294967295u32;
__t369.subdevice = 4294967295u32;
__t369.class = 0u32;
__t369.class_mask = 0u32;
__t369.driver_data = 2u64;
__t369
}, {
let mut __t370: pci_device_id = unsafe { core::mem::zeroed() };
__t370.vendor = 5560u32;
__t370.device = 28710u32;
__t370.subvendor = 4294967295u32;
__t370.subdevice = 4294967295u32;
__t370.class = 0u32;
__t370.class_mask = 0u32;
__t370.driver_data = 1u64;
__t370
}, {
let mut __t371: pci_device_id = unsafe { core::mem::zeroed() };
__t371.vendor = 38672u32;
__t371.device = 38965u32;
__t371.subvendor = 4116u32;
__t371.subdevice = 665u32;
__t371.class = 0u32;
__t371.class_mask = 0u32;
__t371.driver_data = 17u64;
__t371
}, {
let mut __t372: pci_device_id = unsafe { core::mem::zeroed() };
__t372.vendor = 38672u32;
__t372.device = 39169u32;
__t372.subvendor = 40960u32;
__t372.subdevice = 4096u32;
__t372.class = 0u32;
__t372.class_mask = 0u32;
__t372.driver_data = 1u64;
__t372
}, {
let mut __t373: pci_device_id = unsafe { core::mem::zeroed() };
__t373.vendor = 38672u32;
__t373.device = 39186u32;
__t373.subvendor = 40960u32;
__t373.subdevice = 4096u32;
__t373.class = 0u32;
__t373.class_mask = 0u32;
__t373.driver_data = 1u64;
__t373
}, {
let mut __t374: pci_device_id = unsafe { core::mem::zeroed() };
__t374.vendor = 38672u32;
__t374.device = 39202u32;
__t374.subvendor = 40960u32;
__t374.subdevice = 4096u32;
__t374.class = 0u32;
__t374.class_mask = 0u32;
__t374.driver_data = 1u64;
__t374
}, {
let mut __t375: pci_device_id = unsafe { core::mem::zeroed() };
__t375.vendor = 38672u32;
__t375.device = 39172u32;
__t375.subvendor = 40960u32;
__t375.subdevice = 4096u32;
__t375.class = 0u32;
__t375.class_mask = 0u32;
__t375.driver_data = 1u64;
__t375
}, {
let mut __t376: pci_device_id = unsafe { core::mem::zeroed() };
__t376.vendor = 38672u32;
__t376.device = 39168u32;
__t376.subvendor = 40960u32;
__t376.subdevice = 4096u32;
__t376.class = 0u32;
__t376.class_mask = 0u32;
__t376.driver_data = 1u64;
__t376
}, {
let mut __t377: pci_device_id = unsafe { core::mem::zeroed() };
__t377.vendor = 38672u32;
__t377.device = 39168u32;
__t377.subvendor = 40960u32;
__t377.subdevice = 12290u32;
__t377.class = 0u32;
__t377.class_mask = 0u32;
__t377.driver_data = 94u64;
__t377
}, {
let mut __t378: pci_device_id = unsafe { core::mem::zeroed() };
__t378.vendor = 4699u32;
__t378.device = 37120u32;
__t378.subvendor = 40960u32;
__t378.subdevice = 4096u32;
__t378.driver_data = 1u64;
__t378
}, {
let mut __t379: pci_device_id = unsafe { core::mem::zeroed() };
__t379.vendor = 38672u32;
__t379.device = 39013u32;
__t379.subvendor = 40960u32;
__t379.subdevice = 4096u32;
__t379.class = 0u32;
__t379.class_mask = 0u32;
__t379.driver_data = 1u64;
__t379
}, {
let mut __t380: pci_device_id = unsafe { core::mem::zeroed() };
__t380.vendor = 38672u32;
__t380.device = 39013u32;
__t380.subvendor = 40960u32;
__t380.subdevice = 12290u32;
__t380.class = 0u32;
__t380.class_mask = 0u32;
__t380.driver_data = 17u64;
__t380
}, {
let mut __t381: pci_device_id = unsafe { core::mem::zeroed() };
__t381.vendor = 38672u32;
__t381.device = 39013u32;
__t381.subvendor = 40960u32;
__t381.subdevice = 12292u32;
__t381.class = 0u32;
__t381.class_mask = 0u32;
__t381.driver_data = 18u64;
__t381
}, {
let mut __t382: pci_device_id = unsafe { core::mem::zeroed() };
__t382.vendor = 4699u32;
__t382.device = 37120u32;
__t382.subvendor = 40960u32;
__t382.subdevice = 4096u32;
__t382.class = 0u32;
__t382.class_mask = 0u32;
__t382.driver_data = 1u64;
__t382
}, {
let mut __t383: pci_device_id = unsafe { core::mem::zeroed() };
__t383.vendor = 32902u32;
__t383.device = 11878u32;
__t383.subvendor = 4294967295u32;
__t383.subdevice = 4294967295u32;
__t383.class = 0u32;
__t383.class_mask = 0u32;
__t383.driver_data = 92u64;
__t383
}, {
let mut __t384: pci_device_id = unsafe { core::mem::zeroed() };
__t384.vendor = 4277u32;
__t384.device = 49153u32;
__t384.subvendor = 4294967295u32;
__t384.subdevice = 4294967295u32;
__t384.class = 0u32;
__t384.class_mask = 0u32;
__t384.driver_data = 93u64;
__t384
}, {
let mut __t385: pci_device_id = unsafe { core::mem::zeroed() };
__t385.vendor = 5348u32;
__t385.device = 5642u32;
__t385.subvendor = 4294967295u32;
__t385.subdevice = 4294967295u32;
__t385.class = 0u32;
__t385.class_mask = 0u32;
__t385.driver_data = 95u64;
__t385
}, {
let mut __t386: pci_device_id = unsafe { core::mem::zeroed() };
__t386.vendor = 21362u32;
__t386.device = 26738u32;
__t386.subvendor = 4294967295u32;
__t386.subdevice = 4294967295u32;
__t386.class = 0u32;
__t386.class_mask = 0u32;
__t386.driver_data = 17u64;
__t386
}, {
let mut __t387: pci_device_id = unsafe { core::mem::zeroed() };
__t387.vendor = 17224u32;
__t387.device = 13395u32;
__t387.subvendor = 4294967295u32;
__t387.subdevice = 4294967295u32;
__t387.class = 0u32;
__t387.class_mask = 0u32;
__t387.driver_data = 18u64;
__t387
}, {
let mut __t388: pci_device_id = unsafe { core::mem::zeroed() };
__t388.vendor = 17224u32;
__t388.device = 20550u32;
__t388.subvendor = 4294967295u32;
__t388.subdevice = 4294967295u32;
__t388.class = 0u32;
__t388.class_mask = 0u32;
__t388.driver_data = 17u64;
__t388
}, {
let mut __t389: pci_device_id = unsafe { core::mem::zeroed() };
__t389.vendor = 17224u32;
__t389.device = 29043u32;
__t389.subvendor = 4294967295u32;
__t389.subdevice = 4294967295u32;
__t389.class = 0u32;
__t389.class_mask = 0u32;
__t389.driver_data = 18u64;
__t389
}, {
let mut __t390: pci_device_id = unsafe { core::mem::zeroed() };
__t390.vendor = 7168u32;
__t390.device = 12883u32;
__t390.subvendor = 4294967295u32;
__t390.subdevice = 4294967295u32;
__t390.class = 0u32;
__t390.class_mask = 0u32;
__t390.driver_data = 102u64;
__t390
}, {
let mut __t391: pci_device_id = unsafe { core::mem::zeroed() };
__t391.vendor = 7168u32;
__t391.device = 13424u32;
__t391.subvendor = 4294967295u32;
__t391.subdevice = 4294967295u32;
__t391.class = 0u32;
__t391.class_mask = 0u32;
__t391.driver_data = 103u64;
__t391
}, {
let mut __t392: pci_device_id = unsafe { core::mem::zeroed() };
__t392.vendor = 7168u32;
__t392.device = 14419u32;
__t392.subvendor = 4294967295u32;
__t392.subdevice = 4294967295u32;
__t392.class = 0u32;
__t392.class_mask = 0u32;
__t392.driver_data = 104u64;
__t392
}, {
let mut __t393: pci_device_id = unsafe { core::mem::zeroed() };
__t393.vendor = 4332u32;
__t393.device = 33130u32;
__t393.subvendor = 4294967295u32;
__t393.subdevice = 4294967295u32;
__t393.class = 0u32;
__t393.class_mask = 0u32;
__t393.driver_data = 1u64;
__t393
}, {
let mut __t394: pci_device_id = unsafe { core::mem::zeroed() };
__t394.vendor = 4332u32;
__t394.device = 33131u32;
__t394.subvendor = 4294967295u32;
__t394.subdevice = 4294967295u32;
__t394.class = 0u32;
__t394.class_mask = 0u32;
__t394.driver_data = 1u64;
__t394
}, {
let mut __t395: pci_device_id = unsafe { core::mem::zeroed() };
__t395.vendor = 5281u32;
__t395.device = 8u32;
__t395.subvendor = 4294967295u32;
__t395.subdevice = 4294967295u32;
__t395.class = 0u32;
__t395.class_mask = 0u32;
__t395.driver_data = 9u64;
__t395
}, {
let mut __t396: pci_device_id = unsafe { core::mem::zeroed() };
__t396.vendor = 7209u32;
__t396.device = 4356u32;
__t396.subvendor = 4294967295u32;
__t396.subdevice = 4294967295u32;
__t396.driver_data = 96u64;
__t396
}, {
let mut __t397: pci_device_id = unsafe { core::mem::zeroed() };
__t397.vendor = 7209u32;
__t397.device = 4360u32;
__t397.subvendor = 4294967295u32;
__t397.subdevice = 4294967295u32;
__t397.driver_data = 97u64;
__t397
}, {
let mut __t398: pci_device_id = unsafe { core::mem::zeroed() };
__t398.vendor = 7209u32;
__t398.device = 4370u32;
__t398.subvendor = 4294967295u32;
__t398.subdevice = 4294967295u32;
__t398.driver_data = 98u64;
__t398
}, {
let mut __t399: pci_device_id = unsafe { core::mem::zeroed() };
__t399.vendor = 7209u32;
__t399.device = 4612u32;
__t399.subvendor = 4294967295u32;
__t399.subdevice = 4294967295u32;
__t399.driver_data = 99u64;
__t399
}, {
let mut __t400: pci_device_id = unsafe { core::mem::zeroed() };
__t400.vendor = 7209u32;
__t400.device = 4616u32;
__t400.subvendor = 4294967295u32;
__t400.subdevice = 4294967295u32;
__t400.driver_data = 100u64;
__t400
}, {
let mut __t401: pci_device_id = unsafe { core::mem::zeroed() };
__t401.vendor = 7209u32;
__t401.device = 4626u32;
__t401.subvendor = 4294967295u32;
__t401.subdevice = 4294967295u32;
__t401.driver_data = 101u64;
__t401
}, {
let mut __t402: pci_device_id = unsafe { core::mem::zeroed() };
__t402.vendor = 5633u32;
__t402.device = 2048u32;
__t402.subvendor = 4294967295u32;
__t402.subdevice = 4294967295u32;
__t402.driver_data = 12u64;
__t402
}, {
let mut __t403: pci_device_id = unsafe { core::mem::zeroed() };
__t403.vendor = 5633u32;
__t403.device = 43009u32;
__t403.subvendor = 4294967295u32;
__t403.subdevice = 4294967295u32;
__t403.driver_data = 12u64;
__t403
}, {
let mut __t404: pci_device_id = unsafe { core::mem::zeroed() };
__t404.vendor = 7439u32;
__t404.device = 33360u32;
__t404.subvendor = 4294967295u32;
__t404.subdevice = 4294967295u32;
__t404.driver_data = 1u64;
__t404
}, {
let mut __t405: pci_device_id = unsafe { core::mem::zeroed() };
__t405.vendor = 4294967295u32;
__t405.device = 4294967295u32;
__t405.subvendor = 4294967295u32;
__t405.subdevice = 4294967295u32;
__t405.class = 458752u32;
__t405.class_mask = 16776960u32;
__t405.driver_data = 0u64;
__t405
}, {
let mut __t406: pci_device_id = unsafe { core::mem::zeroed() };
__t406.vendor = 4294967295u32;
__t406.device = 4294967295u32;
__t406.subvendor = 4294967295u32;
__t406.subdevice = 4294967295u32;
__t406.class = 459520u32;
__t406.class_mask = 16776960u32;
__t406.driver_data = 0u64;
__t406
}, {
let mut __t407: pci_device_id = unsafe { core::mem::zeroed() };
__t407.vendor = 4294967295u32;
__t407.device = 4294967295u32;
__t407.subvendor = 4294967295u32;
__t407.subdevice = 4294967295u32;
__t407.class = 459264u32;
__t407.class_mask = 16776960u32;
__t407.driver_data = 0u64;
__t407
}, {
let mut __t408: pci_device_id = unsafe { core::mem::zeroed() };
__t408
}]);

#[unsafe(link_section = ".text.slate_distinct._8250_pci.serial8250_io_error_detected")]
extern "C" fn serial8250_io_error_detected(dev: *mut pci_dev, state: u32) -> u32 {
let r#priv: *mut serial_private = pci_get_drvdata(dev) as *mut serial_private;
if state == 3 {
return 4;
}
if !r#priv.is_null() {
pciserial_detach_ports(r#priv);
}
unsafe { pci_disable_device(dev) };
3
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.serial8250_io_slot_reset")]
extern "C" fn serial8250_io_slot_reset(dev: *mut pci_dev) -> u32 {
let rc: i32 = unsafe { pci_enable_device(dev) };
if rc != 0 {
return 4;
}
unsafe { pci_restore_state(dev) };
5
}

#[unsafe(link_section = ".text.slate_distinct._8250_pci.serial8250_io_resume")]
extern "C" fn serial8250_io_resume(dev: *mut pci_dev) {
let r#priv: *mut serial_private = pci_get_drvdata(dev) as *mut serial_private;
if r#priv.is_null() {
return;
}
let new: *mut serial_private = pciserial_init_ports(dev, unsafe { (*r#priv).board });
if !IS_ERR(new as *const ()) {
pci_set_drvdata(dev, new as *mut ());
unsafe { kfree(r#priv as *const ()) };
}
}

static serial8250_err_handler: __SlateConst<pci_error_handlers> = __SlateConst({
let mut __t0: pci_error_handlers = unsafe { core::mem::zeroed() };
__t0.error_detected = Some(serial8250_io_error_detected);
__t0.slot_reset = Some(serial8250_io_slot_reset);
__t0.resume = Some(serial8250_io_resume);
__t0
});

static mut serial_pci_driver: pci_driver = {
let mut __t0: pci_driver = unsafe { core::mem::zeroed() };
__t0.name = (b"serial\0".as_ptr() as *mut u8) as *const u8;
__t0.id_table = unsafe { core::ptr::addr_of!(serial_pci_tbl.0) as *const pci_device_id };
__t0.probe = Some(pciserial_init_one);
__t0.remove = Some(pciserial_remove_one);
__t0.err_handler = unsafe { core::ptr::addr_of!(serial8250_err_handler.0) };
__t0.driver = {
let mut __t1: device_driver = unsafe { core::mem::zeroed() };
__t1.pm = unsafe { core::ptr::addr_of!(pciserial_pm_ops.0) };
__t1
};
__t0
};

#[unsafe(link_section = ".init.text")]
extern "C" fn serial_pci_driver_init() -> i32 {
unsafe { __pci_register_driver(core::ptr::addr_of_mut!(serial_pci_driver), core::ptr::null_mut::<module>(), c"8250_pci".as_ptr() as *const u8) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial_pci_driver_init_603: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(serial_pci_driver_init)) };

#[unsafe(link_section = ".exit.text")]
extern "C" fn serial_pci_driver_exit() {
unsafe { pci_unregister_driver(core::ptr::addr_of_mut!(serial_pci_driver)) };
}

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_serial_pci_driver_exit: Option<unsafe extern "C" fn()> = Some(serial_pci_driver_exit);

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_604: [u8; 47] = [56, 50, 53, 48, 95, 112, 99, 105, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 116, 116, 121, 47, 115, 101, 114, 105, 97, 108, 47, 56, 50, 53, 48, 47, 56, 50, 53, 48, 95, 112, 99, 105, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_605: [u8; 21] = [56, 50, 53, 48, 95, 112, 99, 105, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_606: [u8; 64] = [56, 50, 53, 48, 95, 112, 99, 105, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 71, 101, 110, 101, 114, 105, 99, 32, 56, 50, 53, 48, 47, 49, 54, 120, 53, 48, 32, 80, 67, 73, 32, 115, 101, 114, 105, 97, 108, 32, 112, 114, 111, 98, 101, 32, 109, 111, 100, 117, 108, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_607: [u8; 35] = [56, 50, 53, 48, 95, 112, 99, 105, 46, 105, 109, 112, 111, 114, 116, 95, 110, 115, 61, 83, 69, 82, 73, 65, 76, 95, 56, 50, 53, 48, 95, 80, 67, 73, 0];

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";
