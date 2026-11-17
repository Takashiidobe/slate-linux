
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
//! Universal Host Controller Interface driver for USB.
//!
//! Maintainer: Alan Stern <stern@rowland.harvard.edu>
//!
//! (C) Copyright 1999 Linus Torvalds
//! (C) Copyright 1999-2002 Johannes Erdfelt, johannes@erdfelt.com
//! (C) Copyright 1999 Randy Dunlap
//! (C) Copyright 1999 Georg Acher, acher@in.tum.de
//! (C) Copyright 1999 Deti Fliegl, deti@fliegl.de
//! (C) Copyright 1999 Thomas Sailer, sailer@ife.ee.ethz.ch
//! (C) Copyright 1999 Roman Weissgaerber, weissg@vienna.at
//! (C) Copyright 2000 Yggdrasil Computing, Inc. (port of new PCI interface
//!               support from usb-ohci.c by Adam Richter, adam@yggdrasil.com).
//! (C) Copyright 1999 Gregory P. Smith (from usb-ohci.c)
//! (C) Copyright 2004-2007 Alan Stern, stern@rowland.harvard.edu
//!
//! Intel documents this fairly well, and as far as I know there
//! are no royalties or anything like that, but even so there are
//! people who decided that they want to do the same thing in a
//! completely different way.
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
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
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
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! USB directions
//!
//! This bit flag is used in endpoint descriptors' bEndpointAddress field.
//! It's also one of three fields in control requests bRequestType.
//! Runtime PM flag argument bits
//! This is arbitrary.
//! From USB 2.0 spec Table 11-13, offset 7, a hub can
//! have up to 255 ports. The most yet reported is 10.
//! Upcoming hardware might raise that limit.
//! Because the arrays need to add a bit for hub status data, we
//! use 31, so plus one evens out to four bytes.
//! Command register
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
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
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
// Cross-device link
// No such device
// Invalid argument
// Not a typewriter
// Text file busy
// File too large
// No space left on device
// Broken pipe
// Math result not representable
// Invalid system call number
// Out of streams resources
// Protocol error
// Value too large for defined data type
// Illegal byte sequence
// Operation not supported on transport endpoint
// Cannot send after transport endpoint shutdown
// Connection timed out
// Operation now in progress
// Remote I/O error
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
// clamp - return a value clamped to a given range with typechecking
// @val: current value
// @lo: lowest allowable value
// @hi: highest allowable value
//
// This macro checks @val/@lo/@hi to make sure they have compatible
// signedness.
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
// Set the IDR flag and the IDR_FREE tag
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// dev_WARN*() acts like dev_printk(), but with the key difference of
// using WARN/WARN_ONCE to include file/line information and a backtrace.
// Generate SD flag bits
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
// Hub Class feature numbers
// See USB 2.0 spec Table 11-17
// L2 suspend
// wPortStatus bit field
// See USB 2.0 spec Table 11-21
// bits 6 to 7 are reserved
// wPortChange bit field
// See USB 2.0 spec Table 11-22 and USB 2.0 LPM ECN Table-4.10
// Bits 0 to 5 shown, bits 6 to 15 are reserved
// no power switching
// per-port Over-current reporting
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
macro_rules! BIT_MASK {
($nr:expr) => {
UL!() << ($nr) % 64
};
}

macro_rules! BIT_WORD {
($nr:expr) => {
($nr) / 64
};
}

macro_rules! BIT_WORD__2 {
($nr:expr) => {
($nr) / 64
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! CLR_RH_PORTSTAT {
($x:expr, $status:expr, $uhci:expr, $port_addr:expr) => {
$status = uhci_readw(($uhci) as *const uhci_hcd, ($port_addr) as i32) as i32;
let __v: i32 = $status;
let __v2: i32 = __v & !(2048 | 8 | 2 | (8192 | 16384 | 32768));
$status = __v2;
let __v3: i32 = $status;
let __v4: i32 = __v3 & !$x;
$status = __v4;
let __v5: i32 = $status;
let __v6: i32 = __v5 | (2048 | 8 | 2) & $x;
$status = __v6;
uhci_writew(($uhci) as *const uhci_hcd, ($status) as u16, ($port_addr) as i32);
};
}

macro_rules! HCD_HW_ACCESSIBLE {
($hcd:expr) => {
(unsafe { (*$hcd).flags }) & 1u64
};
}

macro_rules! HCD_WAKEUP_PENDING {
($hcd:expr) => {
(unsafe { (*$hcd).flags }) & 16u64
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

macro_rules! LINK_TO_QH {
($uhci:expr, $qh:expr) => {
UHCI_PTR_QH!($uhci) | cpu_to_hc32(($uhci) as *const uhci_hcd, (unsafe { (*$qh).dma_handle }) as u32)
};
}

macro_rules! LINK_TO_TD {
($uhci:expr, $td:expr) => {
cpu_to_hc32(($uhci) as *const uhci_hcd, (unsafe { (*$td).dma_handle }) as u32)
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

/// Scan the URBs in a QH's queue
macro_rules! QH_FINISHED_UNLINKING {
($qh:expr, $uhci:expr) => {
(unsafe { (*$qh).state }) == 2 && unsafe { (*$uhci).frame_number }.wrapping_add(unsafe { (*$uhci).is_stopped }) != unsafe { (*$qh).unlink_frame }
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! SET_RH_PORTSTAT {
($x:expr, $status:expr, $uhci:expr, $port_addr:expr) => {
$status = uhci_readw(($uhci) as *const uhci_hcd, ($port_addr) as i32) as i32;
let __v: i32 = $status;
let __v2: i32 = __v | $x;
$status = __v2;
let __v3: i32 = $status;
let __v4: i32 = __v3 & !(2048 | 8 | 2 | (8192 | 16384 | 32768));
$status = __v4;
uhci_writew(($uhci) as *const uhci_hcd, ($status) as u16, ($port_addr) as i32);
};
}

/// int128, int64, ..., int1 = 2, 3, ..., 9
macro_rules! SKEL_INDEX {
($exponent:expr) => {
9i32.wrapping_sub($exponent)
};
}

macro_rules! UHCI_PTR_QH {
($uhci:expr) => {
cpu_to_hc32(($uhci) as *const uhci_hcd, 2u32)
};
}

macro_rules! UHCI_PTR_TERM {
($uhci:expr) => {
cpu_to_hc32(($uhci) as *const uhci_hcd, 1u32)
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

macro_rules! _AC__2 {
() => {
16045481047390945280u64
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __WRITE_ONCE {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut list_head, $val) };
if true {
break;
}
}
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

macro_rules! ___wait_is_interruptible {
($state:expr) => {
false || ($state) & (1 | 256) != 0
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

macro_rules! __cmp_once__4 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = ($y) as $type;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once__7 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
__v = if __v2 > __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once__2 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = $y;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

/// __ffs - find first set bit in word
/// @word: The word to search
///
/// Undefined if no bit exists, so code should check against 0 first.
macro_rules! __ffs__2 {
($word:expr) => {
if unsafe { __slate_intrinsic_a39e427d3b6c2df4($word) } { ((($word) as u64).trailing_zeros() as i32) as u64 } else { variable__ffs(($word) as u64) }
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

macro_rules! __le16_to_cpu {
($x:expr) => {
($x) as u16
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
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

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
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

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! alloc_hooks {
($_do_alloc:expr) => {
{
let mut __v: *mut ();
let mut __v2: *mut ();
let mut __v3: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
}
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
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
macro_rules! container_of__9 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__38!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
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
__v = ((unsafe { (($ptr) as *const u8).wrapping_offset(-(offsetof__5!() as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__13 {
($ptr:expr) => {
{
let mut __v: *mut urb_priv;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__49!() as isize)) }) as *mut ()) as *mut urb_priv;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__22 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__18!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__24 {
($ptr:expr) => {
{
let mut __v: *mut uhci_qh;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__40!() as isize)) }) as *mut ()) as *mut uhci_qh;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__6 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__13!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__58 {
($ptr:expr) => {
{
let mut __v: *mut uhci_td;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__21!() as isize)) }) as *mut ()) as *mut uhci_td;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__61 {
($ptr:expr) => {
{
let mut __v: *mut uhci_hcd;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__30!() as isize)) }) as *mut ()) as *mut uhci_hcd;
__v
}
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
};
}

macro_rules! kcsan_check_atomic_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 7i32);
};
}

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
};
}

macro_rules! kcsan_mb {
() => {
while false {
}
};
}

macro_rules! kcsan_wmb {
() => {
while false {
}
};
}

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

/// Shortcuts
macro_rules! kmem_cache_zalloc {
($_k:expr, $_flags:expr) => {
alloc_hooks!(unsafe { kmem_cache_alloc_noprof($_k, ($_flags) | 256u32) })
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

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr, $type:ty) => {
container_of__9!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__2 {
($ptr:expr, $type:ty) => {
container_of__6!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__13 {
($ptr:expr) => {
container_of__13!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__24 {
($ptr:expr) => {
container_of__24!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__30 {
($ptr:expr) => {
container_of__58!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__9 {
($ptr:expr, $type:ty) => {
container_of__22!($ptr, $type)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).$member) }) as *const list_head, ($head) as *const list_head)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__2 {
($ptr:expr) => {
list_entry__30!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__26 {
($ptr:expr) => {
list_entry__13!(unsafe { (*$ptr).next })
};
}

/// list_last_entry - get the last element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_last_entry__4 {
($ptr:expr) => {
list_entry__24!(unsafe { (*$ptr).prev })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__6 {
($pos:expr, $member:ident) => {
list_entry__13!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__2 {
($pos:expr, $member:ident) => {
list_entry__30!(unsafe { (*$pos).$member.next })
};
}

/// list_prepare_entry - prepare a pos entry for use in list_for_each_entry_continue()
/// @pos:	the type * to use as a start point
/// @head:	the head of the list
/// @member:	the name of the list_head within the struct.
///
/// Prepares a pos entry for use as a start point in list_for_each_entry_continue().
macro_rules! list_prepare_entry {
($pos:expr, $head:expr) => {
{
let __v: *mut urb_priv = $pos;
let __v2: *mut urb_priv = if !__v.is_null() {
__v
} else {
list_entry__13!($head)
};
__v2
}
};
}

/// list_prev_entry - get the prev element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_prev_entry__5 {
($pos:expr, $member:ident) => {
list_entry__24!(unsafe { (*$pos).$member.prev })
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t__3 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__7!($type, $x, $y)
};
}

macro_rules! might_sleep {
() => {
loop {
might_resched();
if true {
break;
}
}
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__2!($type, $x, $y)
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__3 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__4!($type, $x, $y)
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__18 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, fl_list) as u64
};
}

macro_rules! offsetof__5 {
() => {
584u64
};
}

macro_rules! offsetof__21 {
() => {
core::mem::offset_of!(uhci_td, list) as u64
};
}

macro_rules! offsetof__30 {
() => {
core::mem::offset_of!(uhci_hcd, fsbr_timer) as u64
};
}

macro_rules! offsetof__38 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, node) as u64
};
}

macro_rules! offsetof__40 {
() => {
core::mem::offset_of!(uhci_qh, node) as u64
};
}

macro_rules! offsetof__49 {
() => {
core::mem::offset_of!(urb_priv, node) as u64
};
}

macro_rules! offsetof__13 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, list) as u64
};
}

/// pci_register_driver() must be a macro so KBUILD_MODNAME can be expanded
macro_rules! pci_register_driver__4 {
($driver:expr) => {
unsafe { __pci_register_driver($driver, core::ptr::null_mut::<module>(), c"uhci_hcd".as_ptr() as *const u8) }
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
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

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
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

/// These macros should be used after a dma_map_sg call has been done
/// to get bus addresses of each of the SG entries and their lengths.
/// You should only work with the number of sg entries dma_map_sg
/// returns, or alternatively stop on the first sg_dma_len(sg) which
/// is 0.
macro_rules! sg_dma_address {
($sg:expr) => {
unsafe { (*$sg).dma_address }
};
}

macro_rules! sg_dma_len {
($sg:expr) => {
unsafe { (*$sg).dma_length }
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

/// for TD <info>: (a.k.a. Token)
macro_rules! td_token {
($uhci:expr, $td:expr) => {
hc32_to_cpu(($uhci) as *const uhci_hcd, unsafe { (*$td).token })
};
}

macro_rules! test_bit__3 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

/// time_after - returns true if the time a is after time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Do this with "<0" and ">=0" to only test the sign of the result. A
/// good compiler would generate better code (and a really good compiler
/// wouldn't care). Gcc is currently neither.
///
/// Return: %true is time a is after time b, otherwise %false.
macro_rules! time_after__2 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__3!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_after_eq - returns true if the time a is after or the same as time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is after or the same as time b, otherwise %false.
macro_rules! time_after_eq {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__3!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($a.wrapping_sub($b) as i64) >= 0
}
};
}

macro_rules! timer_container_of__5 {
($callback_timer:expr) => {
container_of__61!($callback_timer)
};
}

/// timer_setup - prepare a timer for first use
/// @timer: the timer in question
/// @callback: the function to call when timer expires
/// @flags: any TIMER_* flags
///
/// Regular timer initialization should use either DEFINE_TIMER() above,
/// or timer_setup(). For timers on the stack, timer_setup_on_stack() must
/// be used and must be balanced with a call to timer_destroy_on_stack().
macro_rules! timer_setup {
($timer:expr, $callback:expr, $flags:expr) => {
__timer_init!($timer, $callback, $flags);
};
}

macro_rules! to_pci_dev {
($n:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(pci_dev) as isize)) }) as *mut ()) as *mut pci_dev;
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

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__2 {
($type:ty) => {
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
__v
}
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__3 {
($type:ty) => {
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == (core::ptr::addr_of_mut!(__v3) as *mut $type);
__v = 1;
__v
}
};
}

/// 1-based
macro_rules! uhci_actual_length {
($ctrl_sts:expr) => {
$ctrl_sts.wrapping_add(1) & 2047
};
}

macro_rules! uhci_dev {
($u:expr) => {
unsafe { (*uhci_to_hcd($u)).self_.controller }
};
}

macro_rules! uhci_expected_length {
($token:expr) => {
(($token) >> 21i32).wrapping_add(1) & 2047
};
}

macro_rules! uhci_explen__2 {
($len:expr) => {
($len.wrapping_sub(1) & 2047) << 21i32
};
}

macro_rules! uhci_explen {
($len:expr) => {
($len.wrapping_sub(1) & 2047) << 21i32
};
}

/// Utility macro for comparing frame numbers
macro_rules! uhci_frame_before_eq__2 {
($f1:expr, $f2:expr) => {
0 <= ($f2.wrapping_sub(($f1) as u32) as i32)
};
}

/// Utility macro for comparing frame numbers
macro_rules! uhci_frame_before_eq {
($f1:expr, $f2:expr) => {
0 <= ($f2.wrapping_sub($f1) as i32)
};
}

macro_rules! uhci_maxerr {
($err:expr) => {
($err) << 27i32
};
}

macro_rules! uhci_packetid {
($token:expr) => {
($token) & 255
};
}

macro_rules! uhci_packetout {
($token:expr) => {
uhci_packetid!($token) != 105
};
}

macro_rules! uhci_status_bits {
($ctrl_sts:expr) => {
($ctrl_sts) & 16121856
};
}

macro_rules! uhci_toggle {
($token:expr) => {
($token) >> 19i32 & 1
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

/// The D0/D1 toggle bits ... USE WITH CAUTION (they're almost hcd-internal)
macro_rules! usb_gettoggle__2 {
($dev:expr, $ep:expr, $out:expr) => {
(unsafe { *((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset((($out) as i32) as isize) }) >> $ep & 1
};
}

macro_rules! usb_packetid {
($pipe:expr) => {
if usb_pipein!($pipe) != 0 { 105i32 } else { 225i32 }
};
}

macro_rules! usb_pipeendpoint {
($pipe:expr) => {
($pipe) >> 15i32 & 15
};
}

macro_rules! usb_pipein {
($pipe:expr) => {
($pipe) & 128
};
}

macro_rules! usb_pipeout {
($pipe:expr) => {
usb_pipein!($pipe) == 0
};
}

macro_rules! usb_settoggle__3 {
($dev:expr, $ep:expr, $out:expr, $bit:expr) => {
unsafe {
*((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset((($out) as i32) as isize) = *((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset((($out) as i32) as isize) & (!(1i32 << $ep) as u32) | ($bit) << $ep;
}
};
}

macro_rules! usb_settoggle__2 {
($dev:expr, $ep:expr, $out:expr, $bit:expr) => {
unsafe {
*((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset((($out) as i32) as isize) = *((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset((($out) as i32) as isize) & (!(1i32 << $ep) as u32) | ((($bit) << $ep) as u32);
}
};
}

macro_rules! wake_up_all {
($x:expr) => {
unsafe { __wake_up($x, 3u32, 0i32, core::ptr::null_mut::<()>()) };
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__might_resched: static_call_key;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_bool: kernel_param_ops;
    static mut usb_hcd_pci_pm_ops: dev_pm_ops;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __compiletime_assert_0();
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_29();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn schedule();
    fn __init_waitqueue_head(arg0: *mut wait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn init_wait_entry(arg0: *mut wait_queue_entry, arg1: i32);
    fn prepare_to_wait_event(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32) -> i64;
    fn finish_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn timer_delete_sync(arg0: *mut timer_list) -> i32;
    fn timer_delete(arg0: *mut timer_list) -> i32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn __kmem_cache_create_args(arg0: *const u8, arg1: u32, arg2: *mut kmem_cache_args, arg3: u32) -> *mut kmem_cache;
    fn kmem_cache_destroy(arg0: *mut kmem_cache);
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn kmem_cache_alloc_noprof(arg0: *mut kmem_cache, arg1: u32) -> *mut ();
    fn kmem_cache_free(arg0: *mut kmem_cache, arg1: *mut ());
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn device_set_wakeup_capable(arg0: *mut device, arg1: bool);
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn synchronize_irq(arg0: u32);
    fn pci_write_config_byte(arg0: *const pci_dev, arg1: i32, arg2: u8) -> i32;
    fn pci_write_config_word(arg0: *const pci_dev, arg1: i32, arg2: u16) -> i32;
    fn __pci_register_driver(arg0: *mut pci_driver, arg1: *mut module, arg2: *const u8) -> i32;
    fn pci_unregister_driver(arg0: *mut pci_driver);
    fn dma_pool_create_node(arg0: *const u8, arg1: *mut device, arg2: u64, arg3: u64, arg4: u64, arg5: i32) -> *mut dma_pool;
    fn dma_pool_destroy(arg0: *mut dma_pool);
    fn dma_pool_alloc(arg0: *mut dma_pool, arg1: u32, arg2: *mut u64) -> *mut ();
    fn dma_pool_free(arg0: *mut dma_pool, arg1: *mut (), arg2: u64);
    fn dma_alloc_attrs(arg0: *mut device, arg1: u64, arg2: *mut u64, arg3: u32, arg4: u64) -> *mut ();
    fn dma_free_attrs(arg0: *mut device, arg1: u64, arg2: *mut (), arg3: u64, arg4: u64);
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn msleep(arg0: u32);
    fn debugfs_remove(arg0: *mut dentry);
    fn debugfs_lookup_and_remove(arg0: *const u8, arg1: *mut dentry);
    fn usb_disabled() -> i32;
    fn usb_hcd_link_urb_to_ep(arg0: *mut usb_hcd, arg1: *mut urb) -> i32;
    fn usb_hcd_check_unlink_urb(arg0: *mut usb_hcd, arg1: *mut urb, arg2: i32) -> i32;
    fn usb_hcd_unlink_urb_from_ep(arg0: *mut usb_hcd, arg1: *mut urb);
    fn usb_hcd_giveback_urb(arg0: *mut usb_hcd, arg1: *mut urb, arg2: i32);
    fn usb_hcd_pci_probe(arg0: *mut pci_dev, arg1: *const hc_driver) -> i32;
    fn usb_hcd_pci_remove(arg0: *mut pci_dev);
    fn usb_hc_died(arg0: *mut usb_hcd);
    fn usb_hcd_poll_rh_status(arg0: *mut usb_hcd);
    fn usb_hcd_start_port_resume(arg0: *mut usb_bus, arg1: i32);
    fn usb_hcd_end_port_resume(arg0: *mut usb_bus, arg1: i32);
    fn usb_calc_bus_time(arg0: i32, arg1: i32, arg2: i32, arg3: i32) -> i64;
    fn usb_root_hub_lost_power(arg0: *mut usb_device);
    fn usb_hcd_resume_root_hub(arg0: *mut usb_hcd);
    fn dmi_get_system_info(arg0: i32) -> *const u8;
    fn __compiletime_assert_531();
    fn __compiletime_assert_534();
    fn __compiletime_assert_537();
    fn __compiletime_assert_553();
    fn __compiletime_assert_554();
    fn __compiletime_assert_557();
    fn __compiletime_assert_558();
    fn uhci_reset_hc(arg0: *mut pci_dev, arg1: u64);
    fn uhci_check_and_reset_hc(arg0: *mut pci_dev, arg1: u64) -> i32;
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_e1c548fa55baa574(arg0: i64) -> bool;
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


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

// kcsan_check_read - check regular read access for races
//
// @ptr: address of access
// @size: size of access
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
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_key {
    func: *mut (),
    __slate_anon_1: __SlateRecord_static_call_types_h_65,
}


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
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


#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
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


// check_add_overflow() - Calculate addition with overflow checking
// @a: first addend
// @b: second addend
// @d: pointer to store sum
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted addition, regardless of whether
// wrap-around occurred.
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
struct qrwlock {
    __slate_anon_0: __SlateRecord_qrwlock_types_h_27,
    wait_lock: qspinlock,
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


fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_0() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).next, list);
if true {
break;
}
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).prev, list);
if true {
break;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
}

fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
}

#[inline(always)]
fn __list_add(new: *mut list_head, prev: *mut list_head, next: *mut list_head) {
if !__list_add_valid(new, prev, next) {
return;
}
unsafe {
(*next).prev = new;
}
unsafe {
(*new).next = next;
}
unsafe {
(*new).prev = prev;
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_2() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, new);
if true {
break;
}
}
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[inline(always)]
fn list_add(new: *mut list_head, head: *mut list_head) {
__list_add(new, head, unsafe { (*head).next });
}

fn list_add_tail(new: *mut list_head, head: *mut list_head) {
__list_add(new, unsafe { (*head).prev }, head);
}

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_5() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, next);
if true {
break;
}
}
}

fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
}

fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_del_init(entry: *mut list_head) {
__list_del_entry(entry);
INIT_LIST_HEAD(entry);
}

fn list_move(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add(list, head);
}

fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_6() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry_reverse - iterate backwards over list of given type.
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry_continue - continue iteration over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
//
// Continue to iterate over list of given type, continuing after
// the current position.
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
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


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn arch_set_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock orb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) ((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
}

#[inline(always)]
fn arch_clear_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock andb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) !((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
}

#[inline(always)]
fn arch_test_and_set_bit(mut nr: i64, mut addr: *mut u64) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock  btsq  {2:r}, ({0:r})\nsetc {1}", in(reg) addr, lateout(reg_byte) __t0, in(reg) nr as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn variable_test_bit(mut nr: i64, mut addr: *const u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut u64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn arch_test_bit(nr: u64, addr: *const u64) -> bool {
let __v0: i32 = if unsafe { __slate_intrinsic_f217c4720f04f6db(nr) } {
constant_test_bit(nr as i64, addr) as i32
} else {
variable_test_bit(nr as i64, addr) as i32
};
__v0 != 0
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


#[inline(always)]
fn variable__ffs(mut word: u64) -> u64 {
unsafe {
let __t0: u64;
core::arch::asm!("tzcnt {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
}

// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
#[inline(always)]
fn instrument_atomic_check_alignment(v: *const (), size: u64) {
if IS_ENABLED!() != 0 {
let mut mask: u32 = size.wrapping_sub(1) as u32;
if IS_ENABLED!() != 0 {
mask = ((mask as u64) & (core::mem::size_of::<__SlateRecord_instrumented_h_66>() as u64).wrapping_sub(1)) as u32;
}
let mut __v2: i64;
let __ret_warn_on: i32 = ((v as u64) & (mask as u64) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("32: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 32b - ., 3; .popsection\n# {0}", const 32 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_instrument_atomic_check_alignment_3, sym __slate_asm_string_instrument_atomic_check_alignment_4, const 67 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("33: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 33b - ., 4; .popsection\n# {0}", const 33 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}

#[inline(always)]
fn instrument_atomic_read(v: *const (), size: u64) {
kasan_check_read(v, size as u32);
kcsan_check_atomic_read!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn instrument_atomic_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_read_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn test_and_set_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
    __slate_bits_0: __slate_bits::__SlateBits121U0,
}


// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
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
    __slate_bits_0: __slate_bits::__SlateBits126U0,
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


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_42: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
    __slate_bits_0: __slate_bits::__SlateBits127U0,
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
struct seccomp_filter {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
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


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
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
struct workqueue_struct {
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
struct rseq {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node_linked {
    node: rb_node,
    prev: *mut rb_node_linked,
    next: *mut rb_node_linked,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
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


#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
struct user_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct load_weight {
    weight: u64,
    inv_weight: u32,
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
struct uprobe_xol_ops {
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
    __slate_bits_0: __slate_bits::__SlateBits281U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
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


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
    __slate_bits_0: __slate_bits::__SlateBits70U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits70U1,
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
    __slate_bits_2: __slate_bits::__SlateBits70U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct wait_queue_entry {
    flags: u32,
    private: *mut (),
    func: Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>,
    entry: list_head,
}


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
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


// explicit shadow
// wait_event_interruptible - sleep until a condition gets true
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
//
// The process is put to sleep (TASK_INTERRUPTIBLE) until the
// @condition evaluates to true or a signal is received.
// The @condition is checked each time the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
//
// The function will return -ERESTARTSYS if it was interrupted by a
// signal and 0 if @condition evaluated to true.
#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
struct completion {
    done: u32,
    wait: swait_queue_head,
}


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
fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
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


#[inline(always)]
fn msecs_to_jiffies(m: u32) -> u64 {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(m) } {
if (m as i32) < 0 {
return 4611686018427387902u64;
}
return _msecs_to_jiffies(m);
} else {
return unsafe { __msecs_to_jiffies(m) };
}
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
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
struct folio_queue {
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
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
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
struct kmem_cache {
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
struct __SlateRecord_mm_types_h_41 {
    f: u64,
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
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
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
struct key_type {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
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
    __slate_bits_0: __slate_bits::__SlateBits408U0,
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


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
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
struct tty_struct {
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
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
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
struct keyring_index_key {
    hash: u64,
    __slate_anon_1: __SlateRecord_key_h_117,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *const u8,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
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


fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
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


fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
    __slate_bits_0: __slate_bits::__SlateBits284U0,
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
if !(__native_word__4!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
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
struct rcu_sync {
    gp_state: i32,
    gp_count: i32,
    gp_wait: wait_queue_head,
    cb_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct cdev {
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
struct io_uring_cmd {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qstr {
    __slate_anon_0: __SlateRecord_dcache_h_50,
    name: *const u8,
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
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
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
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
    __slate_bits_0: __slate_bits::__SlateBits565U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache_args {
    align: u32,
    useroffset: u32,
    usersize: u32,
    freeptr_offset: u32,
    use_freeptr_offset: bool,
    ctor: Option<unsafe extern "C" fn(*mut ())>,
    sheaf_capacity: u32,
}


fn __kmem_cache_create(name: *const u8, size: u32, align: u32, flags: u32, ctor: Option<unsafe extern "C" fn(*mut ())>) -> *mut kmem_cache {
let mut kmem_args: kmem_cache_args = {
let mut __t0: kmem_cache_args = unsafe { core::mem::zeroed() };
__t0.align = align;
__t0.ctor = ctor;
__t0
};
unsafe { __kmem_cache_create_args(name, size, core::ptr::addr_of_mut!(kmem_args), flags) }
}

fn __kmem_cache_default_args(name: *const u8, size: u32, args: *mut kmem_cache_args, flags: u32) -> *mut kmem_cache {
let mut kmem_default_args: kmem_cache_args = {
let mut __t0: kmem_cache_args = unsafe { core::mem::zeroed() };
__t0
};
// Make sure we don't get passed garbage.
let mut __v0: i64;
let __ret_warn_on: i32 = !args.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("390: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 390b - ., 3; .popsection\n# {0}", const 390 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___kmem_cache_default_args_1, sym __slate_asm_string___kmem_cache_default_args_2, const 448 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("391: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 391b - ., 4; .popsection\n# {0}", const 391 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut kmem_cache;
}
unsafe { __kmem_cache_create_args(name, size, core::ptr::addr_of_mut!(kmem_default_args), flags) }
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


// kmem_cache_create - Create a kmem cache.
// @__name: A string which is used in /proc/slabinfo to identify this cache.
// @__object_size: The size of objects to be created in this cache.
// @__args: Optional arguments, see &struct kmem_cache_args. Passing %NULL
// 	    means defaults will be used for all the arguments.
//
// This is currently implemented as a macro using ``_Generic()`` to call
// either the new variant of the function, or a legacy one.
//
// The new variant has 4 parameters:
// ``kmem_cache_create(name, object_size, args, flags)``
//
// See __kmem_cache_create_args() which implements this.
//
// The legacy variant has 5 parameters:
// ``kmem_cache_create(name, object_size, align, flags, ctor)``
//
// The align and ctor parameters map to the respective fields of
// &struct kmem_cache_args
//
// Context: Cannot be called within a interrupt, but can be interrupted.
//
// Return: a pointer to the cache on success, NULL on failure.
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
struct wake_irq {
}


// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list {
    next: u32,
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
struct hotplug_slot {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
}


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
struct kernfs_elem_attr {
    ops: *const kernfs_ops,
    open: *mut kernfs_open_node,
    size: i64,
    notify_next: *mut kernfs_node,
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
    __slate_bits_0: __slate_bits::__SlateBits633U0,
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
struct attribute {
    name: *const u8,
    mode: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_pool {
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


// Permissions on a sysfs file: you didn't miss the 0 prefix did you?
// USER_READABLE >= GROUP_READABLE >= OTHER_READABLE
// USER_WRITABLE >= GROUP_WRITABLE
// OTHER_WRITABLE?  Generally considered a bad idea.
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
    __slate_bits_0: __slate_bits::__SlateBits577U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
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


// One for each parameter, describing how to use it.  Some files do
// multiple of these per line, so can't just use MODULE_INFO.
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


// module_param - typesafe helper for a module/cmdline parameter
// @name: the variable to alter, and exposed parameter name.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// @name becomes the module parameter, or (prefixed by KBUILD_MODNAME and a
// ".") the kernel commandline parameter.  Note that - is changed to _, so
// the user can use "foo-bar=1" even for variable "foo_bar".
//
// @perm is 0 if the variable is not to appear in sysfs, or 0444
// for world-readable, 0644 for root-writable, etc.  Note that if it
// is writable, you may need to use kernel_param_lock() around
// accesses (esp. charp, which can be kfreed when it changes).
//
// The @type is simply pasted to refer to a param_ops_##type and a
// param_check_##type: for convenience many standard types are provided but
// you can create your own by defining those variables.
//
// Standard types are:
// 	byte, hexint, short, ushort, int, uint, long, ulong
// 	charp: a character pointer
// 	bool: a bool, values 0/1, y/n, Y/N.
// 	invbool: the above, only sense-reversed (N = true).
// module_param_named - typesafe helper for a renamed module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @value: the actual lvalue to alter.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// Usually it's a good idea to have variable names and user-exposed names the
// same, but that's harder if the variable must be non-static or is inside a
// structure.  This allows exposure under a different name.
// module_param_cb - general callback for a module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @ops: the set & get operations for this parameter.
// @arg: args for @ops
// @perm: visibility in sysfs.
//
// The ops can have NULL set or get functions.
// This is the fundamental function for registering boot/module parameters.
// All the helper functions
//
// The macros to do compile-time type checking stolen from Jakub
// Jelinek, who IIRC came up with this idea for the 2.4 module init code.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ch9_h_1010 {
    legacy_padding: u32,
    __slate_anon_1: __SlateRecord_ch9_h_1013,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch9_h_1013 {
    __empty_bmSublinkSpeedAttr: __SlateRecord_ch9_h_1013_2,
    bmSublinkSpeedAttr: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch9_h_1013_2 {
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
struct ep_device {
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
// Soft module dependencies. See man modprobe.d for details.
// Example: MODULE_SOFTDEP("pre: module-foo module-bar post: module-baz")
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
// Author(s), use "Name <email>" or just "Name", for multiple
// authors use multiple MODULE_AUTHOR() statements/lines.
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
struct mon_bus {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
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
    __slate_bits_0: __slate_bits::__SlateBits680U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits680U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits680U2,
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
    __slate_bits_3: __slate_bits::__SlateBits680U3,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_phy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_phy_roothub {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gen_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct clk {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct reset_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_archdata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uhci_hcd_c_63 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uhci_hcd_c_63_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uhci_hcd_c_63_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uhci_hcd_c_63_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uhci_hcd_c_63_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uhci_hcd_c_63_6 {
}


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
    pub struct __SlateBits281U0 {
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
    pub struct __SlateBits70U0 {
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
    pub struct __SlateBits70U1 {
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
    pub struct __SlateBits284U0 {
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
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits70U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits121U0 {
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
    pub struct __SlateBits127U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits565U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits633U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits577U0 {
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
    pub struct __SlateBits126U0 {
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
    pub struct __SlateBits408U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits680U0 {
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
    pub struct __SlateBits677U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits680U1 {
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
    pub struct __SlateBits680U2 {
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
    pub struct __SlateBits680U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits788U0 {
        #[bits(1)]
        pub is_b_host: u32,
        #[bits(1)]
        pub b_hnp_enable: u32,
        #[bits(1)]
        pub no_stop_on_short: u32,
        #[bits(1)]
        pub no_sg_constraint: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits784U0 {
        #[bits(1)]
        pub sysfs_files_created: u32,
        #[bits(1)]
        pub ep_devs_created: u32,
        #[bits(1)]
        pub unregistering: u32,
        #[bits(1)]
        pub needs_remote_wakeup: u32,
        #[bits(1)]
        pub needs_altsetting0: u32,
        #[bits(1)]
        pub needs_binding: u32,
        #[bits(1)]
        pub resetting_device: u32,
        #[bits(1)]
        pub authorized: u32,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits778U0 {
        #[bits(1)]
        pub can_submit: u32,
        #[bits(1)]
        pub persist_enabled: u32,
        #[bits(1)]
        pub reset_in_progress: u32,
        #[bits(1)]
        pub have_langid: u32,
        #[bits(1)]
        pub authorized: u32,
        #[bits(1)]
        pub authenticated: u32,
        #[bits(1)]
        pub lpm_capable: u32,
        #[bits(1)]
        pub lpm_devinit_allow: u32,
        #[bits(1)]
        pub usb2_hw_lpm_capable: u32,
        #[bits(1)]
        pub usb2_hw_lpm_besl_capable: u32,
        #[bits(1)]
        pub usb2_hw_lpm_enabled: u32,
        #[bits(1)]
        pub usb2_hw_lpm_allowed: u32,
        #[bits(1)]
        pub usb3_lpm_u1_enabled: u32,
        #[bits(1)]
        pub usb3_lpm_u2_enabled: u32,
        #[bits(2, access = na)]
        pub __slate_pad_14: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits778U1 {
        #[bits(1)]
        pub do_remote_wakeup: u32,
        #[bits(1)]
        pub reset_resume: u32,
        #[bits(1)]
        pub port_is_suspended: u32,
        #[bits(1)]
        pub offload_pm_locked: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits778U2 {
        #[bits(1)]
        pub use_generic_driver: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits796U0 {
        #[bits(1)]
        pub poisoned: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits800U0 {
        #[bits(1)]
        pub rh_registered: u32,
        #[bits(1)]
        pub rh_pollable: u32,
        #[bits(1)]
        pub msix_enabled: u32,
        #[bits(1)]
        pub msi_enabled: u32,
        #[bits(1)]
        pub skip_phy_initialization: u32,
        #[bits(1)]
        pub uses_new_polling: u32,
        #[bits(1)]
        pub has_tt: u32,
        #[bits(1)]
        pub amd_resume_bug: u32,
        #[bits(1)]
        pub can_do_streams: u32,
        #[bits(1)]
        pub tpl_support: u32,
        #[bits(1)]
        pub cant_recv_wakeups: u32,
        #[bits(5, access = na)]
        pub __slate_pad_11: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits807U0 {
        #[bits(1)]
        pub initial_toggle: u32,
        #[bits(1)]
        pub needs_fixup: u32,
        #[bits(1)]
        pub is_stopped: u32,
        #[bits(1)]
        pub wait_expired: u32,
        #[bits(1)]
        pub bandwidth_reserved: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits810U0 {
        #[bits(1)]
        pub scan_in_progress: u32,
        #[bits(1)]
        pub need_rescan: u32,
        #[bits(1)]
        pub dead: u32,
        #[bits(1)]
        pub RD_enable: u32,
        #[bits(1)]
        pub is_initialized: u32,
        #[bits(1)]
        pub fsbr_is_on: u32,
        #[bits(1)]
        pub fsbr_is_wanted: u32,
        #[bits(1)]
        pub fsbr_expiring: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits810U1 {
        #[bits(1)]
        pub oc_low: u32,
        #[bits(1)]
        pub wait_for_hp: u32,
        #[bits(1)]
        pub big_endian_mmio: u32,
        #[bits(1)]
        pub big_endian_desc: u32,
        #[bits(1)]
        pub is_aspeed: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits438U0 {
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
    pub struct __SlateBits727U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits722U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits722U1 {
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
    pub struct __SlateBits722U2 {
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
    pub struct __SlateBits722U3 {
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
    pub struct __SlateBits722U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits812U0 {
        #[bits(1)]
        pub fsbr: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
struct device_link {
    supplier: *mut device,
    s_node: list_head,
    consumer: *mut device,
    c_node: list_head,
    link_dev: device,
    status: i32,
    flags: u32,
    rpm_active: refcount_struct,
    kref: kref,
    rm_work: work_struct,
    supplier_preactivated: bool,
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
    __slate_bits_0: __slate_bits::__SlateBits677U0,
}


fn device_may_wakeup(dev: *mut device) -> bool {
((unsafe { (*dev).power.__slate_bits_0.__get_can_wakeup() }) as i32) != 0 && !unsafe { (*dev).power.wakeup }.is_null()
}

fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
}

#[inline(always)]
fn __outb(mut value: u8, mut port: u16) {
unsafe {
core::arch::asm!("outb %al, %dx", in("ax") (value as u8) as u32, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[inline(always)]
fn __outw(mut value: u16, mut port: u16) {
unsafe {
core::arch::asm!("outw %ax, %dx", in("ax") value as u16, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[inline(always)]
fn __inw(mut port: u16) -> u16 {
let mut value: u16 = 0;
unsafe {
let __t0: u16;
core::arch::asm!("inw %dx, %ax", lateout("ax") __t0, in("dx") port as u16, options(nostack, att_syntax));
value = unsafe { core::mem::transmute::<u16, u16>(__t0 as u16) };
}
value
}

#[inline(always)]
fn __outl(mut value: u32, mut port: u16) {
unsafe {
core::arch::asm!("outl %eax, %dx", in("ax") value as u32, in("dx") port as u16, options(nostack, att_syntax));
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
    __slate_bits_0: __slate_bits::__SlateBits727U0,
    kobj: kobject,
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
    __slate_bits_0: __slate_bits::__SlateBits722U0,
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
    __slate_bits_1: __slate_bits::__SlateBits722U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits722U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits722U3,
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
    __slate_bits_4: __slate_bits::__SlateBits722U4,
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
    __slate_bits_0: __slate_bits::__SlateBits438U0,
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


core::arch::global_asm!(".section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_uhci_hcd__562_930_uhci_hcd_init6:\t\t\t\n.long\t{0} - .\t\n.previous\t\t\t\t\t\n", sym uhci_hcd_init, options(att_syntax));

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


#[repr(C)]
#[derive(Clone, Copy)]
struct sg_table {
    sgl: *mut scatterlist,
    nents: u32,
    orig_nents: u32,
}


// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
fn __sg_flags(sg: *mut scatterlist) -> u32 {
((unsafe { (*sg).page_link }) & (1 | 2)) as u32
}

fn sg_chain_ptr(sg: *mut scatterlist) -> *mut scatterlist {
((unsafe { (*sg).page_link }) & !(1 | 2)) as *mut scatterlist
}

fn sg_is_chain(sg: *mut scatterlist) -> bool {
(__sg_flags(sg) as u64) & 1 != 0
}

fn sg_is_last(sg: *mut scatterlist) -> bool {
(__sg_flags(sg) as u64) & 2 != 0
}

fn sg_next(mut sg: *mut scatterlist) -> *mut scatterlist {
if sg_is_last(sg) {
return core::ptr::null_mut::<scatterlist>();
}
let __v0: *mut scatterlist = sg;
sg = __v0.wrapping_offset(1 as isize);
if unlikely!(sg_is_chain(sg)) != 0 {
sg = sg_chain_ptr(sg);
}
sg
}

fn dma_pool_create(name: *const u8, dev: *mut device, size: u64, align: u64, boundary: u64) -> *mut dma_pool {
unsafe { dma_pool_create_node(name, dev, size, align, boundary, 1i32.wrapping_neg()) }
}

fn dma_pool_zalloc(pool: *mut dma_pool, mem_flags: u32, handle: *mut u64) -> *mut () {
unsafe { dma_pool_alloc(pool, mem_flags | (BIT!(8i32) as u32), handle) }
}

// PCI_DEVICE_CLASS - macro used to describe a specific PCI device class
// @dev_class: the class, subclass, prog-if triple for this device
// @dev_class_mask: the class mask for this device
//
// This macro is used to create a struct pci_device_id that matches a
// specific PCI class.  The vendor, device, subvendor, and subdevice
// fields will be set to PCI_ANY_ID.
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
fn pci_get_drvdata(pdev: *mut pci_dev) -> *mut () {
dev_get_drvdata((unsafe { core::ptr::addr_of_mut!((*pdev).dev) }) as *const device)
}

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
fn dma_alloc_coherent(dev: *mut device, size: u64, dma_handle: *mut u64, gfp: u32) -> *mut () {
unsafe { dma_alloc_attrs(dev, size, dma_handle, gfp, if gfp & (BIT!(13i32) as u32) != 0 { 256u64 } else { 0u64 }) }
}

fn dma_free_coherent(dev: *mut device, size: u64, cpu_addr: *mut (), dma_handle: u64) {
unsafe { dma_free_attrs(dev, size, cpu_addr, dma_handle, 0u64) };
return;
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


// to device
// to host
// Status Type
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_device_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bcdUSB: u16,
    bDeviceClass: u8,
    bDeviceSubClass: u8,
    bDeviceProtocol: u8,
    bMaxPacketSize0: u8,
    idVendor: u16,
    idProduct: u16,
    bcdDevice: u16,
    iManufacturer: u8,
    iProduct: u8,
    iSerialNumber: u8,
    bNumConfigurations: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_config_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wTotalLength: u16,
    bNumInterfaces: u8,
    bConfigurationValue: u8,
    iConfiguration: u8,
    bmAttributes: u8,
    bMaxPower: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bInterfaceNumber: u8,
    bAlternateSetting: u8,
    bNumEndpoints: u8,
    bInterfaceClass: u8,
    bInterfaceSubClass: u8,
    bInterfaceProtocol: u8,
    iInterface: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_endpoint_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bEndpointAddress: u8,
    bmAttributes: u8,
    wMaxPacketSize: u16,
    bInterval: u8,
    bRefresh: u8,
    bSynchAddress: u8,
}


// Endpoints
// in bEndpointAddress
// in bmAttributes
// The USB 3.0 spec redefines bits 5:4 of bmAttributes as interrupt ep type.
fn usb_endpoint_type(epd: *const usb_endpoint_descriptor) -> i32 {
((unsafe { (*epd).bmAttributes }) as i32) & 3
}

fn usb_endpoint_dir_in(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bEndpointAddress }) as i32) & 128 == 128) as i32
}

fn usb_endpoint_maxp(epd: *const usb_endpoint_descriptor) -> i32 {
(__le16_to_cpu!(unsafe { (*epd).wMaxPacketSize }) as i32) & 2047
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_eusb2_isoc_ep_comp_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wMaxPacketSize: u16,
    dwBytesPerInterval: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ssp_isoc_ep_comp_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wReseved: u16,
    dwBytesPerInterval: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ss_ep_comp_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bMaxBurst: u8,
    bmAttributes: u8,
    wBytesPerInterval: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface_assoc_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bFirstInterface: u8,
    bInterfaceCount: u8,
    bFunctionClass: u8,
    bFunctionSubClass: u8,
    bFunctionProtocol: u8,
    iFunction: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_bos_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wTotalLength: u16,
    bNumDeviceCaps: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ext_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bmAttributes: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ss_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bmAttributes: u8,
    wSpeedSupported: u16,
    bFunctionalitySupport: u8,
    bU1devExitLat: u8,
    bU2DevExitLat: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_ss_container_id_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bReserved: u8,
    ContainerID: [u8; 16],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ssp_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bReserved: u8,
    bmAttributes: u32,
    wFunctionalitySupport: u16,
    wReserved: u16,
    __slate_anon_7: __SlateRecord_ch9_h_1010,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_ptm_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_endpoint {
    desc: usb_endpoint_descriptor,
    ss_ep_comp: usb_ss_ep_comp_descriptor,
    ssp_isoc_ep_comp: usb_ssp_isoc_ep_comp_descriptor,
    eusb2_isoc_ep_comp: usb_eusb2_isoc_ep_comp_descriptor,
    urb_list: list_head,
    hcpriv: *mut (),
    ep_dev: *mut ep_device,
    extra: *mut u8,
    extralen: i32,
    enabled: i32,
    streams: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_interface {
    desc: usb_interface_descriptor,
    extralen: i32,
    extra: *mut u8,
    endpoint: *mut usb_host_endpoint,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface {
    altsetting: *mut usb_host_interface,
    cur_altsetting: *mut usb_host_interface,
    num_altsetting: u32,
    intf_assoc: *mut usb_interface_assoc_descriptor,
    minor: i32,
    condition: u32,
    __slate_bits_0: __slate_bits::__SlateBits784U0,
    wireless_status: u32,
    wireless_status_work: work_struct,
    dev: device,
    usb_dev: *mut device,
    reset_ws: work_struct,
}


// this maximum is arbitrary
// USB Resume Timer: Every Host controller driver should drive the resume
// signalling on the bus for the amount of time defined by this macro.
//
// That way we will have a 'stable' behavior among all HCDs supported by Linux.
//
// Note that the USB Specification states we should drive resume for *at least*
// 20 ms, but it doesn't give an upper bound. This creates two possible
// situations which we want to avoid:
//
// (a) sometimes an msleep(20) might expire slightly before 20 ms, which causes
// us to fail USB Electrical Tests, thus failing Certification
//
// (b) Some (many) devices actually need more than 20 ms of resume signalling,
// and while we can argue that's against the USB Specification, we don't have
// control over which devices a certification laboratory will be using for
// certification. If CertLab uses a device which was tested against Windows and
// that happens to have relaxed resume signalling rules, we might fall into
// situations where we fail interoperability and electrical tests.
//
// In order to avoid both conditions, we're using a 40 ms resume timeout, which
// should cope with both LPJ calibration errors and devices not following every
// detail of the USB Specification.
// ms
#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface_cache {
    num_altsetting: u32,
    r#ref: kref,
    altsetting: [usb_host_interface; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_config {
    desc: usb_config_descriptor,
    string: *mut u8,
    intf_assoc: [*mut usb_interface_assoc_descriptor; 16],
    interface: [*mut usb_interface; 32],
    intf_cache: [*mut usb_interface_cache; 32],
    extra: *mut u8,
    extralen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_bos {
    desc: *mut usb_bos_descriptor,
    ext_cap: *mut usb_ext_cap_descriptor,
    ss_cap: *mut usb_ss_cap_descriptor,
    ssp_cap: *mut usb_ssp_cap_descriptor,
    ss_id: *mut usb_ss_container_id_descriptor,
    ptm_cap: *mut usb_ptm_cap_descriptor,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_bus {
    controller: *mut device,
    sysdev: *mut device,
    busnum: i32,
    bus_name: *const u8,
    uses_pio_for_control: u8,
    otg_port: u8,
    __slate_bits_0: __slate_bits::__SlateBits788U0,
    sg_tablesize: u32,
    devnum_next: i32,
    devnum_next_mutex: mutex,
    devmap: [u64; 2],
    root_hub: *mut usb_device,
    hs_companion: *mut usb_bus,
    bandwidth_allocated: i32,
    bandwidth_int_reqs: i32,
    bandwidth_isoc_reqs: i32,
    resuming_ports: u32,
    mon_bus: *mut mon_bus,
    monitored: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb2_lpm_parameters {
    besl: u32,
    timeout: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb3_lpm_parameters {
    mel: u32,
    pel: u32,
    sel: u32,
    timeout: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_device {
    devnum: i32,
    devpath: [u8; 16],
    route: u32,
    state: u32,
    speed: u32,
    rx_lanes: u32,
    tx_lanes: u32,
    ssp_rate: u32,
    tt: *mut usb_tt,
    ttport: i32,
    toggle: [u32; 2],
    parent: *mut usb_device,
    bus: *mut usb_bus,
    ep0: usb_host_endpoint,
    dev: device,
    descriptor: usb_device_descriptor,
    bos: *mut usb_host_bos,
    config: *mut usb_host_config,
    actconfig: *mut usb_host_config,
    ep_in: [*mut usb_host_endpoint; 16],
    ep_out: [*mut usb_host_endpoint; 16],
    rawdescriptors: *mut *mut u8,
    bus_mA: u16,
    portnum: u8,
    level: u8,
    devaddr: u8,
    __slate_bits_0: __slate_bits::__SlateBits778U0,
    string_langid: i32,
    product: *mut u8,
    manufacturer: *mut u8,
    serial: *mut u8,
    filelist: list_head,
    maxchild: i32,
    quirks: u32,
    urbnum: __SlateRecord_types_h_190,
    active_duration: u64,
    connect_time: u64,
    __slate_bits_1: __slate_bits::__SlateBits778U1,
    offload_usage: i32,
    offload_lock: spinlock,
    tunnel_mode: u32,
    usb4_link: *mut device_link,
    slot_id: i32,
    l1_params: usb2_lpm_parameters,
    u1_params: usb3_lpm_parameters,
    u2_params: usb3_lpm_parameters,
    lpm_disable_count: u32,
    hub_delay: u16,
    __slate_bits_2: __slate_bits::__SlateBits778U2,
}


// urb->transfer_flags:
//
// Note: URB_DIR_IN/OUT is automatically set in usb_submit_urb().
// report short reads as errors
// iso-only; use the first unexpired
// slot in the schedule
// Finish bulk OUT with short packet
// The following flags are used internally by usbcore and HCDs
// Transfer from device to host
#[repr(C)]
#[derive(Clone, Copy)]
struct usb_iso_packet_descriptor {
    offset: u32,
    length: u32,
    actual_length: u32,
    status: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_anchor {
    urb_list: list_head,
    wait: wait_queue_head,
    lock: spinlock,
    suspend_wakeups: __SlateRecord_types_h_190,
    __slate_bits_0: __slate_bits::__SlateBits796U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct urb {
    kref: kref,
    unlinked: i32,
    hcpriv: *mut (),
    use_count: __SlateRecord_types_h_190,
    reject: __SlateRecord_types_h_190,
    urb_list: list_head,
    anchor_list: list_head,
    anchor: *mut usb_anchor,
    dev: *mut usb_device,
    ep: *mut usb_host_endpoint,
    pipe: u32,
    stream_id: u32,
    status: i32,
    transfer_flags: u32,
    transfer_buffer: *mut (),
    transfer_dma: u64,
    sg: *mut scatterlist,
    sgt: *mut sg_table,
    num_mapped_sgs: i32,
    num_sgs: i32,
    transfer_buffer_length: u32,
    actual_length: u32,
    setup_packet: *mut u8,
    setup_dma: u64,
    start_frame: i32,
    number_of_packets: i32,
    interval: i32,
    error_count: i32,
    context: *mut (),
    complete: Option<unsafe extern "C" fn(*mut urb)>,
    iso_frame_desc: [usb_iso_packet_descriptor; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct giveback_urb_bh {
    running: bool,
    high_prio: bool,
    lock: spinlock,
    head: list_head,
    bh: work_struct,
    completing_ep: *mut usb_host_endpoint,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_hcd {
    self_: usb_bus,
    kref: kref,
    product_desc: *const u8,
    speed: i32,
    irq_descr: [u8; 24],
    rh_timer: timer_list,
    status_urb: *mut urb,
    wakeup_work: work_struct,
    died_work: work_struct,
    driver: *const hc_driver,
    usb_phy: *mut usb_phy,
    phy_roothub: *mut usb_phy_roothub,
    flags: u64,
    dev_policy: u32,
    __slate_bits_0: __slate_bits::__SlateBits800U0,
    irq: u32,
    regs: *mut (),
    rsrc_start: u64,
    rsrc_len: u64,
    power_budget: u32,
    high_prio_bh: giveback_urb_bh,
    low_prio_bh: giveback_urb_bh,
    address0_mutex: *mut mutex,
    bandwidth_mutex: *mut mutex,
    shared_hcd: *mut usb_hcd,
    primary_hcd: *mut usb_hcd,
    pool: [*mut dma_pool; 4],
    state: i32,
    localmem_pool: *mut gen_pool,
    hcd_priv: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hc_driver {
    description: *const u8,
    product_desc: *const u8,
    hcd_priv_size: u64,
    irq: Option<unsafe extern "C" fn(*mut usb_hcd) -> u32>,
    flags: i32,
    reset: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    start: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    pci_suspend: Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>,
    pci_resume: Option<unsafe extern "C" fn(*mut usb_hcd, pm_message) -> i32>,
    pci_poweroff_late: Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>,
    stop: Option<unsafe extern "C" fn(*mut usb_hcd)>,
    shutdown: Option<unsafe extern "C" fn(*mut usb_hcd)>,
    get_frame_number: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    urb_enqueue: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, u32) -> i32>,
    urb_dequeue: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, i32) -> i32>,
    map_urb_for_dma: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, u32) -> i32>,
    unmap_urb_for_dma: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb)>,
    endpoint_disable: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_host_endpoint)>,
    endpoint_reset: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_host_endpoint)>,
    hub_status_data: Option<unsafe extern "C" fn(*mut usb_hcd, *mut u8) -> i32>,
    hub_control: Option<unsafe extern "C" fn(*mut usb_hcd, u16, u16, u16, *mut u8, u16) -> i32>,
    bus_suspend: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    bus_resume: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    start_port_reset: Option<unsafe extern "C" fn(*mut usb_hcd, u32) -> i32>,
    get_resuming_ports: Option<unsafe extern "C" fn(*mut usb_hcd) -> u64>,
    relinquish_port: Option<unsafe extern "C" fn(*mut usb_hcd, i32)>,
    port_handed_over: Option<unsafe extern "C" fn(*mut usb_hcd, i32) -> i32>,
    clear_tt_buffer_complete: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_host_endpoint)>,
    alloc_dev: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    free_dev: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device)>,
    alloc_streams: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut *mut usb_host_endpoint, u32, u32, u32) -> i32>,
    free_streams: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut *mut usb_host_endpoint, u32, u32) -> i32>,
    add_endpoint: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>,
    drop_endpoint: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>,
    check_bandwidth: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    reset_bandwidth: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device)>,
    address_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, u32) -> i32>,
    enable_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    update_hub_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_tt, u32) -> i32>,
    reset_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    update_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    set_usb2_hw_lpm: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, i32) -> i32>,
    enable_usb3_lpm_timeout: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, u32) -> i32>,
    disable_usb3_lpm_timeout: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, u32) -> i32>,
    find_raw_port_number: Option<unsafe extern "C" fn(*mut usb_hcd, i32) -> i32>,
    port_power: Option<unsafe extern "C" fn(*mut usb_hcd, i32, bool) -> i32>,
    submit_single_step_set_feature: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_tt {
    hub: *mut usb_device,
    multi: i32,
    think_time: u32,
    hcpriv: *mut (),
    lock: spinlock,
    clear_list: list_head,
    clear_work: work_struct,
}


// class requests from the USB 2.0 hub spec, table 11-15
// GetBusState and SetHubDescriptor are optional, omitted
// Run/Stop
// Global Suspend Mode
// Force Global Resume
// Config Flag (sw only)
// Max Packet (0 = 32, 1 = 64)
// Status register
// Interrupt due to IOC
// Interrupt due to error
// Resume Detect
// Host System Error: PCI problems
// Host Controller Process Error:
// the schedule is buggy
// HC Halted
// Interrupt enable register
// Timeout/CRC error enable
// Resume interrupt enable
// Interrupt On Complete enable
// Short packet interrupt enable
// Frame length is exactly 1 ms
// USB port status and control registers
// Current Connect Status
// ("device present")
// Connect Status Change
// Port Enable
// Port Enable Change
// Resume Detect
// Low Speed Device Attached
// Port Reset
// OC and OCC from Intel 430TX and later (not UHCI 1.1d spec)
// Over Current condition
// Over Current Change R/WC
// Suspend
// PCI legacy support register
// only PIRQ enable set
// PCI Intel-specific resume-enable register
// in the frame list [array]
// in an SOF packet
// Periodic scheduling length
// When no queues need Full-Speed Bandwidth Reclamation,
// delay this long before turning FSBR off
// If a queue hasn't advanced after this much time, assume it is stuck
// One role of a QH is to hold a queue of TDs for some endpoint.  One QH goes
// with each endpoint, and qh->element (updated by the HC) is either:
//   - the next unprocessed TD in the endpoint's queue, or
//   - UHCI_PTR_TERM (when there's no more traffic for this endpoint).
//
// The other role of a QH is to serve as a "skeleton" framelist entry, so we
// can easily splice a QH for some endpoint into the schedule at the right
// place.  Then qh->element is UHCI_PTR_TERM.
//
// In the schedule, qh->link maintains a list of QHs seen by the HC:
//     skel1 --> ep1-qh --> ep2-qh --> ... --> skel2 --> ...
//
// qh->node is the software equivalent of qh->link.  The differences
// are that the software list is doubly-linked and QHs in the UNLINKING
// state are on the software list but not the hardware schedule.
//
// For bookkeeping purposes we maintain QHs even for Isochronous endpoints,
// but they never get added to the hardware schedule.
// QH is not being used
// QH has been removed from the
// schedule but the hardware may
// still be using it
// QH is on the schedule
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct uhci_qh {
    link: u32,
    element: u32,
    dma_handle: u64,
    node: list_head,
    hep: *mut usb_host_endpoint,
    udev: *mut usb_device,
    queue: list_head,
    dummy_td: *mut uhci_td,
    post_td: *mut uhci_td,
    iso_packet_desc: *mut usb_iso_packet_descriptor,
    advance_jiffies: u64,
    unlink_frame: u32,
    period: u32,
    phase: i16,
    load: i16,
    iso_frame: u32,
    state: i32,
    r#type: i32,
    skel: i32,
    __slate_bits_0: __slate_bits::__SlateBits807U0,
}


// We need a special accessor for the element pointer because it is
// subject to asynchronous updates by the controller.
// for TD <status>:
// Short Packet Detect
// Low Speed Device
// Isochronous Select
// Interrupt on Complete
// TD Active
// TD Stalled
// Data Buffer Error
// Babble Detected
// CRC/Time Out Error
// Bit Stuff Error
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct uhci_td {
    link: u32,
    status: u32,
    token: u32,
    buffer: u32,
    dma_handle: u64,
    list: list_head,
    frame: i32,
    fl_list: list_head,
}


// We need a special accessor for the control/status word because it is
// subject to asynchronous updates by the controller.
// The following entries refer to sublists of skel_async_qh
#[repr(C)]
#[derive(Clone, Copy)]
struct uhci_hcd {
    io_addr: u64,
    regs: *mut (),
    qh_pool: *mut dma_pool,
    td_pool: *mut dma_pool,
    term_td: *mut uhci_td,
    skelqh: [*mut uhci_qh; 11],
    next_qh: *mut uhci_qh,
    lock: spinlock,
    frame_dma_handle: u64,
    frame: *mut u32,
    frame_cpu: *mut *mut (),
    rh_state: u32,
    auto_stop_time: u64,
    frame_number: u32,
    is_stopped: u32,
    last_iso_frame: u32,
    cur_iso_frame: u32,
    __slate_bits_0: __slate_bits::__SlateBits810U0,
    fsbr_timer: timer_list,
    __slate_bits_1: __slate_bits::__SlateBits810U1,
    port_c_suspend: u64,
    resuming_ports: u64,
    ports_timeout: u64,
    idle_qh_list: list_head,
    rh_numports: i32,
    waitqh: wait_queue_head,
    num_waiting: i32,
    total_load: i32,
    load: [i16; 32],
    clk: *mut clk,
    rsts: *mut reset_control,
    reset_hc: Option<unsafe extern "C" fn(*mut uhci_hcd)>,
    check_and_reset_hc: Option<unsafe extern "C" fn(*mut uhci_hcd) -> i32>,
    configure_hc: Option<unsafe extern "C" fn(*mut uhci_hcd)>,
    resume_detect_interrupts_are_broken: Option<unsafe extern "C" fn(*mut uhci_hcd) -> i32>,
    global_suspend_mode_is_broken: Option<unsafe extern "C" fn(*mut uhci_hcd) -> i32>,
}


fn hcd_to_uhci(hcd: *mut usb_hcd) -> *mut uhci_hcd {
(unsafe { core::ptr::addr_of_mut!((*hcd).hcd_priv) as *mut u64 }) as *mut uhci_hcd
}

fn uhci_to_hcd(uhci: *mut uhci_hcd) -> *mut usb_hcd {
container_of__2!(uhci as *mut (), usb_hcd)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct urb_priv {
    node: list_head,
    urb: *mut urb,
    qh: *mut uhci_qh,
    td_list: list_head,
    __slate_bits_0: __slate_bits::__SlateBits812U0,
}


fn uhci_is_aspeed(uhci: *const uhci_hcd) -> bool {
IS_ENABLED!() != 0 && ((unsafe { (*uhci).__slate_bits_1.__get_is_aspeed() }) as i32) != 0
}

fn uhci_writel(uhci: *const uhci_hcd, val: u32, reg: i32) {
__outl(val, unsafe { (*uhci).io_addr }.wrapping_add(reg as u64) as u16);
}

fn uhci_readw(uhci: *const uhci_hcd, reg: i32) -> u16 {
__inw(unsafe { (*uhci).io_addr }.wrapping_add(reg as u64) as u16)
}

fn uhci_writew(uhci: *const uhci_hcd, val: u16, reg: i32) {
__outw(val, unsafe { (*uhci).io_addr }.wrapping_add(reg as u64) as u16);
}

fn uhci_writeb(uhci: *const uhci_hcd, val: u8, reg: i32) {
__outb(val, unsafe { (*uhci).io_addr }.wrapping_add(reg as u64) as u16);
}

fn cpu_to_hc32(uhci: *const uhci_hcd, x: u32) -> u32 {
x
}

fn hc32_to_cpu(uhci: *const uhci_hcd, x: u32) -> u32 {
x
}

// Version Information
/// for flakey hardware, ignore overcurrent indicators
static mut ignore_oc: bool = unsafe { core::mem::zeroed() };

fn __check_ignore_oc() -> *mut bool {
unsafe { core::ptr::addr_of_mut!(ignore_oc) }
}

static __param_str_ignore_oc: __SlateAlign16<[u8; 19]> = __SlateAlign16([117u8, 104u8, 99u8, 105u8, 95u8, 104u8, 99u8, 100u8, 46u8, 105u8, 103u8, 110u8, 111u8, 114u8, 101u8, 95u8, 111u8, 99u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_ignore_oc: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_ignore_oc.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_bool) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_uhci_hcd_c_63>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_uhci_hcd_c_63_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_uhci_hcd_c_63_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_uhci_hcd_c_63_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_uhci_hcd_c_63_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_uhci_hcd_c_63_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(ignore_oc) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_520: [u8; 33] = [117, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 105, 103, 110, 111, 114, 101, 95, 111, 99, 58, 98, 111, 111, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_521: [u8; 64] = [117, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 61, 105, 103, 110, 111, 114, 101, 95, 111, 99, 58, 105, 103, 110, 111, 114, 101, 32, 104, 97, 114, 100, 119, 97, 114, 101, 32, 111, 118, 101, 114, 99, 117, 114, 114, 101, 110, 116, 32, 105, 110, 100, 105, 99, 97, 116, 105, 111, 110, 115, 0];

// debug = 0, no debugging messages
// debug = 1, dump failed URBs except for stalls
// debug = 2, dump all failed URBs (including stalls)
//            show all queues in /sys/kernel/debug/uhci/[pci_addr]
// debug = 3, show all TDs in URBs when dumping
/// urb_priv
static mut uhci_up_cachep: *mut kmem_cache = unsafe { core::mem::zeroed() };

// Calculate the link pointer DMA value for the first Skeleton QH in a frame.
fn uhci_frame_skel_link(uhci: *mut uhci_hcd, frame: i32) -> u32 {
let mut skelnum: i32 = 0;
// The interrupt queues will be interleaved as evenly as possible.
// There's not much to be done about period-1 interrupts; they have
// to occur in every frame.  But we can schedule period-2 interrupts
// in odd-numbered frames, period-4 interrupts in frames congruent
// to 2 (mod 4), and so on.  This way each frame only has two
// interrupt QHs, which will help spread out bandwidth utilization.
//
// ffs (Find First bit Set) does exactly what we need:
// 1,3,5,...  => ffs = 0 => use period-2 QH = skelqh[8],
// 2,6,10,... => ffs = 1 => use period-4 QH = skelqh[7], etc.
// ffs >= 7 => not on any high-period queue, so use
// 	period-1 QH = skelqh[9].
// Add in UHCI_NUMFRAMES to insure at least one bit is set.
skelnum = 8i32.wrapping_sub(__ffs__2!(frame | 1024) as i32);
if skelnum <= 1 {
skelnum = 9;
}
LINK_TO_QH!(uhci, unsafe { *((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(skelnum as isize) })
}

static mut uhci_debugfs_root: *mut dentry = unsafe { core::mem::zeroed() };

fn lprintk(buf: &u8) {
}

fn uhci_show_qh(uhci: &uhci_hcd, qh: &uhci_qh, buf: &u8, len: i32, space: i32) -> i32 {
0
}

fn uhci_sprint_schedule(uhci: &uhci_hcd, buf: &u8, len: i32) -> i32 {
0
}

fn uhci_set_next_interrupt(uhci: *mut uhci_hcd) {
if (unsafe { (*uhci).is_stopped }) != 0 {
unsafe { mod_timer(core::ptr::addr_of_mut!((*uhci_to_hcd(uhci)).rh_timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies))) };
}
let __v0: u32 = cpu_to_hc32(uhci as *const uhci_hcd, 16777216u32);
let __v1: *mut uhci_td = unsafe { (*uhci).term_td };
unsafe {
(*__v1).status = (*__v1).status | __v0;
}
}

fn uhci_clear_next_interrupt(uhci: *mut uhci_hcd) {
let __v0: u32 = !cpu_to_hc32(uhci as *const uhci_hcd, 16777216u32);
let __v1: *mut uhci_td = unsafe { (*uhci).term_td };
unsafe {
(*__v1).status = (*__v1).status & __v0;
}
}

fn uhci_fsbr_on(uhci: *mut uhci_hcd) {
// The terminating skeleton QH always points back to the first
// FSBR QH.  Make the last async QH point to the terminating
// skeleton QH.
unsafe {
(*uhci).__slate_bits_0.__set_fsbr_is_on(1u32);
}
let lqh: *mut uhci_qh = list_entry!(unsafe { (*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(9 as isize))).node.prev }, uhci_qh);
unsafe {
(*lqh).link = LINK_TO_QH!(uhci, unsafe { *((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(10 as isize) });
}
}

fn uhci_fsbr_off(uhci: *mut uhci_hcd) {
// Remove the link from the last async QH to the terminating
// skeleton QH.
unsafe {
(*uhci).__slate_bits_0.__set_fsbr_is_on(0u32);
}
let lqh: *mut uhci_qh = list_entry!(unsafe { (*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(9 as isize))).node.prev }, uhci_qh);
unsafe {
(*lqh).link = UHCI_PTR_TERM!(uhci);
}
}

fn uhci_add_fsbr(uhci: *mut uhci_hcd, urb: *mut urb) {
let urbp: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
unsafe {
(*urbp).__slate_bits_0.__set_fsbr(1u32);
}
}

fn uhci_urbp_wants_fsbr(uhci: *mut uhci_hcd, urbp: *mut urb_priv) {
if ((unsafe { (*urbp).__slate_bits_0.__get_fsbr() }) as i32) != 0 {
unsafe {
(*uhci).__slate_bits_0.__set_fsbr_is_wanted(1u32);
}
if ((unsafe { (*uhci).__slate_bits_0.__get_fsbr_is_on() }) as i32) == 0 {
uhci_fsbr_on(uhci);
} else {
if ((unsafe { (*uhci).__slate_bits_0.__get_fsbr_expiring() }) as i32) != 0 {
unsafe {
(*uhci).__slate_bits_0.__set_fsbr_expiring(0u32);
}
unsafe { timer_delete(core::ptr::addr_of_mut!((*uhci).fsbr_timer)) };
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_fsbr_timeout")]
extern "C" fn uhci_fsbr_timeout(t: *mut timer_list) {
let uhci: *mut uhci_hcd = timer_container_of__5!(t);
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*uhci).lock) }, flags);
if ((unsafe { (*uhci).__slate_bits_0.__get_fsbr_expiring() }) as i32) != 0 {
unsafe {
(*uhci).__slate_bits_0.__set_fsbr_expiring(0u32);
}
uhci_fsbr_off(uhci);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*uhci).lock) }, flags);
}

fn uhci_alloc_td(uhci: *mut uhci_hcd) -> *mut uhci_td {
let mut dma_handle: u64 = 0;
let mut td: *mut uhci_td = unsafe { core::mem::zeroed() };
td = (unsafe { dma_pool_alloc((*uhci).td_pool, (BIT!(5i32) as u32) | (BIT!(11i32) as u32), core::ptr::addr_of_mut!(dma_handle)) }) as *mut uhci_td;
if td.is_null() {
return core::ptr::null_mut::<uhci_td>();
}
unsafe {
(*td).dma_handle = dma_handle;
}
unsafe {
(*td).frame = 1i32.wrapping_neg();
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*td).list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*td).fl_list) });
td
}

fn uhci_free_td(uhci: *mut uhci_hcd, td: *mut uhci_td) {
if list_empty((unsafe { core::ptr::addr_of_mut!((*td).list) }) as *const list_head) == 0 {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: td %p still in list!\n".as_ptr() as *const u8, dev_driver_string((*uhci_to_hcd(uhci)).self_.controller as *const device), dev_name((*uhci_to_hcd(uhci)).self_.controller as *const device), td) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_uhci_free_td_3, sym __slate_asm_string_uhci_free_td_4, const 127 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, dev_driver_string((*uhci_to_hcd(uhci)).self_.controller as *const device), dev_name((*uhci_to_hcd(uhci)).self_.controller as *const device), td) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*td).fl_list) }) as *const list_head) == 0 {
let mut __v6: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: td %p still in fl_list!\n".as_ptr() as *const u8, dev_driver_string((*uhci_to_hcd(uhci)).self_.controller as *const device), dev_name((*uhci_to_hcd(uhci)).self_.controller as *const device), td) };
let mut __v7: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v7 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v8: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_uhci_free_td_9, sym __slate_asm_string_uhci_free_td_10, const 129 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v8 = bug;
unsafe { __v7.unwrap()(__v8, dev_driver_string((*uhci_to_hcd(uhci)).self_.controller as *const device), dev_name((*uhci_to_hcd(uhci)).self_.controller as *const device), td) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
}
unsafe { dma_pool_free((*uhci).td_pool, td as *mut (), (*td).dma_handle) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_522: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_523: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn uhci_fill_td(uhci: *mut uhci_hcd, td: *mut uhci_td, status: u32, token: u32, buffer: u32) {
unsafe {
(*td).status = cpu_to_hc32(uhci as *const uhci_hcd, status);
}
unsafe {
(*td).token = cpu_to_hc32(uhci as *const uhci_hcd, token);
}
unsafe {
(*td).buffer = cpu_to_hc32(uhci as *const uhci_hcd, buffer);
}
}

fn uhci_add_td_to_urbp(td: *mut uhci_td, urbp: *mut urb_priv) {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*td).list) }, unsafe { core::ptr::addr_of_mut!((*urbp).td_list) });
}

fn uhci_remove_td_from_urbp(td: *mut uhci_td) {
list_del_init(unsafe { core::ptr::addr_of_mut!((*td).list) });
}

fn uhci_insert_td_in_frame_list(uhci: *mut uhci_hcd, td: *mut uhci_td, mut framenum: u32) {
framenum &= 1023u32;
unsafe {
(*td).frame = framenum as i32;
}
// Is there a TD already mapped there?
if !unsafe { *unsafe { (*uhci).frame_cpu }.wrapping_offset(framenum as isize) }.is_null() {
let mut ftd: *mut uhci_td = unsafe { core::mem::zeroed() };
ftd = (unsafe { *unsafe { (*uhci).frame_cpu }.wrapping_offset(framenum as isize) }) as *mut uhci_td;
let ltd: *mut uhci_td = list_entry__9!(unsafe { (*ftd).fl_list.prev }, uhci_td);
list_add_tail(unsafe { core::ptr::addr_of_mut!((*td).fl_list) }, unsafe { core::ptr::addr_of_mut!((*ftd).fl_list) });
unsafe {
(*td).link = (*ltd).link;
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
(*ltd).link = LINK_TO_TD!(uhci, td);
}
} else {
unsafe {
(*td).link = *unsafe { (*uhci).frame }.wrapping_offset(framenum as isize);
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
*unsafe { (*uhci).frame }.wrapping_offset(framenum as isize) = LINK_TO_TD!(uhci, td);
}
unsafe {
*unsafe { (*uhci).frame_cpu }.wrapping_offset(framenum as isize) = td as *mut ();
}
}
}

fn uhci_remove_td_from_frame_list(uhci: *mut uhci_hcd, td: *mut uhci_td) {
// If it's not inserted, don't remove it
if (unsafe { (*td).frame }) == 1i32.wrapping_neg() {
let mut __v0: i64;
let __ret_warn_on: i32 = (list_empty((unsafe { core::ptr::addr_of_mut!((*td).fl_list) }) as *const list_head) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("524: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 524b - ., 3; .popsection\n# {0}", const 524 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_uhci_remove_td_from_frame_list_1, sym __slate_asm_string_uhci_remove_td_from_frame_list_2, const 187 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("525: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 525b - ., 4; .popsection\n# {0}", const 525 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
return;
}
if (unsafe { *unsafe { (*uhci).frame_cpu }.wrapping_offset((*td).frame as isize) }) == (td as *mut ()) {
if list_empty((unsafe { core::ptr::addr_of_mut!((*td).fl_list) }) as *const list_head) != 0 {
unsafe {
*unsafe { (*uhci).frame }.wrapping_offset((*td).frame as isize) = (*td).link;
}
unsafe {
*unsafe { (*uhci).frame_cpu }.wrapping_offset((*td).frame as isize) = core::ptr::null_mut::<()>();
}
} else {
let ntd: *mut uhci_td = list_entry__9!(unsafe { (*td).fl_list.next }, uhci_td);
unsafe {
*unsafe { (*uhci).frame }.wrapping_offset((*td).frame as isize) = LINK_TO_TD!(uhci, ntd);
}
unsafe {
*unsafe { (*uhci).frame_cpu }.wrapping_offset((*td).frame as isize) = ntd as *mut ();
}
}
} else {
let ptd: *mut uhci_td = list_entry__9!(unsafe { (*td).fl_list.prev }, uhci_td);
unsafe {
(*ptd).link = (*td).link;
}
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*td).fl_list) });
unsafe {
(*td).frame = 1i32.wrapping_neg();
}
}

fn uhci_remove_tds_from_frame(uhci: *mut uhci_hcd, mut framenum: u32) {
let mut ftd: *mut uhci_td = unsafe { core::mem::zeroed() };
framenum &= 1023u32;
ftd = (unsafe { *unsafe { (*uhci).frame_cpu }.wrapping_offset(framenum as isize) }) as *mut uhci_td;
if !ftd.is_null() {
let ltd: *mut uhci_td = list_entry__9!(unsafe { (*ftd).fl_list.prev }, uhci_td);
unsafe {
*unsafe { (*uhci).frame }.wrapping_offset(framenum as isize) = (*ltd).link;
}
unsafe {
*unsafe { (*uhci).frame_cpu }.wrapping_offset(framenum as isize) = core::ptr::null_mut::<()>();
}
while list_empty((unsafe { core::ptr::addr_of_mut!((*ftd).fl_list) }) as *const list_head) == 0 {
list_del_init(unsafe { (*ftd).fl_list.prev });
}
}
}

fn uhci_unlink_isochronous_tds(uhci: *mut uhci_hcd, urb: *mut urb) {
let urbp: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
let mut td: *mut uhci_td = unsafe { core::mem::zeroed() };
td = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*urbp).td_list) });
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*urbp).td_list) }, list) == 0 {
uhci_remove_td_from_frame_list(uhci, td);
td = list_next_entry__2!(td, list);
}
}

fn uhci_alloc_qh(uhci: *mut uhci_hcd, udev: *mut usb_device, hep: *mut usb_host_endpoint) -> *mut uhci_qh {
let mut dma_handle: u64 = 0;
let mut qh: *mut uhci_qh = unsafe { core::mem::zeroed() };
qh = dma_pool_zalloc(unsafe { (*uhci).qh_pool }, (BIT!(5i32) as u32) | (BIT!(11i32) as u32), core::ptr::addr_of_mut!(dma_handle)) as *mut uhci_qh;
if qh.is_null() {
return core::ptr::null_mut::<uhci_qh>();
}
unsafe {
(*qh).dma_handle = dma_handle;
}
unsafe {
(*qh).element = UHCI_PTR_TERM!(uhci);
}
unsafe {
(*qh).link = UHCI_PTR_TERM!(uhci);
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*qh).queue) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*qh).node) });
if !udev.is_null() {
// Normal QH
unsafe {
(*qh).r#type = usb_endpoint_type(core::ptr::addr_of_mut!((*hep).desc) as *const usb_endpoint_descriptor);
}
if (unsafe { (*qh).r#type }) != 1 {
unsafe {
(*qh).dummy_td = uhci_alloc_td(uhci);
}
if unsafe { (*qh).dummy_td }.is_null() {
unsafe { dma_pool_free((*uhci).qh_pool, qh as *mut (), dma_handle) };
return core::ptr::null_mut::<uhci_qh>();
}
}
unsafe {
(*qh).state = 1;
}
unsafe {
(*qh).hep = hep;
}
unsafe {
(*qh).udev = udev;
}
unsafe {
(*hep).hcpriv = qh as *mut ();
}
if (unsafe { (*qh).r#type }) == 3 || (unsafe { (*qh).r#type }) == 1 {
unsafe {
(*qh).load = (usb_calc_bus_time(((*udev).speed as u32) as i32, usb_endpoint_dir_in(core::ptr::addr_of_mut!((*hep).desc) as *const usb_endpoint_descriptor), ((*qh).r#type == 1) as i32, usb_endpoint_maxp(core::ptr::addr_of_mut!((*hep).desc) as *const usb_endpoint_descriptor)) / 1000).wrapping_add(1) as i16;
}
}
} else {
// Skeleton QH
unsafe {
(*qh).state = 3;
}
unsafe {
(*qh).r#type = 1i32.wrapping_neg();
}
}
qh
}

fn uhci_free_qh(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
let mut __v0: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*qh).state }) != 1 && !unsafe { (*qh).udev }.is_null()) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("526: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 526b - ., 3; .popsection\n# {0}", const 526 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_uhci_free_qh_1, sym __slate_asm_string_uhci_free_qh_2, const 294 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("527: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 527b - ., 4; .popsection\n# {0}", const 527 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).queue) }) as *const list_head) == 0 {
let mut __v5: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: qh %p list not empty!\n".as_ptr() as *const u8, dev_driver_string((*uhci_to_hcd(uhci)).self_.controller as *const device), dev_name((*uhci_to_hcd(uhci)).self_.controller as *const device), qh) };
let mut __v6: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v6 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v7: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_uhci_free_qh_8, sym __slate_asm_string_uhci_free_qh_9, const 296 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v7 = bug;
unsafe { __v6.unwrap()(__v7, dev_driver_string((*uhci_to_hcd(uhci)).self_.controller as *const device), dev_name((*uhci_to_hcd(uhci)).self_.controller as *const device), qh) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
}
list_del(unsafe { core::ptr::addr_of_mut!((*qh).node) });
if !unsafe { (*qh).udev }.is_null() {
unsafe {
(*(*qh).hep).hcpriv = core::ptr::null_mut::<()>();
}
if !unsafe { (*qh).dummy_td }.is_null() {
uhci_free_td(uhci, unsafe { (*qh).dummy_td });
}
}
unsafe { dma_pool_free((*uhci).qh_pool, qh as *mut (), (*qh).dma_handle) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_528: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn uhci_cleanup_queue(uhci: *mut uhci_hcd, qh: *mut uhci_qh, urb: *mut urb) -> i32 {
let mut ptd: core::mem::MaybeUninit<*mut uhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut uhci_td = core::ptr::addr_of_mut!(ptd) as *mut *mut uhci_td;
let mut purbp: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut urb_priv = core::ptr::addr_of_mut!(purbp) as *mut *mut urb_priv;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut td: core::mem::MaybeUninit<*mut uhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut uhci_td = core::ptr::addr_of_mut!(td) as *mut *mut uhci_td;
let mut urbp: *mut urb_priv = unsafe { core::mem::zeroed() };
unsafe {
urbp = (*urb).hcpriv as *mut urb_priv;
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
ret = 1;
// Isochronous pipes don't use toggles and their TD link pointers
// get adjusted during uhci_urb_dequeue().  But since their queues
// cannot truly be stopped, we have to watch out for dequeues
// occurring after the nominal unlink frame.
if (*qh).r#type == 1 {
ret = (unsafe { (*uhci).frame_number }.wrapping_add((*uhci).is_stopped) != (*qh).unlink_frame) as i32;
} else {
// If the URB isn't first on its queue, adjust the link pointer
// of the last TD in the previous URB.  The toggle doesn't need
// to be saved since this URB can't be executing yet.
if (*qh).queue.next != core::ptr::addr_of_mut!((*urbp).node) {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_8 = list_entry!(unsafe { (*urbp).node.prev }, urb_priv);
let __ret_warn_on: i32 = (list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_8)).td_list) as *const list_head) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("529: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 529b - ., 3; .popsection\n# {0}", const 529 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_uhci_cleanup_queue_17, sym __slate_asm_string_uhci_cleanup_queue_18, const 339 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("530: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 530b - ., 4; .popsection\n# {0}", const 530 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
*__slate_slot_7 = list_entry__2!(unsafe { (*(*__slate_slot_8)).td_list.prev }, uhci_td);
*__slate_slot_10 = list_entry__2!(unsafe { (*urbp).td_list.prev }, uhci_td);
(*(*__slate_slot_7)).link = (*(*__slate_slot_10)).link;
} else {
while false {
}
// If the QH element pointer is UHCI_PTR_TERM then then currently
// executing URB has already been unlinked, so this one isn't it.
let __v3: u32 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*qh).element) as *const u32);
if __v3 != UHCI_PTR_TERM!(uhci) {
(*qh).element = UHCI_PTR_TERM!(uhci);
// Control pipes don't have to worry about toggles
if (*qh).r#type != 0 {
let __ret_warn_on: i32 = (list_empty(core::ptr::addr_of_mut!((*urbp).td_list) as *const list_head) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("532: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 532b - ., 3; .popsection\n# {0}", const 532 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_uhci_cleanup_queue_13, sym __slate_asm_string_uhci_cleanup_queue_14, const 359 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("533: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 533b - ., 4; .popsection\n# {0}", const 533 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// Save the next toggle value
*__slate_slot_10 = list_entry__2!(unsafe { (*urbp).td_list.next }, uhci_td);
(*qh).__slate_bits_0.__set_needs_fixup(1u32);
let __t0 = uhci_toggle!(td_token!(uhci, *__slate_slot_10));
(*qh).__slate_bits_0.__set_initial_toggle(__t0);
}
}
}
}
return ret;
}
}

fn uhci_fixup_toggles(uhci: *mut uhci_hcd, qh: *mut uhci_qh, skip_first: i32) {
let mut urbp: *mut urb_priv = core::ptr::null_mut::<urb_priv>();
let mut td: *mut uhci_td = unsafe { core::mem::zeroed() };
let mut toggle: u32 = ((unsafe { (*qh).__slate_bits_0.__get_initial_toggle() }) as i32) as u32;
let mut pipe: u32 = 0;
// Fixups for a short transfer start with the second URB in the
// queue (the short URB is the first).
if skip_first != 0 {
urbp = list_entry!(unsafe { (*qh).queue.next }, urb_priv);
} else {
let mut __v1: u32;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_534() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*qh).element) as *const u32) };
if __v1 != UHCI_PTR_TERM!(uhci) {
toggle = 2;
}
}
// When starting with the first URB, if the QH element pointer is
// still valid then we know the URB's toggles are okay.
// Fix up the toggle for the URBs in the queue.  Normally this
// loop won't run more than once: When an error or short transfer
// occurs, the queue usually gets emptied.
urbp = list_prepare_entry!(urbp, unsafe { core::ptr::addr_of_mut!((*qh).queue) });
urbp = list_next_entry__6!(urbp, node);
while list_entry_is_head!(urbp, unsafe { core::ptr::addr_of_mut!((*qh).queue) }, node) == 0 {
// If the first TD has the right toggle value, we don't
// need to change any toggles in this URB
td = list_entry__2!(unsafe { (*urbp).td_list.next }, uhci_td);
let __v8: bool = toggle > 1 || uhci_toggle!(td_token!(uhci, td)) == toggle;
if __v8 {
td = list_entry__2!(unsafe { (*urbp).td_list.prev }, uhci_td);
toggle = uhci_toggle!(td_token!(uhci, td)) ^ 1;
// Otherwise all the toggles in the URB have to be switched
} else {
td = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*urbp).td_list) });
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*urbp).td_list) }, list) == 0 {
let __v11: u32 = cpu_to_hc32(uhci as *const uhci_hcd, 524288u32);
unsafe {
(*td).token = (*td).token ^ __v11;
}
toggle ^= 1;
td = list_next_entry__2!(td, list);
}
}
urbp = list_next_entry__6!(urbp, node);
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
pipe = unsafe { (*(*list_entry!(unsafe { (*qh).queue.next }, urb_priv)).urb).pipe };
usb_settoggle__3!(unsafe { (*qh).udev }, pipe >> 15i32 & 15, pipe & 128 == 0, toggle);
unsafe {
(*qh).__slate_bits_0.__set_needs_fixup(0u32);
}
}

fn link_iso(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*qh).node) }, unsafe { core::ptr::addr_of_mut!((*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(1 as isize))).node) });
// Isochronous QHs aren't linked by the hardware
}

fn link_interrupt(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*qh).node) }, unsafe { core::ptr::addr_of_mut!((*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset((*qh).skel as isize))).node) });
let pqh: *mut uhci_qh = list_entry!(unsafe { (*qh).node.prev }, uhci_qh);
unsafe {
(*qh).link = (*pqh).link;
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
(*pqh).link = LINK_TO_QH!(uhci, qh);
}
}

fn link_async(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
let mut pqh: *mut uhci_qh = unsafe { core::mem::zeroed() };
let mut link_to_new_qh: u32 = 0;
// Find the predecessor QH for our new one and insert it in the list.
// The list of QHs is expected to be short, so linear search won't
// take too long.
pqh = list_last_entry__4!(unsafe { core::ptr::addr_of_mut!((*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(9 as isize))).node) });
while list_entry_is_head!(pqh, unsafe { core::ptr::addr_of_mut!((*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(9 as isize))).node) }, node) == 0 {
if (unsafe { (*pqh).skel }) <= unsafe { (*qh).skel } {
break;
}
pqh = list_prev_entry__5!(pqh, node);
}
list_add(unsafe { core::ptr::addr_of_mut!((*qh).node) }, unsafe { core::ptr::addr_of_mut!((*pqh).node) });
// Link it into the schedule
unsafe {
(*qh).link = (*pqh).link;
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
link_to_new_qh = LINK_TO_QH!(uhci, qh);
unsafe {
(*pqh).link = link_to_new_qh;
}
// If this is now the first FSBR QH, link the terminating skeleton
// QH to it.
if (unsafe { (*pqh).skel }) < 21 && (unsafe { (*qh).skel }) >= 21 {
unsafe {
(*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(10 as isize))).link = link_to_new_qh;
}
}
}

fn uhci_activate_qh(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
let mut __v0: i64;
let __ret_warn_on: i32 = (list_empty((unsafe { core::ptr::addr_of_mut!((*qh).queue) }) as *const list_head) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("535: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 535b - ., 3; .popsection\n# {0}", const 535 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_uhci_activate_qh_1, sym __slate_asm_string_uhci_activate_qh_2, const 482 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("536: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 536b - ., 4; .popsection\n# {0}", const 536 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
// Set the element pointer if it isn't set already.
// This isn't needed for Isochronous queues, but it doesn't hurt.
let mut __v5: u32;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_537() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*qh).element) as *const u32) };
if __v5 == UHCI_PTR_TERM!(uhci) {
let urbp: *mut urb_priv = list_entry!(unsafe { (*qh).queue.next }, urb_priv);
let td: *mut uhci_td = list_entry__2!(unsafe { (*urbp).td_list.next }, uhci_td);
unsafe {
(*qh).element = LINK_TO_TD!(uhci, td);
}
}
// Treat the queue as if it has just advanced
unsafe {
(*qh).__slate_bits_0.__set_wait_expired(0u32);
}
unsafe {
(*qh).advance_jiffies = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
if (unsafe { (*qh).state }) == 3 {
return;
}
unsafe {
(*qh).state = 3;
}
// Move the QH from its old list to the correct spot in the appropriate
// skeleton's list
if qh == unsafe { (*uhci).next_qh } {
unsafe {
(*uhci).next_qh = list_entry!(unsafe { (*qh).node.next }, uhci_qh);
}
}
list_del(unsafe { core::ptr::addr_of_mut!((*qh).node) });
if (unsafe { (*qh).skel }) == 1 {
link_iso(uhci, qh);
} else {
if (unsafe { (*qh).skel }) < 9 {
link_interrupt(uhci, qh);
} else {
link_async(uhci, qh);
}
}
}

fn unlink_interrupt(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
let pqh: *mut uhci_qh = list_entry!(unsafe { (*qh).node.prev }, uhci_qh);
unsafe {
(*pqh).link = (*qh).link;
}
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
}
if true {
break;
}
}
}

fn unlink_async(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
let link_to_next_qh: u32 = unsafe { (*qh).link };
let pqh: *mut uhci_qh = list_entry!(unsafe { (*qh).node.prev }, uhci_qh);
unsafe {
(*pqh).link = link_to_next_qh;
}
// If this was the old first FSBR QH, link the terminating skeleton
// QH to the next (new first FSBR) QH.
if (unsafe { (*pqh).skel }) < 21 && (unsafe { (*qh).skel }) >= 21 {
unsafe {
(*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(10 as isize))).link = link_to_next_qh;
}
}
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
}
if true {
break;
}
}
}

fn uhci_unlink_qh(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
if (unsafe { (*qh).state }) == 2 {
return;
}
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*qh).state }) != 3 || unsafe { (*qh).udev }.is_null()) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("538: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 538b - ., 3; .popsection\n# {0}", const 538 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_uhci_unlink_qh_1, sym __slate_asm_string_uhci_unlink_qh_2, const 555 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("539: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 539b - ., 4; .popsection\n# {0}", const 539 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe {
(*qh).state = 2;
}
// Unlink the QH from the schedule and record when we did it
if (unsafe { (*qh).skel }) != 1 {
if (unsafe { (*qh).skel }) < 9 {
unlink_interrupt(uhci, qh);
} else {
unlink_async(uhci, qh);
}
}
uhci_get_current_frame_number(uhci);
unsafe {
(*qh).unlink_frame = (*uhci).frame_number;
}
// Force an interrupt so we know when the QH is fully unlinked
if list_empty((unsafe { core::ptr::addr_of_mut!((*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(0 as isize))).node) }) as *const list_head) != 0 || (unsafe { (*uhci).is_stopped }) != 0 {
uhci_set_next_interrupt(uhci);
}
// Move the QH from its old list to the end of the unlinking list
if qh == unsafe { (*uhci).next_qh } {
unsafe {
(*uhci).next_qh = list_entry!(unsafe { (*qh).node.next }, uhci_qh);
}
}
list_move_tail(unsafe { core::ptr::addr_of_mut!((*qh).node) }, unsafe { core::ptr::addr_of_mut!((*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(0 as isize))).node) });
}

fn uhci_make_qh_idle(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*qh).state }) == 3) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("540: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 540b - ., 3; .popsection\n# {0}", const 540 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_uhci_make_qh_idle_1, sym __slate_asm_string_uhci_make_qh_idle_2, const 588 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("541: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 541b - ., 4; .popsection\n# {0}", const 541 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if qh == unsafe { (*uhci).next_qh } {
unsafe {
(*uhci).next_qh = list_entry!(unsafe { (*qh).node.next }, uhci_qh);
}
}
list_move(unsafe { core::ptr::addr_of_mut!((*qh).node) }, unsafe { core::ptr::addr_of_mut!((*uhci).idle_qh_list) });
unsafe {
(*qh).state = 1;
}
// Now that the QH is idle, its post_td isn't being used
if !unsafe { (*qh).post_td }.is_null() {
uhci_free_td(uhci, unsafe { (*qh).post_td });
unsafe {
(*qh).post_td = core::ptr::null_mut::<uhci_td>();
}
}
// If anyone is waiting for a QH to become idle, wake them up
if (unsafe { (*uhci).num_waiting }) != 0 {
wake_up_all!(unsafe { core::ptr::addr_of_mut!((*uhci).waitqh) });
}
}

fn uhci_highest_load(uhci: *mut uhci_hcd, mut phase: i32, period: i32) -> i32 {
let mut highest_load: i32 = (unsafe { *((*uhci).load.as_mut_ptr() as *mut i16).wrapping_offset(phase as isize) }) as i32;
phase = phase.wrapping_add(period);
while phase < 32 {
highest_load = max_t__3!(i32, highest_load, unsafe { *((*uhci).load.as_mut_ptr() as *mut i16).wrapping_offset(phase as isize) });
phase = phase.wrapping_add(period);
}
highest_load
}

fn uhci_check_bandwidth(uhci: *mut uhci_hcd, qh: *mut uhci_qh) -> i32 {
let mut minimax_load: i32 = 0;
// Find the optimal phase (unless it is already set) and get
// its load value.
if ((unsafe { (*qh).phase }) as i32) >= 0 {
minimax_load = uhci_highest_load(uhci, (unsafe { (*qh).phase }) as i32, (unsafe { (*qh).period }) as i32);
} else {
let mut load: i32 = 0;
let max_phase: i32 = min_t__2!(i32, 32i32, unsafe { (*qh).period });
unsafe {
(*qh).phase = 0;
}
minimax_load = uhci_highest_load(uhci, (unsafe { (*qh).phase }) as i32, (unsafe { (*qh).period }) as i32);
for phase in 1..max_phase {
load = uhci_highest_load(uhci, phase, (unsafe { (*qh).period }) as i32);
if load < minimax_load {
minimax_load = load;
unsafe {
(*qh).phase = phase as i16;
}
}
}
}
// Maximum allowable periodic bandwidth is 90%, or 900 us per frame
if minimax_load.wrapping_add((unsafe { (*qh).load }) as i32) > 900 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, uhci_dev!(uhci) as *const device, c"bandwidth allocation failed: period %d, phase %d, %d + %d us\n".as_ptr() as *const u8, (*qh).period, (*qh).phase as i32, minimax_load, (*qh).load as i32) };
}
return 28i32.wrapping_neg();
}
0
}

fn uhci_reserve_bandwidth(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
let mut i: i32 = 0;
let load: i32 = (unsafe { (*qh).load }) as i32;
let mut p: *mut u8 = c"??".as_ptr() as *mut u8;
i = (unsafe { (*qh).phase }) as i32;
while i < 32 {
let __v0: *mut i16 = unsafe { (*uhci).load.as_mut_ptr() as *mut i16 }.wrapping_offset(i as isize);
unsafe {
*__v0 = (*__v0 as i32).wrapping_add(load) as i16;
}
let __v4: i32 = unsafe { (*uhci).total_load };
unsafe {
(*uhci).total_load = __v4.wrapping_add(load);
}
i = (i as u32).wrapping_add(unsafe { (*qh).period }) as i32;
}
unsafe {
(*uhci_to_hcd(uhci)).self_.bandwidth_allocated = (*uhci).total_load / 32;
}
match unsafe { (*qh).r#type } {
3 => {
let __v9: *mut usb_hcd = uhci_to_hcd(uhci);
let __v10: i32 = unsafe { (*__v9).self_.bandwidth_int_reqs };
unsafe {
(*__v9).self_.bandwidth_int_reqs = __v10.wrapping_add(1);
}
p = c"INT".as_ptr() as *mut u8;
}
1 => {
let __v13: *mut usb_hcd = uhci_to_hcd(uhci);
let __v14: i32 = unsafe { (*__v13).self_.bandwidth_isoc_reqs };
unsafe {
(*__v13).self_.bandwidth_isoc_reqs = __v14.wrapping_add(1);
}
p = c"ISO".as_ptr() as *mut u8;
}
_ => {
}
}
unsafe {
(*qh).__slate_bits_0.__set_bandwidth_reserved(1u32);
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, uhci_dev!(uhci) as *const device, c"%s dev %d ep%02x-%s, period %d, phase %d, %d us\n".as_ptr() as *const u8, c"reserve".as_ptr() as *mut u8, (*(*qh).udev).devnum, (*(*qh).hep).desc.bEndpointAddress as i32, p, (*qh).period, (*qh).phase as i32, load) };
}
}

fn uhci_release_bandwidth(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
let mut i: i32 = 0;
let load: i32 = (unsafe { (*qh).load }) as i32;
let mut p: *mut u8 = c"??".as_ptr() as *mut u8;
i = (unsafe { (*qh).phase }) as i32;
while i < 32 {
let __v0: *mut i16 = unsafe { (*uhci).load.as_mut_ptr() as *mut i16 }.wrapping_offset(i as isize);
unsafe {
*__v0 = (*__v0 as i32).wrapping_sub(load) as i16;
}
let __v4: i32 = unsafe { (*uhci).total_load };
unsafe {
(*uhci).total_load = __v4.wrapping_sub(load);
}
i = (i as u32).wrapping_add(unsafe { (*qh).period }) as i32;
}
unsafe {
(*uhci_to_hcd(uhci)).self_.bandwidth_allocated = (*uhci).total_load / 32;
}
match unsafe { (*qh).r#type } {
3 => {
let __v9: *mut usb_hcd = uhci_to_hcd(uhci);
let __v10: i32 = unsafe { (*__v9).self_.bandwidth_int_reqs };
unsafe {
(*__v9).self_.bandwidth_int_reqs = __v10.wrapping_sub(1);
}
p = c"INT".as_ptr() as *mut u8;
}
1 => {
let __v13: *mut usb_hcd = uhci_to_hcd(uhci);
let __v14: i32 = unsafe { (*__v13).self_.bandwidth_isoc_reqs };
unsafe {
(*__v13).self_.bandwidth_isoc_reqs = __v14.wrapping_sub(1);
}
p = c"ISO".as_ptr() as *mut u8;
}
_ => {
}
}
unsafe {
(*qh).__slate_bits_0.__set_bandwidth_reserved(0u32);
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, uhci_dev!(uhci) as *const device, c"%s dev %d ep%02x-%s, period %d, phase %d, %d us\n".as_ptr() as *const u8, c"release".as_ptr() as *mut u8, (*(*qh).udev).devnum, (*(*qh).hep).desc.bEndpointAddress as i32, p, (*qh).period, (*qh).phase as i32, load) };
}
}

fn uhci_alloc_urb_priv(uhci: *mut uhci_hcd, urb: *mut urb) -> *mut urb_priv {
let mut urbp: *mut urb_priv = unsafe { core::mem::zeroed() };
urbp = kmem_cache_zalloc!(unsafe { uhci_up_cachep }, 2080u32) as *mut urb_priv;
if urbp.is_null() {
return core::ptr::null_mut::<urb_priv>();
}
unsafe {
(*urbp).urb = urb;
}
unsafe {
(*urb).hcpriv = urbp as *mut ();
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*urbp).node) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*urbp).td_list) });
urbp
}

fn uhci_free_urb_priv(uhci: *mut uhci_hcd, urbp: *mut urb_priv) {
let mut td: *mut uhci_td = unsafe { core::mem::zeroed() };
let mut tmp: *mut uhci_td = unsafe { core::mem::zeroed() };
if list_empty((unsafe { core::ptr::addr_of_mut!((*urbp).node) }) as *const list_head) == 0 {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: urb %p still on QH's list!\n".as_ptr() as *const u8, dev_driver_string((*uhci_to_hcd(uhci)).self_.controller as *const device), dev_name((*uhci_to_hcd(uhci)).self_.controller as *const device), (*urbp).urb) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_uhci_free_urb_priv_3, sym __slate_asm_string_uhci_free_urb_priv_4, const 747 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, dev_driver_string((*uhci_to_hcd(uhci)).self_.controller as *const device), dev_name((*uhci_to_hcd(uhci)).self_.controller as *const device), (*urbp).urb) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
td = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*urbp).td_list) });
tmp = list_next_entry__2!(td, list);
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*urbp).td_list) }, list) == 0 {
uhci_remove_td_from_urbp(td);
uhci_free_td(uhci, td);
td = tmp;
tmp = list_next_entry__2!(tmp, list);
}
unsafe { kmem_cache_free(uhci_up_cachep, urbp as *mut ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_546: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn uhci_map_status(status: i32, dir_out: i32) -> i32 {
if status == 0 {
return 0;
}
if status & 131072i32 != 0 {
return 71i32.wrapping_neg();
}
// Bitstuff error
if status & 262144i32 != 0 {
// CRC/Timeout
if dir_out != 0 {
return 71i32.wrapping_neg();
} else {
return 84i32.wrapping_neg();
}
}
if status & 1048576i32 != 0 {
return 75i32.wrapping_neg();
}
// Babble
if status & 2097152i32 != 0 {
return 63i32.wrapping_neg();
}
// Buffer error
if status & 4194304i32 != 0 {
return 32i32.wrapping_neg();
}
// Stalled
0
}

fn uhci_submit_control(uhci: *mut uhci_hcd, urb: *mut urb, qh: *mut uhci_qh) -> i32 {
let mut pktsze: i32 = unsafe { core::mem::zeroed() };
let mut skel: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut i32 = core::ptr::addr_of_mut!(skel) as *mut i32;
let mut urbp: *mut urb_priv = unsafe { core::mem::zeroed() };
let mut plink: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut u32 = core::ptr::addr_of_mut!(plink) as *mut *mut u32;
let mut data: u64 = unsafe { core::mem::zeroed() };
let mut len: i32 = unsafe { core::mem::zeroed() };
let mut maxsze: i32 = unsafe { core::mem::zeroed() };
let mut status: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u64 = core::ptr::addr_of_mut!(status) as *mut u64;
let mut destination: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u64 = core::ptr::addr_of_mut!(destination) as *mut u64;
let mut td: core::mem::MaybeUninit<*mut uhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut *mut uhci_td = core::ptr::addr_of_mut!(td) as *mut *mut uhci_td;
unsafe {
let __slate_auto_bytes = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
maxsze = usb_endpoint_maxp(core::ptr::addr_of_mut!((*(*qh).hep).desc) as *const usb_endpoint_descriptor);
len = (*urb).transfer_buffer_length as i32;
data = (*urb).transfer_dma;
let __slate_auto_bytes_3 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
urbp = (*urb).hcpriv as *mut urb_priv;
let __slate_auto_bytes = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// The "pipe" thing contains the destination in bits 8--18
*__slate_slot_34 = ((*urb).pipe & 524032 | 45) as u64;
// 3 errors, dummy TD remains inactive
*__slate_slot_33 = uhci_maxerr!(3i32) as u64;
if ((*(*urb).dev).speed as u32) == 1 {
*__slate_slot_33 |= 67108864u64;
}
// Build the TD for the control request setup packet
*__slate_slot_35 = (*qh).dummy_td;
uhci_add_td_to_urbp(*__slate_slot_35, urbp);
uhci_fill_td(uhci, *__slate_slot_35, *__slate_slot_33 as u32, (*__slate_slot_34 | (uhci_explen!(8i32) as u64)) as u32, (*urb).setup_dma as u32);
*__slate_slot_29 = core::ptr::addr_of_mut!((*(*__slate_slot_35)).link);
*__slate_slot_33 |= 8388608u64;
// If direction is "send", change the packet ID from SETUP (0x2D)
// to OUT (0xE1).  Else change it from SETUP to IN (0x69) and
// set Short Packet Detect (SPD) for all data packets.
//
// 0-length transfers always get treated as "send".
if usb_pipeout!(unsafe { (*urb).pipe }) || len == 0 {
*__slate_slot_34 ^= 204u64;
} else {
*__slate_slot_34 ^= 68u64;
*__slate_slot_33 |= 536870912u64;
}
// Build the DATA TDs
'__join_0: {
while len > 0 {
pktsze = maxsze;
if len <= pktsze {
// The last data packet
pktsze = len;
let __v14: u64 = *__slate_slot_33 & 18446744073172680703u64;
*__slate_slot_33 = __v14;
}
*__slate_slot_35 = uhci_alloc_td(uhci);
if (*__slate_slot_35).is_null() {
break '__join_0;
} else {
*(*__slate_slot_29) = LINK_TO_TD!(uhci, *__slate_slot_35);
// Alternate Data0/1 (start with Data1)
*__slate_slot_34 ^= 524288u64;
uhci_add_td_to_urbp(*__slate_slot_35, urbp);
uhci_fill_td(uhci, *__slate_slot_35, *__slate_slot_33 as u32, (*__slate_slot_34 | (uhci_explen!(pktsze) as u64)) as u32, data as u32);
*__slate_slot_29 = core::ptr::addr_of_mut!((*(*__slate_slot_35)).link);
data = data.wrapping_add(pktsze as u64);
len = len.wrapping_sub(pktsze);
}
}
// Build the final TD for control status
*__slate_slot_35 = uhci_alloc_td(uhci);
if !(*__slate_slot_35).is_null() {
*(*__slate_slot_29) = LINK_TO_TD!(uhci, *__slate_slot_35);
// Change direction for the status transaction
*__slate_slot_34 ^= 136u64;
*__slate_slot_34 |= 524288u64; // End in Data1
uhci_add_td_to_urbp(*__slate_slot_35, urbp);
uhci_fill_td(uhci, *__slate_slot_35, (*__slate_slot_33 | 16777216u64) as u32, (*__slate_slot_34 | (uhci_explen!(0i32) as u64)) as u32, 0u32);
*__slate_slot_29 = core::ptr::addr_of_mut!((*(*__slate_slot_35)).link);
// Build the new dummy TD and activate the old one
*__slate_slot_35 = uhci_alloc_td(uhci);
if !(*__slate_slot_35).is_null() {
*(*__slate_slot_29) = LINK_TO_TD!(uhci, *__slate_slot_35);
uhci_fill_td(uhci, *__slate_slot_35, 0u32, (225 | uhci_explen!(0i32)) as u32, 0u32);
loop {
kcsan_wmb!();
core::arch::asm!("sfence", options(nostack, att_syntax));
if true {
break;
}
}
let __v3: u32 = cpu_to_hc32(uhci as *const uhci_hcd, 8388608u32);
let __v2: *mut uhci_td = (*qh).dummy_td;
(*__v2).status = (*__v2).status | __v3;
(*qh).dummy_td = *__slate_slot_35;
// Low-speed transfers get a different queue, and won't hog the bus.
// Also, some devices enumerate better without FSBR; the easiest way
// to do that is to put URBs on the low-speed queue while the device
// isn't in the CONFIGURED state.
if ((*(*urb).dev).speed as u32) == 1 || ((*(*urb).dev).state as u32) != 7 {
*__slate_slot_27 = 20;
} else {
*__slate_slot_27 = 21;
uhci_add_fsbr(uhci, urb);
}
if (*qh).state != 3 {
(*qh).skel = *__slate_slot_27;
}
return 0;
}
}
}
uhci_remove_td_from_urbp((*qh).dummy_td); // Remove the dummy TD from the td_list so it doesn't get freed
return 12i32.wrapping_neg();
}
}

fn uhci_submit_common(uhci: *mut uhci_hcd, urb: *mut urb, qh: *mut uhci_qh) -> i32 {
let mut pktsze: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut sg: core::mem::MaybeUninit<*mut scatterlist> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut scatterlist = core::ptr::addr_of_mut!(sg) as *mut *mut scatterlist;
let mut toggle: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut u32 = core::ptr::addr_of_mut!(toggle) as *mut u32;
let mut urbp: *mut urb_priv = unsafe { core::mem::zeroed() };
let mut plink: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut *mut u32 = core::ptr::addr_of_mut!(plink) as *mut *mut u32;
let mut data: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u64 = core::ptr::addr_of_mut!(data) as *mut u64;
let mut this_sg_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut i32 = core::ptr::addr_of_mut!(this_sg_len) as *mut i32;
let mut len: i32 = unsafe { core::mem::zeroed() };
let mut maxsze: i32 = unsafe { core::mem::zeroed() };
let mut status: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut u64 = core::ptr::addr_of_mut!(status) as *mut u64;
let mut destination: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut u64 = core::ptr::addr_of_mut!(destination) as *mut u64;
let mut td: core::mem::MaybeUninit<*mut uhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut *mut uhci_td = core::ptr::addr_of_mut!(td) as *mut *mut uhci_td;
unsafe {
let __slate_auto_bytes = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
maxsze = usb_endpoint_maxp(core::ptr::addr_of_mut!((*(*qh).hep).desc) as *const usb_endpoint_descriptor);
len = (*urb).transfer_buffer_length as i32;
let __slate_auto_bytes_3 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
urbp = (*urb).hcpriv as *mut urb_priv;
let __slate_auto_bytes_6 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if len < 0 {
return 22i32.wrapping_neg();
} else {
// The "pipe" thing contains the destination in bits 8--18
*__slate_slot_39 = ((*urb).pipe & 524032 | (usb_packetid!(unsafe { (*urb).pipe }) as u32)) as u64;
*__slate_slot_31 = usb_gettoggle__2!(unsafe { (*urb).dev }, usb_pipeendpoint!(unsafe { (*urb).pipe }), usb_pipeout!(unsafe { (*urb).pipe }));
// 3 errors, dummy TD remains inactive
*__slate_slot_38 = uhci_maxerr!(3i32) as u64;
if ((*(*urb).dev).speed as u32) == 1 {
*__slate_slot_38 |= 67108864u64;
}
if usb_pipein!(unsafe { (*urb).pipe }) != 0 {
*__slate_slot_38 |= 536870912u64;
}
*__slate_slot_29 = (*urb).num_mapped_sgs;
if len > 0 && *__slate_slot_29 > 0 {
*__slate_slot_30 = (*urb).sg;
*__slate_slot_34 = sg_dma_address!(*__slate_slot_30);
// urb->transfer_buffer_length may be smaller than the
// size of the scatterlist (or vice versa)
*__slate_slot_35 = min_t!(i32, sg_dma_len!(*__slate_slot_30), len);
} else {
*__slate_slot_30 = core::ptr::null_mut::<scatterlist>();
*__slate_slot_34 = (*urb).transfer_dma;
*__slate_slot_35 = len;
}
// Build the DATA TDs
*__slate_slot_33 = core::ptr::null_mut::<u32>();
*__slate_slot_40 = (*qh).dummy_td;
'__join_0: {
loop {
// Allow zero length packets
pktsze = maxsze;
if len <= pktsze {
// The last packet
pktsze = len;
if (*urb).transfer_flags & 1 == 0 {
let __v22: u64 = *__slate_slot_38 & 18446744073172680703u64;
*__slate_slot_38 = __v22;
}
}
if !(*__slate_slot_33).is_null() {
*__slate_slot_40 = uhci_alloc_td(uhci);
if (*__slate_slot_40).is_null() {
break '__join_0;
} else {
*(*__slate_slot_33) = LINK_TO_TD!(uhci, *__slate_slot_40);
}
}
uhci_add_td_to_urbp(*__slate_slot_40, urbp);
uhci_fill_td(uhci, *__slate_slot_40, *__slate_slot_38 as u32, (*__slate_slot_39 | (uhci_explen!(pktsze) as u64) | ((*__slate_slot_31 << 19i32) as u64)) as u32, *__slate_slot_34 as u32);
*__slate_slot_33 = core::ptr::addr_of_mut!((*(*__slate_slot_40)).link);
*__slate_slot_38 |= 8388608u64;
*__slate_slot_31 ^= 1;
let __v17: u64 = *__slate_slot_34;
*__slate_slot_34 = __v17.wrapping_add(pktsze as u64);
let __v15: i32 = *__slate_slot_35;
*__slate_slot_35 = __v15.wrapping_sub(pktsze);
len = len.wrapping_sub(maxsze);
if *__slate_slot_35 <= 0 {
let __v11: i32 = *__slate_slot_29;
let __v10: i32 = __v11.wrapping_sub(1);
*__slate_slot_29 = __v10;
if __v10 <= 0 || len <= 0 {
break;
} else {
*__slate_slot_30 = sg_next(*__slate_slot_30);
*__slate_slot_34 = sg_dma_address!(*__slate_slot_30);
*__slate_slot_35 = min_t!(i32, sg_dma_len!(*__slate_slot_30), len);
}
}
}
// URB_ZERO_PACKET means adding a 0-length packet, if direction
// is OUT and the transfer_length was an exact multiple of maxsze,
// hence (len = transfer_length - N * maxsze) == 0
// however, if transfer_length == 0, the zero packet was already
// prepared above.
if (*urb).transfer_flags & 64 != 0 && usb_pipeout!(unsafe { (*urb).pipe }) && len == 0 && (*urb).transfer_buffer_length > 0 {
*__slate_slot_40 = uhci_alloc_td(uhci);
if (*__slate_slot_40).is_null() {
break '__join_0;
} else {
*(*__slate_slot_33) = LINK_TO_TD!(uhci, *__slate_slot_40);
uhci_add_td_to_urbp(*__slate_slot_40, urbp);
uhci_fill_td(uhci, *__slate_slot_40, *__slate_slot_38 as u32, (*__slate_slot_39 | (uhci_explen!(0i32) as u64) | ((*__slate_slot_31 << 19i32) as u64)) as u32, *__slate_slot_34 as u32);
*__slate_slot_33 = core::ptr::addr_of_mut!((*(*__slate_slot_40)).link);
*__slate_slot_31 ^= 1;
}
}
// Set the interrupt-on-completion flag on the last packet.
// A more-or-less typical 4 KB URB (= size of one memory page)
// will require about 3 ms to transfer; that's a little on the
// fast side but not enough to justify delaying an interrupt
// more than 2 or 3 URBs, so we will ignore the URB_NO_INTERRUPT
// flag setting.
let __v7: u32 = cpu_to_hc32(uhci as *const uhci_hcd, 16777216u32);
let __v6: *mut uhci_td = *__slate_slot_40;
(*__v6).status = (*__v6).status | __v7;
// Build the new dummy TD and activate the old one
*__slate_slot_40 = uhci_alloc_td(uhci);
if !(*__slate_slot_40).is_null() {
*(*__slate_slot_33) = LINK_TO_TD!(uhci, *__slate_slot_40);
uhci_fill_td(uhci, *__slate_slot_40, 0u32, (225 | uhci_explen!(0i32)) as u32, 0u32);
loop {
kcsan_wmb!();
core::arch::asm!("sfence", options(nostack, att_syntax));
if true {
break;
}
}
let __v3: u32 = cpu_to_hc32(uhci as *const uhci_hcd, 8388608u32);
let __v2: *mut uhci_td = (*qh).dummy_td;
(*__v2).status = (*__v2).status | __v3;
(*qh).dummy_td = *__slate_slot_40;
usb_settoggle__3!(unsafe { (*urb).dev }, (*urb).pipe >> 15i32 & 15, (*urb).pipe & 128 == 0, *__slate_slot_31);
return 0;
}
}
uhci_remove_td_from_urbp((*qh).dummy_td); // Remove the dummy TD from the td_list so it doesn't get freed
return 12i32.wrapping_neg();
}
}
}

fn uhci_submit_bulk(uhci: *mut uhci_hcd, urb: *mut urb, qh: *mut uhci_qh) -> i32 {
let mut ret: i32 = 0;
// Can't have low-speed bulk transfers
if ((unsafe { (*(*urb).dev).speed }) as u32) == 1 {
return 22i32.wrapping_neg();
}
if (unsafe { (*qh).state }) != 3 {
unsafe {
(*qh).skel = 22;
}
}
ret = uhci_submit_common(uhci, urb, qh);
if ret == 0 {
uhci_add_fsbr(uhci, urb);
}
ret
}

fn uhci_submit_interrupt(uhci: *mut uhci_hcd, urb: *mut urb, qh: *mut uhci_qh) -> i32 {
let mut ret: i32 = 0;
// USB 1.1 interrupt transfers only involve one packet per interval.
// Drivers can submit URBs of any length, but longer ones will need
// multiple intervals to complete.
if ((unsafe { (*qh).__slate_bits_0.__get_bandwidth_reserved() }) as i32) == 0 {
// Figure out which power-of-two queue to use
let mut exponent: i32 = 7;
while exponent >= 0 {
if 1i32 << exponent <= unsafe { (*urb).interval } {
break;
}
exponent = exponent.wrapping_sub(1);
}
if exponent < 0 {
return 22i32.wrapping_neg();
}
// If the slot is full, try a lower period
loop {
unsafe {
(*qh).period = (1i32 << exponent) as u32;
}
unsafe {
(*qh).skel = SKEL_INDEX!(exponent);
}
// For now, interrupt phase is fixed by the layout
// of the QH lists.
unsafe {
(*qh).phase = ((*qh).period / 2 & 31u32) as i16;
}
ret = uhci_check_bandwidth(uhci, qh);
let __v4: bool = if ret != 0 {
let __v6: i32 = exponent.wrapping_sub(1);
exponent = __v6;
__v6 >= 0
} else {
false
};
if !__v4 {
break;
}
}
if ret != 0 {
return ret;
}
} else {
if (unsafe { (*qh).period }) > ((unsafe { (*urb).interval }) as u32) {
return 22i32.wrapping_neg();
}
}
// Can't decrease the period
ret = uhci_submit_common(uhci, urb, qh);
if ret == 0 {
unsafe {
(*urb).interval = (*qh).period as i32;
}
if ((unsafe { (*qh).__slate_bits_0.__get_bandwidth_reserved() }) as i32) == 0 {
uhci_reserve_bandwidth(uhci, qh);
}
}
ret
}

fn uhci_fixup_short_transfer(uhci: *mut uhci_hcd, qh: *mut uhci_qh, urbp: *mut urb_priv) -> i32 {
let mut td: *mut uhci_td = unsafe { core::mem::zeroed() };
let mut tmp: *mut list_head = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
td = list_entry__2!(unsafe { (*urbp).td_list.prev }, uhci_td);
if (unsafe { (*qh).r#type }) == 0 {
// When a control transfer is short, we have to restart
// the queue at the status stage transaction, which is
// the last TD.
let mut __v1: i64;
let __ret_warn_on: i32 = (list_empty((unsafe { core::ptr::addr_of_mut!((*urbp).td_list) }) as *const list_head) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("551: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 551b - ., 3; .popsection\n# {0}", const 551 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_uhci_fixup_short_transfer_2, sym __slate_asm_string_uhci_fixup_short_transfer_3, const 1135 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("552: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 552b - ., 4; .popsection\n# {0}", const 552 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
unsafe {
(*qh).element = LINK_TO_TD!(uhci, td);
}
tmp = unsafe { (*td).list.prev };
ret = 115i32.wrapping_neg();
} else {
// When a bulk/interrupt transfer is short, we have to
// fix up the toggles of the following URBs on the queue
// before restarting the queue at the next URB.
unsafe {
let __t0 = uhci_toggle!(td_token!(uhci, unsafe { (*qh).post_td })) ^ 1;
(*qh).__slate_bits_0.__set_initial_toggle(__t0);
}
uhci_fixup_toggles(uhci, qh, 1i32);
if list_empty((unsafe { core::ptr::addr_of_mut!((*urbp).td_list) }) as *const list_head) != 0 {
td = unsafe { (*qh).post_td };
}
unsafe {
(*qh).element = (*td).link;
}
tmp = unsafe { (*urbp).td_list.prev };
ret = 0;
}
// Remove all the TDs we skipped over, from tmp back to the start
while tmp != unsafe { core::ptr::addr_of_mut!((*urbp).td_list) } {
td = list_entry__2!(tmp, uhci_td);
tmp = unsafe { (*tmp).prev };
uhci_remove_td_from_urbp(td);
uhci_free_td(uhci, td);
}
ret
}

fn uhci_result_common(uhci: *mut uhci_hcd, urb: *mut urb) -> i32 {
let mut len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(len) as *mut i32;
let mut ctrlstat: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(ctrlstat) as *mut u32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut tmp: core::mem::MaybeUninit<*mut uhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut uhci_td = core::ptr::addr_of_mut!(tmp) as *mut *mut uhci_td;
let mut td: core::mem::MaybeUninit<*mut uhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut uhci_td = core::ptr::addr_of_mut!(td) as *mut *mut uhci_td;
let mut qh: *mut uhci_qh = unsafe { core::mem::zeroed() };
let mut urbp: *mut urb_priv = unsafe { core::mem::zeroed() };
unsafe {
urbp = (*urb).hcpriv as *mut urb_priv;
qh = (*urbp).qh;
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
ret = 0;
*__slate_slot_11 = list_first_entry__2!(core::ptr::addr_of_mut!((*urbp).td_list));
let __v7: *mut uhci_td = list_next_entry__2!(*__slate_slot_11, list);
*__slate_slot_10 = __v7;
'__join_3: {
while list_entry_is_head!(*__slate_slot_11, core::ptr::addr_of_mut!((*urbp).td_list), list) == 0 {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
while false {
}
let __v4: u32 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_11)).status) as *const u32);
*__slate_slot_6 = hc32_to_cpu(uhci as *const uhci_hcd, __v4);
*__slate_slot_9 = uhci_status_bits!(*__slate_slot_6);
if *__slate_slot_9 & 8388608u32 != 0 {
return 115i32.wrapping_neg();
} else {
*__slate_slot_5 = uhci_actual_length!(*__slate_slot_6) as i32;
let __v2: u32 = (*urb).actual_length;
(*urb).actual_length = __v2.wrapping_add(*__slate_slot_5 as u32);
if *__slate_slot_9 != 0 {
ret = uhci_map_status(*__slate_slot_9 as i32, uhci_packetout!(td_token!(uhci, *__slate_slot_11)) as i32);
// Did we receive a short packet?
} else {
if (*__slate_slot_5 as u32) < uhci_expected_length!(td_token!(uhci, *__slate_slot_11)) {
// For control transfers, go to the status TD if
// this isn't already the last data TD
if (*qh).r#type == 0 {
if (*(*__slate_slot_11)).list.next != (*urbp).td_list.prev {
ret = 1;
}
} else {
if (*urb).transfer_flags & 1 != 0 {
ret = 121i32.wrapping_neg();
} else {
if core::ptr::addr_of_mut!((*(*__slate_slot_11)).list) != (*urbp).td_list.prev {
ret = 1;
}
}
}
// For bulk and interrupt, this may be an error
// Fixup needed only if this isn't the URB's last TD
}
}
uhci_remove_td_from_urbp(*__slate_slot_11);
if !unsafe { (*qh).post_td }.is_null() {
uhci_free_td(uhci, (*qh).post_td);
}
(*qh).post_td = *__slate_slot_11;
if ret != 0 {
break '__join_3;
} else {
*__slate_slot_11 = *__slate_slot_10;
let __v0: *mut uhci_td = list_next_entry__2!(*__slate_slot_10, list);
*__slate_slot_10 = __v0;
}
}
}
return ret;
}
if ret < 0 {
// Note that the queue has stopped and save
// the next toggle value
(*qh).element = UHCI_PTR_TERM!(uhci);
(*qh).__slate_bits_0.__set_is_stopped(1u32);
let __t0 = ((*qh).r#type != 0) as u32;
(*qh).__slate_bits_0.__set_needs_fixup(__t0);
let __t1 = uhci_toggle!(td_token!(uhci, *__slate_slot_11)) ^ (((ret == 121i32.wrapping_neg()) as i32) as u32);
(*qh).__slate_bits_0.__set_initial_toggle(__t1);
} else {
ret = uhci_fixup_short_transfer(uhci, qh, urbp);
}
// Short packet received
return ret;
}
// Some debugging code
// Print the chain for debugging
}

fn uhci_submit_isochronous(uhci: *mut uhci_hcd, urb: *mut urb, qh: *mut uhci_qh) -> i32 {
let mut td: *mut uhci_td = core::ptr::null_mut::<uhci_td>(); // Since urb->number_of_packets > 0
let mut i: i32 = 0;
let mut frame: u32 = 0;
let mut next: u32 = 0;
let mut destination: u64 = 0;
let status: u64 = 41943040u64;
let urbp: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
// Values must not be too big (could overflow below)
if (unsafe { (*urb).interval }) >= 1024 || (unsafe { (*urb).number_of_packets }) >= 1024 {
return 27i32.wrapping_neg();
}
uhci_get_current_frame_number(uhci);
// Check the period and figure out the starting frame number
if ((unsafe { (*qh).__slate_bits_0.__get_bandwidth_reserved() }) as i32) == 0 {
unsafe {
(*qh).period = (*urb).interval as u32;
}
unsafe {
(*qh).phase = 1i32.wrapping_neg() as i16;
}
// Find the best phase
i = uhci_check_bandwidth(uhci, qh);
if i != 0 {
return i;
}
// Allow a little time to allocate the TDs
next = unsafe { (*uhci).frame_number }.wrapping_add(10);
frame = (unsafe { (*qh).phase }) as u32;
// Round up to the first available slot
frame = frame.wrapping_add(next.wrapping_sub(frame).wrapping_add(unsafe { (*qh).period }).wrapping_sub(1) & unsafe { (*qh).period }.wrapping_neg());
} else {
if (unsafe { (*qh).period }) != ((unsafe { (*urb).interval }) as u32) {
return 22i32.wrapping_neg(); // Can't change the period
} else {
next = unsafe { (*uhci).frame_number }.wrapping_add(1);
// Find the next unused frame
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).queue) }) as *const list_head) != 0 {
frame = unsafe { (*qh).iso_frame };
} else {
let lurb: *mut urb = unsafe { (*list_entry!(unsafe { (*qh).queue.prev }, urb_priv)).urb };
frame = unsafe { (*lurb).start_frame }.wrapping_add(unsafe { (*lurb).number_of_packets }.wrapping_mul(unsafe { (*lurb).interval })) as u32;
}
// Fell behind?
if !uhci_frame_before_eq!(next, frame) {
// USB_ISO_ASAP: Round up to the first available slot
if (unsafe { (*urb).transfer_flags }) & 2 != 0 {
frame = frame.wrapping_add(next.wrapping_sub(frame).wrapping_add(unsafe { (*qh).period }).wrapping_sub(1) & unsafe { (*qh).period }.wrapping_neg());
} else {
if !uhci_frame_before_eq!(next, frame.wrapping_add((unsafe { (*urb).number_of_packets }.wrapping_sub(1) as u32).wrapping_mul(unsafe { (*qh).period }))) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, uhci_dev!(uhci) as *const device, c"iso underrun %p (%u+%u < %u)\n".as_ptr() as *const u8, urb, frame, (unsafe { (*urb).number_of_packets }.wrapping_sub(1) as u32).wrapping_mul((*qh).period), next) };
}
}
}
// Not ASAP: Use the next slot in the stream,
// no matter what.
}
}
}
// Make sure we won't have to go too far into the future
if uhci_frame_before_eq!(unsafe { (*uhci).last_iso_frame }.wrapping_add(1024), frame.wrapping_add(unsafe { (*urb).number_of_packets }.wrapping_mul(unsafe { (*urb).interval }) as u32)) {
return 27i32.wrapping_neg();
}
unsafe {
(*urb).start_frame = frame as i32;
}
destination = ((unsafe { (*urb).pipe }) & 524032 | (usb_packetid!(unsafe { (*urb).pipe }) as u32)) as u64;
i = 0;
while i < unsafe { (*urb).number_of_packets } {
td = uhci_alloc_td(uhci);
if td.is_null() {
return 12i32.wrapping_neg();
}
uhci_add_td_to_urbp(td, urbp);
uhci_fill_td(uhci, td, status as u32, (destination | (uhci_explen__2!(unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(i as isize)).length }) as u64)) as u32, unsafe { (*urb).transfer_dma }.wrapping_add((unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(i as isize)).offset }) as u64) as u32);
i = i.wrapping_add(1);
}
// Set the interrupt-on-completion flag on the last packet.
let __v8: u32 = cpu_to_hc32(uhci as *const uhci_hcd, 16777216u32);
unsafe {
(*td).status = (*td).status | __v8;
}
// Add the TDs to the frame list
frame = (unsafe { (*urb).start_frame }) as u32;
td = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*urbp).td_list) });
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*urbp).td_list) }, list) == 0 {
uhci_insert_td_in_frame_list(uhci, td, frame);
frame = frame.wrapping_add(unsafe { (*qh).period });
td = list_next_entry__2!(td, list);
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).queue) }) as *const list_head) != 0 {
unsafe {
(*qh).iso_packet_desc = (core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(0 as isize);
}
unsafe {
(*qh).iso_frame = (*urb).start_frame as u32;
}
}
unsafe {
(*qh).skel = 1;
}
if ((unsafe { (*qh).__slate_bits_0.__get_bandwidth_reserved() }) as i32) == 0 {
uhci_reserve_bandwidth(uhci, qh);
}
0
}

fn uhci_result_isochronous(uhci: *mut uhci_hcd, urb: *mut urb) -> i32 {
let mut td: *mut uhci_td = unsafe { core::mem::zeroed() };
let mut tmp: *mut uhci_td = unsafe { core::mem::zeroed() };
let urbp: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
let qh: *mut uhci_qh = unsafe { (*urbp).qh };
td = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*urbp).td_list) });
tmp = list_next_entry__2!(td, list);
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*urbp).td_list) }, list) == 0 {
let mut ctrlstat: u32 = 0;
let mut status: i32 = 0;
let mut actlength: i32 = 0;
if uhci_frame_before_eq!(unsafe { (*uhci).cur_iso_frame }, unsafe { (*qh).iso_frame }) {
return 115i32.wrapping_neg();
}
uhci_remove_tds_from_frame(uhci, unsafe { (*qh).iso_frame });
let mut __v3: u32;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_554() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*td).status) as *const u32) };
ctrlstat = hc32_to_cpu(uhci as *const uhci_hcd, __v3);
if ctrlstat & 8388608u32 != 0 {
status = 18i32.wrapping_neg(); // TD was added too late?
} else {
status = uhci_map_status(uhci_status_bits!(ctrlstat) as i32, usb_pipeout!(unsafe { (*urb).pipe }) as i32);
actlength = uhci_actual_length!(ctrlstat) as i32;
let __v6: u32 = unsafe { (*urb).actual_length };
unsafe {
(*urb).actual_length = __v6.wrapping_add(actlength as u32);
}
unsafe {
(*(*qh).iso_packet_desc).actual_length = actlength as u32;
}
unsafe {
(*(*qh).iso_packet_desc).status = status;
}
}
if status != 0 {
let __v9: i32 = unsafe { (*urb).error_count };
unsafe {
(*urb).error_count = __v9.wrapping_add(1);
}
}
uhci_remove_td_from_urbp(td);
uhci_free_td(uhci, td);
let __v12: u32 = unsafe { (*qh).iso_frame };
unsafe {
(*qh).iso_frame = __v12.wrapping_add((*qh).period);
}
let __v15: *mut usb_iso_packet_descriptor = unsafe { (*qh).iso_packet_desc };
unsafe {
(*qh).iso_packet_desc = __v15.wrapping_offset(1 as isize);
}
td = tmp;
tmp = list_next_entry__2!(tmp, list);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_urb_enqueue")]
extern "C" fn uhci_urb_enqueue(hcd: *mut usb_hcd, urb: *mut urb, mem_flags: u32) -> i32 {
let mut qh: core::mem::MaybeUninit<*mut uhci_qh> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut uhci_qh = core::ptr::addr_of_mut!(qh) as *mut *mut uhci_qh;
let mut urbp: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut urb_priv = core::ptr::addr_of_mut!(urbp) as *mut *mut urb_priv;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut uhci: *mut uhci_hcd = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
uhci = hcd_to_uhci(hcd);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*uhci).lock), *__slate_slot_2);
*__slate_slot_4 = usb_hcd_link_urb_to_ep(hcd, urb);
if *__slate_slot_4 == 0 {
'__join_2: {
*__slate_slot_4 = 12i32.wrapping_neg();
*__slate_slot_1 = uhci_alloc_urb_priv(uhci, urb);
if !(*__slate_slot_1).is_null() {
'__join_3: {
if !unsafe { (*(*urb).ep).hcpriv }.is_null() {
*__slate_slot_0 = (*(*urb).ep).hcpriv as *mut uhci_qh;
} else {
*__slate_slot_0 = uhci_alloc_qh(uhci, (*urb).dev, (*urb).ep);
if (*__slate_slot_0).is_null() {
break '__join_3;
}
}
(*(*__slate_slot_1)).qh = *__slate_slot_0;
let __t0: i32 = (*(*__slate_slot_0)).r#type;
if __t0 == 0 {
*__slate_slot_4 = uhci_submit_control(uhci, urb, *__slate_slot_0);
} else {
if __t0 == 2 {
*__slate_slot_4 = uhci_submit_bulk(uhci, urb, *__slate_slot_0);
} else {
if __t0 == 3 {
*__slate_slot_4 = uhci_submit_interrupt(uhci, urb, *__slate_slot_0);
} else {
if __t0 == 1 {
(*urb).error_count = 0;
*__slate_slot_4 = uhci_submit_isochronous(uhci, urb, *__slate_slot_0);
}
}
}
}
if *__slate_slot_4 != 0 {
if (*(*__slate_slot_0)).state == 1 {
uhci_make_qh_idle(uhci, *__slate_slot_0);
}
// Reclaim unused QH
} else {
// Add this URB to the QH
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_1)).node), core::ptr::addr_of_mut!((*(*__slate_slot_0)).queue));
// If the new URB is the first and only one on this QH then either
// the QH is new and idle or else it's unlinked and waiting to
// become idle, so we can activate it right away.  But only if the
// queue isn't stopped.
if (*(*__slate_slot_0)).queue.next == core::ptr::addr_of_mut!((*(*__slate_slot_1)).node) && ((*(*__slate_slot_0)).__slate_bits_0.__get_is_stopped() as i32) == 0 {
uhci_activate_qh(uhci, *__slate_slot_0);
uhci_urbp_wants_fsbr(uhci, *__slate_slot_1);
break '__join_2;
} else {
break '__join_2;
}
}
}
uhci_free_urb_priv(uhci, *__slate_slot_1);
}
}
if *__slate_slot_4 != 0 {
usb_hcd_unlink_urb_from_ep(hcd, urb);
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*uhci).lock), *__slate_slot_2);
return *__slate_slot_4;
}
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_urb_dequeue")]
extern "C" fn uhci_urb_dequeue(hcd: *mut usb_hcd, urb: *mut urb, status: i32) -> i32 {
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut qh: core::mem::MaybeUninit<*mut uhci_qh> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut uhci_qh = core::ptr::addr_of_mut!(qh) as *mut *mut uhci_qh;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut uhci: *mut uhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
uhci = hcd_to_uhci(hcd);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*uhci).lock), *__slate_slot_2);
*__slate_slot_0 = usb_hcd_check_unlink_urb(hcd, urb, status);
if *__slate_slot_0 == 0 {
*__slate_slot_1 = (*((*urb).hcpriv as *mut urb_priv)).qh;
// Remove Isochronous TDs from the frame list ASAP
if (*(*__slate_slot_1)).r#type == 1 {
uhci_unlink_isochronous_tds(uhci, urb);
loop {
kcsan_mb!();
core::arch::asm!("mfence", options(nostack, att_syntax));
if true {
break;
}
}
// If the URB has already started, update the QH unlink time
uhci_get_current_frame_number(uhci);
if uhci_frame_before_eq__2!(unsafe { (*urb).start_frame }, unsafe { (*uhci).frame_number }) {
(*(*__slate_slot_1)).unlink_frame = (*uhci).frame_number;
}
}
uhci_unlink_qh(uhci, *__slate_slot_1);
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*uhci).lock), *__slate_slot_2);
return *__slate_slot_0;
}
}

fn uhci_giveback_urb(uhci: *mut uhci_hcd, qh: *mut uhci_qh, urb: *mut urb, status: i32) {
let urbp: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
if (unsafe { (*qh).r#type }) == 0 {
// Subtract off the length of the SETUP packet from
// urb->actual_length.
let __v0: u32 = min_t!(u32, 8i32, unsafe { (*urb).actual_length });
let __v5: u32 = unsafe { (*urb).actual_length };
unsafe {
(*urb).actual_length = __v5.wrapping_sub(__v0);
}
} else {
if (unsafe { (*qh).r#type }) == 1 && (unsafe { (*urbp).node.prev }) == unsafe { core::ptr::addr_of_mut!((*qh).queue) } && (unsafe { (*urbp).node.next }) != unsafe { core::ptr::addr_of_mut!((*qh).queue) } {
let nurb: *mut urb = unsafe { (*list_entry!(unsafe { (*urbp).node.next }, urb_priv)).urb };
unsafe {
(*qh).iso_packet_desc = (core::ptr::addr_of_mut!((*nurb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(0 as isize);
}
unsafe {
(*qh).iso_frame = (*nurb).start_frame as u32;
}
}
}
// When giving back the first URB in an Isochronous queue,
// reinitialize the QH's iso-related members for the next URB.
// Take the URB off the QH's queue.  If the queue is now empty,
// this is a perfect time for a toggle fixup.
list_del_init(unsafe { core::ptr::addr_of_mut!((*urbp).node) });
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).queue) }) as *const list_head) != 0 && ((unsafe { (*qh).__slate_bits_0.__get_needs_fixup() }) as i32) != 0 {
usb_settoggle__2!(unsafe { (*urb).dev }, (unsafe { (*urb).pipe }) >> 15i32 & 15, (unsafe { (*urb).pipe }) & 128 == 0, (unsafe { (*qh).__slate_bits_0.__get_initial_toggle() }) as i32);
unsafe {
(*qh).__slate_bits_0.__set_needs_fixup(0u32);
}
}
uhci_free_urb_priv(uhci, urbp);
unsafe { usb_hcd_unlink_urb_from_ep(uhci_to_hcd(uhci), urb) };
spin_unlock(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
unsafe { usb_hcd_giveback_urb(uhci_to_hcd(uhci), urb, status) };
spin_lock(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
// If the queue is now empty, we can unlink the QH and give up its
// reserved bandwidth.
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).queue) }) as *const list_head) != 0 {
uhci_unlink_qh(uhci, qh);
if ((unsafe { (*qh).__slate_bits_0.__get_bandwidth_reserved() }) as i32) != 0 {
uhci_release_bandwidth(uhci, qh);
}
}
}

fn uhci_scan_qh(uhci: *mut uhci_hcd, qh: *mut uhci_qh) {
let mut status: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(status) as *mut i32;
let mut urb: core::mem::MaybeUninit<*mut urb> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut urb = core::ptr::addr_of_mut!(urb) as *mut *mut urb;
let mut urbp: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut urb_priv = core::ptr::addr_of_mut!(urbp) as *mut *mut urb_priv;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
'__join_20: {
loop {
if list_empty(core::ptr::addr_of_mut!((*qh).queue) as *const list_head) == 0 {
*__slate_slot_7 = list_entry!(unsafe { (*qh).queue.next }, urb_priv);
*__slate_slot_6 = (*(*__slate_slot_7)).urb;
if (*qh).r#type == 1 {
*__slate_slot_5 = uhci_result_isochronous(uhci, *__slate_slot_6);
} else {
*__slate_slot_5 = uhci_result_common(uhci, *__slate_slot_6);
}
if *__slate_slot_5 == 115i32.wrapping_neg() {
break '__join_20;
} else {
// Dequeued but completed URBs can't be given back unless
// the QH is stopped or has finished unlinking.
if (*(*__slate_slot_6)).unlinked != 0 {
if QH_FINISHED_UNLINKING!(qh, uhci) {
(*qh).__slate_bits_0.__set_is_stopped(1u32);
} else {
if ((*qh).__slate_bits_0.__get_is_stopped() as i32) == 0 {
break;
}
}
}
uhci_giveback_urb(uhci, qh, *__slate_slot_6, *__slate_slot_5);
if *__slate_slot_5 < 0 {
break '__join_20;
}
}
} else {
break '__join_20;
}
}
return;
}
// If the QH is neither stopped nor finished unlinking (normal case),
// our work here is done.
if QH_FINISHED_UNLINKING!(qh, uhci) {
(*qh).__slate_bits_0.__set_is_stopped(1u32);
} else {
if ((*qh).__slate_bits_0.__get_is_stopped() as i32) == 0 {
return;
}
}
// Otherwise give back each of the dequeued URBs
'__loop_15: loop {
*__slate_slot_7 = list_first_entry__26!(core::ptr::addr_of_mut!((*qh).queue));
loop {
if list_entry_is_head!(*__slate_slot_7, core::ptr::addr_of_mut!((*qh).queue), node) == 0 {
*__slate_slot_6 = (*(*__slate_slot_7)).urb;
if (*(*__slate_slot_6)).unlinked != 0 {
break;
} else {
*__slate_slot_7 = list_next_entry__6!(*__slate_slot_7, node);
}
} else {
break '__loop_15;
}
}
// Fix up the TD links and save the toggles for
// non-Isochronous queues.  For Isochronous queues,
// test for too-recent dequeues.
if uhci_cleanup_queue(uhci, qh, *__slate_slot_6) == 0 {
(*qh).__slate_bits_0.__set_is_stopped(0u32);
return;
} else {
uhci_giveback_urb(uhci, qh, *__slate_slot_6, 0i32);
}
}
(*qh).__slate_bits_0.__set_is_stopped(0u32);
// There are no more dequeued URBs.  If there are still URBs on the
// queue, the QH can now be re-activated.
if list_empty(core::ptr::addr_of_mut!((*qh).queue) as *const list_head) == 0 {
if ((*qh).__slate_bits_0.__get_needs_fixup() as i32) != 0 {
uhci_fixup_toggles(uhci, qh, 0i32);
}
// If the first URB on the queue wants FSBR but its time
// limit has expired, set the next TD to interrupt on
// completion before reactivating the QH.
*__slate_slot_7 = list_entry!(unsafe { (*qh).queue.next }, urb_priv);
if ((*(*__slate_slot_7)).__slate_bits_0.__get_fsbr() as i32) != 0 && ((*qh).__slate_bits_0.__get_wait_expired() as i32) != 0 {
let td: *mut uhci_td = list_entry__2!(unsafe { (*(*__slate_slot_7)).td_list.next }, uhci_td);
let __v3: u32 = cpu_to_hc32(uhci as *const uhci_hcd, 16777216u32);
(*td).status = (*td).status | __v3;
}
uhci_activate_qh(uhci, qh);
} else {
if QH_FINISHED_UNLINKING!(qh, uhci) {
uhci_make_qh_idle(uhci, qh);
}
}
// The queue is empty.  The QH can become idle if it is fully
// unlinked.
}
}

fn uhci_advance_check(uhci: *mut uhci_hcd, qh: *mut uhci_qh) -> i32 {
// Detect the Intel bug and work around it
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut td: core::mem::MaybeUninit<*mut uhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut uhci_td = core::ptr::addr_of_mut!(td) as *mut *mut uhci_td;
let mut urbp: *mut urb_priv = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
urbp = core::ptr::null_mut::<urb_priv>();
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 1;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if (*qh).r#type != 1 {
// Treat an UNLINKING queue as though it hasn't advanced.
// This is okay because reactivation will treat it as though
// it has advanced, and if it is going to become IDLE then
// this doesn't matter anyway.  Furthermore it's possible
// for an UNLINKING queue not to have any URBs at all, or
// for its first URB not to have any TDs (if it was dequeued
// just as it completed).  So it's not easy in any case to
// test whether such queues have advanced.
if (*qh).state != 3 {
urbp = core::ptr::null_mut::<urb_priv>();
*__slate_slot_3 = 0;
} else {
urbp = list_entry!(unsafe { (*qh).queue.next }, urb_priv);
*__slate_slot_5 = list_entry__2!(unsafe { (*urbp).td_list.next }, uhci_td);
while false {
}
let __v2: u32 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_5)).status) as *const u32);
*__slate_slot_3 = hc32_to_cpu(uhci as *const uhci_hcd, __v2);
if *__slate_slot_3 & 8388608u32 == 0 {
// We're okay, the queue has advanced
(*qh).__slate_bits_0.__set_wait_expired(0u32);
(*qh).advance_jiffies = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
break '__join_0;
} else {
ret = (*uhci).is_stopped as i32;
}
}
// The queue hasn't advanced; check for timeout
if ((*qh).__slate_bits_0.__get_wait_expired() as i32) == 0 {
if time_after__2!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), unsafe { (*qh).advance_jiffies }.wrapping_add(msecs_to_jiffies(200u32))) {
if !unsafe { (*qh).post_td }.is_null() {
while false {
}
let __v0: u32 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*qh).element) as *const u32);
__v1 = __v0 == LINK_TO_TD!(uhci, unsafe { (*qh).post_td });
} else {
__v1 = false;
}
if __v1 {
(*qh).element = (*(*qh).post_td).link;
(*qh).advance_jiffies = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
ret = 1;
} else {
(*qh).__slate_bits_0.__set_wait_expired(1u32);
// If the current URB wants FSBR, unlink it temporarily
// so that we can safely set the next TD to interrupt on
// completion.  That way we'll know as soon as the queue
// starts moving again.
if !urbp.is_null() && ((*urbp).__slate_bits_0.__get_fsbr() as i32) != 0 && *__slate_slot_3 & 16777216u32 == 0 {
uhci_unlink_qh(uhci, qh);
}
}
} else {
// Unmoving but not-yet-expired queues keep FSBR alive
if !urbp.is_null() {
uhci_urbp_wants_fsbr(uhci, urbp);
}
}
}
}
}
return ret;
}
}

fn uhci_scan_schedule(uhci: *mut uhci_hcd) {
let mut qh: core::mem::MaybeUninit<*mut uhci_qh> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut uhci_qh = core::ptr::addr_of_mut!(qh) as *mut *mut uhci_qh;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Don't allow re-entrant calls
if ((*uhci).__slate_bits_0.__get_scan_in_progress() as i32) != 0 {
(*uhci).__slate_bits_0.__set_need_rescan(1u32);
return;
} else {
(*uhci).__slate_bits_0.__set_scan_in_progress(1u32);
loop {
(*uhci).__slate_bits_0.__set_need_rescan(0u32);
(*uhci).__slate_bits_0.__set_fsbr_is_wanted(0u32);
uhci_clear_next_interrupt(uhci);
uhci_get_current_frame_number(uhci);
(*uhci).cur_iso_frame = (*uhci).frame_number;
// Go through all the QH queues and process the URBs in each one
*__slate_slot_4 = 0;
while *__slate_slot_4 < 10i32 {
(*uhci).next_qh = list_entry!(unsafe { (*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(*__slate_slot_4 as isize))).node.next }, uhci_qh);
loop {
let __v2: *mut uhci_qh = (*uhci).next_qh;
*__slate_slot_3 = __v2;
if __v2 != *((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(*__slate_slot_4 as isize) {
(*uhci).next_qh = list_entry!(unsafe { (*(*__slate_slot_3)).node.next }, uhci_qh);
if uhci_advance_check(uhci, *__slate_slot_3) != 0 {
uhci_scan_qh(uhci, *__slate_slot_3);
if (*(*__slate_slot_3)).state == 3 {
uhci_urbp_wants_fsbr(uhci, list_entry!(unsafe { (*(*__slate_slot_3)).queue.next }, urb_priv));
}
}
} else {
break;
}
}
let __v1: i32 = *__slate_slot_4;
*__slate_slot_4 = __v1.wrapping_add(1);
}
(*uhci).last_iso_frame = (*uhci).cur_iso_frame;
if ((*uhci).__slate_bits_0.__get_need_rescan() as i32) == 0 {
break;
}
}
(*uhci).__slate_bits_0.__set_scan_in_progress(0u32);
if ((*uhci).__slate_bits_0.__get_fsbr_is_on() as i32) != 0 && ((*uhci).__slate_bits_0.__get_fsbr_is_wanted() as i32) == 0 && ((*uhci).__slate_bits_0.__get_fsbr_expiring() as i32) == 0 {
(*uhci).__slate_bits_0.__set_fsbr_expiring(1u32);
mod_timer(core::ptr::addr_of_mut!((*uhci).fsbr_timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(10u32)));
}
if list_empty(core::ptr::addr_of_mut!((*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(0 as isize))).node) as *const list_head) != 0 {
uhci_clear_next_interrupt(uhci);
} else {
uhci_set_next_interrupt(uhci);
}
}
}
}

static root_hub_hub_des: [u8; 9] = [9, 41, 2, 10, 0, 1, 0, 0, 255];

// status change bits:  nonzero writes will clear
// suspend/resume bits: port suspended or port resuming
fn any_ports_active(uhci: *mut uhci_hcd) -> i32 {
let mut port: i32 = 0;
while port < unsafe { (*uhci).rh_numports } {
let __v0: bool = (uhci_readw(uhci as *const uhci_hcd, 16i32.wrapping_add(port.wrapping_mul(2))) as i32) & (1 | (2048 | 8 | 2)) != 0 || test_bit!(port, unsafe { core::ptr::addr_of_mut!((*uhci).port_c_suspend) }) != 0;
if __v0 {
return 1;
}
port = port.wrapping_add(1);
}
0
}

fn get_hub_status_data(uhci: *mut uhci_hcd, buf: *mut u8) -> i32 {
let mut port: i32 = 0;
let mut mask: i32 = 2048 | 8 | 2;
// Some boards (both VIA and Intel apparently) report bogus
// overcurrent indications, causing massive log spam unless
// we completely ignore them.  This doesn't seem to be a problem
// with the chipset so much as with the way it is connected on
// the motherboard; if the overcurrent input is left to float
// then it may constantly register false positives.
if unsafe { ignore_oc } {
let __v1: i32 = mask & !2048;
mask = __v1;
}
unsafe {
*buf = 0;
}
while port < unsafe { (*uhci).rh_numports } {
let __v2: bool = (uhci_readw(uhci as *const uhci_hcd, 16i32.wrapping_add(port.wrapping_mul(2))) as i32) & mask != 0 || test_bit!(port, unsafe { core::ptr::addr_of_mut!((*uhci).port_c_suspend) }) != 0;
if __v2 {
unsafe {
*buf = ((*buf as i32) | 1i32 << port.wrapping_add(1)) as u8;
}
}
port = port.wrapping_add(1);
}
((unsafe { *buf }) != 0) as i32
}

fn uhci_finish_suspend(uhci: *mut uhci_hcd, port: i32, port_addr: u64) {
let mut status: i32 = 0;
if (uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32) & (4096 | 64) != 0 {
CLR_RH_PORTSTAT!(4160i32, status, uhci, port_addr);
if test_bit!(port, unsafe { core::ptr::addr_of_mut!((*uhci).resuming_ports) }) != 0 {
set_bit(port as i64, (unsafe { core::ptr::addr_of_mut!((*uhci).port_c_suspend) }) as *mut u64);
}
// The controller won't actually turn off the RD bit until
// it has had a chance to send a low-speed EOP sequence,
// which is supposed to take 3 bit times (= 2 microseconds).
// Experiments show that some controllers take longer, so
// we'll poll for completion.
for i in 0i32..10i32 {
if (uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32) & (4096 | 64) == 0 {
break;
}
udelay(1u64);
}
}
clear_bit(port as i64, (unsafe { core::ptr::addr_of_mut!((*uhci).resuming_ports) }) as *mut u64);
unsafe { usb_hcd_end_port_resume(core::ptr::addr_of_mut!((*uhci_to_hcd(uhci)).self_), port) };
}

fn wait_for_HP(uhci: *mut uhci_hcd, port_addr: u64) {
let mut i: i32 = 10;
while i < 250 {
if (uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32) & 2 != 0 {
return;
}
udelay(10u64);
i = i.wrapping_add(10);
}
// Log a warning?
}

fn uhci_check_ports(uhci: *mut uhci_hcd) {
let mut port: u32 = 0;
let mut port_addr: u64 = 0;
let mut status: i32 = 0;
while port < ((unsafe { (*uhci).rh_numports }) as u32) {
port_addr = 16u32.wrapping_add(2u32.wrapping_mul(port)) as u64;
status = uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32;
if unlikely__2!(status & 512) != 0 {
if time_after_eq!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, unsafe { (*uhci).ports_timeout }) {
CLR_RH_PORTSTAT!(512i32, status, uhci, port_addr);
udelay(10u64);
// HP's server management chip requires
// a longer delay.
if ((unsafe { (*uhci).__slate_bits_1.__get_wait_for_hp() }) as i32) != 0 {
wait_for_HP(uhci, port_addr);
}
// If the port was enabled before, turning
// reset on caused a port enable change.
// Turning reset off causes a port connect
// status change.  Clear these changes.
CLR_RH_PORTSTAT!(10i32, status, uhci, port_addr);
SET_RH_PORTSTAT!(4i32, status, uhci, port_addr);
}
}
if unlikely__2!(status & 64) != 0 {
if test_bit__3!(port, unsafe { core::ptr::addr_of_mut!((*uhci).resuming_ports) }) == 0 {
// Port received a wakeup request
set_bit(port as i64, (unsafe { core::ptr::addr_of_mut!((*uhci).resuming_ports) }) as *mut u64);
unsafe {
(*uhci).ports_timeout = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(40u32));
}
unsafe { usb_hcd_start_port_resume(core::ptr::addr_of_mut!((*uhci_to_hcd(uhci)).self_), port as i32) };
// Make sure we see the port again
// after the resuming period is over.
unsafe { mod_timer(core::ptr::addr_of_mut!((*uhci_to_hcd(uhci)).rh_timer), (*uhci).ports_timeout) };
} else {
if time_after_eq!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, unsafe { (*uhci).ports_timeout }) {
uhci_finish_suspend(uhci, port as i32, port_addr);
}
}
}
port = port.wrapping_add(1);
}
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_hub_status_data")]
extern "C" fn uhci_hub_status_data(hcd: *mut usb_hcd, buf: *mut u8) -> i32 {
let mut status: i32 = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut uhci: *mut uhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
uhci = hcd_to_uhci(hcd);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
status = 0;
spin_lock_irqsave!(core::ptr::addr_of_mut!((*uhci).lock), *__slate_slot_1);
uhci_scan_schedule(uhci);
if !(HCD_HW_ACCESSIBLE!(hcd) == 0 || ((*uhci).__slate_bits_0.__get_dead() as i32) != 0) {
uhci_check_ports(uhci);
status = get_hub_status_data(uhci, buf);
let __t0: u32 = (*uhci).rh_state as u32;
if __t0 == 1 {
if status != 0 || (*uhci).resuming_ports != 0 {
status = 1;
usb_hcd_resume_root_hub(hcd);
}
// if port change, ask to be resumed
} else {
if __t0 == 2 {
if status != 0 {
wakeup_rh(uhci);
}
// if port change, auto start
} else {
if __t0 == 5 {
if any_ports_active(uhci) == 0 {
(*uhci).rh_state = 6;
(*uhci).auto_stop_time = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(1000);
}
// are any devices attached?
} else {
if __t0 == 6 {
if any_ports_active(uhci) != 0 {
(*uhci).rh_state = 5;
} else {
if time_after_eq!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), unsafe { (*uhci).auto_stop_time }) && ((*uhci).__slate_bits_1.__get_wait_for_hp() as i32) == 0 {
suspend_rh(uhci, 2u32);
}
}
// auto-stop if nothing connected for 1 second
}
}
}
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*uhci).lock), *__slate_slot_1);
return status;
}
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_hub_control")]
extern "C" fn uhci_hub_control(hcd: *mut usb_hcd, typeReq: u16, wValue: u16, wIndex: u16, buf: *mut u8, wLength: u16) -> i32 {
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_58: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut wPortStatus: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_59: *mut u16 = core::ptr::addr_of_mut!(wPortStatus) as *mut u16;
let mut wPortChange: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_60: *mut u16 = core::ptr::addr_of_mut!(wPortChange) as *mut u16;
let mut port_addr: u64 = unsafe { core::mem::zeroed() };
let mut port: u32 = unsafe { core::mem::zeroed() };
let mut retval: i32 = unsafe { core::mem::zeroed() };
let mut lstatus: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_64: *mut i32 = core::ptr::addr_of_mut!(lstatus) as *mut i32;
let mut status: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_65: *mut i32 = core::ptr::addr_of_mut!(status) as *mut i32;
let mut uhci: *mut uhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
uhci = hcd_to_uhci(hcd);
let __slate_auto_bytes = __slate_slot_65 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_64 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
retval = 0;
port = (wIndex as i32).wrapping_sub(1) as u32;
port_addr = 16u32.wrapping_add(2u32.wrapping_mul(port)) as u64;
let __slate_auto_bytes_2 = __slate_slot_60 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_3 = __slate_slot_59 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_58 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if HCD_HW_ACCESSIBLE!(hcd) == 0 || ((*uhci).__slate_bits_0.__get_dead() as i32) != 0 {
return 110i32.wrapping_neg();
} else {
'__join_0: {
'__join_1: {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*uhci).lock), *__slate_slot_58);
let __t3: i32 = typeReq as i32;
if __t3 == 40960 {
*(buf as *mut u32) = 0;
retval = 4; // hub power
break '__join_0;
} else {
if __t3 == 41728 {
if port >= ((*uhci).rh_numports as u32) {
break '__join_1;
} else {
uhci_check_ports(uhci);
*__slate_slot_65 = uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32;
// Intel controllers report the OverCurrent bit active on.
// VIA controllers report it active off, so we'll adjust the
// bit value.  (It's not standardized in the UHCI spec.)
if ((*uhci).__slate_bits_1.__get_oc_low() as i32) != 0 {
*__slate_slot_65 ^= 1024;
}
// UHCI doesn't support C_RESET (always false)
*__slate_slot_64 = 0;
*__slate_slot_60 = 0;
if *__slate_slot_65 & 2 != 0 {
*__slate_slot_60 = ((*__slate_slot_60 as i32) | 1) as u16;
}
if *__slate_slot_65 & 8 != 0 {
*__slate_slot_60 = ((*__slate_slot_60 as i32) | 2) as u16;
}
if *__slate_slot_65 & 2048 != 0 && !ignore_oc {
*__slate_slot_60 = ((*__slate_slot_60 as i32) | 8) as u16;
}
if test_bit__3!(port, core::ptr::addr_of_mut!((*uhci).port_c_suspend)) != 0 {
*__slate_slot_60 = ((*__slate_slot_60 as i32) | 4) as u16;
*__slate_slot_64 |= 1;
}
if test_bit__3!(port, core::ptr::addr_of_mut!((*uhci).resuming_ports)) != 0 {
*__slate_slot_64 |= 4;
}
// UHCI has no power switching (always on)
*__slate_slot_59 = 256;
if *__slate_slot_65 & 1 != 0 {
*__slate_slot_59 = ((*__slate_slot_59 as i32) | 1) as u16;
}
if *__slate_slot_65 & 4 != 0 {
*__slate_slot_59 = ((*__slate_slot_59 as i32) | 2) as u16;
if *__slate_slot_65 & (4096 | 64) != 0 {
*__slate_slot_59 = ((*__slate_slot_59 as i32) | 4) as u16;
}
}
if *__slate_slot_65 & 1024 != 0 {
*__slate_slot_59 = ((*__slate_slot_59 as i32) | 8) as u16;
}
if *__slate_slot_65 & 512 != 0 {
*__slate_slot_59 = ((*__slate_slot_59 as i32) | 16) as u16;
}
if *__slate_slot_65 & 256 != 0 {
*__slate_slot_59 = ((*__slate_slot_59 as i32) | 512) as u16;
}
if *__slate_slot_60 != 0 {
}
*(buf as *mut u16) = *__slate_slot_59;
*(buf.wrapping_offset(2 as isize) as *mut u16) = *__slate_slot_60;
retval = 4;
break '__join_0;
}
} else {
if __t3 != 8195 {
if __t3 != 8193 {
if __t3 == 8963 {
if port >= ((*uhci).rh_numports as u32) {
break '__join_1;
} else {
let __t1: i32 = wValue as i32;
if __t1 == 2 {
*__slate_slot_65 = uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32;
*__slate_slot_65 |= 4096;
let __v28: i32 = *__slate_slot_65 & !(2048 | 8 | 2 | (8192 | 16384 | 32768));
*__slate_slot_65 = __v28;
uhci_writew(uhci as *const uhci_hcd, *__slate_slot_65 as u16, port_addr as i32);
break '__join_0;
} else {
if __t1 == 4 {
*__slate_slot_65 = uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32;
*__slate_slot_65 |= 512;
let __v24: i32 = *__slate_slot_65 & !(2048 | 8 | 2 | (8192 | 16384 | 32768));
*__slate_slot_65 = __v24;
uhci_writew(uhci as *const uhci_hcd, *__slate_slot_65 as u16, port_addr as i32);
// Reset terminates Resume signalling
uhci_finish_suspend(uhci, port as i32, port_addr);
// USB v2.0 7.1.7.5
(*uhci).ports_timeout = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(40u32));
break '__join_0;
} else {
if __t1 == 8 {
break '__join_0;
} else {
break '__join_1;
}
}
}
}
} else {
if __t3 == 8961 {
if port >= ((*uhci).rh_numports as u32) {
break '__join_1;
} else {
let __t0: i32 = wValue as i32;
if __t0 == 1 {
*__slate_slot_65 = uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32;
let __v22: i32 = *__slate_slot_65 & !(2048 | 8 | 2 | (8192 | 16384 | 32768));
*__slate_slot_65 = __v22;
let __v20: i32 = *__slate_slot_65 & !4;
*__slate_slot_65 = __v20;
*__slate_slot_65 |= (2048 | 8 | 2) & 4;
uhci_writew(uhci as *const uhci_hcd, *__slate_slot_65 as u16, port_addr as i32);
// Disable terminates Resume signalling
uhci_finish_suspend(uhci, port as i32, port_addr);
break '__join_0;
} else {
if __t0 == 17 {
*__slate_slot_65 = uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32;
let __v16: i32 = *__slate_slot_65 & !(2048 | 8 | 2 | (8192 | 16384 | 32768));
*__slate_slot_65 = __v16;
let __v14: i32 = *__slate_slot_65 & !8;
*__slate_slot_65 = __v14;
*__slate_slot_65 |= (2048 | 8 | 2) & 8;
uhci_writew(uhci as *const uhci_hcd, *__slate_slot_65 as u16, port_addr as i32);
break '__join_0;
} else {
if __t0 == 2 {
if (uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32) & 4096 == 0 {
// Make certain the port isn't suspended
uhci_finish_suspend(uhci, port as i32, port_addr);
break '__join_0;
} else {
if !test_and_set_bit(port as i64, core::ptr::addr_of_mut!((*uhci).resuming_ports) as *mut u64) {
SET_RH_PORTSTAT!(64i32, *__slate_slot_65, uhci, port_addr);
// The controller won't allow RD to be set
// if the port is disabled.  When this happens
// just skip the Resume signalling.
if (uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32) & 64 == 0 {
uhci_finish_suspend(uhci, port as i32, port_addr);
} else {
(*uhci).ports_timeout = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(20u32));
}
// USB v2.0 7.1.7.7
break '__join_0;
} else {
break '__join_0;
}
}
} else {
if __t0 == 18 {
clear_bit(port as i64, core::ptr::addr_of_mut!((*uhci).port_c_suspend) as *mut u64);
break '__join_0;
} else {
if __t0 == 8 {
break '__join_1;
} else {
if __t0 == 16 {
*__slate_slot_65 = uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32;
let __v10: i32 = *__slate_slot_65 & !(2048 | 8 | 2 | (8192 | 16384 | 32768));
*__slate_slot_65 = __v10;
let __v8: i32 = *__slate_slot_65 & !2;
*__slate_slot_65 = __v8;
*__slate_slot_65 |= (2048 | 8 | 2) & 2;
uhci_writew(uhci as *const uhci_hcd, *__slate_slot_65 as u16, port_addr as i32);
break '__join_0;
} else {
if __t0 == 19 {
*__slate_slot_65 = uhci_readw(uhci as *const uhci_hcd, port_addr as i32) as i32;
let __v4: i32 = *__slate_slot_65 & !(2048 | 8 | 2 | (8192 | 16384 | 32768));
*__slate_slot_65 = __v4;
let __v2: i32 = *__slate_slot_65 & !2048;
*__slate_slot_65 = __v2;
*__slate_slot_65 |= (2048 | 8 | 2) & 2048;
uhci_writew(uhci as *const uhci_hcd, *__slate_slot_65 as u16, port_addr as i32);
break '__join_0;
} else {
if __t0 == 20 {
break '__join_0;
} else {
break '__join_1;
}
}
}
}
}
}
}
}
}
} else {
if __t3 == 40966 {
retval = min_t__3!(u32, core::mem::size_of::<[u8; 9]>() as u64, wLength) as i32;
memcpy(buf as *mut (), (core::ptr::addr_of!(root_hub_hub_des) as *const u8) as *const (), retval as u64);
if retval > 2 {
*buf.wrapping_offset(2 as isize) = (*uhci).rh_numports as u8;
break '__join_0;
} else {
break '__join_0;
}
} else {
break '__join_1;
}
}
}
}
}
}
}
let __t2: i32 = wValue as i32;
if __t2 != 1 {
if __t2 != 0 {
break '__join_1;
}
}
// We don't implement these
break '__join_0;
}
retval = 32i32.wrapping_neg();
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*uhci).lock), *__slate_slot_58);
return retval;
}
}
// UHCI has no power switching
// UHCI has no power switching
// this driver won't report these
}

/// Finish up a host controller reset and update the recorded state.
fn finish_reset(uhci: *mut uhci_hcd) {
// HCRESET doesn't affect the Suspend, Reset, and Resume Detect
// bits in the port status and control registers.
// We have to clear them by hand.
let mut port: i32 = 0;
while port < unsafe { (*uhci).rh_numports } {
uhci_writew(uhci as *const uhci_hcd, 0u16, 16i32.wrapping_add(port.wrapping_mul(2)));
port = port.wrapping_add(1);
}
unsafe {
(*uhci).resuming_ports = 0;
}
unsafe {
(*uhci).port_c_suspend = 0;
}
unsafe {
(*uhci).rh_state = 0;
}
unsafe {
(*uhci).is_stopped = 9999;
}
clear_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*uhci_to_hcd(uhci)).flags) }) as *mut u64);
}

/// Last rites for a defunct/nonfunctional controller
/// or one we don't want to use any more.
fn uhci_hc_died(uhci: *mut uhci_hcd) {
uhci_get_current_frame_number(uhci);
unsafe { unsafe { (*uhci).reset_hc }.unwrap()(uhci) };
finish_reset(uhci);
unsafe {
(*uhci).__slate_bits_0.__set_dead(1u32);
}
// The current frame may already be partway finished
let __v1: u32 = unsafe { (*uhci).frame_number };
unsafe {
(*uhci).frame_number = __v1.wrapping_add(1);
}
}

/// Initialize a controller that was newly discovered or has lost power
/// or otherwise been reset while it was suspended.  In none of these cases
/// can we be sure of its previous state.
fn check_and_reset_hc(uhci: *mut uhci_hcd) {
if (unsafe { unsafe { (*uhci).check_and_reset_hc }.unwrap()(uhci) }) != 0 {
finish_reset(uhci);
}
}

/// Store the basic register settings needed by the controller.
fn configure_hc(uhci: *mut uhci_hcd) {
// Set the frame length to the default: 1 ms exactly
uhci_writeb(uhci as *const uhci_hcd, 64u8, 12i32);
// Store the frame list base address
uhci_writel(uhci as *const uhci_hcd, (unsafe { (*uhci).frame_dma_handle }) as u32, 8i32);
// Set the current frame number
uhci_writew(uhci as *const uhci_hcd, ((unsafe { (*uhci).frame_number }) & 2047) as u16, 6i32);
// perform any arch/bus specific configuration
if (unsafe { (*uhci).configure_hc }) != None {
unsafe { unsafe { (*uhci).configure_hc }.unwrap()(uhci) };
}
}

fn resume_detect_interrupts_are_broken(uhci: *mut uhci_hcd) -> i32 {
// If we have to ignore overcurrent events then almost by definition
// we can't depend on resume-detect interrupts.
//
// Those interrupts also don't seem to work on ASpeed SoCs.
let __v0: bool = (unsafe { ignore_oc }) || uhci_is_aspeed(uhci as *const uhci_hcd);
if __v0 {
return 1;
}
let __v1: i32 = if (unsafe { (*uhci).resume_detect_interrupts_are_broken }) != None {
unsafe { unsafe { (*uhci).resume_detect_interrupts_are_broken }.unwrap()(uhci) }
} else {
0
};
__v1
}

fn global_suspend_mode_is_broken(uhci: *mut uhci_hcd) -> i32 {
let __v0: i32 = if (unsafe { (*uhci).global_suspend_mode_is_broken }) != None {
unsafe { unsafe { (*uhci).global_suspend_mode_is_broken }.unwrap()(uhci) }
} else {
0
};
__v0
}

fn suspend_rh(uhci: *mut uhci_hcd, new_state: u32) {
let mut auto_stop: i32 = 0;
let mut int_enable: i32 = 2;
// Start off by assuming Resume-Detect interrupts and EGSM work
// and that remote wakeups should be enabled.
let mut egsm_enable: i32 = 8;
let mut wakeup_enable: i32 = 1;
let rhdev: *mut usb_device = unsafe { (*uhci_to_hcd(uhci)).self_.root_hub };
auto_stop = ((new_state as u32) == 2) as i32;
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*rhdev).dev) as *const device, c"%s%s\n".as_ptr() as *const u8, c"suspend_rh".as_ptr() as *mut u8, if auto_stop != 0 { c" (auto-stop)".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}
// In auto-stop mode, we must be able to detect new connections.
// The user can force us to poll by disabling remote wakeup;
// otherwise we will use the EGSM/RD mechanism.
if auto_stop != 0 {
if !device_may_wakeup(unsafe { core::ptr::addr_of_mut!((*rhdev).dev) }) {
int_enable = 0;
egsm_enable = 0;
}
} else {
if ((unsafe { (*rhdev).__slate_bits_1.__get_do_remote_wakeup() }) as i32) == 0 {
wakeup_enable = 0;
}
}
// In bus-suspend mode, we use the wakeup setting specified
// for the root hub.
// UHCI doesn't distinguish between wakeup requests from downstream
// devices and local connect/disconnect events.  There's no way to
// enable one without the other; both are controlled by EGSM.  Thus
// if wakeups are disallowed then EGSM must be turned off -- in which
// case remote wakeup requests from downstream during system sleep
// will be lost.
//
// In addition, if EGSM is broken then we can't use it.  Likewise,
// if Resume-Detect interrupts are broken then we can't use them.
//
// Finally, neither EGSM nor RD is useful by itself.  Without EGSM,
// the RD status bit will never get set.  Without RD, the controller
// won't generate interrupts to tell the system about wakeup events.
let __v0: bool = wakeup_enable == 0 || global_suspend_mode_is_broken(uhci) != 0;
let __v1: bool = __v0 || resume_detect_interrupts_are_broken(uhci) != 0;
if __v1 {
int_enable = 0;
egsm_enable = 0;
}
unsafe {
let __t0 = (int_enable != 0) as u32;
(*uhci).__slate_bits_0.__set_RD_enable(__t0);
}
uhci_writew(uhci as *const uhci_hcd, int_enable as u16, 4i32);
uhci_writew(uhci as *const uhci_hcd, (egsm_enable | 64) as u16, 0i32);
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
}
if true {
break;
}
}
udelay(5u64);
// If we're auto-stopping then no devices have been attached
// for a while, so there shouldn't be any active URBs and the
// controller should stop after a few microseconds.  Otherwise
// we will give the controller one frame to stop.
let __v4: bool = auto_stop == 0 && (uhci_readw(uhci as *const uhci_hcd, 2i32) as i32) & 32 == 0;
if __v4 {
unsafe {
(*uhci).rh_state = 4;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
unsafe { msleep(1u32) };
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
if ((unsafe { (*uhci).__slate_bits_0.__get_dead() }) as i32) != 0 {
return;
}
}
if (uhci_readw(uhci as *const uhci_hcd, 2i32) as i32) & 32 == 0 {
while false {
}
unsafe { _dev_warn(uhci_dev!(uhci) as *const device, c"Controller not stopped yet!\n".as_ptr() as *const u8) };
}
uhci_get_current_frame_number(uhci);
unsafe {
(*uhci).rh_state = new_state;
}
unsafe {
(*uhci).is_stopped = 9999;
}
// If remote wakeup is enabled but either EGSM or RD interrupts
// doesn't work, then we won't get an interrupt when a wakeup event
// occurs.  Thus the suspended root hub needs to be polled.
if wakeup_enable != 0 && (int_enable == 0 || egsm_enable == 0) {
set_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*uhci_to_hcd(uhci)).flags) }) as *mut u64);
} else {
clear_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*uhci_to_hcd(uhci)).flags) }) as *mut u64);
}
uhci_scan_schedule(uhci);
uhci_fsbr_off(uhci);
}

fn start_rh(uhci: *mut uhci_hcd) {
unsafe {
(*uhci).is_stopped = 0;
}
// Clear stale status bits on Aspeed as we get a stale HCH
// which causes problems later on
if uhci_is_aspeed(uhci as *const uhci_hcd) {
uhci_writew(uhci as *const uhci_hcd, uhci_readw(uhci as *const uhci_hcd, 2i32), 2i32);
}
// Mark it configured and running with a 64-byte max packet.
// All interrupts are enabled, even though RESUME won't do anything.
uhci_writew(uhci as *const uhci_hcd, 193u16, 0i32);
uhci_writew(uhci as *const uhci_hcd, 15u16, 4i32);
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
(*uhci).rh_state = 5;
}
set_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*uhci_to_hcd(uhci)).flags) }) as *mut u64);
}

fn wakeup_rh(uhci: *mut uhci_hcd) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*uhci_to_hcd(uhci)).self_.root_hub).dev) as *const device, c"%s%s\n".as_ptr() as *const u8, c"wakeup_rh".as_ptr() as *mut u8, if ((*uhci).rh_state as u32) == 2 { c" (auto-start)".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}
// If we are auto-stopped then no devices are attached so there's
// no need for wakeup signals.  Otherwise we send Global Resume
// for 20 ms.
if ((unsafe { (*uhci).rh_state }) as u32) == 1 {
let mut egsm: u32 = 0;
// Keep EGSM on if it was set before
egsm = ((uhci_readw(uhci as *const uhci_hcd, 0i32) as i32) & 8) as u32;
unsafe {
(*uhci).rh_state = 3;
}
uhci_writew(uhci as *const uhci_hcd, (80u32 | egsm) as u16, 0i32);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
unsafe { msleep(20u32) };
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
if ((unsafe { (*uhci).__slate_bits_0.__get_dead() }) as i32) != 0 {
return;
}
// End Global Resume and wait for EOP to be sent
uhci_writew(uhci as *const uhci_hcd, 64u16, 0i32);
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
}
if true {
break;
}
}
udelay(4u64);
if (uhci_readw(uhci as *const uhci_hcd, 0i32) as i32) & 16 != 0 {
while false {
}
unsafe { _dev_warn(uhci_dev!(uhci) as *const device, c"FGR not stopped yet!\n".as_ptr() as *const u8) };
}
}
start_rh(uhci);
// Restart root hub polling
unsafe { mod_timer(core::ptr::addr_of_mut!((*uhci_to_hcd(uhci)).rh_timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies))) };
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_irq")]
extern "C" fn uhci_irq(hcd: *mut usb_hcd) -> u32 {
let mut status: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u16 = core::ptr::addr_of_mut!(status) as *mut u16;
let mut uhci: *mut uhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
uhci = hcd_to_uhci(hcd);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
// Read the interrupt status, and write it back to clear the
// interrupt cause.  Contrary to the UHCI specification, the
// "HC Halted" status bit is persistent: it is RO, not R/WC.
*__slate_slot_0 = uhci_readw(uhci as *const uhci_hcd, 2i32);
if (*__slate_slot_0 as i32) & !32 == 0 {
return 0;
} else {
'__join_0: {
// shared interrupt, not mine
uhci_writew(uhci as *const uhci_hcd, *__slate_slot_0, 2i32); // Clear it
spin_lock(core::ptr::addr_of_mut!((*uhci).lock));
if unlikely!(((*uhci).__slate_bits_0.__get_is_initialized() as i32) == 0) == 0 {
// not yet configured
if (*__slate_slot_0 as i32) & !(1 | 2 | 4) != 0 {
if (*__slate_slot_0 as i32) & 8 != 0 {
while false {
}
_dev_err(uhci_dev!(uhci) as *const device, c"host system error, PCI problems?\n".as_ptr() as *const u8);
}
if (*__slate_slot_0 as i32) & 16 != 0 {
while false {
}
_dev_err(uhci_dev!(uhci) as *const device, c"host controller process error, something bad happened!\n".as_ptr() as *const u8);
}
if (*__slate_slot_0 as i32) & 32 != 0 {
if ((*uhci).rh_state as u32) >= 5 {
while false {
}
_dev_err(uhci_dev!(uhci) as *const device, c"host controller halted, very bad!\n".as_ptr() as *const u8);
uhci_hc_died(uhci);
usb_hc_died(hcd);
// Force a callback in case there are
// pending unlinks
mod_timer(core::ptr::addr_of_mut!((*hcd).rh_timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)));
}
}
}
if (*__slate_slot_0 as i32) & 4 != 0 {
spin_unlock(core::ptr::addr_of_mut!((*uhci).lock));
usb_hcd_poll_rh_status(hcd);
break '__join_0;
} else {
uhci_scan_schedule(uhci);
}
}
spin_unlock(core::ptr::addr_of_mut!((*uhci).lock));
}
return 1;
}
}
// Print the schedule for debugging
}

/// Store the current frame number in uhci->frame_number if the controller
/// is running.  Expand from 11 bits (of which we use only 10) to a
/// full-sized integer.
///
/// Like many other parts of the driver, this code relies on being polled
/// more than once per second as long as the controller is running.
fn uhci_get_current_frame_number(uhci: *mut uhci_hcd) {
if (unsafe { (*uhci).is_stopped }) == 0 {
let delta: u32 = ((uhci_readw(uhci as *const uhci_hcd, 6i32) as i32) as u32).wrapping_sub(unsafe { (*uhci).frame_number }) & 1023u32;
let __v1: u32 = unsafe { (*uhci).frame_number };
unsafe {
(*uhci).frame_number = __v1.wrapping_add(delta);
}
}
}

/// De-allocate all resources
fn release_uhci(uhci: *mut uhci_hcd) {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
unsafe {
(*uhci).__slate_bits_0.__set_is_initialized(0u32);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
unsafe { debugfs_lookup_and_remove((*uhci_to_hcd(uhci)).self_.bus_name, uhci_debugfs_root) };
for i in 0i32..11i32 {
uhci_free_qh(uhci, unsafe { *((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(i as isize) });
}
uhci_free_td(uhci, unsafe { (*uhci).term_td });
unsafe { dma_pool_destroy((*uhci).qh_pool) };
unsafe { dma_pool_destroy((*uhci).td_pool) };
unsafe { kfree((*uhci).frame_cpu as *const ()) };
dma_free_coherent(uhci_dev!(uhci), 1024u64.wrapping_mul(core::mem::size_of::<u32>() as u64), (unsafe { (*uhci).frame }) as *mut (), unsafe { (*uhci).frame_dma_handle });
}

/// Allocate a frame list, and then setup the skeleton
///
/// The hardware doesn't really know any difference
/// in the queues, but the order does matter for the
/// protocols higher up.  The order in which the queues
/// are encountered by the hardware is:
///
///  - All isochronous events are handled before any
///    of the queues. We don't do that here, because
///    we'll create the actual TD entries on demand.
///  - The first queue is the high-period interrupt queue.
///  - The second queue is the period-1 interrupt and async
///    (low-speed control, full-speed control, then bulk) queue.
///  - The third queue is the terminating bandwidth reclamation queue,
///    which contains no members, loops back to itself, and is present
///    only when FSBR is on and there are no full-speed control or bulk QHs.
#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_start")]
extern "C" fn uhci_start(hcd: *mut usb_hcd) -> i32 {
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: spinlock = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut retval: i32 = unsafe { core::mem::zeroed() };
let mut uhci: *mut uhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
{
uhci = hcd_to_uhci(hcd);
retval = 16i32.wrapping_neg();
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
(*hcd).__slate_bits_0.__set_uses_new_polling(1u32);
// Accept arbitrarily long scatter-gather lists
if unsafe { (*hcd).localmem_pool }.is_null() {
(*hcd).self_.sg_tablesize = 4294967295u32;
}
}
loop {
spinlock_check(core::ptr::addr_of_mut!((*uhci).lock));
*core::ptr::addr_of_mut!((*uhci).lock) = *{
__slate_compound_22 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_22)
};
if true {
break;
}
}
timer_setup!(core::ptr::addr_of_mut!((*uhci).fsbr_timer), Some(uhci_fsbr_timeout), 0i32);
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*uhci).idle_qh_list));
loop {
__init_waitqueue_head(core::ptr::addr_of_mut!((*uhci).waitqh), c"&uhci->waitqh".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key));
if true {
break;
}
}
(*uhci).frame = dma_alloc_coherent(uhci_dev!(uhci), 1024u64.wrapping_mul(core::mem::size_of::<u32>() as u64), core::ptr::addr_of_mut!((*uhci).frame_dma_handle), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *mut u32;
if unsafe { (*uhci).frame }.is_null() {
while false {
}
_dev_err(uhci_dev!(uhci) as *const device, c"unable to allocate consistent memory for frame list\n".as_ptr() as *const u8);
} else {
__obj_size = size_mul(core::mem::size_of::<*mut ()>() as u64, 1024u64);
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_9 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_10 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
alloc_tag_restore!();
} else {
*__slate_slot_10 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
}
(*uhci).frame_cpu = *__slate_slot_10 as *mut *mut ();
if !unsafe { (*uhci).frame_cpu }.is_null() {
(*uhci).td_pool = dma_pool_create(c"uhci_td".as_ptr() as *const u8, uhci_dev!(uhci), core::mem::size_of::<uhci_td>() as u64, 16u64, 0u64);
if unsafe { (*uhci).td_pool }.is_null() {
while false {
}
_dev_err(uhci_dev!(uhci) as *const device, c"unable to create td dma_pool\n".as_ptr() as *const u8);
} else {
(*uhci).qh_pool = dma_pool_create(c"uhci_qh".as_ptr() as *const u8, uhci_dev!(uhci), core::mem::size_of::<uhci_qh>() as u64, 16u64, 0u64);
if unsafe { (*uhci).qh_pool }.is_null() {
while false {
}
_dev_err(uhci_dev!(uhci) as *const device, c"unable to create qh dma_pool\n".as_ptr() as *const u8);
} else {
(*uhci).term_td = uhci_alloc_td(uhci);
if unsafe { (*uhci).term_td }.is_null() {
while false {
}
_dev_err(uhci_dev!(uhci) as *const device, c"unable to allocate terminating TD\n".as_ptr() as *const u8);
} else {
*__slate_slot_15 = 0;
'__join_22: {
while *__slate_slot_15 < 11 {
*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(*__slate_slot_15 as isize) = uhci_alloc_qh(uhci, core::ptr::null_mut::<usb_device>(), core::ptr::null_mut::<usb_host_endpoint>());
if unsafe { *((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(*__slate_slot_15 as isize) }.is_null() {
break '__join_22;
} else {
let __v8: i32 = *__slate_slot_15;
*__slate_slot_15 = __v8.wrapping_add(1);
}
}
// 8 Interrupt queues; link all higher int queues to int1 = async
*__slate_slot_15 = 2i32;
while *__slate_slot_15 < 9 {
(*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(*__slate_slot_15 as isize))).link = LINK_TO_QH!(uhci, unsafe { *((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(9 as isize) });
let __v6: i32 = *__slate_slot_15;
*__slate_slot_15 = __v6.wrapping_add(1);
}
(*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(9 as isize))).link = UHCI_PTR_TERM!(uhci);
(*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(10 as isize))).link = LINK_TO_QH!(uhci, unsafe { *((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(10 as isize) });
// This dummy TD is to work around a bug in Intel PIIX controllers
uhci_fill_td(uhci, (*uhci).term_td, 0u32, (uhci_explen!(0i32) | 32512i32 | 105) as u32, 0u32);
(*(*uhci).term_td).link = UHCI_PTR_TERM!(uhci);
let __v4: u32 = LINK_TO_TD!(uhci, unsafe { (*uhci).term_td });
(*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(10 as isize))).element = __v4;
(*(*((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(9 as isize))).element = __v4;
// Fill the frame list: make all entries point to the proper
// interrupt queue.
*__slate_slot_15 = 0;
while *__slate_slot_15 < 1024 {
// Only place we don't use the frame list routines
*unsafe { (*uhci).frame }.wrapping_offset(*__slate_slot_15 as isize) = uhci_frame_skel_link(uhci, *__slate_slot_15);
let __v3: i32 = *__slate_slot_15;
*__slate_slot_15 = __v3.wrapping_add(1);
}
// Some architectures require a full mb() to enforce completion of
// the memory writes above before the I/O transfers in configure_hc().
loop {
kcsan_mb!();
core::arch::asm!("mfence", options(nostack, att_syntax));
if true {
break;
}
}
spin_lock_irq(core::ptr::addr_of_mut!((*uhci).lock));
configure_hc(uhci);
(*uhci).__slate_bits_0.__set_is_initialized(1u32);
start_rh(uhci);
spin_unlock_irq(core::ptr::addr_of_mut!((*uhci).lock));
return 0;
}
while false {
}
_dev_err(uhci_dev!(uhci) as *const device, c"unable to allocate QH\n".as_ptr() as *const u8);
*__slate_slot_15 = 0;
while *__slate_slot_15 < 11 {
if !unsafe { *((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(*__slate_slot_15 as isize) }.is_null() {
uhci_free_qh(uhci, *((*uhci).skelqh.as_mut_ptr() as *mut *mut uhci_qh).wrapping_offset(*__slate_slot_15 as isize));
}
let __v1: i32 = *__slate_slot_15;
*__slate_slot_15 = __v1.wrapping_add(1);
}
uhci_free_td(uhci, (*uhci).term_td);
}
dma_pool_destroy((*uhci).qh_pool);
}
dma_pool_destroy((*uhci).td_pool);
}
kfree((*uhci).frame_cpu as *const ());
}
dma_free_coherent(uhci_dev!(uhci), 1024u64.wrapping_mul(core::mem::size_of::<u32>() as u64), (*uhci).frame as *mut (), (*uhci).frame_dma_handle);
}
debugfs_lookup_and_remove((*hcd).self_.bus_name, uhci_debugfs_root);
return retval;
}
// error exits:
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_stop")]
extern "C" fn uhci_stop(hcd: *mut usb_hcd) {
let uhci: *mut uhci_hcd = hcd_to_uhci(hcd);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
if HCD_HW_ACCESSIBLE!(hcd) != 0 && ((unsafe { (*uhci).__slate_bits_0.__get_dead() }) as i32) == 0 {
uhci_hc_died(uhci);
}
uhci_scan_schedule(uhci);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
unsafe { synchronize_irq((*hcd).irq) };
unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*uhci).fsbr_timer)) };
release_uhci(uhci);
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_rh_suspend")]
extern "C" fn uhci_rh_suspend(hcd: *mut usb_hcd) -> i32 {
let uhci: *mut uhci_hcd = hcd_to_uhci(hcd);
let mut rc: i32 = 0;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
if HCD_HW_ACCESSIBLE!(hcd) == 0 {
rc = 108i32.wrapping_neg();
} else {
if ((unsafe { (*uhci).__slate_bits_0.__get_dead() }) as i32) == 0 {
if ((unsafe { (*(*hcd).self_.root_hub).__slate_bits_1.__get_do_remote_wakeup() }) as i32) != 0 && (unsafe { (*uhci).resuming_ports }) != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, uhci_dev!(uhci) as *const device, c"suspend failed because a port is resuming\n".as_ptr() as *const u8) };
}
rc = 16i32.wrapping_neg();
} else {
suspend_rh(uhci, 1u32);
}
}
}
// Dead controllers tell no tales
// Once the controller is stopped, port resumes that are already
// in progress won't complete.  Hence if remote wakeup is enabled
// for the root hub and any ports are in the middle of a resume or
// remote wakeup, we must fail the suspend.
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
rc
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_rh_resume")]
extern "C" fn uhci_rh_resume(hcd: *mut usb_hcd) -> i32 {
let uhci: *mut uhci_hcd = hcd_to_uhci(hcd);
let mut rc: i32 = 0;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
if HCD_HW_ACCESSIBLE!(hcd) == 0 {
rc = 108i32.wrapping_neg();
} else {
if ((unsafe { (*uhci).__slate_bits_0.__get_dead() }) as i32) == 0 {
wakeup_rh(uhci);
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
rc
}

/// Wait until a particular device/endpoint's QH is idle, and free it
#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_hcd_endpoint_disable")]
extern "C" fn uhci_hcd_endpoint_disable(hcd: *mut usb_hcd, hep: *mut usb_host_endpoint) {
let mut __int: i64 = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __ret_1: i32 = unsafe { core::mem::zeroed() };
let mut qh: core::mem::MaybeUninit<*mut uhci_qh> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut uhci_qh = core::ptr::addr_of_mut!(qh) as *mut *mut uhci_qh;
let mut uhci: *mut uhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
uhci = hcd_to_uhci(hcd);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
spin_lock_irq(core::ptr::addr_of_mut!((*uhci).lock));
*__slate_slot_12 = (*hep).hcpriv as *mut uhci_qh;
if !(*__slate_slot_12).is_null() {
while (*(*__slate_slot_12)).state != 1 {
let __v10: i32 = (*uhci).num_waiting;
(*uhci).num_waiting = __v10.wrapping_add(1);
spin_unlock_irq(core::ptr::addr_of_mut!((*uhci).lock));
__ret_1 = 0;
might_sleep!();
if (*(*__slate_slot_12)).state != 1 {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret = 0;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_5), if false { 1i32 } else { 0i32 });
'__join_4: {
'__join_5: {
loop {
__int = prepare_to_wait_event(core::ptr::addr_of_mut!((*uhci).waitqh), core::ptr::addr_of_mut!(*__slate_slot_5), 1i32);
if (*(*__slate_slot_12)).state == 1 {
break '__join_5;
} else {
if ___wait_is_interruptible!(1i32) && __int != 0 {
break;
} else {
schedule();
if (*(*__slate_slot_12)).state == 1 {
break '__join_5;
}
}
}
}
__ret = __int;
break '__join_4;
}
finish_wait(core::ptr::addr_of_mut!((*uhci).waitqh), core::ptr::addr_of_mut!(*__slate_slot_5));
}
__ret_1 = __ret as i32;
}
spin_lock_irq(core::ptr::addr_of_mut!((*uhci).lock));
let __v1: i32 = (*uhci).num_waiting;
(*uhci).num_waiting = __v1.wrapping_sub(1);
}
uhci_free_qh(uhci, *__slate_slot_12);
}
spin_unlock_irq(core::ptr::addr_of_mut!((*uhci).lock));
}
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_hcd_get_frame_number")]
extern "C" fn uhci_hcd_get_frame_number(hcd: *mut usb_hcd) -> i32 {
let uhci: *mut uhci_hcd = hcd_to_uhci(hcd);
let mut frame_number: u32 = 0;
let mut delta: u32 = 0;
// Minimize latency by avoiding the spinlock
frame_number = unsafe { (*uhci).frame_number };
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
delta = ((uhci_readw(uhci as *const uhci_hcd, 6i32) as i32) as u32).wrapping_sub(frame_number) & 1023u32;
frame_number.wrapping_add(delta) as i32
}

/// Determines number of ports on controller
fn uhci_count_ports(hcd: *mut usb_hcd) -> i32 {
let uhci: *mut uhci_hcd = hcd_to_uhci(hcd);
let io_size: u32 = (unsafe { (*hcd).rsrc_len }) as u32;
// The UHCI spec says devices must have 2 ports, and goes on to say
// they may have more but gives no way to determine how many there
// are.  However according to the UHCI spec, Bit 7 of the port
// status and control register is always set to 1.  So we try to
// use this to our advantage.  Another common failure mode when
// a nonexistent register is addressed is to return all ones, so
// we test for that also.
let mut port: i32 = 0;
while (port as u32) < io_size.wrapping_sub(16) / 2 {
let portstatus: u32 = uhci_readw(uhci as *const uhci_hcd, 16i32.wrapping_add(port.wrapping_mul(2))) as u32;
if portstatus & 128 == 0 || portstatus == 65535 {
break;
}
port = port.wrapping_add(1);
}
if false {
while false {
}
unsafe { _dev_info(uhci_dev!(uhci) as *const device, c"detected %d ports\n".as_ptr() as *const u8, port) };
}
// Anything greater than 7 is weird so we'll ignore it.
if port > 7 {
while false {
}
unsafe { _dev_info(uhci_dev!(uhci) as *const device, c"port count misdetected? forcing to 2 ports\n".as_ptr() as *const u8) };
port = 2;
}
port
}

static hcd_name: [u8; 9] = [117, 104, 99, 105, 95, 104, 99, 100, 0];

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_pci_reset_hc")]
extern "C" fn uhci_pci_reset_hc(uhci: *mut uhci_hcd) {
unsafe { uhci_reset_hc(to_pci_dev!(unsafe { (*uhci_to_hcd(uhci)).self_.controller }), (*uhci).io_addr) };
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_pci_check_and_reset_hc")]
extern "C" fn uhci_pci_check_and_reset_hc(uhci: *mut uhci_hcd) -> i32 {
unsafe { uhci_check_and_reset_hc(to_pci_dev!(unsafe { (*uhci_to_hcd(uhci)).self_.controller }), (*uhci).io_addr) }
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_pci_configure_hc")]
extern "C" fn uhci_pci_configure_hc(uhci: *mut uhci_hcd) {
let pdev: *mut pci_dev = to_pci_dev!(unsafe { (*uhci_to_hcd(uhci)).self_.controller });
// Enable PIRQ
unsafe { pci_write_config_word(pdev as *const pci_dev, 192i32, 8192u16) };
// Disable platform-specific non-PME# wakeup
if ((unsafe { (*pdev).vendor }) as i32) == 32902 {
unsafe { pci_write_config_byte(pdev as *const pci_dev, 196i32, 0u8) };
}
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_pci_resume_detect_interrupts_are_broken")]
extern "C" fn uhci_pci_resume_detect_interrupts_are_broken(uhci: *mut uhci_hcd) -> i32 {
let mut port: i32 = 0;
match (unsafe { (*to_pci_dev!(unsafe { (*uhci_to_hcd(uhci)).self_.controller })).vendor }) as i32 {
6048 => {
return 1;
// Genesys Logic's GL880S controllers don't generate
// resume-detect interrupts.
}
32902 => {
port = 0;
while port < unsafe { (*uhci).rh_numports } {
if (__inw(unsafe { (*uhci).io_addr }.wrapping_add(16).wrapping_add(port.wrapping_mul(2) as u64) as u16) as i32) & 1024 != 0 {
return 1;
}
port = port.wrapping_add(1);
}
// Some of Intel's USB controllers have a bug that causes
// resume-detect interrupts if any port has an over-current
// condition.  To make matters worse, some motherboards
// hardwire unused USB ports' over-current inputs active!
// To prevent problems, we will not enable resume-detect
// interrupts if any ports are OC.
}
_ => {
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_pci_global_suspend_mode_is_broken")]
extern "C" fn uhci_pci_global_suspend_mode_is_broken(uhci: *mut uhci_hcd) -> i32 {
let mut port: i32 = 0;
// One of Asus's motherboards has a bug which causes it to
// wake up immediately from suspend-to-RAM if any of the ports
// are connected.  In such cases we will not set EGSM.
let sys_info: *const u8 = unsafe { dmi_get_system_info(14i32) };
let __v0: bool = !sys_info.is_null() && (unsafe { strcmp(sys_info, core::ptr::addr_of!(bad_Asus_board) as *const u8) }) == 0;
if __v0 {
port = 0;
while port < unsafe { (*uhci).rh_numports } {
if (__inw(unsafe { (*uhci).io_addr }.wrapping_add(16).wrapping_add(port.wrapping_mul(2) as u64) as u16) as i32) & 1 != 0 {
return 1;
}
port = port.wrapping_add(1);
}
}
0
}

static bad_Asus_board: [u8; 6] = [65, 55, 86, 56, 88, 0];

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_pci_init")]
extern "C" fn uhci_pci_init(hcd: *mut usb_hcd) -> i32 {
let uhci: *mut uhci_hcd = hcd_to_uhci(hcd);
unsafe {
(*uhci).io_addr = (*hcd).rsrc_start;
}
unsafe {
(*uhci).rh_numports = uhci_count_ports(hcd);
}
// Intel controllers report the OverCurrent bit active on.  VIA
// and ZHAOXIN controllers report it active off, so we'll adjust
// the bit value.  (It's not standardized in the UHCI spec.)
let __v0: bool = ((unsafe { (*to_pci_dev!(unsafe { (*uhci_to_hcd(uhci)).self_.controller })).vendor }) as i32) == 4358 || ((unsafe { (*to_pci_dev!(unsafe { (*uhci_to_hcd(uhci)).self_.controller })).vendor }) as i32) == 7447;
if __v0 {
unsafe {
(*uhci).__slate_bits_1.__set_oc_low(1u32);
}
}
// HP's server management chip requires a longer port reset delay.
if ((unsafe { (*to_pci_dev!(unsafe { (*uhci_to_hcd(uhci)).self_.controller })).vendor }) as i32) == 4156 {
unsafe {
(*uhci).__slate_bits_1.__set_wait_for_hp(1u32);
}
}
// Intel controllers use non-PME wakeup signalling
if ((unsafe { (*to_pci_dev!(unsafe { (*uhci_to_hcd(uhci)).self_.controller })).vendor }) as i32) == 32902 {
unsafe { device_set_wakeup_capable(uhci_dev!(uhci), true) };
}
// Set up pointers to PCI-specific functions
unsafe {
(*uhci).reset_hc = Some(uhci_pci_reset_hc);
}
unsafe {
(*uhci).check_and_reset_hc = Some(uhci_pci_check_and_reset_hc);
}
unsafe {
(*uhci).configure_hc = Some(uhci_pci_configure_hc);
}
unsafe {
(*uhci).resume_detect_interrupts_are_broken = Some(uhci_pci_resume_detect_interrupts_are_broken);
}
unsafe {
(*uhci).global_suspend_mode_is_broken = Some(uhci_pci_global_suspend_mode_is_broken);
}
// Kick BIOS off this hardware and reset if the controller
// isn't already safely quiescent.
check_and_reset_hc(uhci);
0
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_shutdown")]
extern "C" fn uhci_shutdown(pdev: *mut pci_dev) {
let hcd: *mut usb_hcd = pci_get_drvdata(pdev) as *mut usb_hcd;
uhci_hc_died(hcd_to_uhci(hcd));
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_pci_suspend")]
extern "C" fn uhci_pci_suspend(hcd: *mut usb_hcd, do_wakeup: bool) -> i32 {
let mut __slate_compound_3: pm_message = unsafe { core::mem::zeroed() };
let mut rc: i32 = unsafe { core::mem::zeroed() };
let mut pdev: *mut pci_dev = unsafe { core::mem::zeroed() };
let mut uhci: *mut uhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
uhci = hcd_to_uhci(hcd);
pdev = to_pci_dev!(unsafe { (*uhci_to_hcd(uhci)).self_.controller });
rc = 0;
spin_lock_irq(core::ptr::addr_of_mut!((*uhci).lock));
if !(HCD_HW_ACCESSIBLE!(hcd) == 0 || ((*uhci).__slate_bits_0.__get_dead() as i32) != 0) {
// Already suspended or dead
// All PCI host controllers are required to disable IRQ generation
// at the source, so we must turn off PIRQ.
pci_write_config_word(pdev as *const pci_dev, 192i32, 0u16);
clear_bit(2i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
// Enable platform-specific non-PME# wakeup
if do_wakeup {
if ((*pdev).vendor as i32) == 32902 {
pci_write_config_byte(pdev as *const pci_dev, 196i32, 3u8);
}
}
}
clear_bit(0i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
spin_unlock_irq(core::ptr::addr_of_mut!((*uhci).lock));
synchronize_irq((*hcd).irq);
// Check for race with a wakeup request
if do_wakeup && HCD_WAKEUP_PENDING!(hcd) != 0 {
uhci_pci_resume(hcd, *{
__slate_compound_3 = pm_message { event: 2 };
core::ptr::addr_of_mut!(__slate_compound_3)
});
rc = 16i32.wrapping_neg();
}
return rc;
}
}

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_pci_resume")]
extern "C" fn uhci_pci_resume(hcd: *mut usb_hcd, msg: pm_message) -> i32 {
let hibernated: bool = msg.event == 64;
let uhci: *mut uhci_hcd = hcd_to_uhci(hcd);
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, uhci_dev!(uhci) as *const device, c"%s\n".as_ptr() as *const u8, c"uhci_pci_resume".as_ptr() as *mut u8) };
}
// Since we aren't in D3 any more, it's safe to set this flag
// even if the controller was dead.
set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*hcd).flags) }) as *mut u64);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
// Make sure resume from hibernation re-enumerates everything
if hibernated {
unsafe { unsafe { (*uhci).reset_hc }.unwrap()(uhci) };
finish_reset(uhci);
} else {
check_and_reset_hc(uhci);
}
// The firmware may have changed the controller settings during
// a system wakeup.  Check it and reconfigure to avoid problems.
configure_hc(uhci);
// Tell the core if the controller had to be reset
if ((unsafe { (*uhci).rh_state }) as u32) == 0 {
unsafe { usb_root_hub_lost_power((*hcd).self_.root_hub) };
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*uhci).lock) });
// If interrupts don't work and remote wakeup is enabled then
// the suspended root hub needs to be polled.
if ((unsafe { (*uhci).__slate_bits_0.__get_RD_enable() }) as i32) == 0 && ((unsafe { (*(*hcd).self_.root_hub).__slate_bits_1.__get_do_remote_wakeup() }) as i32) != 0 {
set_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*hcd).flags) }) as *mut u64);
}
// Does the root hub have a port wakeup pending?
unsafe { usb_hcd_poll_rh_status(hcd) };
0
}

static uhci_driver: __SlateConst<hc_driver> = __SlateConst({
let mut __t0: hc_driver = unsafe { core::mem::zeroed() };
__t0.description = unsafe { core::ptr::addr_of!(hcd_name) as *const u8 };
__t0.product_desc = (b"UHCI Host Controller\0".as_ptr() as *mut u8) as *const u8;
__t0.hcd_priv_size = core::mem::size_of::<uhci_hcd>() as u64;
__t0.irq = Some(uhci_irq);
__t0.flags = 18i32;
__t0.reset = Some(uhci_pci_init);
__t0.start = Some(uhci_start);
__t0.pci_suspend = Some(uhci_pci_suspend);
__t0.pci_resume = Some(uhci_pci_resume);
__t0.stop = Some(uhci_stop);
__t0.get_frame_number = Some(uhci_hcd_get_frame_number);
__t0.urb_enqueue = Some(uhci_urb_enqueue);
__t0.urb_dequeue = Some(uhci_urb_dequeue);
__t0.endpoint_disable = Some(uhci_hcd_endpoint_disable);
__t0.hub_status_data = Some(uhci_hub_status_data);
__t0.hub_control = Some(uhci_hub_control);
__t0.bus_suspend = Some(uhci_rh_suspend);
__t0.bus_resume = Some(uhci_rh_resume);
__t0
});

static uhci_pci_ids: __SlateAlign16<[pci_device_id; 2]> = __SlateAlign16([{
let mut __t0: pci_device_id = unsafe { core::mem::zeroed() };
__t0.vendor = 4294967295u32;
__t0.device = 4294967295u32;
__t0.subvendor = 4294967295u32;
__t0.subdevice = 4294967295u32;
__t0.class = 787200u32;
__t0.class_mask = 4294967295u32;
__t0
}, {
let mut __t1: pci_device_id = unsafe { core::mem::zeroed() };
__t1
}]);

#[unsafe(link_section = ".text.slate_distinct.uhci_hcd.uhci_pci_probe")]
extern "C" fn uhci_pci_probe(dev: *mut pci_dev, id: *const pci_device_id) -> i32 {
unsafe { usb_hcd_pci_probe(dev, core::ptr::addr_of!(uhci_driver.0)) }
}

static mut uhci_pci_driver: pci_driver = {
let mut __t0: pci_driver = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(hcd_name) as *const u8 };
__t0.id_table = unsafe { core::ptr::addr_of!(uhci_pci_ids.0) as *const pci_device_id };
__t0.probe = Some(uhci_pci_probe);
__t0.remove = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut pci_dev)>>(usb_hcd_pci_remove as *const ()) };
__t0.shutdown = Some(uhci_shutdown);
__t0.driver = {
let mut __t1: device_driver = unsafe { core::mem::zeroed() };
__t1.pm = unsafe { core::ptr::addr_of!(usb_hcd_pci_pm_ops) };
__t1
};
__t0
};

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_561: [u8; 31] = [117, 104, 99, 105, 95, 104, 99, 100, 46, 115, 111, 102, 116, 100, 101, 112, 61, 112, 114, 101, 58, 32, 101, 104, 99, 105, 95, 112, 99, 105, 0];

#[unsafe(link_section = ".init.text")]
extern "C" fn uhci_hcd_init() -> i32 {
let mut retval: i32 = unsafe { core::mem::zeroed() };
unsafe {
retval = 12i32.wrapping_neg();
if usb_disabled() != 0 {
return 19i32.wrapping_neg();
} else {
uhci_up_cachep = __kmem_cache_create(c"uhci_urb_priv".as_ptr() as *const u8, core::mem::size_of::<urb_priv>() as u32, 0u32, 0u32, None);
if !unsafe { uhci_up_cachep }.is_null() {
retval = pci_register_driver__4!(core::ptr::addr_of_mut!(uhci_pci_driver));
if retval < 0 {
kmem_cache_destroy(uhci_up_cachep);
} else {
return 0;
}
}
return retval;
}
}
}

#[unsafe(link_section = ".exit.text")]
extern "C" fn uhci_hcd_cleanup() {
unsafe { pci_unregister_driver(core::ptr::addr_of_mut!(uhci_pci_driver)) };
unsafe { kmem_cache_destroy(uhci_up_cachep) };
unsafe { debugfs_remove(uhci_debugfs_root) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_uhci_hcd_init_563: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(uhci_hcd_init)) };

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_uhci_hcd_cleanup: Option<unsafe extern "C" fn()> = Some(uhci_hcd_cleanup);

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_564: [u8; 151] = [117, 104, 99, 105, 95, 104, 99, 100, 46, 97, 117, 116, 104, 111, 114, 61, 76, 105, 110, 117, 115, 32, 39, 70, 114, 111, 100, 111, 32, 82, 97, 98, 98, 105, 116, 39, 32, 84, 111, 114, 118, 97, 108, 100, 115, 44, 32, 74, 111, 104, 97, 110, 110, 101, 115, 32, 69, 114, 100, 102, 101, 108, 116, 44, 32, 82, 97, 110, 100, 121, 32, 68, 117, 110, 108, 97, 112, 44, 32, 71, 101, 111, 114, 103, 32, 65, 99, 104, 101, 114, 44, 32, 68, 101, 116, 105, 32, 70, 108, 105, 101, 103, 108, 44, 32, 84, 104, 111, 109, 97, 115, 32, 83, 97, 105, 108, 101, 114, 44, 32, 82, 111, 109, 97, 110, 32, 87, 101, 105, 115, 115, 103, 97, 101, 114, 98, 101, 114, 44, 32, 65, 108, 97, 110, 32, 83, 116, 101, 114, 110, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_565: [u8; 68] = [117, 104, 99, 105, 95, 104, 99, 100, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 85, 83, 66, 32, 85, 110, 105, 118, 101, 114, 115, 97, 108, 32, 72, 111, 115, 116, 32, 67, 111, 110, 116, 114, 111, 108, 108, 101, 114, 32, 73, 110, 116, 101, 114, 102, 97, 99, 101, 32, 100, 114, 105, 118, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_566: [u8; 40] = [117, 104, 99, 105, 95, 104, 99, 100, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 117, 115, 98, 47, 104, 111, 115, 116, 47, 117, 104, 99, 105, 45, 104, 99, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_567: [u8; 21] = [117, 104, 99, 105, 95, 104, 99, 100, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string___kmem_cache_default_args_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmem_cache_default_args_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_uhci_activate_qh_1: [u8; 1] = *b"\0";

static __slate_asm_string_uhci_activate_qh_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_cleanup_queue_13: [u8; 1] = *b"\0";

static __slate_asm_string_uhci_cleanup_queue_14: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_cleanup_queue_17: [u8; 1] = *b"\0";

static __slate_asm_string_uhci_cleanup_queue_18: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_fixup_short_transfer_2: [u8; 1] = *b"\0";

static __slate_asm_string_uhci_fixup_short_transfer_3: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_free_qh_1: [u8; 1] = *b"\0";

static __slate_asm_string_uhci_free_qh_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_free_qh_8: [u8; 30] = *b"%s %s: qh %p list not empty!\n\0";

static __slate_asm_string_uhci_free_qh_9: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_free_td_10: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_free_td_3: [u8; 29] = *b"%s %s: td %p still in list!\n\0";

static __slate_asm_string_uhci_free_td_4: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_free_td_9: [u8; 32] = *b"%s %s: td %p still in fl_list!\n\0";

static __slate_asm_string_uhci_free_urb_priv_3: [u8; 35] = *b"%s %s: urb %p still on QH's list!\n\0";

static __slate_asm_string_uhci_free_urb_priv_4: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_make_qh_idle_1: [u8; 1] = *b"\0";

static __slate_asm_string_uhci_make_qh_idle_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_remove_td_from_frame_list_1: [u8; 1] = *b"\0";

static __slate_asm_string_uhci_remove_td_from_frame_list_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";

static __slate_asm_string_uhci_unlink_qh_1: [u8; 1] = *b"\0";

static __slate_asm_string_uhci_unlink_qh_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/uhci-q.c\0";
