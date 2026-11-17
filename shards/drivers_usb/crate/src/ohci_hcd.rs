
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
//! SPDX-License-Identifier: GPL-1.0+
//!
//! Open Host Controller Interface (OHCI) driver for USB.
//!
//! Maintainer: Alan Stern <stern@rowland.harvard.edu>
//!
//! (C) Copyright 1999 Roman Weissgaerber <weissg@vienna.at>
//! (C) Copyright 2000-2004 David Brownell <dbrownell@users.sourceforge.net>
//!
//! [ Initialisation is based on Linus'  ]
//! [ uhci code and gregs ohci fragments ]
//! [ (C) Copyright 1999 Linus Torvalds  ]
//! [ (C) Copyright 1999 Gregory P. Smith]
//!
//!
//! OHCI is the main "non-Intel/VIA" standard for USB 1.1 host controller
//! interfaces (though some non-x86 Intel chips use it).  It supports
//! smarter hardware than UHCI.  A download link for the spec available
//! through the https://www.usb.org website.
//!
//! This file is licenced under the GPL.
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
//! # chars in a path name including nul
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
//! New user namespace
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
//! number of env pointers
//! Generic info of form tag = "info"
//! Resource type
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
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! This is arbitrary.
//! From USB 2.0 spec Table 11-13, offset 7, a hub can
//! have up to 255 ports. The most yet reported is 10.
//! Upcoming hardware might raise that limit.
//! Because the arrays need to add a bit for hub status data, we
//! use 31, so plus one evens out to four bytes.
//! __get_unaligned_t - read an unaligned value from memory.
//! @type:	the type to load from the pointer.
//! @ptr:	the pointer to load from.
//!
//! Use memcpy to affect an unaligned type sized load avoiding undefined behavior
//! from approaches like type punning that require -fno-strict-aliasing in order
//! to be correct. As type may be const, use __unqual_scalar_typeof to map to a
//! non-const type - you can't memcpy into a const type. The
//! __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
//! expression rather than type, a pointer is used to avoid warnings about mixing
//! the use of 0 and NULL. The void* cast silences ubsan warnings.
//! Version for use where "next" is the address of a local variable
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
// Dummy printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// round_down - round down to next specified power of 2
// @x: the value to round
// @y: multiple to round down to (must be a power of 2)
//
// Rounds @x down to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding down, use rounddown() below.
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
// Broken pipe
// Math result not representable
// Invalid system call number
// Timer expired
// Out of streams resources
// Communication error on send
// Protocol error
// Value too large for defined data type
// Illegal byte sequence
// Message too long
// Operation not supported on transport endpoint
// Cannot send after transport endpoint shutdown
// Operation already in progress
// Operation now in progress
// Remote I/O error
// Memory page has hardware error
// Operation is not supported
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// in_range - Determine if a value lies within a range.
// @val: Value to test.
// @start: First value in range.
// @len: Number of values in range.
//
// This is more efficient than "if (start <= val && val < (start + len))".
// It also gives a different answer if @start + @len overflows the size of
// the type by a sufficient amount to encompass @val.  Decide for yourself
// which behaviour you want, or prove that start + len never overflow.
// Do not blindly replace one form with the other.
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
// SCHED_ISO: reserved but not implemented yet
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
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
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
// Error values that may be returned by PCI functions
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
// task_css_check - obtain css for (task, subsys) w/ extra access conds
// @task: the target task
// @subsys_id: the target subsystem ID
// @__c: extra condition expression to be passed to rcu_dereference_check()
//
// Return the cgroup_subsys_state for the (@task, @subsys_id) pair.  The
// synchronization rules are the same as task_css_set_check().
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

macro_rules! FSMP {
($fi:expr) => {
32767 & 6u32.wrapping_mul($fi.wrapping_sub(210)) / 7
};
}

macro_rules! HCD_HW_ACCESSIBLE {
($hcd:expr) => {
(unsafe { (*$hcd).flags }) & 1u64
};
}

macro_rules! HCD_POLL_RH {
($hcd:expr) => {
(unsafe { (*$hcd).flags }) & 4u64
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

macro_rules! OKAY_TO_TAKEBACK {
($ohci:expr, $ed:expr) => {
(unsafe { (*$ohci).wdh_cnt }.wrapping_sub(unsafe { (*$ed).takeback_wdh_cnt }) as i32) >= 0
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! TD_CC_GET {
($td_p:expr) => {
($td_p) >> 28i32 & 15
};
}

macro_rules! TD_DI_SET {
($X:expr) => {
(($X) & 7) << 21i32
};
}

/// sizeof (struct td) ~= 64 == 2^6 ...
macro_rules! TD_HASH_FUNC {
($td_dma:expr) => {
(($td_dma) ^ ($td_dma) >> 6i32) % 64
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

macro_rules! __cmp_once__5 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: i32;
let __v2: $type = ($x) as $type;
let __v3: $type = $y;
__v = if (__v2 as i32) > (__v3 as i32) { __v2 as i32 } else { __v3 as i32 };
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

/// __flex_counter() - Get pointer to counter member for the given
///                    flexible array, if it was annotated with __counted_by()
/// @FAM: Pointer to flexible array member of an addressable struct instance
///
/// For example, with:
///
/// 	struct foo {
/// 		int counter;
/// 		short array[] __counted_by(counter);
/// 	} *p;
///
/// __flex_counter(p->array) will resolve to &p->counter.
///
/// Note that Clang may not allow this to be assigned to a separate
/// variable; it must be used directly.
///
/// If p->array is unannotated, this returns (void *)NULL.
macro_rules! __flex_counter {
($__obj_ptr:expr) => {
unsafe { core::ptr::addr_of_mut!((*$__obj_ptr).length) }
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

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __le16_to_cpu {
($x:expr) => {
($x) as u16
};
}

macro_rules! __must_be_array__27 {
() => {
core::mem::size_of::<__SlateRecord_ohci_hcd_c_193_4>() as i32
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

/// __put_unaligned_t - write an unaligned value to memory.
/// @type:	the type of the value to store.
/// @val:	the value to store.
/// @ptr:	the pointer to store to.
///
/// Use memcpy to affect an unaligned type sized store avoiding undefined
/// behavior from approaches like type punning that require -fno-strict-aliasing
/// in order to be correct. The void* cast silences ubsan warnings.
macro_rules! __put_unaligned_t {
($type:ty, $val:expr, $ptr:expr) => {
loop {
let mut __put_unaligned_val: $type = $val;
unsafe { __builtin_memcpy($ptr, core::ptr::addr_of_mut!(__put_unaligned_val) as *const (), core::mem::size_of::<$type>() as u64) };
if true {
break;
}
}
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

/// __set_flex_counter() - Set the counter associated with the given flexible
///                        array member that has been annoated by __counted_by().
/// @FAM: Instance of flexible array member within a given struct.
/// @COUNT: Value to store to the __counted_by annotated @FAM_PTR's counter.
///
/// This is a no-op if no annotation exists. Count needs to be checked with
/// overflows_flex_counter_type() before using this function.
macro_rules! __set_flex_counter__2 {
($COUNT:expr, $__obj_ptr:expr) => {
let mut __v: u16;
let __v2: u16 = ($COUNT) as u16;
unsafe {
*__flex_counter!($__obj_ptr) = __v2;
}
__v = __v2;
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
macro_rules! __sign_use {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
};
}

macro_rules! __swab16 {
($x:expr) => {
unsafe { __slate_intrinsic_1a0c079c71348b00(($x) as u16) }
};
}

macro_rules! __swab32 {
($x:expr) => {
unsafe { __slate_intrinsic_e94e9cf815168b51(($x) as u32) }
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
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

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

/// only little endian
macro_rules! big_endian_desc {
() => {
0i32
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
macro_rules! container_of__4 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__11!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__21 {
($ptr:expr) => {
{
let mut __v: *mut ohci_hcd;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__51!() as isize)) }) as *mut ()) as *mut ohci_hcd;
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
macro_rules! container_of__27 {
($ptr:expr) => {
{
let mut __v: *mut ed;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__15!() as isize)) }) as *mut ()) as *mut ed;
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
macro_rules! container_of__29 {
($ptr:expr) => {
{
let mut __v: *mut urb_priv;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__56!() as isize)) }) as *mut ()) as *mut urb_priv;
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
macro_rules! container_of__34 {
($ptr:expr) => {
{
let mut __v: *mut td;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__60!() as isize)) }) as *mut ()) as *mut td;
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
macro_rules! container_of__7 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!() as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! dbg_port {
($hc:expr, $label:expr, $num:expr, $value:expr) => {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(($hc) as *const ohci_hcd)).self_.controller as *const device, c"%s roothub.portstatus [%d] = 0x%08x%s%s%s%s%s%s%s%s%s%s%s%s\n".as_ptr() as *const u8, $label, ($num) as i32, $value, if ($value) & 1048576 != 0 { c" PRSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 524288 != 0 { c" OCIC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 262144 != 0 { c" PSSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 131072 != 0 { c" PESC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 65536 != 0 { c" CSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 512 != 0 { c" LSDA".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 256 != 0 { c" PPS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 16 != 0 { c" PRS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 8 != 0 { c" POCI".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 4 != 0 { c" PSS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 2 != 0 { c" PES".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($value) & 1 != 0 { c" CCS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}
};
}

macro_rules! dbg_port_sw {
($hc:expr, $num:expr, $next:expr, $size:expr, $temp:expr) => {
loop {
if !$next.is_null() {
let s_len: u32 = (unsafe { scnprintf(*$next, *$size as u64, c"roothub.portstatus [%d] 0x%08x%s%s%s%s%s%s%s%s%s%s%s%s\n".as_ptr() as *const u8, $num, $temp, if ($temp) & 1048576 != 0 { c" PRSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 524288 != 0 { c" OCIC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 262144 != 0 { c" PSSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 131072 != 0 { c" PESC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 65536 != 0 { c" CSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 512 != 0 { c" LSDA".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 256 != 0 { c" PPS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 16 != 0 { c" PRS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 8 != 0 { c" POCI".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 4 != 0 { c" PSS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 2 != 0 { c" PES".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 1 != 0 { c" CCS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) }) as u32;
let __v: *mut u32 = $size;
let __v2: u32 = unsafe { *__v };
unsafe {
*__v = __v2.wrapping_sub(s_len);
}
let __v4: *mut *mut u8 = $next;
let __v5: *mut u8 = unsafe { *__v4 };
unsafe {
*__v4 = __v5.wrapping_offset(s_len as isize);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(($hc) as *const ohci_hcd)).self_.controller as *const device, c"roothub.portstatus [%d] 0x%08x%s%s%s%s%s%s%s%s%s%s%s%s\n".as_ptr() as *const u8, $num, $temp, if ($temp) & 1048576 != 0 { c" PRSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 524288 != 0 { c" OCIC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 262144 != 0 { c" PSSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 131072 != 0 { c" PESC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 65536 != 0 { c" CSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 512 != 0 { c" LSDA".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 256 != 0 { c" PPS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 16 != 0 { c" PRS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 8 != 0 { c" POCI".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 4 != 0 { c" PSS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 2 != 0 { c" PES".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ($temp) & 1 != 0 { c" CCS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}
}
if true {
break;
}
}
};
}

/// debugfs_create_file - create a file in the debugfs filesystem
/// @name: a pointer to a string containing the name of the file to create.
/// @mode: the permission that the file should have.
/// @parent: a pointer to the parent dentry for this file.  This should be a
///          directory dentry if set.  If this parameter is NULL, then the
///          file will be created in the root of the debugfs filesystem.
/// @data: a pointer to something that the caller will want to get to later
///        on.  The inode.i_private pointer will point to this value on
///        the open() call.
/// @fops: a pointer to a struct file_operations or struct debugfs_short_fops that
///        should be used for this file.
///
/// This is the basic "create a file" function for debugfs.  It allows for a
/// wide range of flexibility in creating a file, or a directory (if you want
/// to create a directory, the debugfs_create_dir() function is
/// recommended to be used instead.)
///
/// This function will return a pointer to a dentry if it succeeds.  This
/// pointer must be passed to the debugfs_remove() function when the file is
/// to be removed (no automatic cleanup happens if your module is unloaded,
/// you are responsible here.)  If an error occurs, ERR_PTR(-ERROR) will be
/// returned.
///
/// If debugfs is not enabled in the kernel, the value -%ENODEV will be
/// returned.
///
/// If fops points to a struct debugfs_short_fops, then simple_open() will be
/// used for the open, and only read/write/llseek are supported and are proxied,
/// so no module reference or release are needed.
///
/// NOTE: it's expected that most callers should _ignore_ the errors returned
/// by this function. Other debugfs functions handle the fact that the "dentry"
/// passed to them could be an error and they don't crash in that case.
/// Drivers should generally work fine even if debugfs fails to init anyway.
macro_rules! debugfs_create_file {
($name:expr, $mode:expr, $parent:expr, $data:expr, $fops:expr) => {
unsafe { debugfs_create_file_full(($name) as *const u8, ($mode) as u16, $parent, ($data) as *mut (), core::ptr::null::<()>(), $fops) };
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
size_mul($count, (core::mem::size_of::<*mut td>() as u64).wrapping_add(__must_be_array__27!() as u64))
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
1i32.wrapping_neg() < 1
};
}

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
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
macro_rules! list_entry__19 {
($ptr:expr) => {
container_of__27!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__20 {
($ptr:expr) => {
container_of__29!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__3 {
($ptr:expr, $type:ty) => {
container_of__4!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__14 {
($ptr:expr, $type:ty) => {
container_of__7!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__8 {
($ptr:expr) => {
container_of__34!($ptr)
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
macro_rules! list_first_entry__4 {
($ptr:expr, $type:ty) => {
list_entry__3!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__19 {
($ptr:expr) => {
list_entry__8!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__22 {
($ptr:expr) => {
list_entry__20!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__24 {
($ptr:expr) => {
list_entry__19!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__30 {
($ptr:expr, $type:ty) => {
list_entry__14!(unsafe { (*$ptr).next }, $type)
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__21 {
($pos:expr, $member:ident) => {
list_entry__20!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__7 {
($pos:expr, $member:ident) => {
list_entry__8!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__24 {
($pos:expr, $member:ident) => {
list_entry__19!(unsafe { (*$pos).$member.next })
};
}

/// list_prepare_entry - prepare a pos entry for use in list_for_each_entry_continue()
/// @pos:	the type * to use as a start point
/// @head:	the head of the list
/// @member:	the name of the list_head within the struct.
///
/// Prepares a pos entry for use as a start point in list_for_each_entry_continue().
macro_rules! list_prepare_entry__2 {
($pos:expr, $head:expr) => {
{
let __v: *mut td = $pos;
let __v2: *mut td = if !__v.is_null() {
__v
} else {
list_entry__8!($head)
};
__v2
}
};
}

/// list_prev_entry - get the prev element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_prev_entry__2 {
($pos:expr, $member:ident) => {
list_entry__8!(unsafe { (*$pos).$member.prev })
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__5!($type, $x, $y)
};
}

/// mdelay - Inserting a delay based on milliseconds with busy waiting
/// @n:	requested delay in milliseconds
///
/// See udelay() for basic information about mdelay() and it's variants.
///
/// Please double check, whether mdelay() is the right way to go or whether a
/// refactoring of the code is the better variant to be able to use msleep()
/// instead.
macro_rules! mdelay {
($n:expr) => {
if true && ($n) <= 5 {
udelay($n.wrapping_mul(1000) as u64);
} else {
let mut __ms: u64 = ($n) as u64;
loop {
let __v: u64 = __ms;
__ms = __v.wrapping_sub(1);
if __v == 0 {
break;
}
udelay(1000u64);
}
}
};
}

/// mdelay - Inserting a delay based on milliseconds with busy waiting
/// @n:	requested delay in milliseconds
///
/// See udelay() for basic information about mdelay() and it's variants.
///
/// Please double check, whether mdelay() is the right way to go or whether a
/// refactoring of the code is the better variant to be able to use msleep()
/// instead.
macro_rules! mdelay__2 {
($n:expr) => {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) }) && ($n) <= 5 {
udelay($n.wrapping_mul(1000) as u64);
} else {
let mut __ms: u64 = ($n) as u64;
loop {
let __v: u64 = __ms;
__ms = __v.wrapping_sub(1);
if __v == 0 {
break;
}
udelay(1000u64);
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

macro_rules! offsetof__15 {
() => {
core::mem::offset_of!(ed, in_use_list) as u64
};
}

macro_rules! offsetof__5 {
() => {
584u64
};
}

macro_rules! offsetof__11 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, td_list) as u64
};
}

macro_rules! offsetof__51 {
() => {
1080u64
};
}

macro_rules! offsetof__7 {
() => {
24u64
};
}

macro_rules! offsetof__56 {
() => {
16u64
};
}

macro_rules! offsetof__60 {
() => {
core::mem::offset_of!(td, td_list) as u64
};
}

macro_rules! ohci_readl__2 {
($o:expr, $r:expr) => {
_ohci_readl(($o) as *const ohci_hcd, $r);
};
}

macro_rules! ohci_readl {
($o:expr, $r:expr) => {
_ohci_readl(($o) as *const ohci_hcd, $r)
};
}

macro_rules! ohci_writel__2 {
($o:expr, $v:expr, $r:expr) => {
_ohci_writel(($o) as *const ohci_hcd, $v, $r);
};
}

macro_rules! ohci_writel {
($o:expr, $v:expr, $r:expr) => {
_ohci_writel(($o) as *const ohci_hcd, ($v) as u32, $r);
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

/// AMD-756 (D2 rev) reports corrupt register contents in some cases.
/// The erratum (#4) description is incorrect.  AMD's workaround waits
/// till some bits (mostly reserved) are clear; ok for all revs.
macro_rules! read_roothub {
($hc:expr, $register:ident, $mask:expr) => {
{
let mut __v: u32;
let mut __v2: u32 = ohci_readl!($hc, unsafe { core::ptr::addr_of_mut!((*(*$hc).regs).roothub.$register) });
if __v2 == (1i32.wrapping_neg() as u32) {
unsafe {
(*$hc).rh_state = 0;
}
} else {
if (unsafe { (*$hc).flags }) & 1 != 0 {
while __v2 & $mask != 0 {
__v2 = ohci_readl!($hc, unsafe { core::ptr::addr_of_mut!((*(*$hc).regs).roothub.$register) });
}
}
}
__v = __v2;
__v
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

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size__4 {
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
macro_rules! struct_size_t__5 {
($count:expr) => {
struct_size__4!($count)
};
}

/// wrap-aware logic morphed from <linux/jiffies.h>
macro_rules! tick_before {
($t1:expr, $t2:expr) => {
((((($t1) as i16) as i32).wrapping_sub((($t2) as i16) as i32) as i16) as i32) < 0
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
macro_rules! time_after {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__2!(u64) != 0 && typecheck__3!(u64) != 0;
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

/// time_before - returns true if the time a is before time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is before time b, otherwise %false.
macro_rules! time_before {
($a:expr, $b:expr) => {
time_after!($b, $a)
};
}

macro_rules! timer_container_of__3 {
($callback_timer:expr) => {
container_of__21!($callback_timer)
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
macro_rules! usb_gettoggle {
($dev:expr, $ep:expr, $out:expr) => {
(unsafe { *((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset(($out) as isize) }) >> $ep & 1
};
}

macro_rules! usb_pipecontrol {
($pipe:expr) => {
usb_pipetype!($pipe) == 2
};
}

macro_rules! usb_pipedevice {
($pipe:expr) => {
($pipe) >> 8i32 & 127
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

macro_rules! usb_pipetype {
($pipe:expr) => {
($pipe) >> 30i32 & 3
};
}

macro_rules! usb_settoggle {
($dev:expr, $ep:expr, $out:expr, $bit:expr) => {
unsafe {
*((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset(($out) as isize) = *((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset(($out) as isize) & (!(1i32 << $ep) as u32) | ((($bit) << $ep) as u32);
}
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_bool: kernel_param_ops;
    static mut usb_debug_root: *mut dentry;
    static mut kstat: kernel_stat;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __compiletime_assert_0();
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_29();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn scnprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn schedule_timeout_uninterruptible(arg0: i64) -> i64;
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn timer_delete_sync(arg0: *mut timer_list) -> i32;
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn flush_work(arg0: *mut work_struct) -> bool;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn default_llseek(arg0: *mut file, arg1: i64, arg2: i32) -> i64;
    fn simple_read_from_buffer(arg0: *mut (), arg1: u64, arg2: *mut i64, arg3: *const (), arg4: u64) -> i64;
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn device_set_wakeup_capable(arg0: *mut device, arg1: bool);
    fn synchronize_irq(arg0: u32);
    fn free_irq(arg0: u32, arg1: *mut ()) -> *const ();
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
    fn usb_disabled() -> i32;
    fn usb_hcd_link_urb_to_ep(arg0: *mut usb_hcd, arg1: *mut urb) -> i32;
    fn usb_hcd_check_unlink_urb(arg0: *mut usb_hcd, arg1: *mut urb, arg2: i32) -> i32;
    fn usb_hcd_unlink_urb_from_ep(arg0: *mut usb_hcd, arg1: *mut urb);
    fn usb_hcd_giveback_urb(arg0: *mut usb_hcd, arg1: *mut urb, arg2: i32);
    fn usb_hc_died(arg0: *mut usb_hcd);
    fn usb_hcd_poll_rh_status(arg0: *mut usb_hcd);
    fn usb_calc_bus_time(arg0: i32, arg1: i32, arg2: i32, arg3: i32) -> i64;
    fn usb_root_hub_lost_power(arg0: *mut usb_device);
    fn usb_hcd_resume_root_hub(arg0: *mut usb_hcd);
    fn debugfs_create_file_full(arg0: *const u8, arg1: u16, arg2: *mut dentry, arg3: *mut (), arg4: *const (), arg5: *const file_operations) -> *mut dentry;
    fn debugfs_create_file_short(arg0: *const u8, arg1: u16, arg2: *mut dentry, arg3: *mut (), arg4: *const (), arg5: *const debugfs_short_fops) -> *mut dentry;
    fn debugfs_create_dir(arg0: *const u8, arg1: *mut dentry) -> *mut dentry;
    fn debugfs_remove(arg0: *mut dentry);
    fn gen_pool_dma_alloc_align(arg0: *mut gen_pool, arg1: u64, arg2: *mut u64, arg3: i32) -> *mut ();
    fn gen_pool_dma_zalloc_align(arg0: *mut gen_pool, arg1: u64, arg2: *mut u64, arg3: i32) -> *mut ();
    fn gen_pool_free_owner(arg0: *mut gen_pool, arg1: u64, arg2: u64, arg3: *mut *mut ());
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn usb_amd_dev_put();
    fn usb_amd_quirk_pll_disable();
    fn usb_amd_quirk_pll_enable();
    fn sb800_prefetch(arg0: *mut device, arg1: i32);
    fn __compiletime_assert_649();
    fn __compiletime_assert_661();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.bswap.i16"]
    fn __slate_intrinsic_1a0c079c71348b00(arg0: u16) -> u16;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_e1c548fa55baa574(arg0: i64) -> bool;
    #[link_name = "llvm.bswap.i32"]
    fn __slate_intrinsic_e94e9cf815168b51(arg0: u32) -> u32;
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
struct __SlateRecord_types_h_202 {
    refcnt: __SlateRecord_types_h_190,
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
fn size_add(addend1: u64, addend2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_add_overflow!(addend1, addend2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
#[inline(always)]
fn __swab16p(p: *const u16) -> u16 {
__swab16!(unsafe { *p })
}

#[inline(always)]
fn __swab32p(p: *const u32) -> u32 {
__swab32!(unsafe { *p })
}

#[inline(always)]
fn __le32_to_cpup(p: *const u32) -> u32 {
unsafe { *p }
}

#[inline(always)]
fn __le16_to_cpup(p: *const u16) -> u16 {
unsafe { *p }
}

#[inline(always)]
fn __be32_to_cpup(p: *const u32) -> u32 {
__swab32p(p as *const u32)
}

#[inline(always)]
fn __be16_to_cpup(p: *const u16) -> u16 {
__swab16p(p as *const u16)
}

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
struct io_bitmap {
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
// list_for_each_safe - iterate over a list safe against removal of list entry
// @pos:	the &struct list_head to use as a loop cursor.
// @n:		another &struct list_head to use as temporary storage
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
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
// list_for_each_entry_continue_reverse - iterate backwards from the given point
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
//
// Start to iterate over list of given type backwards, continuing after
// the current position.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


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

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
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
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits122U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits127U0,
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
struct futex_private_hash {
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
    __slate_bits_0: __slate_bits::__SlateBits128U0,
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
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
struct task_cputime {
    stime: u64,
    utime: u64,
    sum_exec_runtime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct tlbflush_unmap_batch {
    arch: arch_tlbflush_unmap_batch,
    flush_required: bool,
    writable: bool,
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
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
struct vm_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits284U0,
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
    __slate_bits_0: __slate_bits::__SlateBits73U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits73U1,
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
    __slate_bits_2: __slate_bits::__SlateBits73U2,
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
// This thread should not be frozen
// I am kswapd
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

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
struct uprobe_xol_ops {
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
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
}

fn schedule_work(work: *mut work_struct) -> bool {
queue_work(unsafe { system_percpu_wq }, work)
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
struct ldt_struct {
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
struct mem_cgroup {
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
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
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct atomic_notifier_head {
    lock: spinlock,
    head: *mut notifier_block,
}


// Suits me
// Don't call further
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
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
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
struct irq_domain {
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysctl_h_254 {
    __slate_anon_0: __SlateRecord_sysctl_h_255,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sysctl_h_255 {
    ctl_table: *const ctl_table,
    ctl_table_size: i32,
    used: i32,
    count: i32,
    nreg: i32,
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
    __slate_bits_0: __slate_bits::__SlateBits420U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table {
    procname: *const u8,
    data: *mut (),
    maxlen: i32,
    mode: u16,
    proc_handler: Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>,
    poll: *mut ctl_table_poll,
    extra1: *mut (),
    extra2: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_node {
    node: rb_node,
    header: *mut ctl_table_header,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_header {
    __slate_anon_0: __SlateRecord_sysctl_h_254,
    unregistering: *mut completion,
    ctl_table_arg: *const ctl_table,
    root: *mut ctl_table_root,
    set: *mut ctl_table_set,
    parent: *mut ctl_dir,
    node: *mut ctl_node,
    inodes: hlist_head,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_dir {
    header: ctl_table_header,
    root: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_set {
    is_seen: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>,
    dir: ctl_dir,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_root {
    default_set: ctl_table_set,
    lookup: Option<unsafe extern "C" fn(*mut ctl_table_root) -> *mut ctl_table_set>,
    set_ownership: Option<unsafe extern "C" fn(*mut ctl_table_header, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    permissions: Option<unsafe extern "C" fn(*mut ctl_table_header, *const ctl_table) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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
// XA_STATE() - Declare an XArray operation state.
// @name: Name of this operation state (usually xas).
// @array: Array to operate on.
// @index: Initial index of interest.
//
// Declare and initialise an xa_state on the stack.
// xas_for_each() - Iterate over a range of an XArray.
// @xas: XArray operation state.
// @entry: Entry retrieved from the array.
// @max: Maximum index to retrieve from array.
//
// The loop body will be executed for each entry present in the xarray
// between the current xas position and @max.  @entry will be set to
// the entry retrieved from the xarray.  It is safe to delete entries
// from the array in the loop body.  You should hold either the RCU lock
// or the xa_lock while iterating.  If you need to drop the lock, call
// xas_pause() first.
#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
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
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
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
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
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
struct tty_struct {
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
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
}

fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
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
    __slate_bits_0: __slate_bits::__SlateBits287U0,
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
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
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
struct __SlateRecord_projid_h_24 {
    val: u32,
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
struct request_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
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
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
    __slate_bits_0: __slate_bits::__SlateBits592U0,
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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

// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
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
// kmalloc_objs - Allocate an array of the given type
// @VAR_OR_TYPE: Variable or type to allocate an array of.
// @COUNT: How many elements in the array.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to array of @VAR_OR_TYPE on success,
// or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
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
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
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


// XArray tags, for tagging dirty and writeback pages in the pagecache.
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
struct idr {
    idr_rt: xarray,
    idr_base: u32,
    idr_next: u32,
}


// The IDR API does not expose the tagging functionality of the radix tree
// to users.  Use tag 0 to track whether a node has free space below it.
// Set the IDR flag and the IDR_FREE tag
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
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
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
    __slate_bits_0: __slate_bits::__SlateBits661U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute {
    name: *const u8,
    mode: u16,
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
    __slate_bits_0: __slate_bits::__SlateBits604U0,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct module_attribute {
    attr: attribute,
    show: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *const u8, u64) -> i64>,
    setup: Option<unsafe extern "C" fn(*mut module, *const u8)>,
    test: Option<unsafe extern "C" fn(*mut module) -> i32>,
    free: Option<unsafe extern "C" fn(*mut module)>,
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
// Author(s), use "Name <email>" or just "Name", for multiple
// authors use multiple MODULE_AUTHOR() statements/lines.
// What your module does.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_tree_node {
    r#mod: *mut module,
    node: latch_tree_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_pool {
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
    __slate_bits_0: __slate_bits::__SlateBits707U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits707U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits707U2,
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
    __slate_bits_3: __slate_bits::__SlateBits707U3,
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
struct dev_archdata {
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
struct ep_device {
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
    __slate_bits_0: __slate_bits::__SlateBits704U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mon_bus {
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

fn readl(mut addr: *const ()) -> u32 {
let mut ret: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movl ({1:r}),{0:e}", lateout(reg) __t0, in(reg) addr as *mut u32, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
ret
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
struct ipc_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
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
union __SlateRecord_user_namespace_h_26 {
    __slate_anon_0: __SlateRecord_user_namespace_h_27,
    __slate_anon_1: __SlateRecord_user_namespace_h_31,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_27 {
    extent: [uid_gid_extent; 5],
    nr_extents: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_31 {
    forward: *mut uid_gid_extent,
    reverse: *mut uid_gid_extent,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct binfmt_misc {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cgroup_defs_h_644 {
    __slate_anon_0: __SlateRecord_cgroup_defs_h_645,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_646,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645 {
    __empty_ancestors: __SlateRecord_cgroup_defs_h_645_2,
    ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_646 {
    _root_ancestor: *mut cgroup,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_648,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648 {
    __empty__low_ancestors: __SlateRecord_cgroup_defs_h_648_2,
    _low_ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648_2 {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct disk_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_cluster_info {
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
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulator {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_phy_hdmi_h_27 {
    tmds_char_rate: u64,
    frl: __SlateRecord_phy_hdmi_h_29,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_phy_hdmi_h_29 {
    rate_per_lane: u8,
    lanes: u8,
}


// Endpoints
// in bEndpointAddress
// in bmAttributes
// The USB 3.0 spec redefines bits 5:4 of bmAttributes as interrupt ep type.
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


#[repr(C)]
#[derive(Clone, Copy)]
struct extcon_dev {
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
struct usb_gadget {
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
    __slate_bits_0: __slate_bits::__SlateBits788U0,
    wireless_status: u32,
    wireless_status_work: work_struct,
    dev: device,
    usb_dev: *mut device,
    reset_ws: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_phy_roothub {
}


// this maximum is arbitrary
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
union __SlateRecord_ch11_h_269 {
    hs: __SlateRecord_ch11_h_270,
    ss: __SlateRecord_ch11_h_276,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch11_h_270 {
    DeviceRemovable: [u8; 4],
    PortPwrCtrlMask: [u8; 4],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch11_h_276 {
    bHubHdrDecLat: u8,
    wHubDelay: u16,
    DeviceRemovable: u16,
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
    __slate_bits_0: __slate_bits::__SlateBits792U0,
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
    __slate_bits_0: __slate_bits::__SlateBits782U0,
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
    __slate_bits_1: __slate_bits::__SlateBits782U1,
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
    __slate_bits_2: __slate_bits::__SlateBits782U2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ohci_roothub_regs {
    a: u32,
    b: u32,
    status: u32,
    portstatus: [u32; 15],
}


// urb->transfer_flags:
//
// Note: URB_DIR_IN/OUT is automatically set in usb_submit_urb().
// report short reads as errors
// iso-only; use the first unexpired
// slot in the schedule
// Finish bulk OUT with short packet
// HINT: no non-error interrupt
// needed
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
    __slate_bits_0: __slate_bits::__SlateBits800U0,
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


// NOTE:  these are not the standard USB_ENDPOINT_XFER_* values!!
//
// (yet ... they're the values used by usbfs)
fn usb_pipe_endpoint(dev: *mut usb_device, pipe: u32) -> *mut usb_host_endpoint {
let eps: *mut *mut usb_host_endpoint = if usb_pipein!(pipe) != 0 { unsafe { (*dev).ep_in.as_mut_ptr() as *mut *mut usb_host_endpoint } } else { unsafe { (*dev).ep_out.as_mut_ptr() as *mut *mut usb_host_endpoint } };
unsafe { *eps.wrapping_offset(usb_pipeendpoint!(pipe) as isize) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_106 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_106_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_106_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_106_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_106_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_106_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_112 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_112_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_112_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_112_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_112_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_112_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ohci_hcd_c_193_4 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits560U0 {
        #[bits(1)]
        pub for_kupdate: u32,
        #[bits(1)]
        pub for_background: u32,
        #[bits(1)]
        pub tagged_writepages: u32,
        #[bits(1)]
        pub range_cyclic: u32,
        #[bits(1)]
        pub for_sync: u32,
        #[bits(1)]
        pub unpinned_netfs_wb: u32,
        #[bits(1)]
        pub no_cgroup_owner: u32,
        #[bits(1, access = na)]
        pub __slate_pad_7: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits592U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits284U0 {
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
    pub struct __SlateBits73U0 {
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
    pub struct __SlateBits73U1 {
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
    pub struct __SlateBits661U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits827U0 {
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub implicit_on_dfl: bool,
        #[bits(1)]
        pub threaded: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits287U0 {
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
    pub struct __SlateBits73U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits122U0 {
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
    pub struct __SlateBits128U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits707U0 {
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
    pub struct __SlateBits704U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits707U1 {
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
    pub struct __SlateBits707U2 {
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
    pub struct __SlateBits707U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits604U0 {
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
    pub struct __SlateBits127U0 {
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
    pub struct __SlateBits420U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits792U0 {
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
    pub struct __SlateBits788U0 {
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
    pub struct __SlateBits782U0 {
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
    pub struct __SlateBits782U1 {
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
    pub struct __SlateBits782U2 {
        #[bits(1)]
        pub use_generic_driver: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits800U0 {
        #[bits(1)]
        pub poisoned: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits853U0 {
        #[bits(1)]
        pub ssc: u8,
        #[bits(1)]
        pub set_rate: u8,
        #[bits(1)]
        pub set_lanes: u8,
        #[bits(1)]
        pub set_voltages: u8,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits880U0 {
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
    pub struct __SlateBits900U0 {
        #[bits(1)]
        pub autostop: u32,
        #[bits(1)]
        pub working: u32,
        #[bits(1)]
        pub restart_work: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
}

fn usb_maxpacket(udev: *mut usb_device, pipe: i32) -> u16 {
let ep: *mut usb_host_endpoint = usb_pipe_endpoint(udev, pipe as u32);
if ep.is_null() {
return 0;
}
// NOTE:  only 0x07ff bits are for packet size...
usb_endpoint_maxp((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) as u16
}

#[repr(C)]
#[derive(Clone, Copy)]
struct property {
    name: *mut u8,
    length: i32,
    value: *mut (),
    next: *mut property,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
    name: *const u8,
    phandle: u32,
    full_name: *const u8,
    fwnode: fwnode_handle,
    properties: *mut property,
    deadprops: *mut property,
    parent: *mut device_node,
    child: *mut device_node,
    sibling: *mut device_node,
    _flags: u64,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
    version: u16,
    ac_exitcode: u32,
    ac_flag: u8,
    ac_nice: u8,
    cpu_count: u64,
    cpu_delay_total: u64,
    blkio_count: u64,
    blkio_delay_total: u64,
    swapin_count: u64,
    swapin_delay_total: u64,
    cpu_run_real_total: u64,
    cpu_run_virtual_total: u64,
    ac_comm: [u8; 32],
    ac_sched: u8,
    ac_pad: [u8; 3],
    __slate_pad_15: [u8; 4],
    ac_uid: u32,
    ac_gid: u32,
    ac_pid: u32,
    ac_ppid: u32,
    ac_btime: u32,
    ac_etime: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
    coremem: u64,
    virtmem: u64,
    hiwater_rss: u64,
    hiwater_vm: u64,
    read_char: u64,
    write_char: u64,
    read_syscalls: u64,
    write_syscalls: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
    nvcsw: u64,
    nivcsw: u64,
    ac_utimescaled: u64,
    ac_stimescaled: u64,
    cpu_scaled_run_real_total: u64,
    freepages_count: u64,
    freepages_delay_total: u64,
    thrashing_count: u64,
    thrashing_delay_total: u64,
    ac_btime64: u64,
    compact_count: u64,
    compact_delay_total: u64,
    ac_tgid: u32,
    ac_tgetime: u64,
    ac_exe_dev: u64,
    ac_exe_inode: u64,
    wpcopy_count: u64,
    wpcopy_delay_total: u64,
    irq_count: u64,
    irq_delay_total: u64,
    cpu_delay_max: u64,
    cpu_delay_min: u64,
    blkio_delay_max: u64,
    blkio_delay_min: u64,
    swapin_delay_max: u64,
    swapin_delay_min: u64,
    freepages_delay_max: u64,
    freepages_delay_min: u64,
    thrashing_delay_max: u64,
    thrashing_delay_min: u64,
    compact_delay_max: u64,
    compact_delay_min: u64,
    wpcopy_delay_max: u64,
    wpcopy_delay_min: u64,
    irq_delay_max: u64,
    irq_delay_min: u64,
    cpu_delay_max_ts: __kernel_timespec,
    blkio_delay_max_ts: __kernel_timespec,
    swapin_delay_max_ts: __kernel_timespec,
    freepages_delay_max_ts: __kernel_timespec,
    thrashing_delay_max_ts: __kernel_timespec,
    compact_delay_max_ts: __kernel_timespec,
    wpcopy_delay_max_ts: __kernel_timespec,
    irq_delay_max_ts: __kernel_timespec,
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
struct ns_tree_root {
    ns_rb: rb_root,
    ns_list_head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_node {
    ns_node: rb_node,
    ns_list_entry: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree {
    ns_id: u64,
    __ns_ref_active: __SlateRecord_types_h_190,
    ns_unified_node: ns_tree_node,
    ns_tree_node: ns_tree_node,
    ns_owner_node: ns_tree_node,
    ns_owner_root: ns_tree_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
    __slate_anon_0: __SlateRecord_ns_common_types_h_112,
    ns_type: u32,
    stashed: *mut dentry,
    ops: *const proc_ns_operations,
    inum: u32,
    __slate_anon_5: ns_tree,
    ns_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
    count: refcount_struct,
    uts_ns: *mut uts_namespace,
    ipc_ns: *mut ipc_namespace,
    mnt_ns: *mut mnt_namespace,
    pid_ns_for_children: *mut pid_namespace,
    net_ns: *mut net,
    time_ns: *mut time_namespace,
    time_ns_for_children: *mut time_namespace,
    cgroup_ns: *mut cgroup_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_extent {
    first: u32,
    lower_first: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_map {
    __slate_anon_0: __SlateRecord_user_namespace_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
    uid_map: uid_gid_map,
    gid_map: uid_gid_map,
    projid_map: uid_gid_map,
    parent: *mut user_namespace,
    level: i32,
    owner: __SlateRecord_uidgid_types_h_9,
    group: __SlateRecord_uidgid_types_h_13,
    ns: ns_common,
    flags: u64,
    parent_could_setfcap: bool,
    keyring_name_list: list_head,
    user_keyring_register: *mut key,
    keyring_sem: rw_semaphore,
    work: work_struct,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    ucounts: *mut ucounts,
    ucount_max: [i64; 11],
    rlimit_max: [i64; 4],
    binfmt_misc: *mut binfmt_misc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
    node: hlist_nulls_node,
    ns: *mut user_namespace,
    uid: __SlateRecord_uidgid_types_h_9,
    rcu: callback_head,
    count: __SlateRecord_types_h_202,
    ucount: [__SlateRecord_types_h_197; 11],
    rlimit: [__SlateRecord_types_h_197; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_stat {
    irqs_sum: u64,
    softirqs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct u64_stats_sync {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_bpf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct psi_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_file {
    kn: *mut kernfs_node,
    notified_at: u64,
    notify_timer: timer_list,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys_state {
    cgroup: *mut cgroup,
    ss: *mut cgroup_subsys,
    refcnt: percpu_ref,
    rstat_cpu: *mut css_rstat_cpu,
    sibling: list_head,
    children: list_head,
    id: i32,
    flags: u32,
    serial_nr: u64,
    online_cnt: __SlateRecord_types_h_190,
    destroy_work: work_struct,
    destroy_rwork: rcu_work,
    parent: *mut cgroup_subsys_state,
    nr_descendants: i32,
    nr_populated_csets: i32,
    nr_populated_children: i32,
    kill_finish_work: work_struct,
    rstat_flush_next: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
    subsys: [*mut cgroup_subsys_state; 14],
    refcount: refcount_struct,
    dom_cset: *mut css_set,
    dfl_cgrp: *mut cgroup,
    nr_tasks: i32,
    tasks: list_head,
    mg_tasks: list_head,
    dying_tasks: list_head,
    task_iters: list_head,
    e_cset_node: [list_head; 14],
    threaded_csets: list_head,
    threaded_csets_node: list_head,
    hlist: hlist_node,
    cgrp_links: list_head,
    mg_src_preload_node: list_head,
    mg_dst_preload_node: list_head,
    mg_node: list_head,
    mg_src_cgrp: *mut cgroup,
    mg_dst_cgrp: *mut cgroup,
    mg_dst_cset: *mut css_set,
    dead: bool,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_base_stat {
    cputime: task_cputime,
    ntime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_rstat_cpu {
    updated_children: *mut cgroup_subsys_state,
    updated_next: *mut cgroup_subsys_state,
    lnode: llist_node,
    owner: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_rstat_base_cpu {
    bsync: u64_stats_sync,
    bstat: cgroup_base_stat,
    last_bstat: cgroup_base_stat,
    subtree_bstat: cgroup_base_stat,
    last_subtree_bstat: cgroup_base_stat,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_freezer_state {
    freeze: bool,
    e_freeze: bool,
    nr_frozen_descendants: i32,
    nr_frozen_tasks: i32,
    freeze_seq: seqcount_spinlock,
    freeze_start_nsec: u64,
    frozen_nsec: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup {
    self_: cgroup_subsys_state,
    flags: u64,
    level: i32,
    max_depth: i32,
    nr_descendants: i32,
    nr_dying_descendants: i32,
    max_descendants: i32,
    nr_populated_domain_children: i32,
    nr_populated_threaded_children: i32,
    nr_threaded_children: i32,
    kill_seq: u32,
    kn: *mut kernfs_node,
    procs_file: cgroup_file,
    events_file: cgroup_file,
    psi_files: [cgroup_file; 0],
    subtree_control: u32,
    subtree_ss_mask: u32,
    old_subtree_control: u32,
    old_subtree_ss_mask: u32,
    subsys: [*mut cgroup_subsys_state; 14],
    nr_dying_subsys: [i32; 14],
    root: *mut cgroup_root,
    cset_links: list_head,
    e_csets: [list_head; 14],
    dom_cgrp: *mut cgroup,
    old_dom_cgrp: *mut cgroup,
    rstat_base_cpu: *mut cgroup_rstat_base_cpu,
    _pad_: cacheline_padding,
    last_bstat: cgroup_base_stat,
    bstat: cgroup_base_stat,
    prev_cputime: prev_cputime,
    pidlists: list_head,
    pidlist_mutex: mutex,
    offline_waitq: wait_queue_head,
    release_agent_work: work_struct,
    psi: *mut psi_group,
    bpf: cgroup_bpf,
    freezer: cgroup_freezer_state,
    __slate_anon_38: __SlateRecord_cgroup_defs_h_644,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_root {
    kf_root: *mut kernfs_root,
    subsys_mask: u32,
    hierarchy_id: i32,
    root_list: list_head,
    rcu: callback_head,
    nr_cgrps: __SlateRecord_types_h_190,
    flags: u32,
    release_agent_path: [u8; 4096],
    name: [u8; 64],
    cgrp: cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cftype {
    name: [u8; 64],
    private: u64,
    max_write_len: u64,
    flags: u32,
    file_offset: u32,
    ss: *mut cgroup_subsys,
    node: list_head,
    kf_ops: *mut kernfs_ops,
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    read_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> u64>,
    read_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> i64>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    write_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, u64) -> i32>,
    write_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, i64) -> i32>,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    lockdep_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys {
    css_alloc: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> *mut cgroup_subsys_state>,
    css_online: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> i32>,
    css_offline: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_released: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_free: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_reset: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_killed: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_rstat_flush: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, i32)>,
    css_extra_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    css_local_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    can_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset) -> i32>,
    cancel_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    can_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set) -> i32>,
    cancel_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set)>,
    fork: Option<unsafe extern "C" fn(*mut task_struct)>,
    exit: Option<unsafe extern "C" fn(*mut task_struct)>,
    release: Option<unsafe extern "C" fn(*mut task_struct)>,
    bind: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    __slate_bits_0: __slate_bits::__SlateBits827U0,
    id: i32,
    name: *const u8,
    legacy_name: *const u8,
    root: *mut cgroup_root,
    css_idr: idr,
    cfts: list_head,
    dfl_cftypes: *mut cftype,
    legacy_cftypes: *mut cftype,
    depends_on: u32,
    rstat_ss_lock: spinlock,
    lhead: *mut llist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
    ns: ns_common,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    root_cset: *mut css_set,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fprop_local_percpu {
    events: percpu_counter,
    period: u32,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bdi_writeback {
    bdi: *mut backing_dev_info,
    state: u64,
    last_old_flush: u64,
    b_dirty: list_head,
    b_io: list_head,
    b_more_io: list_head,
    b_dirty_time: list_head,
    list_lock: spinlock,
    writeback_inodes: __SlateRecord_types_h_190,
    stat: [percpu_counter; 5],
    bw_time_stamp: u64,
    dirtied_stamp: u64,
    written_stamp: u64,
    write_bandwidth: u64,
    avg_write_bandwidth: u64,
    dirty_ratelimit: u64,
    balanced_dirty_ratelimit: u64,
    completions: fprop_local_percpu,
    dirty_exceeded: i32,
    start_all_reason: u32,
    work_lock: spinlock,
    work_list: list_head,
    dwork: delayed_work,
    bw_dwork: delayed_work,
    bdi_node: list_head,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_ohci_hub_status_data: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ohci_hub_control: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ohci_setup: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ohci_restart: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ohci_suspend: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ohci_resume: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ohci_init_driver: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_ohci_hcd__670_1336_ohci_hcd_mod_init6:\t\t\t\n.long\t{7} - .\t\n.previous\t\t\t\t\t\n", sym ohci_hub_status_data, sym ohci_hub_control, sym ohci_setup, sym ohci_restart, sym ohci_suspend, sym ohci_resume, sym ohci_init_driver, sym ohci_hcd_mod_init, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
    id: u64,
    rb_node: rb_node,
    bdi_list: list_head,
    ra_pages: u64,
    io_pages: u64,
    refcnt: kref,
    capabilities: u32,
    min_ratio: u32,
    max_ratio: u32,
    max_prop_frac: u32,
    tot_write_bandwidth: __SlateRecord_types_h_197,
    last_bdp_sleep: u64,
    wb: bdi_writeback,
    wb_list: list_head,
    wb_waitq: wait_queue_head,
    dev: *mut device,
    dev_name: [u8; 64],
    owner: *mut device,
    debug_dir: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
    bd_start_sect: u64,
    bd_nr_sectors: u64,
    bd_disk: *mut gendisk,
    bd_queue: *mut request_queue,
    bd_stats: *mut disk_stats,
    bd_stamp: u64,
    __bd_flags: __SlateRecord_types_h_190,
    bd_dev: u32,
    bd_mapping: *mut address_space,
    bd_openers: __SlateRecord_types_h_190,
    bd_size_lock: spinlock,
    bd_claiming: *mut (),
    bd_holder: *mut (),
    bd_holder_ops: *const blk_holder_ops,
    bd_holder_lock: mutex,
    bd_holders: i32,
    bd_holder_dir: *mut kobject,
    bd_fsfreeze_count: __SlateRecord_types_h_190,
    bd_fsfreeze_mutex: mutex,
    bd_meta_info: *mut partition_meta_info,
    bd_writers: i32,
    bd_security: *mut (),
    bd_device: device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_batch {
    nr: u8,
    i: u8,
    percpu_pvec_drained: bool,
    folios: [*mut folio; 31],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
    nr_to_write: i64,
    pages_skipped: i64,
    range_start: i64,
    range_end: i64,
    sync_mode: u32,
    __slate_bits_0: __slate_bits::__SlateBits560U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
    folio: *mut folio,
    bit_nr: i32,
    wait: wait_queue_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
    file: *mut file,
    mapping: *mut address_space,
    ra: *mut file_ra_state,
    _index: u64,
    _nr_pages: u32,
    _batch_count: u32,
    dropbehind: bool,
    _workingset: bool,
    _pflags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
    reclaimed: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_sequential_cluster {
    next: [u32; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
    users: percpu_ref,
    flags: u64,
    prio: i16,
    list: plist_node,
    r#type: i8,
    max: u32,
    cluster_info: *mut swap_cluster_info,
    free_clusters: list_head,
    full_clusters: list_head,
    nonfull_clusters: [list_head; 1],
    frag_clusters: [list_head; 1],
    pages: u32,
    inuse_pages: __SlateRecord_types_h_197,
    global_cluster: *mut swap_sequential_cluster,
    global_cluster_lock: spinlock,
    swap_extent_root: rb_root,
    bdev: *mut block_device,
    swap_file: *mut file,
    comp: completion,
    lock: spinlock,
    discard_work: work_struct,
    reclaim_work: work_struct,
    discard_clusters: list_head,
    avail_list: plist_node,
    ops: *const swap_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_dp {
    link_rate: u32,
    lanes: u32,
    voltage: [u32; 4],
    pre: [u32; 4],
    __slate_bits_0: __slate_bits::__SlateBits853U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_hdmi {
    bpc: u32,
    __slate_anon_1: __SlateRecord_phy_hdmi_h_27,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_lvds {
    bits_per_lane_and_dclk_cycle: u32,
    differential_clk_rate: u64,
    lanes: u32,
    is_slave: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_mipi_dphy {
    clk_miss: u32,
    clk_post: u32,
    clk_pre: u32,
    clk_prepare: u32,
    clk_settle: u32,
    clk_term_en: u32,
    clk_trail: u32,
    clk_zero: u32,
    d_term_en: u32,
    eot: u32,
    hs_exit: u32,
    hs_prepare: u32,
    hs_settle: u32,
    hs_skip: u32,
    hs_trail: u32,
    hs_zero: u32,
    init: u32,
    lpx: u32,
    ta_get: u32,
    ta_go: u32,
    ta_sure: u32,
    wakeup: u32,
    hs_clk_rate: u64,
    lp_clk_rate: u64,
    lanes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union phy_notify {
    ufs_state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union phy_configure_opts {
    mipi_dphy: phy_configure_opts_mipi_dphy,
    dp: phy_configure_opts_dp,
    lvds: phy_configure_opts_lvds,
    hdmi: phy_configure_opts_hdmi,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_ops {
    init: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    power_on: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    power_off: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    set_mode: Option<unsafe extern "C" fn(*mut phy, u32, i32) -> i32>,
    set_media: Option<unsafe extern "C" fn(*mut phy, u32) -> i32>,
    set_speed: Option<unsafe extern "C" fn(*mut phy, i32) -> i32>,
    configure: Option<unsafe extern "C" fn(*mut phy, *mut phy_configure_opts) -> i32>,
    validate: Option<unsafe extern "C" fn(*mut phy, u32, i32, *mut phy_configure_opts) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    calibrate: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    connect: Option<unsafe extern "C" fn(*mut phy, i32) -> i32>,
    disconnect: Option<unsafe extern "C" fn(*mut phy, i32) -> i32>,
    notify_phystate: Option<unsafe extern "C" fn(*mut phy, phy_notify) -> i32>,
    release: Option<unsafe extern "C" fn(*mut phy)>,
    owner: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_attrs {
    bus_width: u32,
    max_link_rate: u32,
    mode: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy {
    dev: device,
    id: i32,
    ops: *const phy_ops,
    mutex: mutex,
    lockdep_key: lock_class_key,
    init_count: i32,
    power_count: i32,
    attrs: phy_attrs,
    pwr: *mut regulator,
    debugfs: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_phy_io_ops {
    read: Option<unsafe extern "C" fn(*mut usb_phy, u32) -> i32>,
    write: Option<unsafe extern "C" fn(*mut usb_phy, u32, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_charger_current {
    sdp_min: u32,
    sdp_max: u32,
    dcp_min: u32,
    dcp_max: u32,
    cdp_min: u32,
    cdp_max: u32,
    aca_min: u32,
    aca_max: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_phy {
    dev: *mut device,
    label: *const u8,
    flags: u32,
    r#type: u32,
    last_event: u32,
    otg: *mut usb_otg,
    io_dev: *mut device,
    io_ops: *mut usb_phy_io_ops,
    io_priv: *mut (),
    edev: *mut extcon_dev,
    id_edev: *mut extcon_dev,
    vbus_nb: notifier_block,
    id_nb: notifier_block,
    type_nb: notifier_block,
    chg_type: u32,
    chg_state: u32,
    chg_cur: usb_charger_current,
    chg_work: work_struct,
    notifier: atomic_notifier_head,
    port_status: u16,
    port_change: u16,
    head: list_head,
    init: Option<unsafe extern "C" fn(*mut usb_phy) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut usb_phy)>,
    set_vbus: Option<unsafe extern "C" fn(*mut usb_phy, i32) -> i32>,
    set_power: Option<unsafe extern "C" fn(*mut usb_phy, u32) -> i32>,
    set_suspend: Option<unsafe extern "C" fn(*mut usb_phy, i32) -> i32>,
    set_wakeup: Option<unsafe extern "C" fn(*mut usb_phy, bool) -> i32>,
    notify_connect: Option<unsafe extern "C" fn(*mut usb_phy, u32) -> i32>,
    notify_disconnect: Option<unsafe extern "C" fn(*mut usb_phy, u32) -> i32>,
    charger_detect: Option<unsafe extern "C" fn(*mut usb_phy) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_otg {
    default_a: u8,
    phy: *mut phy,
    usb_phy: *mut usb_phy,
    host: *mut usb_bus,
    gadget: *mut usb_gadget,
    state: u32,
    set_host: Option<unsafe extern "C" fn(*mut usb_otg, *mut usb_bus) -> i32>,
    set_peripheral: Option<unsafe extern "C" fn(*mut usb_otg, *mut usb_gadget) -> i32>,
    set_vbus: Option<unsafe extern "C" fn(*mut usb_otg, bool) -> i32>,
    start_srp: Option<unsafe extern "C" fn(*mut usb_otg) -> i32>,
    start_hnp: Option<unsafe extern "C" fn(*mut usb_otg) -> i32>,
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
    __slate_bits_0: __slate_bits::__SlateBits880U0,
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


// Hub Class feature numbers
// See USB 2.0 spec Table 11-17
// L2 suspend
// All ports power control at once
// per-port power control
// no power switching
// All ports Over-Current reporting
// per-port Over-current reporting
// No Over-current Protection support
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_hub_descriptor {
    bDescLength: u8,
    bDescriptorType: u8,
    bNbrPorts: u8,
    wHubCharacteristics: u16,
    bPwrOn2PwrGood: u8,
    bHubContrCurrent: u8,
    u: __SlateRecord_ch11_h_269,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct debugfs_short_fops {
    read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gen_pool {
    lock: spinlock,
    chunks: list_head,
    min_alloc_order: i32,
    algo: Option<unsafe extern "C" fn(*mut u64, u64, u64, u32, *mut (), *mut gen_pool, u64) -> u64>,
    data: *mut (),
    name: *const u8,
}


fn gen_pool_free(pool: *mut gen_pool, addr: u64, size: u64) {
unsafe { gen_pool_free_owner(pool, addr, size, core::ptr::null_mut::<*mut ()>()) };
}

fn put_unaligned_le32(val: u32, p: *mut ()) {
__put_unaligned_t!(u32, val, p);
}

// For initializing controller (mask in an HCFS mode too)
static hcd_name: [u8; 9] = [111, 104, 99, 105, 95, 104, 99, 100, 0];

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct ed {
    hwINFO: u32,
    hwTailP: u32,
    hwHeadP: u32,
    hwNextED: u32,
    dma: u64,
    dummy: *mut td,
    ed_next: *mut ed,
    ed_prev: *mut ed,
    td_list: list_head,
    in_use_list: list_head,
    state: u8,
    r#type: u8,
    branch: u8,
    interval: u16,
    load: u16,
    last_iso: u16,
    tick: u16,
    takeback_wdh_cnt: u32,
    pending_td: *mut td,
}


#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct td {
    hwINFO: u32,
    hwCBP: u32,
    hwNextTD: u32,
    hwBE: u32,
    hwPSW: [u16; 2],
    index: u8,
    ed: *mut ed,
    td_hash: *mut td,
    next_dl_td: *mut td,
    urb: *mut urb,
    td_dma: u64,
    data_dma: u64,
    td_list: list_head,
}


// strip hw status in low addr bits
// Hardware transfer status codes -- CC from td->hwINFO or td->hwPSW
static cc_to_error: __SlateAlign16<[i32; 16]> = __SlateAlign16([0i32, 84i32.wrapping_neg(), 71i32.wrapping_neg(), 84i32.wrapping_neg(), 32i32.wrapping_neg(), 62i32.wrapping_neg(), 71i32.wrapping_neg(), 71i32.wrapping_neg(), 75i32.wrapping_neg(), 121i32.wrapping_neg(), 5i32.wrapping_neg(), 5i32.wrapping_neg(), 70i32.wrapping_neg(), 63i32.wrapping_neg(), 114i32.wrapping_neg(), 114i32.wrapping_neg()]);

#[repr(C, align(256))]
#[derive(Clone, Copy)]
struct ohci_hcca {
    int_table: [u32; 32],
    frame_no: u32,
    done_head: u32,
    reserved_for_hc: [u8; 116],
    what: [u8; 4],
}


#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct ohci_regs {
    revision: u32,
    control: u32,
    cmdstatus: u32,
    intrstatus: u32,
    intrenable: u32,
    intrdisable: u32,
    hcca: u32,
    ed_periodcurrent: u32,
    ed_controlhead: u32,
    ed_controlcurrent: u32,
    ed_bulkhead: u32,
    ed_bulkcurrent: u32,
    donehead: u32,
    fminterval: u32,
    fmremaining: u32,
    fmnumber: u32,
    periodicstart: u32,
    lsthresh: u32,
    roothub: ohci_roothub_regs,
}


// HcControl (control) register masks
// control/bulk service ratio
// periodic list enable
// isochronous enable
// control list enable
// bulk list enable
// host controller functional state
// interrupt routing
// remote wakeup connected
// remote wakeup enable
// pre-shifted values for HCFS
// HcCommandStatus (cmdstatus) register masks
// host controller reset
// control list filled
// bulk list filled
// ownership change request
// scheduling overrun count
// masks used with interrupt registers:
// HcInterruptStatus (intrstatus)
// HcInterruptEnable (intrenable)
// HcInterruptDisable (intrdisable)
// scheduling overrun
// writeback of done_head
// start frame
// resume detect
// unrecoverable error
// frame number overflow
// root hub status change
// ownership change
// master interrupt enable
// roothub.portstatus [i] bits
// current connect status
// port enable status
// port suspend status
// port over current indicator
// port reset status
// port power status
// low speed device attached
// connect status change
// port enable status change
// port suspend status change
// over current indicator change
// port reset status change
// roothub.status bits
// local power status
// over current indicator
// device remote wakeup enable
// local power status change
// over current indicator change
// clear remote wakeup enable
// roothub.b masks
// device removable flags
// port power control mask
// roothub.a masks
// number of downstream ports
// power switching mode
// no power switching
// device type (mbz)
// over current protection mode
// no over current protection
// power on to power good time
#[repr(C)]
#[derive(Clone, Copy)]
struct urb_priv {
    ed: *mut ed,
    length: u16,
    td_cnt: u16,
    pending: list_head,
    td: [*mut td; 0],
}


// power'o'two
#[repr(C)]
#[derive(Clone, Copy)]
struct ohci_hcd {
    lock: spinlock,
    regs: *mut ohci_regs,
    hcca: *mut ohci_hcca,
    hcca_dma: u64,
    ed_rm_list: *mut ed,
    ed_bulktail: *mut ed,
    ed_controltail: *mut ed,
    periodic: [*mut ed; 32],
    start_hnp: Option<unsafe extern "C" fn(*mut ohci_hcd)>,
    td_cache: *mut dma_pool,
    ed_cache: *mut dma_pool,
    td_hash: [*mut td; 64],
    dl_start: *mut td,
    dl_end: *mut td,
    pending: list_head,
    eds_in_use: list_head,
    rh_state: u32,
    num_ports: i32,
    load: [i32; 32],
    hc_control: u32,
    next_statechange: u64,
    fminterval: u32,
    __slate_bits_0: __slate_bits::__SlateBits900U0,
    flags: u64,
    prev_frame_no: u32,
    wdh_cnt: u32,
    prev_wdh_cnt: u32,
    prev_donehead: u32,
    io_watchdog: timer_list,
    nec_work: work_struct,
    debug_dir: *mut dentry,
    r#priv: [u64; 0],
}


fn quirk_nec(ohci: *mut ohci_hcd) -> i32 {
((unsafe { (*ohci).flags }) & 64) as i32
}

fn quirk_zfmicro(ohci: *mut ohci_hcd) -> i32 {
((unsafe { (*ohci).flags }) & 32) as i32
}

fn quirk_amdiso(ohci: *mut ohci_hcd) -> i32 {
((unsafe { (*ohci).flags }) & 512) as i32
}

fn quirk_amdprefetch(ohci: *mut ohci_hcd) -> i32 {
((unsafe { (*ohci).flags }) & 1024) as i32
}

fn hcd_to_ohci(hcd: *mut usb_hcd) -> *mut ohci_hcd {
(unsafe { core::ptr::addr_of_mut!((*hcd).hcd_priv) as *mut u64 }) as *mut ohci_hcd
}

fn ohci_to_hcd(ohci: *const ohci_hcd) -> *mut usb_hcd {
container_of__2!(ohci as *mut (), usb_hcd)
}

fn _ohci_readl(ohci: *const ohci_hcd, regs: *mut u32) -> u32 {
readl(regs as *const ())
}

fn _ohci_writel(ohci: *const ohci_hcd, val: u32, regs: *mut u32) {
writel(val, regs as *mut ());
}

fn cpu_to_hc16(ohci: *const ohci_hcd, x: u16) -> u16 {
(if big_endian_desc!() != 0 { __swab16!(x) as i32 } else { x as i32 }) as u16
}

fn cpu_to_hc32(ohci: *const ohci_hcd, x: u32) -> u32 {
if big_endian_desc!() != 0 { __swab32!(x) } else { x }
}

fn hc16_to_cpup(ohci: *const ohci_hcd, x: *const u16) -> u16 {
let __v0: i32 = if big_endian_desc!() != 0 {
__be16_to_cpup(x as *const u16) as i32
} else {
__le16_to_cpup(x as *const u16) as i32
};
__v0 as u16
}

fn hc32_to_cpu(ohci: *const ohci_hcd, x: u32) -> u32 {
if big_endian_desc!() != 0 { __swab32!(x) } else { x }
}

fn hc32_to_cpup(ohci: *const ohci_hcd, x: *const u32) -> u32 {
let __v0: u32 = if big_endian_desc!() != 0 {
__be32_to_cpup(x as *const u32)
} else {
__le32_to_cpup(x as *const u32)
};
__v0
}

fn ohci_frame_no(ohci: *const ohci_hcd) -> u16 {
let mut tmp: u32 = 0;
if big_endian_desc!() != 0 {
tmp = __be32_to_cpup((unsafe { core::ptr::addr_of_mut!((*(*ohci).hcca).frame_no) }) as *const u32);
if (unsafe { (*ohci).flags }) & 128 == 0 {
tmp >>= 16i32;
}
} else {
tmp = __le32_to_cpup((unsafe { core::ptr::addr_of_mut!((*(*ohci).hcca).frame_no) }) as *const u32);
}
tmp as u16
}

fn ohci_hwPSWp(ohci: *const ohci_hcd, td: *const td, index: i32) -> *mut u16 {
(if big_endian_desc!() != 0 { unsafe { (*td).hwPSW.as_ptr() as *const u16 }.wrapping_offset((index ^ 1) as isize) } else { unsafe { (*td).hwPSW.as_ptr() as *const u16 }.wrapping_offset(index as isize) }) as *mut u16
}

fn ohci_hwPSW(ohci: *const ohci_hcd, td: *const td, index: i32) -> u16 {
hc16_to_cpup(ohci, ohci_hwPSWp(ohci, td, index) as *const u16)
}

// 12000 bits per frame (-1)
fn periodic_reinit(ohci: *mut ohci_hcd) {
let fi: u32 = (unsafe { (*ohci).fminterval }) & 16383;
let fit: u32 = ohci_readl!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).fminterval) }) & 2147483648u32;
ohci_writel__2!(ohci, fit ^ 2147483648u32 | unsafe { (*ohci).fminterval }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).fminterval) });
ohci_writel__2!(ohci, 9u32.wrapping_mul(fi) / 10 & 16383, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).periodicstart) });
}

fn roothub_a(hc: *mut ohci_hcd) -> u32 {
read_roothub!(hc, a, 4228898816u32)
}

fn roothub_b(hc: *mut ohci_hcd) -> u32 {
ohci_readl!(hc, unsafe { core::ptr::addr_of_mut!((*(*hc).regs).roothub.b) })
}

fn roothub_status(hc: *mut ohci_hcd) -> u32 {
ohci_readl!(hc, unsafe { core::ptr::addr_of_mut!((*(*hc).regs).roothub.status) })
}

fn roothub_portstatus(hc: *mut ohci_hcd, i: i32) -> u32 {
let mut __v0: u32;
let mut temp: u32 = ohci_readl!(hc, unsafe { (*(*hc).regs).roothub.portstatus.as_mut_ptr() as *mut u32 }.wrapping_offset(i as isize));
if temp == (1i32.wrapping_neg() as u32) {
unsafe {
(*hc).rh_state = 0;
}
} else {
if (unsafe { (*hc).flags }) & 1 != 0 {
while temp & 4292934880 != 0 {
temp = ohci_readl!(hc, unsafe { (*(*hc).regs).roothub.portstatus.as_mut_ptr() as *mut u32 }.wrapping_offset(i as isize));
}
}
}
__v0 = temp;
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ohci_driver_overrides {
    product_desc: *const u8,
    extra_priv_size: u64,
    reset: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
}


#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.io_watchdog_func")]
extern "C" fn io_watchdog_func(t: *mut timer_list) {
let mut tmp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(tmp) as *mut u32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut active_cnt: i32 = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut prev_frame_no: u32 = unsafe { core::mem::zeroed() };
let mut frame_no: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(frame_no) as *mut u32;
let mut td_next: core::mem::MaybeUninit<*mut td> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut td = core::ptr::addr_of_mut!(td_next) as *mut *mut td;
let mut td_start: core::mem::MaybeUninit<*mut td> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut td = core::ptr::addr_of_mut!(td_start) as *mut *mut td;
let mut td: core::mem::MaybeUninit<*mut td> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut td = core::ptr::addr_of_mut!(td) as *mut *mut td;
let mut ed: core::mem::MaybeUninit<*mut ed> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut ed = core::ptr::addr_of_mut!(ed) as *mut *mut ed;
let mut head: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u32 = core::ptr::addr_of_mut!(head) as *mut u32;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut takeback_all_pending: bool = unsafe { core::mem::zeroed() };
let mut ohci: *mut ohci_hcd = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
'__join_31: {
{
ohci = timer_container_of__3!(t);
takeback_all_pending = false;
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
prev_frame_no = 4294967040;
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_9);
// One way to lose track of completed TDs is if the controller
// never writes back the done queue head.  If it hasn't been
// written back since the last time this function ran and if it
// was non-empty at that time, something is badly wrong with the
// hardware.
*__slate_slot_17 = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).intrstatus));
if *__slate_slot_17 & 2u32 == 0 && (*ohci).wdh_cnt == (*ohci).prev_wdh_cnt {
if (*ohci).prev_donehead != 0 {
while false {
}
_dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"HcDoneHead not written back; disabled\n".as_ptr() as *const u8);
break '__join_31;
} else {
// No write back because the done queue was empty
takeback_all_pending = true;
}
}
}
// Check every ED which might have pending TDs
*__slate_slot_15 = list_first_entry__24!(core::ptr::addr_of_mut!((*ohci).eds_in_use));
while list_entry_is_head!(*__slate_slot_15, core::ptr::addr_of_mut!((*ohci).eds_in_use), in_use_list) == 0 {
if !unsafe { (*(*__slate_slot_15)).pending_td }.is_null() {
if takeback_all_pending || OKAY_TO_TAKEBACK!(ohci, *__slate_slot_15) {
hc32_to_cpu(ohci as *const ohci_hcd, (*(*__slate_slot_15)).hwINFO);
add_to_done_list(ohci, (*(*__slate_slot_15)).pending_td);
}
}
// Starting from the latest pending TD,
*__slate_slot_14 = (*(*__slate_slot_15)).pending_td;
// or the last TD on the done list,
if (*__slate_slot_14).is_null() {
*__slate_slot_12 = list_first_entry__19!(core::ptr::addr_of_mut!((*(*__slate_slot_15)).td_list));
while list_entry_is_head!(*__slate_slot_12, core::ptr::addr_of_mut!((*(*__slate_slot_15)).td_list), td_list) == 0 {
if unsafe { (*(*__slate_slot_12)).next_dl_td }.is_null() {
break;
} else {
*__slate_slot_14 = *__slate_slot_12;
*__slate_slot_12 = list_next_entry__7!(*__slate_slot_12, td_list);
}
}
}
while false {
}
// find the last TD processed by the controller.
let __v7: u32 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_15)).hwHeadP) as *const u32);
*__slate_slot_16 = hc32_to_cpu(ohci as *const ohci_hcd, __v7) & 4294967264u32;
*__slate_slot_13 = *__slate_slot_14;
*__slate_slot_12 = list_prepare_entry__2!(*__slate_slot_14, core::ptr::addr_of_mut!((*(*__slate_slot_15)).td_list));
*__slate_slot_12 = list_next_entry__7!(*__slate_slot_12, td_list);
while list_entry_is_head!(*__slate_slot_12, core::ptr::addr_of_mut!((*(*__slate_slot_15)).td_list), td_list) == 0 {
if *__slate_slot_16 == ((*(*__slate_slot_12)).td_dma as u32) {
break;
} else {
*__slate_slot_14 = *__slate_slot_12; // head pointer has passed this TD
*__slate_slot_12 = list_next_entry__7!(*__slate_slot_12, td_list);
}
}
if *__slate_slot_14 != *__slate_slot_13 {
// In case a WDH cycle is in progress, we will wait
// for the next two cycles to complete before assuming
// this TD will never get on the done queue.
(*(*__slate_slot_15)).takeback_wdh_cnt = unsafe { (*ohci).wdh_cnt }.wrapping_add(2);
(*(*__slate_slot_15)).pending_td = *__slate_slot_14;
}
*__slate_slot_15 = list_next_entry__24!(*__slate_slot_15, in_use_list);
}
ohci_work(ohci);
if ((*ohci).rh_state as u32) == 2 {
// Sometimes a controller just stops working.  We can tell
// by checking that the frame counter has advanced since
// the last time we ran.
//
// But be careful: Some controllers violate the spec by
// stopping their frame counter when no ports are active.
*__slate_slot_11 = ohci_frame_no(ohci as *const ohci_hcd) as u32;
if *__slate_slot_11 == (*ohci).prev_frame_no {
active_cnt = 0;
let __slate_auto_bytes_8 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_5 = 0;
while *__slate_slot_5 < (*ohci).num_ports {
*__slate_slot_4 = roothub_portstatus(ohci, *__slate_slot_5);
// Enabled and not suspended?
if *__slate_slot_4 & 2 != 0 && *__slate_slot_4 & 4 == 0 {
active_cnt = active_cnt.wrapping_add(1);
}
let __v1: i32 = *__slate_slot_5;
*__slate_slot_5 = __v1.wrapping_add(1);
}
if active_cnt > 0 {
while false {
}
_dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"frame counter not updating; disabled\n".as_ptr() as *const u8);
break '__join_31;
}
}
if list_empty(core::ptr::addr_of_mut!((*ohci).eds_in_use) as *const list_head) == 0 {
prev_frame_no = *__slate_slot_11;
(*ohci).prev_wdh_cnt = (*ohci).wdh_cnt;
(*ohci).prev_donehead = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).donehead));
mod_timer(core::ptr::addr_of_mut!((*ohci).io_watchdog), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(275u32)));
break '__join_0;
} else {
break '__join_0;
}
} else {
break '__join_0;
}
}
usb_hc_died(ohci_to_hcd(ohci as *const ohci_hcd));
ohci_dump(ohci);
_ohci_shutdown(ohci_to_hcd(ohci as *const ohci_hcd));
}
(*ohci).prev_frame_no = prev_frame_no;
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_9);
}
}

fn ohci_rh_suspend(ohci: *mut ohci_hcd, autostop: i32) -> i32 {
let mut temp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u32 = core::ptr::addr_of_mut!(temp) as *mut u32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut portstat: *mut u32 = unsafe { core::mem::zeroed() };
let mut status: i32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
'__join_15: {
status = 0;
(*ohci).hc_control = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
let __t0: u32 = (*ohci).hc_control & 192u32;
if __t0 == 64 {
let __v25: u32 = (*ohci).hc_control & 4294967103u32;
(*ohci).hc_control = __v25;
(*ohci).hc_control = (*ohci).hc_control | 0u32;
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
break '__join_15;
} else {
if __t0 == 0 {
break '__join_15;
} else {
if __t0 == 192 {
if ((*ohci).__slate_bits_0.__get_autostop() as i32) == 0 {
break '__join_0;
}
}
}
}
// First stop any processing
if autostop == 0 && (*ohci).hc_control & 60u32 != 0 {
let __v19: u32 = (*ohci).hc_control & 4294967235u32;
(*ohci).hc_control = __v19;
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
(*ohci).hc_control = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
ohci_writel!(ohci, 4i32, core::ptr::addr_of_mut!((*(*ohci).regs).intrstatus));
// sched disables take effect on the next frame,
// then the last WDH could take 6+ msec
(*ohci).__slate_bits_0.__set_autostop(0u32);
spin_unlock_irq(core::ptr::addr_of_mut!((*ohci).lock));
msleep(8u32);
spin_lock_irq(core::ptr::addr_of_mut!((*ohci).lock));
}
update_done_list(ohci);
ohci_work(ohci);
// All ED unlinks should be finished, no need for SOF interrupts
ohci_writel!(ohci, 4i32, core::ptr::addr_of_mut!((*(*ohci).regs).intrdisable));
// Some controllers don't handle "global" suspend properly if
// there are unsuspended ports.  For these controllers, put all
// the enabled ports into suspend before suspending the root hub.
if (*ohci).flags & 2048 != 0 {
portstat = (*(*ohci).regs).roothub.portstatus.as_mut_ptr() as *mut u32;
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_17 = 0;
while *__slate_slot_17 < (*ohci).num_ports {
*__slate_slot_16 = ohci_readl!(ohci, portstat);
if *__slate_slot_16 & 6u32 == 2 {
ohci_writel!(ohci, 4i32, portstat);
}
let __v15: i32 = *__slate_slot_17;
*__slate_slot_17 = __v15.wrapping_add(1);
let __v13: *mut u32 = portstat;
portstat = __v13.wrapping_offset(1 as isize);
}
}
// maybe resume can wake root hub
if ((*(*ohci_to_hcd(ohci as *const ohci_hcd)).self_.root_hub).__slate_bits_1.__get_do_remote_wakeup() as i32) != 0 || autostop != 0 {
(*ohci).hc_control = (*ohci).hc_control | 1024u32;
} else {
ohci_writel!(ohci, 72i32, core::ptr::addr_of_mut!((*(*ohci).regs).intrdisable));
let __v9: u32 = (*ohci).hc_control & 4294966271u32;
(*ohci).hc_control = __v9;
}
// Suspend hub ... this is the "global (to this bus) suspend" mode,
// which doesn't imply ports will first be individually suspended.
let __v3: u32 = (*ohci).hc_control & 4294967103u32;
(*ohci).hc_control = __v3;
(*ohci).hc_control = (*ohci).hc_control | 192u32;
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
// no resumes until devices finish suspending
if autostop == 0 {
(*ohci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(5u32));
(*ohci).__slate_bits_0.__set_autostop(0u32);
(*ohci).rh_state = 1;
break '__join_0;
} else {
break '__join_0;
}
}
status = 16i32.wrapping_neg();
}
return status;
}
}

fn find_head(mut ed: *mut ed) -> *mut ed {
// for bulk and control lists
while !unsafe { (*ed).ed_prev }.is_null() {
ed = unsafe { (*ed).ed_prev };
}
ed
}

fn ohci_rh_resume(ohci: *mut ohci_hcd) -> i32 {
let mut autostopped: i32 = unsafe { core::mem::zeroed() };
let mut status: i32 = unsafe { core::mem::zeroed() };
let mut enables: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(enables) as *mut u32;
let mut temp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(temp) as *mut u32;
let mut hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
unsafe {
hcd = ohci_to_hcd(ohci as *const ohci_hcd);
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
status = 115i32.wrapping_neg();
autostopped = (*ohci).__slate_bits_0.__get_autostop() as i32;
(*ohci).__slate_bits_0.__set_autostop(0u32);
(*ohci).hc_control = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
if (*ohci).hc_control & 316u32 != 0 {
// this can happen after resuming a swsusp snapshot
if ((*ohci).rh_state as u32) != 2 {
status = 16i32.wrapping_neg();
// this happens when pmcore resumes HC then root
} else {
status = 0;
}
} else {
let __t0: u32 = (*ohci).hc_control & 192u32;
if __t0 == 192 {
let __v22: u32 = (*ohci).hc_control & 4294967043u32;
(*ohci).hc_control = __v22;
(*ohci).hc_control = (*ohci).hc_control | 64u32;
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
} else {
if __t0 == 64 {
// HCFS changes sometime after INTR_RD
} else {
if __t0 == 128 {
// this can happen after resuming a swsusp snapshot
status = 16i32.wrapping_neg();
} else {
// RESET, we lost power
status = 16i32.wrapping_neg();
}
}
}
}
if status == 16i32.wrapping_neg() {
if autostopped == 0 {
spin_unlock_irq(core::ptr::addr_of_mut!((*ohci).lock));
status = ohci_restart(ohci);
usb_root_hub_lost_power((*hcd).self_.root_hub);
spin_lock_irq(core::ptr::addr_of_mut!((*ohci).lock));
}
return status;
} else {
if status != 115i32.wrapping_neg() {
return status;
} else {
if autostopped == 0 {
spin_unlock_irq(core::ptr::addr_of_mut!((*ohci).lock));
// Some controllers (lucent erratum) need extra-long delays
msleep(33u32); // usb 11.5.1.10
// 32 msec counter
*__slate_slot_28 = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
*__slate_slot_28 &= 192u32;
if *__slate_slot_28 != 64u32 {
while false {
}
_dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"controller won't resume\n".as_ptr() as *const u8);
spin_lock_irq(core::ptr::addr_of_mut!((*ohci).lock));
return 16i32.wrapping_neg();
} else {
// disable old schedule state, reinit from scratch
ohci_writel!(ohci, 0i32, core::ptr::addr_of_mut!((*(*ohci).regs).ed_controlhead));
ohci_writel!(ohci, 0i32, core::ptr::addr_of_mut!((*(*ohci).regs).ed_controlcurrent));
ohci_writel!(ohci, 0i32, core::ptr::addr_of_mut!((*(*ohci).regs).ed_bulkhead));
ohci_writel!(ohci, 0i32, core::ptr::addr_of_mut!((*(*ohci).regs).ed_bulkcurrent));
ohci_writel!(ohci, 0i32, core::ptr::addr_of_mut!((*(*ohci).regs).ed_periodcurrent));
ohci_writel__2!(ohci, (*ohci).hcca_dma as u32, core::ptr::addr_of_mut!((*(*ohci).regs).hcca));
// Sometimes PCI D3 suspend trashes frame timings ...
periodic_reinit(ohci);
// The following code is executed with ohci->lock held and
// irqs disabled if and only if autostopped is true.  This
// will cause sparse to warn about a "context imbalance".
}
}
ohci_writel!(ohci, -2147483558i32, core::ptr::addr_of_mut!((*(*ohci).regs).intrenable)); // interrupts might have been disabled
if !unsafe { (*ohci).ed_rm_list }.is_null() {
ohci_writel!(ohci, 4i32, core::ptr::addr_of_mut!((*(*ohci).regs).intrenable));
}
// Then re-enable operations
ohci_writel!(ohci, 128i32, core::ptr::addr_of_mut!((*(*ohci).regs).control));
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
if autostopped == 0 {
msleep(3u32);
}
*__slate_slot_28 = (*ohci).hc_control;
*__slate_slot_28 &= 512u32;
*__slate_slot_28 |= 131u32;
(*ohci).hc_control = *__slate_slot_28;
ohci_writel__2!(ohci, *__slate_slot_28, core::ptr::addr_of_mut!((*(*ohci).regs).control));
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
// TRSMRCY
if autostopped == 0 {
msleep(10u32);
spin_lock_irq(core::ptr::addr_of_mut!((*ohci).lock));
}
// now ohci->lock is always held and irqs are always disabled
// keep it alive for more than ~5x suspend + resume costs
(*ohci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(300u32));
// maybe turn schedules back on
*__slate_slot_27 = 0;
*__slate_slot_28 = 0;
if unsafe { (*ohci).ed_rm_list }.is_null() {
if !unsafe { (*ohci).ed_controltail }.is_null() {
ohci_writel!(ohci, unsafe { (*find_head((*ohci).ed_controltail)).dma }, core::ptr::addr_of_mut!((*(*ohci).regs).ed_controlhead));
*__slate_slot_27 |= 16u32;
*__slate_slot_28 |= 2u32;
}
if !unsafe { (*ohci).ed_bulktail }.is_null() {
ohci_writel!(ohci, unsafe { (*find_head((*ohci).ed_bulktail)).dma }, core::ptr::addr_of_mut!((*(*ohci).regs).ed_bulkhead));
*__slate_slot_27 |= 32u32;
*__slate_slot_28 |= 4u32;
}
}
if (*hcd).self_.bandwidth_isoc_reqs != 0 || (*hcd).self_.bandwidth_int_reqs != 0 {
*__slate_slot_27 |= 12u32;
}
if *__slate_slot_27 != 0 {
(*ohci).hc_control = (*ohci).hc_control | *__slate_slot_27;
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
if *__slate_slot_28 != 0 {
ohci_writel__2!(ohci, *__slate_slot_28, core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus));
}
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
}
(*ohci).rh_state = 2;
return 0;
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_bus_suspend")]
extern "C" fn ohci_bus_suspend(hcd: *mut usb_hcd) -> i32 {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
let mut rc: i32 = 0;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ohci).lock) });
if unlikely!(HCD_HW_ACCESSIBLE!(hcd) == 0) != 0 {
rc = 108i32.wrapping_neg();
} else {
rc = ohci_rh_suspend(ohci, 0i32);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ohci).lock) });
if rc == 0 {
unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*ohci).io_watchdog)) };
unsafe {
(*ohci).prev_frame_no = 4294967040;
}
}
rc
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_bus_resume")]
extern "C" fn ohci_bus_resume(hcd: *mut usb_hcd) -> i32 {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
let mut rc: i32 = 0;
if time_before!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, unsafe { (*ohci).next_statechange }) {
unsafe { msleep(5u32) };
}
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ohci).lock) });
if unlikely!(HCD_HW_ACCESSIBLE!(hcd) == 0) != 0 {
rc = 108i32.wrapping_neg();
} else {
rc = ohci_rh_resume(ohci);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ohci).lock) });
// poll until we know a device is connected or we autostop
if rc == 0 {
unsafe { usb_hcd_poll_rh_status(hcd) };
}
rc
}

fn ohci_root_hub_state_changes(ohci: *mut ohci_hcd, changed: i32, any_connected: i32, rhsc_status: i32) -> i32 {
let mut poll_rh: i32 = 1;
let mut rhsc_enable: i32 = 0;
// Some broken controllers never turn off RHSC in the interrupt
// status register.  For their sake we won't re-enable RHSC
// interrupts if the interrupt bit is already active.
rhsc_enable = (ohci_readl!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrenable) }) & 64u32) as i32;
match (unsafe { (*ohci).hc_control }) & 192u32 {
128 => {
if rhsc_enable == 0 && rhsc_status == 0 && changed == 0 {
rhsc_enable = 64i32;
ohci_writel!(ohci, rhsc_enable, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrenable) });
}
// If no status changes are pending, enable RHSC interrupts.
// Keep on polling until we know a device is connected
// and RHSC is enabled, or until we autostop.
if ((unsafe { (*ohci).__slate_bits_0.__get_autostop() }) as i32) == 0 {
let __v0: bool = any_connected != 0 || !device_may_wakeup(unsafe { core::ptr::addr_of_mut!((*(*ohci_to_hcd(ohci as *const ohci_hcd)).self_.root_hub).dev) });
if __v0 {
if rhsc_enable != 0 {
poll_rh = 0;
}
} else {
unsafe {
(*ohci).__slate_bits_0.__set_autostop(1u32);
}
unsafe {
(*ohci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(1000);
}
}
// if no devices have been attached for one second, autostop
} else {
if changed != 0 || any_connected != 0 {
unsafe {
(*ohci).__slate_bits_0.__set_autostop(0u32);
}
unsafe {
(*ohci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(300u32));
}
} else {
if time_after_eq!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, unsafe { (*ohci).next_statechange }) && unsafe { (*ohci).ed_rm_list }.is_null() && (unsafe { (*ohci).hc_control }) & 60u32 == 0 {
ohci_rh_suspend(ohci, 1i32);
if rhsc_enable != 0 {
poll_rh = 0;
}
}
}
}
}
64 | 192 => {
if changed != 0 {
if ((unsafe { (*ohci).__slate_bits_0.__get_autostop() }) as i32) != 0 {
ohci_rh_resume(ohci);
} else {
unsafe { usb_hcd_resume_root_hub(ohci_to_hcd(ohci as *const ohci_hcd)) };
}
// If remote wakeup is disabled, stop polling
} else {
let __v9: bool = ((unsafe { (*ohci).__slate_bits_0.__get_autostop() }) as i32) == 0 && ((unsafe { (*(*ohci_to_hcd(ohci as *const ohci_hcd)).self_.root_hub).__slate_bits_1.__get_do_remote_wakeup() }) as i32) == 0;
if __v9 {
poll_rh = 0;
} else {
// If no status changes are pending,
// enable RHSC interrupts
if rhsc_enable == 0 && rhsc_status == 0 {
rhsc_enable = 64i32;
ohci_writel!(ohci, rhsc_enable, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrenable) });
}
// Keep polling until RHSC is enabled
if rhsc_enable != 0 {
poll_rh = 0;
}
}
}
// if there is a port change, autostart or ask to be resumed
}
_ => {
}
}
poll_rh
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_hub_status_data")]
extern "C" fn ohci_hub_status_data(hcd: *mut usb_hcd, buf: *mut u8) -> i32 {
// undocumented erratum seen on at least rev D
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut rhsc_status: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(rhsc_status) as *mut i32;
let mut any_connected: i32 = unsafe { core::mem::zeroed() };
let mut length: i32 = unsafe { core::mem::zeroed() };
let mut changed: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut ohci: *mut ohci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ohci = hcd_to_ohci(hcd);
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
changed = 0;
length = 1;
any_connected = 0;
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_14);
if HCD_HW_ACCESSIBLE!(hcd) != 0 {
if (*ohci).flags & 1 != 0 {
__v13 = roothub_a(ohci) & 255u32 > 15;
} else {
__v13 = false;
}
if __v13 {
while false {
}
_dev_warn((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"bogus NDP, rereads as NDP=%d\n".as_ptr() as *const u8, ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).roothub.a)) & 255u32);
// retry later; "should not happen"
} else {
// init status
if roothub_status(ohci) & 196608u32 != 0 {
changed = 1;
*buf.wrapping_offset(0 as isize) = 1;
} else {
*buf.wrapping_offset(0 as isize) = 0;
}
if (*ohci).num_ports > 7 {
*buf.wrapping_offset(1 as isize) = 0;
length = length.wrapping_add(1);
}
// Clear the RHSC status flag before reading the port statuses
ohci_writel!(ohci, 64i32, core::ptr::addr_of_mut!((*(*ohci).regs).intrstatus));
*__slate_slot_15 = (ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).intrstatus)) & 64u32) as i32;
// look at each port
*__slate_slot_19 = 0;
while *__slate_slot_19 < (*ohci).num_ports {
let status: u32 = roothub_portstatus(ohci, *__slate_slot_19);
// can't autostop if ports are connected
any_connected = ((any_connected as u32) | status & 1) as i32;
if status & 2031616u32 != 0 {
changed = 1;
if *__slate_slot_19 < 7 {
let __v4: *mut u8 = buf.wrapping_offset(0 as isize);
*__v4 = ((*__v4 as i32) | 1i32 << (*__slate_slot_19).wrapping_add(1)) as u8;
} else {
let __v7: *mut u8 = buf.wrapping_offset(1 as isize);
*__v7 = ((*__v7 as i32) | 1i32 << (*__slate_slot_19).wrapping_sub(7)) as u8;
}
}
let __v1: i32 = *__slate_slot_19;
*__slate_slot_19 = __v1.wrapping_add(1);
}
if ohci_root_hub_state_changes(ohci, changed, any_connected, *__slate_slot_15) != 0 {
set_bit(2i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
} else {
clear_bit(2i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
}
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_14);
return if changed != 0 { length } else { 0 };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ohci_hub_status_data_639: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, *mut u8) -> i32>, *mut ()>(Some(ohci_hub_status_data)) };

fn ohci_hub_descriptor(ohci: *mut ohci_hcd, desc: *mut usb_hub_descriptor) {
let mut rh: u32 = roothub_a(ohci);
let mut temp: u16 = 0;
unsafe {
(*desc).bDescriptorType = 41u8;
}
unsafe {
(*desc).bPwrOn2PwrGood = ((rh & 4278190080u32) >> 24i32) as u8;
}
unsafe {
(*desc).bHubContrCurrent = 0;
}
unsafe {
(*desc).bNbrPorts = (*ohci).num_ports as u8;
}
temp = 1i32.wrapping_add((unsafe { (*ohci).num_ports }) / 8) as u16;
unsafe {
(*desc).bDescLength = 7i32.wrapping_add(2i32.wrapping_mul(temp as i32)) as u8;
}
temp = 0u16;
if rh & 512u32 != 0 {
temp = ((temp as i32) | 2) as u16;
}
// no power switching?
if rh & 256u32 != 0 {
temp = ((temp as i32) | 1) as u16;
}
// per-port power switching?
if rh & 4096u32 != 0 {
temp = ((temp as i32) | 16) as u16;
} else {
if rh & 2048u32 != 0 {
temp = ((temp as i32) | 8) as u16;
}
}
// no overcurrent reporting?
// per-port overcurrent reporting?
unsafe {
(*desc).wHubCharacteristics = temp;
}
// ports removable, and usb 1.0 legacy PortPwrCtrlMask
rh = roothub_b(ohci);
unsafe { memset((core::ptr::addr_of_mut!((*desc).u.hs.DeviceRemovable) as *mut u8) as *mut (), 255i32, core::mem::size_of::<[u8; 4]>() as u64) };
unsafe {
(core::ptr::addr_of_mut!((*desc).u.hs.DeviceRemovable) as *mut u8).wrapping_offset(0 as isize).write_unaligned((rh & 65535) as u8);
}
if (unsafe { (*ohci).num_ports }) > 7 {
unsafe {
(core::ptr::addr_of_mut!((*desc).u.hs.DeviceRemovable) as *mut u8).wrapping_offset(1 as isize).write_unaligned(((rh & 65535) >> 8i32) as u8);
}
unsafe {
(core::ptr::addr_of_mut!((*desc).u.hs.DeviceRemovable) as *mut u8).wrapping_offset(2 as isize).write_unaligned(255u8);
}
} else {
unsafe {
(core::ptr::addr_of_mut!((*desc).u.hs.DeviceRemovable) as *mut u8).wrapping_offset(1 as isize).write_unaligned(255u8);
}
}
}

// See usb 7.1.7.5:  root hubs must issue at least 50 msec reset signaling,
// not necessarily continuous ... to guard against resume signaling.
// this timer value might be vendor-specific ...
fn root_port_reset(ohci: *mut ohci_hcd, port: u32) -> i32 {
let portstat: *mut u32 = unsafe { (*(*ohci).regs).roothub.portstatus.as_mut_ptr() as *mut u32 }.wrapping_offset(port as isize);
let mut temp: u32 = 0;
let mut now: u16 = ohci_readl!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).fmnumber) }) as u16;
let reset_done: u16 = (now as i32).wrapping_add(50) as u16;
let mut limit_1: i32 = 5i32;
// build a "continuous enough" reset signal, with up to
// 3msec gap between pulses.  scheduler HZ==100 must work;
// this might need to be deadline-scheduled.
loop {
let mut limit_2: i32 = 0;
// spin until any current reset finishes
limit_2 = 20i32;
loop {
let __v2: i32 = limit_2.wrapping_sub(1);
limit_2 = __v2;
if __v2 < 0 {
break;
}
temp = ohci_readl!(ohci, portstat);
// handle e.g. CardBus eject
if temp == !0 {
return 108i32.wrapping_neg();
}
if temp & 16 == 0 {
break;
}
udelay(500u64);
}
// timeout (a hardware error) has been observed when
// EHCI sets CF while this driver is resetting a port;
// presumably other disconnect paths might do it too.
if limit_2 < 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"port[%d] reset timeout, stat %08x\n".as_ptr() as *const u8, port, temp) };
}
break;
}
if temp & 1 == 0 {
break;
}
if temp & 1048576 != 0 {
ohci_writel!(ohci, 1048576i32, portstat);
}
// start the next reset, sleep till it's probably done
ohci_writel!(ohci, 16i32, portstat);
unsafe { msleep(10u32) };
now = ohci_readl!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).fmnumber) }) as u16;
let __v4: bool = if tick_before!(now, reset_done) {
let __v6: i32 = limit_1.wrapping_sub(1);
limit_1 = __v6;
__v6 >= 0
} else {
false
};
if !__v4 {
break;
}
}
// caller synchronizes using PRSC ... and handles PRS
// still being set when this returns.
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_hub_control")]
extern "C" fn ohci_hub_control(hcd: *mut usb_hcd, typeReq: u16, wValue: u16, mut wIndex: u16, buf: *mut u8, wLength: u16) -> i32 {
let mut retval: i32 = unsafe { core::mem::zeroed() };
let mut temp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(temp) as *mut u32;
let mut ports: i32 = unsafe { core::mem::zeroed() };
let mut ohci: *mut ohci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ohci = hcd_to_ohci(hcd);
ports = (*ohci).num_ports;
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
retval = 0;
if unlikely!(HCD_HW_ACCESSIBLE!(hcd) == 0) != 0 {
return 108i32.wrapping_neg();
} else {
'__join_0: {
'__join_1: {
let __t4: i32 = typeReq as i32;
if __t4 == 8193 {
let __t3: i32 = wValue as i32;
if __t3 == 1 {
ohci_writel!(ohci, 131072i32, core::ptr::addr_of_mut!((*(*ohci).regs).roothub.status));
break '__join_0;
} else {
if __t3 == 0 {
break '__join_0;
}
}
} else {
if __t4 == 8961 {
if !(wIndex == 0 || (wIndex as i32) > ports) {
wIndex = (wIndex as i32).wrapping_sub(1) as u16;
let __t2: i32 = wValue as i32;
if __t2 == 1 {
*__slate_slot_7 = 1;
} else {
if __t2 == 17 {
*__slate_slot_7 = 131072;
} else {
if __t2 == 2 {
*__slate_slot_7 = 8;
} else {
if __t2 == 18 {
*__slate_slot_7 = 262144;
} else {
if __t2 == 8 {
*__slate_slot_7 = 512;
} else {
if __t2 == 16 {
*__slate_slot_7 = 65536;
} else {
if __t2 == 19 {
*__slate_slot_7 = 524288;
} else {
if __t2 == 20 {
*__slate_slot_7 = 1048576;
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
ohci_writel__2!(ohci, *__slate_slot_7, ((*(*ohci).regs).roothub.portstatus.as_mut_ptr() as *mut u32).wrapping_offset((wIndex as i32) as isize));
// ohci_readl (ohci, &ohci->regs->roothub.portstatus [wIndex]);
break '__join_0;
}
} else {
if __t4 == 40966 {
ohci_hub_descriptor(ohci, buf as *mut usb_hub_descriptor);
break '__join_0;
} else {
if __t4 == 40960 {
*__slate_slot_7 = roothub_status(ohci) & !(2147483648 | 32768);
put_unaligned_le32(*__slate_slot_7, buf as *mut ());
break '__join_0;
} else {
if __t4 == 41728 {
if !(wIndex == 0 || (wIndex as i32) > ports) {
wIndex = (wIndex as i32).wrapping_sub(1) as u16;
*__slate_slot_7 = roothub_portstatus(ohci, wIndex as i32);
put_unaligned_le32(*__slate_slot_7, buf as *mut ());
if *(buf.wrapping_offset(2 as isize) as *mut u16) != 0 {
dbg_port!(ohci, c"GetStatus".as_ptr() as *mut u8, wIndex, *__slate_slot_7);
}
// only if wPortChange is interesting
break '__join_0;
}
} else {
if __t4 == 8195 {
let __t1: i32 = wValue as i32;
if __t1 == 1 {
break '__join_0;
} else {
if __t1 == 0 {
break '__join_0;
}
}
} else {
if __t4 == 8963 {
if !(wIndex == 0 || (wIndex as i32) > ports) {
wIndex = (wIndex as i32).wrapping_sub(1) as u16;
let __t0: i32 = wValue as i32;
if __t0 == 2 {
ohci_writel!(ohci, 4i32, ((*(*ohci).regs).roothub.portstatus.as_mut_ptr() as *mut u32).wrapping_offset((wIndex as i32) as isize));
break '__join_0;
} else {
if __t0 == 8 {
ohci_writel!(ohci, 256i32, ((*(*ohci).regs).roothub.portstatus.as_mut_ptr() as *mut u32).wrapping_offset((wIndex as i32) as isize));
break '__join_0;
} else {
if __t0 == 4 {
retval = root_port_reset(ohci, wIndex as u32);
break '__join_0;
}
}
}
}
}
}
}
}
}
}
}
}
retval = 32i32.wrapping_neg(); // "protocol stall" on error
}
return retval;
}
}
// FIXME:  this can be cleared, yes?
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ohci_hub_control_640: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, u16, u16, u16, *mut u8, u16) -> i32>, *mut ()>(Some(ohci_hub_control)) };

fn ohci_dump_intr_mask(ohci: *mut ohci_hcd, label: *mut u8, mask: u32, next: *mut *mut u8, size: *mut u32) {
loop {
if !next.is_null() {
let s_len: u32 = (unsafe { scnprintf(*next, *size as u64, c"%s 0x%08x%s%s%s%s%s%s%s%s%s\n".as_ptr() as *const u8, label, mask, if mask & 2147483648u32 != 0 { c" MIE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 1073741824u32 != 0 { c" OC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 64u32 != 0 { c" RHSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 32u32 != 0 { c" FNO".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 16u32 != 0 { c" UE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 8u32 != 0 { c" RD".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 4u32 != 0 { c" SF".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 2u32 != 0 { c" WDH".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 1u32 != 0 { c" SO".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) }) as u32;
let __v1: u32 = unsafe { *size };
unsafe {
*size = __v1.wrapping_sub(s_len);
}
let __v4: *mut u8 = unsafe { *next };
unsafe {
*next = __v4.wrapping_offset(s_len as isize);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"%s 0x%08x%s%s%s%s%s%s%s%s%s\n".as_ptr() as *const u8, label, mask, if mask & 2147483648u32 != 0 { c" MIE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 1073741824u32 != 0 { c" OC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 64u32 != 0 { c" RHSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 32u32 != 0 { c" FNO".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 16u32 != 0 { c" UE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 8u32 != 0 { c" RD".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 4u32 != 0 { c" SF".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 2u32 != 0 { c" WDH".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if mask & 1u32 != 0 { c" SO".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}
}
if true {
break;
}
}
}

fn maybe_print_eds(ohci: *mut ohci_hcd, label: *mut u8, value: u32, next: *mut *mut u8, size: *mut u32) {
if value != 0 {
loop {
if !next.is_null() {
let s_len: u32 = (unsafe { scnprintf(*next, *size as u64, c"%s %08x\n".as_ptr() as *const u8, label, value) }) as u32;
let __v1: u32 = unsafe { *size };
unsafe {
*size = __v1.wrapping_sub(s_len);
}
let __v4: *mut u8 = unsafe { *next };
unsafe {
*next = __v4.wrapping_offset(s_len as isize);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"%s %08x\n".as_ptr() as *const u8, label, value) };
}
}
if true {
break;
}
}
}
}

fn hcfs2string(state: i32) -> *mut u8 {
match state {
0 => {
return c"reset".as_ptr() as *mut u8;
}
64 => {
return c"resume".as_ptr() as *mut u8;
}
128 => {
return c"operational".as_ptr() as *mut u8;
}
192 => {
return c"suspend".as_ptr() as *mut u8;
}
_ => {
}
}
c"?".as_ptr() as *mut u8
}

fn rh_state_string(ohci: *mut ohci_hcd) -> *const u8 {
match (unsafe { (*ohci).rh_state }) as u32 {
0 => {
return c"halted".as_ptr() as *const u8;
}
1 => {
return c"suspended".as_ptr() as *const u8;
}
2 => {
return c"running".as_ptr() as *const u8;
}
_ => {
}
}
c"?".as_ptr() as *const u8
}

fn ohci_dump_status(controller: *mut ohci_hcd, next: *mut *mut u8, size: *mut u32) {
let regs: *mut ohci_regs = unsafe { (*controller).regs };
let mut temp: u32 = ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).revision) }) & 255;
loop {
if !next.is_null() {
let s_len: u32 = (unsafe { scnprintf(*next, *size as u64, c"OHCI %d.%d, %s legacy support registers, rh state %s\n".as_ptr() as *const u8, 3 & temp >> 4i32, temp & 15, if temp & 256 != 0 { c"with".as_ptr() as *mut u8 } else { c"NO".as_ptr() as *mut u8 }, rh_state_string(controller)) }) as u32;
let __v1: u32 = unsafe { *size };
unsafe {
*size = __v1.wrapping_sub(s_len);
}
let __v4: *mut u8 = unsafe { *next };
unsafe {
*next = __v4.wrapping_offset(s_len as isize);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(controller as *const ohci_hcd)).self_.controller as *const device, c"OHCI %d.%d, %s legacy support registers, rh state %s\n".as_ptr() as *const u8, 3 & temp >> 4i32, temp & 15, if temp & 256 != 0 { c"with".as_ptr() as *mut u8 } else { c"NO".as_ptr() as *mut u8 }, rh_state_string(controller)) };
}
}
if true {
break;
}
}
temp = ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).control) });
loop {
if !next.is_null() {
let s_len: u32 = (unsafe { scnprintf(*next, *size as u64, c"control 0x%03x%s%s%s HCFS=%s%s%s%s%s CBSR=%d\n".as_ptr() as *const u8, temp, if temp & 1024u32 != 0 { c" RWE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 512u32 != 0 { c" RWC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 256u32 != 0 { c" IR".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, hcfs2string((temp & 192u32) as i32), if temp & 32u32 != 0 { c" BLE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 16u32 != 0 { c" CLE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 8u32 != 0 { c" IE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 4u32 != 0 { c" PLE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, temp & 3u32) }) as u32;
let __v8: u32 = unsafe { *size };
unsafe {
*size = __v8.wrapping_sub(s_len);
}
let __v11: *mut u8 = unsafe { *next };
unsafe {
*next = __v11.wrapping_offset(s_len as isize);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(controller as *const ohci_hcd)).self_.controller as *const device, c"control 0x%03x%s%s%s HCFS=%s%s%s%s%s CBSR=%d\n".as_ptr() as *const u8, temp, if temp & 1024u32 != 0 { c" RWE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 512u32 != 0 { c" RWC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 256u32 != 0 { c" IR".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, hcfs2string((temp & 192u32) as i32), if temp & 32u32 != 0 { c" BLE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 16u32 != 0 { c" CLE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 8u32 != 0 { c" IE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 4u32 != 0 { c" PLE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, temp & 3u32) };
}
}
if true {
break;
}
}
temp = ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).cmdstatus) });
loop {
if !next.is_null() {
let s_len: u32 = (unsafe { scnprintf(*next, *size as u64, c"cmdstatus 0x%05x SOC=%d%s%s%s%s\n".as_ptr() as *const u8, temp, (temp & 196608u32) >> 16i32, if temp & 8u32 != 0 { c" OCR".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 4u32 != 0 { c" BLF".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 2u32 != 0 { c" CLF".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 1u32 != 0 { c" HCR".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) }) as u32;
let __v15: u32 = unsafe { *size };
unsafe {
*size = __v15.wrapping_sub(s_len);
}
let __v18: *mut u8 = unsafe { *next };
unsafe {
*next = __v18.wrapping_offset(s_len as isize);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(controller as *const ohci_hcd)).self_.controller as *const device, c"cmdstatus 0x%05x SOC=%d%s%s%s%s\n".as_ptr() as *const u8, temp, (temp & 196608u32) >> 16i32, if temp & 8u32 != 0 { c" OCR".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 4u32 != 0 { c" BLF".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 2u32 != 0 { c" CLF".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 1u32 != 0 { c" HCR".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}
}
if true {
break;
}
}
ohci_dump_intr_mask(controller, c"intrstatus".as_ptr() as *mut u8, ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).intrstatus) }), next, size);
ohci_dump_intr_mask(controller, c"intrenable".as_ptr() as *mut u8, ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).intrenable) }), next, size);
// intrdisable always same as intrenable
maybe_print_eds(controller, c"ed_periodcurrent".as_ptr() as *mut u8, ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).ed_periodcurrent) }), next, size);
maybe_print_eds(controller, c"ed_controlhead".as_ptr() as *mut u8, ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).ed_controlhead) }), next, size);
maybe_print_eds(controller, c"ed_controlcurrent".as_ptr() as *mut u8, ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).ed_controlcurrent) }), next, size);
maybe_print_eds(controller, c"ed_bulkhead".as_ptr() as *mut u8, ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).ed_bulkhead) }), next, size);
maybe_print_eds(controller, c"ed_bulkcurrent".as_ptr() as *mut u8, ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).ed_bulkcurrent) }), next, size);
maybe_print_eds(controller, c"donehead".as_ptr() as *mut u8, ohci_readl!(controller, unsafe { core::ptr::addr_of_mut!((*regs).donehead) }), next, size);
}

fn ohci_dump_roothub(controller: *mut ohci_hcd, verbose: i32, next: *mut *mut u8, size: *mut u32) {
let mut temp: u32 = roothub_a(controller);
let mut i: u32 = 0;
if temp == !0 {
return;
}
if verbose != 0 {
loop {
if !next.is_null() {
let s_len: u32 = (unsafe { scnprintf(*next, *size as u64, c"roothub.a %08x POTPGT=%d%s%s%s%s%s NDP=%d(%d)\n".as_ptr() as *const u8, temp, (temp & 4278190080u32) >> 24i32 & 255, if temp & 4096u32 != 0 { c" NOCP".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 2048u32 != 0 { c" OCPM".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 1024u32 != 0 { c" DT".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 512u32 != 0 { c" NPS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 256u32 != 0 { c" PSM".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, temp & 255u32, (*controller).num_ports) }) as u32;
let __v1: u32 = unsafe { *size };
unsafe {
*size = __v1.wrapping_sub(s_len);
}
let __v4: *mut u8 = unsafe { *next };
unsafe {
*next = __v4.wrapping_offset(s_len as isize);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(controller as *const ohci_hcd)).self_.controller as *const device, c"roothub.a %08x POTPGT=%d%s%s%s%s%s NDP=%d(%d)\n".as_ptr() as *const u8, temp, (temp & 4278190080u32) >> 24i32 & 255, if temp & 4096u32 != 0 { c" NOCP".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 2048u32 != 0 { c" OCPM".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 1024u32 != 0 { c" DT".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 512u32 != 0 { c" NPS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 256u32 != 0 { c" PSM".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, temp & 255u32, (*controller).num_ports) };
}
}
if true {
break;
}
}
temp = roothub_b(controller);
loop {
if !next.is_null() {
let s_len: u32 = (unsafe { scnprintf(*next, *size as u64, c"roothub.b %08x PPCM=%04x DR=%04x\n".as_ptr() as *const u8, temp, (temp & 4294901760) >> 16i32, temp & 65535) }) as u32;
let __v8: u32 = unsafe { *size };
unsafe {
*size = __v8.wrapping_sub(s_len);
}
let __v11: *mut u8 = unsafe { *next };
unsafe {
*next = __v11.wrapping_offset(s_len as isize);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(controller as *const ohci_hcd)).self_.controller as *const device, c"roothub.b %08x PPCM=%04x DR=%04x\n".as_ptr() as *const u8, temp, (temp & 4294901760) >> 16i32, temp & 65535) };
}
}
if true {
break;
}
}
temp = roothub_status(controller);
loop {
if !next.is_null() {
let s_len: u32 = (unsafe { scnprintf(*next, *size as u64, c"roothub.status %08x%s%s%s%s%s%s\n".as_ptr() as *const u8, temp, if temp & 2147483648 != 0 { c" CRWE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 131072 != 0 { c" OCIC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 65536 != 0 { c" LPSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 32768 != 0 { c" DRWE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 2 != 0 { c" OCI".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 1 != 0 { c" LPS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) }) as u32;
let __v15: u32 = unsafe { *size };
unsafe {
*size = __v15.wrapping_sub(s_len);
}
let __v18: *mut u8 = unsafe { *next };
unsafe {
*next = __v18.wrapping_offset(s_len as isize);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(controller as *const ohci_hcd)).self_.controller as *const device, c"roothub.status %08x%s%s%s%s%s%s\n".as_ptr() as *const u8, temp, if temp & 2147483648 != 0 { c" CRWE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 131072 != 0 { c" OCIC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 65536 != 0 { c" LPSC".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 32768 != 0 { c" DRWE".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 2 != 0 { c" OCI".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if temp & 1 != 0 { c" LPS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}
}
if true {
break;
}
}
}
while i < ((unsafe { (*controller).num_ports }) as u32) {
temp = roothub_portstatus(controller, i as i32);
dbg_port_sw!(controller, i, next, size, temp);
i = i.wrapping_add(1);
}
}

fn ohci_dump(controller: *mut ohci_hcd) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(controller as *const ohci_hcd)).self_.controller as *const device, c"OHCI controller state\n".as_ptr() as *const u8) };
}
// dumps some of the state we know about
ohci_dump_status(controller, core::ptr::null_mut::<*mut u8>(), core::ptr::null_mut::<u32>());
if !unsafe { (*controller).hcca }.is_null() {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(controller as *const ohci_hcd)).self_.controller as *const device, c"hcca frame #%04x\n".as_ptr() as *const u8, ohci_frame_no(controller as *const ohci_hcd) as i32) };
}
}
ohci_dump_roothub(controller, 1i32, core::ptr::null_mut::<*mut u8>(), core::ptr::null_mut::<u32>());
}

static data0: [u8; 6] = [68, 65, 84, 65, 48, 0];

static data1: [u8; 6] = [68, 65, 84, 65, 49, 0];

static debug_async_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(default_llseek as *const ()) };
__t0.read = Some(debug_output);
__t0.open = Some(debug_async_open);
__t0.release = Some(debug_close);
__t0
});

static debug_periodic_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(default_llseek as *const ()) };
__t0.read = Some(debug_output);
__t0.open = Some(debug_periodic_open);
__t0.release = Some(debug_close);
__t0
});

static debug_registers_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(default_llseek as *const ()) };
__t0.read = Some(debug_output);
__t0.open = Some(debug_registers_open);
__t0.release = Some(debug_close);
__t0
});

static mut ohci_debug_root: *mut dentry = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct debug_buffer {
    fill_func: Option<unsafe extern "C" fn(*mut debug_buffer) -> i64>,
    ohci: *mut ohci_hcd,
    mutex: mutex,
    count: u64,
    page: *mut u8,
}


fn show_list(ohci: *mut ohci_hcd, mut buf: *mut u8, count: u64, mut ed: *mut ed) -> i64 {
let mut temp: u32 = 0;
let mut size: u32 = count as u32;
if ed.is_null() {
return 0;
}
// print first --> last
while !unsafe { (*ed).ed_prev }.is_null() {
ed = unsafe { (*ed).ed_prev };
}
// dump a snapshot of the bulk or control schedule
while !ed.is_null() {
let mut info: u32 = hc32_to_cpu(ohci as *const ohci_hcd, unsafe { (*ed).hwINFO });
let headp: u32 = hc32_to_cpu(ohci as *const ohci_hcd, unsafe { (*ed).hwHeadP });
let mut entry: *mut list_head = unsafe { core::mem::zeroed() };
let mut td: *mut td = unsafe { core::mem::zeroed() };
temp = (unsafe { scnprintf(buf, size as u64, c"ed/%p %cs dev%d ep%d%s max %d %08x%s%s %s".as_ptr() as *const u8, ed, if info & 8192u32 != 0 { 108i32 } else { 102i32 }, info & 127, info >> 7i32 & 15, if info & 4096u32 != 0 { c"in".as_ptr() as *mut u8 } else { c"out".as_ptr() as *mut u8 }, 1023 & info >> 16i32, info, if info & 16384u32 != 0 { c" s".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if headp & 1 != 0 { c" H".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if headp & 2 != 0 { core::ptr::addr_of!(data1) as *const u8 } else { core::ptr::addr_of!(data0) as *const u8 }) }) as u32;
size = size.wrapping_sub(temp);
let __v3: *mut u8 = buf;
buf = __v3.wrapping_offset(temp as isize);
entry = unsafe { (*core::ptr::addr_of_mut!((*ed).td_list)).next };
while list_is_head(entry as *const list_head, (unsafe { core::ptr::addr_of_mut!((*ed).td_list) }) as *const list_head) == 0 {
let mut cbp: u32 = 0;
let mut be: u32 = 0;
td = list_entry__3!(entry, td);
info = hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*td).hwINFO) }) as *const u32);
cbp = hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*td).hwCBP) }) as *const u32);
be = hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*td).hwBE) }) as *const u32);
let mut __v6: *mut u8;
let mut pid: *mut u8 = unsafe { core::mem::zeroed() };
match info & 1572864 {
0 => {
pid = c"setup".as_ptr() as *mut u8;
}
1048576 => {
pid = c"in".as_ptr() as *mut u8;
}
524288 => {
pid = c"out".as_ptr() as *mut u8;
}
_ => {
pid = c"(?)".as_ptr() as *mut u8;
}
}
__v6 = pid;
temp = (unsafe { scnprintf(buf, size as u64, c"\n\ttd %p %s %d cc=%x urb %p (%08x)".as_ptr() as *const u8, td, __v6, if cbp != 0 { be.wrapping_add(1).wrapping_sub(cbp) } else { 0u32 }, TD_CC_GET!(info), (*td).urb, info) }) as u32;
size = size.wrapping_sub(temp);
let __v10: *mut u8 = buf;
buf = __v10.wrapping_offset(temp as isize);
entry = unsafe { (*entry).next };
}
temp = (unsafe { scnprintf(buf, size as u64, c"\n".as_ptr() as *const u8) }) as u32;
size = size.wrapping_sub(temp);
let __v15: *mut u8 = buf;
buf = __v15.wrapping_offset(temp as isize);
ed = unsafe { (*ed).ed_next };
}
count.wrapping_sub(size as u64) as i64
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.fill_async_buffer")]
extern "C" fn fill_async_buffer(buf: *mut debug_buffer) -> i64 {
let mut ohci: *mut ohci_hcd = unsafe { core::mem::zeroed() };
let mut temp: u64 = 0;
let mut size: u64 = 0;
let mut flags: u64 = 0;
ohci = unsafe { (*buf).ohci };
size = _AC!() << 12i32;
// display control and bulk lists together, for simplicity
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
temp = show_list(ohci, unsafe { (*buf).page }, size, unsafe { (*ohci).ed_controltail }) as u64;
let __v5: i64 = show_list(ohci, unsafe { (*buf).page }.wrapping_offset(temp as isize), size.wrapping_sub(temp), unsafe { (*ohci).ed_bulktail });
temp = temp.wrapping_add(__v5 as u64);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
temp as i64
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.fill_periodic_buffer")]
extern "C" fn fill_periodic_buffer(buf: *mut debug_buffer) -> i64 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut ohci: *mut ohci_hcd = unsafe { core::mem::zeroed() };
let mut ed: *mut ed = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
let mut temp: u32 = 0;
let mut size: u32 = 0;
let mut seen_count: u32 = 0;
let mut next: *mut u8 = unsafe { core::mem::zeroed() };
let mut __v0: *mut *mut ed;
let __obj_size: u64 = size_mul(core::mem::size_of::<*mut ed>() as u64, 64u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 2080u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 2080u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut *mut ed;
let seen: *mut *mut ed = __v0;
if seen.is_null() {
return 0;
}
ohci = unsafe { (*buf).ohci };
next = unsafe { (*buf).page };
size = (_AC!() << 12i32) as u32;
temp = (unsafe { scnprintf(next, size as u64, c"size = %d\n".as_ptr() as *const u8, 32i32) }) as u32;
size = size.wrapping_sub(temp);
let __v8: *mut u8 = next;
next = __v8.wrapping_offset(temp as isize);
// dump a snapshot of the periodic schedule (and load)
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
for i in 0u32..32u32 {
ed = unsafe { *((*ohci).periodic.as_mut_ptr() as *mut *mut ed).wrapping_offset(i as isize) };
if !ed.is_null() {
temp = (unsafe { scnprintf(next, size as u64, c"%2d [%3d]:".as_ptr() as *const u8, i, *((*ohci).load.as_mut_ptr() as *mut i32).wrapping_offset(i as isize)) }) as u32;
size = size.wrapping_sub(temp);
let __v18: *mut u8 = next;
next = __v18.wrapping_offset(temp as isize);
loop {
temp = (unsafe { scnprintf(next, size as u64, c" ed%d/%p".as_ptr() as *const u8, (*ed).interval as i32, ed) }) as u32;
size = size.wrapping_sub(temp);
let __v22: *mut u8 = next;
next = __v22.wrapping_offset(temp as isize);
temp = 0;
while temp < seen_count {
if (unsafe { *seen.wrapping_offset(temp as isize) }) == ed {
break;
}
temp = temp.wrapping_add(1);
}
// show more info the first time around
if temp == seen_count {
let info: u32 = hc32_to_cpu(ohci as *const ohci_hcd, unsafe { (*ed).hwINFO });
let mut entry: *mut list_head = unsafe { core::mem::zeroed() };
let mut qlen: u32 = 0;
// qlen measured here in TDs, not urbs
entry = unsafe { (*core::ptr::addr_of_mut!((*ed).td_list)).next };
while list_is_head(entry as *const list_head, (unsafe { core::ptr::addr_of_mut!((*ed).td_list) }) as *const list_head) == 0 {
qlen = qlen.wrapping_add(1);
entry = unsafe { (*entry).next };
}
temp = (unsafe { scnprintf(next, size as u64, c" (%cs dev%d ep%d%s-%s qlen %u max %d %08x%s%s)".as_ptr() as *const u8, if info & 8192u32 != 0 { 108i32 } else { 102i32 }, info & 127, info >> 7i32 & 15, if info & 4096u32 != 0 { c"in".as_ptr() as *mut u8 } else { c"out".as_ptr() as *mut u8 }, if info & 32768u32 != 0 { c"iso".as_ptr() as *mut u8 } else { c"int".as_ptr() as *mut u8 }, qlen, 1023 & info >> 16i32, info, if info & 16384u32 != 0 { c" K".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if (*ed).hwHeadP & cpu_to_hc32(ohci as *const ohci_hcd, 1u32) != 0 { c" H".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) }) as u32;
size = size.wrapping_sub(temp);
let __v32: *mut u8 = next;
next = __v32.wrapping_offset(temp as isize);
if seen_count < 64 {
let __v34: u32 = seen_count;
seen_count = __v34.wrapping_add(1);
unsafe {
*seen.wrapping_offset(__v34 as isize) = ed;
}
}
ed = unsafe { (*ed).ed_next };
} else {
// we've seen it and what's after
temp = 0;
ed = core::ptr::null_mut::<ed>();
}
if ed.is_null() {
break;
}
}
temp = (unsafe { scnprintf(next, size as u64, c"\n".as_ptr() as *const u8) }) as u32;
size = size.wrapping_sub(temp);
let __v39: *mut u8 = next;
next = __v39.wrapping_offset(temp as isize);
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
unsafe { kfree(seen as *const ()) };
(_AC!() << 12i32).wrapping_sub(size as u64) as i64
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.fill_registers_buffer")]
extern "C" fn fill_registers_buffer(buf: *mut debug_buffer) -> i64 {
let mut s_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(s_len) as *mut u32;
let mut s_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut u32 = core::ptr::addr_of_mut!(s_len) as *mut u32;
let mut rdata: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut u32 = core::ptr::addr_of_mut!(rdata) as *mut u32;
let mut next: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut *mut u8 = core::ptr::addr_of_mut!(next) as *mut *mut u8;
let mut size: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut u32 = core::ptr::addr_of_mut!(size) as *mut u32;
let mut temp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut u32 = core::ptr::addr_of_mut!(temp) as *mut u32;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut regs: core::mem::MaybeUninit<*mut ohci_regs> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut *mut ohci_regs = core::ptr::addr_of_mut!(regs) as *mut *mut ohci_regs;
let mut ohci: core::mem::MaybeUninit<*mut ohci_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut *mut ohci_hcd = core::ptr::addr_of_mut!(ohci) as *mut *mut ohci_hcd;
let mut hcd: core::mem::MaybeUninit<*mut usb_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut *mut usb_hcd = core::ptr::addr_of_mut!(hcd) as *mut *mut usb_hcd;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
*__slate_slot_43 = (*buf).ohci;
*__slate_slot_44 = ohci_to_hcd(*__slate_slot_43 as *const ohci_hcd);
*__slate_slot_42 = (*(*__slate_slot_43)).regs;
*__slate_slot_38 = (*buf).page;
*__slate_slot_39 = (_AC!() << 12i32) as u32;
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_43)).lock), *__slate_slot_41);
// dump driver info, then registers in spec order
loop {
let __slate_auto_bytes = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_36 = scnprintf(*core::ptr::addr_of_mut!(*__slate_slot_38), *core::ptr::addr_of_mut!(*__slate_slot_39) as u64, c"bus %s, device %s\n%s\n%s\n".as_ptr() as *const u8, (*(*(*(*__slate_slot_44)).self_.controller).bus).name, dev_name((*(*__slate_slot_44)).self_.controller as *const device), (*(*__slate_slot_44)).product_desc, core::ptr::addr_of!(hcd_name) as *const u8) as u32;
let __v35: *mut u32 = core::ptr::addr_of_mut!(*__slate_slot_39);
let __v34: u32 = *__v35;
*__v35 = __v34.wrapping_sub(*__slate_slot_36);
let __v32: *mut *mut u8 = core::ptr::addr_of_mut!(*__slate_slot_38);
let __v31: *mut u8 = *__v32;
*__v32 = __v31.wrapping_offset(*__slate_slot_36 as isize);
if true {
break;
}
}
if HCD_HW_ACCESSIBLE!(*__slate_slot_44) == 0 {
let __v29: i32 = scnprintf(*__slate_slot_38, *__slate_slot_39 as u64, c"SUSPENDED (no register access)\n".as_ptr() as *const u8);
let __v28: u32 = *__slate_slot_39;
*__slate_slot_39 = __v28.wrapping_sub(__v29 as u32);
} else {
ohci_dump_status(*__slate_slot_43, core::ptr::addr_of_mut!(*__slate_slot_38), core::ptr::addr_of_mut!(*__slate_slot_39));
// hcca
if !unsafe { (*(*__slate_slot_43)).hcca }.is_null() {
loop {
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_26 = scnprintf(*core::ptr::addr_of_mut!(*__slate_slot_38), *core::ptr::addr_of_mut!(*__slate_slot_39) as u64, c"hcca frame 0x%04x\n".as_ptr() as *const u8, ohci_frame_no(*__slate_slot_43 as *const ohci_hcd) as i32) as u32;
let __v25: *mut u32 = core::ptr::addr_of_mut!(*__slate_slot_39);
let __v24: u32 = *__v25;
*__v25 = __v24.wrapping_sub(*__slate_slot_26);
let __v22: *mut *mut u8 = core::ptr::addr_of_mut!(*__slate_slot_38);
let __v21: *mut u8 = *__v22;
*__v22 = __v21.wrapping_offset(*__slate_slot_26 as isize);
if true {
break;
}
}
}
// other registers mostly affect frame timings
*__slate_slot_37 = ohci_readl!(*__slate_slot_43, core::ptr::addr_of_mut!((*(*__slate_slot_42)).fminterval));
*__slate_slot_40 = scnprintf(*__slate_slot_38, *__slate_slot_39 as u64, c"fmintvl 0x%08x %sFSMPS=0x%04x FI=0x%04x\n".as_ptr() as *const u8, *__slate_slot_37, if *__slate_slot_37 >> 31i32 != 0 { c"FIT ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, *__slate_slot_37 >> 16i32 & 61439, *__slate_slot_37 & 65535) as u32;
let __v19: u32 = *__slate_slot_39;
*__slate_slot_39 = __v19.wrapping_sub(*__slate_slot_40);
let __v17: *mut u8 = *__slate_slot_38;
*__slate_slot_38 = __v17.wrapping_offset(*__slate_slot_40 as isize);
*__slate_slot_37 = ohci_readl!(*__slate_slot_43, core::ptr::addr_of_mut!((*(*__slate_slot_42)).fmremaining));
*__slate_slot_40 = scnprintf(*__slate_slot_38, *__slate_slot_39 as u64, c"fmremaining 0x%08x %sFR=0x%04x\n".as_ptr() as *const u8, *__slate_slot_37, if *__slate_slot_37 >> 31i32 != 0 { c"FRT ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, *__slate_slot_37 & 16383) as u32;
let __v15: u32 = *__slate_slot_39;
*__slate_slot_39 = __v15.wrapping_sub(*__slate_slot_40);
let __v13: *mut u8 = *__slate_slot_38;
*__slate_slot_38 = __v13.wrapping_offset(*__slate_slot_40 as isize);
*__slate_slot_37 = ohci_readl!(*__slate_slot_43, core::ptr::addr_of_mut!((*(*__slate_slot_42)).periodicstart));
*__slate_slot_40 = scnprintf(*__slate_slot_38, *__slate_slot_39 as u64, c"periodicstart 0x%04x\n".as_ptr() as *const u8, *__slate_slot_37 & 16383) as u32;
let __v11: u32 = *__slate_slot_39;
*__slate_slot_39 = __v11.wrapping_sub(*__slate_slot_40);
let __v9: *mut u8 = *__slate_slot_38;
*__slate_slot_38 = __v9.wrapping_offset(*__slate_slot_40 as isize);
*__slate_slot_37 = ohci_readl!(*__slate_slot_43, core::ptr::addr_of_mut!((*(*__slate_slot_42)).lsthresh));
*__slate_slot_40 = scnprintf(*__slate_slot_38, *__slate_slot_39 as u64, c"lsthresh 0x%04x\n".as_ptr() as *const u8, *__slate_slot_37 & 16383) as u32;
let __v7: u32 = *__slate_slot_39;
*__slate_slot_39 = __v7.wrapping_sub(*__slate_slot_40);
let __v5: *mut u8 = *__slate_slot_38;
*__slate_slot_38 = __v5.wrapping_offset(*__slate_slot_40 as isize);
*__slate_slot_40 = scnprintf(*__slate_slot_38, *__slate_slot_39 as u64, c"hub poll timer %s\n".as_ptr() as *const u8, if HCD_POLL_RH!(ohci_to_hcd(*__slate_slot_43 as *const ohci_hcd)) != 0 { c"ON".as_ptr() as *mut u8 } else { c"off".as_ptr() as *mut u8 }) as u32;
let __v3: u32 = *__slate_slot_39;
*__slate_slot_39 = __v3.wrapping_sub(*__slate_slot_40);
let __v1: *mut u8 = *__slate_slot_38;
*__slate_slot_38 = __v1.wrapping_offset(*__slate_slot_40 as isize);
// roothub
ohci_dump_roothub(*__slate_slot_43, 1i32, core::ptr::addr_of_mut!(*__slate_slot_38), core::ptr::addr_of_mut!(*__slate_slot_39));
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_43)).lock), *__slate_slot_41);
return (_AC!() << 12i32).wrapping_sub(*__slate_slot_39 as u64) as i64;
}
}

fn alloc_buffer(ohci: *mut ohci_hcd, fill_func: Option<unsafe extern "C" fn(*mut debug_buffer) -> i64>) -> *mut debug_buffer {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut buf: *mut debug_buffer = unsafe { core::mem::zeroed() };
let mut __v0: *mut debug_buffer;
let __obj_size: u64 = size_mul(core::mem::size_of::<debug_buffer>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut debug_buffer;
buf = __v0;
if !buf.is_null() {
unsafe {
(*buf).ohci = ohci;
}
unsafe {
(*buf).fill_func = fill_func;
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*buf).mutex) }, c"&buf->mutex".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key) });
if true {
break;
}
}
}
buf
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

fn fill_buffer(buf: *mut debug_buffer) -> i32 {
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
if unsafe { (*buf).page }.is_null() {
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(4096u64, 3264u32, *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(4096u64, 3264u32, *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
});
}
(*buf).page = *__slate_slot_1 as *mut u8;
}
if unsafe { (*buf).page }.is_null() {
*__slate_slot_4 = 12i32.wrapping_neg();
} else {
*__slate_slot_4 = unsafe { (*buf).fill_func }.unwrap()(buf) as i32;
if *__slate_slot_4 >= 0 {
(*buf).count = *__slate_slot_4 as u64;
*__slate_slot_4 = 0;
}
}
return *__slate_slot_4;
}
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.debug_output")]
extern "C" fn debug_output(file: *mut file, user_buf: *mut u8, len: u64, offset: *mut i64) -> i64 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut buf: *mut debug_buffer = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
buf = (*file).private_data as *mut debug_buffer;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
mutex_lock(core::ptr::addr_of_mut!((*buf).mutex));
if (*buf).count == 0 {
*__slate_slot_0 = fill_buffer(buf);
if *__slate_slot_0 != 0 {
mutex_unlock(core::ptr::addr_of_mut!((*buf).mutex));
break '__join_0;
}
}
mutex_unlock(core::ptr::addr_of_mut!((*buf).mutex));
*__slate_slot_0 = simple_read_from_buffer(user_buf as *mut (), len, offset, (*buf).page as *const (), (*buf).count) as i32;
}
return *__slate_slot_0 as i64;
}
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.debug_close")]
extern "C" fn debug_close(inode: *mut inode, file: *mut file) -> i32 {
let buf: *mut debug_buffer = (unsafe { (*file).private_data }) as *mut debug_buffer;
unsafe { kfree((*buf).page as *const ()) };
unsafe { kfree(buf as *const ()) };
0
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.debug_async_open")]
extern "C" fn debug_async_open(inode: *mut inode, file: *mut file) -> i32 {
unsafe {
(*file).private_data = alloc_buffer((*inode).i_private as *mut ohci_hcd, Some(fill_async_buffer)) as *mut ();
}
if !unsafe { (*file).private_data }.is_null() { 0 } else { 12i32.wrapping_neg() }
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.debug_periodic_open")]
extern "C" fn debug_periodic_open(inode: *mut inode, file: *mut file) -> i32 {
unsafe {
(*file).private_data = alloc_buffer((*inode).i_private as *mut ohci_hcd, Some(fill_periodic_buffer)) as *mut ();
}
if !unsafe { (*file).private_data }.is_null() { 0 } else { 12i32.wrapping_neg() }
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.debug_registers_open")]
extern "C" fn debug_registers_open(inode: *mut inode, file: *mut file) -> i32 {
unsafe {
(*file).private_data = alloc_buffer((*inode).i_private as *mut ohci_hcd, Some(fill_registers_buffer)) as *mut ();
}
if !unsafe { (*file).private_data }.is_null() { 0 } else { 12i32.wrapping_neg() }
}

fn create_debug_files(ohci: *mut ohci_hcd) {
let bus: *mut usb_bus = unsafe { core::ptr::addr_of_mut!((*ohci_to_hcd(ohci as *const ohci_hcd)).self_) };
let root: *mut dentry = unsafe { debugfs_create_dir((*bus).bus_name, ohci_debug_root) };
unsafe {
(*ohci).debug_dir = root;
}
debugfs_create_file!(c"async".as_ptr() as *mut u8, 292i32, root, ohci, unsafe { core::ptr::addr_of!(debug_async_fops.0) });
debugfs_create_file!(c"periodic".as_ptr() as *mut u8, 292i32, root, ohci, unsafe { core::ptr::addr_of!(debug_periodic_fops.0) });
debugfs_create_file!(c"registers".as_ptr() as *mut u8, 292i32, root, ohci, unsafe { core::ptr::addr_of!(debug_registers_fops.0) });
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"created debug files\n".as_ptr() as *const u8) };
}
}

fn remove_debug_files(ohci: *mut ohci_hcd) {
unsafe { debugfs_remove((*ohci).debug_dir) };
}

fn ohci_hcd_init(ohci: *mut ohci_hcd) {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
unsafe {
(*ohci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*ohci).lock) });
unsafe {
*core::ptr::addr_of_mut!((*ohci).lock) = *{
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
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ohci).pending) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ohci).eds_in_use) });
}

fn ohci_mem_init(ohci: *mut ohci_hcd) -> i32 {
// HCs with local memory allocate from localmem_pool so there's
// no need to create the below dma pools.
if !unsafe { (*ohci_to_hcd(ohci as *const ohci_hcd)).localmem_pool }.is_null() {
return 0;
}
unsafe {
(*ohci).td_cache = dma_pool_create(c"ohci_td".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller, core::mem::size_of::<td>() as u64, 32u64, 0u64);
}
// byte alignment
// no page-crossing issues
if unsafe { (*ohci).td_cache }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*ohci).ed_cache = dma_pool_create(c"ohci_ed".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller, core::mem::size_of::<ed>() as u64, 16u64, 0u64);
}
// byte alignment
// no page-crossing issues
if unsafe { (*ohci).ed_cache }.is_null() {
unsafe { dma_pool_destroy((*ohci).td_cache) };
return 12i32.wrapping_neg();
}
0
}

fn ohci_mem_cleanup(ohci: *mut ohci_hcd) {
unsafe { dma_pool_destroy((*ohci).td_cache) };
unsafe {
(*ohci).td_cache = core::ptr::null_mut::<dma_pool>();
}
unsafe { dma_pool_destroy((*ohci).ed_cache) };
unsafe {
(*ohci).ed_cache = core::ptr::null_mut::<dma_pool>();
}
}

fn dma_to_td(hc: *mut ohci_hcd, mut td_dma: u64) -> *mut td {
let mut td: *mut td = unsafe { core::mem::zeroed() };
let __v1: u64 = td_dma & 4294967264u64;
td_dma = __v1;
td = unsafe { *((*hc).td_hash.as_mut_ptr() as *mut *mut td).wrapping_offset(TD_HASH_FUNC!(td_dma) as isize) };
while !td.is_null() && (unsafe { (*td).td_dma }) != td_dma {
td = unsafe { (*td).td_hash };
}
td
}

fn td_alloc(hc: *mut ohci_hcd, mem_flags: u32) -> *mut td {
let mut dma: u64 = 0;
let mut td: *mut td = unsafe { core::mem::zeroed() };
let hcd: *mut usb_hcd = ohci_to_hcd(hc as *const ohci_hcd);
if !unsafe { (*hcd).localmem_pool }.is_null() {
td = (unsafe { gen_pool_dma_zalloc_align((*hcd).localmem_pool, core::mem::size_of::<td>() as u64, core::ptr::addr_of_mut!(dma), 32i32) }) as *mut td;
} else {
td = dma_pool_zalloc(unsafe { (*hc).td_cache }, mem_flags, core::ptr::addr_of_mut!(dma)) as *mut td;
}
if !td.is_null() {
// in case hc fetches it, make it look dead
unsafe {
(*td).hwNextTD = cpu_to_hc32(hc as *const ohci_hcd, dma as u32);
}
unsafe {
(*td).td_dma = dma;
}
// hashed in td_fill
}
td
}

fn td_free(hc: *mut ohci_hcd, td: *mut td) {
let mut prev: *mut *mut td = unsafe { (*hc).td_hash.as_mut_ptr() as *mut *mut td }.wrapping_offset(TD_HASH_FUNC!(unsafe { (*td).td_dma }) as isize);
let hcd: *mut usb_hcd = ohci_to_hcd(hc as *const ohci_hcd);
while !unsafe { *prev }.is_null() && (unsafe { *prev }) != td {
prev = unsafe { core::ptr::addr_of_mut!((*(*prev)).td_hash) };
}
if !unsafe { *prev }.is_null() {
unsafe {
*prev = (*td).td_hash;
}
} else {
if (unsafe { (*td).hwINFO }) & cpu_to_hc32(hc as *const ohci_hcd, 131072u32) != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(hc as *const ohci_hcd)).self_.controller as *const device, c"no hash for td %p\n".as_ptr() as *const u8, td) };
}
}
}
if !unsafe { (*hcd).localmem_pool }.is_null() {
gen_pool_free(unsafe { (*hcd).localmem_pool }, td as u64, core::mem::size_of::<td>() as u64);
} else {
unsafe { dma_pool_free((*hc).td_cache, td as *mut (), (*td).td_dma) };
}
}

fn ed_alloc(hc: *mut ohci_hcd, mem_flags: u32) -> *mut ed {
let mut dma: u64 = 0;
let mut ed: *mut ed = unsafe { core::mem::zeroed() };
let hcd: *mut usb_hcd = ohci_to_hcd(hc as *const ohci_hcd);
if !unsafe { (*hcd).localmem_pool }.is_null() {
ed = (unsafe { gen_pool_dma_zalloc_align((*hcd).localmem_pool, core::mem::size_of::<ed>() as u64, core::ptr::addr_of_mut!(dma), 16i32) }) as *mut ed;
} else {
ed = dma_pool_zalloc(unsafe { (*hc).ed_cache }, mem_flags, core::ptr::addr_of_mut!(dma)) as *mut ed;
}
if !ed.is_null() {
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ed).td_list) });
unsafe {
(*ed).dma = dma;
}
}
ed
}

fn ed_free(hc: *mut ohci_hcd, ed: *mut ed) {
let hcd: *mut usb_hcd = ohci_to_hcd(hc as *const ohci_hcd);
if !unsafe { (*hcd).localmem_pool }.is_null() {
gen_pool_free(unsafe { (*hcd).localmem_pool }, ed as u64, core::mem::size_of::<ed>() as u64);
} else {
unsafe { dma_pool_free((*hc).ed_cache, ed as *mut (), (*ed).dma) };
}
}

fn urb_free_priv(hc: *mut ohci_hcd, urb_priv: *mut urb_priv) {
let last: i32 = ((unsafe { (*urb_priv).length }) as i32).wrapping_sub(1);
if last >= 0 {
let mut i: i32 = 0;
while i <= last {
let td: *mut td = unsafe { *(core::ptr::addr_of_mut!((*urb_priv).td) as *mut *mut td).wrapping_offset(i as isize) };
if !td.is_null() {
td_free(hc, td);
}
i = i.wrapping_add(1);
}
}
list_del(unsafe { core::ptr::addr_of_mut!((*urb_priv).pending) });
unsafe { kfree(urb_priv as *const ()) };
}

fn finish_urb(ohci: *mut ohci_hcd, mut urb: *mut urb, mut status: i32) {
// stop periodic dma if it's not needed
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut urb_priv: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut urb_priv = core::ptr::addr_of_mut!(urb_priv) as *mut *mut urb_priv;
let mut ep: *mut usb_host_endpoint = unsafe { core::mem::zeroed() };
let mut dev: *mut device = unsafe { core::mem::zeroed() };
unsafe {
dev = (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller;
ep = (*urb).ep;
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// ASSERT (urb->hcpriv != 0);
loop {
urb_free_priv(ohci, (*urb).hcpriv as *mut urb_priv);
(*urb).hcpriv = core::ptr::null_mut::<()>();
if likely!(status == 115i32.wrapping_neg()) != 0 {
status = 0;
}
let __t0: u32 = usb_pipetype!(unsafe { (*urb).pipe });
if __t0 == 0 {
let __v9: *mut usb_hcd = ohci_to_hcd(ohci as *const ohci_hcd);
let __v8: i32 = (*__v9).self_.bandwidth_isoc_reqs;
(*__v9).self_.bandwidth_isoc_reqs = __v8.wrapping_sub(1);
if (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.bandwidth_isoc_reqs == 0 {
if quirk_amdiso(ohci) != 0 {
usb_amd_quirk_pll_enable();
}
if quirk_amdprefetch(ohci) != 0 {
sb800_prefetch(dev, 0i32);
}
}
} else {
if __t0 == 1 {
let __v6: *mut usb_hcd = ohci_to_hcd(ohci as *const ohci_hcd);
let __v5: i32 = (*__v6).self_.bandwidth_int_reqs;
(*__v6).self_.bandwidth_int_reqs = __v5.wrapping_sub(1);
}
}
// urb->complete() can reenter this HCD
usb_hcd_unlink_urb_from_ep(ohci_to_hcd(ohci as *const ohci_hcd), urb);
spin_unlock(core::ptr::addr_of_mut!((*ohci).lock));
usb_hcd_giveback_urb(ohci_to_hcd(ohci as *const ohci_hcd), urb, status);
spin_lock(core::ptr::addr_of_mut!((*ohci).lock));
if (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.bandwidth_isoc_reqs == 0 {
__v3 = (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.bandwidth_int_reqs == 0;
} else {
__v3 = false;
}
if __v3 {
let __v0: u32 = (*ohci).hc_control & 4294967283u32;
(*ohci).hc_control = __v0;
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
}
// An isochronous URB that is sumitted too late won't have any TDs
// (marked by the fact that the td_cnt value is larger than the
// actual number of TDs).  If the next URB on this endpoint is like
// that, give it back now.
if list_empty(core::ptr::addr_of_mut!((*ep).urb_list) as *const list_head) == 0 {
urb = list_first_entry__30!(core::ptr::addr_of_mut!((*ep).urb_list), urb);
*__slate_slot_10 = (*urb).hcpriv as *mut urb_priv;
if ((*(*__slate_slot_10)).td_cnt as i32) > ((*(*__slate_slot_10)).length as i32) {
status = 0;
} else {
break;
}
} else {
break;
}
}
}
}

fn balance(ohci: *mut ohci_hcd, mut interval: i32, load: i32) -> i32 {
let mut branch: i32 = 28i32.wrapping_neg();
// iso periods can be huge; iso tds specify frame numbers
if interval > 32 {
interval = 32;
}
// search for the least loaded schedule branch of that period
// that has enough bandwidth left unreserved.
for i in 0..interval {
if branch < 0 || (unsafe { *((*ohci).load.as_mut_ptr() as *mut i32).wrapping_offset(branch as isize) }) > unsafe { *((*ohci).load.as_mut_ptr() as *mut i32).wrapping_offset(i as isize) } {
let mut j: i32 = 0;
// usb 1.1 says 90% of one frame
j = i;
while j < 32 {
if unsafe { *((*ohci).load.as_mut_ptr() as *mut i32).wrapping_offset(j as isize) }.wrapping_add(load) > 900 {
break;
}
j = j.wrapping_add(interval);
}
if j >= 32 {
branch = i;
}
}
}
branch
}

fn periodic_link(ohci: *mut ohci_hcd, ed: *mut ed) {
let mut i: u32 = 0;
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"link %sed %p branch %d [%dus.], interval %d\n".as_ptr() as *const u8, if (*ed).hwINFO & cpu_to_hc32(ohci as *const ohci_hcd, 32768u32) != 0 { c"iso ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, ed, (*ed).branch as i32, (*ed).load as i32, (*ed).interval as i32) };
}
i = (unsafe { (*ed).branch }) as u32;
while i < 32 {
let mut prev: *mut *mut ed = unsafe { (*ohci).periodic.as_mut_ptr() as *mut *mut ed }.wrapping_offset(i as isize);
let mut prev_p: *mut u32 = unsafe { (*(*ohci).hcca).int_table.as_mut_ptr() as *mut u32 }.wrapping_offset(i as isize);
let mut here: *mut ed = unsafe { *prev };
// sorting each branch by period (slow before fast)
// lets us share the faster parts of the tree.
// (plus maybe: put interrupt eds before iso)
while !here.is_null() && ed != here {
if ((unsafe { (*ed).interval }) as i32) > ((unsafe { (*here).interval }) as i32) {
break;
}
prev = unsafe { core::ptr::addr_of_mut!((*here).ed_next) };
prev_p = unsafe { core::ptr::addr_of_mut!((*here).hwNextED) };
here = unsafe { *prev };
}
if ed != here {
unsafe {
(*ed).ed_next = here;
}
if !here.is_null() {
unsafe {
(*ed).hwNextED = *prev_p;
}
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
*prev = ed;
}
unsafe {
*prev_p = cpu_to_hc32(ohci as *const ohci_hcd, (*ed).dma as u32);
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
}
let __v5: *mut i32 = unsafe { (*ohci).load.as_mut_ptr() as *mut i32 }.wrapping_offset(i as isize);
let __v6: i32 = unsafe { *__v5 };
unsafe {
*__v5 = __v6.wrapping_add((*ed).load as i32);
}
i = i.wrapping_add(((unsafe { (*ed).interval }) as i32) as u32);
}
let __v11: *mut usb_hcd = ohci_to_hcd(ohci as *const ohci_hcd);
let __v12: i32 = unsafe { (*__v11).self_.bandwidth_allocated };
unsafe {
(*__v11).self_.bandwidth_allocated = __v12.wrapping_add(((*ed).load as i32) / ((*ed).interval as i32));
}
}

fn ed_schedule(ohci: *mut ohci_hcd, ed: *mut ed) -> i32 {
let mut branch: i32 = 0;
unsafe {
(*ed).ed_prev = core::ptr::null_mut::<ed>();
}
unsafe {
(*ed).ed_next = core::ptr::null_mut::<ed>();
}
unsafe {
(*ed).hwNextED = 0;
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
// we care about rm_list when setting CLE/BLE in case the HC was at
// work on some TD when CLE/BLE was turned off, and isn't quiesced
// yet.  finish_unlinks() restarts as needed, some upcoming INTR_SF.
//
// control and bulk EDs are doubly linked (ed_next, ed_prev), but
// periodic ones are singly linked (ed_next). that's because the
// periodic schedule encodes a tree like figure 3-5 in the ohci
// spec:  each qh can have several "previous" nodes, and the tree
// doesn't have unused/idle descriptors.
'__slate_break_12: {
match (unsafe { (*ed).r#type }) as i32 {
2 => {
if unsafe { (*ohci).ed_controltail }.is_null() {
let mut __v2: i64;
let __ret_warn_on: i32 = ((unsafe { (*ohci).hc_control }) & 16u32 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("641: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 641b - ., 3; .popsection\n# {0}", const 641 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ed_schedule_3, sym __slate_asm_string_ed_schedule_4, const 205 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("642: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 642b - ., 4; .popsection\n# {0}", const 642 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
ohci_writel!(ohci, unsafe { (*ed).dma }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).ed_controlhead) });
} else {
unsafe {
(*(*ohci).ed_controltail).ed_next = ed;
}
unsafe {
(*(*ohci).ed_controltail).hwNextED = cpu_to_hc32(ohci as *const ohci_hcd, (*ed).dma as u32);
}
}
unsafe {
(*ed).ed_prev = (*ohci).ed_controltail;
}
if unsafe { (*ohci).ed_controltail }.is_null() && unsafe { (*ohci).ed_rm_list }.is_null() {
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
(*ohci).hc_control = (*ohci).hc_control | 16u32;
}
ohci_writel!(ohci, 0i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).ed_controlcurrent) });
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) });
}
unsafe {
(*ohci).ed_controltail = ed;
}
}
3 => {
if unsafe { (*ohci).ed_bulktail }.is_null() {
let mut __v13: i64;
let __ret_warn_on: i32 = ((unsafe { (*ohci).hc_control }) & 32u32 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("643: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 643b - ., 3; .popsection\n# {0}", const 643 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ed_schedule_14, sym __slate_asm_string_ed_schedule_15, const 226 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("644: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 644b - ., 4; .popsection\n# {0}", const 644 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v13 = (__ret_warn_on != 0) as i64;
ohci_writel!(ohci, unsafe { (*ed).dma }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).ed_bulkhead) });
} else {
unsafe {
(*(*ohci).ed_bulktail).ed_next = ed;
}
unsafe {
(*(*ohci).ed_bulktail).hwNextED = cpu_to_hc32(ohci as *const ohci_hcd, (*ed).dma as u32);
}
}
unsafe {
(*ed).ed_prev = (*ohci).ed_bulktail;
}
if unsafe { (*ohci).ed_bulktail }.is_null() && unsafe { (*ohci).ed_rm_list }.is_null() {
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
(*ohci).hc_control = (*ohci).hc_control | 32u32;
}
ohci_writel!(ohci, 0i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).ed_bulkcurrent) });
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) });
}
unsafe {
(*ohci).ed_bulktail = ed;
}
break '__slate_break_12;
// case PIPE_INTERRUPT:
//
// case PIPE_ISOCHRONOUS:
}
_ => {
branch = balance(ohci, (unsafe { (*ed).interval }) as i32, (unsafe { (*ed).load }) as i32);
if branch < 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"ERR %d, interval %d msecs, load %d\n".as_ptr() as *const u8, branch, (*ed).interval as i32, (*ed).load as i32) };
}
// FIXME if there are TDs queued, fail them!
return branch;
}
unsafe {
(*ed).branch = branch as u8;
}
periodic_link(ohci, ed);
}
}
}
// the HC may not see the schedule updates yet, but if it does
// then they'll be properly ordered.
unsafe {
(*ed).state = 2;
}
0
}

fn periodic_unlink(ohci: *mut ohci_hcd, ed: *mut ed) {
let mut i: i32 = (unsafe { (*ed).branch }) as i32;
while i < 32 {
let mut temp: *mut ed = unsafe { core::mem::zeroed() };
let mut prev: *mut *mut ed = unsafe { (*ohci).periodic.as_mut_ptr() as *mut *mut ed }.wrapping_offset(i as isize);
let mut prev_p: *mut u32 = unsafe { (*(*ohci).hcca).int_table.as_mut_ptr() as *mut u32 }.wrapping_offset(i as isize);
loop {
let __v0: bool = if !unsafe { *prev }.is_null() {
let __v1: *mut ed = unsafe { *prev };
temp = __v1;
__v1 != ed
} else {
false
};
if !__v0 {
break;
}
prev_p = unsafe { core::ptr::addr_of_mut!((*temp).hwNextED) };
prev = unsafe { core::ptr::addr_of_mut!((*temp).ed_next) };
}
if !unsafe { *prev }.is_null() {
unsafe {
*prev_p = (*ed).hwNextED;
}
unsafe {
*prev = (*ed).ed_next;
}
}
let __v3: *mut i32 = unsafe { (*ohci).load.as_mut_ptr() as *mut i32 }.wrapping_offset(i as isize);
let __v4: i32 = unsafe { *__v3 };
unsafe {
*__v3 = __v4.wrapping_sub((*ed).load as i32);
}
i = i.wrapping_add((unsafe { (*ed).interval }) as i32);
}
let __v9: *mut usb_hcd = ohci_to_hcd(ohci as *const ohci_hcd);
let __v10: i32 = unsafe { (*__v9).self_.bandwidth_allocated };
unsafe {
(*__v9).self_.bandwidth_allocated = __v10.wrapping_sub(((*ed).load as i32) / ((*ed).interval as i32));
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"unlink %sed %p branch %d [%dus.], interval %d\n".as_ptr() as *const u8, if (*ed).hwINFO & cpu_to_hc32(ohci as *const ohci_hcd, 32768u32) != 0 { c"iso ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, ed, (*ed).branch as i32, (*ed).load as i32, (*ed).interval as i32) };
}
}

fn ed_deschedule(ohci: *mut ohci_hcd, ed: *mut ed) {
let __v0: u32 = cpu_to_hc32(ohci as *const ohci_hcd, 16384u32);
unsafe {
(*ed).hwINFO = (*ed).hwINFO | __v0;
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
(*ed).state = 1;
}
// To deschedule something from the control or bulk list, just
// clear CLE/BLE and wait.  There's no safe way to scrub out list
// head/current registers until later, and "later" isn't very
// tightly specified.  Figure 6-5 and Section 6.4.2.2 show how
// the HC is reading the ED queues (while we modify them).
//
// For now, ed_schedule() is "later".  It might be good paranoia
// to scrub those registers in finish_unlinks(), in case of bugs
// that make the HC try to use them.
'__slate_break_9: {
match (unsafe { (*ed).r#type }) as i32 {
2 => {
if unsafe { (*ed).ed_prev }.is_null() {
if (unsafe { (*ed).hwNextED }) == 0 {
let __v8: u32 = (unsafe { (*ohci).hc_control }) & 4294967279u32;
unsafe {
(*ohci).hc_control = __v8;
}
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) });
// a ohci_readl() later syncs CLE with the HC
} else {
ohci_writel__2!(ohci, hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*ed).hwNextED) }) as *const u32), unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).ed_controlhead) });
}
} else {
unsafe {
(*(*ed).ed_prev).ed_next = (*ed).ed_next;
}
unsafe {
(*(*ed).ed_prev).hwNextED = (*ed).hwNextED;
}
}
// remove ED from the HC's list:
// remove ED from the HCD's list:
if (unsafe { (*ohci).ed_controltail }) == ed {
unsafe {
(*ohci).ed_controltail = (*ed).ed_prev;
}
if !unsafe { (*ohci).ed_controltail }.is_null() {
unsafe {
(*(*ohci).ed_controltail).ed_next = core::ptr::null_mut::<ed>();
}
}
} else {
if !unsafe { (*ed).ed_next }.is_null() {
unsafe {
(*(*ed).ed_next).ed_prev = (*ed).ed_prev;
}
}
}
}
3 => {
if unsafe { (*ed).ed_prev }.is_null() {
if (unsafe { (*ed).hwNextED }) == 0 {
let __v12: u32 = (unsafe { (*ohci).hc_control }) & 4294967263u32;
unsafe {
(*ohci).hc_control = __v12;
}
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) });
// a ohci_readl() later syncs BLE with the HC
} else {
ohci_writel__2!(ohci, hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*ed).hwNextED) }) as *const u32), unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).ed_bulkhead) });
}
} else {
unsafe {
(*(*ed).ed_prev).ed_next = (*ed).ed_next;
}
unsafe {
(*(*ed).ed_prev).hwNextED = (*ed).hwNextED;
}
}
// remove ED from the HC's list:
// remove ED from the HCD's list:
if (unsafe { (*ohci).ed_bulktail }) == ed {
unsafe {
(*ohci).ed_bulktail = (*ed).ed_prev;
}
if !unsafe { (*ohci).ed_bulktail }.is_null() {
unsafe {
(*(*ohci).ed_bulktail).ed_next = core::ptr::null_mut::<ed>();
}
}
} else {
if !unsafe { (*ed).ed_next }.is_null() {
unsafe {
(*(*ed).ed_next).ed_prev = (*ed).ed_prev;
}
}
}
break '__slate_break_9;
// case PIPE_INTERRUPT:
//
// case PIPE_ISOCHRONOUS:
}
_ => {
periodic_unlink(ohci, ed);
}
}
}
}

fn ed_get(ohci: *mut ohci_hcd, ep: *mut usb_host_endpoint, udev: *mut usb_device, pipe: u32, mut interval: i32) -> *mut ed {
let mut info: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(info) as *mut u32;
let mut is_out: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(is_out) as *mut i32;
let mut td: core::mem::MaybeUninit<*mut td> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut td = core::ptr::addr_of_mut!(td) as *mut *mut td;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut ed: core::mem::MaybeUninit<*mut ed> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut ed = core::ptr::addr_of_mut!(ed) as *mut *mut ed;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_14);
*__slate_slot_15 = (*ep).hcpriv as *mut ed;
if (*__slate_slot_15).is_null() {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_15 = ed_alloc(ohci, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
if (*__slate_slot_15).is_null() {
// out of memory
} else {
// dummy td; end of td list for ed
*__slate_slot_13 = td_alloc(ohci, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
if (*__slate_slot_13).is_null() {
// out of memory
ed_free(ohci, *__slate_slot_15);
*__slate_slot_15 = core::ptr::null_mut::<ed>();
} else {
(*(*__slate_slot_15)).dummy = *__slate_slot_13;
(*(*__slate_slot_15)).hwTailP = cpu_to_hc32(ohci as *const ohci_hcd, (*(*__slate_slot_13)).td_dma as u32);
(*(*__slate_slot_15)).hwHeadP = (*(*__slate_slot_15)).hwTailP;
// ED_C, ED_H zeroed
(*(*__slate_slot_15)).state = 0;
*__slate_slot_12 = (((*ep).desc.bEndpointAddress as i32) & 128 == 0) as i32;
// FIXME usbcore changes dev->devnum before SET_ADDRESS
// succeeds ... otherwise we wouldn't need "pipe".
*__slate_slot_11 = usb_pipedevice!(pipe);
(*(*__slate_slot_15)).r#type = usb_pipetype!(pipe) as u8;
let __v9: u32 = *__slate_slot_11 | (((((*ep).desc.bEndpointAddress as i32) & !128) << 7i32) as u32);
*__slate_slot_11 = __v9;
let __v8: i32 = usb_endpoint_maxp(core::ptr::addr_of_mut!((*ep).desc) as *const usb_endpoint_descriptor) << 16i32;
*__slate_slot_11 |= __v8 as u32;
if ((*udev).speed as u32) == 1 {
*__slate_slot_11 |= 8192u32;
}
// only control transfers store pids in tds
if ((*(*__slate_slot_15)).r#type as i32) != 2 {
*__slate_slot_11 |= (if *__slate_slot_12 != 0 { 2048i32 } else { 4096i32 }) as u32;
if ((*(*__slate_slot_15)).r#type as i32) != 3 {
// periodic transfers...
if ((*(*__slate_slot_15)).r#type as i32) == 0 {
*__slate_slot_11 |= 32768u32;
} else {
if interval > 32 {
interval = 32;
}
}
// iso can be bigger
(*(*__slate_slot_15)).interval = interval as u16;
(*(*__slate_slot_15)).load = (usb_calc_bus_time(((*udev).speed as u32) as i32, (*__slate_slot_12 == 0) as i32, (((*(*__slate_slot_15)).r#type as i32) == 0) as i32, usb_endpoint_maxp(core::ptr::addr_of_mut!((*ep).desc) as *const usb_endpoint_descriptor)) / 1000) as u16;
}
}
(*(*__slate_slot_15)).hwINFO = cpu_to_hc32(ohci as *const ohci_hcd, *__slate_slot_11);
(*ep).hcpriv = *__slate_slot_15 as *mut ();
}
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_14);
return *__slate_slot_15;
}
}

fn start_ed_unlink(ohci: *mut ohci_hcd, ed: *mut ed) {
let __v0: u32 = cpu_to_hc32(ohci as *const ohci_hcd, 134217728u32);
unsafe {
(*ed).hwINFO = (*ed).hwINFO | __v0;
}
ed_deschedule(ohci, ed);
// rm_list is just singly linked, for simplicity
unsafe {
(*ed).ed_next = (*ohci).ed_rm_list;
}
unsafe {
(*ed).ed_prev = core::ptr::null_mut::<ed>();
}
unsafe {
(*ohci).ed_rm_list = ed;
}
// enable SOF interrupt
ohci_writel!(ohci, 4i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrstatus) });
ohci_writel!(ohci, 4i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrenable) });
// flush those writes, and get latest HCCA contents
ohci_readl!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) });
// SF interrupt might get delayed; record the frame counter value that
// indicates when the HC isn't looking at it, so concurrent unlinks
// behave.  frame_no wraps every 2^16 msec, and changes right before
// SF is triggered.
unsafe {
(*ed).tick = (ohci_frame_no(ohci as *const ohci_hcd) as i32).wrapping_add(1) as u16;
}
}

fn td_fill(ohci: *mut ohci_hcd, mut info: u32, mut data: u64, len: i32, urb: *mut urb, index: i32) {
let mut td: *mut td = unsafe { core::mem::zeroed() };
let urb_priv: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
let is_iso: i32 = (info & 65536) as i32;
let mut hash: i32 = 0;
// ASSERT (index < urb_priv->length);
// aim for only one interrupt per urb.  mostly applies to control
// and iso; other urbs rarely need more than one TD per urb.
// this way, only final tds (or ones with an error) cause IRQs.
// at least immediately; use DI=6 in case any control request is
// tempted to die part way through.  (and to force the hc to flush
// its donelist soonish, even on unlink paths.)
//
// NOTE: could delay interrupts even for the last TD, and get fewer
// interrupts ... increasing per-urb latency by sharing interrupts.
// Drivers that queue bulk urbs may request that behavior.
if index != ((unsafe { (*urb_priv).length }) as i32).wrapping_sub(1) || (unsafe { (*urb).transfer_flags }) & 128 != 0 {
let __v0: u32 = info;
info = __v0 | (TD_DI_SET!(6i32) as u32);
}
// use this td as the next dummy
let td_pt: *mut td = unsafe { *(core::ptr::addr_of_mut!((*urb_priv).td) as *mut *mut td).wrapping_offset(index as isize) };
// fill the old dummy TD
let __v2: *mut td = unsafe { (*(*urb_priv).ed).dummy };
unsafe {
*(core::ptr::addr_of_mut!((*urb_priv).td) as *mut *mut td).wrapping_offset(index as isize) = __v2;
}
td = __v2;
unsafe {
(*(*urb_priv).ed).dummy = td_pt;
}
unsafe {
(*td).ed = (*urb_priv).ed;
}
unsafe {
(*td).next_dl_td = core::ptr::null_mut::<td>();
}
unsafe {
(*td).index = index as u8;
}
unsafe {
(*td).urb = urb;
}
unsafe {
(*td).data_dma = data;
}
if len == 0 {
data = 0;
}
unsafe {
(*td).hwINFO = cpu_to_hc32(ohci as *const ohci_hcd, info);
}
if is_iso != 0 {
unsafe {
(*td).hwCBP = cpu_to_hc32(ohci as *const ohci_hcd, (data & 4294963200) as u32);
}
unsafe {
*ohci_hwPSWp(ohci as *const ohci_hcd, td as *const td, 0i32) = cpu_to_hc16(ohci as *const ohci_hcd, (data & 4095 | 57344) as u16);
}
} else {
unsafe {
(*td).hwCBP = cpu_to_hc32(ohci as *const ohci_hcd, data as u32);
}
}
if data != 0 {
unsafe {
(*td).hwBE = cpu_to_hc32(ohci as *const ohci_hcd, data.wrapping_add(len as u64).wrapping_sub(1) as u32);
}
} else {
unsafe {
(*td).hwBE = 0;
}
}
unsafe {
(*td).hwNextTD = cpu_to_hc32(ohci as *const ohci_hcd, (*td_pt).td_dma as u32);
}
// append to queue
list_add_tail(unsafe { core::ptr::addr_of_mut!((*td).td_list) }, unsafe { core::ptr::addr_of_mut!((*(*td).ed).td_list) });
// hash it for later reverse mapping
hash = TD_HASH_FUNC!(unsafe { (*td).td_dma }) as i32;
unsafe {
(*td).td_hash = *((*ohci).td_hash.as_mut_ptr() as *mut *mut td).wrapping_offset(hash as isize);
}
unsafe {
*((*ohci).td_hash.as_mut_ptr() as *mut *mut td).wrapping_offset(hash as isize) = td;
}
// HC might read the TD (or cachelines) right away ...
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
(*(*td).ed).hwTailP = (*td).hwNextTD;
}
}

fn td_submit_urb(ohci: *mut ohci_hcd, urb: *mut urb) {
let urb_priv: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
let dev: *mut device = unsafe { (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller };
let mut data: u64 = 0;
let mut data_len: i32 = (unsafe { (*urb).transfer_buffer_length }) as i32;
let mut cnt: i32 = 0;
let mut info: u32 = 0;
let is_out: i32 = usb_pipeout!(unsafe { (*urb).pipe }) as i32;
let mut periodic: i32 = 0;
let mut i: i32 = 0;
let mut this_sg_len: i32 = 0;
let mut n: i32 = 0;
let mut sg: *mut scatterlist = unsafe { core::mem::zeroed() };
// OHCI handles the bulk/interrupt data toggles itself.  We just
// use the device toggle bits for resetting, and rely on the fact
// that resetting toggle is meaningless if the endpoint is active.
if usb_gettoggle!(unsafe { (*urb).dev }, usb_pipeendpoint!(unsafe { (*urb).pipe }), is_out) == 0 {
usb_settoggle!(unsafe { (*urb).dev }, (unsafe { (*urb).pipe }) >> 15i32 & 15, is_out, 1i32);
let __v0: u32 = !cpu_to_hc32(ohci as *const ohci_hcd, 2u32);
let __v1: *mut ed = unsafe { (*urb_priv).ed };
unsafe {
(*__v1).hwHeadP = (*__v1).hwHeadP & __v0;
}
}
list_add(unsafe { core::ptr::addr_of_mut!((*urb_priv).pending) }, unsafe { core::ptr::addr_of_mut!((*ohci).pending) });
i = unsafe { (*urb).num_mapped_sgs };
if data_len > 0 && i > 0 {
sg = unsafe { (*urb).sg };
data = sg_dma_address!(sg);
// urb->transfer_buffer_length may be smaller than the
// size of the scatterlist (or vice versa)
this_sg_len = min_t!(i32, sg_dma_len!(sg), data_len);
} else {
sg = core::ptr::null_mut::<scatterlist>();
if data_len != 0 {
data = unsafe { (*urb).transfer_dma };
} else {
data = 0;
}
this_sg_len = data_len;
}
// NOTE:  TD_CC is set so we can tell which TDs the HC processed by
// using TD_CC_GET, as well as by seeing them on the done list.
// (CC = NotAccessed ... 0x0F, or 0x0E in PSWs for ISO.)
// Bulk and interrupt are identical except for where in the schedule
// their EDs live.
// ... and periodic urbs have extra accounting
// TDs _could_ transfer up to 8K each
// maybe avoid ED halt on final TD short read
// maybe kickstart bulk list
// control manages DATA0/DATA1 toggle per-request; SETUP resets it,
// any DATA phase works normally, and the STATUS ack is special.
// NOTE:  mishandles transfers >8K, some >4K
// maybe kickstart control list
// ISO has no retransmit, so no toggle; and it uses special TDs.
// Each TD could handle multiple consecutive frames (interval 1);
// we could often reduce the number of TDs here.
// FIXME scheduling should handle frame counter
//
// roll-around ... exotic case (and OHCI has
//
// a 2^16 iso range, vs other HCs max of 2^10)
'__slate_break_7: {
match (unsafe { (*(*urb_priv).ed).r#type }) as i32 {
1 => {
// Bulk and interrupt are identical except for where in the schedule
// their EDs live.
let __v8: *mut usb_hcd = ohci_to_hcd(ohci as *const ohci_hcd);
let __v9: i32 = unsafe { (*__v8).self_.bandwidth_int_reqs };
unsafe {
(*__v8).self_.bandwidth_int_reqs = __v9.wrapping_add(1);
}
let __v11: bool = __v9 == 0 && (unsafe { (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.bandwidth_isoc_reqs }) == 0;
periodic = __v11 as i32; // ... and periodic urbs have extra accounting
info = if is_out != 0 { 0 | 4026531840 | 524288 } else { 0 | 4026531840 | 1048576 };
// TDs _could_ transfer up to 8K each
loop {
let mut _v60: i32;
let mut __UNIQUE_ID_x__647: i32 = this_sg_len;
let mut __UNIQUE_ID_y__648: i32 = 4096;
loop {
if __types_ok__2!(__UNIQUE_ID_x__647, __UNIQUE_ID_y__648) == 0 {
unsafe { __compiletime_assert_649() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
_v60 = if __UNIQUE_ID_x__647 < __UNIQUE_ID_y__648 { __UNIQUE_ID_x__647 } else { __UNIQUE_ID_y__648 };
n = _v60;
// maybe avoid ED halt on final TD short read
if n >= data_len || i == 1 && n >= this_sg_len {
if (unsafe { (*urb).transfer_flags }) & 1 == 0 {
let _v61: u32 = info;
let _v62: u32 = _v61 | 262144;
info = _v62;
}
}
td_fill(ohci, info, data, n, urb, cnt);
let _v63: i32 = this_sg_len;
let _v64: i32 = _v63.wrapping_sub(n);
this_sg_len = _v64;
let _v65: i32 = data_len;
let _v66: i32 = _v65.wrapping_sub(n);
data_len = _v66;
let _v67: u64 = data;
let _v68: u64 = _v67.wrapping_add(n as u64);
data = _v68;
let _v69: i32 = cnt;
let _v70: i32 = _v69.wrapping_add(1);
cnt = _v70;
if this_sg_len <= 0 {
let _v71: i32 = i;
let _v72: i32 = _v71.wrapping_sub(1);
i = _v72;
if _v72 <= 0 || data_len <= 0 {
break;
}
sg = sg_next(sg);
data = sg_dma_address!(sg);
this_sg_len = min_t!(i32, sg_dma_len!(sg), data_len);
}
}
if (unsafe { (*urb).transfer_flags }) & 64 != 0 && cnt < ((unsafe { (*urb_priv).length }) as i32) {
td_fill(ohci, info, 0u64, 0i32, urb, cnt);
let _v73: i32 = cnt;
let _v74: i32 = _v73.wrapping_add(1);
cnt = _v74;
}
// maybe kickstart bulk list
if ((unsafe { (*(*urb_priv).ed).r#type }) as i32) == 3 {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
ohci_writel!(ohci, 4i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus) });
}
break '__slate_break_7;
// control manages DATA0/DATA1 toggle per-request; SETUP resets it,
// any DATA phase works normally, and the STATUS ack is special.
}
3 => {
info = if is_out != 0 { 0 | 4026531840 | 524288 } else { 0 | 4026531840 | 1048576 };
// TDs _could_ transfer up to 8K each
loop {
let mut __v12: i32;
let __UNIQUE_ID_x__647: i32 = this_sg_len;
let __UNIQUE_ID_y__648: i32 = 4096;
loop {
if __types_ok__2!(__UNIQUE_ID_x__647, __UNIQUE_ID_y__648) == 0 {
unsafe { __compiletime_assert_649() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = if __UNIQUE_ID_x__647 < __UNIQUE_ID_y__648 { __UNIQUE_ID_x__647 } else { __UNIQUE_ID_y__648 };
n = __v12;
// maybe avoid ED halt on final TD short read
if n >= data_len || i == 1 && n >= this_sg_len {
if (unsafe { (*urb).transfer_flags }) & 1 == 0 {
info |= 262144;
}
}
td_fill(ohci, info, data, n, urb, cnt);
this_sg_len = this_sg_len.wrapping_sub(n);
data_len = data_len.wrapping_sub(n);
data = data.wrapping_add(n as u64);
cnt = cnt.wrapping_add(1);
if this_sg_len <= 0 {
let __v25: i32 = i.wrapping_sub(1);
i = __v25;
if __v25 <= 0 || data_len <= 0 {
break;
}
sg = sg_next(sg);
data = sg_dma_address!(sg);
this_sg_len = min_t!(i32, sg_dma_len!(sg), data_len);
}
}
if (unsafe { (*urb).transfer_flags }) & 64 != 0 && cnt < ((unsafe { (*urb_priv).length }) as i32) {
td_fill(ohci, info, 0u64, 0i32, urb, cnt);
cnt = cnt.wrapping_add(1);
}
// maybe kickstart bulk list
if ((unsafe { (*(*urb_priv).ed).r#type }) as i32) == 3 {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
ohci_writel!(ohci, 4i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus) });
}
break '__slate_break_7;
// control manages DATA0/DATA1 toggle per-request; SETUP resets it,
// any DATA phase works normally, and the STATUS ack is special.
}
2 => {
info = 4026531840 | 0 | 33554432;
let __v34: i32 = cnt;
cnt = __v34.wrapping_add(1);
td_fill(ohci, info, unsafe { (*urb).setup_dma }, 8i32, urb, __v34);
if data_len > 0 {
info = 4026531840 | 262144 | 50331648;
info |= (if is_out != 0 { 524288i32 } else { 1048576i32 }) as u32;
// NOTE:  mishandles transfers >8K, some >4K
let __v38: i32 = cnt;
cnt = __v38.wrapping_add(1);
td_fill(ohci, info, data, data_len, urb, __v38);
}
info = if is_out != 0 || data_len == 0 { 4026531840 | 1048576 | 50331648 } else { 4026531840 | 524288 | 50331648 };
let __v40: i32 = cnt;
cnt = __v40.wrapping_add(1);
td_fill(ohci, info, data, 0i32, urb, __v40);
// maybe kickstart control list
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
ohci_writel!(ohci, 2i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus) });
break '__slate_break_7;
// ISO has no retransmit, so no toggle; and it uses special TDs.
// Each TD could handle multiple consecutive frames (interval 1);
// we could often reduce the number of TDs here.
}
0 => {
cnt = (unsafe { (*urb_priv).td_cnt }) as i32;
while cnt < unsafe { (*urb).number_of_packets } {
let mut frame: i32 = unsafe { (*urb).start_frame };
// FIXME scheduling should handle frame counter
//
// roll-around ... exotic case (and OHCI has
//
// a 2^16 iso range, vs other HCs max of 2^10)
frame = frame.wrapping_add(cnt.wrapping_mul(unsafe { (*urb).interval }));
frame &= 65535;
td_fill(ohci, 4026531840 | 65536 | (frame as u32), data.wrapping_add((unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(cnt as isize)).offset }) as u64), (unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(cnt as isize)).length }) as i32, urb, cnt);
cnt = cnt.wrapping_add(1);
}
if (unsafe { (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.bandwidth_isoc_reqs }) == 0 {
if quirk_amdiso(ohci) != 0 {
unsafe { usb_amd_quirk_pll_disable() };
}
if quirk_amdprefetch(ohci) != 0 {
unsafe { sb800_prefetch(dev, 1i32) };
}
}
let __v51: *mut usb_hcd = ohci_to_hcd(ohci as *const ohci_hcd);
let __v52: i32 = unsafe { (*__v51).self_.bandwidth_isoc_reqs };
unsafe {
(*__v51).self_.bandwidth_isoc_reqs = __v52.wrapping_add(1);
}
let __v54: bool = __v52 == 0 && (unsafe { (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.bandwidth_int_reqs }) == 0;
periodic = __v54 as i32;
}
_ => {
}
}
}
// start periodic dma if needed
if periodic != 0 {
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
(*ohci).hc_control = (*ohci).hc_control | 12u32;
}
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) });
}
// ASSERT (urb_priv->length == cnt);
}

fn td_done(ohci: *mut ohci_hcd, urb: *mut urb, td: *mut td) -> i32 {
let tdINFO: u32 = hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*td).hwINFO) }) as *const u32);
let mut cc: i32 = 0;
let mut status: i32 = 115i32.wrapping_neg();
list_del(unsafe { core::ptr::addr_of_mut!((*td).td_list) });
// ISO ... drivers see per-TD length/status
if tdINFO & 65536 != 0 {
let tdPSW: u16 = ohci_hwPSW(ohci as *const ohci_hcd, td as *const td, 0i32);
let mut dlen: i32 = 0;
// NOTE:  assumes FC in tdINFO == 0, and that
// only the first of 0..MAXPSW psws is used.
cc = (tdPSW as i32) >> 12i32 & 15;
if tdINFO & 4026531840 != 0 {
return status;
}
// hc didn't touch?
if usb_pipeout!(unsafe { (*urb).pipe }) {
dlen = (unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(((*td).index as i32) as isize)).length }) as i32;
} else {
// short reads are always OK for ISO
if cc == 9 {
cc = 0;
}
dlen = (tdPSW as i32) & 1023;
}
let __v1: u32 = unsafe { (*urb).actual_length };
unsafe {
(*urb).actual_length = __v1.wrapping_add(dlen as u32);
}
unsafe {
(*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(((*td).index as i32) as isize)).actual_length = dlen as u32;
}
unsafe {
(*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(((*td).index as i32) as isize)).status = cc_to_error.0[cc as usize];
}
if cc != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"urb %p iso td %p (%d) len %d cc %d\n".as_ptr() as *const u8, urb, td, 1i32.wrapping_add((*td).index as i32), dlen, cc) };
}
}
// BULK, INT, CONTROL ... drivers see aggregate length/status,
// except that "setup" bytes aren't counted and "short" transfers
// might not be reported as errors.
} else {
let r#type: i32 = usb_pipetype!(unsafe { (*urb).pipe }) as i32;
let tdBE: u32 = hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*td).hwBE) }) as *const u32);
cc = TD_CC_GET!(tdINFO) as i32;
// update packet status if needed (short is normally ok)
if cc == 9 && (unsafe { (*urb).transfer_flags }) & 1 == 0 {
cc = 0;
}
if cc != 0 && cc < 14 {
status = unsafe { cc_to_error.0[cc as usize] };
}
// count all non-empty packets except control SETUP packet
if (r#type != 2 || ((unsafe { (*td).index }) as i32) != 0) && tdBE != 0 {
if (unsafe { (*td).hwCBP }) == 0 {
unsafe {
(*urb).actual_length = ((*urb).actual_length as u64).wrapping_add((tdBE as u64).wrapping_sub((*td).data_dma).wrapping_add(1)) as u32;
}
} else {
let __v6: u64 = (hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*td).hwCBP) }) as *const u32) as u64).wrapping_sub(unsafe { (*td).data_dma });
unsafe {
(*urb).actual_length = ((*urb).actual_length as u64).wrapping_add(__v6) as u32;
}
}
}
if cc != 0 && cc < 14 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"urb %p td %p (%d) cc %d, len=%d/%d\n".as_ptr() as *const u8, urb, td, 1i32.wrapping_add((*td).index as i32), cc, (*urb).actual_length, (*urb).transfer_buffer_length) };
}
}
}
status
}

fn ed_halted(ohci: *mut ohci_hcd, td: *mut td, cc: i32) {
let urb: *mut urb = unsafe { (*td).urb };
let urb_priv: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
let ed: *mut ed = unsafe { (*td).ed };
let mut tmp: *mut list_head = unsafe { (*td).td_list.next };
let toggle: u32 = (unsafe { (*ed).hwHeadP }) & cpu_to_hc32(ohci as *const ohci_hcd, 2u32);
// clear ed halt; this is the td that caused it, but keep it inactive
// until its urb->complete() has a chance to clean up.
let __v0: u32 = cpu_to_hc32(ohci as *const ohci_hcd, 16384u32);
unsafe {
(*ed).hwINFO = (*ed).hwINFO | __v0;
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
let __v6: u32 = !cpu_to_hc32(ohci as *const ohci_hcd, 1u32);
unsafe {
(*ed).hwHeadP = (*ed).hwHeadP & __v6;
}
// Get rid of all later tds from this urb.  We don't have
// to be careful: no errors and nothing was transferred.
// Also patch the ed so it looks as if those tds completed normally.
while tmp != unsafe { core::ptr::addr_of_mut!((*ed).td_list) } {
let mut next: *mut td = unsafe { core::mem::zeroed() };
next = list_entry__3!(tmp, td);
tmp = unsafe { (*next).td_list.next };
if (unsafe { (*next).urb }) != urb {
break;
}
// NOTE: if multi-td control DATA segments get supported,
// this urb had one of them, this td wasn't the last td
// in that segment (TD_R clear), this ed halted because
// of a short read, _and_ URB_SHORT_NOT_OK is clear ...
// then we need to leave the control STATUS packet queued
// and clear ED_SKIP.
list_del(unsafe { core::ptr::addr_of_mut!((*next).td_list) });
unsafe {
(*urb_priv).td_cnt = ((*urb_priv).td_cnt as i32).wrapping_add(1) as u16;
}
unsafe {
(*ed).hwHeadP = (*next).hwNextTD | toggle;
}
}
// help for troubleshooting:  report anything that
// looks odd ... that doesn't include protocol stalls
// (or maybe some other things)
match cc {
9 => {
if (unsafe { (*urb).transfer_flags }) & 1 != 0 {
if !usb_pipecontrol!(unsafe { (*urb).pipe }) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"urb %p path %s ep%d%s %08x cc %d --> status %d\n".as_ptr() as *const u8, urb, (*(*urb).dev).devpath.as_mut_ptr() as *mut u8, usb_pipeendpoint!(unsafe { (*urb).pipe }), if usb_pipein!(unsafe { (*urb).pipe }) != 0 { c"in".as_ptr() as *mut u8 } else { c"out".as_ptr() as *mut u8 }, hc32_to_cpu(ohci as *const ohci_hcd, (*td).hwINFO), cc, cc_to_error.0[cc as usize]) };
}
}
}
}
4 => {
if !usb_pipecontrol!(unsafe { (*urb).pipe }) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"urb %p path %s ep%d%s %08x cc %d --> status %d\n".as_ptr() as *const u8, urb, (*(*urb).dev).devpath.as_mut_ptr() as *mut u8, usb_pipeendpoint!(unsafe { (*urb).pipe }), if usb_pipein!(unsafe { (*urb).pipe }) != 0 { c"in".as_ptr() as *mut u8 } else { c"out".as_ptr() as *mut u8 }, hc32_to_cpu(ohci as *const ohci_hcd, (*td).hwINFO), cc, cc_to_error.0[cc as usize]) };
}
}
}
_ => {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"urb %p path %s ep%d%s %08x cc %d --> status %d\n".as_ptr() as *const u8, urb, (*(*urb).dev).devpath.as_mut_ptr() as *mut u8, usb_pipeendpoint!(unsafe { (*urb).pipe }), if usb_pipein!(unsafe { (*urb).pipe }) != 0 { c"in".as_ptr() as *mut u8 } else { c"out".as_ptr() as *mut u8 }, hc32_to_cpu(ohci as *const ohci_hcd, (*td).hwINFO), cc, cc_to_error.0[cc as usize]) };
}
}
}
}

fn add_to_done_list(ohci: *mut ohci_hcd, td: *mut td) {
let mut td2: *mut td = unsafe { core::mem::zeroed() };
let mut td_prev: *mut td = unsafe { core::mem::zeroed() };
let mut ed: *mut ed = unsafe { core::mem::zeroed() };
if !unsafe { (*td).next_dl_td }.is_null() {
return;
}
// Already on the list
// Add all the TDs going back until we reach one that's on the list
ed = unsafe { (*td).ed };
let __v0: *mut td = td;
td_prev = __v0;
td2 = __v0;
td2 = list_prev_entry__2!(td2, td_list);
while list_entry_is_head!(td2, unsafe { core::ptr::addr_of_mut!((*ed).td_list) }, td_list) == 0 {
if !unsafe { (*td2).next_dl_td }.is_null() {
break;
}
unsafe {
(*td2).next_dl_td = td_prev;
}
td_prev = td2;
td2 = list_prev_entry__2!(td2, td_list);
}
if !unsafe { (*ohci).dl_end }.is_null() {
unsafe {
(*(*ohci).dl_end).next_dl_td = td_prev;
}
} else {
unsafe {
(*ohci).dl_start = td_prev;
}
}
// Make td->next_dl_td point to td itself, to mark the fact
// that td is on the done list.
let __v4: *mut td = td;
unsafe {
(*td).next_dl_td = __v4;
}
unsafe {
(*ohci).dl_end = __v4;
}
// Did we just add the latest pending TD?
td2 = unsafe { (*ed).pending_td };
if !td2.is_null() && !unsafe { (*td2).next_dl_td }.is_null() {
unsafe {
(*ed).pending_td = core::ptr::null_mut::<td>();
}
}
}

fn update_done_list(ohci: *mut ohci_hcd) {
let mut td_dma: u32 = 0;
let mut td: *mut td = core::ptr::null_mut::<td>();
td_dma = hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*(*ohci).hcca).done_head) }) as *const u32);
unsafe {
(*(*ohci).hcca).done_head = 0;
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
// get TD from hc's singly linked list, and
// add to ours.  ed->td_list changes later.
while td_dma != 0 {
let mut cc: i32 = 0;
td = dma_to_td(ohci, td_dma as u64);
if td.is_null() {
while false {
}
unsafe { _dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"bad entry %8x\n".as_ptr() as *const u8, td_dma) };
break;
}
let __v4: u32 = cpu_to_hc32(ohci as *const ohci_hcd, 131072u32);
unsafe {
(*td).hwINFO = (*td).hwINFO | __v4;
}
cc = TD_CC_GET!(hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*td).hwINFO) }) as *const u32)) as i32;
// Non-iso endpoints can halt on error; un-halt,
// and dequeue any other TDs from this urb.
// No other TD could have caused the halt.
let __v8: bool = cc != 0 && (unsafe { (*(*td).ed).hwHeadP }) & cpu_to_hc32(ohci as *const ohci_hcd, 1u32) != 0;
if __v8 {
ed_halted(ohci, td, cc);
}
td_dma = hc32_to_cpup(ohci as *const ohci_hcd, (unsafe { core::ptr::addr_of_mut!((*td).hwNextTD) }) as *const u32);
add_to_done_list(ohci, td);
}
}

fn finish_unlinks(ohci: *mut ohci_hcd) {
let mut control: u32 = unsafe { core::mem::zeroed() };
let mut command: u32 = unsafe { core::mem::zeroed() };
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut tdINFO: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(tdINFO) as *mut u32;
let mut savebits: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(savebits) as *mut u32;
let mut urb_priv: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut urb_priv = core::ptr::addr_of_mut!(urb_priv) as *mut *mut urb_priv;
let mut urb: core::mem::MaybeUninit<*mut urb> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut urb = core::ptr::addr_of_mut!(urb) as *mut *mut urb;
let mut td: core::mem::MaybeUninit<*mut td> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut td = core::ptr::addr_of_mut!(td) as *mut *mut td;
let mut head: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut u32 = core::ptr::addr_of_mut!(head) as *mut u32;
let mut td: core::mem::MaybeUninit<*mut td> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut *mut td = core::ptr::addr_of_mut!(td) as *mut *mut td;
let mut prev: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut *mut u32 = core::ptr::addr_of_mut!(prev) as *mut *mut u32;
let mut modified: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut i32 = core::ptr::addr_of_mut!(modified) as *mut i32;
let mut completed: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut i32 = core::ptr::addr_of_mut!(completed) as *mut i32;
let mut tmp: core::mem::MaybeUninit<*mut list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_45: *mut *mut list_head = core::ptr::addr_of_mut!(tmp) as *mut *mut list_head;
let mut entry: core::mem::MaybeUninit<*mut list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_46: *mut *mut list_head = core::ptr::addr_of_mut!(entry) as *mut *mut list_head;
let mut last: core::mem::MaybeUninit<*mut *mut ed> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut *mut *mut ed = core::ptr::addr_of_mut!(last) as *mut *mut *mut ed;
let mut ed: core::mem::MaybeUninit<*mut ed> = core::mem::MaybeUninit::uninit();
let __slate_slot_49: *mut *mut ed = core::ptr::addr_of_mut!(ed) as *mut *mut ed;
let mut tick: u32 = unsafe { core::mem::zeroed() };
unsafe {
tick = ohci_frame_no(ohci as *const ohci_hcd) as u32;
let __slate_auto_bytes_10 = __slate_slot_49 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_11 = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 8usize);
'__loop_52: loop {
*__slate_slot_48 = core::ptr::addr_of_mut!((*ohci).ed_rm_list);
*__slate_slot_49 = *(*__slate_slot_48);
loop {
if !(*__slate_slot_49).is_null() {
'__join_21: {
'__join_50: {
let __slate_auto_bytes_5 = __slate_slot_46 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_45 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
// only take off EDs that the HC isn't using, accounting for
// frame counter wraps and EDs with partially retired TDs
if !(likely!(((*ohci).rh_state as u32) == 2) != 0 && tick_before!(tick, unsafe { (*(*__slate_slot_49)).tick })) {
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_49)).td_list) as *const list_head) == 0 {
let __slate_auto_bytes_4 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_41 = list_first_entry__4!(core::ptr::addr_of_mut!((*(*__slate_slot_49)).td_list), td);
// INTR_WDH may need to clean up first
*__slate_slot_40 = hc32_to_cpu(ohci as *const ohci_hcd, (*(*__slate_slot_49)).hwHeadP) & 4294967264u32;
if (*(*__slate_slot_41)).td_dma != (*__slate_slot_40 as u64) && ((*ohci).rh_state as u32) == 2 {
break '__join_50;
} else {
// Don't mess up anything already on the done list
if !unsafe { (*(*__slate_slot_41)).next_dl_td }.is_null() {
break '__join_50;
}
}
}
// ED's now officially unlinked, hc doesn't see
let __v39: u32 = !cpu_to_hc32(ohci as *const ohci_hcd, 1u32);
let __v38: *mut ed = *__slate_slot_49;
(*__v38).hwHeadP = (*__v38).hwHeadP & __v39;
(*(*__slate_slot_49)).hwNextED = 0;
loop {
kcsan_wmb!();
core::arch::asm!("sfence", options(nostack, att_syntax));
if true {
break;
}
}
let __v35: u32 = !cpu_to_hc32(ohci as *const ohci_hcd, 134234112u32);
let __v34: *mut ed = *__slate_slot_49;
(*__v34).hwINFO = (*__v34).hwINFO & __v35;
// reentrancy:  if we drop the schedule lock, someone might
// have modified this list.  normally it's just prepending
// entries (which we'd ignore), but paranoia won't hurt.
*(*__slate_slot_48) = (*(*__slate_slot_49)).ed_next;
(*(*__slate_slot_49)).ed_next = core::ptr::null_mut::<ed>();
*__slate_slot_43 = 0;
// unlink urbs as requested, but rescan the list after
// we call a completion since it might have unlinked
// another (earlier) urb
//
// When we get here, the HC doesn't see this ed.  But it
// must not be rescheduled until all completed URBs have
// been given back to the driver.
loop {
*__slate_slot_44 = 0;
*__slate_slot_42 = core::ptr::addr_of_mut!((*(*__slate_slot_49)).hwHeadP);
*__slate_slot_46 = (*core::ptr::addr_of_mut!((*(*__slate_slot_49)).td_list)).next;
*__slate_slot_45 = (*(*__slate_slot_46)).next;
while list_is_head(*__slate_slot_46 as *const list_head, core::ptr::addr_of_mut!((*(*__slate_slot_49)).td_list) as *const list_head) == 0 {
let __slate_auto_bytes_12 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_30 = list_entry__3!(*__slate_slot_46, td);
*__slate_slot_29 = (*(*__slate_slot_30)).urb;
*__slate_slot_28 = (*(*(*__slate_slot_30)).urb).hcpriv as *mut urb_priv;
if (*(*__slate_slot_29)).unlinked == 0 {
*__slate_slot_42 = core::ptr::addr_of_mut!((*(*__slate_slot_30)).hwNextTD);
} else {
// patch pointer hc uses
*__slate_slot_27 = *(*__slate_slot_42) & !cpu_to_hc32(ohci as *const ohci_hcd, 4294967264u32);
*(*__slate_slot_42) = (*(*__slate_slot_30)).hwNextTD | *__slate_slot_27;
// If this was unlinked, the TD may not have been
// retired ... so manually save the data toggle.
// The controller ignores the value we save for
// control and ISO endpoints.
*__slate_slot_26 = hc32_to_cpup(ohci as *const ohci_hcd, core::ptr::addr_of_mut!((*(*__slate_slot_30)).hwINFO) as *const u32);
if *__slate_slot_26 & 50331648 == 33554432 {
let __v21: u32 = !cpu_to_hc32(ohci as *const ohci_hcd, 2u32);
let __v20: *mut ed = *__slate_slot_49;
(*__v20).hwHeadP = (*__v20).hwHeadP & __v21;
} else {
if *__slate_slot_26 & 50331648 == 50331648 {
let __v25: u32 = cpu_to_hc32(ohci as *const ohci_hcd, 2u32);
let __v24: *mut ed = *__slate_slot_49;
(*__v24).hwHeadP = (*__v24).hwHeadP | __v25;
}
}
// HC may have partly processed this TD
td_done(ohci, *__slate_slot_29, *__slate_slot_30);
let __v17: *mut urb_priv = *__slate_slot_28;
(*__v17).td_cnt = ((*__v17).td_cnt as i32).wrapping_add(1) as u16;
// if URB is done, clean up
if ((*(*__slate_slot_28)).td_cnt as i32) >= ((*(*__slate_slot_28)).length as i32) {
*__slate_slot_44 = 1;
*__slate_slot_43 = 1;
finish_urb(ohci, *__slate_slot_29, 0i32);
}
}
*__slate_slot_46 = *__slate_slot_45;
*__slate_slot_45 = (*(*__slate_slot_46)).next;
}
if *__slate_slot_44 != 0 {
__v13 = list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_49)).td_list) as *const list_head) == 0;
} else {
__v13 = false;
}
if !__v13 {
break;
}
}
// If no TDs are queued, ED is now idle.
// Otherwise, if the HC is running, reschedule.
// If the HC isn't running, add ED back to the
// start of the list for later processing.
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_49)).td_list) as *const list_head) != 0 {
(*(*__slate_slot_49)).state = 0;
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_49)).in_use_list));
} else {
if ((*ohci).rh_state as u32) == 2 {
ed_schedule(ohci, *__slate_slot_49);
} else {
(*(*__slate_slot_49)).ed_next = (*ohci).ed_rm_list;
(*ohci).ed_rm_list = *__slate_slot_49;
// Don't loop on the same ED
if *__slate_slot_48 == core::ptr::addr_of_mut!((*ohci).ed_rm_list) {
*__slate_slot_48 = core::ptr::addr_of_mut!((*(*__slate_slot_49)).ed_next);
}
}
}
if *__slate_slot_43 != 0 {
continue '__loop_52;
} else {
break '__join_21;
}
}
}
*__slate_slot_48 = core::ptr::addr_of_mut!((*(*__slate_slot_49)).ed_next);
}
*__slate_slot_49 = *(*__slate_slot_48);
} else {
break '__loop_52;
}
}
}
// maybe reenable control and bulk lists
if ((*ohci).rh_state as u32) == 2 && unsafe { (*ohci).ed_rm_list }.is_null() {
command = 0;
control = 0;
if !unsafe { (*ohci).ed_controltail }.is_null() {
command |= 2u32;
if quirk_zfmicro(ohci) != 0 {
mdelay!(1i32);
}
if (*ohci).hc_control & 16u32 == 0 {
control |= 16u32;
ohci_writel!(ohci, 0i32, core::ptr::addr_of_mut!((*(*ohci).regs).ed_controlcurrent));
}
}
if !unsafe { (*ohci).ed_bulktail }.is_null() {
command |= 4u32;
if quirk_zfmicro(ohci) != 0 {
mdelay!(1i32);
}
if (*ohci).hc_control & 32u32 == 0 {
control |= 32u32;
ohci_writel!(ohci, 0i32, core::ptr::addr_of_mut!((*(*ohci).regs).ed_bulkcurrent));
}
}
// CLE/BLE to enable, CLF/BLF to (maybe) kickstart
if control != 0 {
(*ohci).hc_control = (*ohci).hc_control | control;
if quirk_zfmicro(ohci) != 0 {
mdelay!(1i32);
}
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
}
if command != 0 {
if quirk_zfmicro(ohci) != 0 {
mdelay!(1i32);
}
ohci_writel__2!(ohci, command, core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus));
}
}
}
}

fn takeback_td(ohci: *mut ohci_hcd, mut td: *mut td) {
let urb: *mut urb = unsafe { (*td).urb };
let urb_priv: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
let ed: *mut ed = unsafe { (*td).ed };
let mut status: i32 = 0;
// update URB's length and status from TD
status = td_done(ohci, urb, td);
unsafe {
(*urb_priv).td_cnt = ((*urb_priv).td_cnt as i32).wrapping_add(1) as u16;
}
// If all this urb's TDs are done, call complete()
if ((unsafe { (*urb_priv).td_cnt }) as i32) >= ((unsafe { (*urb_priv).length }) as i32) {
finish_urb(ohci, urb, status);
}
// clean schedule:  unlink EDs that are no longer busy
if list_empty((unsafe { core::ptr::addr_of_mut!((*ed).td_list) }) as *const list_head) != 0 {
if ((unsafe { (*ed).state }) as i32) == 2 {
start_ed_unlink(ohci, ed);
}
// ... reenabling halted EDs only after fault cleanup
} else {
if (unsafe { (*ed).hwINFO }) & cpu_to_hc32(ohci as *const ohci_hcd, 134234112u32) == cpu_to_hc32(ohci as *const ohci_hcd, 16384u32) {
td = list_entry__3!(unsafe { (*ed).td_list.next }, td);
if (unsafe { (*td).hwINFO }) & cpu_to_hc32(ohci as *const ohci_hcd, 131072u32) == 0 {
let __v4: u32 = !cpu_to_hc32(ohci as *const ohci_hcd, 16384u32);
unsafe {
(*ed).hwINFO = (*ed).hwINFO & __v4;
}
// ... hc may need waking-up
match (unsafe { (*ed).r#type }) as i32 {
2 => {
ohci_writel!(ohci, 2i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus) });
}
3 => {
ohci_writel!(ohci, 4i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus) });
}
_ => {
}
}
}
}
}
}

fn process_done_list(ohci: *mut ohci_hcd) {
while !unsafe { (*ohci).dl_start }.is_null() {
let td: *mut td = unsafe { (*ohci).dl_start };
if td == unsafe { (*ohci).dl_end } {
unsafe {
(*ohci).dl_end = core::ptr::null_mut::<td>();
}
unsafe {
(*ohci).dl_start = core::ptr::null_mut::<td>();
}
} else {
unsafe {
(*ohci).dl_start = (*td).next_dl_td;
}
}
takeback_td(ohci, td);
}
}

fn ohci_work(ohci: *mut ohci_hcd) {
unsafe {
if ((*ohci).__slate_bits_0.__get_working() as i32) != 0 {
(*ohci).__slate_bits_0.__set_restart_work(1u32);
return;
} else {
(*ohci).__slate_bits_0.__set_working(1u32);
loop {
process_done_list(ohci);
if !unsafe { (*ohci).ed_rm_list }.is_null() {
finish_unlinks(ohci);
}
if ((*ohci).__slate_bits_0.__get_restart_work() as i32) != 0 {
(*ohci).__slate_bits_0.__set_restart_work(0u32);
} else {
break;
}
}
(*ohci).__slate_bits_0.__set_working(0u32);
}
}
}

// On architectures with edge-triggered interrupts we must never return
// IRQ_NONE.
/// Some boards misreport power switching/overcurrent
static mut distrust_firmware: bool = unsafe { core::mem::zeroed() };

fn __check_distrust_firmware() -> *mut bool {
unsafe { core::ptr::addr_of_mut!(distrust_firmware) }
}

static __param_str_distrust_firmware: __SlateAlign16<[u8; 27]> = __SlateAlign16([111u8, 104u8, 99u8, 105u8, 95u8, 104u8, 99u8, 100u8, 46u8, 100u8, 105u8, 115u8, 116u8, 114u8, 117u8, 115u8, 116u8, 95u8, 102u8, 105u8, 114u8, 109u8, 119u8, 97u8, 114u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_distrust_firmware: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_distrust_firmware.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_bool) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_ohci_hcd_c_106>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_106_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_106_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_106_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_106_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_106_6>() as u64) as i32).wrapping_add(0i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(distrust_firmware) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_652: [u8; 41] = [111, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 100, 105, 115, 116, 114, 117, 115, 116, 95, 102, 105, 114, 109, 119, 97, 114, 101, 58, 98, 111, 111, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_653: [u8; 82] = [111, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 61, 100, 105, 115, 116, 114, 117, 115, 116, 95, 102, 105, 114, 109, 119, 97, 114, 101, 58, 116, 114, 117, 101, 32, 116, 111, 32, 100, 105, 115, 116, 114, 117, 115, 116, 32, 102, 105, 114, 109, 119, 97, 114, 101, 32, 112, 111, 119, 101, 114, 47, 111, 118, 101, 114, 99, 117, 114, 114, 101, 110, 116, 32, 115, 101, 116, 117, 112, 0];

/// Some boards leave IR set wrongly, since they fail BIOS/SMM handshakes
static mut no_handshake: bool = unsafe { core::mem::zeroed() };

fn __check_no_handshake() -> *mut bool {
unsafe { core::ptr::addr_of_mut!(no_handshake) }
}

static __param_str_no_handshake: __SlateAlign16<[u8; 22]> = __SlateAlign16([111u8, 104u8, 99u8, 105u8, 95u8, 104u8, 99u8, 100u8, 46u8, 110u8, 111u8, 95u8, 104u8, 97u8, 110u8, 100u8, 115u8, 104u8, 97u8, 107u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_no_handshake: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_no_handshake.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_bool) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_ohci_hcd_c_112>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_112_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_112_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_112_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_112_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ohci_hcd_c_112_6>() as u64) as i32).wrapping_add(0i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(no_handshake) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_654: [u8; 36] = [111, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 110, 111, 95, 104, 97, 110, 100, 115, 104, 97, 107, 101, 58, 98, 111, 111, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_655: [u8; 70] = [111, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 61, 110, 111, 95, 104, 97, 110, 100, 115, 104, 97, 107, 101, 58, 116, 114, 117, 101, 32, 40, 110, 111, 116, 32, 100, 101, 102, 97, 117, 108, 116, 41, 32, 100, 105, 115, 97, 98, 108, 101, 115, 32, 66, 73, 79, 83, 32, 104, 97, 110, 100, 115, 104, 97, 107, 101, 0];

fn number_of_tds(urb: *mut urb) -> i32 {
let mut len: i32 = 0;
let mut i: i32 = 0;
let mut num: i32 = 0;
let mut this_sg_len: i32 = 0;
let mut sg: *mut scatterlist = unsafe { core::mem::zeroed() };
len = (unsafe { (*urb).transfer_buffer_length }) as i32;
i = unsafe { (*urb).num_mapped_sgs };
if len > 0 && i > 0 {
// Scatter-gather transfer
num = 0;
sg = unsafe { (*urb).sg };
loop {
this_sg_len = min_t!(i32, sg_dma_len!(sg), len);
num = num.wrapping_add(this_sg_len.wrapping_add(4096).wrapping_sub(1) / 4096);
len = len.wrapping_sub(this_sg_len);
let __v8: i32 = i.wrapping_sub(1);
i = __v8;
if __v8 <= 0 || len <= 0 {
break;
}
sg = sg_next(sg);
}
} else {
// Non-SG transfer
// one TD for every 4096 Bytes (could be up to 8K)
num = len.wrapping_add(4096).wrapping_sub(1) / 4096;
}
num
}

/// queue up an urb for anything except the root hub
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_urb_enqueue")]
extern "C" fn ohci_urb_enqueue(hcd: *mut usb_hcd, urb: *mut urb, mem_flags: u32) -> i32 {
let mut __slate_compound_53: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_55: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut frame: u16 = unsafe { core::mem::zeroed() };
// Start up the I/O watchdog timer, if it's not running
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut length: u16 = unsafe { core::mem::zeroed() };
let mut frame_1: u16 = unsafe { core::mem::zeroed() };
let mut next: u16 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_ptr: *mut urb_priv = unsafe { core::mem::zeroed() };
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut __count: u64 = unsafe { core::mem::zeroed() };
let mut __v31: bool = unsafe { core::mem::zeroed() };
let mut retval: i32 = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut size: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut pipe: u32 = unsafe { core::mem::zeroed() };
let mut urb_priv: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut *mut urb_priv = core::ptr::addr_of_mut!(urb_priv) as *mut *mut urb_priv;
let mut ed: core::mem::MaybeUninit<*mut ed> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut *mut ed = core::ptr::addr_of_mut!(ed) as *mut *mut ed;
let mut ohci: *mut ohci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ohci = hcd_to_ohci(hcd);
let __slate_auto_bytes_2 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
pipe = (*urb).pipe;
let __slate_auto_bytes_4 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
size = 0;
let __slate_auto_bytes_5 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
retval = 0;
// every endpoint has a ed, locate and maybe (re)initialize it
*__slate_slot_41 = ed_get(ohci, (*urb).ep, (*urb).dev, pipe, (*urb).interval);
if (*__slate_slot_41).is_null() {
return 12i32.wrapping_neg();
} else {
'__join_37: {
// for the private part of the URB we need the number of TDs (size)
let __t0: i32 = (*(*__slate_slot_41)).r#type as i32;
if __t0 == 2 {
if (*urb).transfer_buffer_length > 4096 {
return 90i32.wrapping_neg();
} else {
// td_submit_urb() doesn't yet handle these
// 1 TD for setup, 1 for ACK, plus ...
size = 2;
// case PIPE_INTERRUPT:
//
// case PIPE_BULK:
}
} else {
if __t0 == 0 {
size = (*urb).number_of_packets; // number of packets from URB
break '__join_37;
}
}
let __v34: i32 = number_of_tds(urb);
size = size.wrapping_add(__v34);
// maybe a zero-length packet to wrap it up
if size == 0 {
size = size.wrapping_add(1);
} else {
if (*urb).transfer_flags & 64 != 0 {
__v31 = (*urb).transfer_buffer_length % ((usb_maxpacket((*urb).dev, pipe as i32) as i32) as u32) == 0;
} else {
__v31 = false;
}
if __v31 {
size = size.wrapping_add(1);
}
}
}
__count = size as u64;
__obj_size = struct_size_t__5!(__count);
let __slate_auto_bytes_1 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_19 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_20 = _kzalloc_noprof(__obj_size, mem_flags, *{
__slate_compound_53 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_53)
});
alloc_tag_restore!();
} else {
*__slate_slot_20 = _kzalloc_noprof(__obj_size, mem_flags, *{
__slate_compound_55 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_55)
});
}
__obj_ptr = *__slate_slot_20 as *mut urb_priv;
if !__obj_ptr.is_null() {
__set_flex_counter__2!(__count, __obj_ptr);
}
// allocate the private part of the URB
*__slate_slot_40 = __obj_ptr;
if (*__slate_slot_40).is_null() {
return 12i32.wrapping_neg();
} else {
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_40)).pending));
(*(*__slate_slot_40)).length = size as u16;
(*(*__slate_slot_40)).ed = *__slate_slot_41;
// allocate the TDs (deferring hash chain updates)
*__slate_slot_38 = 0;
while *__slate_slot_38 < size {
*(core::ptr::addr_of_mut!((*(*__slate_slot_40)).td) as *mut *mut td).wrapping_offset(*__slate_slot_38 as isize) = td_alloc(ohci, mem_flags);
if unsafe { *(core::ptr::addr_of_mut!((*(*__slate_slot_40)).td) as *mut *mut td).wrapping_offset(*__slate_slot_38 as isize) }.is_null() {
(*(*__slate_slot_40)).length = *__slate_slot_38 as u16;
urb_free_priv(ohci, *__slate_slot_40);
return 12i32.wrapping_neg();
} else {
let __v18: i32 = *__slate_slot_38;
*__slate_slot_38 = __v18.wrapping_add(1);
}
}
'__join_2: {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_36);
// don't submit to a dead HC
if HCD_HW_ACCESSIBLE!(hcd) == 0 {
retval = 19i32.wrapping_neg();
} else {
if ((*ohci).rh_state as u32) != 2 {
retval = 19i32.wrapping_neg();
} else {
retval = usb_hcd_link_urb_to_ep(hcd, urb);
if retval == 0 {
// schedule the ed if needed
if ((*(*__slate_slot_41)).state as i32) == 0 {
retval = ed_schedule(ohci, *__slate_slot_41);
if retval < 0 {
usb_hcd_unlink_urb_from_ep(hcd, urb);
break '__join_2;
} else {
if (*ohci).prev_frame_no == 4294967040 {
__v8 = list_empty(core::ptr::addr_of_mut!((*ohci).eds_in_use) as *const list_head) != 0;
} else {
__v8 = false;
}
if __v8 && (*ohci).flags & 4096 == 0 {
(*ohci).prev_frame_no = ohci_frame_no(ohci as *const ohci_hcd) as u32;
mod_timer(core::ptr::addr_of_mut!((*ohci).io_watchdog), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(275u32)));
}
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_41)).in_use_list), core::ptr::addr_of_mut!((*ohci).eds_in_use));
if ((*(*__slate_slot_41)).r#type as i32) == 0 {
frame = ohci_frame_no(ohci as *const ohci_hcd);
// delay a few frames before the first TD
let __v6: i32 = max_t!(u16, 8i32, unsafe { (*(*__slate_slot_41)).interval });
frame = (frame as i32).wrapping_add(__v6) as u16;
let __v2: u16 = ((frame as i32) & !((*(*__slate_slot_41)).interval as i32).wrapping_sub(1)) as u16;
frame = __v2;
frame = ((frame as i32) | ((*(*__slate_slot_41)).branch as i32)) as u16;
(*urb).start_frame = frame as i32;
(*(*__slate_slot_41)).last_iso = (frame as i32).wrapping_add(((*(*__slate_slot_41)).interval as i32).wrapping_mul(size.wrapping_sub(1))) as u16;
}
}
} else {
if ((*(*__slate_slot_41)).r#type as i32) == 0 {
next = (ohci_frame_no(ohci as *const ohci_hcd) as i32).wrapping_add(1) as u16;
frame_1 = ((*(*__slate_slot_41)).last_iso as i32).wrapping_add((*(*__slate_slot_41)).interval as i32) as u16;
length = ((*(*__slate_slot_41)).interval as i32).wrapping_mul(size.wrapping_sub(1)) as u16;
// Behind the scheduling threshold?
if unlikely!(tick_before!(frame_1, next)) != 0 {
// URB_ISO_ASAP: Round up to the first available slot
if (*urb).transfer_flags & 2 != 0 {
frame_1 = (frame_1 as i32).wrapping_add((next as i32).wrapping_sub(frame_1 as i32).wrapping_add((*(*__slate_slot_41)).interval as i32).wrapping_sub(1) & ((*(*__slate_slot_41)).interval as i32).wrapping_neg()) as u16;
// Not ASAP: Use the next slot in the stream,
// no matter what.
} else {
// Some OHCI hardware doesn't handle late TDs
// correctly.  After retiring them it proceeds
// to the next ED instead of the next TD.
// Therefore we have to omit the late TDs
// entirely.
(*(*__slate_slot_40)).td_cnt = ((((next as i32).wrapping_sub(frame_1 as i32) as u16) as i32).wrapping_add((*(*__slate_slot_41)).interval as i32).wrapping_sub(1) / ((*(*__slate_slot_41)).interval as i32)) as u16;
if ((*(*__slate_slot_40)).td_cnt as i32) >= ((*(*__slate_slot_40)).length as i32) {
let __v13: *mut urb_priv = *__slate_slot_40;
(*__v13).td_cnt = ((*__v13).td_cnt as i32).wrapping_add(1) as u16;
// Mark it
}
}
}
(*urb).start_frame = frame_1 as i32;
(*(*__slate_slot_41)).last_iso = (frame_1 as i32).wrapping_add(length as i32) as u16;
}
}
// fill the TDs and link them to the ed; and
// enable that part of the schedule, if needed
// and update count of queued periodic urbs
(*urb).hcpriv = *__slate_slot_40 as *mut ();
td_submit_urb(ohci, urb);
}
}
}
}
if retval != 0 {
urb_free_priv(ohci, *__slate_slot_40);
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_36);
return retval;
}
}
}
}

/// decouple the URB from the HC queues (TDs, urb_priv).
/// reporting is always done
/// asynchronously, and we might be dealing with an urb that's
/// partially transferred, or an ED with other urbs being unlinked.
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_urb_dequeue")]
extern "C" fn ohci_urb_dequeue(hcd: *mut usb_hcd, urb: *mut urb, status: i32) -> i32 {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
let mut flags: u64 = 0;
let mut rc: i32 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
rc = unsafe { usb_hcd_check_unlink_urb(hcd, urb, status) };
if rc == 0 {
// Unless an IRQ completed the unlink while it was being
// handed to us, flag it for unlink and giveback, and force
// some upcoming INTR_SF to call finish_unlinks()
let urb_priv: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
if ((unsafe { (*(*urb_priv).ed).state }) as i32) == 2 {
start_ed_unlink(ohci, unsafe { (*urb_priv).ed });
}
if ((unsafe { (*ohci).rh_state }) as u32) != 2 {
// With HC dead, we can clean up right away
ohci_work(ohci);
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
rc
}

// frees config/altsetting state for endpoints,
// including ED memory, dummy TD, and bulk/intr data toggle
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_endpoint_disable")]
extern "C" fn ohci_endpoint_disable(hcd: *mut usb_hcd, ep: *mut usb_host_endpoint) {
let mut limit: u32 = unsafe { core::mem::zeroed() };
let mut ed: *mut ed = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut ohci: *mut ohci_hcd = unsafe { core::mem::zeroed() };
let mut __slate_state: usize = 14usize;
unsafe {
loop {
match __slate_state {
0 => {
(*ep).hcpriv = core::ptr::null_mut::<()>();
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_4);
break;
}
1 => {
_dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"leak ed %p (#%02x) state %d%s\n".as_ptr() as *const u8, ed, (*ep).desc.bEndpointAddress as i32, (*ed).state as i32, if list_empty(core::ptr::addr_of_mut!((*ed).td_list) as *const list_head) != 0 { c"".as_ptr() as *mut u8 } else { c" (has tds)".as_ptr() as *mut u8 });
// caller was supposed to have unlinked any requests;
// that's not our job.  can't recover; must leak ed.
td_free(ohci, (*ed).dummy);
__slate_state = 0usize;
continue;
}
2 => {
if false {
__slate_state = 2usize;
continue;
} else {
__slate_state = 1usize;
continue;
}
}
3 => {
// fully unlinked
__slate_state = 2usize;
continue;
}
4 => {
td_free(ohci, (*ed).dummy);
ed_free(ohci, ed);
__slate_state = 0usize;
continue;
}
5 => {
if list_empty(core::ptr::addr_of_mut!((*ed).td_list) as *const list_head) != 0 {
__slate_state = 4usize;
continue;
} else {
__slate_state = 3usize;
continue;
}
}
6 => {
// wait for hw to finish?
// major IRQ delivery trouble loses INTR_SF too...
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_4);
schedule_timeout_uninterruptible(1i64);
__slate_state = 12usize;
continue;
}
7 => {
_dev_warn((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"ED unlink timeout\n".as_ptr() as *const u8);
__slate_state = 11usize;
continue;
}
8 => {
if false {
__slate_state = 8usize;
continue;
} else {
__slate_state = 7usize;
continue;
}
}
9 => {
let __v1: u32 = limit;
limit = __v1.wrapping_sub(1);
if __v1 == 0 {
__slate_state = 8usize;
continue;
} else {
__slate_state = 6usize;
continue;
}
}
10 => {
let __t0: i32 = (*ed).state as i32;
if __t0 == 1 {
__slate_state = 9usize;
continue;
} else {
if __t0 == 0 {
__slate_state = 5usize;
continue;
} else {
__slate_state = 2usize;
continue;
}
}
}
11 => {
(*ed).state = 0;
ohci_work(ohci);
__slate_state = 10usize;
continue;
}
12 => {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ohci).lock), *__slate_slot_4);
if ((*ohci).rh_state as u32) != 2 {
__slate_state = 11usize;
continue;
} else {
__slate_state = 10usize;
continue;
}
}
13 => {
return;
}
14 => {
ohci = hcd_to_ohci(hcd);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ed = (*ep).hcpriv as *mut ed;
limit = 1000;
// ASSERT:  any requests/urbs are being unlinked
//
// ASSERT:  nobody can be submitting urbs for this any more
if ed.is_null() {
__slate_state = 13usize;
continue;
} else {
__slate_state = 12usize;
continue;
}
}
_ => {
core::hint::unreachable_unchecked();
}
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_get_frame")]
extern "C" fn ohci_get_frame(hcd: *mut usb_hcd) -> i32 {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
ohci_frame_no(ohci as *const ohci_hcd) as i32
}

fn ohci_usb_reset(ohci: *mut ohci_hcd) {
unsafe {
(*ohci).hc_control = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
}
unsafe {
(*ohci).hc_control = (*ohci).hc_control & 512u32;
}
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) });
unsafe {
(*ohci).rh_state = 0;
}
}

/// ohci_shutdown forcibly disables IRQs and DMA, helping kexec and
/// other cases where the next software may expect clean state from the
/// "firmware".  this is bus-neutral, unlike shutdown() methods.
fn _ohci_shutdown(hcd: *mut usb_hcd) {
let mut ohci: *mut ohci_hcd = unsafe { core::mem::zeroed() };
ohci = hcd_to_ohci(hcd);
ohci_writel__2!(ohci, 4294967295u32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrdisable) });
// Software reset, after which the controller goes into SUSPEND
ohci_writel!(ohci, 1i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus) });
ohci_readl__2!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus) }); // flush the writes
udelay(10u64);
ohci_writel__2!(ohci, unsafe { (*ohci).fminterval }, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).fminterval) });
unsafe {
(*ohci).rh_state = 0;
}
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_shutdown")]
extern "C" fn ohci_shutdown(hcd: *mut usb_hcd) {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
_ohci_shutdown(hcd);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
}

// HC functions
// init memory, and kick BIOS/SMM off
fn ohci_init(ohci: *mut ohci_hcd) -> i32 {
let mut ret: i32 = 0;
let hcd: *mut usb_hcd = ohci_to_hcd(ohci as *const ohci_hcd);
// Accept arbitrarily long scatter-gather lists
if unsafe { (*hcd).localmem_pool }.is_null() {
unsafe {
(*hcd).self_.sg_tablesize = 4294967295u32;
}
}
if unsafe { distrust_firmware } {
unsafe {
(*ohci).flags = (*ohci).flags | 256;
}
}
unsafe {
(*ohci).rh_state = 0;
}
unsafe {
(*ohci).regs = (*hcd).regs as *mut ohci_regs;
}
// REVISIT this BIOS handshake is now moved into PCI "quirks", and
// was never needed for most non-PCI systems ... remove the code?
// SMM owns the HC?  not for long!
let __v3: bool = !unsafe { no_handshake } && ohci_readl!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) }) & 256u32 != 0;
if __v3 {
// this timeout is arbitrary.  we make it long, so systems
// depending on usb keyboards may be usable even if the
// BIOS/SMM code seems pretty broken.
// arbitrary: five seconds
let mut temp: u32 = 500;
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"USB HC TakeOver from BIOS/SMM\n".as_ptr() as *const u8) };
}
ohci_writel!(ohci, 1073741824i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrenable) });
ohci_writel!(ohci, 8i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus) });
while ohci_readl!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) }) & 256u32 != 0 {
unsafe { msleep(10u32) };
let __v5: u32 = temp.wrapping_sub(1);
temp = __v5;
if __v5 == 0 {
while false {
}
unsafe { _dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"USB HC takeover failed!  (BIOS/SMM bug)\n".as_ptr() as *const u8) };
return 16i32.wrapping_neg();
}
}
ohci_usb_reset(ohci);
}
// Disable HC interrupts
ohci_writel!(ohci, -2147483648i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrdisable) });
// flush the writes, and save key bits like RWC
if ohci_readl!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).control) }) & 512u32 != 0 {
unsafe {
(*ohci).hc_control = (*ohci).hc_control | 512u32;
}
}
// Read the number of ports unless overridden
if (unsafe { (*ohci).num_ports }) == 0 {
unsafe {
(*ohci).num_ports = (roothub_a(ohci) & 255u32) as i32;
}
}
if !unsafe { (*ohci).hcca }.is_null() {
return 0;
}
timer_setup!(unsafe { core::ptr::addr_of_mut!((*ohci).io_watchdog) }, Some(io_watchdog_func), 0i32);
unsafe {
(*ohci).prev_frame_no = 4294967040;
}
if !unsafe { (*hcd).localmem_pool }.is_null() {
unsafe {
(*ohci).hcca = gen_pool_dma_alloc_align((*hcd).localmem_pool, core::mem::size_of::<ohci_hcca>() as u64, core::ptr::addr_of_mut!((*ohci).hcca_dma), 256i32) as *mut ohci_hcca;
}
} else {
unsafe {
(*ohci).hcca = dma_alloc_coherent((*hcd).self_.controller, core::mem::size_of::<ohci_hcca>() as u64, core::ptr::addr_of_mut!((*ohci).hcca_dma), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *mut ohci_hcca;
}
}
if unsafe { (*ohci).hcca }.is_null() {
return 12i32.wrapping_neg();
}
let __v11: i32 = ohci_mem_init(ohci);
ret = __v11;
if __v11 < 0 {
ohci_stop(hcd);
} else {
create_debug_files(ohci);
}
ret
}

/// Start an OHCI controller, set the BUS operational
/// resets USB and controller
/// enable interrupts
fn ohci_run(ohci: *mut ohci_hcd) -> i32 {
// some OHCI implementations are finicky about how they init.
// bogus values here mean not even enumeration could work.
let mut __v19: bool = unsafe { core::mem::zeroed() };
let mut hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
let mut val: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut u32 = core::ptr::addr_of_mut!(val) as *mut u32;
let mut mask: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut u32 = core::ptr::addr_of_mut!(mask) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let first: i32 = ((*ohci).fminterval == 0) as i32;
hcd = ohci_to_hcd(ohci as *const ohci_hcd);
(*ohci).rh_state = 0;
// boot firmware should have set this up (5.1.1.3.1)
if first != 0 {
*__slate_slot_39 = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).fminterval));
(*ohci).fminterval = *__slate_slot_39 & 16383;
if (*ohci).fminterval != 11999 {
}
let __v35: u32 = (*ohci).fminterval;
let __v34: u32 = __v35 | FSMP!(unsafe { (*ohci).fminterval }) << 16i32;
(*ohci).fminterval = __v34;
// also: power/overcurrent flags in roothub.a
}
// Reset USB nearly "by the book".  RemoteWakeupConnected has
// to be checked in case boot firmware (BIOS/SMM/...) has set up
// wakeup in a way the bus isn't aware of (e.g., legacy PCI PM).
// If the bus glue detected wakeup capability then it should
// already be enabled; if so we'll just enable it again.
if (*ohci).hc_control & 512u32 != 0 {
device_set_wakeup_capable((*hcd).self_.controller, true);
}
'__join_23: {
let __t0: u32 = (*ohci).hc_control & 192u32;
if __t0 == 128 {
*__slate_slot_39 = 0;
break '__join_23;
} else {
if __t0 != 192 {
if __t0 != 64 {
(*ohci).hc_control = (*ohci).hc_control & 512u32;
(*ohci).hc_control = (*ohci).hc_control | 0u32;
*__slate_slot_39 = 50; // msec wait
break '__join_23;
}
}
}
(*ohci).hc_control = (*ohci).hc_control & 512u32;
(*ohci).hc_control = (*ohci).hc_control | 64u32;
*__slate_slot_39 = 10; // msec wait
}
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
// flush the writes
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
msleep(*__slate_slot_39);
memset((*ohci).hcca as *mut (), 0i32, core::mem::size_of::<ohci_hcca>() as u64);
// 2msec timelimit here means no irqs/preempt
spin_lock_irq(core::ptr::addr_of_mut!((*ohci).lock));
'__join_20: {
'__join_7: {
loop {
ohci_writel!(ohci, 1i32, core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus)); // HC Reset requires max 10 us delay
*__slate_slot_39 = 30; // ... allow extra time
while ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).cmdstatus)) & 1u32 != 0 {
let __v21: u32 = *__slate_slot_39;
let __v20: u32 = __v21.wrapping_sub(1);
*__slate_slot_39 = __v20;
if __v20 == 0 {
break '__join_20;
} else {
udelay(1u64);
}
}
// now we're in the SUSPEND state ... must go OPERATIONAL
// within 2msec else HC enters RESUME
//
// ... but some hardware won't init fmInterval "by the book"
// (SiS, OPTi ...), so reset again instead.  SiS doesn't need
// this if we write fmInterval after we're OPERATIONAL.
// Unclear about ALi, ServerWorks, and others ... this could
// easily be a longstanding bug in chip init on Linux.
if (*ohci).flags & 4 != 0 {
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
// flush those writes
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
}
// Tell the controller where the control and bulk lists are
// The lists are empty now.
ohci_writel!(ohci, 0i32, core::ptr::addr_of_mut!((*(*ohci).regs).ed_controlhead));
ohci_writel!(ohci, 0i32, core::ptr::addr_of_mut!((*(*ohci).regs).ed_bulkhead));
// a reset clears this
ohci_writel__2!(ohci, (*ohci).hcca_dma as u32, core::ptr::addr_of_mut!((*(*ohci).regs).hcca));
periodic_reinit(ohci);
if ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).fminterval)) & 1073676288 == 0 {
__v19 = true;
} else {
__v19 = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).periodicstart)) == 0;
}
if __v19 {
if (*ohci).flags & 4 == 0 {
(*ohci).flags = (*ohci).flags | 4;
} else {
break '__join_7;
}
} else {
break;
}
}
// use rhsc irqs after hub_wq is allocated
set_bit(2i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
(*hcd).__slate_bits_0.__set_uses_new_polling(1u32);
// start controller operations
(*ohci).hc_control = (*ohci).hc_control & 512u32;
(*ohci).hc_control = (*ohci).hc_control | 131u32;
ohci_writel__2!(ohci, unsafe { (*ohci).hc_control }, core::ptr::addr_of_mut!((*(*ohci).regs).control));
(*ohci).rh_state = 2;
// wake on ConnectStatusChange, matching external hubs
ohci_writel!(ohci, 32768i32, core::ptr::addr_of_mut!((*(*ohci).regs).roothub.status));
// Choose the interrupts we care about now, others later on demand
*__slate_slot_40 = 2147483738u32;
ohci_writel!(ohci, -1i32, core::ptr::addr_of_mut!((*(*ohci).regs).intrstatus));
ohci_writel__2!(ohci, *__slate_slot_40, core::ptr::addr_of_mut!((*(*ohci).regs).intrenable));
// handle root hub init quirks ...
*__slate_slot_39 = roothub_a(ohci);
// Configure for per-port over-current protection by default
let __v8: u32 = *__slate_slot_39 & 4294963199u32;
*__slate_slot_39 = __v8;
*__slate_slot_39 |= 2048u32;
if (*ohci).flags & 2 != 0 {
// NSC 87560 and maybe others.
// Ganged power switching, no over-current protection.
*__slate_slot_39 |= 4096u32;
let __v0: u32 = *__slate_slot_39 & 16774399u32;
*__slate_slot_39 = __v0;
} else {
if (*ohci).flags & 1 != 0 || (*ohci).flags & 256 != 0 {
// hub power always on; required for AMD-756 and some
// Mac platforms.
*__slate_slot_39 |= 512u32;
}
}
ohci_writel__2!(ohci, *__slate_slot_39, core::ptr::addr_of_mut!((*(*ohci).regs).roothub.a));
ohci_writel!(ohci, 65536i32, core::ptr::addr_of_mut!((*(*ohci).regs).roothub.status));
ohci_writel__2!(ohci, if *__slate_slot_39 & 512u32 != 0 { 0u32 } else { 4294901760u32 }, core::ptr::addr_of_mut!((*(*ohci).regs).roothub.b));
// flush those writes
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
(*ohci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(300u32));
spin_unlock_irq(core::ptr::addr_of_mut!((*ohci).lock));
// POTPGT delay is bits 24-31, in 2 ms units.
mdelay__2!(*__slate_slot_39 >> 23i32 & 510);
ohci_dump(ohci);
return 0;
}
spin_unlock_irq(core::ptr::addr_of_mut!((*ohci).lock));
while false {
}
_dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"init err (%08x %04x)\n".as_ptr() as *const u8, ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).fminterval)), ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).periodicstart)));
return 75i32.wrapping_neg();
}
spin_unlock_irq(core::ptr::addr_of_mut!((*ohci).lock));
while false {
}
_dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"USB HC reset timed out!\n".as_ptr() as *const u8);
return 1i32.wrapping_neg();
}
// case OHCI_USB_RESET:
}

// ohci_setup routine for generic controller initialization
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_setup")]
extern "C" fn ohci_setup(hcd: *mut usb_hcd) -> i32 {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
ohci_hcd_init(ohci);
ohci_init(ohci)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ohci_setup_660: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>, *mut ()>(Some(ohci_setup)) };

/// ohci_start routine for generic controller start of all OHCI bus glue
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_start")]
extern "C" fn ohci_start(hcd: *mut usb_hcd) -> i32 {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
let ret: i32 = ohci_run(ohci);
if ret < 0 {
while false {
}
unsafe { _dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"can't start\n".as_ptr() as *const u8) };
ohci_stop(hcd);
}
ret
}

// an interrupt happens
// Some OHCI controllers are known to lose track of completed TDs.  They
// don't add the TDs to the hardware done queue, which means we never see
// them as being completed.
//
// This watchdog routine checks for such problems.  Without some way to
// tell when those TDs have completed, we would never take their EDs off
// the unlink list.  As a result, URBs could never be dequeued and
// endpoints could never be released.
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_irq")]
extern "C" fn ohci_irq(hcd: *mut usb_hcd) -> u32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut ints: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(ints) as *mut i32;
let mut regs: *mut ohci_regs = unsafe { core::mem::zeroed() };
let mut ohci: *mut ohci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ohci = hcd_to_ohci(hcd);
regs = (*ohci).regs;
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Read interrupt status (and flush pending writes).  We ignore the
// optimization of checking the LSB of hcca->done_head; it doesn't
// work on all systems (edge triggering for OHCI can be a factor).
*__slate_slot_7 = ohci_readl!(ohci, core::ptr::addr_of_mut!((*regs).intrstatus)) as i32;
// Check for an all 1's result which is a typical consequence
// of dead, unclocked, or unplugged (CardBus...) devices
loop {
if (*__slate_slot_7 as u32) == !0 {
(*ohci).rh_state = 0;
usb_hc_died(hcd);
return 1;
} else {
// We only care about interrupts that are enabled
let __v6: u32 = ohci_readl!(ohci, core::ptr::addr_of_mut!((*regs).intrenable));
*__slate_slot_7 = ((*__slate_slot_7 as u32) & __v6) as i32;
// interrupt for some other device?
if *__slate_slot_7 == 0 || unlikely!(((*ohci).rh_state as u32) == 0) != 0 {
return 0;
} else {
if *__slate_slot_7 & 16i32 != 0 {
// e.g. due to PCI Master/Target Abort
if quirk_nec(ohci) != 0 {
// Workaround for a silicon bug in some NEC chips used
// in Apple's PowerBooks. Adapted from Darwin code.
while false {
}
_dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"OHCI Unrecoverable Error, scheduling NEC chip restart\n".as_ptr() as *const u8);
ohci_writel!(ohci, 16i32, core::ptr::addr_of_mut!((*regs).intrdisable));
schedule_work(core::ptr::addr_of_mut!((*ohci).nec_work));
} else {
while false {
}
_dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"OHCI Unrecoverable Error, disabled\n".as_ptr() as *const u8);
(*ohci).rh_state = 0;
usb_hc_died(hcd);
}
ohci_dump(ohci);
ohci_usb_reset(ohci);
}
if *__slate_slot_7 & 64i32 != 0 {
(*ohci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(300u32));
ohci_writel!(ohci, 72i32, core::ptr::addr_of_mut!((*regs).intrstatus));
// NOTE: Vendors didn't always make the same implementation
// choices for RHSC.  Many followed the spec; RHSC triggers
// on an edge, like setting and maybe clearing a port status
// change bit.  With others it's level-triggered, active
// until hub_wq clears all the port status change bits.  We'll
// always disable it here and rely on polling until hub_wq
// re-enables it.
ohci_writel!(ohci, 64i32, core::ptr::addr_of_mut!((*regs).intrdisable));
usb_hcd_poll_rh_status(hcd);
} else {
if *__slate_slot_7 & 8i32 != 0 {
ohci_writel!(ohci, 8i32, core::ptr::addr_of_mut!((*regs).intrstatus));
set_bit(2i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
if ((*ohci).__slate_bits_0.__get_autostop() as i32) != 0 {
spin_lock(core::ptr::addr_of_mut!((*ohci).lock));
ohci_rh_resume(ohci);
spin_unlock(core::ptr::addr_of_mut!((*ohci).lock));
} else {
usb_hcd_resume_root_hub(hcd);
}
}
}
// For connect and disconnect events, we expect the controller
// to turn on RHSC along with RD.  But for remote wakeup events
// this might not happen.
spin_lock(core::ptr::addr_of_mut!((*ohci).lock));
if *__slate_slot_7 & 2i32 != 0 {
update_done_list(ohci);
}
// could track INTR_SO to reduce available PCI/... bandwidth
// handle any pending URB/ED unlinks, leaving INTR_SF enabled
// when there's still unlinking to be done (next frame).
ohci_work(ohci);
if *__slate_slot_7 & 4i32 != 0 && unsafe { (*ohci).ed_rm_list }.is_null() && ((*ohci).rh_state as u32) == 2 {
ohci_writel!(ohci, 4i32, core::ptr::addr_of_mut!((*regs).intrdisable));
}
if ((*ohci).rh_state as u32) == 2 {
ohci_writel!(ohci, *__slate_slot_7, core::ptr::addr_of_mut!((*regs).intrstatus));
if *__slate_slot_7 & 2i32 != 0 {
let __v2: u32 = (*ohci).wdh_cnt;
(*ohci).wdh_cnt = __v2.wrapping_add(1);
}
ohci_writel!(ohci, -2147483648i32, core::ptr::addr_of_mut!((*regs).intrenable));
// flush those writes
ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
}
spin_unlock(core::ptr::addr_of_mut!((*ohci).lock));
// repeat until all enabled interrupts are handled
if ((*ohci).rh_state as u32) != 0 {
*__slate_slot_7 = ohci_readl!(ohci, core::ptr::addr_of_mut!((*regs).intrstatus)) as i32;
if *__slate_slot_7 != 0 {
__v0 = (*__slate_slot_7 as u32) & ohci_readl!(ohci, core::ptr::addr_of_mut!((*regs).intrenable)) != 0;
} else {
__v0 = false;
}
if !__v0 {
break;
}
} else {
break;
}
}
}
}
return 1;
}
}

#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_stop")]
extern "C" fn ohci_stop(hcd: *mut usb_hcd) {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
ohci_dump(ohci);
if quirk_nec(ohci) != 0 {
unsafe { flush_work(core::ptr::addr_of_mut!((*ohci).nec_work)) };
}
unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*ohci).io_watchdog)) };
unsafe {
(*ohci).prev_frame_no = 4294967040;
}
ohci_writel!(ohci, -2147483648i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrdisable) });
ohci_usb_reset(ohci);
unsafe { free_irq((*hcd).irq, hcd as *mut ()) };
unsafe {
(*hcd).irq = 0;
}
if quirk_amdiso(ohci) != 0 {
unsafe { usb_amd_dev_put() };
}
remove_debug_files(ohci);
ohci_mem_cleanup(ohci);
if !unsafe { (*ohci).hcca }.is_null() {
if !unsafe { (*hcd).localmem_pool }.is_null() {
gen_pool_free(unsafe { (*hcd).localmem_pool }, (unsafe { (*ohci).hcca }) as u64, core::mem::size_of::<ohci_hcca>() as u64);
} else {
dma_free_coherent(unsafe { (*hcd).self_.controller }, core::mem::size_of::<ohci_hcca>() as u64, (unsafe { (*ohci).hcca }) as *mut (), unsafe { (*ohci).hcca_dma });
}
unsafe {
(*ohci).hcca = core::ptr::null_mut::<ohci_hcca>();
}
unsafe {
(*ohci).hcca_dma = 0;
}
}
}

/// must not be called from interrupt context
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_restart")]
extern "C" fn ohci_restart(ohci: *mut ohci_hcd) -> i32 {
let mut temp: i32 = 0;
// empty the interrupt branches
let mut i: i32 = 0;
let mut r#priv: *mut urb_priv = unsafe { core::mem::zeroed() };
ohci_init(ohci);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ohci).lock) });
unsafe {
(*ohci).rh_state = 0;
}
// Recycle any "live" eds/tds (and urbs).
if list_empty((unsafe { core::ptr::addr_of_mut!((*ohci).pending) }) as *const list_head) == 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"abort schedule...\n".as_ptr() as *const u8) };
}
}
r#priv = list_first_entry__22!(unsafe { core::ptr::addr_of_mut!((*ohci).pending) });
while list_entry_is_head!(r#priv, unsafe { core::ptr::addr_of_mut!((*ohci).pending) }, pending) == 0 {
let urb: *mut urb = unsafe { (*(*(core::ptr::addr_of_mut!((*r#priv).td) as *mut *mut td).wrapping_offset(0 as isize))).urb };
let ed: *mut ed = unsafe { (*r#priv).ed };
match (unsafe { (*ed).state }) as i32 {
2 => {
unsafe {
(*ed).state = 1;
}
let __v2: u32 = cpu_to_hc32(ohci as *const ohci_hcd, 134217728u32);
unsafe {
(*ed).hwINFO = (*ed).hwINFO | __v2;
}
ed_deschedule(ohci, ed);
unsafe {
(*ed).ed_next = (*ohci).ed_rm_list;
}
unsafe {
(*ed).ed_prev = core::ptr::null_mut::<ed>();
}
unsafe {
(*ohci).ed_rm_list = ed;
}
}
1 => {
}
_ => {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"bogus ed %p state %d\n".as_ptr() as *const u8, ed, (*ed).state as i32) };
}
}
}
if (unsafe { (*urb).unlinked }) == 0 {
unsafe {
(*urb).unlinked = 108i32.wrapping_neg();
}
}
r#priv = list_next_entry__21!(r#priv, pending);
}
ohci_work(ohci);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ohci).lock) });
// paranoia, in case that didn't work:
while i < 32 {
unsafe {
*((*ohci).load.as_mut_ptr() as *mut i32).wrapping_offset(i as isize) = 0;
}
i = i.wrapping_add(1);
}
i = 0;
while i < 32 {
unsafe {
*((*(*ohci).hcca).int_table.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = 0;
}
i = i.wrapping_add(1);
}
// no EDs to remove
unsafe {
(*ohci).ed_rm_list = core::ptr::null_mut::<ed>();
}
// empty control and bulk lists
unsafe {
(*ohci).ed_controltail = core::ptr::null_mut::<ed>();
}
unsafe {
(*ohci).ed_bulktail = core::ptr::null_mut::<ed>();
}
let __v14: i32 = ohci_run(ohci);
temp = __v14;
if __v14 < 0 {
while false {
}
unsafe { _dev_err((*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"can't restart, %d\n".as_ptr() as *const u8, temp) };
return temp;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"restart complete\n".as_ptr() as *const u8) };
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ohci_restart_662: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ohci_hcd) -> i32>, *mut ()>(Some(ohci_restart)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_suspend")]
extern "C" fn ohci_suspend(hcd: *mut usb_hcd, do_wakeup: bool) -> i32 {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
let mut flags: u64 = 0;
let mut rc: i32 = 0;
// Disable irq emission and mark HW unaccessible. Use
// the spinlock to properly synchronize with possible pending
// RH suspend or resume activity.
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
ohci_writel!(ohci, -2147483648i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrdisable) });
ohci_readl!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrdisable) });
clear_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*hcd).flags) }) as *mut u64);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ohci).lock) }, flags);
unsafe { synchronize_irq((*hcd).irq) };
if do_wakeup && HCD_WAKEUP_PENDING!(hcd) != 0 {
ohci_resume(hcd, false);
rc = 16i32.wrapping_neg();
}
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ohci_suspend_663: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>, *mut ()>(Some(ohci_suspend)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_resume")]
extern "C" fn ohci_resume(hcd: *mut usb_hcd, hibernated: bool) -> i32 {
let ohci: *mut ohci_hcd = hcd_to_ohci(hcd);
let mut port: i32 = 0;
let mut need_reinit: bool = false;
set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*hcd).flags) }) as *mut u64);
// Make sure resume from hibernation re-enumerates everything
if hibernated {
ohci_usb_reset(ohci);
}
// See if the controller is already running or has been reset
unsafe {
(*ohci).hc_control = ohci_readl!(ohci, core::ptr::addr_of_mut!((*(*ohci).regs).control));
}
if (unsafe { (*ohci).hc_control }) & 316u32 != 0 {
need_reinit = true;
} else {
match (unsafe { (*ohci).hc_control }) & 192u32 {
0 | 128 => {
need_reinit = true;
}
_ => {
}
}
}
// If needed, reinitialize and suspend the root hub
if need_reinit {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ohci).lock) });
ohci_rh_resume(ohci);
ohci_rh_suspend(ohci, 0i32);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ohci).lock) });
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ohci_to_hcd(ohci as *const ohci_hcd)).self_.controller as *const device, c"powerup ports\n".as_ptr() as *const u8) };
}
port = 0;
while port < unsafe { (*ohci).num_ports } {
ohci_writel!(ohci, 256i32, unsafe { (*(*ohci).regs).roothub.portstatus.as_mut_ptr() as *mut u32 }.wrapping_offset(port as isize));
port = port.wrapping_add(1);
}
ohci_writel!(ohci, -2147483648i32, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrenable) });
ohci_readl__2!(ohci, unsafe { core::ptr::addr_of_mut!((*(*ohci).regs).intrenable) });
unsafe { msleep(20u32) };
}
// Normally just turn on port power and enable interrupts
unsafe { usb_hcd_resume_root_hub(hcd) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ohci_resume_664: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>, *mut ()>(Some(ohci_resume)) };

// Generic structure: This gets copied for platform drivers so that
// individual entries can be overridden as needed.
/// generic hardware linkage
/// basic lifecycle operations
/// managing i/o requests and associated device resources
/// scheduling support
/// root hub support
static ohci_hc_driver: __SlateConst<hc_driver> = __SlateConst({
let mut __t0: hc_driver = unsafe { core::mem::zeroed() };
__t0.description = unsafe { core::ptr::addr_of!(hcd_name) as *const u8 };
__t0.product_desc = (b"OHCI Host Controller\0".as_ptr() as *mut u8) as *const u8;
__t0.hcd_priv_size = 1160u64;
__t0.irq = Some(ohci_irq);
__t0.flags = 19i32;
__t0.reset = Some(ohci_setup);
__t0.start = Some(ohci_start);
__t0.stop = Some(ohci_stop);
__t0.shutdown = Some(ohci_shutdown);
__t0.get_frame_number = Some(ohci_get_frame);
__t0.urb_enqueue = Some(ohci_urb_enqueue);
__t0.urb_dequeue = Some(ohci_urb_dequeue);
__t0.endpoint_disable = Some(ohci_endpoint_disable);
__t0.hub_status_data = Some(ohci_hub_status_data);
__t0.hub_control = Some(ohci_hub_control);
__t0.bus_suspend = Some(ohci_bus_suspend);
__t0.bus_resume = Some(ohci_bus_resume);
__t0.start_port_reset = None;
__t0
});

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ohci_hcd.ohci_init_driver")]
extern "C" fn ohci_init_driver(drv: *mut hc_driver, over: *const ohci_driver_overrides) {
// Copy the generic table to drv and then apply the overrides
unsafe {
*drv = ohci_hc_driver.0;
}
if !over.is_null() {
unsafe {
(*drv).product_desc = (*over).product_desc;
}
let __v1: u64 = unsafe { (*drv).hcd_priv_size };
unsafe {
(*drv).hcd_priv_size = __v1.wrapping_add((*over).extra_priv_size);
}
if (unsafe { (*over).reset }) != None {
unsafe {
(*drv).reset = (*over).reset;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ohci_init_driver_665: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hc_driver, *const ohci_driver_overrides)>, *mut ()>(Some(ohci_init_driver)) };

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_666: [u8; 51] = [111, 104, 99, 105, 95, 104, 99, 100, 46, 97, 117, 116, 104, 111, 114, 61, 82, 111, 109, 97, 110, 32, 87, 101, 105, 115, 115, 103, 97, 101, 114, 98, 101, 114, 44, 32, 68, 97, 118, 105, 100, 32, 66, 114, 111, 119, 110, 101, 108, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_667: [u8; 66] = [111, 104, 99, 105, 95, 104, 99, 100, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 85, 83, 66, 32, 49, 46, 49, 32, 39, 79, 112, 101, 110, 39, 32, 72, 111, 115, 116, 32, 67, 111, 110, 116, 114, 111, 108, 108, 101, 114, 32, 40, 79, 72, 67, 73, 41, 32, 68, 114, 105, 118, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_668: [u8; 40] = [111, 104, 99, 105, 95, 104, 99, 100, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 117, 115, 98, 47, 104, 111, 115, 116, 47, 111, 104, 99, 105, 45, 104, 99, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_669: [u8; 21] = [111, 104, 99, 105, 95, 104, 99, 100, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

#[unsafe(link_section = ".init.text")]
extern "C" fn ohci_hcd_mod_init() -> i32 {
let retval: i32 = 0;
if (unsafe { usb_disabled() }) != 0 {
return 19i32.wrapping_neg();
}
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017%s: block sizes: ed %zd td %zd\n".as_ptr() as *const u8, core::ptr::addr_of!(hcd_name) as *const u8, core::mem::size_of::<ed>() as u64, core::mem::size_of::<td>() as u64) };
}
__v0 = 0;
unsafe {
ohci_debug_root = debugfs_create_dir(c"ohci".as_ptr() as *const u8, usb_debug_root);
}
return retval;
// Error path
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ohci_hcd_mod_init_671: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(ohci_hcd_mod_init)) };

#[unsafe(link_section = ".exit.text")]
extern "C" fn ohci_hcd_mod_exit() {
unsafe { debugfs_remove(ohci_debug_root) };
}

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_ohci_hcd_mod_exit: Option<unsafe extern "C" fn()> = Some(ohci_hcd_mod_exit);

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_ed_schedule_14: [u8; 1] = *b"\0";

static __slate_asm_string_ed_schedule_15: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ohci-q.c\0";

static __slate_asm_string_ed_schedule_3: [u8; 1] = *b"\0";

static __slate_asm_string_ed_schedule_4: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ohci-q.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
