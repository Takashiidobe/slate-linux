
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
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! Copyright (C) 2007-2010 Advanced Micro Devices, Inc.
//! Author: Joerg Roedel <jroedel@suse.de>
//!         Leo Duran <leo.duran@amd.com>
//! This counts to 15. Any more, it will return 16th argument.
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
//! __struct_group() - Create a mirrored named and anonyomous struct
//!
//! @TAG: The tag name for the named sub-struct (usually empty)
//! @NAME: The identifier name of the mirrored sub-struct
//! @ATTRS: Any struct attributes (usually empty)
//! @MEMBERS: The member declarations for the mirrored structs
//!
//! Used to create an anonymous union of two structs with identical layout
//! and size: one anonymous and one named. The former's members can be used
//! normally without sub-struct naming, and the latter can be used to
//! reason about the start, end, and size of the group of struct members.
//! The named struct can also be explicitly tagged for layer reuse (C only),
//! as well as both having struct attributes appended.
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
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
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
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
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! upper_32_bits - return bits 32-63 of a number
//! @n: the number we're accessing
//!
//! A basic shift-right of a 64- or 32-bit quantity.  Use this to suppress
//! the "right shift count >= width of type" warning when that quantity is
//! 32-bits.
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! Sign Flag
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! writeable
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
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! signal pending
//! If !LOCKDEP, the value is meaningless
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
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
//! wait_queue_entry::flags
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
//! Docs say 0 for future compat.
//! Each PCI slot may be a combo card with its own bus.  4 IRQ pins per slot.
//! Helper macro to avoid gfp flags if they are the default one
//! Keep unconverted code working
//! number of env pointers
//! Generate SD flag indexes
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
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! AMD RD890 Chipset
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! The maximum expected number of CSI-2 data lanes.
//!
//! This number is not expected to ever have to be equal to or greater than the
//! number of bits in an unsigned long variable, but if it needs to be increased
//! above that limit, code will need to be adjusted accordingly.
//! DWORD 3
//! ***DANGEROUS*** don't add guard page
//! Monochrome Text Display
//! number of edd_info structs starting at EDDBUF
//! This is the maximum number of entries in struct boot_params::e820_table
//! (the zeropage), which is part of the x86 boot protocol ABI:
//! Various well-known legacy memory ranges in physical memory:
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Aperture control register bits.
//! Legacy last prefixes
//! REX2 M0
//! SNP Page State Change NAE event
//!   The VMGEXIT_PSC_MAX_ENTRY determines the size of the PSC structure, which
//!   is a local stack variable in set_pages_state(). Do not increase this value
//!   without evaluating the impact to stack usage.
//!
//!   Use VMGEXIT_PSC_MAX_COUNT in cases where the actual GHCB-defined max value
//!   is needed, such as when processing GHCB requests on the hypervisor side.
//! Architectural interrupt line count.
//! All input flush parameters should be in single page. The max flush
//! count is equal with how many entries of union hv_gpa_page_range can
//! be populated into the input parameter page.
//! HV_SPA_PAGE_RANGE_ADDITIONAL_PAGES_BITS
//! NOTE: Should use hv_input_set_partition_property_ex_header to compute this
//! size, but hv_input_get_partition_property_ex is identical so it suffices
//! For AVIC, the max index allowed for physical APIC ID table is 0xfe (254), as
//! 0xff is a broadcast to all CPUs, i.e. can't be targeted individually.
//! SNP CPUID table, as defined by the SNP Firmware ABI, Revision 0.9,
//! Section 8.14.2.6. Also noted there is the SNP firmware-enforced limit
//! of 64 entries per CPUID table.
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Number of chars for args
//! The per-cpu notes area is a list of notes terminated by a "NULL"
//! note header.  For kdump, the code in vmcore.c runs in the context
//! of the second kernel to combine them into one note.
//! The artificial cap on the number of segments passed to kexec_load.
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! For IS_ENABLED(CONFIG_CRASH_DUMP)
//! FIELD_GET() - extract a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_reg:  value of entire bitfield
//!
//! FIELD_GET() extracts the field specified by @_mask from the
//! bitfield passed in as @_reg by masking and shifting it down.
//! Generate the exported function signatures from iommu_pt.h
//! Maximum number of IOMMUs supported
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// struct_group() - Wrap a set of declarations in a mirrored struct
//
// @NAME: The identifier name of the mirrored sub-struct
// @MEMBERS: The member declarations for the mirrored structs
//
// Used to create an anonymous union of two structs with identical
// layout and size: one anonymous and one named. The former can be
// used normally without sub-struct naming, and the latter can be
// used to reason about the start, end, and size of the group of
// struct members.
// no tag
// no attrs
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// __TRAILING_OVERLAP() - Overlap a flexible-array member with trailing
// 			  members.
//
// Creates a union between a flexible-array member (FAM) in a struct and a set
// of additional members that would otherwise follow it.
//
// Beware that, as this helper encloses TYPE NAME and MEMBERS in the same
// union, designated initializers for MEMBERS may overwrite portions
// previously initialized through NAME.
//
// For example::
//
// 	struct flex {
// 		size_t count;
// 		u8 fam[];
// 	};
//
// 	struct composite {
// 		...
// 		__TRAILING_OVERLAP(struct flex, flex, fam, __packed,
// 			u8 data;
// 		);
// 	} __packed;
//
// 	static struct composite comp = {
// 		.flex = {
// 			.count = 1,
// 		},
// 		.data = 2,
// 	};
//
// In the example above, .flex and .data initialize different views of the same
// union storage. Since .data is initialized last, it _may_ overwrite portions
// previously initialized through .flex, leading to .flex.count being zeroed
// out.
//
// A couple of alternatives are shown below.
//
// a) Initialize only one view of the overlapped storage and assign the rest
//    at runtime::
//
// 	static struct composite comp = {
// 		.flex = {
// 			.count = 1,
// 		},
// 	};
//
// 	static void foo(void)
// 	{
// 		comp.data = 2;
// 		...
// 	}
//
// b) Alternatively, replace designated initializers with runtime assignments::
//
// 	static void foo(void)
// 	{
// 		struct composite comp;
//
// 		comp.flex.count = 1;
// 		comp.data = 2;
// 		...
// 	}
//
// Compiler Explorer test code: https://godbolt.org/z/voM4E36dT
//
// For another example of the above see commit 5e54510a9389 ("acpi: nfit:
// intel: avoid multiple -Wflex-array-member-not-at-end warnings")
//
// Link: https://git.kernel.org/linus/5e54510a9389caa9
//
// @TYPE: Flexible structure type name, including "struct" keyword.
// @NAME: Name for a variable to define.
// @FAM: The flexible-array member within @TYPE
// @ATTRS: Any struct attributes (usually empty)
// @MEMBERS: Trailing overlapping members.
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
// Make the optimizer believe the variable can be manipulated arbitrarily.
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
// Force a compilation error if a constant expression is not a power of 2
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
// Compile time check that field has an expected offset
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
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
// Combine multiple asm inline constraint args into a single arg for passing to
// another macro.
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
// informational
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// FW_BUG
// Add this to a message where you are sure the firmware is buggy or behaves
// really stupid or out of spec. Be aware that the responsible BIOS developer
// should be able to fix this issue or at least get a concrete idea of the
// problem by reading your message without the need of looking at the kernel
// code.
//
// Use it for definite and high priority BIOS bugs.
//
// FW_WARN
// Use it for not that clear (e.g. could the kernel messed up things already?)
// and medium priority BIOS bugs.
//
// FW_INFO
// Use this one if you want to tell the user or vendor about something
// suspicious, but generally harmless related to the firmware.
//
// Use it for information or very low priority BIOS bugs.
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
// pr_err - Print an error-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_ERR loglevel. It uses pr_fmt() to
// generate the format string.
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
// pr_notice - Print a notice-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_NOTICE loglevel. It uses pr_fmt() to
// generate the format string.
// pr_info - Print an info-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_INFO loglevel. It uses pr_fmt() to
// generate the format string.
// pr_cont - Continues a previous log message in the same line.
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_CONT loglevel. It should only be
// used when continuing a log message with no newline ('\n') enclosed. Otherwise
// it defaults back to KERN_DEFAULT loglevel.
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
// "pse" Page Size Extensions
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "acpi" ACPI via MSR
// "sse"
// "pdpe1gb" GB pages
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
// "cx16" CMPXCHG16B instruction
// "x2apic" X2APIC
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// Xen paravirtual guest
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
// "hwp" Intel Hardware P-states
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
// "11ap" Bad local APIC aka 11AP
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
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
// lower_32_bits - return bits 0-31 of a number
// @n: the number we're accessing
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
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
// Overflow Flag
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Generate SD flag bits
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
// Device class
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
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
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// Fixed by ACPI spec
// DWORD 5
// CGA Display
// EGA/VGA in Monochrome Mode
// EGA in Color Mode
// VGA+ in Color Mode
// VESA VGA in graphic mode
// ACER PICA-61 local S3 video
// MIPS Magnum 4000 G364 video
// Various SGI graphics hardware
// DEC TGA
// Sun frame buffer.
// Sun PCI based frame buffer.
// PowerMacintosh frame buffer.
// EFI graphic mode
// Frame buffer base is 64-bit
// max number of signatures to store
// K8 On-cpu GART registers
// 0x66
//
// LPFX1
// 0x67
// x86-64 REX prefix
// 0x4X
// AVX VEX prefixes
// 2-bytes VEX prefix
// 3-bytes VEX prefix
// EVEX prefix
// x86-64 REX2 prefix
// 0xD5
// AMD XOP prefix
// 0x8F
// Escape opcodes
// Group opcodes (1-16)
// Immediates
// VEX bit flags
// VEX3 Byte2
// VEX bit fields
// EVEX Byte1
// VEX3 Byte1
// VEX2.M always 1
// VEX3 Byte2, VEX2 Byte1
// XOP Byte2
// XOP Byte3
// for KVM_GET_LAPIC and KVM_SET_LAPIC
// Define the number of synthetic interrupt sources.
// Define the number of synthetic timers
// HvCallModifySparseGpaPageHostVisibility hypercall
// Data structures for HVCALL_MMIO_READ and HVCALL_MMIO_WRITE
// linux side we create long version of flags to use long bit ops on flags
// The maximum array size of HV_GENERIC_SET (vp_set) buffer
// partition id
// vp index
// PT_TRACE_* event enable flags
// The driver should setup its domain struct like
// 	union {
// 		struct iommu_domain domain;
// 		struct pt_iommu_xxx xx;
// 	};
// PT_IOMMU_CHECK_DOMAIN(struct mock_iommu_domain, xx.iommu, domain);
//
// Which creates an alias between driver_domain.domain and
// driver_domain.xx.iommu.domain. This is to avoid a mass rename of existing
// driver_domain.domain users.
// amdv1_mock is used by the iommufd selftest
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
macro_rules! ACPI_FAILURE {
($a:expr) => {
$a
};
}

/// @a is a power of 2 value
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

macro_rules! AMD_IOMMU_GUEST_IR_GA {
($x:expr) => {
($x) == 2 || ($x) == 1
};
}

macro_rules! AMD_IOMMU_GUEST_IR_VAPIC {
($x:expr) => {
($x) == 2
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__9 {
() => {
((core::mem::size_of::<[*const u8; 10]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__38!() as u64)
};
}

macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
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

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

/// Macros to insert the exception code classes
macro_rules! EXCEP_ENV {
($code:expr) => {
(($code) | 0) as u32
};
}

macro_rules! GENMASK_INPUT_CHECK__26 {
() => {
core::mem::size_of::<__SlateRecord_init_c_2444>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__52 {
() => {
core::mem::size_of::<__SlateRecord_init_c_4046>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__60 {
() => {
core::mem::size_of::<__SlateRecord_init_c_4054>() as i32
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__24 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__52!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__57 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__60!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__59 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__26!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK_ULL__7 {
($h:expr, $l:expr) => {
GENMASK_TYPE__57!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__15 {
($h:expr, $l:expr) => {
GENMASK_TYPE__24!(u64, $h, $l)
};
}

macro_rules! GENMASK__36 {
($h:expr, $l:expr) => {
GENMASK_TYPE__59!(u64, $h, $l)
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

macro_rules! IVRS_GET_SBDF_ID {
($seg:expr, $bus:expr, $dev:expr, $fn:expr) => {
(($seg) & 65535) << 16i32 | (($bus) & 255) << 8i32 | (($dev) & 31) << 3i32 | ($fn) & 7
};
}

/// to align the pointer to the (next) page boundary
macro_rules! PAGE_ALIGN {
($addr:expr) => {
ALIGN!($addr, 4096u64)
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

/// return bus from PCI devid = ((u16)bus_number) << 8) | devfn
macro_rules! PCI_BUS_NUM__3 {
($x:expr) => {
($x) >> 8i32 & 255
};
}

/// return bus from PCI devid = ((u16)bus_number) << 8) | devfn
macro_rules! PCI_BUS_NUM {
($x:expr) => {
(($x) as i32) >> 8i32 & 255
};
}

/// return bus from PCI devid = ((u16)bus_number) << 8) | devfn
macro_rules! PCI_BUS_NUM__2 {
($x:expr) => {
($x) >> 8i32 & 255
};
}

/// The PCI interface treats multi-function devices as independent
/// devices.  The slot/function address of each device is encoded
/// in a single byte as follows:
///
/// 	7:3 = slot
/// 	2:0 = function
macro_rules! PCI_DEVFN {
($slot:expr, $func:expr) => {
(($slot) & 31) << 3i32 | ($func) & 7
};
}

macro_rules! PCI_FUNC__3 {
($devfn:expr) => {
($devfn) & 7
};
}

macro_rules! PCI_FUNC {
($devfn:expr) => {
(($devfn) as i32) & 7
};
}

macro_rules! PCI_SEG_DEVID_TO_SBDF__2 {
($seg:expr, $devid:expr) => {
((($seg) as u32) & 65535) << 16i32 | ($devid) & 65535
};
}

macro_rules! PCI_SEG_DEVID_TO_SBDF {
($seg:expr, $devid:expr) => {
((($seg) as u32) & 65535) << 16i32 | (((($devid) as i32) & 65535) as u32)
};
}

macro_rules! PCI_SLOT__2 {
($devfn:expr) => {
($devfn) >> 3i32 & 31
};
}

macro_rules! PCI_SLOT {
($devfn:expr) => {
(($devfn) as i32) >> 3i32 & 31
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

macro_rules! _AC__3 {
() => {
18446744071562067968u64
};
}

macro_rules! _AC__2 {
() => {
16045481047390945280u64
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

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u64).wrapping_sub(1))
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

macro_rules! __bf_cast_unsigned {
($x:expr) => {
($x) as u64
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

macro_rules! __must_be_array__38 {
() => {
core::mem::size_of::<__SlateRecord_init_c_2278>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
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

macro_rules! __pa {
($x:expr) => {
__phys_addr!(($x) as u64)
};
}

macro_rules! __phys_addr {
($x:expr) => {
__phys_addr_nodebug($x)
};
}

macro_rules! __ratelimit__17 {
($state:expr) => {
unsafe { ___ratelimit($state, c"amd_iommu_restart_log".as_ptr() as *const u8) }
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
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
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
macro_rules! __sign_use__6 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u64>() as u64) < 4) as i32)) }
};
}

macro_rules! __sme_clr {
($x:expr) => {
$x
};
}

macro_rules! __sme_set {
($x:expr) => {
$x
};
}

/// We need to compute the minimum and maximum values representable in a given
/// type. These macros may also be useful elsewhere. It would seem more obvious
/// to do something like:
///
/// #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
/// #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
///
/// Unfortunately, the middle expressions, strictly speaking, have
/// undefined behaviour, and at least some versions of gcc warn about
/// the type_max expression (but not if -fsanitize=undefined is in
/// effect; in that case, the warning is deferred to runtime...).
///
/// The slightly excessive casting in type_min is to make sure the
/// macros also produce sensible values for the exotic type _Bool. [The
/// overflow checkers only almost work for _Bool, but that's
/// a-feature-not-a-bug, since people shouldn't be doing arithmetic on
/// _Bools. Besides, the gcc builtins don't allow _Bool* as third
/// argument.]
///
/// Idea stolen from
/// https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
/// credit to Christian Biere.
macro_rules! __type_half_max {
() => {
1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type!() as i32) as u64)
};
}

macro_rules! __type_max {
() => {
__type_half_max!().wrapping_sub(1).wrapping_add(__type_half_max!())
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok__7 {
($ux:expr, $uy:expr) => {
__sign_use__6!($ux) & __sign_use!($uy)
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

macro_rules! _static_cpu_has {
($bit:expr) => {
__static_cpu_has__const1(($bit) as u16) as i32
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

macro_rules! boot_cpu_has {
($bit:expr) => {
cpu_has!(unsafe { core::ptr::addr_of_mut!(boot_cpu_data) }, $bit)
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
macro_rules! container_of__23 {
($ptr:expr) => {
{
let mut __v: *mut amd_iommu_pci_seg;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__44!() as isize)) }) as *mut ()) as *mut amd_iommu_pci_seg;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__10!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__73 {
($ptr:expr) => {
{
let mut __v: *mut amd_iommu;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__148!() as isize)) }) as *mut ()) as *mut amd_iommu;
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
macro_rules! container_of__25 {
($ptr:expr) => {
{
let mut __v: *mut devid_map;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__41!() as isize)) }) as *mut ()) as *mut devid_map;
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
let mut __v: *mut acpihid_map_entry;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__32!() as isize)) }) as *mut ()) as *mut acpihid_map_entry;
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
macro_rules! container_of__95 {
($ptr:expr) => {
{
let mut __v: *mut ivhd_dte_flags;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__121!() as isize)) }) as *mut ()) as *mut ivhd_dte_flags;
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
macro_rules! container_of__32 {
($ptr:expr) => {
{
let mut __v: *mut unity_map_entry;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__47!() as isize)) }) as *mut ()) as *mut unity_map_entry;
__v
}
};
}

/// This is the default CPU features testing macro to use in code.
///
/// It is for detection of features which need kernel infrastructure to be
/// used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
/// if you want true runtime testing of CPU features, like in hypervisor code
/// where you are supporting a possible guest feature where host support for it
/// is not relevant.
macro_rules! cpu_feature_enabled {
($bit:expr) => {
_static_cpu_has!($bit)
};
}

macro_rules! cpu_has {
($c:expr, $bit:expr) => {
test_cpu_cap!($c, $bit) as i32
};
}

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2__4 {
($n:expr) => {
if unsafe { __slate_intrinsic_a39e427d3b6c2df4($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
1i32.wrapping_neg() < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
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

macro_rules! kvmalloc_node {
($size:expr, $flags:expr, $node:expr) => {
alloc_hooks!(unsafe { __kvmalloc_node_noprof($size, 1u64, $flags, $node) })
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
macro_rules! list_entry__20 {
($ptr:expr) => {
container_of__73!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__10 {
($ptr:expr) => {
container_of__29!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__12 {
($ptr:expr) => {
container_of__32!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__13 {
($ptr:expr) => {
container_of__25!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__14 {
($ptr:expr) => {
container_of__23!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__45 {
($ptr:expr) => {
container_of__95!($ptr)
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
macro_rules! list_first_entry__5 {
($ptr:expr) => {
list_entry__14!(unsafe { (*$ptr).next })
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
list_entry__20!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__32 {
($ptr:expr) => {
list_entry__45!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__23 {
($ptr:expr) => {
list_entry__10!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__33 {
($ptr:expr) => {
list_entry__12!(unsafe { (*$ptr).next })
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
list_entry__13!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__36 {
($pos:expr, $member:ident) => {
list_entry__45!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__18 {
($pos:expr, $member:ident) => {
list_entry__13!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__5 {
($pos:expr, $member:ident) => {
list_entry__14!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__29 {
($pos:expr, $member:ident) => {
list_entry__12!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__34 {
($pos:expr, $member:ident) => {
list_entry__10!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__3 {
($pos:expr, $member:ident) => {
list_entry__20!(unsafe { (*$pos).$member.next })
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__4 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__4!($type, $x, $y)
};
}

macro_rules! offsetof__32 {
() => {
core::mem::offset_of!(acpihid_map_entry, list) as u64
};
}

macro_rules! offsetof__10 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, iommu) as u64
};
}

macro_rules! offsetof__121 {
() => {
core::mem::offset_of!(ivhd_dte_flags, list) as u64
};
}

macro_rules! offsetof__41 {
() => {
core::mem::offset_of!(devid_map, list) as u64
};
}

macro_rules! offsetof__44 {
() => {
core::mem::offset_of!(amd_iommu_pci_seg, list) as u64
};
}

macro_rules! offsetof__47 {
() => {
core::mem::offset_of!(unity_map_entry, list) as u64
};
}

macro_rules! offsetof__148 {
() => {
core::mem::offset_of!(amd_iommu, list) as u64
};
}

macro_rules! release_mem_region {
($start:expr, $n:expr) => {
unsafe { __release_region(core::ptr::addr_of_mut!(iomem_resource), $start, $n) };
};
}

macro_rules! request_mem_region {
($start:expr, $n:expr, $name:expr) => {
unsafe { __request_region(core::ptr::addr_of_mut!(iomem_resource), $start, $n, ($name) as *const u8, 0i32) }
};
}

/// roundup_pow_of_two - round the given value up to nearest power of two
/// @n: parameter
///
/// round the given value up to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! roundup_pow_of_two {
($n:expr) => {
if unsafe { __slate_intrinsic_a39e427d3b6c2df4($n) } { if ($n) == 1 { 1u64 } else { 1u64 << ilog2__4!($n.wrapping_sub(1)).wrapping_add(1) } } else { __roundup_pow_of_two(($n) as u64) }
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

macro_rules! test_cpu_cap {
($c:expr, $bit:expr) => {
arch_test_bit(($bit) as u64, ((unsafe { (*$c).__slate_anon_9.x86_capability.as_mut_ptr() as *mut u32 }) as *mut u64) as *const u64)
};
}

macro_rules! type_max {
() => {
__type_max!()
};
}

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
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

unsafe extern "C" {
    static mut iomem_resource: resource;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut boot_cpu_data: cpuinfo_x86;
    static mut x86_init: x86_init_ops;
    static mut x86_platform: x86_platform_ops;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut x86_vector_domain: *mut irq_domain;
    static mut force_iommu: i32;
    static mut no_iommu: i32;
    static mut iommu_detected: i32;
    static mut apic: *mut apic;
    static mut nr_ioapics: i32;
    static mut elfcorehdr_addr: u64;
    static mut amd_iommu_ops: iommu_ops;
    static mut ioapic_map: list_head;
    static mut hpet_map: list_head;
    static mut acpihid_map: list_head;
    static mut amd_iommu_max_glx_val: i32;
    static mut irte_32_ops: amd_irte_ops;
    static mut irte_128_ops: amd_irte_ops;
    static mut irq_remapping_enabled: i32;
    static mut amd_iommu_irq_ops: irq_remap_ops;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __request_region(arg0: *mut resource, arg1: u64, arg2: u64, arg3: *const u8, arg4: i32) -> *mut resource;
    fn __release_region(arg0: *mut resource, arg1: u64, arg2: u64);
    fn iomem_get_mapping() -> *mut address_space;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __compiletime_assert_12();
    fn __compiletime_assert_13();
    fn __compiletime_assert_14();
    fn __compiletime_assert_17();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_41();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn strchr(arg0: *const u8, arg1: i32) -> *mut u8;
    fn strsep(arg0: *mut *mut u8, arg1: *const u8) -> *mut u8;
    fn strcspn(arg0: *const u8, arg1: *const u8) -> u64;
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn sscanf(arg0: *const u8, arg1: *const u8, ...) -> i32;
    fn irq_to_desc(arg0: u32) -> *mut irq_desc;
    fn sysfs_emit(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_415();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kvmalloc_node_noprof(arg0: u64, arg1: u64, arg2: u32, arg3: i32) -> *mut ();
    fn kvfree(arg0: *const ());
    fn request_threaded_irq(arg0: u32, arg1: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, arg2: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, arg3: u64, arg4: *const u8, arg5: *mut ()) -> i32;
    fn ioremap_encrypted(arg0: u64, arg1: u64) -> *mut ();
    fn ioremap(arg0: u64, arg1: u64) -> *mut ();
    fn iounmap(arg0: *mut ());
    fn memcpy_toio(arg0: *mut (), arg1: *const (), arg2: u64);
    fn memremap(arg0: u64, arg1: u64, arg2: u64) -> *mut ();
    fn memunmap(arg0: *mut ());
    fn pci_get_domain_bus_and_slot(arg0: i32, arg1: u32, arg2: u32) -> *mut pci_dev;
    fn pci_read_config_dword(arg0: *const pci_dev, arg1: i32, arg2: *mut u32) -> i32;
    fn pci_write_config_dword(arg0: *const pci_dev, arg1: i32, arg2: u32) -> i32;
    fn pci_enable_device(arg0: *mut pci_dev) -> i32;
    fn pci_disable_msi(arg0: *mut pci_dev);
    fn pci_enable_msi(arg0: *mut pci_dev) -> i32;
    fn pci_request_acs();
    fn acpi_format_exception(arg0: u32) -> *const u8;
    fn acpi_get_table(arg0: *mut u8, arg1: u32, arg2: *mut *mut acpi_table_header) -> u32;
    fn acpi_put_table(arg0: *mut acpi_table_header);
    fn register_syscore(arg0: *mut syscore);
    fn irq_cfg(arg0: u32) -> *mut irq_cfg;
    fn irqd_cfg(arg0: *mut irq_data) -> *mut irq_cfg;
    fn handle_edge_irq(arg0: *mut irq_desc);
    fn irq_chip_ack_parent(arg0: *mut irq_data);
    fn irq_chip_retrigger_hierarchy(arg0: *mut irq_data) -> i32;
    fn __irq_set_handler(arg0: u32, arg1: Option<unsafe extern "C" fn(*mut irq_desc)>, arg2: i32, arg3: *const u8);
    fn __irq_domain_alloc_fwnode(arg0: u32, arg1: i32, arg2: *const u8, arg3: *mut u64, arg4: *mut fwnode_handle) -> *mut fwnode_handle;
    fn irq_domain_free_fwnode(arg0: *mut fwnode_handle);
    fn irq_domain_instantiate(arg0: *const irq_domain_info) -> *mut irq_domain;
    fn irq_domain_remove(arg0: *mut irq_domain);
    fn irq_domain_get_irq_data(arg0: *mut irq_domain, arg1: u32) -> *mut irq_data;
    fn __irq_domain_alloc_irqs(arg0: *mut irq_domain, arg1: i32, arg2: u32, arg3: i32, arg4: *mut (), arg5: bool, arg6: *const irq_affinity_desc) -> i32;
    fn irq_domain_free_irqs(arg0: u32, arg1: u32);
    fn irq_domain_free_irqs_top(arg0: *mut irq_domain, arg1: u32, arg2: u32);
    fn irq_domain_alloc_irqs_parent(arg0: *mut irq_domain, arg1: u32, arg2: u32, arg3: *mut ()) -> i32;
    fn init_irq_alloc_info(arg0: *mut irq_alloc_info, arg1: *const cpumask);
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn read_pci_config(arg0: u8, arg1: u8, arg2: u8, arg3: u8) -> u32;
    fn read_pci_config_16(arg0: u8, arg1: u8, arg2: u8, arg3: u8) -> u16;
    fn mpc_ioapic_id(arg0: i32) -> i32;
    fn mp_find_ioapic(arg0: u32) -> i32;
    fn set_memory_4k(arg0: u64, arg1: i32) -> i32;
    fn iommu_device_register(arg0: *mut iommu_device, arg1: *const iommu_ops, arg2: *mut device) -> i32;
    fn iommu_device_unregister(arg0: *mut iommu_device);
    fn iommu_device_sysfs_add(arg0: *mut iommu_device, arg1: *mut device, arg2: *mut *const attribute_group, arg3: *const u8, ...) -> i32;
    fn iommu_device_sysfs_remove(arg0: *mut iommu_device);
    fn iommu_set_dma_strict();
    fn amd_iommu_int_thread(arg0: i32, arg1: *mut ()) -> u32;
    fn amd_iommu_int_thread_evtlog(arg0: i32, arg1: *mut ()) -> u32;
    fn amd_iommu_int_thread_pprlog(arg0: i32, arg1: *mut ()) -> u32;
    fn amd_iommu_int_thread_galog(arg0: i32, arg1: *mut ()) -> u32;
    fn amd_iommu_set_rlookup_table(arg0: *mut amd_iommu, arg1: u16);
    fn amd_iommu_init_identity_domain();
    fn amd_iommu_iopf_init(arg0: *mut amd_iommu) -> i32;
    fn amd_iommu_iopf_uninit(arg0: *mut amd_iommu);
    fn amd_iommu_alloc_ppr_log(arg0: *mut amd_iommu) -> i32;
    fn amd_iommu_free_ppr_log(arg0: *mut amd_iommu);
    fn amd_iommu_enable_ppr_log(arg0: *mut amd_iommu);
    fn amd_iommu_flush_all_caches(arg0: *mut amd_iommu);
    fn amd_iommu_create_irq_domain(arg0: *mut amd_iommu) -> i32;
    fn amd_iommu_pdom_id_reserve(arg0: u16, arg1: u32) -> i32;
    fn amd_iommu_pdom_id_destroy();
    fn amd_iommu_apply_ivrs_quirks();
    fn get_dev_table(arg0: *mut amd_iommu) -> *mut dev_table_entry;
    fn iommu_alloc_pages_node_sz(arg0: i32, arg1: u32, arg2: u64) -> *mut ();
    fn iommu_free_pages(arg0: *mut ());
    fn __compiletime_assert_634();
    fn __compiletime_assert_635();
    fn __compiletime_assert_636();
    fn __compiletime_assert_637();
    fn __compiletime_assert_638();
    fn __compiletime_assert_639();
    fn __compiletime_assert_640();
    fn __compiletime_assert_641();
    fn __compiletime_assert_642();
    fn __compiletime_assert_643();
    fn __compiletime_assert_644();
    fn __compiletime_assert_645();
    fn __compiletime_assert_646();
    fn __compiletime_assert_647();
    fn __compiletime_assert_648();
    fn __compiletime_assert_649();
    fn __compiletime_assert_652();
    fn __compiletime_assert_656();
    fn __compiletime_assert_657();
    fn __compiletime_assert_658();
    fn __compiletime_assert_659();
    fn __compiletime_assert_660();
    fn __compiletime_assert_662();
    fn __compiletime_assert_663();
    fn __compiletime_assert_664();
    fn __compiletime_assert_665();
    fn __compiletime_assert_666();
    fn __compiletime_assert_669();
    fn __compiletime_assert_670();
    fn __compiletime_assert_671();
    fn __compiletime_assert_672();
    fn __compiletime_assert_673();
    fn __compiletime_assert_674();
    fn __compiletime_assert_678();
    fn __compiletime_assert_679();
    fn __compiletime_assert_680();
    fn __compiletime_assert_681();
    fn __compiletime_assert_682();
    fn __compiletime_assert_683();
    fn __compiletime_assert_684();
    fn __compiletime_assert_685();
    fn __compiletime_assert_686();
    fn __compiletime_assert_687();
    fn __compiletime_assert_688();
    fn __compiletime_assert_689();
    fn __compiletime_assert_690();
    fn __compiletime_assert_691();
    fn __compiletime_assert_692();
    fn __compiletime_assert_693();
    fn __compiletime_assert_694();
    fn __compiletime_assert_695();
    fn __compiletime_assert_696();
    fn __compiletime_assert_697();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
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


// modpost check for section mismatches during the kernel build.
// A section mismatch happens when there are references from a
// code or data section to an init section (both code or data).
// The init sections are (for most archs) discarded by the kernel
// when early init has completed so all such references are potential bugs.
// For exit sections the same issue exists.
//
// The following markers are used for the cases where the reference to
// the *init / *exit section (code or data) is valid and will teach
// modpost not to issue a warning.  Intended semantics is that a code or
// data tagged __ref* can reference code or data from init section without
// producing a warning (of course, no warning does not mean code is
// correct, so optimally document why the __ref is needed and why it's OK).
//
// The markers follow same syntax rules as __init / __initdata.
#[repr(C)]
#[derive(Clone, Copy)]
struct obs_kernel_param {
    str: *const u8,
    setup_func: Option<unsafe extern "C" fn(*mut u8) -> i32>,
    early: i32,
}


// Only for really core code.  See moduleparam.h for the normal way.
//
// Force the alignment so the compiler doesn't space elements of the
// obs_kernel_param "array" too far apart in .init.setup.
// NOTE: __setup functions return values:
// @fn returns 1 (or non-zero) if the option argument is "handled"
// and returns 0 if the option argument is "not handled".
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
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
struct static_call_site {
    addr: i32,
    key: i32,
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
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_12() };
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
unsafe { __compiletime_assert_13() };
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
unsafe { __compiletime_assert_14() };
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
unsafe { __compiletime_assert_17() };
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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

fn list_is_first(list: *const list_head, head: *const list_head) -> i32 {
((unsafe { (*list).prev }) == (head as *mut list_head)) as i32
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_41() };
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
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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

#[inline(always)]
fn arch_test_bit(nr: u64, addr: *const u64) -> bool {
let __v0: i32 = if unsafe { __slate_intrinsic_f217c4720f04f6db(nr) } {
constant_test_bit(nr as i64, addr) as i32
} else {
variable_test_bit(nr as i64, addr) as i32
};
__v0 != 0
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
fn fls(mut x: u32) -> i32 {
let mut r: i32 = 0;
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(x) } {
return if x != 0 { 32i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRL won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before, except that the
// top 32 bits will be cleared.
//
// We cannot do this on 32 bits because at the very least some
// 486 CPUs did not behave this way.
unsafe {
let __t0: u32;
core::arch::asm!("bsrl {1:e},{0:e}", inlateout(reg) (1 as i32).wrapping_neg() as u32 => __t0, in(reg) x as u32, options(pure, nomem, nostack, att_syntax));
r = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
r.wrapping_add(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn fls64(mut x: u64) -> i32 {
let mut bitpos: i32 = 1i32.wrapping_neg();
if unsafe { __slate_intrinsic_f217c4720f04f6db(x) } {
return if x != 0 { 64i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRQ won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before.
unsafe {
let __t0: u32;
core::arch::asm!("bsrq {1:r},{0:r}", inlateout(reg) bitpos as u32 => __t0, in(reg) x as u64, options(pure, nomem, nostack, att_syntax));
bitpos = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
bitpos.wrapping_add(1)
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


// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
fn fls_long(l: u64) -> u32 {
if ((const { core::mem::size_of::<u64>() }) as u64) == 4 {
return fls(l as u32) as u32;
}
fls64(l) as u32
}

// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
#[repr(C)]
#[derive(Clone, Copy)]
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
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
struct io_bitmap {
}


#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
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


fn __roundup_pow_of_two(n: u64) -> u64 {
1u64 << fls_long(n.wrapping_sub(1))
}

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


fn init_llist_head(list: *mut llist_head) {
unsafe {
(*list).first = core::ptr::null_mut::<llist_node>();
}
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


#[repr(C)]
#[derive(Clone, Copy)]
struct restart_block {
    arch_data: u64,
    r#fn: Option<unsafe extern "C" fn(*mut restart_block) -> i64>,
    __slate_anon_2: __SlateRecord_restart_block_h_29,
}


fn cc_platform_has(attr: u32) -> bool {
false
}

fn cc_platform_clear(attr: u32) {
}

#[inline(always)]
fn __phys_addr_nodebug(mut x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__3!());
// use the carry flag to determine if x was < __START_KERNEL_map
x = y.wrapping_add(if x > y { unsafe { phys_base } } else { _AC__3!().wrapping_sub(unsafe { page_offset_base }) });
x
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_cs {
    __slate_bits_0: __slate_bits::__SlateBits142U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_ss {
    __slate_bits_0: __slate_bits::__SlateBits143U0,
}


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


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x0_0 {
    __slate_bits_0: __slate_bits::__SlateBits147U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_111 {
    hw_cpu_type: u32,
    __slate_anon_1: __SlateRecord_processor_h_113,
    __slate_anon_2: __SlateRecord_processor_h_119,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits175U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_119 {
    __slate_bits_0: __slate_bits::__SlateBits176U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_136 {
    __slate_anon_0: __SlateRecord_processor_h_142,
    x86_vfm: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_142 {
    x86_model: u8,
    x86: u8,
    x86_vendor: u8,
    x86_reserved: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_154 {
    intel_platform_id: u8,
    amd_unused: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_177 {
    x86_capability: [u32; 24],
    x86_capability_alignment: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x1_0 {
    __slate_bits_0: __slate_bits::__SlateBits148U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct leaf_parse_info {
    nr_entries: u32,
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct cpuid_leaves {
    leaf_0x0_0: [leaf_0x0_0; 1],
    leaf_0x0_0_info: leaf_parse_info,
    leaf_0x1_0: [leaf_0x1_0; 1],
    leaf_0x1_0_info: leaf_parse_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuid_table {
    leaves: cpuid_leaves,
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
struct futex_private_hash {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits169U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits170U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuinfo_topology {
    apicid: u32,
    initial_apicid: u32,
    pkg_id: u32,
    die_id: u32,
    cu_id: u32,
    core_id: u32,
    logical_pkg_id: u32,
    logical_die_id: u32,
    logical_core_id: u32,
    amd_node_id: u32,
    llc_id: u32,
    l2c_id: u32,
    __slate_anon_12: __SlateRecord_processor_h_111,
    cpu_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuinfo_x86 {
    __slate_anon_0: __SlateRecord_processor_h_136,
    x86_stepping: u8,
    __slate_anon_2: __SlateRecord_processor_h_154,
    x86_tlbsize: i32,
    vmx_capability: [u32; 5],
    x86_virt_bits: u8,
    x86_phys_bits: u8,
    extended_cpuid_level: u32,
    cpuid_level: i32,
    __slate_anon_9: __SlateRecord_processor_h_177,
    x86_vendor_id: [u8; 16],
    x86_model_id: [u8; 64],
    topo: cpuinfo_topology,
    cpuid: cpuid_table,
    x86_cache_size: u32,
    x86_cache_alignment: i32,
    x86_cache_max_rmid: i32,
    x86_cache_occ_scale: i32,
    x86_cache_mbm_width_offset: i32,
    x86_power: i32,
    loops_per_jiffy: u64,
    ppin: u64,
    x86_clflush_size: u16,
    booted_cores: u16,
    cpu_index: u16,
    smt_active: bool,
    microcode: u32,
    x86_cache_bits: u8,
    __slate_bits_0: __slate_bits::__SlateBits146U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
    __slate_bits_0: __slate_bits::__SlateBits171U0,
    pkru: u32,
}


#[inline(always)]
fn __static_cpu_has__const1(bit: u16) -> bool {
unsafe {
let mut __slate_asm_target: usize = 0usize;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\tjmp 6f\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 3*32+21)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tjmp {4}\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte {0}\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\t\n775:\n.popsection\n.pushsection .altinstr_aux,\"ax\"\n6:\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n testb ${1}, {2}+114(%rip)\n jnz {3}\n jmp {4}\n.popsection\n", const 528 as u16, const 1 as i32, sym boot_cpu_data, label {
__slate_asm_target = 1usize;
}, label {
__slate_asm_target = 2usize;
}, options(nostack, att_syntax));
if __slate_asm_target != 1usize {
if __slate_asm_target == 2usize {
return false;
}
}
return true;
}
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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
}


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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
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
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
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
struct workqueue_struct {
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
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
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
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
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
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
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
struct gendisk {
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
    __slate_bits_0: __slate_bits::__SlateBits307U0,
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
    __slate_bits_0: __slate_bits::__SlateBits72U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits72U1,
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
    __slate_bits_2: __slate_bits::__SlateBits72U2,
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct uprobe_xol_ops {
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
struct xol_area {
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
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct mem_cgroup {
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
struct __SlateRecord_mm_types_h_41 {
    f: u64,
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
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
    __slate_bits_0: __slate_bits::__SlateBits425U0,
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
struct x86_init_mpparse {
    setup_ioapic_ids: Option<unsafe extern "C" fn()>,
    find_mptable: Option<unsafe extern "C" fn()>,
    early_parse_smp_cfg: Option<unsafe extern "C" fn()>,
    parse_smp_cfg: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_resources {
    probe_roms: Option<unsafe extern "C" fn()>,
    reserve_resources: Option<unsafe extern "C" fn()>,
    memory_setup: Option<unsafe extern "C" fn() -> *mut u8>,
    dmi_setup: Option<unsafe extern "C" fn()>,
    realmode_limit: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_irqs {
    pre_vector_init: Option<unsafe extern "C" fn()>,
    intr_init: Option<unsafe extern "C" fn()>,
    intr_mode_select: Option<unsafe extern "C" fn()>,
    intr_mode_init: Option<unsafe extern "C" fn()>,
    create_pci_msi_domain: Option<unsafe extern "C" fn() -> *mut irq_domain>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_oem {
    arch_setup: Option<unsafe extern "C" fn()>,
    banner: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_paging {
    pagetable_init: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_timers {
    setup_percpu_clockev: Option<unsafe extern "C" fn()>,
    timer_init: Option<unsafe extern "C" fn()>,
    wallclock_init: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_iommu {
    iommu_init: Option<unsafe extern "C" fn() -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_pci {
    arch_init: Option<unsafe extern "C" fn() -> i32>,
    init: Option<unsafe extern "C" fn() -> i32>,
    init_irq: Option<unsafe extern "C" fn()>,
    fixup_irqs: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_hyper_init {
    init_platform: Option<unsafe extern "C" fn()>,
    guest_late_init: Option<unsafe extern "C" fn()>,
    x2apic_available: Option<unsafe extern "C" fn() -> bool>,
    msi_ext_dest_id: Option<unsafe extern "C" fn() -> bool>,
    init_mem_mapping: Option<unsafe extern "C" fn()>,
    init_after_bootmem: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_acpi {
    set_root_pointer: Option<unsafe extern "C" fn(u64)>,
    get_root_pointer: Option<unsafe extern "C" fn() -> u64>,
    reduced_hw_early_init: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_guest {
    enc_status_change_prepare: Option<unsafe extern "C" fn(u64, i32, bool) -> i32>,
    enc_status_change_finish: Option<unsafe extern "C" fn(u64, i32, bool) -> i32>,
    enc_tlb_flush_required: Option<unsafe extern "C" fn(bool) -> bool>,
    enc_cache_flush_required: Option<unsafe extern "C" fn() -> bool>,
    enc_kexec_begin: Option<unsafe extern "C" fn()>,
    enc_kexec_finish: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_ops {
    resources: x86_init_resources,
    mpparse: x86_init_mpparse,
    irqs: x86_init_irqs,
    oem: x86_init_oem,
    paging: x86_init_paging,
    timers: x86_init_timers,
    iommu: x86_init_iommu,
    pci: x86_init_pci,
    hyper: x86_hyper_init,
    acpi: x86_init_acpi,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_legacy_devices {
    pnpbios: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_legacy_features {
    i8042: u32,
    rtc: i32,
    warm_reset: i32,
    no_vga: i32,
    reserve_bios_regions: i32,
    devices: x86_legacy_devices,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_hyper_runtime {
    pin_vcpu: Option<unsafe extern "C" fn(i32)>,
    sev_es_hcall_prepare: Option<unsafe extern "C" fn(*mut ghcb, *mut pt_regs)>,
    sev_es_hcall_finish: Option<unsafe extern "C" fn(*mut ghcb, *mut pt_regs) -> bool>,
    is_private_mmio: Option<unsafe extern "C" fn(u64) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_platform_ops {
    calibrate_cpu: Option<unsafe extern "C" fn() -> u64>,
    calibrate_tsc: Option<unsafe extern "C" fn() -> u64>,
    get_wallclock: Option<unsafe extern "C" fn(*mut timespec64)>,
    set_wallclock: Option<unsafe extern "C" fn(*const timespec64) -> i32>,
    iommu_shutdown: Option<unsafe extern "C" fn()>,
    is_untracked_pat_range: Option<unsafe extern "C" fn(u64, u64) -> bool>,
    nmi_init: Option<unsafe extern "C" fn()>,
    get_nmi_reason: Option<unsafe extern "C" fn() -> u8>,
    save_sched_clock_state: Option<unsafe extern "C" fn()>,
    restore_sched_clock_state: Option<unsafe extern "C" fn()>,
    apic_post_init: Option<unsafe extern "C" fn()>,
    legacy: x86_legacy_features,
    set_legacy_features: Option<unsafe extern "C" fn()>,
    realmode_reserve: Option<unsafe extern "C" fn()>,
    realmode_init: Option<unsafe extern "C" fn()>,
    hyper: x86_hyper_runtime,
    guest: x86_guest,
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
struct user_namespace {
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
    __slate_bits_0: __slate_bits::__SlateBits492U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
struct attribute {
    name: *const u8,
    mode: u16,
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
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits494U0,
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
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
    __slate_bits_0: __slate_bits::__SlateBits519U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits519U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits519U2,
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
    __slate_bits_3: __slate_bits::__SlateBits519U3,
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
struct wake_irq {
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
struct dev_pm_qos {
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
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
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


fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
    __slate_bits_0: __slate_bits::__SlateBits310U0,
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
unsafe { __compiletime_assert_388() };
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_409() };
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
    __slate_bits_0: __slate_bits::__SlateBits657U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
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
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
unsafe { __compiletime_assert_415() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("416: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 416b - ., 3; .popsection\n# {0}", const 416 as i32, options(nostack, att_syntax));
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
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
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
struct inode_state_flags {
    __state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct device_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group {
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
struct msi_device_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_108 {
    show: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *mut u8) -> i64>,
    show_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *mut u8) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_114 {
    store: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *const u8, u64) -> i64>,
    store_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *const u8, u64) -> i64>,
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
struct kernel_param_ops {
    flags: u32,
    set: Option<unsafe extern "C" fn(*const u8, *const kernel_param) -> i32>,
    get: Option<unsafe extern "C" fn(*mut u8, *const kernel_param) -> i32>,
    free: Option<unsafe extern "C" fn(*mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_interrupt_h_125 {
    dev_id: *mut (),
    percpu_dev_id: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
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
struct hotplug_slot {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mod_tree_node {
    r#mod: *mut module,
    node: latch_tree_node,
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
struct module_memory {
    base: *mut (),
    is_rox: bool,
    size: u32,
    mtn: mod_tree_node,
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
}


// DEVICE_ATTR - Define a device attribute.
// @_name: Attribute name.
// @_mode: File mode.
// @_show: Show handler. Optional, but mandatory if attribute is readable.
// @_store: Store handler. Optional, but mandatory if attribute is writable.
//
// Convenience macro for defining a struct device_attribute.
//
// For example, ``DEVICE_ATTR(foo, 0644, foo_show, foo_store);`` expands to:
//
// .. code-block:: c
//
// 	struct device_attribute dev_attr_foo = {
// 		.attr	= { .name = "foo", .mode = 0644 },
// 		.show	= foo_show,
// 		.store	= foo_store,
// 	};
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
    __slate_bits_0: __slate_bits::__SlateBits516U0,
}


fn dev_to_node(dev: *mut device) -> i32 {
unsafe { (*dev).numa_node }
}

fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irqaction {
    handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    __slate_anon_1: __SlateRecord_interrupt_h_125,
    affinity: *const cpumask,
    next: *mut irqaction,
    thread_fn: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    thread: *mut task_struct,
    secondary: *mut irqaction,
    irq: u32,
    flags: u32,
    thread_flags: u64,
    thread_mask: u64,
    name: *const u8,
    dir: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_notify {
    irq: u32,
    kref: kref,
    work: work_struct,
    notify: Option<unsafe extern "C" fn(*mut irq_affinity_notify, *const cpumask)>,
    release: Option<unsafe extern "C" fn(*mut kref)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_desc {
    mask: cpumask,
    __slate_bits_0: __slate_bits::__SlateBits768U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hw_irq_h_82 {
    ioapic: ioapic_alloc_info,
    uv: uv_alloc_info,
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

fn readq(mut addr: *const ()) -> u64 {
let mut ret: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq ({1:r}),{0:r}", lateout(reg) __t0, in(reg) addr as *mut u64, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
ret
}

fn writeq(mut val: u64, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movq {0:r},({1:r})", in(reg) val as u64, in(reg) addr as *mut u64, options(nostack, att_syntax));
}
}

// Let people know that we have them
fn virt_to_phys(address: *mut ()) -> u64 {
__pa!(address)
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
union msi_instance_cookie {
    value: u64,
    ptr: *mut (),
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
    __slate_bits_0: __slate_bits::__SlateBits777U0,
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
    __slate_bits_0: __slate_bits::__SlateBits770U0,
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
    __slate_bits_1: __slate_bits::__SlateBits770U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits770U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits770U3,
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
    __slate_bits_4: __slate_bits::__SlateBits770U4,
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
union __SlateRecord_msi_h_26 {
    __slate_anon_0: __SlateRecord_msi_h_26_2,
    dmar_subhandle: u32,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_26_2 {
    __slate_bits_0: __slate_bits::__SlateBits846U0,
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
    __slate_bits_0: __slate_bits::__SlateBits472U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_49 {
    __slate_anon_0: __SlateRecord_msi_h_49_2,
    __slate_anon_1: __SlateRecord_msi_h_49_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_2 {
    __slate_bits_0: __slate_bits::__SlateBits850U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_3 {
    __slate_bits_0: __slate_bits::__SlateBits851U0,
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
union __SlateRecord_msi_h_62 {
    address_lo: u32,
    arch_addr_lo: x86_msi_addr_lo,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_66 {
    address_hi: u32,
    arch_addr_hi: x86_msi_addr_hi,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_70 {
    data: u32,
    arch_data: x86_msi_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dynids {
    lock: spinlock,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_109 {
    msi_mask: u32,
    msix_ctrl: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits860U0,
    default_irq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_122 {
    mask_pos: u8,
    mask_base: *mut (),
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


// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_202 {
    pci: pci_msi_desc,
    data: msi_desc_data,
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
union __SlateRecord_vmalloc_h_53 {
    next: *mut vm_struct,
    llnode: llist_node,
}


// Bits set in the VMA until the stack is in its final location
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
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ipc_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_table_header {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    asl_compiler_id: [u8; 4],
    asl_compiler_revision: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct syscore_ops {
    suspend: Option<unsafe extern "C" fn(*mut ()) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut ())>,
    shutdown: Option<unsafe extern "C" fn(*mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct syscore {
    node: list_head,
    ops: *const syscore_ops,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iova_bitmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_common_data {
    state_use_accessors: u32,
    node: u32,
    handler_data: *mut (),
    msi_desc: *mut msi_desc,
    affinity: [cpumask; 1],
    effective_affinity: [cpumask; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_data {
    mask: u32,
    irq: u32,
    hwirq: u64,
    common: *mut irq_common_data,
    chip: *mut irq_chip,
    domain: *mut irq_domain,
    parent_data: *mut irq_data,
    chip_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip {
    name: *const u8,
    irq_startup: Option<unsafe extern "C" fn(*mut irq_data) -> u32>,
    irq_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_enable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_disable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_unmask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_eoi: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_set_affinity: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask, bool) -> i32>,
    irq_pre_redirect: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_retrigger: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_set_type: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_set_wake: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_bus_lock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_bus_sync_unlock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_suspend: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_resume: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_pm_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_calc_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_print_chip: Option<unsafe extern "C" fn(*mut irq_data, *mut seq_file)>,
    irq_request_resources: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_release_resources: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_compose_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_write_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_get_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, *mut bool) -> i32>,
    irq_set_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, bool) -> i32>,
    irq_set_vcpu_affinity: Option<unsafe extern "C" fn(*mut irq_data, *mut ()) -> i32>,
    ipi_send_single: Option<unsafe extern "C" fn(*mut irq_data, u32)>,
    ipi_send_mask: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask)>,
    irq_nmi_setup: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_nmi_teardown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_force_complete_move: Option<unsafe extern "C" fn(*mut irq_data)>,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dma_cookie {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dma_msi_cookie {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irqstat {
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_redirect {
    work: irq_work,
    target_cpu: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irq_desc {
    irq_common_data: irq_common_data,
    irq_data: irq_data,
    kstat_irqs: *mut irqstat,
    handle_irq: Option<unsafe extern "C" fn(*mut irq_desc)>,
    action: *mut irqaction,
    status_use_accessors: u32,
    core_internal_state__do_not_mess_with_it: u32,
    depth: u32,
    wake_depth: u32,
    tot_count: u64,
    last_unhandled: u64,
    irq_count: u32,
    irqs_unhandled: u32,
    threads_handled: __SlateRecord_types_h_190,
    threads_handled_last: i32,
    lock: raw_spinlock,
    percpu_enabled: *mut cpumask,
    redirect: irq_redirect,
    affinity_hint: *const cpumask,
    affinity_notify: *mut irq_affinity_notify,
    pending_mask: [cpumask; 1],
    threads_oneshot: u64,
    threads_active: __SlateRecord_types_h_190,
    wait_for_threads: wait_queue_head,
    nr_actions: u32,
    no_suspend_depth: u32,
    cond_suspend_depth: u32,
    force_resume_depth: u32,
    dir: *mut proc_dir_entry,
    refcnt: __SlateRecord_types_h_202,
    rcu: callback_head,
    kobj: kobject,
    request_mutex: mutex,
    parent_irq: i32,
    owner: *mut module,
    name: *const u8,
    resend_node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ioapic_alloc_info {
    pin: i32,
    node: i32,
    __slate_bits_0: __slate_bits::__SlateBits824U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uv_alloc_info {
    limit: i32,
    blade: i32,
    offset: u64,
    name: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_alloc_info {
    r#type: u32,
    flags: u32,
    devid: u32,
    hwirq: u64,
    mask: *const cpumask,
    desc: *mut msi_desc,
    data: *mut (),
    __slate_anon_7: __SlateRecord_hw_irq_h_82,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_cfg {
    dest_apicid: u32,
    vector: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_regs {
    enable: u64,
    disable: u64,
    mask: u64,
    ack: u64,
    eoi: u64,
    r#type: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_type {
    chip: irq_chip,
    regs: irq_chip_regs,
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    r#type: u32,
    mask_cache_priv: u32,
    mask_cache: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_generic {
    lock: raw_spinlock,
    reg_base: *mut (),
    reg_readl: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    reg_writel: Option<unsafe extern "C" fn(u32, *mut ())>,
    suspend: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    resume: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    irq_base: u32,
    irq_cnt: u32,
    mask_cache: u32,
    wake_enabled: u32,
    wake_active: u32,
    num_ct: u32,
    private: *mut (),
    installed: u64,
    unused: u64,
    domain: *mut irq_domain,
    list: list_head,
    chip_types: [irq_chip_type; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iommu_h_234 {
    iova_cookie: *mut iommu_dma_cookie,
    msi_cookie: *mut iommu_dma_msi_cookie,
    iommufd_hwpt: *mut iommufd_hw_pagetable,
    __slate_anon_3: __SlateRecord_iommu_h_238,
    __slate_anon_4: __SlateRecord_iommu_h_242,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_hw_pagetable {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_h_238 {
    handler: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u64, i32, *mut ()) -> i32>,
    handler_token: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_h_242 {
    mm: *mut mm_struct,
    users: i32,
    next: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_chip_generic {
    irqs_per_chip: u32,
    num_chips: u32,
    irq_flags_to_clear: u32,
    irq_flags_to_set: u32,
    gc_flags: u32,
    exit: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    gc: [*mut irq_chip_generic; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_chip_generic_info {
    name: *const u8,
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    irqs_per_chip: u32,
    num_ct: u32,
    irq_flags_to_clear: u32,
    irq_flags_to_set: u32,
    gc_flags: u32,
    init: Option<unsafe extern "C" fn(*mut irq_chip_generic) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iommu_h_364 {
    pgsize: u64,
    pt: __SlateRecord_iommu_h_370,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_h_370 {
    leaf_levels_bitmap: u8,
    table_levels_bitmap: u8,
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
struct of_phandle_args {
    np: *mut device_node,
    args_count: i32,
    args: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec {
    fwnode: *mut fwnode_handle,
    param_count: i32,
    param: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec_info {
    flags: u64,
    affinity: *const cpumask,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_access {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_ops {
    r#match: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, u32) -> i32>,
    select: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, u32) -> i32>,
    map: Option<unsafe extern "C" fn(*mut irq_domain, u32, u64) -> i32>,
    unmap: Option<unsafe extern "C" fn(*mut irq_domain, u32)>,
    xlate: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, *const u32, u32, *mut u64, *mut u32) -> i32>,
    alloc: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32, *mut ()) -> i32>,
    free: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32)>,
    activate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data, bool) -> i32>,
    deactivate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data)>,
    translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
    get_fwspec_info: Option<unsafe extern "C" fn(*mut irq_fwspec, *mut irq_fwspec_info) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
    link: list_head,
    name: *const u8,
    ops: *const irq_domain_ops,
    host_data: *mut (),
    flags: u32,
    mapcount: u32,
    mutex: mutex,
    root: *mut irq_domain,
    fwnode: *mut fwnode_handle,
    bus_token: u32,
    gc: *mut irq_domain_chip_generic,
    dev: *mut device,
    pm_dev: *mut device,
    parent: *mut irq_domain,
    msi_parent_ops: *const msi_parent_ops,
    exit: Option<unsafe extern "C" fn(*mut irq_domain)>,
    hwirq_max: u64,
    revmap_size: u32,
    revmap_tree: xarray,
    revmap: [*mut irq_data; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_hwpt_paging {
}


fn irq_domain_alloc_named_fwnode(name: *const u8) -> *mut fwnode_handle {
unsafe { __irq_domain_alloc_fwnode(1u32, 0i32, name, core::ptr::null_mut::<u64>(), core::ptr::null_mut::<fwnode_handle>()) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_info {
    fwnode: *mut fwnode_handle,
    domain_flags: u32,
    size: u32,
    hwirq_max: u64,
    direct_max: i32,
    hwirq_base: u32,
    virq_base: u32,
    bus_token: u32,
    name_suffix: *const u8,
    ops: *const irq_domain_ops,
    host_data: *mut (),
    dev: *mut device,
    parent: *mut irq_domain,
    dgc_info: *mut irq_domain_chip_generic_info,
    init: Option<unsafe extern "C" fn(*mut irq_domain) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut irq_domain)>,
}


fn irq_domain_create_hierarchy(parent: *mut irq_domain, flags: u32, size: u32, fwnode: *mut fwnode_handle, ops: *const irq_domain_ops, host_data: *mut ()) -> *mut irq_domain {
let mut info: irq_domain_info = unsafe { core::mem::zeroed() };
let __v0: u32 = size;
info = {
let mut __t0: irq_domain_info = unsafe { core::mem::zeroed() };
__t0.fwnode = fwnode;
__t0.domain_flags = flags;
__t0.size = size;
__t0.hwirq_max = (if __v0 != 0 { __v0 } else { 4294967295u32 }) as u64;
__t0.ops = ops;
__t0.host_data = host_data;
__t0.parent = parent;
__t0
};
let d: *mut irq_domain = unsafe { irq_domain_instantiate(core::ptr::addr_of!(info)) };
if IS_ERR(d as *const ()) { core::ptr::null_mut::<irq_domain>() } else { d }
}

fn irq_domain_alloc_irqs(domain: *mut irq_domain, nr_irqs: u32, node: i32, arg: *mut ()) -> i32 {
unsafe { __irq_domain_alloc_irqs(domain, 1i32.wrapping_neg(), nr_irqs, node, arg, false, core::ptr::null::<irq_affinity_desc>()) }
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct x86_msi_data {
    __slate_anon_0: core::mem::MaybeUninit<[u8; 4]>,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct x86_msi_addr_lo {
    __slate_anon_0: __SlateRecord_msi_h_49,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_msi_addr_hi {
    __slate_bits_0: __slate_bits::__SlateBits853U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_msg {
    __slate_anon_0: __SlateRecord_msi_h_62,
    __slate_anon_1: __SlateRecord_msi_h_66,
    __slate_anon_2: __SlateRecord_msi_h_70,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_amd_iommu_types_h_843 {
    data: [u64; 4],
    data128: [u128; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_msi_desc {
    __slate_anon_0: __SlateRecord_msi_h_109,
    msi_attrib: __SlateRecord_msi_h_113,
    __slate_anon_2: __SlateRecord_msi_h_122,
}


#[repr(C)]
#[derive(Clone, Copy)]
union msi_domain_cookie {
    value: u64,
    ptr: *mut (),
    iobase: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc_data {
    dcookie: msi_domain_cookie,
    icookie: msi_instance_cookie,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc {
    irq: u32,
    nvec_used: u32,
    dev: *mut device,
    msg: msi_msg,
    affinity: *mut irq_affinity_desc,
    sysfs_attrs: *mut device_attribute,
    write_msi_msg: Option<unsafe extern "C" fn(*mut msi_desc, *mut ())>,
    write_msi_msg_data: *mut (),
    msi_index: u16,
    __slate_anon_9: __SlateRecord_msi_h_202,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_domain_ops {
    get_hwirq: Option<unsafe extern "C" fn(*mut msi_domain_info, *mut irq_alloc_info) -> u64>,
    msi_init: Option<unsafe extern "C" fn(*mut irq_domain, *mut msi_domain_info, u32, u64, *mut irq_alloc_info) -> i32>,
    msi_free: Option<unsafe extern "C" fn(*mut irq_domain, *mut msi_domain_info, u32)>,
    msi_prepare: Option<unsafe extern "C" fn(*mut irq_domain, *mut device, i32, *mut irq_alloc_info) -> i32>,
    msi_teardown: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_alloc_info)>,
    prepare_desc: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_alloc_info, *mut msi_desc)>,
    set_desc: Option<unsafe extern "C" fn(*mut irq_alloc_info, *mut msi_desc)>,
    domain_alloc_irqs: Option<unsafe extern "C" fn(*mut irq_domain, *mut device, i32) -> i32>,
    domain_free_irqs: Option<unsafe extern "C" fn(*mut irq_domain, *mut device)>,
    msi_translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_init_c_125 {
    __slate_anon_0: __SlateRecord_init_c_125_2,
    ext_hid: __SlateRecord_init_c_125_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_125_2 {
    ext: u32,
    hidh: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_125_3 {
    ext: u32,
    hidh: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_domain_info {
    flags: u32,
    bus_token: u32,
    hwsize: u32,
    ops: *mut msi_domain_ops,
    dev: *mut device,
    chip: *mut irq_chip,
    chip_data: *mut (),
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    handler_data: *mut (),
    handler_name: *const u8,
    alloc_data: *mut irq_alloc_info,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_202 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1188_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_1308_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2094 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2094_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2094_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2094_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2094_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2094_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2102 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2102_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2102_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2102_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2102_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2102_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2181_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2186_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2278 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
    supported_flags: u32,
    required_flags: u32,
    chip_flags: u32,
    bus_select_token: u32,
    bus_select_mask: u32,
    prefix: *const u8,
    init_dev_msi_info: Option<unsafe extern "C" fn(*mut device, *mut irq_domain, *mut irq_domain, *mut msi_domain_info) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2377 {
    __slate_bits_0: __slate_bits::__SlateBits1019U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2444 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_2879_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3019_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3259_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3259_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3259_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3259_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3259_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3259_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3259_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3259_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3276_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3276_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3276_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3276_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3276_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3276_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3276_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3276_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3279_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3279_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3279_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3279_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3279_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3279_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3279_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_3279_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_4046 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_init_c_4054 {
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
    pub struct __SlateBits307U0 {
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
    pub struct __SlateBits72U0 {
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
    pub struct __SlateBits72U1 {
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
    pub struct __SlateBits310U0 {
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
    pub struct __SlateBits72U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits169U0 {
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
    pub struct __SlateBits171U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits425U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits657U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits492U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits494U0 {
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
    pub struct __SlateBits170U0 {
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
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits175U0 {
        #[bits(24)]
        pub intel_native_model_id: u32,
        #[bits(8)]
        pub intel_type: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits176U0 {
        #[bits(16)]
        pub amd_num_processors: u32,
        #[bits(8)]
        pub amd_power_eff_ranking: u32,
        #[bits(4)]
        pub amd_native_model_id: u32,
        #[bits(4)]
        pub amd_type: u32,
    }
    #[bitfields::bitfield([u8; 16], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits147U0 {
        #[bits(32)]
        pub max_std_leaf: u32,
        #[bits(32)]
        pub cpu_vendorid_0: u32,
        #[bits(32)]
        pub cpu_vendorid_2: u32,
        #[bits(32)]
        pub cpu_vendorid_1: u32,
    }
    #[bitfields::bitfield([u8; 16], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits148U0 {
        #[bits(4)]
        pub stepping: u32,
        #[bits(4)]
        pub base_model: u32,
        #[bits(4)]
        pub base_family_id: u32,
        #[bits(2)]
        pub cpu_type: u32,
        #[bits(2, access = na)]
        pub __slate_pad_4: u8,
        #[bits(4)]
        pub ext_model: u32,
        #[bits(8)]
        pub ext_family: u32,
        #[bits(4, access = na)]
        pub __slate_pad_7: u8,
        #[bits(8)]
        pub brand_id: u32,
        #[bits(8)]
        pub clflush_size: u32,
        #[bits(8)]
        pub n_logical_cpu: u32,
        #[bits(8)]
        pub local_apic_id: u32,
        #[bits(1)]
        pub sse3: u32,
        #[bits(1)]
        pub pclmulqdq: u32,
        #[bits(1)]
        pub dtes64: u32,
        #[bits(1)]
        pub monitor: u32,
        #[bits(1)]
        pub dscpl: u32,
        #[bits(1)]
        pub vmx: u32,
        #[bits(1)]
        pub smx: u32,
        #[bits(1)]
        pub est: u32,
        #[bits(1)]
        pub tm2: u32,
        #[bits(1)]
        pub ssse3: u32,
        #[bits(1)]
        pub cntxt_id: u32,
        #[bits(1)]
        pub sdbg: u32,
        #[bits(1)]
        pub fma: u32,
        #[bits(1)]
        pub cx16: u32,
        #[bits(1)]
        pub xtpr_update: u32,
        #[bits(1)]
        pub pdcm: u32,
        #[bits(1, access = na)]
        pub __slate_pad_28: u8,
        #[bits(1)]
        pub pcid: u32,
        #[bits(1)]
        pub dca: u32,
        #[bits(1)]
        pub sse4_1: u32,
        #[bits(1)]
        pub sse4_2: u32,
        #[bits(1)]
        pub x2apic: u32,
        #[bits(1)]
        pub movbe: u32,
        #[bits(1)]
        pub popcnt: u32,
        #[bits(1)]
        pub tsc_deadline_timer: u32,
        #[bits(1)]
        pub aes: u32,
        #[bits(1)]
        pub xsave: u32,
        #[bits(1)]
        pub osxsave: u32,
        #[bits(1)]
        pub avx: u32,
        #[bits(1)]
        pub f16c: u32,
        #[bits(1)]
        pub rdrand: u32,
        #[bits(1)]
        pub guest_status: u32,
        #[bits(1)]
        pub fpu: u32,
        #[bits(1)]
        pub vme: u32,
        #[bits(1)]
        pub de: u32,
        #[bits(1)]
        pub pse: u32,
        #[bits(1)]
        pub tsc: u32,
        #[bits(1)]
        pub msr: u32,
        #[bits(1)]
        pub pae: u32,
        #[bits(1)]
        pub mce: u32,
        #[bits(1)]
        pub cx8: u32,
        #[bits(1)]
        pub apic: u32,
        #[bits(1, access = na)]
        pub __slate_pad_54: u8,
        #[bits(1)]
        pub sep: u32,
        #[bits(1)]
        pub mtrr: u32,
        #[bits(1)]
        pub pge: u32,
        #[bits(1)]
        pub mca: u32,
        #[bits(1)]
        pub cmov: u32,
        #[bits(1)]
        pub pat: u32,
        #[bits(1)]
        pub pse36: u32,
        #[bits(1)]
        pub psn: u32,
        #[bits(1)]
        pub clflush: u32,
        #[bits(1, access = na)]
        pub __slate_pad_64: u8,
        #[bits(1)]
        pub ds: u32,
        #[bits(1)]
        pub acpi: u32,
        #[bits(1)]
        pub mmx: u32,
        #[bits(1)]
        pub fxsr: u32,
        #[bits(1)]
        pub sse: u32,
        #[bits(1)]
        pub sse2: u32,
        #[bits(1)]
        pub selfsnoop: u32,
        #[bits(1)]
        pub htt: u32,
        #[bits(1)]
        pub tm: u32,
        #[bits(1)]
        pub ia64: u32,
        #[bits(1)]
        pub pbe: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits146U0 {
        #[bits(1)]
        pub initialized: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits519U0 {
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
    pub struct __SlateBits516U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits519U1 {
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
    pub struct __SlateBits519U2 {
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
    pub struct __SlateBits519U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits809U0 {
        #[bits(1)]
        pub user_pasid_table: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits741U0 {
        #[bits(1)]
        pub attach_deferred: u32,
        #[bits(1)]
        pub pci_32bit_workaround: u32,
        #[bits(1)]
        pub require_direct: u32,
        #[bits(1)]
        pub shadow_on_flush: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits850U0 {
        #[bits(2)]
        pub reserved_0: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(1)]
        pub redirect_hint: u32,
        #[bits(1)]
        pub reserved_1: u32,
        #[bits(7)]
        pub virt_destid_8_14: u32,
        #[bits(8)]
        pub destid_0_7: u32,
        #[bits(12)]
        pub base_address: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits851U0 {
        #[bits(2)]
        pub dmar_reserved_0: u32,
        #[bits(1)]
        pub dmar_index_15: u32,
        #[bits(1)]
        pub dmar_subhandle_valid: u32,
        #[bits(1)]
        pub dmar_format: u32,
        #[bits(15)]
        pub dmar_index_0_14: u32,
        #[bits(12)]
        pub dmar_base_address: u32,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits853U0 {
        #[bits(8)]
        pub reserved: u32,
        #[bits(24)]
        pub destid_8_31: u32,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits846U0 {
        #[bits(8)]
        pub vector: u32,
        #[bits(3)]
        pub delivery_mode: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(2)]
        pub reserved: u32,
        #[bits(1)]
        pub active_low: u32,
        #[bits(1)]
        pub is_level: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits768U0 {
        #[bits(1)]
        pub is_managed: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits860U0 {
        #[bits(1)]
        pub is_msix: u8,
        #[bits(3)]
        pub multiple: u8,
        #[bits(3)]
        pub multi_cap: u8,
        #[bits(1)]
        pub can_mask: u8,
        #[bits(1)]
        pub is_64: u8,
        #[bits(1)]
        pub is_virtual: u8,
        #[bits(6, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits824U0 {
        #[bits(1)]
        pub is_level: u32,
        #[bits(1)]
        pub active_low: u32,
        #[bits(1)]
        pub valid: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits142U0 {
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
    pub struct __SlateBits143U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits869U0 {
        #[bits(1)]
        pub disable_esr: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(1)]
        pub x2apic_set_max_apicid: u32,
        #[bits(1)]
        pub nmi_to_offline_cpu: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits472U0 {
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
    pub struct __SlateBits777U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits770U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits770U1 {
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
    pub struct __SlateBits770U2 {
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
    pub struct __SlateBits770U3 {
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
    pub struct __SlateBits770U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1019U0 {
        #[bits(2)]
        pub reserved_0: u64,
        #[bits(1)]
        pub dest_mode_logical: u64,
        #[bits(5)]
        pub reserved_1: u64,
        #[bits(24)]
        pub destid_0_23: u64,
        #[bits(8)]
        pub vector: u64,
        #[bits(16)]
        pub reserved_2: u64,
        #[bits(8)]
        pub destid_24_31: u64,
    }
}

// reset direct map and flush TLB on unmap, can't be freed in atomic context
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
    __slate_anon_0: __SlateRecord_vmalloc_h_53,
    addr: *mut (),
    size: u64,
    flags: u64,
    pages: *mut *mut page,
    page_order: u32,
    nr_pages: u64,
    phys_addr: u64,
    caller: *const (),
    requested_size: u64,
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

#[unsafe(no_mangle)]
static mut amd_iommu_snp_en: bool = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
    next: *mut static_call_mod,
    r#mod: *mut module,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct apic {
    eoi: Option<unsafe extern "C" fn()>,
    native_eoi: Option<unsafe extern "C" fn()>,
    write: Option<unsafe extern "C" fn(u32, u32)>,
    read: Option<unsafe extern "C" fn(u32) -> u32>,
    wait_icr_idle: Option<unsafe extern "C" fn()>,
    safe_wait_icr_idle: Option<unsafe extern "C" fn() -> u32>,
    send_IPI: Option<unsafe extern "C" fn(i32, i32)>,
    send_IPI_mask: Option<unsafe extern "C" fn(*const cpumask, i32)>,
    send_IPI_mask_allbutself: Option<unsafe extern "C" fn(*const cpumask, i32)>,
    send_IPI_allbutself: Option<unsafe extern "C" fn(i32)>,
    send_IPI_all: Option<unsafe extern "C" fn(i32)>,
    send_IPI_self: Option<unsafe extern "C" fn(i32)>,
    __slate_bits_0: __slate_bits::__SlateBits869U0,
    calc_dest_apicid: Option<unsafe extern "C" fn(u32) -> u32>,
    icr_read: Option<unsafe extern "C" fn() -> u64>,
    icr_write: Option<unsafe extern "C" fn(u32, u32)>,
    max_apic_id: u32,
    probe: Option<unsafe extern "C" fn() -> i32>,
    setup: Option<unsafe extern "C" fn()>,
    teardown: Option<unsafe extern "C" fn()>,
    acpi_madt_oem_check: Option<unsafe extern "C" fn(*mut u8, *mut u8) -> i32>,
    init_apic_ldr: Option<unsafe extern "C" fn()>,
    cpu_present_to_apicid: Option<unsafe extern "C" fn(i32) -> u32>,
    get_apic_id: Option<unsafe extern "C" fn(u32) -> u32>,
    wakeup_secondary_cpu: Option<unsafe extern "C" fn(u32, u64, u32) -> i32>,
    wakeup_secondary_cpu_64: Option<unsafe extern "C" fn(u32, u64, u32) -> i32>,
    update_vector: Option<unsafe extern "C" fn(u32, u32, bool)>,
    name: *mut u8,
}


// For x2AVIC, the max index allowed for physical APIC ID table is 0x1ff (511).
// With X86_FEATURE_X2AVIC_EXT, the max index is increased to 0xfff (4095).
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ghcb_save_area {
    reserved_0x0: [u8; 203],
    cpl: u8,
    reserved_0xcc: [u8; 116],
    xss: u64,
    reserved_0x148: [u8; 24],
    dr7: u64,
    reserved_0x168: [u8; 16],
    rip: u64,
    reserved_0x180: [u8; 88],
    rsp: u64,
    reserved_0x1e0: [u8; 24],
    rax: u64,
    reserved_0x200: [u8; 264],
    rcx: u64,
    rdx: u64,
    rbx: u64,
    reserved_0x320: [u8; 8],
    rbp: u64,
    rsi: u64,
    rdi: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    reserved_0x380: [u8; 16],
    sw_exit_code: u64,
    sw_exit_info_1: u64,
    sw_exit_info_2: u64,
    sw_scratch: u64,
    reserved_0x3b0: [u8; 56],
    xcr0: u64,
    valid_bitmap: [u8; 16],
    x87_state_gpa: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ghcb {
    save: ghcb_save_area,
    reserved_save: [u8; 1016],
    shared_buffer: [u8; 2032],
    reserved_0xff0: [u8; 10],
    protocol_version: u16,
    ghcb_usage: u32,
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


// MFD_EXEC implied if unset
#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
    idr: idr,
    rcu: callback_head,
    pid_allocated: u32,
    memfd_noexec_scope: i32,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    child_reaper: *mut task_struct,
    pid_cachep: *mut kmem_cache,
    level: u32,
    pid_max: i32,
    parent: *mut pid_namespace,
    bacct: *mut fs_pin,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    reboot: i32,
    ns: ns_common,
    work: work_struct,
}


fn is_kdump_kernel() -> bool {
(unsafe { elfcorehdr_addr }) != 1u64.wrapping_neg()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fault_page_request {
    flags: u32,
    pasid: u32,
    grpid: u32,
    perm: u32,
    addr: u64,
    private_data: [u64; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fault {
    r#type: u32,
    prm: iommu_fault_page_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_page_response {
    pasid: u32,
    grpid: u32,
    code: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iopf_fault {
    fault: iommu_fault,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iopf_group {
    last_fault: iopf_fault,
    faults: list_head,
    fault_count: u64,
    pending_node: list_head,
    work: work_struct,
    attach_handle: *mut iommu_attach_handle,
    fault_param: *mut iommu_fault_param,
    node: list_head,
    cookie: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iopf_queue {
    wq: *mut workqueue_struct,
    devices: list_head,
    lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_domain_geometry {
    aperture_start: u64,
    aperture_end: u64,
    force_aperture: bool,
}


// Domain for use in DMA-API
// implementation
#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_domain {
    r#type: u32,
    cookie_type: u32,
    is_iommupt: bool,
    ops: *const iommu_domain_ops,
    dirty_ops: *const iommu_dirty_ops,
    owner: *const iommu_ops,
    pgsize_bitmap: u64,
    geometry: iommu_domain_geometry,
    iopf_handler: Option<unsafe extern "C" fn(*mut iopf_group) -> i32>,
    __slate_anon_9: __SlateRecord_iommu_h_234,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_pages_list {
    pages: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_iotlb_gather {
    start: u64,
    end: u64,
    __slate_anon_2: __SlateRecord_iommu_h_364,
    freelist: iommu_pages_list,
    queued: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dirty_bitmap {
    bitmap: *mut iova_bitmap,
    gather: *mut iommu_iotlb_gather,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dirty_ops {
    set_dirty_tracking: Option<unsafe extern "C" fn(*mut iommu_domain, bool) -> i32>,
    read_and_clear_dirty: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, *mut iommu_dirty_bitmap) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_user_data {
    r#type: u32,
    uptr: *mut (),
    len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_user_data_array {
    r#type: u32,
    uptr: *mut (),
    entry_len: u64,
    entry_num: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_ops {
    capable: Option<unsafe extern "C" fn(*mut device, u32) -> bool>,
    hw_info: Option<unsafe extern "C" fn(*mut device, *mut u32, *mut u32) -> *mut ()>,
    domain_alloc_identity: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_domain>,
    domain_alloc_paging_flags: Option<unsafe extern "C" fn(*mut device, u32, *const iommu_user_data) -> *mut iommu_domain>,
    domain_alloc_paging: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_domain>,
    domain_alloc_sva: Option<unsafe extern "C" fn(*mut device, *mut mm_struct) -> *mut iommu_domain>,
    domain_alloc_nested: Option<unsafe extern "C" fn(*mut device, *mut iommu_domain, u32, *const iommu_user_data) -> *mut iommu_domain>,
    probe_device: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_device>,
    release_device: Option<unsafe extern "C" fn(*mut device)>,
    probe_finalize: Option<unsafe extern "C" fn(*mut device)>,
    device_group: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_group>,
    get_resv_regions: Option<unsafe extern "C" fn(*mut device, *mut list_head)>,
    of_xlate: Option<unsafe extern "C" fn(*mut device, *const of_phandle_args) -> i32>,
    is_attach_deferred: Option<unsafe extern "C" fn(*mut device) -> bool>,
    page_response: Option<unsafe extern "C" fn(*mut device, *mut iopf_fault, *mut iommu_page_response)>,
    def_domain_type: Option<unsafe extern "C" fn(*mut device) -> i32>,
    get_viommu_size: Option<unsafe extern "C" fn(*mut device, u32) -> u64>,
    viommu_init: Option<unsafe extern "C" fn(*mut iommufd_viommu, *mut iommu_domain, *const iommu_user_data) -> i32>,
    default_domain_ops: *const iommu_domain_ops,
    owner: *mut module,
    identity_domain: *mut iommu_domain,
    blocked_domain: *mut iommu_domain,
    release_domain: *mut iommu_domain,
    default_domain: *mut iommu_domain,
    __slate_bits_0: __slate_bits::__SlateBits809U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_domain_ops {
    attach_dev: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, *mut iommu_domain) -> i32>,
    set_dev_pasid: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u32, *mut iommu_domain) -> i32>,
    map_pages: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, u64, i32, u32, *mut u64) -> i32>,
    unmap_pages: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, *mut iommu_iotlb_gather) -> u64>,
    flush_iotlb_all: Option<unsafe extern "C" fn(*mut iommu_domain)>,
    iotlb_sync_map: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64) -> i32>,
    iotlb_sync: Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_iotlb_gather)>,
    cache_invalidate_user: Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_user_data_array) -> i32>,
    iova_to_phys: Option<unsafe extern "C" fn(*mut iommu_domain, u64) -> u64>,
    enforce_cache_coherency: Option<unsafe extern "C" fn(*mut iommu_domain) -> bool>,
    set_pgtable_quirks: Option<unsafe extern "C" fn(*mut iommu_domain, u64) -> i32>,
    free: Option<unsafe extern "C" fn(*mut iommu_domain)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_device {
    list: list_head,
    ops: *const iommu_ops,
    fwnode: *mut fwnode_handle,
    dev: *mut device,
    singleton_group: *mut iommu_group,
    max_pasids: u32,
    ready: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fault_param {
    lock: mutex,
    users: refcount_struct,
    rcu: callback_head,
    dev: *mut device,
    queue: *mut iopf_queue,
    queue_list: list_head,
    partial: list_head,
    faults: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_iommu {
    lock: mutex,
    fault_param: *mut iommu_fault_param,
    fwspec: *mut iommu_fwspec,
    iommu_dev: *mut iommu_device,
    r#priv: *mut (),
    max_pasids: u32,
    __slate_bits_0: __slate_bits::__SlateBits741U0,
}


fn dev_to_iommu_device(dev: *mut device) -> *mut iommu_device {
dev_get_drvdata(dev as *const device) as *mut iommu_device
}

#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fwspec {
    iommu_fwnode: *mut fwnode_handle,
    flags: u32,
    num_ids: u32,
    ids: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_attach_handle {
    domain: *mut iommu_domain,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_mm_data {
    pasid: u32,
    mm: *mut mm_struct,
    sva_domains: list_head,
    mm_list_elm: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_object {
    wait_cnt: refcount_struct,
    users: refcount_struct,
    r#type: u32,
    id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_viommu {
    obj: iommufd_object,
    ictx: *mut iommufd_ctx,
    iommu_dev: *mut iommu_device,
    hwpt: *mut iommufd_hwpt_paging,
    ops: *const iommufd_viommu_ops,
    vdevs: xarray,
    veventqs: list_head,
    veventqs_rwsem: rw_semaphore,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_vdevice {
    obj: iommufd_object,
    viommu: *mut iommufd_viommu,
    idev: *mut iommufd_device,
    virt_id: u64,
    destroy: Option<unsafe extern "C" fn(*mut iommufd_vdevice)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_hw_queue {
    obj: iommufd_object,
    viommu: *mut iommufd_viommu,
    access: *mut iommufd_access,
    base_addr: u64,
    length: u64,
    r#type: u32,
    destroy: Option<unsafe extern "C" fn(*mut iommufd_hw_queue)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_viommu_ops {
    destroy: Option<unsafe extern "C" fn(*mut iommufd_viommu)>,
    alloc_domain_nested: Option<unsafe extern "C" fn(*mut iommufd_viommu, u32, *const iommu_user_data) -> *mut iommu_domain>,
    cache_invalidate: Option<unsafe extern "C" fn(*mut iommufd_viommu, *mut iommu_user_data_array) -> i32>,
    vdevice_size: u64,
    vdevice_init: Option<unsafe extern "C" fn(*mut iommufd_vdevice) -> i32>,
    get_hw_queue_size: Option<unsafe extern "C" fn(*mut iommufd_viommu, u32) -> u64>,
    hw_queue_init_phys: Option<unsafe extern "C" fn(*mut iommufd_hw_queue, u32, u64) -> i32>,
}


// some size calculation constants
// Capability offsets used by the driver
// Used offsets into the MMIO space
// Extended Feature 2 Bits
// Note:
// The current driver only support 16-bit PASID.
// Currently, hardware only implement upto 16-bit PASID
// even though the spec says it could have upto 20 bits.
// MMIO status bits
// feature control bits
// macros and definitions for device table entries
// constants to configure the command buffer
// 512 entries
// 512 entries
// This bitmap is used to advertise the page sizes our hardware support
// to the IOMMU core, which will then use this information to split
// physically contiguous memory regions it is mapping into page sizes
// that we support.
//
// 512GB Pages are not supported due to a hardware bug
// Page sizes >= the 52 bit max physical address of the CPU are not supported.
// Special mode where page-sizes are limited to 4 KiB
// 4K, 2MB, 1G page sizes are supported
// Bit value definition for DTE fields
// IOMMU capabilities
// IOMMU IVINFO
// IOMMU Feature Reporting Field (for IVHD type 10h
// IOMMU HATDIS for IVHD type 11h and 40h
// IOMMU Extended Feature Register (EFR)
#[unsafe(no_mangle)]
static mut amd_iommu_dump: bool = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut amd_iommu_snp_mode0_sup: bool = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut amd_iommu_np_cache: bool = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct irq_remap_table {
    lock: raw_spinlock,
    min_index: u32,
    table: *mut u32,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut amdr_ivrs_remap_support: bool = unsafe { core::mem::zeroed() };

// Make iterating over all pci segment easier
// Make iterating over all IOMMUs easier
#[repr(C)]
#[derive(Clone, Copy)]
struct amd_iommu_pci_seg {
    list: list_head,
    dev_data_list: llist_head,
    id: u16,
    last_bdf: u16,
    dev_table_size: u32,
    dev_table: *mut dev_table_entry,
    rlookup_table: *mut *mut amd_iommu,
    irq_lookup_table: *mut *mut irq_remap_table,
    old_dev_tbl_cpy: *mut dev_table_entry,
    alias_table: *mut u16,
    unity_map: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct amd_iommu {
    list: list_head,
    index: i32,
    lock: raw_spinlock,
    dev: *mut pci_dev,
    root_pdev: *mut pci_dev,
    mmio_phys: u64,
    mmio_phys_end: u64,
    mmio_base: *mut u8,
    cap: u32,
    acpi_flags: u8,
    features: u64,
    features2: u64,
    devid: u16,
    cap_ptr: u16,
    pci_seg: *mut amd_iommu_pci_seg,
    cmd_buf: *mut u8,
    cmd_buf_head: u32,
    cmd_buf_tail: u32,
    evt_buf: *mut u8,
    evt_irq_name: [u8; 16],
    ppr_log: *mut u8,
    ppr_irq_name: [u8; 16],
    ga_log: *mut u8,
    ga_irq_name: [u8; 16],
    ga_log_tail: *mut u8,
    int_enabled: bool,
    need_sync: bool,
    irtcachedis_enabled: bool,
    iommu: iommu_device,
    stored_addr_lo: u32,
    stored_addr_hi: u32,
    stored_l1: [[u32; 18]; 6],
    stored_l2: [u32; 131],
    max_banks: u8,
    max_counters: u8,
    ir_domain: *mut irq_domain,
    irte_ops: *mut amd_irte_ops,
    flags: u32,
    cmd_sem: *mut u64,
    cmd_sem_val: u64,
    cmd_sem_paddr: u64,
    iopf_queue: *mut iopf_queue,
    iopfq_name: [u8; 32],
}


fn dev_to_amd_iommu(dev: *mut device) -> *mut amd_iommu {
let iommu: *mut iommu_device = dev_to_iommu_device(dev);
container_of__7!(iommu, amd_iommu)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct acpihid_map_entry {
    list: list_head,
    uid: [u8; 256],
    hid: [u8; 9],
    devid: u32,
    root_devid: u32,
    cmd_line: bool,
    group: *mut iommu_group,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct devid_map {
    list: list_head,
    id: u8,
    devid: u32,
    cmd_line: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_table_entry {
    __slate_anon_0: __SlateRecord_amd_iommu_types_h_843,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ivhd_dte_flags {
    list: list_head,
    segid: u16,
    devid_first: u16,
    devid_last: u16,
    dte: dev_table_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unity_map_entry {
    list: list_head,
    devid_start: u16,
    devid_end: u16,
    address_start: u64,
    address_end: u64,
    prot: i32,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut amd_iommu_force_isolation: bool = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut amd_iommu_efr: u64 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut amd_iommu_efr2: u64 = unsafe { core::mem::zeroed() };

fn get_ioapic_devid(id: i32) -> i32 {
let mut entry: *mut devid_map = unsafe { core::mem::zeroed() };
entry = list_first_entry__19!(unsafe { core::ptr::addr_of_mut!(ioapic_map) });
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!(ioapic_map) }, list) == 0 {
if ((unsafe { (*entry).id }) as i32) == id {
return (unsafe { (*entry).devid }) as i32;
}
entry = list_next_entry__18!(entry, list);
}
22i32.wrapping_neg()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct amd_irte_ops {
    prepare: Option<unsafe extern "C" fn(*mut (), u32, bool, u8, u32, i32)>,
    activate: Option<unsafe extern "C" fn(*mut amd_iommu, *mut (), u16, u16)>,
    deactivate: Option<unsafe extern "C" fn(*mut amd_iommu, *mut (), u16, u16)>,
    set_affinity: Option<unsafe extern "C" fn(*mut amd_iommu, *mut (), u16, u16, u8, u32)>,
    get: Option<unsafe extern "C" fn(*mut irq_remap_table, i32) -> *mut ()>,
    set_allocated: Option<unsafe extern "C" fn(*mut irq_remap_table, i32)>,
    is_allocated: Option<unsafe extern "C" fn(*mut irq_remap_table, i32) -> bool>,
    clear_allocated: Option<unsafe extern "C" fn(*mut irq_remap_table, i32)>,
}


fn amd_iommu_debugfs_setup() {
}

#[unsafe(no_mangle)]
static mut amd_iommu_hpt_vasize: u8 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut amd_iommu_hatdis: bool = unsafe { core::mem::zeroed() };

fn is_rd890_iommu(pdev: *mut pci_dev) -> bool {
((unsafe { (*pdev).vendor }) as i32) == 4098 && ((unsafe { (*pdev).device }) as i32) == 23075
}

fn check_feature(mask: u64) -> bool {
(unsafe { amd_iommu_efr }) & mask != 0
}

fn check_feature2(mask: u64) -> bool {
(unsafe { amd_iommu_efr2 }) & mask != 0
}

fn amd_iommu_v2_pgtbl_supported() -> bool {
check_feature(BIT_ULL!(48i32)) && check_feature(BIT_ULL!(4i32))
}

fn amd_iommu_gt_ppr_supported() -> bool {
let __v0: bool = amd_iommu_v2_pgtbl_supported() && check_feature(BIT_ULL!(1i32));
__v0 && check_feature(BIT_ULL!(50i32))
}

fn iommu_virt_to_phys(vaddr: *mut ()) -> u64 {
__sme_set!(virt_to_phys(vaddr as *mut ()))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct irq_remap_ops {
    capability: i32,
    prepare: Option<unsafe extern "C" fn() -> i32>,
    enable: Option<unsafe extern "C" fn() -> i32>,
    disable: Option<unsafe extern "C" fn()>,
    reenable: Option<unsafe extern "C" fn(i32) -> i32>,
    enable_faulting: Option<unsafe extern "C" fn(u32) -> i32>,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_amd_iommu_snp_en: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous", sym amd_iommu_snp_en, options(att_syntax));

fn iommu_alloc_pages_sz(gfp: u32, size: u64) -> *mut () {
unsafe { iommu_alloc_pages_node_sz(1i32.wrapping_neg(), gfp, size) }
}

// definitions for the ACPI scanning code
// ACPI table definitions
//
// These data structures are laid over the table to parse the important values
// out of it.
/// structure describing one IOMMU in the ACPI table. Typically followed by one
/// or more ivhd_entrys.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ivhd_header {
    r#type: u8,
    flags: u8,
    length: u16,
    devid: u16,
    cap_ptr: u16,
    mmio_phys: u64,
    pci_seg: u16,
    info: u16,
    efr_attr: u32,
    /// Following only valid on IVHD type 11h and 40h
    /// Exact copy of MMIO_EXT_FEATURES
    efr_reg: u64,
    efr_reg2: u64,
}


/// A device entry describing which devices a specific IOMMU translates and
/// which requestor ids they use.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ivhd_entry {
    r#type: u8,
    devid: u16,
    flags: u8,
    __slate_anon_3: __SlateRecord_init_c_125,
    cid: u64,
    uidf: u8,
    uidl: u8,
    uid: u8,
}


#[unsafe(no_mangle)]
static mut amd_iommu_evtlog_size: i32 = 8192;

#[unsafe(no_mangle)]
static mut amd_iommu_pprlog_size: i32 = 8192;

/// An AMD IOMMU memory definition structure. It defines things like exclusion
/// ranges for devices and regions that should be unity mapped.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ivmd_header {
    r#type: u8,
    flags: u8,
    length: u16,
    devid: u16,
    aux: u16,
    pci_seg: u16,
    resv: [u8; 6],
    range_start: u64,
    range_length: u64,
}


#[unsafe(link_section = ".data..read_mostly")]
static mut amd_iommu_irq_remap: bool = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut amd_iommu_pgtable: u32 = 1;

/// Virtual address size
/// Guest page table level
#[unsafe(no_mangle)]
static mut amd_iommu_gpt_level: i32 = 4;

#[unsafe(no_mangle)]
static mut amd_iommu_guest_ir: i32 = 2;

static mut amd_iommu_xt_mode: i32 = 0;

static mut amd_iommu_detected: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut amd_iommu_disabled: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut amd_iommu_force_enable: bool = unsafe { core::mem::zeroed() };

static mut amd_iommu_irtcachedis: bool = unsafe { core::mem::zeroed() };

static mut amd_iommu_target_ivhd_type: i32 = unsafe { core::mem::zeroed() };

/// Global EFR and EFR2 registers
/// Host (v1) page table is not supported
/// SNP is enabled on the system?
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_amd_iommu_snp_en_622: *mut () = (unsafe { core::ptr::addr_of_mut!(amd_iommu_snp_en) }) as *mut ();

/// SNP page mode 0 support
/// list of all PCI segments
#[unsafe(no_mangle)]
static mut amd_iommu_pci_seg_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) }, prev: unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) } };

/// list of all AMD IOMMUs in the system
#[unsafe(no_mangle)]
static mut amd_iommu_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, prev: unsafe { core::ptr::addr_of_mut!(amd_iommu_list) } };

/// list of all IVHD device entry settings
#[unsafe(no_mangle)]
static mut amd_ivhd_dev_flags_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(amd_ivhd_dev_flags_list) }, prev: unsafe { core::ptr::addr_of_mut!(amd_ivhd_dev_flags_list) } };

/// Number of IOMMUs present in the system
static mut amd_iommus_present: i32 = unsafe { core::mem::zeroed() };

/// IOMMUs have a non-present cache?
#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut amd_iommu_iotlb_sup: bool = 1i32 != 0i32;

#[unsafe(link_section = ".data..read_mostly")]
static mut amd_iommu_pc_present: bool = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..ro_after_init")]
static mut amd_iommu_pgsize_bitmap: u64 = (((core::mem::size_of::<__SlateRecord_init_c_202>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(51u64)) ^ 549755813888u64;

// Early ioapic and hpet maps from kernel command line
#[unsafe(link_section = ".init.data")]
static mut early_ioapic_map: __SlateAlign16<[devid_map; 4]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(link_section = ".init.data")]
static mut early_hpet_map: __SlateAlign16<[devid_map; 4]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(link_section = ".init.data")]
static mut early_acpihid_map: __SlateAlign16<[acpihid_map_entry; 4]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(link_section = ".init.data")]
static mut early_ioapic_map_size: i32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut early_hpet_map_size: i32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut early_acpihid_map_size: i32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut cmdline_maps: bool = unsafe { core::mem::zeroed() };

static mut init_state: u32 = 0;

static mut amd_iommu_pre_enabled: bool = 1i32 != 0i32;

#[unsafe(link_section = ".init.data")]
static mut amd_iommu_ivinfo: u32 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
extern "C" fn translation_pre_enabled(iommu: *mut amd_iommu) -> bool {
(unsafe { (*iommu).flags }) & 1u32 != 0
}

fn clear_translation_pre_enabled(__v0: *mut amd_iommu) {
let __v2: u32 = (unsafe { (*__v0).flags }) & 4294967294u32;
unsafe {
(*__v0).flags = __v2;
}
}

fn init_translation_status(iommu: *mut amd_iommu) {
let ctrl: u64 = readq(unsafe { (*iommu).mmio_base }.wrapping_offset(24 as isize) as *const ());
if ctrl & 1u64 != 0 {
unsafe {
(*iommu).flags = (*iommu).flags | 1u32;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn amd_iommu_get_num_iommus() -> i32 {
unsafe { amd_iommus_present }
}

#[unsafe(no_mangle)]
extern "C" fn amd_iommu_ht_range_ignore() -> bool {
check_feature2(BIT_ULL!(11i32))
}

/// Iterate through all the IOMMUs to get common EFR
/// masks among all IOMMUs and warn if found inconsistency.
#[unsafe(link_section = ".init.text")]
fn get_global_efr() {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
let tmp: u64 = unsafe { (*iommu).features };
let tmp2: u64 = unsafe { (*iommu).features2 };
if list_is_first((unsafe { core::ptr::addr_of_mut!((*iommu).list) }) as *const list_head, (unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }) as *const list_head) != 0 {
unsafe {
amd_iommu_efr = tmp;
}
unsafe {
amd_iommu_efr2 = tmp2;
}
} else {
if !((unsafe { amd_iommu_efr }) == tmp && (unsafe { amd_iommu_efr2 }) == tmp2) {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x013AMD-Vi: [Firmware Bug]: Found inconsistent EFR/EFR2 %#llx,%#llx (global %#llx,%#llx) on iommu%d (%04x:%02x:%02x.%01x).\n".as_ptr() as *const u8, tmp, tmp2, amd_iommu_efr, amd_iommu_efr2, (*iommu).index, (*(*iommu).pci_seg).id as i32, PCI_BUS_NUM!(unsafe { (*iommu).devid }), PCI_SLOT!(unsafe { (*iommu).devid }), PCI_FUNC!(unsafe { (*iommu).devid })) };
unsafe {
amd_iommu_efr = amd_iommu_efr & tmp;
}
unsafe {
amd_iommu_efr2 = amd_iommu_efr2 & tmp2;
}
}
}
iommu = list_next_entry__3!(iommu, list);
}
let mut __v9: i32;
while false {
}
__v9 = unsafe { _printk(c"\x016AMD-Vi: Using global IVHD EFR:%#llx, EFR2:%#llx\n".as_ptr() as *const u8, amd_iommu_efr, amd_iommu_efr2) };
}

/// For IVHD type 0x11/0x40, EFR is also available via IVHD.
/// Default to IVHD EFR since it is available sooner
/// (i.e. before PCI init).
#[unsafe(link_section = ".init.text")]
fn early_iommu_features_init(iommu: *mut amd_iommu, h: *mut ivhd_header) {
if ((unsafe { amd_iommu_ivinfo }) as u64) & BIT!(0i32) != 0 {
unsafe {
(*iommu).features = (*h).efr_reg;
}
unsafe {
(*iommu).features2 = (*h).efr_reg2;
}
}
if ((unsafe { amd_iommu_ivinfo }) as u64) & BIT!(1i32) != 0 {
unsafe {
amdr_ivrs_remap_support = true;
}
}
}

// Access to l1 and l2 indexed register spaces
fn iommu_read_l1(iommu: *mut amd_iommu, l1: u16, address: u8) -> u32 {
let mut val: u32 = 0;
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 248i32, ((address as i32) | (l1 as i32) << 16i32) as u32) };
unsafe { pci_read_config_dword((*iommu).dev as *const pci_dev, 252i32, core::ptr::addr_of_mut!(val)) };
val
}

fn iommu_write_l1(iommu: *mut amd_iommu, l1: u16, address: u8, val: u32) {
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 248i32, ((address as i32) | (l1 as i32) << 16i32 | -2147483648i32) as u32) };
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 252i32, val) };
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 248i32, ((address as i32) | (l1 as i32) << 16i32) as u32) };
}

fn iommu_read_l2(iommu: *mut amd_iommu, address: u8) -> u32 {
let mut val: u32 = 0;
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 240i32, address as u32) };
unsafe { pci_read_config_dword((*iommu).dev as *const pci_dev, 244i32, core::ptr::addr_of_mut!(val)) };
val
}

fn iommu_write_l2(iommu: *mut amd_iommu, address: u8, val: u32) {
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 240i32, ((address as i32) | 256i32) as u32) };
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 244i32, val) };
}

// AMD IOMMU MMIO register space handling functions
//
// These functions are used to program the IOMMU device registers in
// MMIO space required for that driver.
fn iommu_set_cwwb_range(iommu: *mut amd_iommu) {
let start: u64 = iommu_virt_to_phys((unsafe { (*iommu).cmd_sem }) as *mut ());
let mut entry: u64 = start & 4503599627366400;
if !check_feature(BIT_ULL!(63i32)) {
return;
}
// Note:
// Re-purpose Exclusion base/limit registers for Completion wait
// write-back base/limit.
unsafe { memcpy_toio(unsafe { (*iommu).mmio_base }.wrapping_offset(32 as isize) as *mut (), core::ptr::addr_of_mut!(entry) as *const (), core::mem::size_of::<u64>() as u64) };
// Note:
// Default to 4 Kbytes, which can be specified by setting base
// address equal to the limit address.
unsafe { memcpy_toio(unsafe { (*iommu).mmio_base }.wrapping_offset(40 as isize) as *mut (), core::ptr::addr_of_mut!(entry) as *const (), core::mem::size_of::<u64>() as u64) };
}

/// Programs the physical address of the device table into the IOMMU hardware
fn iommu_set_device_table(iommu: *mut amd_iommu) {
let mut entry: u64 = 0;
let dev_table_size: u32 = unsafe { (*(*iommu).pci_seg).dev_table_size };
let dev_table: *mut () = (unsafe { get_dev_table(iommu) }) as *mut ();
loop {
if unlikely!(unsafe { (*iommu).mmio_base }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("623: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 623b - ., 3; .popsection\n# {0}", const 623 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_iommu_set_device_table_0, sym __slate_asm_string_iommu_set_device_table_1, const 391 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if true {
break;
}
}
if is_kdump_kernel() {
return;
}
entry = iommu_virt_to_phys(dev_table);
entry |= (dev_table_size >> 12i32).wrapping_sub(1) as u64;
unsafe { memcpy_toio(unsafe { (*iommu).mmio_base }.wrapping_offset(0 as isize) as *mut (), core::ptr::addr_of_mut!(entry) as *const (), core::mem::size_of::<u64>() as u64) };
}

fn iommu_feature_set(iommu: *mut amd_iommu, val: u64, mut mask: u64, shift: u8) {
let mut ctrl: u64 = readq(unsafe { (*iommu).mmio_base }.wrapping_offset(24 as isize) as *const ());
mask <<= shift as i32;
let __v3: u64 = ctrl & !mask;
ctrl = __v3;
ctrl |= val << (shift as i32) & mask;
writeq(ctrl, unsafe { (*iommu).mmio_base }.wrapping_offset(24 as isize) as *mut ());
}

/// Generic functions to enable/disable certain features of the IOMMU.
#[unsafe(no_mangle)]
extern "C" fn iommu_feature_enable(iommu: *mut amd_iommu, bit: u8) {
iommu_feature_set(iommu, 1u64, 1u64, bit);
}

fn iommu_feature_disable(iommu: *mut amd_iommu, bit: u8) {
iommu_feature_set(iommu, 0u64, 1u64, bit);
}

/// Function to enable the hardware
fn iommu_enable(iommu: *mut amd_iommu) {
iommu_feature_enable(iommu, 0u8);
}

fn iommu_disable(iommu: *mut amd_iommu) {
if unsafe { (*iommu).mmio_base }.is_null() {
return;
}
// Disable command buffer
iommu_feature_disable(iommu, 12u8);
// Disable event logging and event interrupts
iommu_feature_disable(iommu, 3u8);
iommu_feature_disable(iommu, 2u8);
// Disable IOMMU GA_LOG
iommu_feature_disable(iommu, 28u8);
iommu_feature_disable(iommu, 29u8);
// Disable IOMMU PPR logging
iommu_feature_disable(iommu, 13u8);
iommu_feature_disable(iommu, 14u8);
// Disable IOMMU hardware itself
iommu_feature_disable(iommu, 0u8);
// Clear IRTE cache disabling bit
iommu_feature_disable(iommu, 59u8);
}

/// mapping and unmapping functions for the IOMMU MMIO space. Each AMD IOMMU in
/// the system has one.
#[unsafe(link_section = ".init.text")]
fn iommu_map_mmio_space(address: u64, end: u64) -> *mut u8 {
if request_mem_region!(address, end, c"amd_iommu".as_ptr() as *mut u8).is_null() {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013AMD-Vi: Can not reserve memory region %llx-%llx for mmio\n".as_ptr() as *const u8, address, end) };
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x013AMD-Vi: This is a BIOS bug. Please contact your hardware vendor\n".as_ptr() as *const u8) };
return core::ptr::null_mut::<u8>();
}
(unsafe { ioremap(address, end) }) as *mut u8
}

#[unsafe(link_section = ".init.text")]
fn iommu_unmap_mmio_space(iommu: *mut amd_iommu) {
if !unsafe { (*iommu).mmio_base }.is_null() {
unsafe { iounmap((*iommu).mmio_base as *mut ()) };
}
release_mem_region!(unsafe { (*iommu).mmio_phys }, unsafe { (*iommu).mmio_phys_end });
}

fn get_ivhd_header_size(h: *mut ivhd_header) -> u32 {
let mut size: u32 = 0;
match (unsafe { (*h).r#type }) as i32 {
16 => {
size = 24;
}
17 | 64 => {
size = 40;
}
_ => {
}
}
size
}

// The functions below belong to the first pass of AMD IOMMU ACPI table
// parsing. In this pass we try to find out the highest device id this
// code has to handle. Upon this information the size of the shared data
// structures is determined later.
/// This function calculates the length of a given IVHD entry
fn ivhd_entry_length(ivhd: *mut u8) -> i32 {
let r#type: u32 = (unsafe { (*(ivhd as *mut ivhd_entry)).r#type }) as u32;
if r#type < 128 {
return 4i32 << (((unsafe { *ivhd }) as i32) >> 6i32);
} else {
if r#type == 240 {
// For ACPI_HID, offset 21 is uid len
return ((unsafe { *ivhd.wrapping_offset(21 as isize) }) as i32).wrapping_add(22);
}
}
0
}

/// After reading the highest device id from the IOMMU PCI capability header
/// this function looks if there is a higher device id defined in the ACPI table
#[unsafe(link_section = ".init.text")]
fn find_last_devid_from_ivhd(h: *mut ivhd_header) -> i32 {
let mut p: *mut u8 = h as *mut u8;
let mut end: *mut u8 = h as *mut u8;
let mut last_devid: i32 = 22i32.wrapping_neg();
let ivhd_size: u32 = get_ivhd_header_size(h);
if ivhd_size == 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013AMD-Vi: Unsupported IVHD type %#x\n".as_ptr() as *const u8, (*h).r#type as i32) };
return 22i32.wrapping_neg();
}
let __v2: *mut u8 = p;
p = __v2.wrapping_offset(ivhd_size as isize);
let __v4: *mut u8 = end;
end = __v4.wrapping_offset(((unsafe { (*h).length }) as i32) as isize);
while p < end {
match (unsafe { (*(p as *mut ivhd_entry)).r#type }) as i32 {
1 => {
return 65535; // Use maximum BDF value for DEV_ALL
}
2 | 4 | 66 | 70 => {
if ((unsafe { (*(p as *mut ivhd_entry)).devid }) as i32) > last_devid {
last_devid = (unsafe { (*(p as *mut ivhd_entry)).devid }) as i32;
}
// all the above subfield types refer to device ids
}
_ => {
}
}
let __v7: i32 = ivhd_entry_length(p);
let __v8: *mut u8 = p;
p = __v8.wrapping_offset(__v7 as isize);
}
let mut __v11: i64;
let __ret_warn_on: i32 = (p != end) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("624: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 624b - ., 3; .popsection\n# {0}", const 624 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_find_last_devid_from_ivhd_12, sym __slate_asm_string_find_last_devid_from_ivhd_13, const 561 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("625: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 625b - ., 4; .popsection\n# {0}", const 625 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v11 = (__ret_warn_on != 0) as i64;
last_devid
}

#[unsafe(link_section = ".init.text")]
fn check_ivrs_checksum(table: *mut acpi_table_header) -> i32 {
let mut i: i32 = 0;
let mut checksum: u8 = 0;
let p: *mut u8 = table as *mut u8;
while (i as u32) < unsafe { (*table).length } {
checksum = (checksum as i32).wrapping_add((unsafe { *p.wrapping_offset(i as isize) }) as i32) as u8;
i = i.wrapping_add(1);
}
if (checksum as i32) != 0 {
// ACPI table corrupt
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013AMD-Vi: [Firmware Bug]: IVRS invalid checksum\n".as_ptr() as *const u8) };
return 19i32.wrapping_neg();
}
0
}

/// Iterate over all IVHD entries in the ACPI table and find the highest device
/// id which we need to handle. This is the first of three functions which parse
/// the ACPI table. So we check the checksum here.
#[unsafe(link_section = ".init.text")]
fn find_last_devid_acpi(table: *mut acpi_table_header, pci_seg: u16) -> i32 {
let mut p: *mut u8 = table as *mut u8;
let mut end: *mut u8 = table as *mut u8;
let mut last_devid: i32 = 0;
let mut last_bdf: i32 = 0;
let __v0: *mut u8 = p;
p = __v0.wrapping_offset(48 as isize);
let __v2: *mut u8 = end;
end = __v2.wrapping_offset((unsafe { (*table).length }) as isize);
while p < end {
let h: *mut ivhd_header = p as *mut ivhd_header;
if ((unsafe { (*h).pci_seg }) as i32) == (pci_seg as i32) && ((unsafe { (*h).r#type }) as i32) == unsafe { amd_iommu_target_ivhd_type } {
last_devid = find_last_devid_from_ivhd(h);
if last_devid < 0 {
return 22i32.wrapping_neg();
}
if last_devid > last_bdf {
last_bdf = last_devid;
}
}
let __v4: *mut u8 = p;
p = __v4.wrapping_offset(((unsafe { (*h).length }) as i32) as isize);
}
let mut __v7: i64;
let __ret_warn_on: i32 = (p != end) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("626: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 626b - ., 3; .popsection\n# {0}", const 626 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_find_last_devid_acpi_8, sym __slate_asm_string_find_last_devid_acpi_9, const 609 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("627: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 627b - ., 4; .popsection\n# {0}", const 627 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
last_bdf
}

// The following functions belong to the code path which parses the ACPI table
// the second time. In this ACPI parsing iteration we allocate IOMMU specific
// data structures, initialize the per PCI segment device/alias/rlookup table
// and also basically initialize the hardware.
/// Allocate per PCI segment device table
#[unsafe(link_section = ".init.text")]
fn alloc_dev_table(pci_seg: *mut amd_iommu_pci_seg) -> i32 {
unsafe {
(*pci_seg).dev_table = iommu_alloc_pages_sz(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(2i32) as u32), (*pci_seg).dev_table_size as u64) as *mut dev_table_entry;
}
if unsafe { (*pci_seg).dev_table }.is_null() {
return 12i32.wrapping_neg();
}
0
}

fn free_dev_table(pci_seg: *mut amd_iommu_pci_seg) {
if is_kdump_kernel() {
unsafe { memunmap((*pci_seg).dev_table as *mut ()) };
} else {
unsafe { iommu_free_pages((*pci_seg).dev_table as *mut ()) };
}
unsafe {
(*pci_seg).dev_table = core::ptr::null_mut::<dev_table_entry>();
}
}

/// Allocate per PCI segment IOMMU rlookup table.
#[unsafe(link_section = ".init.text")]
fn alloc_rlookup_table(pci_seg: *mut amd_iommu_pci_seg) -> i32 {
let mut __v0: *mut *mut amd_iommu;
let __obj_size: u64 = size_mul(core::mem::size_of::<*mut amd_iommu>() as u64, ((unsafe { (*pci_seg).last_bdf }) as i32).wrapping_add(1) as u64);
__v0 = kvmalloc_node!(__obj_size, 3520u32, 1i32.wrapping_neg()) as *mut *mut amd_iommu;
unsafe {
(*pci_seg).rlookup_table = __v0;
}
if unsafe { (*pci_seg).rlookup_table }.is_null() {
return 12i32.wrapping_neg();
}
0
}

fn free_rlookup_table(pci_seg: *mut amd_iommu_pci_seg) {
unsafe { kvfree((*pci_seg).rlookup_table as *const ()) };
unsafe {
(*pci_seg).rlookup_table = core::ptr::null_mut::<*mut amd_iommu>();
}
}

#[unsafe(link_section = ".init.text")]
fn alloc_irq_lookup_table(pci_seg: *mut amd_iommu_pci_seg) -> i32 {
let mut __v0: *mut *mut irq_remap_table;
let __obj_size: u64 = size_mul(core::mem::size_of::<*mut irq_remap_table>() as u64, ((unsafe { (*pci_seg).last_bdf }) as i32).wrapping_add(1) as u64);
__v0 = kvmalloc_node!(__obj_size, 3520u32, 1i32.wrapping_neg()) as *mut *mut irq_remap_table;
unsafe {
(*pci_seg).irq_lookup_table = __v0;
}
if unsafe { (*pci_seg).irq_lookup_table }.is_null() {
return 12i32.wrapping_neg();
}
0
}

fn free_irq_lookup_table(pci_seg: *mut amd_iommu_pci_seg) {
unsafe { kvfree((*pci_seg).irq_lookup_table as *const ()) };
unsafe {
(*pci_seg).irq_lookup_table = core::ptr::null_mut::<*mut irq_remap_table>();
}
}

#[unsafe(link_section = ".init.text")]
fn alloc_alias_table(pci_seg: *mut amd_iommu_pci_seg) -> i32 {
// let all alias entries point to itself
let mut i: i32 = 0;
let mut __v0: *mut u16;
let __obj_size: u64 = size_mul(core::mem::size_of::<u16>() as u64, ((unsafe { (*pci_seg).last_bdf }) as i32).wrapping_add(1) as u64);
__v0 = kvmalloc_node!(__obj_size, 3264u32, 1i32.wrapping_neg()) as *mut u16;
unsafe {
(*pci_seg).alias_table = __v0;
}
if unsafe { (*pci_seg).alias_table }.is_null() {
return 12i32.wrapping_neg();
}
while i <= ((unsafe { (*pci_seg).last_bdf }) as i32) {
unsafe {
*unsafe { (*pci_seg).alias_table }.wrapping_offset(i as isize) = i as u16;
}
i = i.wrapping_add(1);
}
0
}

#[unsafe(link_section = ".init.text")]
fn free_alias_table(pci_seg: *mut amd_iommu_pci_seg) {
unsafe { kvfree((*pci_seg).alias_table as *const ()) };
unsafe {
(*pci_seg).alias_table = core::ptr::null_mut::<u16>();
}
}

fn iommu_memremap(paddr: u64, size: u64) -> *mut () {
let mut phys: u64 = 0;
if paddr == 0 {
return core::ptr::null_mut::<()>();
}
// Obtain true physical address in kdump kernel when SME is enabled.
// Currently, previous kernel with SME enabled and kdump kernel
// with SME support disabled is not supported.
phys = __sme_clr!(paddr);
if cc_platform_has(1u32) {
return unsafe { ioremap_encrypted(phys, size) };
} else {
return unsafe { memremap(phys, size, 1u64) };
}
}

/// Allocates the command buffer. This buffer is per AMD IOMMU. We can
/// write commands to that buffer later and the IOMMU will execute them
/// asynchronously
#[unsafe(link_section = ".init.text")]
fn alloc_command_buffer(iommu: *mut amd_iommu) -> i32 {
unsafe {
(*iommu).cmd_buf = iommu_alloc_pages_sz(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), 8192u64) as *mut u8;
}
if !unsafe { (*iommu).cmd_buf }.is_null() { 0 } else { 12i32.wrapping_neg() }
}

/// Interrupt handler has processed all pending events and adjusted head
/// and tail pointer. Reset overflow mask and restart logging again.
#[unsafe(no_mangle)]
extern "C" fn amd_iommu_restart_log(iommu: *mut amd_iommu, evt_type: *const u8, cntrl_intr: u8, cntrl_log: u8, status_run_mask: u32, status_overflow_mask: u32) {
let status: u32 = readl(unsafe { (*iommu).mmio_base }.wrapping_offset(8224 as isize) as *const ());
if status & status_run_mask != 0 {
return;
}
if __ratelimit__17!(unsafe { core::ptr::addr_of_mut!(_rs) }) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016AMD-Vi: IOMMU %s log restarting\n".as_ptr() as *const u8, evt_type) };
}
iommu_feature_disable(iommu, cntrl_log);
iommu_feature_disable(iommu, cntrl_intr);
writel(status_overflow_mask, unsafe { (*iommu).mmio_base }.wrapping_offset(8224 as isize) as *mut ());
iommu_feature_enable(iommu, cntrl_intr);
iommu_feature_enable(iommu, cntrl_log);
}

static mut _rs: ratelimit_state = {
let mut __t0: ratelimit_state = unsafe { core::mem::zeroed() };
__t0.lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.interval = 5000i32;
__t0.burst = 10i32;
__t0.flags = 0u32;
__t0
};

/// This function restarts event logging in case the IOMMU experienced
/// an event log buffer overflow.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.amd_init.amd_iommu_restart_event_logging")]
extern "C" fn amd_iommu_restart_event_logging(iommu: *mut amd_iommu) {
amd_iommu_restart_log(iommu, c"Event".as_ptr() as *const u8, 3u8, 2u8, BIT!(3i32) as u32, BIT!(0i32) as u32);
}

/// This function restarts event logging in case the IOMMU experienced
/// GA log overflow.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.amd_init.amd_iommu_restart_ga_log")]
extern "C" fn amd_iommu_restart_ga_log(iommu: *mut amd_iommu) {
amd_iommu_restart_log(iommu, c"GA".as_ptr() as *const u8, 29u8, 28u8, BIT!(8i32) as u32, BIT!(9i32) as u32);
}

/// This function resets the command buffer if the IOMMU stopped fetching
/// commands from it.
fn amd_iommu_reset_cmd_buffer(iommu: *mut amd_iommu) {
iommu_feature_disable(iommu, 12u8);
writel(0u32, unsafe { (*iommu).mmio_base }.wrapping_offset(8192 as isize) as *mut ());
writel(0u32, unsafe { (*iommu).mmio_base }.wrapping_offset(8200 as isize) as *mut ());
unsafe {
(*iommu).cmd_buf_head = 0;
}
unsafe {
(*iommu).cmd_buf_tail = 0;
}
iommu_feature_enable(iommu, 12u8);
}

/// This function writes the command buffer address to the hardware and
/// enables it.
fn iommu_enable_command_buffer(iommu: *mut amd_iommu) {
let mut entry: u64 = 0;
loop {
if unlikely!(unsafe { (*iommu).cmd_buf }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("628: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 628b - ., 3; .popsection\n# {0}", const 628 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_iommu_enable_command_buffer_0, sym __slate_asm_string_iommu_enable_command_buffer_1, const 803 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if true {
break;
}
}
if !is_kdump_kernel() {
// Command buffer is re-used for kdump kernel and setting
// of MMIO register is not required.
entry = iommu_virt_to_phys((unsafe { (*iommu).cmd_buf }) as *mut ());
entry |= 648518346341351424u64;
unsafe { memcpy_toio(unsafe { (*iommu).mmio_base }.wrapping_offset(8 as isize) as *mut (), core::ptr::addr_of_mut!(entry) as *const (), core::mem::size_of::<u64>() as u64) };
}
amd_iommu_reset_cmd_buffer(iommu);
}

/// This function disables the command buffer
fn iommu_disable_command_buffer(iommu: *mut amd_iommu) {
iommu_feature_disable(iommu, 12u8);
}

#[unsafe(link_section = ".init.text")]
fn free_command_buffer(iommu: *mut amd_iommu) {
unsafe { iommu_free_pages((*iommu).cmd_buf as *mut ()) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn iommu_alloc_4k_pages(iommu: *mut amd_iommu, gfp: u32, mut size: u64) -> *mut () {
let mut nid: i32 = 0;
let __v0: i32 = if !unsafe { (*iommu).dev }.is_null() {
dev_to_node(unsafe { core::ptr::addr_of_mut!((*(*iommu).dev).dev) })
} else {
1i32.wrapping_neg()
};
nid = __v0;
size = PAGE_ALIGN!(size);
let buf: *mut () = unsafe { iommu_alloc_pages_node_sz(nid, gfp, size) };
if buf.is_null() {
return core::ptr::null_mut::<()>();
}
let __v1: bool = check_feature(BIT_ULL!(63i32)) && (unsafe { set_memory_4k(buf as u64, (size / (_AC!() << 12i32)) as i32) }) != 0;
if __v1 {
unsafe { iommu_free_pages(buf) };
return core::ptr::null_mut::<()>();
}
buf
}

/// allocates the memory where the IOMMU will log its events to
#[unsafe(link_section = ".init.text")]
fn alloc_event_buffer() -> i32 {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
unsafe {
(*iommu).evt_buf = iommu_alloc_4k_pages(iommu, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), amd_iommu_evtlog_size as u64) as *mut u8;
}
if unsafe { (*iommu).evt_buf }.is_null() {
return 12i32.wrapping_neg();
}
iommu = list_next_entry__3!(iommu, list);
}
0
}

fn iommu_enable_event_buffer() {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
let mut entry: u64 = 0;
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
loop {
if unlikely!(unsafe { (*iommu).evt_buf }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("629: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 629b - ., 3; .popsection\n# {0}", const 629 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_iommu_enable_event_buffer_1, sym __slate_asm_string_iommu_enable_event_buffer_2, const 872 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if true {
break;
}
}
if !is_kdump_kernel() {
// Event buffer is re-used for kdump kernel and setting
// of MMIO register is not required.
entry = iommu_virt_to_phys((unsafe { (*iommu).evt_buf }) as *mut ());
entry |= if (unsafe { amd_iommu_evtlog_size }) == 8192 { 648518346341351424u64 } else { 1080863910568919040u64 };
unsafe { memcpy_toio(unsafe { (*iommu).mmio_base }.wrapping_offset(16 as isize) as *mut (), core::ptr::addr_of_mut!(entry) as *const (), core::mem::size_of::<u64>() as u64) };
}
// set head and tail to zero manually
writel(0u32, unsafe { (*iommu).mmio_base }.wrapping_offset(8208 as isize) as *mut ());
writel(0u32, unsafe { (*iommu).mmio_base }.wrapping_offset(8216 as isize) as *mut ());
iommu_feature_enable(iommu, 2u8);
iommu = list_next_entry__3!(iommu, list);
}
}

/// This function disables the event log buffer
fn iommu_disable_event_buffer(iommu: *mut amd_iommu) {
iommu_feature_disable(iommu, 2u8);
}

#[unsafe(link_section = ".init.text")]
fn free_event_buffer(iommu: *mut amd_iommu) {
unsafe { iommu_free_pages((*iommu).evt_buf as *mut ()) };
}

fn free_ga_log(iommu: *mut amd_iommu) {
unsafe { iommu_free_pages((*iommu).ga_log as *mut ()) };
unsafe {
(*iommu).ga_log = core::ptr::null_mut::<u8>();
}
unsafe { iommu_free_pages((*iommu).ga_log_tail as *mut ()) };
unsafe {
(*iommu).ga_log_tail = core::ptr::null_mut::<u8>();
}
}

fn iommu_ga_log_enable(iommu: *mut amd_iommu) -> i32 {
let mut status: u32 = 0;
let mut i: u32 = 0;
let mut entry: u64 = 0;
if unsafe { (*iommu).ga_log }.is_null() {
return 22i32.wrapping_neg();
}
entry = iommu_virt_to_phys((unsafe { (*iommu).ga_log }) as *mut ()) | 576460752303423488u64;
unsafe { memcpy_toio(unsafe { (*iommu).mmio_base }.wrapping_offset(224 as isize) as *mut (), core::ptr::addr_of_mut!(entry) as *const (), core::mem::size_of::<u64>() as u64) };
entry = iommu_virt_to_phys((unsafe { (*iommu).ga_log_tail }) as *mut ()) & BIT_ULL!(52i32).wrapping_sub(1) & !7;
unsafe { memcpy_toio(unsafe { (*iommu).mmio_base }.wrapping_offset(232 as isize) as *mut (), core::ptr::addr_of_mut!(entry) as *const (), core::mem::size_of::<u64>() as u64) };
writel(0u32, unsafe { (*iommu).mmio_base }.wrapping_offset(8256 as isize) as *mut ());
writel(0u32, unsafe { (*iommu).mmio_base }.wrapping_offset(8264 as isize) as *mut ());
iommu_feature_enable(iommu, 29u8);
iommu_feature_enable(iommu, 28u8);
while i < 2000000 {
status = readl(unsafe { (*iommu).mmio_base }.wrapping_offset(8224 as isize) as *const ());
if (status as u64) & BIT!(8i32) != 0 {
break;
}
udelay(10u64);
i = i.wrapping_add(1);
}
let mut __v3: i64;
let __ret_warn_on: i32 = (i >= 2000000) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("630: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 630b - ., 3; .popsection\n# {0}", const 630 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_ga_log_enable_4, sym __slate_asm_string_iommu_ga_log_enable_5, const 948 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("631: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 631b - ., 4; .popsection\n# {0}", const 631 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
if __v3 != 0 {
return 22i32.wrapping_neg();
}
0
}

fn iommu_init_ga_log(iommu: *mut amd_iommu) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut __v3: i32 = unsafe { core::mem::zeroed() };
let mut nid: i32 = unsafe { core::mem::zeroed() };
unsafe {
if !unsafe { (*iommu).dev }.is_null() {
__v3 = dev_to_node(core::ptr::addr_of_mut!((*(*iommu).dev).dev));
} else {
__v3 = 1i32.wrapping_neg();
}
nid = __v3;
__ret_warn_on = !(!(!AMD_IOMMU_GUEST_IR_VAPIC!(unsafe { amd_iommu_guest_ir }))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("632: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 632b - ., 3; .popsection\n# {0}", const 632 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_init_ga_log_5, sym __slate_asm_string_iommu_init_ga_log_6, const 958 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("633: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 633b - ., 4; .popsection\n# {0}", const 633 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return 22i32.wrapping_neg();
} else {
if !unsafe { (*iommu).ga_log }.is_null() && !unsafe { (*iommu).ga_log_tail }.is_null() {
return 0;
} else {
(*iommu).ga_log = iommu_alloc_pages_node_sz(nid, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), 4096u64) as *mut u8;
if !unsafe { (*iommu).ga_log }.is_null() {
(*iommu).ga_log_tail = iommu_alloc_pages_node_sz(nid, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), 8u64) as *mut u8;
if !unsafe { (*iommu).ga_log_tail }.is_null() {
return 0;
}
}
free_ga_log(iommu);
return 22i32.wrapping_neg();
}
}
}
}

#[unsafe(link_section = ".init.text")]
fn alloc_cwwb_sem(iommu: *mut amd_iommu) -> i32 {
unsafe {
(*iommu).cmd_sem = iommu_alloc_4k_pages(iommu, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), 1u64) as *mut u64;
}
if unsafe { (*iommu).cmd_sem }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*iommu).cmd_sem_paddr = iommu_virt_to_phys((*iommu).cmd_sem as *mut ());
}
0
}

#[unsafe(link_section = ".init.text")]
fn remap_event_buffer() -> i32 {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
let mut paddr: u64 = 0;
let mut __v0: i64;
let __ret_do_once: bool = true;
let mut __v1: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely!(__ret_cond) != 0 && unlikely!(!unsafe { __already_done__993 }) != 0 {
unsafe {
__already_done__993 = true;
}
__ret_once = true;
}
__v1 = __ret_once as i64;
if __v1 != 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016AMD-Vi: Re-using event buffer from the previous kernel\n".as_ptr() as *const u8) };
}
__v0 = __ret_do_once as i64;
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
paddr = readq(unsafe { (*iommu).mmio_base }.wrapping_offset(16 as isize) as *const ()) & 4503599627366400;
unsafe {
(*iommu).evt_buf = iommu_memremap(paddr, amd_iommu_evtlog_size as u64) as *mut u8;
}
if unsafe { (*iommu).evt_buf }.is_null() {
return 12i32.wrapping_neg();
}
iommu = list_next_entry__3!(iommu, list);
}
0
}

#[unsafe(link_section = ".data..once")]
static mut __already_done__993: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn remap_command_buffer(iommu: *mut amd_iommu) -> i32 {
let mut paddr: u64 = 0;
let mut __v0: i64;
let __ret_do_once: bool = true;
let mut __v1: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely!(__ret_cond) != 0 && unlikely!(!unsafe { __already_done__1008 }) != 0 {
unsafe {
__already_done__1008 = true;
}
__ret_once = true;
}
__v1 = __ret_once as i64;
if __v1 != 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016AMD-Vi: Re-using command buffer from the previous kernel\n".as_ptr() as *const u8) };
}
__v0 = __ret_do_once as i64;
paddr = readq(unsafe { (*iommu).mmio_base }.wrapping_offset(8 as isize) as *const ()) & 4503599627366400;
unsafe {
(*iommu).cmd_buf = iommu_memremap(paddr, 8192u64) as *mut u8;
}
if !unsafe { (*iommu).cmd_buf }.is_null() { 0 } else { 12i32.wrapping_neg() }
}

#[unsafe(link_section = ".data..once")]
static mut __already_done__1008: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn remap_or_alloc_cwwb_sem(iommu: *mut amd_iommu) -> i32 {
let mut paddr: u64 = 0;
if check_feature(BIT_ULL!(63i32)) {
// When SNP is enabled, the exclusion base register is used for the
// completion wait buffer (CWB) address. Read and re-use it.
let mut __v0: i64;
let __ret_do_once: bool = true;
let mut __v1: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely!(__ret_cond) != 0 && unlikely!(!unsafe { __already_done__1024 }) != 0 {
unsafe {
__already_done__1024 = true;
}
__ret_once = true;
}
__v1 = __ret_once as i64;
if __v1 != 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016AMD-Vi: Re-using CWB buffers from the previous kernel\n".as_ptr() as *const u8) };
}
__v0 = __ret_do_once as i64;
paddr = readq(unsafe { (*iommu).mmio_base }.wrapping_offset(32 as isize) as *const ()) & 4503599627366400;
unsafe {
(*iommu).cmd_sem = iommu_memremap(paddr, _AC!() << 12i32) as *mut u64;
}
if unsafe { (*iommu).cmd_sem }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*iommu).cmd_sem_paddr = paddr;
}
} else {
return alloc_cwwb_sem(iommu);
}
0
}

#[unsafe(link_section = ".data..once")]
static mut __already_done__1024: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn alloc_iommu_buffers(iommu: *mut amd_iommu) -> i32 {
let mut ret: i32 = 0;
// Reuse/Remap the previous kernel's allocated completion wait
// command and event buffers for kdump boot.
if is_kdump_kernel() {
ret = remap_or_alloc_cwwb_sem(iommu);
if ret != 0 {
return ret;
}
ret = remap_command_buffer(iommu);
if ret != 0 {
return ret;
}
} else {
ret = alloc_cwwb_sem(iommu);
if ret != 0 {
return ret;
}
ret = alloc_command_buffer(iommu);
if ret != 0 {
return ret;
}
}
0
}

#[unsafe(link_section = ".init.text")]
fn free_cwwb_sem(iommu: *mut amd_iommu) {
if !unsafe { (*iommu).cmd_sem }.is_null() {
unsafe { iommu_free_pages((*iommu).cmd_sem as *mut ()) };
}
}

#[unsafe(link_section = ".init.text")]
fn unmap_cwwb_sem(iommu: *mut amd_iommu) {
if !unsafe { (*iommu).cmd_sem }.is_null() {
if check_feature(BIT_ULL!(63i32)) {
unsafe { memunmap((*iommu).cmd_sem as *mut ()) };
} else {
unsafe { iommu_free_pages((*iommu).cmd_sem as *mut ()) };
}
}
}

#[unsafe(link_section = ".init.text")]
fn unmap_command_buffer(iommu: *mut amd_iommu) {
unsafe { memunmap((*iommu).cmd_buf as *mut ()) };
}

#[unsafe(link_section = ".init.text")]
fn unmap_event_buffer(iommu: *mut amd_iommu) {
unsafe { memunmap((*iommu).evt_buf as *mut ()) };
}

#[unsafe(link_section = ".init.text")]
fn free_iommu_buffers(iommu: *mut amd_iommu) {
if is_kdump_kernel() {
unmap_cwwb_sem(iommu);
unmap_command_buffer(iommu);
unmap_event_buffer(iommu);
} else {
free_cwwb_sem(iommu);
free_command_buffer(iommu);
free_event_buffer(iommu);
}
}

fn iommu_enable_xt(iommu: *mut amd_iommu) {
// XT mode (32-bit APIC destination ID) requires
// GA mode (128-bit IRTE support) as a prerequisite.
if AMD_IOMMU_GUEST_IR_GA!(unsafe { amd_iommu_guest_ir }) && (unsafe { amd_iommu_xt_mode }) == 1 {
iommu_feature_enable(iommu, 50u8);
}
}

fn iommu_enable_gt(iommu: *mut amd_iommu) {
if !check_feature(BIT_ULL!(4i32)) {
return;
}
iommu_feature_enable(iommu, 16u8);
// This feature needs to be enabled prior to a call
// to iommu_snp_enable(). Since this function is called
// in early_enable_iommu(), it is safe to enable here.
if check_feature2(BIT_ULL!(3i32)) {
iommu_feature_enable(iommu, 58u8);
}
}

/// sets a specific bit in the device table entry.
fn set_dte_bit(dte: *mut dev_table_entry, bit: u8) {
let i: i32 = (bit as i32) >> 6i32 & 3;
let _bit: i32 = (bit as i32) & 63;
let __v0: *mut u64 = unsafe { (*dte).__slate_anon_0.data.as_mut_ptr() as *mut u64 }.wrapping_offset(i as isize);
unsafe {
*__v0 = *__v0 | 1u64 << _bit;
}
}

fn __reuse_device_table(iommu: *mut amd_iommu) -> bool {
let pci_seg: *mut amd_iommu_pci_seg = unsafe { (*iommu).pci_seg };
let mut lo: u32 = 0;
let mut hi: u32 = 0;
let mut old_devtb_size: u32 = 0;
let mut devid: u32 = 0;
let mut old_devtb_phys: u64 = 0;
let mut dom_id: u16 = 0;
let mut dte_v: bool = false;
let mut entry: u64 = 0;
// Each IOMMU use separate device table with the same size
lo = readl(unsafe { (*iommu).mmio_base }.wrapping_offset(0 as isize) as *const ());
hi = readl(unsafe { (*iommu).mmio_base }.wrapping_offset(0 as isize).wrapping_offset(4 as isize) as *const ());
entry = ((hi as u64) << 32i32).wrapping_add(lo as u64);
old_devtb_size = ((entry & !(!(_AC!() << 12i32).wrapping_sub(1))).wrapping_add(1) << 12i32) as u32;
if old_devtb_size != unsafe { (*pci_seg).dev_table_size } {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013AMD-Vi: The device table size of IOMMU:%d is not expected!\n".as_ptr() as *const u8, (*iommu).index) };
return false;
}
// When SME is enabled in the first kernel, the entry includes the
// memory encryption mask(sme_me_mask), we must remove the memory
// encryption mask to obtain the true physical address in kdump kernel.
old_devtb_phys = __sme_clr!(entry) & !(_AC!() << 12i32).wrapping_sub(1);
if old_devtb_phys >= 4294967296 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x013AMD-Vi: The address of old device table is above 4G, not trustworthy!\n".as_ptr() as *const u8) };
return false;
}
// Re-use the previous kernel's device table for kdump.
unsafe {
(*pci_seg).old_dev_tbl_cpy = iommu_memremap(old_devtb_phys, (*pci_seg).dev_table_size as u64) as *mut dev_table_entry;
}
if unsafe { (*pci_seg).old_dev_tbl_cpy }.is_null() {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x013AMD-Vi: Failed to remap memory for reusing old device table!\n".as_ptr() as *const u8) };
return false;
}
while devid <= (((unsafe { (*pci_seg).last_bdf }) as i32) as u32) {
let old_dev_tbl_entry: *mut dev_table_entry = unsafe { (*pci_seg).old_dev_tbl_cpy }.wrapping_offset(devid as isize);
let mut __v6: u64;
loop {
if __bf_cast_unsigned!(1u64) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_634() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v8: u64;
loop {
if false {
unsafe { __compiletime_assert_635() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 1u64 == 0 {
unsafe { __compiletime_assert_636() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !(1u64 >> (1u64.trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_637() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 1u64.wrapping_add(1u64 << (1u64.trailing_zeros() as i32)) & 1u64.wrapping_add(1u64 << (1u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_638() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = ((unsafe { *((*old_dev_tbl_entry).__slate_anon_0.data.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) }) & 1u64) >> (1u64.trailing_zeros() as i32);
__v6 = __v8;
dte_v = __v6 != 0;
let mut __v13: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_init_c_1188_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_639() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v15: u64;
loop {
if false {
unsafe { __compiletime_assert_640() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_init_c_1188_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) == 0 {
unsafe { __compiletime_assert_641() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_init_c_1188_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) >> ((core::mem::size_of::<__SlateRecord_init_c_1188_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_642() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_init_c_1188_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_1188_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_init_c_1188_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_1188_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_643() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v15 = ((unsafe { *((*old_dev_tbl_entry).__slate_anon_0.data.as_mut_ptr() as *mut u64).wrapping_offset(1 as isize) }) & (core::mem::size_of::<__SlateRecord_init_c_1188_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))) >> ((core::mem::size_of::<__SlateRecord_init_c_1188_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32);
__v13 = __v15;
dom_id = __v13 as u16;
if !(!dte_v || dom_id == 0) {
// ID reservation can fail with -ENOSPC when there
// are multiple devices present in the same domain,
// hence check only for -ENOMEM.
if (unsafe { amd_iommu_pdom_id_reserve(dom_id, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) }) == 12i32.wrapping_neg() {
return false;
}
}
devid = devid.wrapping_add(1);
}
true
}

fn reuse_device_table() -> bool {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
let mut pci_seg: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
if !unsafe { amd_iommu_pre_enabled } {
return false;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014AMD-Vi: Translation is already enabled - trying to reuse translation structures\n".as_ptr() as *const u8) };
// All IOMMUs within PCI segment shares common device table.
// Hence reuse device table only once per PCI segment.
pci_seg = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) });
while list_entry_is_head!(pci_seg, unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) }, list) == 0 {
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
if ((unsafe { (*pci_seg).id }) as i32) == ((unsafe { (*(*iommu).pci_seg).id }) as i32) {
if !__reuse_device_table(iommu) {
return false;
}
break;
}
iommu = list_next_entry__3!(iommu, list);
}
pci_seg = list_next_entry__5!(pci_seg, list);
}
true
}

#[unsafe(no_mangle)]
extern "C" fn amd_iommu_get_ivhd_dte_flags(segid: u16, devid: u16) -> *mut dev_table_entry {
let mut e: *mut ivhd_dte_flags = unsafe { core::mem::zeroed() };
let mut best_len: u32 = !0;
let mut dte: *mut dev_table_entry = core::ptr::null_mut::<dev_table_entry>();
e = list_first_entry__32!(unsafe { core::ptr::addr_of_mut!(amd_ivhd_dev_flags_list) });
while list_entry_is_head!(e, unsafe { core::ptr::addr_of_mut!(amd_ivhd_dev_flags_list) }, list) == 0 {
// Need to go through the whole list to find the smallest range,
// which contains the devid.
if ((unsafe { (*e).segid }) as i32) == (segid as i32) && ((unsafe { (*e).devid_first }) as i32) <= (devid as i32) && (devid as i32) <= ((unsafe { (*e).devid_last }) as i32) {
let len: u32 = ((unsafe { (*e).devid_last }) as i32).wrapping_sub((unsafe { (*e).devid_first }) as i32) as u32;
if len < best_len {
dte = unsafe { core::ptr::addr_of_mut!((*e).dte) };
best_len = len;
}
}
e = list_next_entry__36!(e, list);
}
dte
}

fn search_ivhd_dte_flags(segid: u16, first: u16, last: u16) -> bool {
let mut e: *mut ivhd_dte_flags = unsafe { core::mem::zeroed() };
e = list_first_entry__32!(unsafe { core::ptr::addr_of_mut!(amd_ivhd_dev_flags_list) });
while list_entry_is_head!(e, unsafe { core::ptr::addr_of_mut!(amd_ivhd_dev_flags_list) }, list) == 0 {
if ((unsafe { (*e).segid }) as i32) == (segid as i32) && ((unsafe { (*e).devid_first }) as i32) == (first as i32) && ((unsafe { (*e).devid_last }) as i32) == (last as i32) {
return true;
}
e = list_next_entry__36!(e, list);
}
false
}

// This function takes the device specific flags read from the ACPI
// table and sets up the device table entry with that information
#[unsafe(link_section = ".init.text")]
fn set_dev_entry_from_acpi_range(iommu: *mut amd_iommu, first: u16, last: u16, flags: u32, ext_flags: u32) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut dte: dev_table_entry = {
let mut __t0: dev_table_entry = unsafe { core::mem::zeroed() };
__t0
};
// Parse IVHD DTE setting flags and store information
if flags != 0 {
let mut d: *mut ivhd_dte_flags = unsafe { core::mem::zeroed() };
if search_ivhd_dte_flags(unsafe { (*(*iommu).pci_seg).id }, first, last) {
return;
}
let mut __v0: *mut ivhd_dte_flags;
let __obj_size: u64 = size_mul(core::mem::size_of::<ivhd_dte_flags>() as u64, 1u64);
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
__v0 = __v1 as *mut ivhd_dte_flags;
d = __v0;
if d.is_null() {
return;
}
let mut __v6: i32;
if false {
unsafe { _printk(c"\x017AMD-Vi: %s: devid range %#x:%#x\n".as_ptr() as *const u8, c"set_dev_entry_from_acpi_range".as_ptr() as *mut u8, first as i32, last as i32) };
}
__v6 = 0;
if flags & 1 != 0 {
set_dte_bit(core::ptr::addr_of_mut!(dte), 184u8);
}
if flags & 2 != 0 {
set_dte_bit(core::ptr::addr_of_mut!(dte), 185u8);
}
if flags & 4 != 0 {
set_dte_bit(core::ptr::addr_of_mut!(dte), 186u8);
}
if flags & 16 != 0 {
set_dte_bit(core::ptr::addr_of_mut!(dte), 104u8);
}
if flags & 32 != 0 {
set_dte_bit(core::ptr::addr_of_mut!(dte), 105u8);
}
if flags & 64 != 0 {
set_dte_bit(core::ptr::addr_of_mut!(dte), 190u8);
}
if flags & 128 != 0 {
set_dte_bit(core::ptr::addr_of_mut!(dte), 191u8);
}
// Apply erratum 63, which needs info in initial_dte
let mut __v7: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_init_c_1308_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41))) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_644() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v9: u64;
loop {
if false {
unsafe { __compiletime_assert_645() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_init_c_1308_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41)) == 0 {
unsafe { __compiletime_assert_646() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_init_c_1308_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41)) >> ((core::mem::size_of::<__SlateRecord_init_c_1308_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_647() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_init_c_1308_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_1308_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_init_c_1308_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_1308_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_648() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = ((unsafe { dte.__slate_anon_0.data[1] }) & (core::mem::size_of::<__SlateRecord_init_c_1308_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41))) >> ((core::mem::size_of::<__SlateRecord_init_c_1308_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 40i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(41)).trailing_zeros() as i32);
__v7 = __v9;
if __v7 == 1 {
let __v14: *mut u64 = unsafe { dte.__slate_anon_0.data.as_mut_ptr() as *mut u64 }.wrapping_offset(0 as isize);
let __v15: u64 = unsafe { *__v14 };
let __v16: u64 = __v15 | BIT_ULL!(62i32);
unsafe {
*__v14 = __v16;
}
}
unsafe { memcpy(core::ptr::addr_of_mut!((*d).dte) as *mut (), core::ptr::addr_of_mut!(dte) as *const (), core::mem::size_of::<dev_table_entry>() as u64) };
unsafe {
(*d).segid = (*(*iommu).pci_seg).id;
}
unsafe {
(*d).devid_first = first;
}
unsafe {
(*d).devid_last = last;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*d).list) }, unsafe { core::ptr::addr_of_mut!(amd_ivhd_dev_flags_list) });
}
i = first as i32;
while i <= (last as i32) {
if flags != 0 {
let dev_table: *mut dev_table_entry = unsafe { get_dev_table(iommu) };
unsafe { memcpy(dev_table.wrapping_offset(i as isize) as *mut (), core::ptr::addr_of_mut!(dte) as *const (), core::mem::size_of::<dev_table_entry>() as u64) };
}
unsafe { amd_iommu_set_rlookup_table(iommu, i as u16) };
i = i.wrapping_add(1);
}
}

#[unsafe(link_section = ".init.text")]
fn set_dev_entry_from_acpi(iommu: *mut amd_iommu, devid: u16, flags: u32, ext_flags: u32) {
set_dev_entry_from_acpi_range(iommu, devid, devid, flags, ext_flags);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn add_special_device(r#type: u8, id: u8, devid: *mut u32, cmd_line: bool) -> i32 {
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut entry: *mut devid_map = unsafe { core::mem::zeroed() };
let mut list: *mut list_head = unsafe { core::mem::zeroed() };
if (r#type as i32) == 1 {
list = unsafe { core::ptr::addr_of_mut!(ioapic_map) };
} else {
if (r#type as i32) == 2 {
list = unsafe { core::ptr::addr_of_mut!(hpet_map) };
} else {
return 22i32.wrapping_neg();
}
}
entry = list_first_entry__19!(list);
while list_entry_is_head!(entry, list, list) == 0 {
if ((unsafe { (*entry).id }) as i32) == (id as i32) && unsafe { (*entry).cmd_line } {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016AMD-Vi: Command-line override present for %s id %d - ignoring\n".as_ptr() as *const u8, if (r#type as i32) == 1 { c"IOAPIC".as_ptr() as *mut u8 } else { c"HPET".as_ptr() as *mut u8 }, id as i32) };
unsafe {
*devid = (*entry).devid;
}
return 0;
}
entry = list_next_entry__18!(entry, list);
}
let mut __v5: *mut devid_map;
let __obj_size: u64 = size_mul(core::mem::size_of::<devid_map>() as u64, 1u64);
let mut __v6: *mut ();
let mut __v7: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_8 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_8)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
} });
}
__v7 = _res;
__v6 = __v7;
__v5 = __v6 as *mut devid_map;
entry = __v5;
if entry.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*entry).id = id;
}
unsafe {
(*entry).devid = *devid;
}
unsafe {
(*entry).cmd_line = cmd_line;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*entry).list) }, list);
0
}

#[unsafe(link_section = ".init.text")]
fn add_acpi_hid_device(hid: *mut u8, uid: *mut u8, devid: *mut u32, cmd_line: bool) -> i32 {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut entry: *mut acpihid_map_entry = unsafe { core::mem::zeroed() };
let list: *mut list_head = unsafe { core::ptr::addr_of_mut!(acpihid_map) };
entry = list_first_entry__23!(list);
while list_entry_is_head!(entry, list, list) == 0 {
let __v1: bool = (unsafe { strcmp(((*entry).hid.as_mut_ptr() as *mut u8) as *const u8, hid as *const u8) }) != 0 || (unsafe { *uid }) != 0 && (unsafe { *((*entry).uid.as_mut_ptr() as *mut u8) }) != 0 && (unsafe { strcmp(((*entry).uid.as_mut_ptr() as *mut u8) as *const u8, uid as *const u8) }) != 0;
if !(__v1 || !unsafe { (*entry).cmd_line }) {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016AMD-Vi: Command-line override for hid:%s uid:%s\n".as_ptr() as *const u8, hid, uid) };
unsafe {
*devid = (*entry).devid;
}
return 0;
}
entry = list_next_entry__34!(entry, list);
}
let mut __v7: *mut acpihid_map_entry;
let __obj_size: u64 = size_mul(core::mem::size_of::<acpihid_map_entry>() as u64, 1u64);
let mut __v8: *mut ();
let mut __v9: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
} });
}
__v9 = _res;
__v8 = __v9;
__v7 = __v8 as *mut acpihid_map_entry;
entry = __v7;
if entry.is_null() {
return 12i32.wrapping_neg();
}
unsafe { memcpy(((*entry).uid.as_mut_ptr() as *mut u8) as *mut (), uid as *const (), strlen(uid as *const u8)) };
unsafe { memcpy(((*entry).hid.as_mut_ptr() as *mut u8) as *mut (), hid as *const (), strlen(hid as *const u8)) };
unsafe {
(*entry).devid = *devid;
}
unsafe {
(*entry).cmd_line = cmd_line;
}
unsafe {
(*entry).root_devid = (*entry).devid & 4294967288u32;
}
let mut __v13: i32;
while false {
}
__v13 = unsafe { _printk(c"\x016AMD-Vi: %s, add hid:%s, uid:%s, rdevid:%#x\n".as_ptr() as *const u8, if (*entry).cmd_line { c"cmd".as_ptr() as *mut u8 } else { c"ivrs".as_ptr() as *mut u8 }, (*entry).hid.as_mut_ptr() as *mut u8, (*entry).uid.as_mut_ptr() as *mut u8, (*entry).root_devid) };
list_add_tail(unsafe { core::ptr::addr_of_mut!((*entry).list) }, list);
0
}

#[unsafe(link_section = ".init.text")]
fn add_early_maps() -> i32 {
let mut i: i32 = 0;
let mut ret: i32 = 0;
while i < unsafe { early_ioapic_map_size } {
ret = add_special_device(1u8, unsafe { early_ioapic_map.0[i as usize].id }, unsafe { core::ptr::addr_of_mut!(early_ioapic_map.0[i as usize].devid) }, unsafe { early_ioapic_map.0[i as usize].cmd_line });
if ret != 0 {
return ret;
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { early_hpet_map_size } {
ret = add_special_device(2u8, unsafe { early_hpet_map.0[i as usize].id }, unsafe { core::ptr::addr_of_mut!(early_hpet_map.0[i as usize].devid) }, unsafe { early_hpet_map.0[i as usize].cmd_line });
if ret != 0 {
return ret;
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { early_acpihid_map_size } {
ret = add_acpi_hid_device(unsafe { core::ptr::addr_of_mut!(early_acpihid_map.0[i as usize].hid) as *mut u8 }, unsafe { core::ptr::addr_of_mut!(early_acpihid_map.0[i as usize].uid) as *mut u8 }, unsafe { core::ptr::addr_of_mut!(early_acpihid_map.0[i as usize].devid) }, unsafe { early_acpihid_map.0[i as usize].cmd_line });
if ret != 0 {
return ret;
}
i = i.wrapping_add(1);
}
0
}

// Takes a pointer to an AMD IOMMU entry in the ACPI table and
// initializes the hardware and our data structures with it.
#[unsafe(link_section = ".init.text")]
fn init_iommu_from_acpi(iommu: *mut amd_iommu, h: *mut ivhd_header) -> i32 {
let mut p: *mut u8 = h as *mut u8;
let mut end: *mut u8 = p;
let mut flags: u8 = 0;
let mut devid: u16 = 0;
let mut devid_start: u16 = 0;
let mut devid_to: u16 = 0;
let mut seg_id: u16 = 0;
let mut dev_i: u32 = 0;
let mut ext_flags: u32 = 0;
let mut alias: bool = false;
let mut e: *mut ivhd_entry = unsafe { core::mem::zeroed() };
let pci_seg: *mut amd_iommu_pci_seg = unsafe { (*iommu).pci_seg };
let mut ivhd_size: u32 = 0;
let ret: i32 = add_early_maps();
if ret != 0 {
return ret;
}
unsafe { amd_iommu_apply_ivrs_quirks() };
// First save the recommended feature enable bits from ACPI
unsafe {
(*iommu).acpi_flags = (*h).flags;
}
// Done. Now parse the device entries
ivhd_size = get_ivhd_header_size(h);
if ivhd_size == 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013AMD-Vi: Unsupported IVHD type %#x\n".as_ptr() as *const u8, (*h).r#type as i32) };
return 22i32.wrapping_neg();
}
let __v2: *mut u8 = p;
p = __v2.wrapping_offset(ivhd_size as isize);
let __v4: *mut u8 = end;
end = __v4.wrapping_offset(((unsafe { (*h).length }) as i32) as isize);
while p < end {
e = p as *mut ivhd_entry;
seg_id = unsafe { (*pci_seg).id };
match (unsafe { (*e).r#type }) as i32 {
1 => {
loop {
if unsafe { amd_iommu_dump } {
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x016AMD-Vi:   DEV_ALL\t\t\tsetting: %#02x\n".as_ptr() as *const u8, (*e).flags as i32) };
}
if true {
break;
}
}
set_dev_entry_from_acpi_range(iommu, 0u16, unsafe { (*pci_seg).last_bdf }, (unsafe { (*e).flags }) as u32, 0u32);
}
2 => {
loop {
if unsafe { amd_iommu_dump } {
let mut __v10: i32;
while false {
}
__v10 = unsafe { _printk(c"\x016AMD-Vi:   DEV_SELECT\t\t\tdevid: %04x:%02x:%02x.%x flags: %#02x\n".as_ptr() as *const u8, seg_id as i32, PCI_BUS_NUM!(unsafe { (*e).devid }), PCI_SLOT!(unsafe { (*e).devid }), PCI_FUNC!(unsafe { (*e).devid }), (*e).flags as i32) };
}
if true {
break;
}
}
devid = unsafe { (*e).devid };
set_dev_entry_from_acpi(iommu, devid, (unsafe { (*e).flags }) as u32, 0u32);
}
3 => {
loop {
if unsafe { amd_iommu_dump } {
let mut __v13: i32;
while false {
}
__v13 = unsafe { _printk(c"\x016AMD-Vi:   DEV_SELECT_RANGE_START\tdevid: %04x:%02x:%02x.%x flags: %#02x\n".as_ptr() as *const u8, seg_id as i32, PCI_BUS_NUM!(unsafe { (*e).devid }), PCI_SLOT!(unsafe { (*e).devid }), PCI_FUNC!(unsafe { (*e).devid }), (*e).flags as i32) };
}
if true {
break;
}
}
devid_start = unsafe { (*e).devid };
flags = unsafe { (*e).flags };
ext_flags = 0;
alias = false;
}
66 => {
loop {
if unsafe { amd_iommu_dump } {
let mut __v16: i32;
while false {
}
__v16 = unsafe { _printk(c"\x016AMD-Vi:   DEV_ALIAS\t\t\tdevid: %04x:%02x:%02x.%x flags: %#02x devid_to: %02x:%02x.%x\n".as_ptr() as *const u8, seg_id as i32, PCI_BUS_NUM!(unsafe { (*e).devid }), PCI_SLOT!(unsafe { (*e).devid }), PCI_FUNC!(unsafe { (*e).devid }), (*e).flags as i32, PCI_BUS_NUM__3!((*e).__slate_anon_3.__slate_anon_0.ext >> 8i32), PCI_SLOT__2!((*e).__slate_anon_3.__slate_anon_0.ext >> 8i32), PCI_FUNC__3!((*e).__slate_anon_3.__slate_anon_0.ext >> 8i32)) };
}
if true {
break;
}
}
devid = unsafe { (*e).devid };
devid_to = ((unsafe { (*e).__slate_anon_3.__slate_anon_0.ext }) >> 8i32) as u16;
set_dev_entry_from_acpi(iommu, devid, (unsafe { (*e).flags }) as u32, 0u32);
set_dev_entry_from_acpi(iommu, devid_to, (unsafe { (*e).flags }) as u32, 0u32);
unsafe {
*unsafe { (*pci_seg).alias_table }.wrapping_offset((devid as i32) as isize) = devid_to;
}
}
67 => {
loop {
if unsafe { amd_iommu_dump } {
let mut __v19: i32;
while false {
}
__v19 = unsafe { _printk(c"\x016AMD-Vi:   DEV_ALIAS_RANGE\t\tdevid: %04x:%02x:%02x.%x flags: %#02x devid_to: %04x:%02x:%02x.%x\n".as_ptr() as *const u8, seg_id as i32, PCI_BUS_NUM!(unsafe { (*e).devid }), PCI_SLOT!(unsafe { (*e).devid }), PCI_FUNC!(unsafe { (*e).devid }), (*e).flags as i32, seg_id as i32, PCI_BUS_NUM__3!((*e).__slate_anon_3.__slate_anon_0.ext >> 8i32), PCI_SLOT__2!((*e).__slate_anon_3.__slate_anon_0.ext >> 8i32), PCI_FUNC__3!((*e).__slate_anon_3.__slate_anon_0.ext >> 8i32)) };
}
if true {
break;
}
}
devid_start = unsafe { (*e).devid };
flags = unsafe { (*e).flags };
devid_to = ((unsafe { (*e).__slate_anon_3.__slate_anon_0.ext }) >> 8i32) as u16;
ext_flags = 0;
alias = true;
}
70 => {
loop {
if unsafe { amd_iommu_dump } {
let mut __v22: i32;
while false {
}
__v22 = unsafe { _printk(c"\x016AMD-Vi:   DEV_EXT_SELECT\t\tdevid: %04x:%02x:%02x.%x flags: %#02x ext: %08x\n".as_ptr() as *const u8, seg_id as i32, PCI_BUS_NUM!(unsafe { (*e).devid }), PCI_SLOT!(unsafe { (*e).devid }), PCI_FUNC!(unsafe { (*e).devid }), (*e).flags as i32, (*e).__slate_anon_3.__slate_anon_0.ext) };
}
if true {
break;
}
}
devid = unsafe { (*e).devid };
set_dev_entry_from_acpi(iommu, devid, (unsafe { (*e).flags }) as u32, unsafe { (*e).__slate_anon_3.__slate_anon_0.ext });
}
71 => {
loop {
if unsafe { amd_iommu_dump } {
let mut __v25: i32;
while false {
}
__v25 = unsafe { _printk(c"\x016AMD-Vi:   DEV_EXT_SELECT_RANGE\tdevid: %04x:%02x:%02x.%x flags: %#02x ext: %08x\n".as_ptr() as *const u8, seg_id as i32, PCI_BUS_NUM!(unsafe { (*e).devid }), PCI_SLOT!(unsafe { (*e).devid }), PCI_FUNC!(unsafe { (*e).devid }), (*e).flags as i32, (*e).__slate_anon_3.__slate_anon_0.ext) };
}
if true {
break;
}
}
devid_start = unsafe { (*e).devid };
flags = unsafe { (*e).flags };
ext_flags = unsafe { (*e).__slate_anon_3.__slate_anon_0.ext };
alias = false;
}
4 => {
loop {
if unsafe { amd_iommu_dump } {
let mut __v28: i32;
while false {
}
__v28 = unsafe { _printk(c"\x016AMD-Vi:   DEV_RANGE_END\t\tdevid: %04x:%02x:%02x.%x\n".as_ptr() as *const u8, seg_id as i32, PCI_BUS_NUM!(unsafe { (*e).devid }), PCI_SLOT!(unsafe { (*e).devid }), PCI_FUNC!(unsafe { (*e).devid })) };
}
if true {
break;
}
}
devid = unsafe { (*e).devid };
if alias {
dev_i = devid_start as u32;
while dev_i <= ((devid as i32) as u32) {
unsafe {
*unsafe { (*pci_seg).alias_table }.wrapping_offset(dev_i as isize) = devid_to;
}
dev_i = dev_i.wrapping_add(1);
}
set_dev_entry_from_acpi(iommu, devid_to, flags as u32, ext_flags);
}
set_dev_entry_from_acpi_range(iommu, devid_start, devid, flags as u32, ext_flags);
}
72 => {
let mut handle: u8 = 0;
let mut r#type: u8 = 0;
let mut var: *const u8 = unsafe { core::mem::zeroed() };
let mut devid: u32 = 0;
let mut ret: i32 = 0;
handle = ((unsafe { (*e).__slate_anon_3.__slate_anon_0.ext }) & 255) as u8;
devid = PCI_SEG_DEVID_TO_SBDF__2!(seg_id, (unsafe { (*e).__slate_anon_3.__slate_anon_0.ext }) >> 8i32);
r#type = ((unsafe { (*e).__slate_anon_3.__slate_anon_0.ext }) >> 24i32 & 255) as u8;
if (r#type as i32) == 1 {
var = c"IOAPIC".as_ptr() as *const u8;
} else {
if (r#type as i32) == 2 {
var = c"HPET".as_ptr() as *const u8;
} else {
var = c"UNKNOWN".as_ptr() as *const u8;
}
}
loop {
if unsafe { amd_iommu_dump } {
let mut __v34: i32;
while false {
}
__v34 = unsafe { _printk(c"\x016AMD-Vi:   DEV_SPECIAL(%s[%d])\t\tdevid: %04x:%02x:%02x.%x, flags: %#02x\n".as_ptr() as *const u8, var, handle as i32, seg_id as i32, PCI_BUS_NUM__3!(devid), PCI_SLOT__2!(devid), PCI_FUNC__3!(devid), (*e).flags as i32) };
}
if true {
break;
}
}
ret = add_special_device(r#type, handle, core::ptr::addr_of_mut!(devid), false);
if ret != 0 {
return ret;
}
// add_special_device might update the devid in case a
// command-line override is present. So call
// set_dev_entry_from_acpi after add_special_device.
set_dev_entry_from_acpi(iommu, devid as u16, (unsafe { (*e).flags }) as u32, 0u32);
}
240 => {
let mut devid: u32 = 0;
let mut hid: [u8; 9] = [0u8; 9];
let mut uid: __SlateAlign16<[u8; 256]> = __SlateAlign16([0u8; 256]);
let mut ret: i32 = 0;
if ((unsafe { (*h).r#type }) as i32) != 64 {
let mut __v37: i32;
while false {
}
__v37 = unsafe { _printk(c"\x013AMD-Vi: [Firmware Bug]: Invalid IVHD device type %#x\n".as_ptr() as *const u8, (*e).r#type as i32) };
} else {
loop {
if ((const { core::mem::size_of::<__SlateRecord_init_c_125_3>() }) as u64) != 8u64 {
unsafe { __compiletime_assert_649() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe { memcpy(hid.as_mut_ptr() as *mut (), core::ptr::addr_of_mut!((*e).__slate_anon_3.ext_hid) as *const (), 8u64) };
hid[8i32 as usize] = 0;
if (unsafe { *(hid.as_mut_ptr() as *mut u8) }) == 0 {
let mut __v40: i32;
while false {
}
__v40 = unsafe { _printk(c"\x013AMD-Vi: [Firmware Bug]: Invalid HID.\n".as_ptr() as *const u8) };
} else {
uid.0[0] = 0;
match (unsafe { (*e).uidf }) as i32 {
0 => {
if ((unsafe { (*e).uidl }) as i32) != 0 {
let mut __v42: i32;
while false {
}
__v42 = unsafe { _printk(c"\x014AMD-Vi: [Firmware Bug]: Invalid UID length.\n".as_ptr() as *const u8) };
}
}
1 => {
unsafe { sprintf(uid.0.as_mut_ptr() as *mut u8, c"%d".as_ptr() as *const u8, (*e).uid as i32) };
}
2 => {
unsafe { memcpy(uid.0.as_mut_ptr() as *mut (), core::ptr::addr_of_mut!((*e).uid) as *const (), (*e).uidl as u64) };
uid.0[((unsafe { (*e).uidl }) as i32) as usize] = 0;
}
_ => {
}
}
devid = PCI_SEG_DEVID_TO_SBDF!(seg_id, unsafe { (*e).devid });
loop {
if unsafe { amd_iommu_dump } {
let mut __v45: i32;
while false {
}
__v45 = unsafe { _printk(c"\x016AMD-Vi:   DEV_ACPI_HID(%s[%s])\t\tdevid: %04x:%02x:%02x.%x, flags: %#02x\n".as_ptr() as *const u8, hid.as_mut_ptr() as *mut u8, uid.0.as_mut_ptr() as *mut u8, seg_id as i32, PCI_BUS_NUM__3!(devid), PCI_SLOT__2!(devid), PCI_FUNC__3!(devid), (*e).flags as i32) };
}
if true {
break;
}
}
flags = unsafe { (*e).flags };
ret = add_acpi_hid_device(hid.as_mut_ptr() as *mut u8, uid.0.as_mut_ptr() as *mut u8, core::ptr::addr_of_mut!(devid), false);
if ret != 0 {
return ret;
}
// add_special_device might update the devid in case a
// command-line override is present. So call
// set_dev_entry_from_acpi after add_special_device.
set_dev_entry_from_acpi(iommu, devid as u16, (unsafe { (*e).flags }) as u32, 0u32);
}
}
}
_ => {
}
}
let __v48: i32 = ivhd_entry_length(p);
let __v49: *mut u8 = p;
p = __v49.wrapping_offset(__v48 as isize);
}
0
}

// Allocate PCI segment data structure
#[unsafe(link_section = ".init.text")]
fn alloc_pci_segment(id: u16, ivrs_base: *mut acpi_table_header) -> *mut amd_iommu_pci_seg {
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__651: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__650: u64 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut last_bdf: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(last_bdf) as *mut i32;
let mut pci_seg: core::mem::MaybeUninit<*mut amd_iommu_pci_seg> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut amd_iommu_pci_seg = core::ptr::addr_of_mut!(pci_seg) as *mut *mut amd_iommu_pci_seg;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
// First parse ACPI tables to find the largest Bus/Dev/Func we need to
// handle in this PCI segment. Upon this information the shared data
// structures for the PCI segments in the system will be allocated.
*__slate_slot_10 = find_last_devid_acpi(ivrs_base, id);
if *__slate_slot_10 < 0 {
return core::ptr::null_mut::<amd_iommu_pci_seg>();
} else {
__obj_size = size_mul(core::mem::size_of::<amd_iommu_pci_seg>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_14 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_14)
});
}
*__slate_slot_11 = *__slate_slot_5 as *mut amd_iommu_pci_seg;
if (*__slate_slot_11).is_null() {
return core::ptr::null_mut::<amd_iommu_pci_seg>();
} else {
(*(*__slate_slot_11)).last_bdf = *__slate_slot_10 as u16;
loop {
if amd_iommu_dump {
while false {
}
_printk(c"\x016AMD-Vi: PCI segment : 0x%0x, last bdf : 0x%04x\n".as_ptr() as *const u8, id as i32, *__slate_slot_10);
}
if true {
break;
}
}
__UNIQUE_ID_x__650 = roundup_pow_of_two!((*__slate_slot_10).wrapping_add(1).wrapping_mul(32));
__UNIQUE_ID_y__651 = 4096;
loop {
if __types_ok__7!(__UNIQUE_ID_x__650, __UNIQUE_ID_y__651) == 0 {
__compiletime_assert_652();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
(*(*__slate_slot_11)).dev_table_size = (if __UNIQUE_ID_x__650 > (__UNIQUE_ID_y__651 as u64) { __UNIQUE_ID_x__650 } else { __UNIQUE_ID_y__651 as u64 }) as u32;
(*(*__slate_slot_11)).id = id;
init_llist_head(core::ptr::addr_of_mut!((*(*__slate_slot_11)).dev_data_list));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_11)).unity_map));
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_11)).list), core::ptr::addr_of_mut!(amd_iommu_pci_seg_list));
if alloc_dev_table(*__slate_slot_11) == 0 {
if alloc_alias_table(*__slate_slot_11) == 0 {
if alloc_rlookup_table(*__slate_slot_11) != 0 {
free_alias_table(*__slate_slot_11);
} else {
return *__slate_slot_11;
}
}
free_dev_table(*__slate_slot_11);
}
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_11)).list));
kfree(*__slate_slot_11 as *const ());
return core::ptr::null_mut::<amd_iommu_pci_seg>();
}
}
}
}

#[unsafe(link_section = ".init.text")]
fn get_pci_segment(id: u16, ivrs_base: *mut acpi_table_header) -> *mut amd_iommu_pci_seg {
let mut pci_seg: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
pci_seg = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) });
while list_entry_is_head!(pci_seg, unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) }, list) == 0 {
if ((unsafe { (*pci_seg).id }) as i32) == (id as i32) {
return pci_seg;
}
pci_seg = list_next_entry__5!(pci_seg, list);
}
alloc_pci_segment(id, ivrs_base)
}

#[unsafe(link_section = ".init.text")]
fn free_pci_segments() {
let mut pci_seg: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
let mut next: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
pci_seg = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) });
next = list_next_entry__5!(pci_seg, list);
while list_entry_is_head!(pci_seg, unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) }, list) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*pci_seg).list) });
free_irq_lookup_table(pci_seg);
free_rlookup_table(pci_seg);
free_alias_table(pci_seg);
free_dev_table(pci_seg);
unsafe { kfree(pci_seg as *const ()) };
pci_seg = next;
next = list_next_entry__5!(next, list);
}
}

#[unsafe(link_section = ".init.text")]
fn free_sysfs(iommu: *mut amd_iommu) {
if !unsafe { (*iommu).iommu.dev }.is_null() {
unsafe { iommu_device_unregister(core::ptr::addr_of_mut!((*iommu).iommu)) };
unsafe { iommu_device_sysfs_remove(core::ptr::addr_of_mut!((*iommu).iommu)) };
}
}

#[unsafe(link_section = ".init.text")]
fn free_iommu_one(iommu: *mut amd_iommu) {
free_sysfs(iommu);
free_iommu_buffers(iommu);
unsafe { amd_iommu_free_ppr_log(iommu) };
free_ga_log(iommu);
iommu_unmap_mmio_space(iommu);
unsafe { amd_iommu_iopf_uninit(iommu) };
}

#[unsafe(link_section = ".init.text")]
fn free_iommu_all() {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
let mut next: *mut amd_iommu = unsafe { core::mem::zeroed() };
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
next = list_next_entry__3!(iommu, list);
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*iommu).list) });
free_iommu_one(iommu);
unsafe { kfree(iommu as *const ()) };
iommu = next;
next = list_next_entry__3!(next, list);
}
}

/// Family15h Model 10h-1fh erratum 746 (IOMMU Logging May Stall Translations)
/// Workaround:
///     BIOS should disable L2B micellaneous clock gating by setting
///     L2_L2B_CK_GATE_CONTROL[CKGateL2BMiscDisable](D0F2xF4_x90[2]) = 1b
fn amd_iommu_erratum_746_workaround(iommu: *mut amd_iommu) {
let mut value: u32 = 0;
if ((unsafe { boot_cpu_data.__slate_anon_0.__slate_anon_0.x86 }) as i32) != 21 || ((unsafe { boot_cpu_data.__slate_anon_0.__slate_anon_0.x86_model }) as i32) < 16 || ((unsafe { boot_cpu_data.__slate_anon_0.__slate_anon_0.x86_model }) as i32) > 31 {
return;
}
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 240i32, 144u32) };
unsafe { pci_read_config_dword((*iommu).dev as *const pci_dev, 244i32, core::ptr::addr_of_mut!(value)) };
if (value as u64) & BIT!(2i32) != 0 {
return;
}
// Select NB indirect register 0x90 and enable writing
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 240i32, 400u32) };
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 244i32, value | 4) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*(*iommu).dev).dev) as *const device, c"AMD-Vi: Applying erratum 746 workaround\n".as_ptr() as *const u8) };
// Clear the enable writing bit
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, 240i32, 144u32) };
}

/// Family15h Model 30h-3fh (IOMMU Mishandles ATS Write Permission)
/// Workaround:
///     BIOS should enable ATS write permission check by setting
///     L2_DEBUG_3[AtsIgnoreIWDis](D0F2xF4_x47[0]) = 1b
fn amd_iommu_ats_write_check_workaround(iommu: *mut amd_iommu) {
let mut value: u32 = 0;
if ((unsafe { boot_cpu_data.__slate_anon_0.__slate_anon_0.x86 }) as i32) != 21 || ((unsafe { boot_cpu_data.__slate_anon_0.__slate_anon_0.x86_model }) as i32) < 48 || ((unsafe { boot_cpu_data.__slate_anon_0.__slate_anon_0.x86_model }) as i32) > 63 {
return;
}
// Test L2_DEBUG_3[AtsIgnoreIWDis] == 1
value = iommu_read_l2(iommu, 71u8);
if (value as u64) & BIT!(0i32) != 0 {
return;
}
// Set L2_DEBUG_3[AtsIgnoreIWDis] = 1
iommu_write_l2(iommu, 71u8, ((value as u64) | BIT!(0i32)) as u32);
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*(*iommu).dev).dev) as *const device, c"AMD-Vi: Applying ATS write check workaround\n".as_ptr() as *const u8) };
}

/// This function glues the initialization function for one IOMMU
/// together and also allocates the command buffer and programs the
/// hardware. It does NOT enable the IOMMU. This is done afterwards.
#[unsafe(link_section = ".init.text")]
fn init_iommu_one(iommu: *mut amd_iommu, h: *mut ivhd_header, ivrs_base: *mut acpi_table_header) -> i32 {
let mut __slate_compound_0: raw_spinlock = unsafe { core::mem::zeroed() };
let pci_seg: *mut amd_iommu_pci_seg = get_pci_segment(unsafe { (*h).pci_seg }, ivrs_base);
if pci_seg.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*iommu).pci_seg = pci_seg;
}
'__slate_break_1: loop {
unsafe {
*core::ptr::addr_of_mut!((*iommu).lock) = *{
__slate_compound_0 = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t0.val = __SlateRecord_types_h_190 { counter: 0 };
__t0
} } };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
if true {
break;
}
}
unsafe {
(*iommu).cmd_sem_val = 0;
}
// Add IOMMU to internal data structures
list_add_tail(unsafe { core::ptr::addr_of_mut!((*iommu).list) }, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
let __v2: i32 = unsafe { amd_iommus_present };
unsafe {
amd_iommus_present = __v2.wrapping_add(1);
}
unsafe {
(*iommu).index = __v2;
}
if unlikely!((unsafe { (*iommu).index }) >= 32) != 0 {
let mut __v4: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"System has more IOMMUs than supported by this driver\n".as_ptr() as *const u8) };
let mut __v5: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v5 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v6: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_init_iommu_one_7, sym __slate_asm_string_init_iommu_one_8, const 1893 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v6 = bug;
unsafe { __v5.unwrap()(__v6) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
return 38i32.wrapping_neg();
}
// Copy data from ACPI table entry to the iommu struct
unsafe {
(*iommu).devid = (*h).devid;
}
unsafe {
(*iommu).cap_ptr = (*h).cap_ptr;
}
unsafe {
(*iommu).mmio_phys = (*h).mmio_phys;
}
'__slate_break_10: {
match (unsafe { (*h).r#type }) as i32 {
16 => {
if (unsafe { (*h).efr_attr }) != 0 && (unsafe { (*h).efr_attr }) & 122880u32 != 0 && (unsafe { (*h).efr_attr }) & 8257536u32 != 0 {
unsafe {
(*iommu).mmio_phys_end = 524288;
}
} else {
unsafe {
(*iommu).mmio_phys_end = 16384;
}
}
// Check if IVHD EFR contains proper max banks/counters
// GAM requires GA mode.
if (unsafe { (*h).efr_attr }) & 64u32 == 0 {
unsafe {
amd_iommu_guest_ir = 0;
}
}
}
17 | 64 => {
if (unsafe { (*h).efr_reg }) & 512u64 != 0 {
unsafe {
(*iommu).mmio_phys_end = 524288;
}
} else {
unsafe {
(*iommu).mmio_phys_end = 16384;
}
}
if ((unsafe { (*h).efr_attr }) as u64) & BIT!(0i32) != 0 {
let mut __v11: i64;
let __ret_do_once: bool = true;
let mut __v12: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely!(__ret_cond) != 0 && unlikely!(!unsafe { __already_done__1926 }) != 0 {
unsafe {
__already_done__1926 = true;
}
__ret_once = true;
}
__v12 = __ret_once as i64;
if __v12 != 0 {
let mut __v13: i32;
while false {
}
__v13 = unsafe { _printk(c"\x014AMD-Vi: Host Address Translation is not supported.\n".as_ptr() as *const u8) };
}
__v11 = __ret_do_once as i64;
unsafe {
amd_iommu_hatdis = true;
}
}
// XT and GAM require GA mode.
if (unsafe { (*h).efr_reg }) & 128u64 == 0 {
unsafe {
amd_iommu_guest_ir = 0;
}
break '__slate_break_10;
} else {
if (unsafe { (*h).efr_reg }) & BIT!(2i32) != 0 {
unsafe {
amd_iommu_xt_mode = 1;
}
}
}
early_iommu_features_init(iommu, h);
}
_ => {
return 22i32.wrapping_neg();
}
}
}
unsafe {
(*iommu).mmio_base = iommu_map_mmio_space((*iommu).mmio_phys, (*iommu).mmio_phys_end);
}
if unsafe { (*iommu).mmio_base }.is_null() {
return 12i32.wrapping_neg();
}
init_iommu_from_acpi(iommu, h)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_653: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(link_section = ".data..once")]
static mut __already_done__1926: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn init_iommu_one_late(iommu: *mut amd_iommu) -> i32 {
let mut ret: i32 = alloc_iommu_buffers(iommu);
if ret != 0 {
return ret;
}
unsafe {
(*iommu).int_enabled = false;
}
init_translation_status(iommu);
let __v0: bool = translation_pre_enabled(iommu) && !is_kdump_kernel();
if __v0 {
iommu_disable(iommu);
clear_translation_pre_enabled(iommu);
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014AMD-Vi: Translation was enabled for IOMMU:%d but we are not in kdump mode\n".as_ptr() as *const u8, (*iommu).index) };
}
if unsafe { amd_iommu_pre_enabled } {
unsafe {
amd_iommu_pre_enabled = translation_pre_enabled(iommu);
}
}
if unsafe { amd_iommu_irq_remap } {
ret = unsafe { amd_iommu_create_irq_domain(iommu) };
if ret != 0 {
return ret;
}
}
// Make sure IOMMU is not considered to translate itself. The IVRS
// table tells us so, but this is a lie!
unsafe {
*unsafe { (*(*iommu).pci_seg).rlookup_table }.wrapping_offset(((*iommu).devid as i32) as isize) = core::ptr::null_mut::<amd_iommu>();
}
0
}

/// get_highest_supported_ivhd_type - Look up the appropriate IVHD type
/// @ivrs: Pointer to the IVRS header
///
/// This function search through all IVDB of the maximum supported IVHD
fn get_highest_supported_ivhd_type(ivrs: *mut acpi_table_header) -> u8 {
let base: *mut u8 = ivrs as *mut u8;
let mut ivhd: *mut ivhd_header = base.wrapping_offset(48 as isize) as *mut ivhd_header;
let mut last_type: u8 = unsafe { (*ivhd).r#type };
let devid: u16 = unsafe { (*ivhd).devid };
while ((unsafe { (ivhd as *mut u8).offset_from(base as *mut u8) }) as i64) < ((unsafe { (*ivrs).length }) as i64) && ((unsafe { (*ivhd).r#type }) as i32) <= 64 {
let p: *mut u8 = ivhd as *mut u8;
if ((unsafe { (*ivhd).devid }) as i32) == (devid as i32) {
last_type = unsafe { (*ivhd).r#type };
}
ivhd = p.wrapping_offset(((unsafe { (*ivhd).length }) as i32) as isize) as *mut ivhd_header;
}
last_type
}

/// Iterates over all IOMMU entries in the ACPI table, allocates the
/// IOMMU structure and initializes it with init_iommu_one()
#[unsafe(link_section = ".init.text")]
fn init_iommu_all(table: *mut acpi_table_header) -> i32 {
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut p: *mut u8 = table as *mut u8;
let mut end: *mut u8 = table as *mut u8;
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let __v0: *mut u8 = end;
end = __v0.wrapping_offset((unsafe { (*table).length }) as isize);
let __v2: *mut u8 = p;
p = __v2.wrapping_offset(48 as isize);
// Phase 1: Process all IVHD blocks
'__slate_break_18: while p < end {
let h: *mut ivhd_header = p as *mut ivhd_header;
if ((unsafe { *p }) as i32) == unsafe { amd_iommu_target_ivhd_type } {
loop {
if unsafe { amd_iommu_dump } {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016AMD-Vi: device: %04x:%02x:%02x.%01x cap: %04x flags: %01x info %04x\n".as_ptr() as *const u8, (*h).pci_seg as i32, PCI_BUS_NUM!(unsafe { (*h).devid }), PCI_SLOT!(unsafe { (*h).devid }), PCI_FUNC!(unsafe { (*h).devid }), (*h).cap_ptr as i32, (*h).flags as i32, (*h).info as i32) };
}
if true {
break;
}
}
loop {
if unsafe { amd_iommu_dump } {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x016AMD-Vi:        mmio-addr: %016llx\n".as_ptr() as *const u8, (*h).mmio_phys) };
}
if true {
break;
}
}
let mut __v10: *mut amd_iommu;
let __obj_size: u64 = size_mul(core::mem::size_of::<amd_iommu>() as u64, 1u64);
let mut __v11: *mut ();
let mut __v12: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
} });
}
__v12 = _res;
__v11 = __v12;
__v10 = __v11 as *mut amd_iommu;
iommu = __v10;
if iommu.is_null() {
return 12i32.wrapping_neg();
}
ret = init_iommu_one(iommu, h, table);
if ret != 0 {
return ret;
}
}
let __v16: *mut u8 = p;
p = __v16.wrapping_offset(((unsafe { (*h).length }) as i32) as isize);
}
let mut __v19: i64;
let __ret_warn_on: i32 = (p != end) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("654: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 654b - ., 3; .popsection\n# {0}", const 654 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_init_iommu_all_20, sym __slate_asm_string_init_iommu_all_21, const 2053 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("655: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 655b - ., 4; .popsection\n# {0}", const 655 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v19 = (__ret_warn_on != 0) as i64;
// Phase 2 : Early feature support check
get_global_efr();
// Phase 3 : Enabling IOMMU features
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
ret = init_iommu_one_late(iommu);
if ret != 0 {
return ret;
}
iommu = list_next_entry__3!(iommu, list);
}
0
}

fn init_iommu_perf_ctr(iommu: *mut amd_iommu) {
let mut val: u64 = 0;
let pdev: *mut pci_dev = unsafe { (*iommu).dev };
if !check_feature(BIT_ULL!(9i32)) {
return;
}
unsafe {
amd_iommu_pc_present = true;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"AMD-Vi: IOMMU performance counters supported\n".as_ptr() as *const u8) };
val = readl(unsafe { (*iommu).mmio_base }.wrapping_offset(16384 as isize) as *const ()) as u64;
unsafe {
(*iommu).max_banks = (val >> 12i32 & 63) as u8;
}
unsafe {
(*iommu).max_counters = (val >> 7i32 & 15) as u8;
}
return;
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.amd_iommu_show_cap")]
extern "C" fn amd_iommu_show_cap(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let iommu: *mut amd_iommu = dev_to_amd_iommu(dev);
(unsafe { sysfs_emit(buf, c"%x\n".as_ptr() as *const u8, (*iommu).cap) }) as i64
}

static mut dev_attr_cap: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"cap\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_init_c_2094>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2094_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2094_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2094_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2094_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2094_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(amd_iommu_show_cap) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

#[unsafe(link_section = ".text.slate_distinct.amd_init.amd_iommu_show_features")]
extern "C" fn amd_iommu_show_features(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
(unsafe { sysfs_emit(buf, c"%llx:%llx\n".as_ptr() as *const u8, amd_iommu_efr, amd_iommu_efr2) }) as i64
}

static mut dev_attr_features: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"features\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_init_c_2102>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2102_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2102_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2102_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2102_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_init_c_2102_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(amd_iommu_show_features) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

static mut amd_iommu_attrs: __SlateAlign16<[*mut attribute; 3]> = __SlateAlign16([unsafe { core::ptr::addr_of_mut!(dev_attr_cap.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_features.attr) }, core::ptr::null_mut::<attribute>()]);

static mut amd_iommu_group: attribute_group = {
let mut __t0: attribute_group = unsafe { core::mem::zeroed() };
__t0.name = (b"amd-iommu\0".as_ptr() as *mut u8) as *const u8;
__t0.__slate_anon_4 = __SlateRecord_sysfs_h_118 { attrs: unsafe { core::ptr::addr_of_mut!(amd_iommu_attrs.0) as *mut *mut attribute } };
__t0
};

static mut amd_iommu_groups: __SlateAlign16<[*const attribute_group; 2]> = __SlateAlign16([(unsafe { core::ptr::addr_of_mut!(amd_iommu_group) }) as *const attribute_group, core::ptr::null::<attribute_group>()]);

/// Note: IVHD 0x11 and 0x40 also contains exact copy
/// of the IOMMU Extended Feature Register [MMIO Offset 0030h].
/// Default to EFR in IVHD since it is available sooner (i.e. before PCI init).
#[unsafe(link_section = ".init.text")]
fn late_iommu_features_init(iommu: *mut amd_iommu) {
let mut features: u64 = 0;
let mut features2: u64 = 0;
if (unsafe { (*iommu).cap }) & 134217728u32 == 0 {
return;
}
// read extended feature bits
features = readq(unsafe { (*iommu).mmio_base }.wrapping_offset(48 as isize) as *const ());
features2 = readq(unsafe { (*iommu).mmio_base }.wrapping_offset(416 as isize) as *const ());
if (unsafe { amd_iommu_efr }) == 0 {
unsafe {
amd_iommu_efr = features;
}
unsafe {
amd_iommu_efr2 = features2;
}
return;
}
// Sanity check and warn if EFR values from
// IVHD and MMIO conflict.
if features != unsafe { amd_iommu_efr } || features2 != unsafe { amd_iommu_efr2 } {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014AMD-Vi: [Firmware Warn]: EFR mismatch. Use IVHD EFR (%#llx : %#llx), EFR2 (%#llx : %#llx).\n".as_ptr() as *const u8, features, amd_iommu_efr, features2, amd_iommu_efr2) };
}
}

#[unsafe(link_section = ".init.text")]
fn iommu_init_pci(iommu: *mut amd_iommu) -> i32 {
let cap_ptr: i32 = (unsafe { (*iommu).cap_ptr }) as i32;
let mut ret: i32 = 0;
unsafe {
(*iommu).dev = pci_get_domain_bus_and_slot((*(*iommu).pci_seg).id as i32, PCI_BUS_NUM!(unsafe { (*iommu).devid }) as u32, (((*iommu).devid as i32) & 255) as u32);
}
if unsafe { (*iommu).dev }.is_null() {
return 19i32.wrapping_neg();
}
// ACPI _PRT won't have an IRQ for IOMMU
unsafe {
(*(*iommu).dev).__slate_bits_3.__set_irq_managed(1u32);
}
unsafe { pci_read_config_dword((*iommu).dev as *const pci_dev, cap_ptr.wrapping_add(0), core::ptr::addr_of_mut!((*iommu).cap)) };
if (unsafe { (*iommu).cap }) & 16777216u32 == 0 {
unsafe {
amd_iommu_iotlb_sup = false;
}
}
late_iommu_features_init(iommu);
if check_feature(BIT_ULL!(4i32)) {
let mut glxval: i32 = 0;
let mut pasmax: u64 = 0;
let mut __v0: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_init_c_2181_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36))) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_656() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_657() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_init_c_2181_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36)) == 0 {
unsafe { __compiletime_assert_658() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_init_c_2181_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36)) >> ((core::mem::size_of::<__SlateRecord_init_c_2181_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_659() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_init_c_2181_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_2181_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_init_c_2181_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_2181_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_660() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((unsafe { amd_iommu_efr }) & (core::mem::size_of::<__SlateRecord_init_c_2181_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36))) >> ((core::mem::size_of::<__SlateRecord_init_c_2181_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 32i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(36)).trailing_zeros() as i32);
__v0 = __v2;
pasmax = __v0;
unsafe {
(*iommu).iommu.max_pasids = (1i32 << pasmax.wrapping_add(1)).wrapping_sub(1) as u32;
}
loop {
if unlikely__3!((unsafe { (*iommu).iommu.max_pasids }) & 4294901760u32) != 0 {
loop {
unsafe {
core::arch::asm!("661: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 661b - ., 3; .popsection\n# {0}", const 661 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_iommu_init_pci_7, sym __slate_asm_string_iommu_init_pci_8, const 2184 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if true {
break;
}
}
let mut __v12: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_init_c_2186_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_662() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v14: u64;
loop {
if false {
unsafe { __compiletime_assert_663() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_init_c_2186_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) == 0 {
unsafe { __compiletime_assert_664() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_init_c_2186_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) >> ((core::mem::size_of::<__SlateRecord_init_c_2186_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_665() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_init_c_2186_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_2186_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_init_c_2186_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_2186_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_666() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v14 = ((unsafe { amd_iommu_efr }) & (core::mem::size_of::<__SlateRecord_init_c_2186_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))) >> ((core::mem::size_of::<__SlateRecord_init_c_2186_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 14i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32);
__v12 = __v14;
glxval = __v12 as i32;
if (unsafe { amd_iommu_max_glx_val }) == 1i32.wrapping_neg() {
unsafe {
amd_iommu_max_glx_val = glxval;
}
} else {
let mut __v19: i32;
let __UNIQUE_ID_x__667: i32 = unsafe { amd_iommu_max_glx_val };
let __UNIQUE_ID_y__668: i32 = glxval;
loop {
if __types_ok__2!(__UNIQUE_ID_x__667, __UNIQUE_ID_y__668) == 0 {
unsafe { __compiletime_assert_669() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v19 = if __UNIQUE_ID_x__667 < __UNIQUE_ID_y__668 { __UNIQUE_ID_x__667 } else { __UNIQUE_ID_y__668 };
unsafe {
amd_iommu_max_glx_val = __v19;
}
}
iommu_enable_gt(iommu);
}
let __v21: bool = check_feature(BIT_ULL!(1i32)) && (unsafe { amd_iommu_alloc_ppr_log(iommu) }) != 0;
if __v21 {
return 12i32.wrapping_neg();
}
if ((unsafe { (*iommu).cap }) as u64) & 67108864u64 != 0 {
let mut __v22: i32;
while false {
}
__v22 = unsafe { _printk(c"\x016AMD-Vi: Using strict mode due to virtualization\n".as_ptr() as *const u8) };
unsafe { iommu_set_dma_strict() };
unsafe {
amd_iommu_np_cache = true;
}
}
init_iommu_perf_ctr(iommu);
if is_rd890_iommu(unsafe { (*iommu).dev }) {
let mut i: i32 = 0;
unsafe {
(*iommu).root_pdev = pci_get_domain_bus_and_slot((*(*iommu).pci_seg).id as i32, (*(*(*iommu).dev).bus).number as u32, PCI_DEVFN!(0i32, 0i32) as u32);
}
// Some rd890 systems may not be fully reconfigured by the
// BIOS, so it's necessary for us to store this information so
// it can be reprogrammed on resume
unsafe { pci_read_config_dword((*iommu).dev as *const pci_dev, ((*iommu).cap_ptr as i32).wrapping_add(4), core::ptr::addr_of_mut!((*iommu).stored_addr_lo)) };
unsafe { pci_read_config_dword((*iommu).dev as *const pci_dev, ((*iommu).cap_ptr as i32).wrapping_add(8), core::ptr::addr_of_mut!((*iommu).stored_addr_hi)) };
// Low bit locks writes to configuration space
let __v26: u32 = (unsafe { (*iommu).stored_addr_lo }) & 4294967294u32;
unsafe {
(*iommu).stored_addr_lo = __v26;
}
while i < 6 {
for j in 0i32..18i32 {
unsafe {
*((*((*iommu).stored_l1.as_mut_ptr() as *mut [u32; 18]).wrapping_offset(i as isize)).as_mut_ptr() as *mut u32).wrapping_offset(j as isize) = iommu_read_l1(iommu, i as u16, j as u8);
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < 131 {
unsafe {
*((*iommu).stored_l2.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = iommu_read_l2(iommu, i as u8);
}
i = i.wrapping_add(1);
}
}
amd_iommu_erratum_746_workaround(iommu);
amd_iommu_ats_write_check_workaround(iommu);
ret = unsafe { iommu_device_sysfs_add(core::ptr::addr_of_mut!((*iommu).iommu), core::ptr::addr_of_mut!((*(*iommu).dev).dev), core::ptr::addr_of_mut!(amd_iommu_groups.0) as *mut *const attribute_group, c"ivhd%d".as_ptr() as *const u8, (*iommu).index) };
if ret != 0 {
return ret;
}
// Allocate per IOMMU IOPF queue here so that in attach device path,
// PRI capable device can be added to IOPF queue
if amd_iommu_gt_ppr_supported() {
ret = unsafe { amd_iommu_iopf_init(iommu) };
if ret != 0 {
return ret;
}
}
ret = unsafe { iommu_device_register(core::ptr::addr_of_mut!((*iommu).iommu), core::ptr::addr_of!(amd_iommu_ops), core::ptr::null_mut::<device>()) };
if ret != 0 || ((unsafe { amd_iommu_pgtable }) as u32) == 0 {
// Remove sysfs if DMA translation is not supported by the
// IOMMU. Do not return an error to enable IRQ remapping
// in state_next(), DTE[V, TV] must eventually be set to 0.
unsafe { iommu_device_sysfs_remove(core::ptr::addr_of_mut!((*iommu).iommu)) };
}
unsafe { pci_enable_device((*iommu).dev) }
}

fn print_iommu_info() {
let mut i: i32 = 0;
if (unsafe { amd_iommu_efr }) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016AMD-Vi: Extended features (%#llx, %#llx):".as_ptr() as *const u8, amd_iommu_efr, amd_iommu_efr2) };
i = 0;
while (i as u64) < ARRAY_SIZE__9!() {
if check_feature(1u64 << i) {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x01c %s".as_ptr() as *const u8, feat_str.0[i as usize]) };
}
i = i.wrapping_add(1);
}
if check_feature(BIT_ULL!(21i32)) {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x01c GA_vAPIC".as_ptr() as *const u8) };
}
if check_feature(BIT_ULL!(63i32)) {
let mut __v9: i32;
while false {
}
__v9 = unsafe { _printk(c"\x01c SNP".as_ptr() as *const u8) };
}
if check_feature2(BIT_ULL!(1i32)) {
let mut __v11: i32;
while false {
}
__v11 = unsafe { _printk(c"\x01c SEV-TIO".as_ptr() as *const u8) };
}
let mut __v13: i32;
while false {
}
__v13 = unsafe { _printk(c"\x01c\n".as_ptr() as *const u8) };
}
if (unsafe { irq_remapping_enabled }) != 0 {
let mut __v15: i32;
while false {
}
__v15 = unsafe { _printk(c"\x016AMD-Vi: Interrupt remapping enabled\n".as_ptr() as *const u8) };
if (unsafe { amd_iommu_xt_mode }) == 1 {
let mut __v17: i32;
while false {
}
__v17 = unsafe { _printk(c"\x016AMD-Vi: X2APIC enabled\n".as_ptr() as *const u8) };
}
}
if ((unsafe { amd_iommu_pgtable }) as u32) == 2 {
let mut __v19: i32;
while false {
}
__v19 = unsafe { _printk(c"\x016AMD-Vi: V2 page table enabled (Paging mode : %d level)\n".as_ptr() as *const u8, amd_iommu_gpt_level) };
}
}

static feat_str: __SlateAlign16<[*const u8; 10]> = __SlateAlign16([(b"PreF\0".as_ptr() as *mut u8) as *const u8, (b"PPR\0".as_ptr() as *mut u8) as *const u8, (b"X2APIC\0".as_ptr() as *mut u8) as *const u8, (b"NX\0".as_ptr() as *mut u8) as *const u8, (b"GT\0".as_ptr() as *mut u8) as *const u8, (b"[5]\0".as_ptr() as *mut u8) as *const u8, (b"IA\0".as_ptr() as *mut u8) as *const u8, (b"GA\0".as_ptr() as *mut u8) as *const u8, (b"HE\0".as_ptr() as *mut u8) as *const u8, (b"PC\0".as_ptr() as *mut u8) as *const u8]);

#[unsafe(link_section = ".init.text")]
fn amd_iommu_init_pci() -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut pci_seg: core::mem::MaybeUninit<*mut amd_iommu_pci_seg> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut amd_iommu_pci_seg = core::ptr::addr_of_mut!(pci_seg) as *mut *mut amd_iommu_pci_seg;
let mut iommu: core::mem::MaybeUninit<*mut amd_iommu> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut amd_iommu = core::ptr::addr_of_mut!(iommu) as *mut *mut amd_iommu;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Init global identity domain before registering IOMMU
amd_iommu_init_identity_domain();
*__slate_slot_3 = list_first_entry__2!(core::ptr::addr_of_mut!(amd_iommu_list));
'__join_0: {
'__join_11: {
while list_entry_is_head!(*__slate_slot_3, core::ptr::addr_of_mut!(amd_iommu_list), list) == 0 {
*__slate_slot_1 = iommu_init_pci(*__slate_slot_3);
if *__slate_slot_1 != 0 {
break '__join_11;
} else {
// Need to setup range after PCI init
iommu_set_cwwb_range(*__slate_slot_3);
*__slate_slot_3 = list_next_entry__3!(*__slate_slot_3, list);
}
}
// Order is important here to make sure any unity map requirements are
// fulfilled. The unity mappings are created and written to the device
// table during the iommu_init_pci() call.
//
// After that we call init_device_table_dma() to make sure any
// uninitialized DTE will block DMA, and in the end we flush the caches
// of all IOMMUs to make sure the changes to the device table are
// active.
*__slate_slot_2 = list_first_entry__5!(core::ptr::addr_of_mut!(amd_iommu_pci_seg_list));
while list_entry_is_head!(*__slate_slot_2, core::ptr::addr_of_mut!(amd_iommu_pci_seg_list), list) == 0 {
init_device_table_dma(*__slate_slot_2);
*__slate_slot_2 = list_next_entry__5!(*__slate_slot_2, list);
}
*__slate_slot_3 = list_first_entry__2!(core::ptr::addr_of_mut!(amd_iommu_list));
while list_entry_is_head!(*__slate_slot_3, core::ptr::addr_of_mut!(amd_iommu_list), list) == 0 {
amd_iommu_flush_all_caches(*__slate_slot_3);
*__slate_slot_3 = list_next_entry__3!(*__slate_slot_3, list);
}
print_iommu_info();
break '__join_0;
}
while false {
}
_printk(c"\x013AMD-Vi: IOMMU%d: Failed to initialize IOMMU Hardware (error=%d)!\n".as_ptr() as *const u8, (*(*__slate_slot_3)).index, *__slate_slot_1);
}
return *__slate_slot_1;
}
}

// The following functions initialize the MSI interrupts for all IOMMUs
// in the system. It's a bit challenging because there could be multiple
// IOMMUs per PCI BDF but we can call pci_enable_msi(x) only once per
// pci_dev.
fn iommu_setup_msi(iommu: *mut amd_iommu) -> i32 {
let mut r: i32 = unsafe { pci_enable_msi((*iommu).dev) };
if r != 0 {
return r;
}
r = unsafe { request_threaded_irq((*(*iommu).dev).irq, None, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(i32, *mut ()) -> u32>>(amd_iommu_int_thread as *const ()) }, 8192u64, c"AMD-Vi".as_ptr() as *const u8, iommu as *mut ()) };
if r != 0 {
unsafe { pci_disable_msi((*iommu).dev) };
return r;
}
0
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
union intcapxt {
    capxt: u64,
    __slate_anon_1: __SlateRecord_init_c_2377,
}


static mut intcapxt_controller: irq_chip = {
let mut __t0: irq_chip = unsafe { core::mem::zeroed() };
__t0.name = (b"IOMMU-MSI\0".as_ptr() as *mut u8) as *const u8;
__t0.irq_ack = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_data)>>(irq_chip_ack_parent as *const ()) };
__t0.irq_mask = Some(intcapxt_mask_irq);
__t0.irq_unmask = Some(intcapxt_unmask_irq);
__t0.irq_set_affinity = Some(intcapxt_set_affinity);
__t0.irq_retrigger = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_data) -> i32>>(irq_chip_retrigger_hierarchy as *const ()) };
__t0.irq_set_wake = Some(intcapxt_set_wake);
__t0.flags = 4100u64;
__t0
};

#[unsafe(link_section = ".text.slate_distinct.amd_init.intcapxt_irqdomain_activate")]
extern "C" fn intcapxt_irqdomain_activate(domain: *mut irq_domain, irqd: *mut irq_data, reserve: bool) -> i32 {
0
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.intcapxt_irqdomain_deactivate")]
extern "C" fn intcapxt_irqdomain_deactivate(domain: *mut irq_domain, irqd: *mut irq_data) {
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.intcapxt_irqdomain_alloc")]
extern "C" fn intcapxt_irqdomain_alloc(domain: *mut irq_domain, virq: u32, nr_irqs: u32, arg: *mut ()) -> i32 {
let info: *mut irq_alloc_info = arg as *mut irq_alloc_info;
let mut i: i32 = 0;
let mut ret: i32 = 0;
if info.is_null() || ((unsafe { (*info).r#type }) as u32) != 6 {
return 22i32.wrapping_neg();
}
ret = unsafe { irq_domain_alloc_irqs_parent(domain, virq, nr_irqs, arg) };
if ret < 0 {
return ret;
}
i = virq as i32;
while (i as u32) < virq.wrapping_add(nr_irqs) {
let irqd: *mut irq_data = unsafe { irq_domain_get_irq_data(domain, i as u32) };
unsafe {
(*irqd).chip = core::ptr::addr_of_mut!(intcapxt_controller);
}
unsafe {
(*irqd).hwirq = (*info).hwirq;
}
unsafe {
(*irqd).chip_data = (*info).data;
}
unsafe { __irq_set_handler(i as u32, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_desc)>>(handle_edge_irq as *const ()) }, 0i32, c"edge".as_ptr() as *const u8) };
i = i.wrapping_add(1);
}
ret
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.intcapxt_irqdomain_free")]
extern "C" fn intcapxt_irqdomain_free(domain: *mut irq_domain, virq: u32, nr_irqs: u32) {
unsafe { irq_domain_free_irqs_top(domain, virq, nr_irqs) };
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.intcapxt_unmask_irq")]
extern "C" fn intcapxt_unmask_irq(irqd: *mut irq_data) {
let iommu: *mut amd_iommu = (unsafe { (*irqd).chip_data }) as *mut amd_iommu;
let cfg: *mut irq_cfg = unsafe { irqd_cfg(irqd) };
let mut xt: intcapxt = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(xt) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
unsafe {
xt.capxt = 0;
}
let __t0 = ((unsafe { (*apic).__slate_bits_0.__get_dest_mode_logical() }) as i32) as u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(xt.__slate_anon_1.__slate_bits_0));
__t1.__set_dest_mode_logical(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(xt.__slate_anon_1.__slate_bits_0), __t1);
}
let __t2 = (unsafe { (*cfg).vector }) as u64;
unsafe {
let mut __t3 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(xt.__slate_anon_1.__slate_bits_0));
__t3.__set_vector(__t2);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(xt.__slate_anon_1.__slate_bits_0), __t3);
}
let __t4 = ((unsafe { (*cfg).dest_apicid }) as u64) & GENMASK__36!(23i32, 0i32);
unsafe {
let mut __t5 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(xt.__slate_anon_1.__slate_bits_0));
__t5.__set_destid_0_23(__t4);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(xt.__slate_anon_1.__slate_bits_0), __t5);
}
let __t6 = ((unsafe { (*cfg).dest_apicid }) >> 24i32) as u64;
unsafe {
let mut __t7 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(xt.__slate_anon_1.__slate_bits_0));
__t7.__set_destid_24_31(__t6);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(xt.__slate_anon_1.__slate_bits_0), __t7);
}
writeq(unsafe { xt.capxt }, unsafe { (*iommu).mmio_base }.wrapping_offset((unsafe { (*irqd).hwirq }) as isize) as *mut ());
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.intcapxt_mask_irq")]
extern "C" fn intcapxt_mask_irq(irqd: *mut irq_data) {
let iommu: *mut amd_iommu = (unsafe { (*irqd).chip_data }) as *mut amd_iommu;
writeq(0u64, unsafe { (*iommu).mmio_base }.wrapping_offset((unsafe { (*irqd).hwirq }) as isize) as *mut ());
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.intcapxt_set_affinity")]
extern "C" fn intcapxt_set_affinity(irqd: *mut irq_data, mask: *const cpumask, force: bool) -> i32 {
let parent: *mut irq_data = unsafe { (*irqd).parent_data };
let ret: i32 = unsafe { unsafe { (*(*parent).chip).irq_set_affinity }.unwrap()(parent, mask, force) };
if ret < 0 || ret == 2 {
return ret;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.intcapxt_set_wake")]
extern "C" fn intcapxt_set_wake(irqd: *mut irq_data, on: u32) -> i32 {
if on != 0 { 95i32.wrapping_neg() } else { 0 }
}

static intcapxt_domain_ops: __SlateConst<irq_domain_ops> = __SlateConst({
let mut __t0: irq_domain_ops = unsafe { core::mem::zeroed() };
__t0.alloc = Some(intcapxt_irqdomain_alloc);
__t0.free = Some(intcapxt_irqdomain_free);
__t0.activate = Some(intcapxt_irqdomain_activate);
__t0.deactivate = Some(intcapxt_irqdomain_deactivate);
__t0
});

static mut iommu_irqdomain: *mut irq_domain = unsafe { core::mem::zeroed() };

fn iommu_get_irqdomain() -> *mut irq_domain {
// No need for locking here (yet) as the init is single-threaded
if !unsafe { iommu_irqdomain }.is_null() {
return unsafe { iommu_irqdomain };
}
let r#fn: *mut fwnode_handle = irq_domain_alloc_named_fwnode(c"AMD-Vi-MSI".as_ptr() as *const u8);
if r#fn.is_null() {
return core::ptr::null_mut::<irq_domain>();
}
unsafe {
iommu_irqdomain = irq_domain_create_hierarchy(x86_vector_domain, 0u32, 0u32, r#fn, core::ptr::addr_of!(intcapxt_domain_ops.0), core::ptr::null_mut::<()>());
}
if unsafe { iommu_irqdomain }.is_null() {
unsafe { irq_domain_free_fwnode(r#fn) };
}
unsafe { iommu_irqdomain }
}

fn __iommu_setup_intcapxt(iommu: *mut amd_iommu, devname: *const u8, hwirq: i32, thread_fn: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>) -> i32 {
let mut info: irq_alloc_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
let mut irq: i32 = 0;
let mut ret: i32 = 0;
let node: i32 = dev_to_node(unsafe { core::ptr::addr_of_mut!((*(*iommu).dev).dev) });
let domain: *mut irq_domain = iommu_get_irqdomain();
if domain.is_null() {
return 6i32.wrapping_neg();
}
unsafe { init_irq_alloc_info(core::ptr::addr_of_mut!(info), core::ptr::null::<cpumask>()) };
info.r#type = 6;
info.data = iommu as *mut ();
info.hwirq = hwirq as u64;
irq = irq_domain_alloc_irqs(domain, 1u32, node, core::ptr::addr_of_mut!(info) as *mut ());
if irq < 0 {
unsafe { irq_domain_remove(domain) };
return irq;
}
ret = unsafe { request_threaded_irq(irq as u32, None, thread_fn, 8192u64, devname, iommu as *mut ()) };
if ret != 0 {
unsafe { irq_domain_free_irqs(irq as u32, 1u32) };
unsafe { irq_domain_remove(domain) };
return ret;
}
0
}

fn iommu_setup_intcapxt(iommu: *mut amd_iommu) -> i32 {
let mut ret: i32 = 0;
unsafe { snprintf(((*iommu).evt_irq_name.as_mut_ptr() as *mut u8) as *mut u8, core::mem::size_of::<[u8; 16]>() as u64, c"AMD-Vi%d-Evt".as_ptr() as *const u8, (*iommu).index) };
ret = __iommu_setup_intcapxt(iommu, (unsafe { (*iommu).evt_irq_name.as_mut_ptr() as *mut u8 }) as *const u8, 368i32, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(i32, *mut ()) -> u32>>(amd_iommu_int_thread_evtlog as *const ()) });
if ret != 0 {
return ret;
}
unsafe { snprintf(((*iommu).ppr_irq_name.as_mut_ptr() as *mut u8) as *mut u8, core::mem::size_of::<[u8; 16]>() as u64, c"AMD-Vi%d-PPR".as_ptr() as *const u8, (*iommu).index) };
ret = __iommu_setup_intcapxt(iommu, (unsafe { (*iommu).ppr_irq_name.as_mut_ptr() as *mut u8 }) as *const u8, 376i32, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(i32, *mut ()) -> u32>>(amd_iommu_int_thread_pprlog as *const ()) });
if ret != 0 {
return ret;
}
unsafe { snprintf(((*iommu).ga_irq_name.as_mut_ptr() as *mut u8) as *mut u8, core::mem::size_of::<[u8; 16]>() as u64, c"AMD-Vi%d-GA".as_ptr() as *const u8, (*iommu).index) };
ret = __iommu_setup_intcapxt(iommu, (unsafe { (*iommu).ga_irq_name.as_mut_ptr() as *mut u8 }) as *const u8, 384i32, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(i32, *mut ()) -> u32>>(amd_iommu_int_thread_galog as *const ()) });
ret
}

fn iommu_init_irq(iommu: *mut amd_iommu) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !(*iommu).int_enabled {
if amd_iommu_xt_mode == 1 {
*__slate_slot_0 = iommu_setup_intcapxt(iommu);
} else {
if (*(*iommu).dev).msi_cap != 0 {
*__slate_slot_0 = iommu_setup_msi(iommu);
} else {
*__slate_slot_0 = 19i32.wrapping_neg();
}
}
if *__slate_slot_0 != 0 {
return *__slate_slot_0;
} else {
(*iommu).int_enabled = true;
}
}
if amd_iommu_xt_mode == 1 {
iommu_feature_enable(iommu, 51u8);
}
iommu_feature_enable(iommu, 3u8);
return 0;
}
}

// The next functions belong to the third pass of parsing the ACPI
// table. In this last pass the memory mapping requirements are
// gathered (like exclusion and unity mapping ranges).
#[unsafe(link_section = ".init.text")]
fn free_unity_maps() {
let mut entry: *mut unity_map_entry = unsafe { core::mem::zeroed() };
let mut next: *mut unity_map_entry = unsafe { core::mem::zeroed() };
let mut p: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
let mut pci_seg: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
pci_seg = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) });
p = list_next_entry__5!(pci_seg, list);
while list_entry_is_head!(pci_seg, unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) }, list) == 0 {
entry = list_first_entry__33!(unsafe { core::ptr::addr_of_mut!((*pci_seg).unity_map) });
next = list_next_entry__29!(entry, list);
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!((*pci_seg).unity_map) }, list) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*entry).list) });
unsafe { kfree(entry as *const ()) };
entry = next;
next = list_next_entry__29!(next, list);
}
pci_seg = p;
p = list_next_entry__5!(p, list);
}
}

/// called for unity map ACPI definition
#[unsafe(link_section = ".init.text")]
fn init_unity_map_range(m: *mut ivmd_header, ivrs_base: *mut acpi_table_header) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut e: *mut unity_map_entry = core::ptr::null_mut::<unity_map_entry>();
let mut pci_seg: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
let mut s: *mut u8 = unsafe { core::mem::zeroed() };
pci_seg = get_pci_segment(unsafe { (*m).pci_seg }, ivrs_base);
if pci_seg.is_null() {
return 12i32.wrapping_neg();
}
let mut __v0: *mut unity_map_entry;
let __obj_size: u64 = size_mul(core::mem::size_of::<unity_map_entry>() as u64, 1u64);
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
__v0 = __v1 as *mut unity_map_entry;
e = __v0;
if e.is_null() {
return 12i32.wrapping_neg();
}
match (unsafe { (*m).r#type }) as i32 {
33 => {
s = c"IVMD_TYPEi\t\t\t".as_ptr() as *mut u8;
let __v6: u16 = unsafe { (*m).devid };
unsafe {
(*e).devid_end = __v6;
}
unsafe {
(*e).devid_start = __v6;
}
}
32 => {
s = c"IVMD_TYPE_ALL\t\t".as_ptr() as *mut u8;
unsafe {
(*e).devid_start = 0;
}
unsafe {
(*e).devid_end = (*pci_seg).last_bdf;
}
}
34 => {
s = c"IVMD_TYPE_RANGE\t\t".as_ptr() as *mut u8;
unsafe {
(*e).devid_start = (*m).devid;
}
unsafe {
(*e).devid_end = (*m).aux;
}
}
_ => {
unsafe { kfree(e as *const ()) };
return 0;
}
}
unsafe {
(*e).address_start = PAGE_ALIGN!(unsafe { (*m).range_start });
}
unsafe {
(*e).address_end = unsafe { (*e).address_start }.wrapping_add(PAGE_ALIGN!(unsafe { (*m).range_length }));
}
unsafe {
(*e).prot = ((*m).flags as i32) >> 1i32;
}
// Treat per-device exclusion ranges as r/w unity-mapped regions
// since some buggy BIOSes might lead to the overwritten exclusion
// range (exclusion_start and exclusion_length members). This
// happens when there are multiple exclusion ranges (IVMD entries)
// defined in ACPI table.
if ((unsafe { (*m).flags }) as i32) & 8 != 0 {
unsafe {
(*e).prot = 3i32;
}
}
loop {
if unsafe { amd_iommu_dump } {
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x016AMD-Vi: %s devid_start: %04x:%02x:%02x.%x devid_end: %04x:%02x:%02x.%x range_start: %016llx range_end: %016llx flags: %x\n".as_ptr() as *const u8, s, (*m).pci_seg as i32, PCI_BUS_NUM!(unsafe { (*e).devid_start }), PCI_SLOT!(unsafe { (*e).devid_start }), PCI_FUNC!(unsafe { (*e).devid_start }), (*m).pci_seg as i32, PCI_BUS_NUM!(unsafe { (*e).devid_end }), PCI_SLOT!(unsafe { (*e).devid_end }), PCI_FUNC!(unsafe { (*e).devid_end }), (*e).address_start, (*e).address_end, (*m).flags as i32) };
}
if true {
break;
}
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*e).list) }, unsafe { core::ptr::addr_of_mut!((*pci_seg).unity_map) });
0
}

/// iterates over all memory definitions we find in the ACPI table
#[unsafe(link_section = ".init.text")]
fn init_memory_definitions(table: *mut acpi_table_header) -> i32 {
let mut p: *mut u8 = table as *mut u8;
let mut end: *mut u8 = table as *mut u8;
let __v0: *mut u8 = end;
end = __v0.wrapping_offset((unsafe { (*table).length }) as isize);
let __v2: *mut u8 = p;
p = __v2.wrapping_offset(48 as isize);
while p < end {
let m: *mut ivmd_header = p as *mut ivmd_header;
if ((unsafe { (*m).flags }) as i32) & (1 | 8) != 0 {
init_unity_map_range(m, table);
}
let __v4: *mut u8 = p;
p = __v4.wrapping_offset(((unsafe { (*m).length }) as i32) as isize);
}
0
}

/// Init the device table to not allow DMA access for devices
fn init_device_table_dma(pci_seg: *mut amd_iommu_pci_seg) {
let mut devid: u32 = 0;
let dev_table: *mut dev_table_entry = unsafe { (*pci_seg).dev_table };
if dev_table.is_null() || ((unsafe { amd_iommu_pgtable }) as u32) == 0 {
return;
}
while devid <= (((unsafe { (*pci_seg).last_bdf }) as i32) as u32) {
set_dte_bit(dev_table.wrapping_offset(devid as isize), 0u8);
if !unsafe { amd_iommu_snp_en } {
set_dte_bit(dev_table.wrapping_offset(devid as isize), 1u8);
}
devid = devid.wrapping_add(1);
}
}

#[unsafe(link_section = ".init.text")]
fn uninit_device_table_dma(pci_seg: *mut amd_iommu_pci_seg) {
let mut devid: u32 = 0;
let dev_table: *mut dev_table_entry = unsafe { (*pci_seg).dev_table };
if dev_table.is_null() {
return;
}
while devid <= (((unsafe { (*pci_seg).last_bdf }) as i32) as u32) {
unsafe {
*((*dev_table.wrapping_offset(devid as isize)).__slate_anon_0.data.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = 0;
}
unsafe {
*((*dev_table.wrapping_offset(devid as isize)).__slate_anon_0.data.as_mut_ptr() as *mut u64).wrapping_offset(1 as isize) = 0;
}
devid = devid.wrapping_add(1);
}
}

fn init_device_table() {
let mut pci_seg: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
let mut devid: u32 = 0;
if !unsafe { amd_iommu_irq_remap } {
return;
}
pci_seg = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) });
while list_entry_is_head!(pci_seg, unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) }, list) == 0 {
devid = 0;
while devid <= (((unsafe { (*pci_seg).last_bdf }) as i32) as u32) {
set_dte_bit(unsafe { (*pci_seg).dev_table }.wrapping_offset(devid as isize), 128u8);
devid = devid.wrapping_add(1);
}
pci_seg = list_next_entry__5!(pci_seg, list);
}
}

fn iommu_init_flags(iommu: *mut amd_iommu) {
if ((unsafe { (*iommu).acpi_flags }) as i32) & 1 != 0 {
iommu_feature_enable(iommu, 1u8);
} else {
iommu_feature_disable(iommu, 1u8);
}
if ((unsafe { (*iommu).acpi_flags }) as i32) & 2 != 0 {
iommu_feature_enable(iommu, 8u8);
} else {
iommu_feature_disable(iommu, 8u8);
}
if ((unsafe { (*iommu).acpi_flags }) as i32) & 4 != 0 {
iommu_feature_enable(iommu, 9u8);
} else {
iommu_feature_disable(iommu, 9u8);
}
if ((unsafe { (*iommu).acpi_flags }) as i32) & 8 != 0 {
iommu_feature_enable(iommu, 11u8);
} else {
iommu_feature_disable(iommu, 11u8);
}
// make IOMMU memory accesses cache coherent
iommu_feature_enable(iommu, 10u8);
// Set IOTLB invalidation timeout to 1s
iommu_feature_set(iommu, 4u64, 7u64, 5u8);
// Enable Enhanced Peripheral Page Request Handling
if check_feature(BIT_ULL!(50i32)) {
iommu_feature_enable(iommu, 45u8);
}
}

fn iommu_apply_resume_quirks(iommu: *mut amd_iommu) {
// Restore the l1 indirect regs for each of the 6 l1s
let mut i: i32 = 0;
let mut ioc_feature_control: u32 = 0;
let pdev: *mut pci_dev = unsafe { (*iommu).root_pdev };
// RD890 BIOSes may not have completely reconfigured the iommu
if !is_rd890_iommu(unsafe { (*iommu).dev }) || pdev.is_null() {
return;
}
// First, we need to ensure that the iommu is enabled. This is
// controlled by a register in the northbridge
// Select Northbridge indirect register 0x75 and enable writing
unsafe { pci_write_config_dword(pdev as *const pci_dev, 96i32, 245u32) };
unsafe { pci_read_config_dword(pdev as *const pci_dev, 100i32, core::ptr::addr_of_mut!(ioc_feature_control)) };
// Enable the iommu
if ioc_feature_control & 1 == 0 {
unsafe { pci_write_config_dword(pdev as *const pci_dev, 100i32, ioc_feature_control | 1) };
}
// Restore the iommu BAR
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, ((*iommu).cap_ptr as i32).wrapping_add(4), (*iommu).stored_addr_lo) };
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, ((*iommu).cap_ptr as i32).wrapping_add(8), (*iommu).stored_addr_hi) };
while i < 6 {
for j in 0i32..18i32 {
iommu_write_l1(iommu, i as u16, j as u8, unsafe { *((*((*iommu).stored_l1.as_mut_ptr() as *mut [u32; 18]).wrapping_offset(i as isize)).as_mut_ptr() as *mut u32).wrapping_offset(j as isize) });
}
i = i.wrapping_add(1);
}
// Restore the l2 indirect regs
i = 0;
while i < 131 {
iommu_write_l2(iommu, i as u8, unsafe { *((*iommu).stored_l2.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) });
i = i.wrapping_add(1);
}
// Lock PCI setup registers
unsafe { pci_write_config_dword((*iommu).dev as *const pci_dev, ((*iommu).cap_ptr as i32).wrapping_add(4), (*iommu).stored_addr_lo | 1) };
}

fn iommu_enable_ga(iommu: *mut amd_iommu) {
match unsafe { amd_iommu_guest_ir } {
1 | 2 => {
iommu_feature_enable(iommu, 17u8);
unsafe {
(*iommu).irte_ops = core::ptr::addr_of_mut!(irte_128_ops);
}
}
_ => {
unsafe {
(*iommu).irte_ops = core::ptr::addr_of_mut!(irte_32_ops);
}
}
}
}

fn iommu_disable_irtcachedis(iommu: *mut amd_iommu) {
iommu_feature_disable(iommu, 59u8);
}

fn iommu_enable_irtcachedis(iommu: *mut amd_iommu) {
let mut ctrl: u64 = 0;
if !unsafe { amd_iommu_irtcachedis } {
return;
}
// Note:
// The support for IRTCacheDis feature is dertermined by
// checking if the bit is writable.
iommu_feature_enable(iommu, 59u8);
ctrl = readq(unsafe { (*iommu).mmio_base }.wrapping_offset(24 as isize) as *const ());
ctrl &= 576460752303423488u64;
if ctrl != 0 {
unsafe {
(*iommu).irtcachedis_enabled = true;
}
}
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016AMD-Vi: iommu%d (%#06x) : IRT cache is %s\n".as_ptr() as *const u8, (*iommu).index, (*iommu).devid as i32, if (*iommu).irtcachedis_enabled { c"disabled".as_ptr() as *mut u8 } else { c"enabled".as_ptr() as *mut u8 }) };
}

fn iommu_enable_2k_int(iommu: *mut amd_iommu) {
let mut __v0: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_init_c_2879_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9))) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_670() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_671() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_init_c_2879_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9)) == 0 {
unsafe { __compiletime_assert_672() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_init_c_2879_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9)) >> ((core::mem::size_of::<__SlateRecord_init_c_2879_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_673() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_init_c_2879_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_2879_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_init_c_2879_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_2879_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_674() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((unsafe { amd_iommu_efr2 }) & (core::mem::size_of::<__SlateRecord_init_c_2879_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9))) >> ((core::mem::size_of::<__SlateRecord_init_c_2879_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(9)).trailing_zeros() as i32);
__v0 = __v2;
if __v0 != 1 {
return;
}
iommu_feature_set(iommu, 1u64, 3u64, 43u8);
}

fn early_enable_iommu(iommu: *mut amd_iommu) {
iommu_disable(iommu);
iommu_init_flags(iommu);
iommu_set_device_table(iommu);
iommu_enable_command_buffer(iommu);
iommu_enable_gt(iommu);
iommu_enable_ga(iommu);
iommu_enable_xt(iommu);
iommu_enable_irtcachedis(iommu);
iommu_enable_2k_int(iommu);
iommu_enable(iommu);
unsafe { amd_iommu_flush_all_caches(iommu) };
}

/// This function finally enables all IOMMUs found in the system after
/// they have been initialized.
///
/// Or if in kdump kernel and IOMMUs are all pre-enabled, try to reuse
/// the old content of device table entries. Not this case or reuse failed,
/// just continue as normal kernel does.
fn early_enable_iommus() {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
let mut pci_seg: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
if !reuse_device_table() {
// If come here because of failure in reusing device table from old
// kernel with all IOMMUs enabled, print error message and try to
// free allocated old_dev_tbl_cpy.
if unsafe { amd_iommu_pre_enabled } {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013AMD-Vi: Failed to reuse DEV table from previous kernel.\n".as_ptr() as *const u8) };
// Bail out early if unable to remap/reuse DEV table from
// previous kernel if SNP enabled as IOMMU commands will
// time out without DEV table and cause kdump boot panic.
loop {
if unlikely!(check_feature(BIT_ULL!(63i32))) != 0 {
loop {
unsafe {
core::arch::asm!("675: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 675b - ., 3; .popsection\n# {0}", const 675 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_early_enable_iommus_2, sym __slate_asm_string_early_enable_iommus_3, const 2929 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if true {
break;
}
}
}
pci_seg = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) });
while list_entry_is_head!(pci_seg, unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) }, list) == 0 {
if !unsafe { (*pci_seg).old_dev_tbl_cpy }.is_null() {
unsafe { memunmap((*pci_seg).old_dev_tbl_cpy as *mut ()) };
unsafe {
(*pci_seg).old_dev_tbl_cpy = core::ptr::null_mut::<dev_table_entry>();
}
}
pci_seg = list_next_entry__5!(pci_seg, list);
}
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
clear_translation_pre_enabled(iommu);
early_enable_iommu(iommu);
iommu = list_next_entry__3!(iommu, list);
}
} else {
let mut __v13: i32;
while false {
}
__v13 = unsafe { _printk(c"\x016AMD-Vi: Reused DEV table from previous kernel.\n".as_ptr() as *const u8) };
pci_seg = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) });
while list_entry_is_head!(pci_seg, unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) }, list) == 0 {
unsafe { iommu_free_pages((*pci_seg).dev_table as *mut ()) };
unsafe {
(*pci_seg).dev_table = (*pci_seg).old_dev_tbl_cpy;
}
pci_seg = list_next_entry__5!(pci_seg, list);
}
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
iommu_disable_command_buffer(iommu);
iommu_disable_event_buffer(iommu);
iommu_disable_irtcachedis(iommu);
iommu_enable_command_buffer(iommu);
iommu_enable_ga(iommu);
iommu_enable_xt(iommu);
iommu_enable_irtcachedis(iommu);
iommu_enable_2k_int(iommu);
iommu_set_device_table(iommu);
unsafe { amd_iommu_flush_all_caches(iommu) };
iommu = list_next_entry__3!(iommu, list);
}
}
}

fn enable_iommus_ppr() {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
if !amd_iommu_gt_ppr_supported() {
return;
}
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
unsafe { amd_iommu_enable_ppr_log(iommu) };
iommu = list_next_entry__3!(iommu, list);
}
}

fn enable_iommus_vapic() {
let mut status: u32 = 0;
let mut i: u32 = 0;
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
// Disable GALog if already running. It could have been enabled
// in the previous boot before kdump.
status = readl(unsafe { (*iommu).mmio_base }.wrapping_offset(8224 as isize) as *const ());
if (status as u64) & BIT!(8i32) != 0 {
iommu_feature_disable(iommu, 28u8);
iommu_feature_disable(iommu, 29u8);
// Need to set and poll check the GALOGRun bit to zero before
// we can set/ modify GA Log registers safely.
i = 0;
while i < 2000000 {
status = readl(unsafe { (*iommu).mmio_base }.wrapping_offset(8224 as isize) as *const ());
if (status as u64) & BIT!(8i32) == 0 {
break;
}
udelay(10u64);
i = i.wrapping_add(1);
}
let mut __v5: i64;
let __ret_warn_on: i32 = (i >= 2000000) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("676: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 676b - ., 3; .popsection\n# {0}", const 676 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_enable_iommus_vapic_6, sym __slate_asm_string_enable_iommus_vapic_7, const 3006 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("677: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 677b - ., 4; .popsection\n# {0}", const 677 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
return;
}
}
iommu = list_next_entry__3!(iommu, list);
}
if !AMD_IOMMU_GUEST_IR_VAPIC!(unsafe { amd_iommu_guest_ir }) {
return;
}
if !check_feature(BIT_ULL!(21i32)) {
unsafe {
amd_iommu_guest_ir = 1;
}
return;
}
let __v11: bool = if unsafe { amd_iommu_snp_en } {
let mut __v12: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_init_c_3019_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_678() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v14: u64;
loop {
if false {
unsafe { __compiletime_assert_679() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_init_c_3019_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) == 0 {
unsafe { __compiletime_assert_680() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_init_c_3019_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) >> ((core::mem::size_of::<__SlateRecord_init_c_3019_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_681() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_init_c_3019_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_3019_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_init_c_3019_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_3019_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_682() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v14 = ((unsafe { amd_iommu_efr2 }) & (core::mem::size_of::<__SlateRecord_init_c_3019_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))) >> ((core::mem::size_of::<__SlateRecord_init_c_3019_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32);
__v12 = __v14;
__v12 != 1
} else {
false
};
if __v11 {
let mut __v19: i32;
while false {
}
__v19 = unsafe { _printk(c"\x014AMD-Vi: Force to disable Virtual APIC due to SNP\n".as_ptr() as *const u8) };
unsafe {
amd_iommu_guest_ir = 1;
}
return;
}
// Enabling GAM and SNPAVIC support
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
let __v22: bool = iommu_init_ga_log(iommu) != 0 || iommu_ga_log_enable(iommu) != 0;
if __v22 {
return;
}
iommu_feature_enable(iommu, 25u8);
if unsafe { amd_iommu_snp_en } {
iommu_feature_enable(iommu, 61u8);
}
iommu = list_next_entry__3!(iommu, list);
}
unsafe {
amd_iommu_irq_ops.capability = amd_iommu_irq_ops.capability | 1i32;
}
let mut __v27: i32;
while false {
}
__v27 = unsafe { _printk(c"\x016AMD-Vi: Virtual APIC enabled\n".as_ptr() as *const u8) };
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.disable_iommus")]
extern "C" fn disable_iommus() {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
iommu_disable(iommu);
iommu = list_next_entry__3!(iommu, list);
}
if AMD_IOMMU_GUEST_IR_VAPIC!(unsafe { amd_iommu_guest_ir }) {
let __v4: i32 = (unsafe { amd_iommu_irq_ops.capability }) & -2i32;
unsafe {
amd_iommu_irq_ops.capability = __v4;
}
}
}

// Suspend/Resume support
// disable suspend until real resume implemented
#[unsafe(link_section = ".text.slate_distinct.amd_init.amd_iommu_resume")]
extern "C" fn amd_iommu_resume(data: *mut ()) {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
iommu_apply_resume_quirks(iommu);
iommu = list_next_entry__3!(iommu, list);
}
// re-load the hardware
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
early_enable_iommu(iommu);
iommu = list_next_entry__3!(iommu, list);
}
iommu_enable_event_buffer();
amd_iommu_enable_interrupts();
}

#[unsafe(link_section = ".text.slate_distinct.amd_init.amd_iommu_suspend")]
extern "C" fn amd_iommu_suspend(data: *mut ()) -> i32 {
// disable IOMMUs to go out of the way for BIOS
disable_iommus();
0
}

static amd_iommu_syscore_ops: __SlateConst<syscore_ops> = __SlateConst({
let mut __t0: syscore_ops = unsafe { core::mem::zeroed() };
__t0.suspend = Some(amd_iommu_suspend);
__t0.resume = Some(amd_iommu_resume);
__t0
});

static mut amd_iommu_syscore: syscore = {
let mut __t0: syscore = unsafe { core::mem::zeroed() };
__t0.ops = unsafe { core::ptr::addr_of!(amd_iommu_syscore_ops.0) };
__t0
};

#[unsafe(link_section = ".init.text")]
fn free_iommu_resources() {
free_iommu_all();
free_pci_segments();
}

#[unsafe(link_section = ".init.text")]
fn check_sb_ioapic(devid: i32) -> bool {
let bus: u8 = PCI_BUS_NUM__2!(devid) as u8;
let devfn: u8 = (devid & 255) as u8;
let val: u16 = unsafe { read_pci_config_16(bus, PCI_SLOT!(devfn) as u8, PCI_FUNC!(devfn) as u8, 10u8) };
// The SB IOAPIC is integrated into the FCH (Southbridge), which is
// exposed as an SMBus or ISA bridge in PCI config space.
(val as i32) == 3077 || (val as i32) == 1537
}

/// The Southbridge IOAPIC is assigned a GSI Base of 0 (handling interrupts
/// 0 through 23).
#[unsafe(link_section = ".init.text")]
fn get_sb_ioapic_id() -> i32 {
let idx: i32 = unsafe { mp_find_ioapic(0u32) };
if idx < 0 {
return 19i32.wrapping_neg();
}
unsafe { mpc_ioapic_id(idx) }
}

#[unsafe(link_section = ".init.text")]
fn check_ioapic_information() -> bool {
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut devid: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(devid) as *mut i32;
let mut sb_apicid: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(sb_apicid) as *mut i32;
let mut idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(idx) as *mut i32;
let mut has_sb_ioapic: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut bool = core::ptr::addr_of_mut!(has_sb_ioapic) as *mut bool;
let mut ret: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut bool = core::ptr::addr_of_mut!(ret) as *mut bool;
let mut fw_bug: *const u8 = unsafe { core::mem::zeroed() };
unsafe {
{
fw_bug = c"[Firmware Bug]: ".as_ptr() as *const u8;
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_10 = false;
*__slate_slot_11 = true;
// If we have map overrides on the kernel command line the
// messages in this function might not describe firmware bugs
// anymore - so be careful
if cmdline_maps {
fw_bug = c"".as_ptr() as *const u8;
}
}
*__slate_slot_8 = get_sb_ioapic_id();
if *__slate_slot_8 < 0 {
// Lack of SB IOAPIC registration is not a firmware bug,
// e.g. kernel booted with noapic or noacpi.
fw_bug = c"".as_ptr() as *const u8;
} else {
*__slate_slot_9 = 0;
while *__slate_slot_9 < nr_ioapics {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let id: i32 = mpc_ioapic_id(*__slate_slot_9);
*__slate_slot_7 = get_ioapic_devid(id);
if *__slate_slot_7 < 0 {
while false {
}
_printk(c"\x013AMD-Vi: %s: IOAPIC[%d] not in IVRS table\n".as_ptr() as *const u8, fw_bug, id);
*__slate_slot_11 = false;
} else {
if id == *__slate_slot_8 {
__v5 = check_sb_ioapic(*__slate_slot_7);
} else {
__v5 = false;
}
if __v5 {
*__slate_slot_10 = true;
}
}
let __v3: i32 = *__slate_slot_9;
*__slate_slot_9 = __v3.wrapping_add(1);
}
}
if !(*__slate_slot_10) {
while false {
}
// We expect the SB IOAPIC to be listed in the IVRS
// table. The system timer is connected to the SB IOAPIC
// and if we don't have it in the list the system will
// panic at boot time.  This situation usually happens
// when the BIOS is buggy and provides us the wrong
// device id for the IOAPIC in the system.
_printk(c"\x013AMD-Vi: %s: No southbridge IOAPIC found\n".as_ptr() as *const u8, fw_bug);
*__slate_slot_11 = false;
}
if !(*__slate_slot_11) {
while false {
}
_printk(c"\x013AMD-Vi: Disabling interrupt remapping\n".as_ptr() as *const u8);
}
return *__slate_slot_11;
}
}

#[unsafe(link_section = ".init.text")]
fn free_dma_resources() {
unsafe { amd_iommu_pdom_id_destroy() };
free_unity_maps();
}

#[unsafe(link_section = ".init.text")]
fn ivinfo_init(ivrs: *mut ()) {
unsafe {
amd_iommu_ivinfo = *(((ivrs as *const u8).wrapping_offset(36 as isize) as *mut ()) as *mut u32);
}
}

// This is the hardware init function for AMD IOMMU in the system.
// This function is called either from amd_iommu_init or from the interrupt
// remapping setup code.
//
// This function basically parses the ACPI table for AMD IOMMU (IVRS)
// four times:
//
// 	1 pass) Discover the most comprehensive IVHD type to use.
//
// 	2 pass) Find the highest PCI device id the driver has to handle.
// 		Upon this information the size of the data structures is
// 		determined that needs to be allocated.
//
// 	3 pass) Initialize the data structures just allocated with the
// 		information in the ACPI table about available AMD IOMMUs
// 		in the system. It also maps the PCI devices in the
// 		system to specific IOMMUs
//
// 	4 pass) After the basic data structures are allocated and
// 		initialized we update them with information about memory
// 		remapping requirements parsed out of the ACPI table in
// 		this last pass.
//
// After everything is set up the IOMMUs are enabled and the necessary
// hotplug and suspend notifiers are registered.
#[unsafe(link_section = ".init.text")]
fn early_amd_iommu_init() -> i32 {
let mut pci_seg: core::mem::MaybeUninit<*mut amd_iommu_pci_seg> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut amd_iommu_pci_seg = core::ptr::addr_of_mut!(pci_seg) as *mut *mut amd_iommu_pci_seg;
let mut __ret_once: bool = unsafe { core::mem::zeroed() };
// 5 level guest page table
let mut __v12: bool = unsafe { core::mem::zeroed() };
let mut max_vasize: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u8 = core::ptr::addr_of_mut!(max_vasize) as *mut u8;
let mut efr_hats: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u8 = core::ptr::addr_of_mut!(efr_hats) as *mut u8;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut ivrs_base: core::mem::MaybeUninit<*mut acpi_table_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut acpi_table_header = core::ptr::addr_of_mut!(ivrs_base) as *mut *mut acpi_table_header;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_5 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 1usize);
if !amd_iommu_detected {
return 19i32.wrapping_neg();
} else {
*__slate_slot_21 = acpi_get_table(c"IVRS".as_ptr() as *mut u8, 0u32, core::ptr::addr_of_mut!(*__slate_slot_23));
if *__slate_slot_21 == EXCEP_ENV!(5i32) {
return 19i32.wrapping_neg();
} else {
if ACPI_FAILURE!(*__slate_slot_21) != 0 {
let err: *const u8 = acpi_format_exception(*__slate_slot_21);
while false {
}
_printk(c"\x013AMD-Vi: IVRS table error: %s\n".as_ptr() as *const u8, err);
return 22i32.wrapping_neg();
} else {
'__join_0: {
if boot_cpu_has!(141i32) == 0 {
while false {
}
_printk(c"\x013AMD-Vi: Failed to initialize. The CMPXCHG16B feature is required.\n".as_ptr() as *const u8);
*__slate_slot_22 = 22i32.wrapping_neg();
} else {
// Validate checksum here so we don't need to do it when
// we actually parse the table
*__slate_slot_22 = check_ivrs_checksum(*__slate_slot_23);
if *__slate_slot_22 == 0 {
ivinfo_init(*__slate_slot_23 as *mut ());
while false {
}
while false {
}
while false {
}
loop {
if !((core::mem::size_of::<__SlateRecord_init_c_3259_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 15i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(21)) >> ((core::mem::size_of::<__SlateRecord_init_c_3259_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 15i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(21)).trailing_zeros() as i32)) & 0u64 != 0 {
__compiletime_assert_686();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_init_c_3259_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 15i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(21)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_3259_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 15i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(21)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_init_c_3259_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 15i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(21)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_3259_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 15i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(21)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_687();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_19 = (((amd_iommu_ivinfo as u64) & (core::mem::size_of::<__SlateRecord_init_c_3259_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 15i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(21))) >> ((core::mem::size_of::<__SlateRecord_init_c_3259_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 15i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(21)).trailing_zeros() as i32)) as u8;
if *__slate_slot_19 == 0 {
*__slate_slot_19 = 64;
}
amd_iommu_target_ivhd_type = get_highest_supported_ivhd_type(*__slate_slot_23) as i32;
loop {
if amd_iommu_dump {
while false {
}
_printk(c"\x016AMD-Vi: Using IVHD type %#x\n".as_ptr() as *const u8, amd_iommu_target_ivhd_type);
}
if true {
break;
}
}
// now the data structures are allocated and basically initialized
// start the real acpi table scan
*__slate_slot_22 = init_iommu_all(*__slate_slot_23);
if *__slate_slot_22 == 0 {
if cpu_feature_enabled!(528i32) != 0 {
while false {
}
while false {
}
while false {
}
loop {
if !((core::mem::size_of::<__SlateRecord_init_c_3276_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13)) >> ((core::mem::size_of::<__SlateRecord_init_c_3276_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13)).trailing_zeros() as i32)) & 0u64 != 0 {
__compiletime_assert_691();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_init_c_3276_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_3276_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_init_c_3276_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_3276_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_692();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
__v12 = (amd_iommu_efr & (core::mem::size_of::<__SlateRecord_init_c_3276_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13))) >> ((core::mem::size_of::<__SlateRecord_init_c_3276_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13)).trailing_zeros() as i32) == 1;
} else {
__v12 = false;
}
if __v12 {
amd_iommu_gpt_level = 5;
}
while false {
}
while false {
}
while false {
}
loop {
if !((core::mem::size_of::<__SlateRecord_init_c_3279_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 10i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11)) >> ((core::mem::size_of::<__SlateRecord_init_c_3279_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 10i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11)).trailing_zeros() as i32)) & 0u64 != 0 {
__compiletime_assert_696();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_init_c_3279_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 10i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_3279_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 10i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_init_c_3279_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 10i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_init_c_3279_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 10i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_697();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_20 = ((amd_iommu_efr & (core::mem::size_of::<__SlateRecord_init_c_3279_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 10i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11))) >> ((core::mem::size_of::<__SlateRecord_init_c_3279_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 10i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11)).trailing_zeros() as i32)) as u8;
if (*__slate_slot_20 as i32) != 3 {
// efr[HATS] bits specify the maximum host translation level
// supported, with LEVEL 4 being initial max level.
amd_iommu_hpt_vasize = min_t__4!(u32, *__slate_slot_19, (*__slate_slot_20 as i32).wrapping_add(4).wrapping_sub(1).wrapping_mul(9).wrapping_add(21)) as u8;
} else {
let __ret_do_once: bool = true;
let __ret_cond: bool = __ret_do_once;
__ret_once = false;
if unlikely!(__ret_cond) != 0 && unlikely!(!__already_done__3288) != 0 {
__already_done__3288 = true;
__ret_once = true;
}
let __v5: i64 = __ret_once as i64;
if __v5 != 0 {
while false {
}
_printk(c"\x014AMD-Vi: [Firmware Bug]: Disable host address translation due to invalid translation level (%#x).\n".as_ptr() as *const u8, *__slate_slot_20 as i32);
}
amd_iommu_hatdis = true;
}
if (amd_iommu_pgtable as u32) == 2 {
if !amd_iommu_v2_pgtbl_supported() {
while false {
}
_printk(c"\x014AMD-Vi: Cannot enable v2 page table for DMA-API. Fallback to v1.\n".as_ptr() as *const u8);
amd_iommu_pgtable = 1;
}
}
if amd_iommu_hatdis {
// Host (v1) page table is not available. Attempt to use
// Guest (v2) page table.
if amd_iommu_v2_pgtbl_supported() {
amd_iommu_pgtable = 2;
} else {
amd_iommu_pgtable = 0;
}
}
// Disable any previously enabled IOMMUs
if !is_kdump_kernel() || amd_iommu_disabled {
disable_iommus();
}
if amd_iommu_irq_remap {
amd_iommu_irq_remap = check_ioapic_information();
}
if amd_iommu_irq_remap {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_22 = 12i32.wrapping_neg();
*__slate_slot_0 = list_first_entry__5!(core::ptr::addr_of_mut!(amd_iommu_pci_seg_list));
while list_entry_is_head!(*__slate_slot_0, core::ptr::addr_of_mut!(amd_iommu_pci_seg_list), list) == 0 {
if alloc_irq_lookup_table(*__slate_slot_0) != 0 {
break '__join_0;
} else {
*__slate_slot_0 = list_next_entry__5!(*__slate_slot_0, list);
}
}
}
*__slate_slot_22 = init_memory_definitions(*__slate_slot_23);
if *__slate_slot_22 == 0 {
// init the device table
init_device_table();
}
}
}
}
}
acpi_put_table(*__slate_slot_23); // Don't leak any ACPI memory
return *__slate_slot_22;
}
}
}
}
}

#[unsafe(link_section = ".data..once")]
static mut __already_done__3288: bool = unsafe { core::mem::zeroed() };

fn amd_iommu_enable_interrupts() -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut iommu: core::mem::MaybeUninit<*mut amd_iommu> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut amd_iommu = core::ptr::addr_of_mut!(iommu) as *mut *mut amd_iommu;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
*__slate_slot_1 = list_first_entry__2!(core::ptr::addr_of_mut!(amd_iommu_list));
'__join_0: {
while list_entry_is_head!(*__slate_slot_1, core::ptr::addr_of_mut!(amd_iommu_list), list) == 0 {
ret = iommu_init_irq(*__slate_slot_1);
if ret != 0 {
break '__join_0;
} else {
*__slate_slot_1 = list_next_entry__3!(*__slate_slot_1, list);
}
}
// Interrupt handler is ready to process interrupts. Enable
// PPR and GA log interrupt for all IOMMUs.
enable_iommus_vapic();
enable_iommus_ppr();
}
return ret;
}
}

#[unsafe(link_section = ".init.text")]
fn detect_ivrs() -> bool {
let mut pci_id: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(pci_id) as *mut u32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut ivrs_base: core::mem::MaybeUninit<*mut acpi_table_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut acpi_table_header = core::ptr::addr_of_mut!(ivrs_base) as *mut *mut acpi_table_header;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
*__slate_slot_7 = acpi_get_table(c"IVRS".as_ptr() as *mut u8, 0u32, core::ptr::addr_of_mut!(*__slate_slot_8));
if *__slate_slot_7 == EXCEP_ENV!(5i32) {
return false;
} else {
if ACPI_FAILURE!(*__slate_slot_7) != 0 {
let err: *const u8 = acpi_format_exception(*__slate_slot_7);
while false {
}
_printk(c"\x013AMD-Vi: IVRS table error: %s\n".as_ptr() as *const u8, err);
return false;
} else {
'__join_0: {
acpi_put_table(*__slate_slot_8);
if !amd_iommu_force_enable {
// Don't use IOMMU if there is Stoney Ridge graphics
*__slate_slot_6 = 0;
loop {
if *__slate_slot_6 < 32 {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_3 = read_pci_config(0u8, *__slate_slot_6 as u8, 0u8, 0u8);
if *__slate_slot_3 & 65535 == 4098 && *__slate_slot_3 >> 16i32 == 39140 {
break;
} else {
let __v1: i32 = *__slate_slot_6;
*__slate_slot_6 = __v1.wrapping_add(1);
}
} else {
break '__join_0;
}
}
while false {
}
_printk(c"\x016AMD-Vi: Disable IOMMU on Stoney Ridge\n".as_ptr() as *const u8);
return false;
}
}
pci_request_acs(); // Make sure ACS will be enabled during PCI probe
return true;
}
}
}
}

#[unsafe(link_section = ".init.text")]
fn iommu_snp_enable() {
}

fn amd_iommu_apply_erratum_snp() {
}

// AMD IOMMU Initialization State Machine
#[unsafe(link_section = ".init.text")]
fn state_next() -> i32 {
let mut ret: i32 = 0;
'__slate_break_0: {
match (unsafe { init_state }) as u32 {
0 => {
if !detect_ivrs() {
unsafe {
init_state = 7;
}
ret = 19i32.wrapping_neg();
} else {
unsafe {
init_state = 1;
}
}
}
1 => {
if unsafe { amd_iommu_disabled } {
unsafe {
init_state = 9;
}
ret = 22i32.wrapping_neg();
} else {
ret = early_amd_iommu_init();
unsafe {
init_state = (if ret != 0 { 8i32 } else { 2i32 }) as u32;
}
}
}
2 => {
early_enable_iommus();
unsafe {
x86_platform.iommu_shutdown = Some(disable_iommus);
}
unsafe {
init_state = 3;
}
}
3 => {
unsafe { register_syscore(core::ptr::addr_of_mut!(amd_iommu_syscore)) };
iommu_snp_enable();
amd_iommu_apply_erratum_snp();
// Allocate/enable event log buffer
if is_kdump_kernel() {
ret = remap_event_buffer();
} else {
ret = alloc_event_buffer();
}
if ret != 0 {
unsafe {
init_state = 8;
}
} else {
iommu_enable_event_buffer();
ret = amd_iommu_init_pci();
unsafe {
init_state = (if ret != 0 { 8i32 } else { 4i32 }) as u32;
}
}
}
4 => {
ret = amd_iommu_enable_interrupts();
unsafe {
init_state = (if ret != 0 { 8i32 } else { 5i32 }) as u32;
}
}
5 => {
unsafe {
init_state = 6;
}
}
6 => {
break '__slate_break_0; // Nothing to do
}
7..=9 => {
ret = 22i32.wrapping_neg(); // Error states => do nothing
}
_ => {
loop {
unsafe {
core::arch::asm!("700: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 700b - ., 3; .popsection\n# {0}", const 700 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_state_next_1, sym __slate_asm_string_state_next_2, const 3551 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
// Unknown state
}
}
}
if ret != 0 {
free_dma_resources();
if (unsafe { irq_remapping_enabled }) == 0 {
disable_iommus();
free_iommu_resources();
} else {
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
let mut pci_seg: *mut amd_iommu_pci_seg = unsafe { core::mem::zeroed() };
pci_seg = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) });
while list_entry_is_head!(pci_seg, unsafe { core::ptr::addr_of_mut!(amd_iommu_pci_seg_list) }, list) == 0 {
uninit_device_table_dma(pci_seg);
pci_seg = list_next_entry__5!(pci_seg, list);
}
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
unsafe { amd_iommu_flush_all_caches(iommu) };
iommu = list_next_entry__3!(iommu, list);
}
}
}
ret
}

#[unsafe(link_section = ".init.text")]
fn iommu_go_to_state(state: u32) -> i32 {
let mut ret: i32 = 22i32.wrapping_neg();
while ((unsafe { init_state }) as u32) != (state as u32) {
if ((unsafe { init_state }) as u32) == 7 || ((unsafe { init_state }) as u32) == 8 || ((unsafe { init_state }) as u32) == 9 {
break;
}
ret = state_next();
}
// SNP platform initilazation requires IOMMUs to be fully configured.
// If the SNP support on IOMMUs has NOT been checked, simply mark SNP
// as unsupported. If the SNP support on IOMMUs has been checked and
// host SNP support enabled but RMP enforcement has not been enabled
// in IOMMUs, then the system is in a half-baked state, but can limp
// along as all memory should be Hypervisor-Owned in the RMP. WARN,
// but leave SNP as "supported" to avoid confusing the kernel.
let __v1: bool = ret != 0 && cc_platform_has(7u32);
let __v2: bool = if __v1 {
let mut __v3: i64;
let __ret_warn_on: i32 = !(!unsafe { amd_iommu_snp_en }) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("701: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 701b - ., 3; .popsection\n# {0}", const 701 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_go_to_state_4, sym __slate_asm_string_iommu_go_to_state_5, const 3595 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("702: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 702b - ., 4; .popsection\n# {0}", const 702 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
__v3 == 0
} else {
false
};
if __v2 {
cc_platform_clear(7u32);
}
ret
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn amd_iommu_prepare() -> i32 {
let mut ret: i32 = 0;
unsafe {
amd_iommu_irq_remap = true;
}
ret = iommu_go_to_state(2u32);
if ret != 0 {
unsafe {
amd_iommu_irq_remap = false;
}
return ret;
}
if unsafe { amd_iommu_irq_remap } { 0 } else { 19i32.wrapping_neg() }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn amd_iommu_enable() -> i32 {
let ret: i32 = iommu_go_to_state(3u32);
if ret != 0 {
return ret;
}
unsafe {
irq_remapping_enabled = 1;
}
unsafe { amd_iommu_xt_mode }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.amd_init.amd_iommu_disable")]
extern "C" fn amd_iommu_disable() {
amd_iommu_suspend(core::ptr::null_mut::<()>());
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.amd_init.amd_iommu_reenable")]
extern "C" fn amd_iommu_reenable(mode: i32) -> i32 {
amd_iommu_resume(core::ptr::null_mut::<()>());
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.amd_init.amd_iommu_enable_faulting")]
extern "C" fn amd_iommu_enable_faulting(cpu: u32) -> i32 {
// We enable MSI later when PCI is initialized
0
}

/// This is the core init function for AMD IOMMU hardware in the system.
/// This function is called from the generic x86 DMA layer initialization
/// code.
#[unsafe(link_section = ".init.text")]
extern "C" fn amd_iommu_init() -> i32 {
let ret: i32 = iommu_go_to_state(6u32);
if ret == 0 {
amd_iommu_debugfs_setup();
}
ret
}

fn amd_iommu_sme_check() -> bool {
if !cc_platform_has(1u32) || ((unsafe { boot_cpu_data.__slate_anon_0.__slate_anon_0.x86 }) as i32) != 23 {
return true;
}
// For Fam17h, a specific level of support is required
if (unsafe { boot_cpu_data.microcode }) >= 134222341 {
return true;
}
if (unsafe { boot_cpu_data.microcode }) >= 134222118 && (unsafe { boot_cpu_data.microcode }) <= 134222335 {
return true;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x015AMD-Vi: IOMMU not currently supported when SME is active\n".as_ptr() as *const u8) };
false
}

/// Early detect code. This code runs at IOMMU detection time in the DMA
/// layer. It just looks if there is an IVRS ACPI table to detect AMD
/// IOMMUs
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn amd_iommu_detect() {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !(no_iommu != 0 || iommu_detected != 0 && true) {
if amd_iommu_sme_check() {
*__slate_slot_0 = iommu_go_to_state(1u32);
if *__slate_slot_0 == 0 {
amd_iommu_detected = true;
iommu_detected = 1;
x86_init.iommu.iommu_init = Some(amd_iommu_init);
return;
}
}
}
if cc_platform_has(7u32) {
cc_platform_clear(7u32);
}
}
}

// Parsing functions for the AMD IOMMU specific kernel command line
// options.
#[unsafe(link_section = ".init.text")]
extern "C" fn parse_amd_iommu_dump(str: *mut u8) -> i32 {
unsafe {
amd_iommu_dump = true;
}
1
}

#[unsafe(link_section = ".init.text")]
extern "C" fn parse_amd_iommu_intr(mut str: *mut u8) -> i32 {
while (unsafe { *str }) != 0 {
if (unsafe { strncmp(str as *const u8, c"legacy".as_ptr() as *const u8, 6u64) }) == 0 {
unsafe {
amd_iommu_guest_ir = 1;
}
break;
}
if (unsafe { strncmp(str as *const u8, c"vapic".as_ptr() as *const u8, 5u64) }) == 0 {
unsafe {
amd_iommu_guest_ir = 2;
}
break;
}
let __v1: *mut u8 = str;
str = __v1.wrapping_offset(1 as isize);
}
1
}

#[unsafe(link_section = ".init.text")]
extern "C" fn parse_amd_iommu_options(mut str: *mut u8) -> i32 {
if str.is_null() {
return 22i32.wrapping_neg();
}
while (unsafe { *str }) != 0 {
if (unsafe { strncmp(str as *const u8, c"fullflush".as_ptr() as *const u8, 9u64) }) == 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014AMD-Vi: amd_iommu=fullflush deprecated; use iommu.strict=1 instead\n".as_ptr() as *const u8) };
unsafe { iommu_set_dma_strict() };
} else {
if (unsafe { strncmp(str as *const u8, c"force_enable".as_ptr() as *const u8, 12u64) }) == 0 {
unsafe {
amd_iommu_force_enable = true;
}
} else {
if (unsafe { strncmp(str as *const u8, c"off".as_ptr() as *const u8, 3u64) }) == 0 {
unsafe {
amd_iommu_disabled = true;
}
} else {
if (unsafe { strncmp(str as *const u8, c"force_isolation".as_ptr() as *const u8, 15u64) }) == 0 {
unsafe {
amd_iommu_force_isolation = true;
}
} else {
if (unsafe { strncmp(str as *const u8, c"pgtbl_v1".as_ptr() as *const u8, 8u64) }) == 0 {
unsafe {
amd_iommu_pgtable = 1;
}
} else {
if (unsafe { strncmp(str as *const u8, c"pgtbl_v2".as_ptr() as *const u8, 8u64) }) == 0 {
unsafe {
amd_iommu_pgtable = 2;
}
} else {
if (unsafe { strncmp(str as *const u8, c"irtcachedis".as_ptr() as *const u8, 11u64) }) == 0 {
unsafe {
amd_iommu_irtcachedis = true;
}
} else {
if (unsafe { strncmp(str as *const u8, c"nohugepages".as_ptr() as *const u8, 11u64) }) == 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016AMD-Vi: Restricting V1 page-sizes to 4KiB".as_ptr() as *const u8) };
unsafe {
amd_iommu_pgsize_bitmap = _AC!() << 12i32;
}
} else {
if (unsafe { strncmp(str as *const u8, c"v2_pgsizes_only".as_ptr() as *const u8, 15u64) }) == 0 {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016AMD-Vi: Restricting V1 page-sizes to 4KiB/2MiB/1GiB".as_ptr() as *const u8) };
unsafe {
amd_iommu_pgsize_bitmap = _AC!() << 12i32 | 2097152u64 | 1073741824u64;
}
} else {
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x015AMD-Vi: Unknown option - '%s'\n".as_ptr() as *const u8, str) };
}
}
}
}
}
}
}
}
}
let __v8: u64 = unsafe { strcspn(str as *const u8, c",".as_ptr() as *const u8) };
let __v9: *mut u8 = str;
str = __v9.wrapping_offset(__v8 as isize);
while ((unsafe { *str }) as i32) == 44 {
let __v11: *mut u8 = str;
str = __v11.wrapping_offset(1 as isize);
}
}
1
}

#[unsafe(link_section = ".init.text")]
extern "C" fn parse_ivrs_ioapic(str: *mut u8) -> i32 {
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut devid: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(devid) as *mut u32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut id: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(id) as *mut i32;
let mut r#fn: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(r#fn) as *mut u32;
let mut dev: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(dev) as *mut u32;
let mut bus: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(bus) as *mut u32;
let mut seg: u32 = unsafe { core::mem::zeroed() };
unsafe {
seg = 0;
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if sscanf(str as *const u8, c"=%d@%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_9), core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11), core::ptr::addr_of_mut!(*__slate_slot_10)) == 4 {
__v6 = true;
} else {
__v6 = sscanf(str as *const u8, c"=%d@%x:%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_9), core::ptr::addr_of_mut!(seg), core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11), core::ptr::addr_of_mut!(*__slate_slot_10)) == 5;
}
if !__v6 {
if sscanf(str as *const u8, c"[%d]=%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_9), core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11), core::ptr::addr_of_mut!(*__slate_slot_10)) == 4 {
__v5 = true;
} else {
__v5 = sscanf(str as *const u8, c"[%d]=%x:%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_9), core::ptr::addr_of_mut!(seg), core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11), core::ptr::addr_of_mut!(*__slate_slot_10)) == 5;
}
if __v5 {
while false {
}
_printk(c"\x014AMD-Vi: ivrs_ioapic%s option format deprecated; use ivrs_ioapic=%d@%04x:%02x:%02x.%d instead\n".as_ptr() as *const u8, str, *__slate_slot_9, seg, *__slate_slot_12, *__slate_slot_11, *__slate_slot_10);
} else {
while false {
}
_printk(c"\x013AMD-Vi: Invalid command line: ivrs_ioapic%s\n".as_ptr() as *const u8, str);
return 1;
}
}
if early_ioapic_map_size == 4 {
while false {
}
_printk(c"\x013AMD-Vi: Early IOAPIC map overflow - ignoring ivrs_ioapic%s\n".as_ptr() as *const u8, str);
return 1;
} else {
*__slate_slot_7 = IVRS_GET_SBDF_ID!(seg, *__slate_slot_12, *__slate_slot_11, *__slate_slot_10);
cmdline_maps = true;
let __v1: i32 = early_ioapic_map_size;
early_ioapic_map_size = __v1.wrapping_add(1);
*__slate_slot_8 = __v1;
early_ioapic_map.0[*__slate_slot_8 as usize].id = *__slate_slot_9 as u8;
early_ioapic_map.0[*__slate_slot_8 as usize].devid = *__slate_slot_7;
early_ioapic_map.0[*__slate_slot_8 as usize].cmd_line = true;
return 1;
}
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn parse_ivrs_hpet(str: *mut u8) -> i32 {
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut devid: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(devid) as *mut u32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut id: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(id) as *mut i32;
let mut r#fn: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(r#fn) as *mut u32;
let mut dev: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(dev) as *mut u32;
let mut bus: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(bus) as *mut u32;
let mut seg: u32 = unsafe { core::mem::zeroed() };
unsafe {
seg = 0;
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if sscanf(str as *const u8, c"=%d@%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_9), core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11), core::ptr::addr_of_mut!(*__slate_slot_10)) == 4 {
__v6 = true;
} else {
__v6 = sscanf(str as *const u8, c"=%d@%x:%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_9), core::ptr::addr_of_mut!(seg), core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11), core::ptr::addr_of_mut!(*__slate_slot_10)) == 5;
}
if !__v6 {
if sscanf(str as *const u8, c"[%d]=%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_9), core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11), core::ptr::addr_of_mut!(*__slate_slot_10)) == 4 {
__v5 = true;
} else {
__v5 = sscanf(str as *const u8, c"[%d]=%x:%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_9), core::ptr::addr_of_mut!(seg), core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11), core::ptr::addr_of_mut!(*__slate_slot_10)) == 5;
}
if __v5 {
while false {
}
_printk(c"\x014AMD-Vi: ivrs_hpet%s option format deprecated; use ivrs_hpet=%d@%04x:%02x:%02x.%d instead\n".as_ptr() as *const u8, str, *__slate_slot_9, seg, *__slate_slot_12, *__slate_slot_11, *__slate_slot_10);
} else {
while false {
}
_printk(c"\x013AMD-Vi: Invalid command line: ivrs_hpet%s\n".as_ptr() as *const u8, str);
return 1;
}
}
if early_hpet_map_size == 4 {
while false {
}
_printk(c"\x013AMD-Vi: Early HPET map overflow - ignoring ivrs_hpet%s\n".as_ptr() as *const u8, str);
return 1;
} else {
*__slate_slot_7 = IVRS_GET_SBDF_ID!(seg, *__slate_slot_12, *__slate_slot_11, *__slate_slot_10);
cmdline_maps = true;
let __v1: i32 = early_hpet_map_size;
early_hpet_map_size = __v1.wrapping_add(1);
*__slate_slot_8 = __v1;
early_hpet_map.0[*__slate_slot_8 as usize].id = *__slate_slot_9 as u8;
early_hpet_map.0[*__slate_slot_8 as usize].devid = *__slate_slot_7;
early_hpet_map.0[*__slate_slot_8 as usize].cmd_line = true;
return 1;
}
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn parse_ivrs_acpihid(str: *mut u8) -> i32 {
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32; // size with NULL terminator
let mut acpiid: core::mem::MaybeUninit<__SlateAlign16<[u8; 266]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut [u8; 266] = core::ptr::addr_of_mut!(acpiid) as *mut [u8; 266];
let mut addr: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut u8 = core::ptr::addr_of_mut!(addr) as *mut *mut u8;
let mut p: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut u8 = core::ptr::addr_of_mut!(p) as *mut *mut u8;
let mut uid: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut u8 = core::ptr::addr_of_mut!(uid) as *mut *mut u8;
let mut hid: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut u8 = core::ptr::addr_of_mut!(hid) as *mut *mut u8;
let mut r#fn: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(r#fn) as *mut u32;
let mut dev: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(dev) as *mut u32;
let mut bus: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u32 = core::ptr::addr_of_mut!(bus) as *mut u32;
let mut seg: u32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_17: {
seg = 0;
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_17, {
let mut __t0: [u8; 266] = core::mem::zeroed();
__t0
});
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_18 = strchr(str as *const u8, 64i32);
if (*__slate_slot_18).is_null() {
*__slate_slot_18 = strchr(str as *const u8, 61i32);
if (*__slate_slot_18).is_null() {
break '__join_17;
} else {
let __v15: *mut u8 = *__slate_slot_18;
*__slate_slot_18 = __v15.wrapping_offset(1 as isize);
if strlen(*__slate_slot_18 as *const u8) > 265u64 {
break '__join_17;
} else {
if sscanf(str as *const u8, c"[%x:%x.%x]=%s".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_24), core::ptr::addr_of_mut!(*__slate_slot_23), core::ptr::addr_of_mut!(*__slate_slot_22), (*__slate_slot_17).as_mut_ptr() as *mut u8) == 4 {
__v13 = true;
} else {
__v13 = sscanf(str as *const u8, c"[%x:%x:%x.%x]=%s".as_ptr() as *const u8, core::ptr::addr_of_mut!(seg), core::ptr::addr_of_mut!(*__slate_slot_24), core::ptr::addr_of_mut!(*__slate_slot_23), core::ptr::addr_of_mut!(*__slate_slot_22), (*__slate_slot_17).as_mut_ptr() as *mut u8) == 5;
}
if __v13 {
while false {
}
_printk(c"\x014AMD-Vi: ivrs_acpihid%s option format deprecated; use ivrs_acpihid=%s@%04x:%02x:%02x.%d instead\n".as_ptr() as *const u8, str, (*__slate_slot_17).as_mut_ptr() as *mut u8, seg, *__slate_slot_24, *__slate_slot_23, *__slate_slot_22);
} else {
break '__join_17;
}
}
}
} else {
// We have the '@', make it the terminator to get just the acpiid
let __v11: *mut u8 = *__slate_slot_18;
*__slate_slot_18 = __v11.wrapping_offset(1 as isize);
*__v11 = 0;
if strlen(str as *const u8) > 265u64 {
break '__join_17;
} else {
if sscanf(str as *const u8, c"=%s".as_ptr() as *const u8, (*__slate_slot_17).as_mut_ptr() as *mut u8) != 1 {
break '__join_17;
} else {
if sscanf(*__slate_slot_18 as *const u8, c"%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_24), core::ptr::addr_of_mut!(*__slate_slot_23), core::ptr::addr_of_mut!(*__slate_slot_22)) == 3 {
__v9 = true;
} else {
__v9 = sscanf(*__slate_slot_18 as *const u8, c"%x:%x:%x.%x".as_ptr() as *const u8, core::ptr::addr_of_mut!(seg), core::ptr::addr_of_mut!(*__slate_slot_24), core::ptr::addr_of_mut!(*__slate_slot_23), core::ptr::addr_of_mut!(*__slate_slot_22)) == 4;
}
if !__v9 {
break '__join_17;
}
}
}
}
if early_acpihid_map_size == 4 {
while false {
}
_printk(c"\x013AMD-Vi: Early ACPI HID map overflow - ignoring ivrs_acpihid%s\n".as_ptr() as *const u8, str);
return 1;
} else {
*__slate_slot_19 = (*__slate_slot_17).as_mut_ptr() as *mut u8;
*__slate_slot_21 = strsep(core::ptr::addr_of_mut!(*__slate_slot_19), c":".as_ptr() as *const u8);
*__slate_slot_20 = *__slate_slot_19;
if (*__slate_slot_21).is_null() || *(*__slate_slot_21) == 0 || (*__slate_slot_20).is_null() {
while false {
}
_printk(c"\x013AMD-Vi: Invalid command line: hid or uid\n".as_ptr() as *const u8);
return 1;
} else {
// Ignore leading zeroes after ':', so e.g., AMDI0095:00
// will match AMDI0095:0 in the second strcmp in acpi_dev_hid_uid_match
while (*(*__slate_slot_20) as i32) == 48 && *(*__slate_slot_20).wrapping_offset(1 as isize) != 0 {
let __v5: *mut u8 = *__slate_slot_20;
*__slate_slot_20 = __v5.wrapping_offset(1 as isize);
}
if strlen(*__slate_slot_21 as *const u8) >= 9 {
while false {
}
_printk(c"\x013AMD-Vi: Invalid command line: hid is too long\n".as_ptr() as *const u8);
return 1;
} else {
if strlen(*__slate_slot_20 as *const u8) >= 256 {
while false {
}
_printk(c"\x013AMD-Vi: Invalid command line: uid is too long\n".as_ptr() as *const u8);
return 1;
} else {
let __v1: i32 = early_acpihid_map_size;
early_acpihid_map_size = __v1.wrapping_add(1);
*__slate_slot_16 = __v1;
memcpy((core::ptr::addr_of_mut!(early_acpihid_map.0[*__slate_slot_16 as usize].hid) as *mut u8) as *mut (), *__slate_slot_21 as *const (), strlen(*__slate_slot_21 as *const u8));
memcpy((core::ptr::addr_of_mut!(early_acpihid_map.0[*__slate_slot_16 as usize].uid) as *mut u8) as *mut (), *__slate_slot_20 as *const (), strlen(*__slate_slot_20 as *const u8));
early_acpihid_map.0[*__slate_slot_16 as usize].devid = IVRS_GET_SBDF_ID!(seg, *__slate_slot_24, *__slate_slot_23, *__slate_slot_22);
early_acpihid_map.0[*__slate_slot_16 as usize].cmd_line = true;
return 1;
}
}
}
}
}
while false {
}
_printk(c"\x013AMD-Vi: Invalid command line: ivrs_acpihid%s\n".as_ptr() as *const u8, str);
return 1;
}
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_parse_amd_iommu_dump: [u8; 15] = [97, 109, 100, 95, 105, 111, 109, 109, 117, 95, 100, 117, 109, 112, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_parse_amd_iommu_dump: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_parse_amd_iommu_dump) as *const u8 };
__t0.setup_func = Some(parse_amd_iommu_dump);
__t0.early = 0i32;
__t0
};

#[unsafe(link_section = ".init.rodata")]
static __setup_str_parse_amd_iommu_options: [u8; 11] = [97, 109, 100, 95, 105, 111, 109, 109, 117, 61, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_parse_amd_iommu_options: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_parse_amd_iommu_options) as *const u8 };
__t0.setup_func = Some(parse_amd_iommu_options);
__t0.early = 0i32;
__t0
};

#[unsafe(link_section = ".init.rodata")]
static __setup_str_parse_amd_iommu_intr: [u8; 16] = [97, 109, 100, 95, 105, 111, 109, 109, 117, 95, 105, 110, 116, 114, 61, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_parse_amd_iommu_intr: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_parse_amd_iommu_intr) as *const u8 };
__t0.setup_func = Some(parse_amd_iommu_intr);
__t0.early = 0i32;
__t0
};

#[unsafe(link_section = ".init.rodata")]
static __setup_str_parse_ivrs_ioapic: [u8; 12] = [105, 118, 114, 115, 95, 105, 111, 97, 112, 105, 99, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_parse_ivrs_ioapic: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_parse_ivrs_ioapic) as *const u8 };
__t0.setup_func = Some(parse_ivrs_ioapic);
__t0.early = 0i32;
__t0
};

#[unsafe(link_section = ".init.rodata")]
static __setup_str_parse_ivrs_hpet: [u8; 10] = [105, 118, 114, 115, 95, 104, 112, 101, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_parse_ivrs_hpet: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_parse_ivrs_hpet) as *const u8 };
__t0.setup_func = Some(parse_ivrs_hpet);
__t0.early = 0i32;
__t0
};

#[unsafe(link_section = ".init.rodata")]
static __setup_str_parse_ivrs_acpihid: [u8; 13] = [105, 118, 114, 115, 95, 97, 99, 112, 105, 104, 105, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_parse_ivrs_acpihid: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_parse_ivrs_acpihid) as *const u8 };
__t0.setup_func = Some(parse_ivrs_acpihid);
__t0.early = 0i32;
__t0
};

#[unsafe(no_mangle)]
extern "C" fn amd_iommu_pasid_supported() -> bool {
// CPU page table size should match IOMMU guest page table size
if cpu_feature_enabled!(528i32) != 0 && (unsafe { amd_iommu_gpt_level }) != 5 {
return false;
}
if !amd_iommu_gt_ppr_supported() {
return false;
}
// If SNP page mode 0 is not supported, then DTE[Mode]=0 is prohibited
// on SNP-enabled system (i.e. EFR[SNPSup]=1). IOMMUv2 page table
// cannot be used without setting up IOMMUv1 page table.
!unsafe { amd_iommu_snp_en } || (unsafe { amd_iommu_snp_en }) && unsafe { amd_iommu_snp_mode0_sup }
}

#[unsafe(no_mangle)]
extern "C" fn get_amd_iommu(idx: u32) -> *mut amd_iommu {
let mut i: u32 = 0;
let mut iommu: *mut amd_iommu = unsafe { core::mem::zeroed() };
iommu = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!(amd_iommu_list) });
while list_entry_is_head!(iommu, unsafe { core::ptr::addr_of_mut!(amd_iommu_list) }, list) == 0 {
let __v1: u32 = i;
i = __v1.wrapping_add(1);
if __v1 == idx {
return iommu;
}
iommu = list_next_entry__3!(iommu, list);
}
core::ptr::null_mut::<amd_iommu>()
}

// IOMMU EFR Performance Counter support functionality. This code allows
// access to the IOMMU PC functionality.
#[unsafe(no_mangle)]
extern "C" fn amd_iommu_pc_get_max_banks(idx: u32) -> u8 {
let iommu: *mut amd_iommu = get_amd_iommu(idx);
if !iommu.is_null() {
return unsafe { (*iommu).max_banks };
}
0
}

#[unsafe(no_mangle)]
extern "C" fn amd_iommu_pc_supported() -> bool {
unsafe { amd_iommu_pc_present }
}

#[unsafe(no_mangle)]
extern "C" fn amd_iommu_pc_get_max_counters(idx: u32) -> u8 {
let iommu: *mut amd_iommu = get_amd_iommu(idx);
if !iommu.is_null() {
return unsafe { (*iommu).max_counters };
}
0
}

fn iommu_pc_get_set_reg(iommu: *mut amd_iommu, bank: u8, cntr: u8, fxn: u8, value: *mut u64, is_write: bool) -> i32 {
let mut offset: u32 = 0;
let mut max_offset_lim: u32 = 0;
// Make sure the IOMMU PC resource is available
if !unsafe { amd_iommu_pc_present } {
return 19i32.wrapping_neg();
}
// Check for valid iommu and pc register indexing
let mut __v0: i64;
let __ret_warn_on: i32 = (iommu.is_null() || (fxn as i32) > 40 || (fxn as i32) & 7 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("703: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 703b - ., 3; .popsection\n# {0}", const 703 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_pc_get_set_reg_1, sym __slate_asm_string_iommu_pc_get_set_reg_2, const 4033 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("704: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 704b - ., 4; .popsection\n# {0}", const 704 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 19i32.wrapping_neg();
}
offset = ((64 | (bank as i32)) << 12i32 | (cntr as i32) << 8i32 | (fxn as i32)) as u32;
// Limit the offset to the hw defined mmio region aperture
max_offset_lim = ((64 | ((unsafe { (*iommu).max_banks }) as i32)) << 12i32 | ((unsafe { (*iommu).max_counters }) as i32) << 8i32 | 40) as u32;
if offset < 262144 || offset > max_offset_lim {
return 22i32.wrapping_neg();
}
if is_write {
let val: u64 = (unsafe { *value }) & GENMASK_ULL__15!(47i32, 0i32);
writel(val as u32, unsafe { (*iommu).mmio_base }.wrapping_offset(offset as isize) as *mut ());
writel((val >> 32i32) as u32, unsafe { (*iommu).mmio_base }.wrapping_offset(offset as isize).wrapping_offset(4 as isize) as *mut ());
} else {
unsafe {
*value = readl(unsafe { (*iommu).mmio_base }.wrapping_offset(offset as isize).wrapping_offset(4 as isize) as *const ()) as u64;
}
unsafe {
*value = *value << 32i32;
}
let __v8: u32 = readl(unsafe { (*iommu).mmio_base }.wrapping_offset(offset as isize) as *const ());
unsafe {
*value = *value | (__v8 as u64);
}
let __v13: u64 = unsafe { *value };
let __v14: u64 = __v13 & GENMASK_ULL__7!(47i32, 0i32);
unsafe {
*value = __v14;
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn amd_iommu_pc_get_reg(iommu: *mut amd_iommu, bank: u8, cntr: u8, fxn: u8, value: *mut u64) -> i32 {
if iommu.is_null() {
return 22i32.wrapping_neg();
}
iommu_pc_get_set_reg(iommu, bank, cntr, fxn, value, false)
}

#[unsafe(no_mangle)]
extern "C" fn amd_iommu_pc_set_reg(iommu: *mut amd_iommu, bank: u8, cntr: u8, fxn: u8, value: *mut u64) -> i32 {
if iommu.is_null() {
return 22i32.wrapping_neg();
}
iommu_pc_get_set_reg(iommu, bank, cntr, fxn, value, true)
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_early_enable_iommus_2: [u8; 1] = *b"\0";

static __slate_asm_string_early_enable_iommus_3: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_enable_iommus_vapic_6: [u8; 1] = *b"\0";

static __slate_asm_string_enable_iommus_vapic_7: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_find_last_devid_acpi_8: [u8; 1] = *b"\0";

static __slate_asm_string_find_last_devid_acpi_9: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_find_last_devid_from_ivhd_12: [u8; 1] = *b"\0";

static __slate_asm_string_find_last_devid_from_ivhd_13: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_init_iommu_all_20: [u8; 1] = *b"\0";

static __slate_asm_string_init_iommu_all_21: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_init_iommu_one_7: [u8; 54] = *b"System has more IOMMUs than supported by this driver\n\0";

static __slate_asm_string_init_iommu_one_8: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_iommu_enable_command_buffer_0: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_enable_command_buffer_1: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_iommu_enable_event_buffer_1: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_enable_event_buffer_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_iommu_ga_log_enable_4: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_ga_log_enable_5: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_iommu_go_to_state_4: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_go_to_state_5: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_iommu_init_ga_log_5: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_init_ga_log_6: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_iommu_init_pci_7: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_init_pci_8: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_iommu_pc_get_set_reg_1: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_pc_get_set_reg_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_iommu_set_device_table_0: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_set_device_table_1: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";

static __slate_asm_string_state_next_1: [u8; 1] = *b"\0";

static __slate_asm_string_state_next_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/iommu/amd/init.c\0";
