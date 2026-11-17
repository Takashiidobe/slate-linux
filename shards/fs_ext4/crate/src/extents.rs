
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
//! Copyright (c) 2003-2006, Cluster File Systems, Inc, info@clusterfs.com
//! Written by Alex Tomas <alex@clusterfs.com>
//!
//! Architecture independence:
//!   Copyright (c) 2005, Bull S.A.
//!   Written by Pierre Peiffer <pierre.peiffer@bull.net>
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
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! do_div - returns 2 values: calculate remainder and update new dividend
//! @n: uint64_t dividend (will be updated)
//! @base: uint32_t divisor
//!
//! Summary:
//! ``uint32_t remainder = n % base;``
//! ``n = n / base;``
//!
//! Return: (uint32_t)remainder
//!
//! NOTE: macro parameter @n is evaluated multiple times,
//! beware of side effects!
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
//! Internal kernel timer frequency
//! Sign Flag
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
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
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
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
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! Prevent speculative execution past this barrier.
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
//! writeable
//! Operation not permitted
//! File name too long
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
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
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
//! Used in tsk->__state:
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! these are defined by POSIX and also present in glibc's dirent.h
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
//! Helper macro to avoid gfp flags if they are the default one
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
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
//! Bits set in the VMA until the stack is in its final location
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! Generate SD flag indexes
//! Desired design of maximum size and alignment (see RFC2553)
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
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Types of block ranges for iomap mappings:
//! allocate range
//! Mask of all supported fallocate modes.  Only one can be set at a time.
//!
//! In addition to the mode bit, the mode argument can also encode flags.
//! FALLOC_FL_KEEP_SIZE is the only supported flag so far.
//! Returned by EXT4_IOC_GET_ES_CACHE as an additional possible flag.
//! It indicates that the entry in extent status cache is for a hole.
//! Miscellaneous stuff.
//! See the comment in asm-generic/io.h about ioremap_np().
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! Set this bit for virtual address instead of SG list.
//! Largest digest size among all hash algorithms supported by fs-verity.
//! Currently assumed to be <= size of fsverity_descriptor::root_hash.
//! Fast commit tags
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! If CONFIG_KUNIT is not enabled, these stubs quietly disappear.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
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
// __must_not_hold() - function attribute, caller must not hold context lock
//
// Function attribute declaring that the caller must not hold the given context
// lock instance(s).
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// Overflow Flag
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
// informational
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Dummy printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Block device required
// Device or resource busy
// Invalid argument
// Text file busy
// File too large
// No space left on device
// Read-only file system
// Bad CRC detected
// Value too large for defined data type
// Operation not supported on transport endpoint
// Operation now in progress
// Filesystem is corrupted
// Quota exceeded
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// No static inline type checking - see Subtlety (1) above.
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
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// Set the IDR flag and the IDR_FREE tag
// Generate SD flag bits
// Implementation specific max size
// default is extend size
// de-allocates range
// FALLOC_FL_COLLAPSE_RANGE is used to remove a range of a file
// without leaving a hole in the file. The contents of the file beyond
// the range being removed is appended to the start offset of the range
// being removed (i.e. the hole that was punched is "collapsed"),
// resulting in a file layout that looks like the range that was
// removed never existed. As such collapsing a range of a file changes
// the size of the file, reducing it by the same length of the range
// that has been removed by the operation.
//
// Different filesystems may implement different limitations on the
// granularity of the operation. Most will limit operations to
// filesystem block size boundaries, but this boundary may be larger or
// smaller depending on the filesystem and/or the configuration of the
// filesystem or file.
//
// Attempting to collapse a range that crosses the end of the file is
// considered an illegal operation - just use ftruncate(2) if you need
// to collapse a range that crosses EOF.
// FALLOC_FL_ZERO_RANGE is used to convert a range of file to zeros preferably
// without issuing data IO. Blocks should be preallocated for the regions that
// span holes in the file, and the entire range is preferable converted to
// unwritten extents - even though file system may choose to zero out the
// extent or do whatever which will result in reading zeros from the range
// while the range remains allocated for the file.
//
// This can be also used to preallocate blocks past EOF in the same way as
// with fallocate. Flag FALLOC_FL_KEEP_SIZE should cause the inode
// size to remain the same.
// FALLOC_FL_INSERT_RANGE is use to insert space within the file size without
// overwriting any existing data. The contents of the file beyond offset are
// shifted towards right by len bytes to create a hole.  As such, this
// operation will increase the size of the file by len bytes.
//
// Different filesystems may implement different limitations on the granularity
// of the operation. Most will limit operations to filesystem block size
// boundaries, but this boundary may be larger or smaller depending on
// the filesystem and/or the configuration of the filesystem or file.
//
// Attempting to insert space using this flag at OR beyond the end of
// the file is considered an illegal operation - just use ftruncate(2) or
// fallocate(2) with mode 0 for such type of operations.
// FALLOC_FL_WRITE_ZEROES zeroes a specified file range in such a way that
// subsequent writes to that range do not require further changes to the file
// mapping metadata. This flag is beneficial for subsequent pure overwriting
// within this range, as it can save on block allocation and, consequently,
// significant metadata changes. Therefore, filesystems that always require
// out-of-place writes should not support this flag.
//
// Different filesystems may implement different limitations on the
// granularity of the zeroing operation. Most will preferably be accelerated
// by submitting write zeroes command if the backing storage supports, which
// may not physically write zeros to the media.
//
// This flag cannot be specified in conjunction with the FALLOC_FL_KEEP_SIZE.
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// struct hash_alg_common - define properties of message digest
// @digestsize: Size of the result of the transformation. A buffer of this size
// 	        must be available to the @final and @finup calls, so they can
// 	        store the resulting hash into it. For various predefined sizes,
// 	        search include/crypto/ using
// 	        git grep _DIGEST_SIZE include/crypto.
// @statesize: Size of the block for partial state of the transformation. A
// 	       buffer of this size must be passed to the @export function as it
// 	       will save the partial state of the transformation into it. On the
// 	       other side, the @import function will load the state from a
// 	       buffer of this size as well.
// @base: Start of data structure of cipher algorithm. The common data
// 	  structure of crypto_alg contains information common to all ciphers.
// 	  The hash_alg_common data structure now adds the hash-specific
// 	  information.
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
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
/// @a is a power of 2 value
macro_rules! ALIGN__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL__2!($x, $a)
};
}

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

macro_rules! BUFFER_TRACE {
() => {
while false {
}
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

/// Translate a block number to a cluster number
macro_rules! EXT4_B2C {
($sbi:expr, $blk:expr) => {
($blk) >> unsafe { (*$sbi).s_cluster_bits }
};
}

macro_rules! EXT4_BLOCKS_PER_GROUP {
($s:expr) => {
unsafe { (*EXT4_SB($s)).s_blocks_per_group }
};
}

macro_rules! EXT4_BLOCK_ALIGN {
($size:expr, $blkbits:expr) => {
ALIGN__2!($size, 1i32 << (($blkbits) as i32))
};
}

macro_rules! EXT4_BLOCK_ALIGN__2 {
($size:expr, $blkbits:expr) => {
ALIGN__2!($size, 1i32 << $blkbits)
};
}

macro_rules! EXT4_BLOCK_SIZE_BITS {
($s:expr) => {
unsafe { (*$s).s_blocksize_bits }
};
}

macro_rules! EXT4_BLOCK_SIZE {
($s:expr) => {
unsafe { (*$s).s_blocksize }
};
}

/// Translate a cluster number to a block number
macro_rules! EXT4_C2B {
($sbi:expr, $cluster:expr) => {
($cluster) << unsafe { (*$sbi).s_cluster_bits }
};
}

macro_rules! EXT4_CLUSTER_SIZE {
($s:expr) => {
EXT4_BLOCK_SIZE!($s) << unsafe { (*EXT4_SB($s)).s_cluster_bits }
};
}

macro_rules! EXT4_EXTENT_TAIL_OFFSET {
($hdr:expr) => {
(core::mem::size_of::<ext4_extent_header>() as u64).wrapping_add((core::mem::size_of::<ext4_extent>() as u64).wrapping_mul(((unsafe { (*$hdr).eh_max }) as i32) as u64))
};
}

macro_rules! EXT4_I {
($inode:expr) => {
{
let mut __v: *mut ext4_inode_info;
__v = ((unsafe { ((($inode) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(ext4_inode_info) as isize)) }) as *mut ()) as *mut ext4_inode_info;
__v
}
};
}

macro_rules! EXT4_JOURNAL {
($inode:expr) => {
unsafe { (*EXT4_SB((*$inode).i_sb)).s_journal }
};
}

macro_rules! EXT4_LBLK_CMASK {
($s:expr, $lblk:expr) => {
($lblk) & !unsafe { (*$s).s_cluster_ratio }.wrapping_sub(1)
};
}

macro_rules! EXT4_LBLK_COFF {
($s:expr, $lblk:expr) => {
($lblk) & unsafe { (*$s).s_cluster_ratio }.wrapping_sub(1)
};
}

macro_rules! EXT4_LBLK_TO_B {
($inode:expr, $lblk:expr) => {
(($lblk) as i64) << ((unsafe { (*$inode).i_blkbits }) as i32)
};
}

macro_rules! EXT4_MAXQUOTAS_TRANS_BLOCKS {
($sb:expr) => {
3i32.wrapping_mul(EXT4_QUOTA_TRANS_BLOCKS!($sb))
};
}

macro_rules! EXT4_MAX_BLOCKS {
($size:expr, $offset:expr, $blkbits:expr) => {
(EXT4_BLOCK_ALIGN__2!($size.wrapping_add($offset), $blkbits) >> $blkbits).wrapping_sub(($offset) >> $blkbits)
};
}

macro_rules! EXT4_MAX_BLOCKS__2 {
($size:expr, $offset:expr, $blkbits:expr) => {
(EXT4_BLOCK_ALIGN!($size.wrapping_add($offset), $blkbits) >> (($blkbits) as i32)).wrapping_sub(($offset) >> (($blkbits) as i32))
};
}

/// Define the number of metadata blocks we need to account to modify data.
///
/// This include super block, inode block, quota blocks and xattr blocks
macro_rules! EXT4_META_TRANS_BLOCKS {
($sb:expr) => {
6u32.wrapping_add(EXT4_MAXQUOTAS_TRANS_BLOCKS!($sb) as u32)
};
}

/// Translate # of blks to # of clusters
macro_rules! EXT4_NUM_B2C__2 {
($sbi:expr, $blks:expr) => {
$blks.wrapping_add(unsafe { (*$sbi).s_cluster_ratio }).wrapping_sub(1) >> unsafe { (*$sbi).s_cluster_bits }
};
}

/// Mask out the low bits to get the starting block of the cluster
macro_rules! EXT4_PBLK_CMASK {
($s:expr, $pblk:expr) => {
($pblk) & !((unsafe { (*$s).s_cluster_ratio }) as u64).wrapping_sub(1)
};
}

/// Amount of blocks needed for quota update - we know that the structure was
/// allocated so we need to update only data block
macro_rules! EXT4_QUOTA_TRANS_BLOCKS {
($sb:expr) => {
if ext4_quota_capable($sb) { 1i32 } else { 0i32 }
};
}

macro_rules! EXT_FIRST_EXTENT {
($__hdr__:expr) => {
(($__hdr__) as *mut u8).wrapping_offset(core::mem::size_of::<ext4_extent_header>() as isize) as *mut ext4_extent
};
}

macro_rules! EXT_FIRST_INDEX {
($__hdr__:expr) => {
(($__hdr__) as *mut u8).wrapping_offset(core::mem::size_of::<ext4_extent_header>() as isize) as *mut ext4_extent_idx
};
}

macro_rules! EXT_HAS_FREE_INDEX {
($__path__:expr) => {
((unsafe { (*(*$__path__).p_hdr).eh_entries }) as i32) < ((unsafe { (*(*$__path__).p_hdr).eh_max }) as i32)
};
}

macro_rules! EXT_LAST_EXTENT {
($__hdr__:expr) => {
EXT_FIRST_EXTENT!($__hdr__).wrapping_offset(((unsafe { (*$__hdr__).eh_entries }) as i32) as isize).wrapping_offset(-(1 as isize))
};
}

macro_rules! EXT_LAST_INDEX {
($__hdr__:expr) => {
EXT_FIRST_INDEX!($__hdr__).wrapping_offset(((unsafe { (*$__hdr__).eh_entries }) as i32) as isize).wrapping_offset(-(1 as isize))
};
}

macro_rules! EXT_MAX_EXTENT {
($__hdr__:expr) => {
if (unsafe { (*$__hdr__).eh_max }) != 0 { EXT_FIRST_EXTENT!($__hdr__).wrapping_offset(((unsafe { (*$__hdr__).eh_max }) as i32) as isize).wrapping_offset(-(1 as isize)) } else { core::ptr::null_mut::<ext4_extent>() }
};
}

macro_rules! EXT_MAX_INDEX {
($__hdr__:expr) => {
if (unsafe { (*$__hdr__).eh_max }) != 0 { EXT_FIRST_INDEX!($__hdr__).wrapping_offset(((unsafe { (*$__hdr__).eh_max }) as i32) as isize).wrapping_offset(-(1 as isize)) } else { core::ptr::null_mut::<ext4_extent_idx>() }
};
}

macro_rules! IS_ALIGNED {
($x:expr, $a:expr) => {
($x) & (($a) as i64).wrapping_sub(1) == 0
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

macro_rules! IS_ENCRYPTED {
($inode:expr) => {
(unsafe { (*$inode).i_flags }) & 16384u32
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

macro_rules! IS_SYNC {
($inode:expr) => {
__IS_FLG!($inode, BIT!(4i32)) != 0 || (unsafe { (*$inode).i_flags }) & 1u32 != 0
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

macro_rules! S_ISDIR {
($m:expr) => {
(($m) as i32) & 61440 == 16384
};
}

macro_rules! S_ISLNK {
($m:expr) => {
(($m) as i32) & 61440 == 40960
};
}

macro_rules! S_ISREG {
($m:expr) => {
(($m) as i32) & 61440 == 32768
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

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as i64).wrapping_sub(1))
};
}

/// Note that nosuid etc flags are inode-specific: setting some file-system
/// flags just means all the inodes inherit those flags by default. It might be
/// possible to override it selectively if you really wanted to with some
/// ioctl() that is not currently implemented.
///
/// Exception: SB_RDONLY is always applied to the entire file system.
///
/// Unfortunately, it is possible to change a filesystems flags with it mounted
/// with files in use.  This means that all of the inodes will not have their
/// i_flags updated.  Hence, i_flags no longer inherit the superblock mount
/// flags, so these have to be checked separately. -- rmk@arm.uk.linux.org
macro_rules! __IS_FLG {
($inode:expr, $flg:expr) => {
(unsafe { (*(*$inode).i_sb).s_flags }) & $flg
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u32, ($val) as u32) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i64, $val) };
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

/// __acquire_shared() - function to acquire context lock shared
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x with shared
/// access.
macro_rules! __acquire_shared {
() => {
__acquire_shared_ctx_lock!();
};
}

macro_rules! __acquire_shared_ctx_lock {
() => {
while false {
}
};
}

macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
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

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

/// __release_shared() - function to release context lock shared
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x with shared
/// access.
macro_rules! __release_shared {
() => {
__release_shared_ctx_lock!();
};
}

macro_rules! __release_shared_ctx_lock {
() => {
while false {
}
};
}

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask__2 {
($y:expr) => {
$y.wrapping_sub(1) as i64
};
}

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask {
($y:expr) => {
$y.wrapping_sub(1) as i64
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
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u64>() }) as u64) < 4) as i32)) }
};
}

/// Atomic operations are already serializing on x86
macro_rules! __smp_mb__before_atomic {
() => {
while false {
}
};
}

macro_rules! __types_ok {
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
macro_rules! __verify_pcpu_ptr__3 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<__SlateRecord_types_h_197>() as *const ();
if true {
break;
}
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
macro_rules! __verify_pcpu_ptr__4 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i64>() as *const ();
if true {
break;
}
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
macro_rules! __verify_pcpu_ptr__2 {
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

macro_rules! cond_resched__8 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/fs/ext4/extents.c".as_ptr() as *const u8, 3031i32, 0u32);
__v = _cond_resched();
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__5 {
($ptr:expr) => {
{
let mut __v: *mut ext4_io_end_vec;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__13!() as isize)) }) as *mut ()) as *mut ext4_io_end_vec;
__v
}
};
}

macro_rules! debug_normal_state_change {
() => {
while false {
}
};
}

macro_rules! ext4_ext_check {
($inode:expr, $eh:expr, $depth:expr, $pblk:expr) => {
__ext4_ext_check(c"ext4_ext_check_inode".as_ptr() as *const u8, 521u32, $inode, $eh, ($depth) as i32, ($pblk) as u64, 0u32)
};
}

macro_rules! ext4_ext_check__2 {
($inode:expr, $eh:expr, $depth:expr, $pblk:expr) => {
__ext4_ext_check(c"ext4_ext_remove_space".as_ptr() as *const u8, 2977u32, $inode, $eh, $depth, ($pblk) as u64, 0u32)
};
}

macro_rules! ext4_ext_check__3 {
($inode:expr, $eh:expr, $depth:expr, $pblk:expr) => {
__ext4_ext_check(c"ext4_ext_precache".as_ptr() as *const u8, 631u32, $inode, $eh, $depth, ($pblk) as u64, 0u32)
};
}

macro_rules! ext4_ext_dirty {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_swap_extents".as_ptr() as *const u8, 6026u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__2 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_convert_to_initialized".as_ptr() as *const u8, 3713u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__3 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_split_extent_at".as_ptr() as *const u8, 3248u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__4 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_shift_path_extents".as_ptr() as *const u8, 5471u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__5 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_split_extent_at".as_ptr() as *const u8, 3300u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__6 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_split_extent_zeroout".as_ptr() as *const u8, 3399u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__7 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_rm_idx".as_ptr() as *const u8, 2353u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__8 {
($inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_replay_update_ex".as_ptr() as *const u8, 6195u32, core::ptr::null_mut::<jbd2_journal_handle>(), $inode, $path)
};
}

macro_rules! ext4_ext_dirty__9 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_swap_extents".as_ptr() as *const u8, 6022u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__10 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_split_convert_extents".as_ptr() as *const u8, 3897u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__11 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_remove_space".as_ptr() as *const u8, 3088u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__12 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_rm_idx".as_ptr() as *const u8, 2337u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__13 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_insert_index".as_ptr() as *const u8, 1037u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__14 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_insert_extent".as_ptr() as *const u8, 2210u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__15 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_split".as_ptr() as *const u8, 1271u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__16 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_split".as_ptr() as *const u8, 1185u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__17 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_correct_indexes".as_ptr() as *const u8, 1747u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__18 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_correct_indexes".as_ptr() as *const u8, 1767u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__19 {
($inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_replay_shrink_inode".as_ptr() as *const u8, 6226u32, core::ptr::null_mut::<jbd2_journal_handle>(), $inode, $path);
};
}

macro_rules! ext4_ext_dirty__20 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_shift_path_extents".as_ptr() as *const u8, 5454u32, $handle, $inode, $path)
};
}

macro_rules! ext4_ext_dirty__21 {
($handle:expr, $inode:expr, $path:expr) => {
__ext4_ext_dirty(c"ext4_ext_rm_leaf".as_ptr() as *const u8, 2778u32, $handle, $inode, $path)
};
}

macro_rules! ext4_handle_dirty_metadata__16 {
($handle:expr, $inode:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_ext_grow_indepth".as_ptr() as *const u8, 1367u32, $handle, $inode, $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__50 {
($handle:expr, $inode:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_ext_split".as_ptr() as *const u8, 1173u32, $handle, $inode, $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__52 {
($handle:expr, $inode:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_ext_split".as_ptr() as *const u8, 1259u32, $handle, $inode, $bh) }
};
}

macro_rules! ext4_journal_get_create_access__5 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_create_access(c"ext4_ext_split".as_ptr() as *const u8, 1135u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_create_access__6 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_create_access(c"ext4_ext_split".as_ptr() as *const u8, 1214u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_create_access__7 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_create_access(c"ext4_ext_grow_indepth".as_ptr() as *const u8, 1341u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__51 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_ext_get_access".as_ptr() as *const u8, 166u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_start__5 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 5155u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__7 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 5779u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__10 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 4879u32, $type, $nblocks, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__13 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 5683u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__17 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 4745u32, $type, $nblocks, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__24 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 5078u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__25 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 4698u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__45 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 4660u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start_reserved {
($handle:expr, $type:expr) => {
unsafe { __ext4_journal_start_reserved($handle, 5211u32, $type) }
};
}

macro_rules! ext4_journal_start_with_revoke {
($inode:expr, $type:expr, $blocks:expr, $revoke_creds:expr) => {
__ext4_journal_start($inode, 2865u32, $type, $blocks, 0i32, $revoke_creds)
};
}

macro_rules! ext4_journal_stop__4 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_ext_remove_space".as_ptr() as *const u8, 3096u32, $handle) };
};
}

macro_rules! ext4_journal_stop__12 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_alloc_file_blocks".as_ptr() as *const u8, 4672u32, $handle) };
};
}

macro_rules! ext4_journal_stop__14 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_collapse_range".as_ptr() as *const u8, 5726u32, $handle) };
};
}

macro_rules! ext4_journal_stop__17 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_convert_unwritten_io_end_vec".as_ptr() as *const u8, 5225u32, $handle) }
};
}

macro_rules! ext4_journal_stop__27 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_convert_unwritten_extents".as_ptr() as *const u8, 5181u32, $handle) }
};
}

macro_rules! ext4_journal_stop__55 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_insert_range".as_ptr() as *const u8, 5850u32, $handle) };
};
}

macro_rules! ext4_journal_stop__56 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_alloc_file_blocks".as_ptr() as *const u8, 4764u32, $handle) }
};
}

macro_rules! ext4_journal_stop__72 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_alloc_file_blocks".as_ptr() as *const u8, 4721u32, $handle) }
};
}

macro_rules! ext4_journal_stop__73 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_alloc_file_blocks".as_ptr() as *const u8, 4676u32, $handle) }
};
}

macro_rules! ext4_journal_stop__79 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_ext_remove_space".as_ptr() as *const u8, 2887u32, $handle) };
};
}

macro_rules! ext4_journal_stop__87 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_convert_unwritten_extents_atomic".as_ptr() as *const u8, 5103u32, $handle) }
};
}

macro_rules! ext4_journal_stop__92 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_zero_range".as_ptr() as *const u8, 4897u32, $handle) };
};
}

macro_rules! ext4_mark_inode_dirty__3 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_collapse_range".as_ptr() as *const u8, 5717u32) }
};
}

macro_rules! ext4_mark_inode_dirty__4 {
($__i:expr) => {
unsafe { __ext4_mark_inode_dirty(core::ptr::null_mut::<jbd2_journal_handle>(), $__i, c"ext4_ext_replay_update_ex".as_ptr() as *const u8, 6199u32) };
};
}

macro_rules! ext4_mark_inode_dirty__5 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_alloc_file_blocks".as_ptr() as *const u8, 4762u32) }
};
}

macro_rules! ext4_mark_inode_dirty__11 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_insert_range".as_ptr() as *const u8, 5788u32) }
};
}

macro_rules! ext4_mark_inode_dirty__15 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_zero_range".as_ptr() as *const u8, 4888u32) }
};
}

macro_rules! ext4_mark_inode_dirty__21 {
($__i:expr) => {
unsafe { __ext4_mark_inode_dirty(core::ptr::null_mut::<jbd2_journal_handle>(), $__i, c"ext4_ext_replay_shrink_inode".as_ptr() as *const u8, 6217u32) };
};
}

macro_rules! ext4_mark_inode_dirty__27 {
($__i:expr) => {
unsafe { __ext4_mark_inode_dirty(core::ptr::null_mut::<jbd2_journal_handle>(), $__i, c"ext4_ext_replay_shrink_inode".as_ptr() as *const u8, 6228u32) };
};
}

macro_rules! ext4_mark_inode_dirty__31 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_alloc_file_blocks".as_ptr() as *const u8, 4671u32) };
};
}

macro_rules! ext4_mark_inode_dirty__35 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_convert_unwritten_extents_atomic".as_ptr() as *const u8, 5100u32) }
};
}

macro_rules! ext4_mark_inode_dirty__37 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"__ext4_ext_dirty".as_ptr() as *const u8, 207u32) }
};
}

macro_rules! ext4_mark_inode_dirty__40 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_ext_tree_init".as_ptr() as *const u8, 882u32) };
};
}

macro_rules! ext4_mark_inode_dirty__42 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_ext_truncate".as_ptr() as *const u8, 4565u32) }
};
}

macro_rules! ext4_mark_inode_dirty__51 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_convert_unwritten_extents".as_ptr() as *const u8, 5179u32) }
};
}

macro_rules! ext4_mark_inode_dirty__52 {
($__i:expr) => {
unsafe { __ext4_mark_inode_dirty(core::ptr::null_mut::<jbd2_journal_handle>(), $__i, c"ext4_ext_replay_set_iblocks".as_ptr() as *const u8, 6332u32) };
};
}

macro_rules! ext4_mark_inode_dirty__65 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_ext_grow_indepth".as_ptr() as *const u8, 1387u32) }
};
}

macro_rules! ext4_std_error__15 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_ext_insert_index".as_ptr() as *const u8, 1038u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__32 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_zero_range".as_ptr() as *const u8, 4882u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__45 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_alloc_file_blocks".as_ptr() as *const u8, 4771u32, $errno) };
}
if true {
break;
}
}
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
macro_rules! ilog2 {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub($n.leading_zeros() as i32) } } else { if (core::mem::size_of::<u64>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64($n) } }
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
macro_rules! ilog2__5 {
($n:expr) => {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<u32>() as u64) <= 4 { __ilog2_u32($n) } else { __ilog2_u64(($n) as u64) } }
};
}

/// in_range - Determine if a value lies within a range.
/// @val: Value to test.
/// @start: First value in range.
/// @len: Number of values in range.
///
/// This is more efficient than "if (start <= val && val < (start + len))".
/// It also gives a different answer if @start + @len overflows the size of
/// the type by a sufficient amount to encompass @val.  Decide for yourself
/// which behaviour you want, or prove that start + len never overflow.
/// Do not blindly replace one form with the other.
macro_rules! in_range {
($val:expr, $start:expr, $len:expr) => {
{
let __v: i32 = if (core::mem::size_of::<u32>() as u64) | (core::mem::size_of::<u32>() as u64) | (core::mem::size_of::<u32>() as u64) <= (core::mem::size_of::<u32>() as u64) {
in_range32($val, $start, $len) as i32
} else {
in_range64(($val) as u64, ($start) as u64, ($len) as u64) as i32
};
__v
}
};
}

/// in_range - Determine if a value lies within a range.
/// @val: Value to test.
/// @start: First value in range.
/// @len: Number of values in range.
///
/// This is more efficient than "if (start <= val && val < (start + len))".
/// It also gives a different answer if @start + @len overflows the size of
/// the type by a sufficient amount to encompass @val.  Decide for yourself
/// which behaviour you want, or prove that start + len never overflow.
/// Do not blindly replace one form with the other.
macro_rules! in_range__5 {
($val:expr, $start:expr, $len:expr) => {
in_range64(($val) as u64, ($start) as u64, ($len) as u64) as i32
};
}

/// in_range - Determine if a value lies within a range.
/// @val: Value to test.
/// @start: First value in range.
/// @len: Number of values in range.
///
/// This is more efficient than "if (start <= val && val < (start + len))".
/// It also gives a different answer if @start + @len overflows the size of
/// the type by a sufficient amount to encompass @val.  Decide for yourself
/// which behaviour you want, or prove that start + len never overflow.
/// Do not blindly replace one form with the other.
macro_rules! in_range__2 {
($val:expr, $start:expr, $len:expr) => {
in_range32($val, $start, ($len) as u32) as i32
};
}

/// in_range - Determine if a value lies within a range.
/// @val: Value to test.
/// @start: First value in range.
/// @len: Number of values in range.
///
/// This is more efficient than "if (start <= val && val < (start + len))".
/// It also gives a different answer if @start + @len overflows the size of
/// the type by a sufficient amount to encompass @val.  Decide for yourself
/// which behaviour you want, or prove that start + len never overflow.
/// Do not blindly replace one form with the other.
macro_rules! in_range__3 {
($val:expr, $start:expr, $len:expr) => {
in_range32($val, $start, $len) as i32
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
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

macro_rules! kcsan_release {
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
macro_rules! list_entry {
($ptr:expr) => {
container_of__5!($ptr)
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
macro_rules! list_first_entry__6 {
($ptr:expr) => {
list_entry!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__5 {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).$member.next })
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

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, vfs_inode) as u64
};
}

macro_rules! offsetof__13 {
() => {
core::mem::offset_of!(ext4_io_end_vec, list) as u64
};
}

/// order_base_2 - calculate the (rounded up) base 2 order of the argument
/// @n: parameter
///
/// The first few values calculated by this routine:
///  ob2(0) = 0
///  ob2(1) = 0
///  ob2(2) = 1
///  ob2(3) = 2
///  ob2(4) = 2
///  ob2(5) = 3
///  ... and so on.
macro_rules! order_base_2 {
($n:expr) => {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { if ($n) == 0 || ($n) == 1 { 0i32 } else { ilog2__5!($n.wrapping_sub(1)).wrapping_add(1) } } else { __order_base_2(($n) as u64) }
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

macro_rules! read_extent_tree_block {
($inode:expr, $idx:expr, $depth:expr, $flags:expr) => {
__read_extent_tree_block(c"ext4_ext_search_right".as_ptr() as *const u8, 1609u32, $inode, $idx, $depth, $flags)
};
}

macro_rules! read_extent_tree_block__2 {
($inode:expr, $idx:expr, $depth:expr, $flags:expr) => {
__read_extent_tree_block(c"ext4_ext_search_right".as_ptr() as *const u8, 1617u32, $inode, $idx, $depth, $flags)
};
}

macro_rules! read_extent_tree_block__3 {
($inode:expr, $idx:expr, $depth:expr, $flags:expr) => {
__read_extent_tree_block(c"ext4_ext_remove_space".as_ptr() as *const u8, 3023u32, $inode, $idx, $depth, $flags)
};
}

/// round_down - round down to next specified power of 2
/// @x: the value to round
/// @y: multiple to round down to (must be a power of 2)
///
/// Rounds @x down to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding down, use rounddown() below.
macro_rules! round_down {
($x:expr, $y:expr) => {
($x) & !__round_mask__2!($y)
};
}

/// round_down - round down to next specified power of 2
/// @x: the value to round
/// @y: multiple to round down to (must be a power of 2)
///
/// Rounds @x down to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding down, use rounddown() below.
macro_rules! round_down__2 {
($x:expr, $y:expr) => {
($x) & !__round_mask!($y)
};
}

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask!($y)).wrapping_add(1)
};
}

macro_rules! smp_mb__before_atomic {
() => {
loop {
kcsan_mb!();
__smp_mb__before_atomic!();
if true {
break;
}
}
};
}

macro_rules! static_branch_unlikely__13 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const18(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__8 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const9(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__14 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const14(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__9 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const10(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__15 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const15(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__16 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const19(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const1(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__10 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const12(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__17 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const13(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__4 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const4(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__18 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const17(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__3 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const3(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__11 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const8(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__2 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const2(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__6 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const6(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__5 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const5(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__7 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const7(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__19 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const16(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__12 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const11(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
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

macro_rules! test_bit__5 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if false && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit__4 {
($nr:expr, $addr:expr) => {
{
let __v: bool = (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v2
}
};
}

macro_rules! test_bit__3 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit__2 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! trace_set_current_state {
($state_value:expr) => {
loop {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_sched_set_state_tp.key) }) != 0 {
unsafe { __trace_set_current_state($state_value) };
}
if true {
break;
}
}
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
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__might_resched: static_call_key;
    static mut __cpu_online_mask: cpumask;
    static mut cpu_number: i32;
    static mut __tracepoint_sched_set_state_tp: tracepoint;
    static mut __SCK__cond_resched: static_call_key;
    static mut percpu_counter_batch: i32;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut dquot_operations: dquot_operations;
    static mut ext4_iomap_report_ops: iomap_ops;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_ext4_fallocate_enter: static_call_key;
    static mut __tracepoint_ext4_fallocate_enter: tracepoint;
    static mut __SCK__tp_func_ext4_zero_range: static_call_key;
    static mut __tracepoint_ext4_zero_range: tracepoint;
    static mut __SCK__tp_func_ext4_fallocate_exit: static_call_key;
    static mut __tracepoint_ext4_fallocate_exit: tracepoint;
    static mut __SCK__tp_func_ext4_ext_convert_to_initialized_enter: static_call_key;
    static mut __tracepoint_ext4_ext_convert_to_initialized_enter: tracepoint;
    static mut __SCK__tp_func_ext4_ext_convert_to_initialized_fastpath: static_call_key;
    static mut __tracepoint_ext4_ext_convert_to_initialized_fastpath: tracepoint;
    static mut __SCK__tp_func_ext4_ext_map_blocks_enter: static_call_key;
    static mut __tracepoint_ext4_ext_map_blocks_enter: tracepoint;
    static mut __SCK__tp_func_ext4_ext_map_blocks_exit: static_call_key;
    static mut __tracepoint_ext4_ext_map_blocks_exit: tracepoint;
    static mut __SCK__tp_func_ext4_ext_load_extent: static_call_key;
    static mut __tracepoint_ext4_ext_load_extent: tracepoint;
    static mut __SCK__tp_func_ext4_ext_handle_unwritten_extents: static_call_key;
    static mut __tracepoint_ext4_ext_handle_unwritten_extents: tracepoint;
    static mut __SCK__tp_func_ext4_get_implied_cluster_alloc_exit: static_call_key;
    static mut __tracepoint_ext4_get_implied_cluster_alloc_exit: tracepoint;
    static mut __SCK__tp_func_ext4_ext_show_extent: static_call_key;
    static mut __tracepoint_ext4_ext_show_extent: tracepoint;
    static mut __SCK__tp_func_ext4_remove_blocks: static_call_key;
    static mut __tracepoint_ext4_remove_blocks: tracepoint;
    static mut __SCK__tp_func_ext4_ext_rm_leaf: static_call_key;
    static mut __tracepoint_ext4_ext_rm_leaf: tracepoint;
    static mut __SCK__tp_func_ext4_ext_rm_idx: static_call_key;
    static mut __tracepoint_ext4_ext_rm_idx: tracepoint;
    static mut __SCK__tp_func_ext4_ext_remove_space: static_call_key;
    static mut __tracepoint_ext4_ext_remove_space: tracepoint;
    static mut __SCK__tp_func_ext4_ext_remove_space_done: static_call_key;
    static mut __tracepoint_ext4_ext_remove_space_done: tracepoint;
    static mut __SCK__tp_func_ext4_collapse_range: static_call_key;
    static mut __tracepoint_ext4_collapse_range: tracepoint;
    static mut __SCK__tp_func_ext4_insert_range: static_call_key;
    static mut __tracepoint_ext4_insert_range: tracepoint;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn ____wrong_branch_error() -> bool;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memmove(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn io_schedule_timeout(arg0: i64) -> i64;
    fn __trace_set_current_state(arg0: i32);
    fn __SCT__cond_resched() -> i32;
    fn rcu_is_watching() -> bool;
    fn down_read(arg0: *mut rw_semaphore);
    fn down_write(arg0: *mut rw_semaphore);
    fn up_read(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn percpu_counter_add_batch(arg0: *mut percpu_counter, arg1: i64, arg2: i32);
    fn __compiletime_assert_205();
    fn __compiletime_assert_206();
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn __compiletime_assert_384();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_390();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __compiletime_assert_399();
    fn __compiletime_assert_400();
    fn __compiletime_assert_401();
    fn __compiletime_assert_402();
    fn __mark_inode_dirty(arg0: *mut inode, arg1: i32);
    fn file_modified(arg0: *mut file) -> i32;
    fn inode_dio_wait(arg0: *mut inode);
    fn inode_newsize_ok(arg0: *const inode, arg1: i64) -> i32;
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn truncate_pagecache(arg0: *mut inode, arg1: i64);
    fn pagecache_isize_extended(arg0: *mut inode, arg1: i64, arg2: i64);
    fn filemap_write_and_wait_range(arg0: *mut address_space, arg1: i64, arg2: i64) -> i32;
    fn bdev_getblk(arg0: *mut block_device, arg1: u64, arg2: u32, arg3: u32) -> *mut buffer_head;
    fn __brelse(arg0: *mut buffer_head);
    fn unlock_buffer(arg0: *mut buffer_head);
    fn __lock_buffer(arg0: *mut buffer_head);
    fn bh_uptodate_or_lock(arg0: *mut buffer_head) -> i32;
    fn crc32c(arg0: u32, arg1: *const (), arg2: u64) -> u32;
    fn jbd2__journal_restart(arg0: *mut jbd2_journal_handle, arg1: i32, arg2: i32, arg3: u32) -> i32;
    fn jbd2_journal_extend(arg0: *mut jbd2_journal_handle, arg1: i32, arg2: i32) -> i32;
    fn dquot_reclaim_space_nodirty(arg0: *mut inode, arg1: i64);
    fn fiemap_prep(arg0: *mut inode, arg1: *mut fiemap_extent_info, arg2: u64, arg3: *mut u64, arg4: u32) -> i32;
    fn fiemap_fill_next_extent(arg0: *mut fiemap_extent_info, arg1: u64, arg2: u64, arg3: u64, arg4: u32) -> i32;
    fn iomap_iter(arg0: *mut iomap_iter, arg1: *const iomap_ops) -> i32;
    fn __compiletime_assert_602();
    fn __compiletime_assert_605();
    fn iomap_iter_continue(arg0: *const iomap_iter, arg1: *mut iomap, arg2: *mut iomap, arg3: i32) -> i32;
    fn iomap_fiemap(arg0: *mut inode, arg1: *mut fiemap_extent_info, arg2: u64, arg3: u64, arg4: *const iomap_ops) -> i32;
    fn ext4_es_insert_extent(arg0: *mut inode, arg1: u32, arg2: u32, arg3: u64, arg4: u32, arg5: bool);
    fn ext4_es_cache_extent(arg0: *mut inode, arg1: u32, arg2: u32, arg3: u64, arg4: u32);
    fn ext4_es_remove_extent(arg0: *mut inode, arg1: u32, arg2: u32);
    fn ext4_es_find_extent_range(arg0: *mut inode, arg1: Option<unsafe extern "C" fn(*mut extent_status) -> i32>, arg2: u32, arg3: u32, arg4: *mut extent_status);
    fn ext4_es_lookup_extent(arg0: *mut inode, arg1: u32, arg2: *mut u32, arg3: *mut extent_status, arg4: *mut u64) -> i32;
    fn ext4_remove_pending(arg0: *mut inode, arg1: u32);
    fn ext4_is_pending(arg0: *mut inode, arg1: u32) -> bool;
    fn ext4_new_meta_blocks(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: u64, arg3: u32, arg4: *mut u64, arg5: *mut i32) -> u64;
    fn ext4_should_retry_alloc(arg0: *mut super_block, arg1: *mut i32) -> i32;
    fn ext4_inode_to_goal_block(arg0: *mut inode) -> u64;
    fn ext4_fc_mark_ineligible(arg0: *mut super_block, arg1: i32, arg2: *mut jbd2_journal_handle);
    fn ext4_fc_commit(arg0: *mut journal_s, arg1: u32) -> i32;
    fn ext4_fc_record_regions(arg0: *mut super_block, arg1: i32, arg2: u32, arg3: u64, arg4: i32, arg5: i32) -> i32;
    fn ext4_mb_new_blocks(arg0: *mut jbd2_journal_handle, arg1: *mut ext4_allocation_request, arg2: *mut i32) -> u64;
    fn ext4_discard_preallocations(arg0: *mut inode);
    fn ext4_free_blocks(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut buffer_head, arg3: u64, arg4: u64, arg5: i32);
    fn ext4_mb_mark_bb(arg0: *mut super_block, arg1: u64, arg2: i32, arg3: bool);
    fn ext4_check_map_extents_env(arg0: *mut inode);
    fn ext4_get_inode_loc(arg0: *mut inode, arg1: *mut ext4_iloc) -> i32;
    fn ext4_break_layouts(arg0: *mut inode) -> i32;
    fn ext4_truncate_page_cache_block_range(arg0: *mut inode, arg1: i64, arg2: i64) -> i32;
    fn ext4_punch_hole(arg0: *mut file, arg1: i64, arg2: i64) -> i32;
    fn ext4_chunk_trans_blocks(arg0: *mut inode, arg1: i32) -> i32;
    fn ext4_chunk_trans_extent(arg0: *mut inode, arg1: i32) -> i32;
    fn ext4_meta_trans_blocks(arg0: *mut inode, arg1: i32, arg2: i32, arg3: i32) -> i32;
    fn ext4_block_zero_eof(arg0: *mut inode, arg1: i64, arg2: i64) -> i32;
    fn ext4_zero_partial_blocks(arg0: *mut inode, arg1: i64, arg2: i64, arg3: *mut u32) -> i32;
    fn ext4_issue_zeroout(arg0: *mut inode, arg1: u32, arg2: u64, arg3: u32) -> i32;
    fn ext4_read_bh(arg0: *mut buffer_head, arg1: u32, arg2: Option<unsafe extern "C" fn(*mut bio)>, arg3: bool) -> i32;
    fn __ext4_error(arg0: *mut super_block, arg1: *const u8, arg2: u32, arg3: bool, arg4: i32, arg5: u64, arg6: *const u8, ...);
    fn __ext4_error_inode(arg0: *mut inode, arg1: *const u8, arg2: u32, arg3: u64, arg4: i32, arg5: *const u8, ...);
    fn __ext4_std_error(arg0: *mut super_block, arg1: *const u8, arg2: u32, arg3: i32);
    fn __ext4_warning(arg0: *mut super_block, arg1: *const u8, arg2: u32, arg3: *const u8, ...);
    fn __ext4_msg(arg0: *mut super_block, arg1: *const u8, arg2: *const u8, ...);
    fn __compiletime_assert_659();
    fn ext4_update_disksize_before_punch(arg0: *mut inode, arg1: i64, arg2: i64) -> i32;
    fn ext4_convert_inline_data(arg0: *mut inode) -> i32;
    fn ext4_inode_block_valid(arg0: *mut inode, arg1: u64, arg2: u32) -> i32;
    fn ext4_map_blocks(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut ext4_map_blocks, arg3: i32) -> i32;
    fn ext4_orphan_add(arg0: *mut jbd2_journal_handle, arg1: *mut inode) -> i32;
    fn ext4_orphan_del(arg0: *mut jbd2_journal_handle, arg1: *mut inode) -> i32;
    fn __ext4_mark_inode_dirty(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *const u8, arg3: u32) -> i32;
    fn __ext4_journal_get_write_access(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle, arg3: *mut super_block, arg4: *mut buffer_head, arg5: u32) -> i32;
    fn __ext4_journal_get_create_access(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle, arg3: *mut super_block, arg4: *mut buffer_head, arg5: u32) -> i32;
    fn __ext4_handle_dirty_metadata(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle, arg3: *mut inode, arg4: *mut buffer_head) -> i32;
    fn __ext4_journal_start_sb(arg0: *mut inode, arg1: *mut super_block, arg2: u32, arg3: i32, arg4: i32, arg5: i32, arg6: i32) -> *mut jbd2_journal_handle;
    fn __ext4_journal_stop(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle) -> i32;
    fn __ext4_journal_start_reserved(arg0: *mut jbd2_journal_handle, arg1: u32, arg2: i32) -> *mut jbd2_journal_handle;
    fn __ext4_journal_ensure_credits(arg0: *mut jbd2_journal_handle, arg1: i32, arg2: i32, arg3: i32) -> i32;
    fn ext4_inode_journal_mode(arg0: *mut inode) -> i32;
    fn __traceiter_ext4_fallocate_enter(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: i64, arg4: i32) -> i32;
    fn __SCT__tp_func_ext4_fallocate_enter(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: i64, arg4: i32) -> i32;
    fn __compiletime_assert_1038();
    fn __traceiter_ext4_zero_range(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: i64, arg4: i32) -> i32;
    fn __SCT__tp_func_ext4_zero_range(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: i64, arg4: i32) -> i32;
    fn __compiletime_assert_1052();
    fn __traceiter_ext4_fallocate_exit(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: u32, arg4: i32) -> i32;
    fn __SCT__tp_func_ext4_fallocate_exit(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: u32, arg4: i32) -> i32;
    fn __compiletime_assert_1059();
    fn __traceiter_ext4_ext_convert_to_initialized_enter(arg0: *mut (), arg1: *mut inode, arg2: *mut ext4_map_blocks, arg3: *mut ext4_extent) -> i32;
    fn __SCT__tp_func_ext4_ext_convert_to_initialized_enter(arg0: *mut (), arg1: *mut inode, arg2: *mut ext4_map_blocks, arg3: *mut ext4_extent) -> i32;
    fn __compiletime_assert_1094();
    fn __traceiter_ext4_ext_convert_to_initialized_fastpath(arg0: *mut (), arg1: *mut inode, arg2: *mut ext4_map_blocks, arg3: *mut ext4_extent, arg4: *mut ext4_extent) -> i32;
    fn __SCT__tp_func_ext4_ext_convert_to_initialized_fastpath(arg0: *mut (), arg1: *mut inode, arg2: *mut ext4_map_blocks, arg3: *mut ext4_extent, arg4: *mut ext4_extent) -> i32;
    fn __compiletime_assert_1101();
    fn __traceiter_ext4_ext_map_blocks_enter(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u32, arg4: u32) -> i32;
    fn __SCT__tp_func_ext4_ext_map_blocks_enter(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u32, arg4: u32) -> i32;
    fn __compiletime_assert_1108();
    fn __traceiter_ext4_ext_map_blocks_exit(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: *mut ext4_map_blocks, arg4: i32) -> i32;
    fn __SCT__tp_func_ext4_ext_map_blocks_exit(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: *mut ext4_map_blocks, arg4: i32) -> i32;
    fn __compiletime_assert_1122();
    fn __traceiter_ext4_ext_load_extent(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u64) -> i32;
    fn __SCT__tp_func_ext4_ext_load_extent(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u64) -> i32;
    fn __compiletime_assert_1136();
    fn __traceiter_ext4_ext_handle_unwritten_extents(arg0: *mut (), arg1: *mut inode, arg2: *mut ext4_map_blocks, arg3: i32, arg4: u32, arg5: u64) -> i32;
    fn __SCT__tp_func_ext4_ext_handle_unwritten_extents(arg0: *mut (), arg1: *mut inode, arg2: *mut ext4_map_blocks, arg3: i32, arg4: u32, arg5: u64) -> i32;
    fn __compiletime_assert_1185();
    fn __traceiter_ext4_get_implied_cluster_alloc_exit(arg0: *mut (), arg1: *mut super_block, arg2: *mut ext4_map_blocks, arg3: i32) -> i32;
    fn __SCT__tp_func_ext4_get_implied_cluster_alloc_exit(arg0: *mut (), arg1: *mut super_block, arg2: *mut ext4_map_blocks, arg3: i32) -> i32;
    fn __compiletime_assert_1192();
    fn __traceiter_ext4_ext_show_extent(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u64, arg4: u16) -> i32;
    fn __SCT__tp_func_ext4_ext_show_extent(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u64, arg4: u16) -> i32;
    fn __compiletime_assert_1199();
    fn __traceiter_ext4_remove_blocks(arg0: *mut (), arg1: *mut inode, arg2: *mut ext4_extent, arg3: u32, arg4: u64, arg5: *mut partial_cluster) -> i32;
    fn __SCT__tp_func_ext4_remove_blocks(arg0: *mut (), arg1: *mut inode, arg2: *mut ext4_extent, arg3: u32, arg4: u64, arg5: *mut partial_cluster) -> i32;
    fn __compiletime_assert_1206();
    fn __traceiter_ext4_ext_rm_leaf(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: *mut ext4_extent, arg4: *mut partial_cluster) -> i32;
    fn __SCT__tp_func_ext4_ext_rm_leaf(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: *mut ext4_extent, arg4: *mut partial_cluster) -> i32;
    fn __compiletime_assert_1213();
    fn __traceiter_ext4_ext_rm_idx(arg0: *mut (), arg1: *mut inode, arg2: u64) -> i32;
    fn __SCT__tp_func_ext4_ext_rm_idx(arg0: *mut (), arg1: *mut inode, arg2: u64) -> i32;
    fn __compiletime_assert_1220();
    fn __traceiter_ext4_ext_remove_space(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u32, arg4: i32) -> i32;
    fn __SCT__tp_func_ext4_ext_remove_space(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u32, arg4: i32) -> i32;
    fn __compiletime_assert_1227();
    fn __traceiter_ext4_ext_remove_space_done(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u32, arg4: i32, arg5: *mut partial_cluster, arg6: u16) -> i32;
    fn __SCT__tp_func_ext4_ext_remove_space_done(arg0: *mut (), arg1: *mut inode, arg2: u32, arg3: u32, arg4: i32, arg5: *mut partial_cluster, arg6: u16) -> i32;
    fn __compiletime_assert_1234();
    fn __traceiter_ext4_collapse_range(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: i64) -> i32;
    fn __SCT__tp_func_ext4_collapse_range(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: i64) -> i32;
    fn __compiletime_assert_1311();
    fn __traceiter_ext4_insert_range(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: i64) -> i32;
    fn __SCT__tp_func_ext4_insert_range(arg0: *mut (), arg1: *mut inode, arg2: i64, arg3: i64) -> i32;
    fn __compiletime_assert_1318();
    fn __compiletime_assert_1526();
    fn __compiletime_assert_1529();
    fn __compiletime_assert_1544();
    fn __compiletime_assert_1547();
    fn __compiletime_assert_1550();
    fn __compiletime_assert_1553();
    fn __compiletime_assert_1571();
    fn __compiletime_assert_1581();
    fn __compiletime_assert_1584();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.returnaddress.p0"]
    fn __slate_intrinsic_ded315013ea6523(arg0: i32) -> *mut ();
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
struct io_bitmap {
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
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


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


fn le16_add_cpu(var: *mut u16, val: u16) {
unsafe {
*var = (*var as i32).wrapping_add(val as i32) as u16;
}
}

fn le32_add_cpu(var: *mut u32, val: u32) {
unsafe {
*var = unsafe { *var }.wrapping_add(val);
}
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


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_0() };
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
__verify_pcpu_ptr__2!();
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
fn arch_test_and_set_bit_lock(nr: i64, addr: *mut u64) -> bool {
arch_test_and_set_bit(nr, addr)
}

#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
struct static_key_mod {
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
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 3; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
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
core::arch::asm!("4: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 4b - ., 4; .popsection\n# {0}", const 4 as i32, options(nostack, att_syntax));
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
#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

fn test_and_set_bit_lock(nr: i64, addr: *mut u64) -> bool {
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit_lock(nr, addr)
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


fn __order_base_2(n: u64) -> i32 {
if n > 1 { ilog2!(n.wrapping_sub(1)).wrapping_add(1) } else { 0 }
}

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
    __slate_bits_0: __slate_bits::__SlateBits127U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
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
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn IS_ERR_OR_NULL(ptr: *const ()) -> bool {
unlikely!(ptr.is_null()) != 0 || IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_sched_set_state_tp, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fallocate_enter, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[inline(always)]
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_zero_range, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const4(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fallocate_exit, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const5(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_convert_to_initialized_enter, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const6(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_convert_to_initialized_fastpath, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const7(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_map_blocks_enter, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const8(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_map_blocks_exit, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const9(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_load_extent, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const10(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_handle_unwritten_extents, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const11(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_get_implied_cluster_alloc_exit, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn arch_static_branch__const12(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_show_extent, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const13(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_remove_blocks, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn arch_static_branch__const14(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_rm_leaf, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const15(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_rm_idx, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const16(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_remove_space, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const17(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_ext_remove_space_done, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const18(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_collapse_range, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const19(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_insert_range, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
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
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
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
struct static_key_false {
    key: static_key,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits134U0,
}


#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
}

#[inline(always)]
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// max_t - return maximum of two values, using the specified type
// @type: data type to use
// @x: first value
// @y: second value
fn in_range64(val: u64, start: u64, len: u64) -> bool {
val.wrapping_sub(start) < len
}

fn in_range32(val: u32, start: u32, len: u32) -> bool {
val.wrapping_sub(start) < len
}

// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
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


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

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


fn __might_resched(file: *const u8, line: i32, offsets: u32) {
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
#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[inline(always)]
fn cpumask_check(cpu: u32) -> u32 {
cpu_max_bits_warn(cpu, 64u32);
cpu
}

// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit__5!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
}

// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
#[inline(always)]
fn cpu_online(cpu: u32) -> bool {
cpumask_test_cpu(cpu as i32, (unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask)
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
    __slate_bits_0: __slate_bits::__SlateBits135U0,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct audit_context {
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
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
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


fn mem_alloc_profiling_enabled() -> bool {
false
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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
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
struct vm_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_func {
    func: *mut (),
    data: *mut (),
    prio: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits228U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint {
    name: *const u8,
    key: static_key_false,
    static_call_key: *mut static_call_key,
    static_call_tramp: *mut (),
    iterator: *mut (),
    probestub: *mut (),
    funcs: *mut tracepoint_func,
    ext: *mut tracepoint_ext,
}


// If a tracepoint needs to be called from a header file, it is not
// recommended to call it directly, as tracepoints in header files
// may cause side-effects and bloat the kernel. Instead, use
// tracepoint_enabled() to test if the tracepoint is enabled, then if
// it is, call a wrapper function defined in a C file that will then
// call the tracepoint.
//
// For "trace_foo_bar()", you would need to create a wrapper function
// in a C file to call trace_foo_bar():
//   void do_trace_foo_bar(args) { trace_foo_bar(args); }
// Then in the header file, declare the tracepoint:
//   DECLARE_TRACEPOINT(foo_bar);
// And call your wrapper:
//   static inline void some_inlined_function() {
//            [..]
//            if (tracepoint_enabled(foo_bar))
//                    do_trace_foo_bar(args);
//            [..]
//   }
//
// Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//   but is safe to have in headers, where trace_foo_bar_enabled() is not.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
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
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
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
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


// Convenience macros for the sake of set_current_state:
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
    __slate_bits_0: __slate_bits::__SlateBits291U0,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
    __slate_bits_0: __slate_bits::__SlateBits78U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits78U1,
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
    __slate_bits_2: __slate_bits::__SlateBits78U2,
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
// plug has ts that needs updating
// Per-process atomic flags.
// May not gain new privileges.
// Spread page cache over cpuset
// Speculative Store Bypass disabled
// Speculative Store Bypass force disabled
// Indirect branch speculation restricted
// Indirect branch speculation permanently restricted
// Speculative Store Bypass clear on execve()
#[inline(always)]
fn _cond_resched() -> i32 {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__cond_resched as *const ()) };
unsafe { __v0.unwrap()() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__cond_resched_88: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__cond_resched) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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


// Defined as a macro as it is a very low level header included from
// areas that don't even know about current.  This gives the rcu_read_lock()
// nesting depth, but makes sense only if CONFIG_PREEMPT_RCU -- in other
// types of kernel builds, the rcu_read_lock() nesting depth is unknowable.
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


fn rwsem_is_locked(sem: *mut rw_semaphore) -> i32 {
((atomic_long_read((unsafe { core::ptr::addr_of_mut!((*sem).count) }) as *const __SlateRecord_types_h_197) as u64) != 0) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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
struct pidfs_attr {
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_167() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ssp).srcu_ctrp) as *const *mut srcu_ctr) };
scp = __v0;
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__4!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
let mut __v13: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__3!();
let mut __v15: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v16: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v16 = pfo_val__;
tcp_ptr__ = __v16;
let __v18: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_locks) }) as u64);
tcp_ptr__ = __v18;
__v15 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v13 = __v15;
atomic_long_inc(__v13);
}
// Y, and implicit RCU reader.
// Y, and implicit RCU reader.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
__acquire_shared!();
scp
}

fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__4!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
let mut __v12: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__3!();
let mut __v14: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v15: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v15 = pfo_val__;
tcp_ptr__ = __v15;
let __v17: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_unlocks) }) as u64);
tcp_ptr__ = __v17;
__v14 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v12 = __v14;
atomic_long_inc(__v12);
}
// Z, and implicit RCU reader.
// Z, and implicit RCU reader.
}

fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
}

// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
fn srcu_read_lock_fast_notrace(ssp: *mut srcu_struct) -> *mut srcu_ctr {
srcu_check_read_flavor(ssp, 4i32);
let retval: *mut srcu_ctr = __srcu_read_lock_fast(ssp);
retval
}

fn srcu_read_unlock_fast_notrace(ssp: *mut srcu_struct, scp: *mut srcu_ctr) {
srcu_check_read_flavor(ssp, 4i32);
__srcu_read_unlock_fast(ssp, scp);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


#[inline(always)]
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
}

#[inline(always)]
fn class_srcu_fast_notrace_constructor(l: *mut srcu_struct) -> __SlateRecord_srcu_h_645 {
let mut _t: __SlateRecord_srcu_h_645 = {
let mut __t0: __SlateRecord_srcu_h_645 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_srcu_h_645 = core::ptr::addr_of_mut!(_t);
unsafe {
(*_T).scp = srcu_read_lock_fast_notrace((*_T).lock);
}
return _t;
}

#[inline(always)]
fn __class_srcu_fast_notrace_cleanup_ctx(_T: *mut *mut __SlateRecord_srcu_h_645) {
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


fn percpu_counter_add(fbc: *mut percpu_counter, amount: i64) {
unsafe { percpu_counter_add_batch(fbc, amount, percpu_counter_batch) };
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
struct fserror_event {
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
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits424U0,
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
struct fs_context {
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
fn gfpflags_allow_blocking(gfp_flags: u32) -> bool {
gfp_flags & (BIT!(10i32) as u32) != 0
}

// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
fn current_gfp_context(mut flags: u32) -> u32 {
let mut pflags: u32 = 0;
let mut __v0: u32;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_205() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*get_current()).flags) as *const u32) };
pflags = __v0;
if unlikely__3!(pflags & 269221888u32) != 0 {
// NOIO implies both NOIO and NOFS and it is a weaker context
// so always make sure it makes precedence
if pflags & 524288 != 0 {
let __v2: u32 = flags;
flags = __v2 & !((BIT!(6i32) as u32) | (BIT!(7i32) as u32));
} else {
if pflags & 262144 != 0 {
let __v4: u32 = flags;
flags = __v4 & !(BIT!(7i32) as u32);
}
}
if pflags & 268435456 != 0 {
let __v6: u32 = flags;
flags = __v6 & !(BIT!(3i32) as u32);
}
}
flags
}

fn memalloc_retry_wait(mut gfp_flags: u32) {
// We use io_schedule_timeout because waiting for memory
// typically included waiting for dirty pages to be
// written out, which requires IO.
loop {
debug_normal_state_change!();
trace_set_current_state!(2i32);
loop {
while false {
}
__WRITE_ONCE__3!((*get_current()).__state, 2i32);
if true {
break;
}
}
if true {
break;
}
}
gfp_flags = current_gfp_context(gfp_flags);
if gfpflags_allow_blocking(gfp_flags) && gfp_flags & (BIT!(16i32) as u32) == 0 {
unsafe { io_schedule_timeout(1i64) };
} else {
unsafe { io_schedule_timeout(20i64) };
}
// Probably waited already, no need for much more
// Probably didn't wait, and has now released a lock,
// so now is a good time to wait
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct io_uring_cmd {
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
struct rhash_head {
    next: *mut rhash_head,
}


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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits294U0,
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
if !(__native_word__3!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_359() };
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
struct wake_irq {
}


// dquot modified since read
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
struct dev_pm_qos {
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
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
if !(__native_word__6!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_384() };
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
    __slate_bits_0: __slate_bits::__SlateBits580U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
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
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
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
unsafe { __compiletime_assert_390() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("391: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 391b - ., 3; .popsection\n# {0}", const 391 as i32, options(nostack, att_syntax));
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
struct disk_stats {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_types_h_239 {
    bi_cookie: u32,
    __bi_nr_segments: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkcg_gq {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_buffer_head_h_62 {
    b_page: *mut page,
    b_folio: *mut folio,
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
struct mapping_metadata_bhs {
    mapping: *mut address_space,
    lock: spinlock,
    list: list_head,
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
struct request {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_alloc_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_trace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_flush_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pr_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hd_geometry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_report_zones_args {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_queue_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct disk_events {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct badblocks {
}


// Get bit address from inode->i_state to use with wait_var_event()
// infrastructre.
fn i_blocksize(node: *const inode) -> u32 {
(1i32 << ((unsafe { (*node).i_blkbits }) as i32)) as u32
}

fn inode_lock(inode: *mut inode) {
unsafe { down_write(core::ptr::addr_of_mut!((*inode).i_rwsem)) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
}


fn inode_unlock(inode: *mut inode) {
unsafe { up_write(core::ptr::addr_of_mut!((*inode).i_rwsem)) };
}

fn inode_lock_shared(inode: *mut inode) {
unsafe { down_read(core::ptr::addr_of_mut!((*inode).i_rwsem)) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_hw_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tag_set {
}


fn inode_unlock_shared(inode: *mut inode) {
unsafe { up_read(core::ptr::addr_of_mut!((*inode).i_rwsem)) };
}

fn inode_is_locked(inode: *mut inode) -> i32 {
rwsem_is_locked(unsafe { core::ptr::addr_of_mut!((*inode).i_rwsem) })
}

fn filemap_invalidate_lock(mapping: *mut address_space) {
unsafe { down_write(core::ptr::addr_of_mut!((*mapping).invalidate_lock)) };
}

fn filemap_invalidate_unlock(mapping: *mut address_space) {
unsafe { up_write(core::ptr::addr_of_mut!((*mapping).invalidate_lock)) };
}

fn i_size_read(inode: *const inode) -> i64 {
// Pairs with smp_store_release() in i_size_write()
let mut __v0: i64;
let mut ___p1: i64 = 0;
let mut __v1: i64;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_399() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*inode).i_size) as *const i64) };
___p1 = __v1;
loop {
if !__native_word__4!() {
unsafe { __compiletime_assert_400() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v0 = ___p1;
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct jbd2_revoke_table_s {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jbd2_h_499 {
    h_transaction: *mut transaction_s,
    h_journal: *mut journal_s,
}


fn i_size_write(inode: *mut inode, i_size: i64) {
// Pairs with smp_load_acquire() in i_size_read() to ensure
// changes related to inode size (such as page contents) are
// visible before we see the changed inode size.
loop {
kcsan_release!();
loop {
loop {
if !__native_word__4!() {
unsafe { __compiletime_assert_401() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_402() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!(*unsafe { core::ptr::addr_of_mut!((*inode).i_size) }, i_size);
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}

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
struct proc_dir_entry {
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


fn file_inode(f: *const file) -> *mut inode {
unsafe { (*f).f_inode }
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
struct dax_device {
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
fn mark_inode_dirty_sync(inode: *mut inode) {
unsafe { __mark_inode_dirty(inode, 16i32) };
}

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
struct cacheinfo {
    id: u32,
    r#type: u32,
    level: u32,
    coherency_line_size: u32,
    number_of_sets: u32,
    ways_of_associativity: u32,
    physical_line_partition: u32,
    size: u32,
    shared_cpu_map: cpumask,
    attributes: u32,
    fw_token: *mut (),
    disable_sysfs: bool,
    r#priv: *mut (),
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
    __slate_bits_0: __slate_bits::__SlateBits656U0,
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
struct ext4_fc_inode_snap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ext4_h_1077 {
    xattr_sem: rw_semaphore,
    i_ea_iput_node: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ext4_h_1086 {
    i_orphan: list_head,
    i_orphan_idx: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits593U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_locality_group {
}


#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct bvec_iter {
    bi_sector: u64,
    bi_size: u32,
    bi_idx: u32,
    bi_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mb_cache {
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
    __slate_bits_0: __slate_bits::__SlateBits682U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits682U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits682U2,
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
    __slate_bits_3: __slate_bits::__SlateBits682U3,
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
struct proc_ns_operations {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
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
struct compat_robust_list {
    next: u32,
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
    pub struct __SlateBits548U0 {
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
    pub struct __SlateBits580U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits291U0 {
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
    pub struct __SlateBits78U0 {
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
    pub struct __SlateBits78U1 {
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
    pub struct __SlateBits294U0 {
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
    pub struct __SlateBits682U0 {
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
    pub struct __SlateBits679U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits682U1 {
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
    pub struct __SlateBits682U2 {
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
    pub struct __SlateBits682U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits78U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits127U0 {
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
    pub struct __SlateBits135U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits424U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits656U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits593U0 {
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
    pub struct __SlateBits134U0 {
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
    pub struct __SlateBits228U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits770U0 {
        #[bits(1)]
        pub t_synchronous_commit: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits807U0 {
        #[bits(1)]
        pub h_sync: u8,
        #[bits(1)]
        pub h_reserved: u8,
        #[bits(1)]
        pub h_aborted: u8,
        #[bits(1)]
        pub h_invalid: u8,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits679U0,
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


// BLK_STS_AGAIN should only be returned if RQF_NOWAIT is set
// and the bio would block (cf bio_wouldblock_error())
#[repr(C)]
#[derive(Clone, Copy)]
struct bio {
    bi_next: *mut bio,
    bi_bdev: *mut block_device,
    bi_opf: u32,
    bi_flags: u16,
    bi_ioprio: u16,
    bi_write_hint: u8,
    bi_write_stream: u8,
    bi_status: u8,
    bi_bvec_gap_bit: u8,
    __bi_remaining: __SlateRecord_types_h_190,
    bi_io_vec: *mut bio_vec,
    bi_iter: bvec_iter,
    __slate_anon_12: __SlateRecord_blk_types_h_239,
    bi_end_io: Option<unsafe extern "C" fn(*mut bio)>,
    bi_private: *mut (),
    bi_blkg: *mut blkcg_gq,
    issue_time_ns: u64,
    bi_iocost_cost: u64,
    bi_vcnt: u16,
    bi_max_vecs: u16,
    __bi_cnt: __SlateRecord_types_h_190,
    bi_pool: *mut bio_set,
}


fn mapping_gfp_mask(mapping: *const address_space) -> u32 {
unsafe { (*mapping).gfp_mask }
}

fn mapping_gfp_constraint(mapping: *const address_space, gfp_mask: u32) -> u32 {
mapping_gfp_mask(mapping) & gfp_mask
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
struct buffer_head {
    b_state: u64,
    b_this_page: *mut buffer_head,
    __slate_anon_2: __SlateRecord_buffer_head_h_62,
    b_blocknr: u64,
    b_size: u64,
    b_data: *mut u8,
    b_bdev: *mut block_device,
    b_private: *mut (),
    b_assoc_buffers: list_head,
    b_mmb: *mut mapping_metadata_bhs,
    b_count: __SlateRecord_types_h_190,
    b_uptodate_lock: spinlock,
}


// macro tricks to expand the set_buffer_foo(), clear_buffer_foo()
// and buffer_foo() functions.
// To avoid reset buffer flags that are already set, because that causes
// a costly cache line transition, check the flag first.
// test_set_buffer_foo() and test_clear_buffer_foo()
#[inline(always)]
fn buffer_locked(bh: *const buffer_head) -> i32 {
test_bit__2!(2i32, unsafe { core::ptr::addr_of!((*bh).b_state) })
}

#[inline(always)]
fn set_buffer_uptodate(bh: *mut buffer_head) {
// If somebody else already set this uptodate, they will
// have done the memory barrier, and a reader will thus
// see *some* valid buffer state.
//
// Any other serialization (with IO errors or whatever that
// might clear the bit) has to come from other state (eg BH_Lock).
if test_bit__3!(0i32, unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) != 0 {
return;
}
// make it consistent with folio_mark_uptodate
// pairs with smp_load_acquire in buffer_uptodate
smp_mb__before_atomic!();
set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) as *mut u64);
}

fn put_bh(bh: *mut buffer_head) {
smp_mb__before_atomic!();
atomic_dec(unsafe { core::ptr::addr_of_mut!((*bh).b_count) });
}

fn brelse(bh: *mut buffer_head) {
if !bh.is_null() {
unsafe { __brelse(bh) };
}
}

fn __getblk(bdev: *mut block_device, block: u64, size: u32) -> *mut buffer_head {
let mut gfp: u32 = mapping_gfp_constraint((unsafe { (*bdev).bd_mapping }) as *const address_space, !(BIT!(7i32) as u32));
let __v0: u32 = gfp;
gfp = __v0 | ((BIT!(3i32) as u32) | (BIT!(15i32) as u32));
unsafe { bdev_getblk(bdev, block, size, gfp) }
}

fn sb_getblk(sb: *mut super_block, block: u64) -> *mut buffer_head {
__getblk(unsafe { (*sb).s_bdev }, block, (unsafe { (*sb).s_blocksize }) as u32)
}

fn sb_getblk_gfp(sb: *mut super_block, block: u64, gfp: u32) -> *mut buffer_head {
unsafe { bdev_getblk((*sb).s_bdev, block, (*sb).s_blocksize as u32, gfp) }
}

fn trylock_buffer(bh: *mut buffer_head) -> i32 {
likely!(!test_and_set_bit_lock(2i64, (unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) as *mut u64)) as i32
}

fn lock_buffer(bh: *mut buffer_head) {
might_sleep!();
if trylock_buffer(bh) == 0 {
unsafe { __lock_buffer(bh) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct journal_head {
    b_bh: *mut buffer_head,
    b_state_lock: spinlock,
    b_jcount: i32,
    b_jlist: u32,
    b_modified: u32,
    b_frozen_data: *mut u8,
    b_committed_data: *mut u8,
    b_transaction: *mut transaction_s,
    b_next_transaction: *mut transaction_s,
    b_tnext: *mut journal_head,
    b_tprev: *mut journal_head,
    b_cp_transaction: *mut transaction_s,
    b_cpnext: *mut journal_head,
    b_cpprev: *mut journal_head,
    b_triggers: *mut jbd2_buffer_trigger_type,
    b_frozen_triggers: *mut jbd2_buffer_trigger_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_list {
    head: *mut request,
    tail: *mut request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_plug {
    mq_list: rq_list,
    cached_rqs: rq_list,
    cur_ktime: u64,
    nr_ios: u16,
    rq_count: u16,
    multiple_queues: bool,
    has_elevator: bool,
    cb_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mempool {
    lock: spinlock,
    min_nr: i32,
    curr_nr: i32,
    elements: *mut *mut (),
    pool_data: *mut (),
    alloc: Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>,
    free: Option<unsafe extern "C" fn(*mut (), *mut ())>,
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
    head: *mut bio,
    tail: *mut bio,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_set {
    bio_slab: *mut kmem_cache,
    front_pad: u32,
    cache: *mut bio_alloc_cache,
    bio_pool: mempool,
    bvec_pool: mempool,
    back_pad: u32,
    rescue_lock: spinlock,
    rescue_list: bio_list,
    rescue_work: work_struct,
    rescue_workqueue: *mut workqueue_struct,
    cpuhp_dead: hlist_node,
}


// Enough for the string representation of any kind of UUID plus NULL.
// EFI UUID is 36 characters. MSDOS UUID is 11 characters.
#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
    uuid: [u8; 37],
    volname: [u8; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_integrity {
    flags: u8,
    csum_type: u8,
    metadata_size: u8,
    pi_offset: u8,
    interval_exp: u8,
    tag_size: u8,
    pi_tuple_size: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
    major: i32,
    first_minor: i32,
    minors: i32,
    disk_name: [u8; 32],
    events: u16,
    event_flags: u16,
    part_tbl: xarray,
    part0: *mut block_device,
    fops: *const block_device_operations,
    queue: *mut request_queue,
    private_data: *mut (),
    bio_split: bio_set,
    flags: i32,
    state: u64,
    open_mutex: mutex,
    open_partitions: u32,
    bdi: *mut backing_dev_info,
    queue_kobj: kobject,
    slave_dir: *mut kobject,
    slave_bdevs: list_head,
    random: *mut timer_rand_state,
    ev: *mut disk_events,
    cdi: *mut cdrom_device_info,
    node_id: i32,
    bb: *mut badblocks,
    lockdep_map: lockdep_map,
    diskseq: u64,
    open_mode: u32,
    ia_ranges: *mut blk_independent_access_ranges,
    rqos_state_mutex: mutex,
}


// supports a volatile write cache
// supports passing on the FUA bit
// don't modify data until writeback is done
// always completes in submit context
// supports REQ_NOWAIT
// is a zoned device
// do not send FLUSH/FUA commands despite advertising a write cache
#[repr(C)]
#[derive(Clone, Copy)]
struct queue_limits {
    features: u32,
    flags: u32,
    seg_boundary_mask: u64,
    virt_boundary_mask: u64,
    max_hw_sectors: u32,
    max_dev_sectors: u32,
    chunk_sectors: u32,
    max_sectors: u32,
    max_user_sectors: u32,
    max_segment_size: u32,
    max_fast_segment_size: u32,
    physical_block_size: u32,
    logical_block_size: u32,
    alignment_offset: u32,
    io_min: u32,
    io_opt: u32,
    max_discard_sectors: u32,
    max_hw_discard_sectors: u32,
    max_user_discard_sectors: u32,
    max_secure_erase_sectors: u32,
    max_write_zeroes_sectors: u32,
    max_wzeroes_unmap_sectors: u32,
    max_hw_wzeroes_unmap_sectors: u32,
    max_user_wzeroes_unmap_sectors: u32,
    max_hw_zone_append_sectors: u32,
    max_zone_append_sectors: u32,
    discard_granularity: u32,
    discard_alignment: u32,
    zone_write_granularity: u32,
    atomic_write_hw_max: u32,
    atomic_write_max_sectors: u32,
    atomic_write_hw_boundary: u32,
    atomic_write_boundary_sectors: u32,
    atomic_write_hw_unit_min: u32,
    atomic_write_unit_min: u32,
    atomic_write_hw_unit_max: u32,
    atomic_write_unit_max: u32,
    max_segments: u16,
    max_integrity_segments: u16,
    max_discard_segments: u16,
    max_write_streams: u16,
    write_stream_granularity: u32,
    max_open_zones: u32,
    max_active_zones: u32,
    dma_alignment: u32,
    dma_pad_mask: u32,
    integrity: blk_integrity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_independent_access_range {
    kobj: kobject,
    sector: u64,
    nr_sectors: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_independent_access_ranges {
    kobj: kobject,
    sysfs_registered: bool,
    nr_ia_ranges: u32,
    ia_range: [blk_independent_access_range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
    queuedata: *mut (),
    elevator: *mut elevator_queue,
    mq_ops: *const blk_mq_ops,
    queue_ctx: *mut blk_mq_ctx,
    queue_flags: u64,
    rq_timeout: u32,
    queue_depth: u32,
    refs: refcount_struct,
    nr_hw_queues: u32,
    queue_hw_ctx: *mut *mut blk_mq_hw_ctx,
    q_usage_counter: percpu_ref,
    io_lock_cls_key: lock_class_key,
    io_lockdep_map: lockdep_map,
    q_lock_cls_key: lock_class_key,
    q_lockdep_map: lockdep_map,
    last_merge: *mut request,
    queue_lock: spinlock,
    quiesce_depth: i32,
    disk: *mut gendisk,
    mq_kobj: *mut kobject,
    limits: queue_limits,
    dev: *mut device,
    rpm_status: i32,
    pm_only: __SlateRecord_types_h_190,
    stats: *mut blk_queue_stats,
    rq_qos: *mut rq_qos,
    rq_qos_mutex: mutex,
    id: i32,
    nr_requests: u32,
    async_depth: u32,
    timeout: timer_list,
    timeout_work: work_struct,
    nr_active_requests_shared_tags: __SlateRecord_types_h_190,
    sched_shared_tags: *mut blk_mq_tags,
    icq_list: list_head,
    blkcg_pols: [u64; 1],
    root_blkg: *mut blkcg_gq,
    blkg_list: list_head,
    blkcg_mutex: mutex,
    node: i32,
    requeue_lock: spinlock,
    requeue_list: list_head,
    requeue_work: delayed_work,
    blk_trace: *mut blk_trace,
    fq: *mut blk_flush_queue,
    flush_list: list_head,
    elevator_lock: mutex,
    sysfs_lock: mutex,
    limits_lock: mutex,
    unused_hctx_list: list_head,
    unused_hctx_lock: spinlock,
    mq_freeze_depth: i32,
    callback_head: callback_head,
    mq_freeze_wq: wait_queue_head,
    mq_freeze_lock: mutex,
    tag_set: *mut blk_mq_tag_set,
    tag_set_list: list_head,
    debugfs_dir: *mut dentry,
    sched_debugfs_dir: *mut dentry,
    rqos_debugfs_dir: *mut dentry,
    debugfs_mutex: mutex,
}


fn bdev_get_queue(bdev: *mut block_device) -> *mut request_queue {
return unsafe { (*bdev).bd_queue }; // this is never NULL
}

fn bdev_limits(bdev: *mut block_device) -> *mut queue_limits {
unsafe { core::ptr::addr_of_mut!((*bdev_get_queue(bdev)).limits) }
}

fn bdev_write_zeroes_unmap_sectors(bdev: *mut block_device) -> u32 {
unsafe { (*bdev_limits(bdev)).max_wzeroes_unmap_sectors }
}

fn blksize_bits(size: u32) -> u32 {
order_base_2!(size >> 9i32).wrapping_add(9) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct block_device_operations {
    submit_bio: Option<unsafe extern "C" fn(*mut bio)>,
    poll_bio: Option<unsafe extern "C" fn(*mut bio, *mut io_comp_batch, u32) -> i32>,
    open: Option<unsafe extern "C" fn(*mut gendisk, u32) -> i32>,
    release: Option<unsafe extern "C" fn(*mut gendisk)>,
    ioctl: Option<unsafe extern "C" fn(*mut block_device, u32, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut block_device, u32, u32, u64) -> i32>,
    check_events: Option<unsafe extern "C" fn(*mut gendisk, u32) -> u32>,
    unlock_native_capacity: Option<unsafe extern "C" fn(*mut gendisk)>,
    getgeo: Option<unsafe extern "C" fn(*mut gendisk, *mut hd_geometry) -> i32>,
    set_read_only: Option<unsafe extern "C" fn(*mut block_device, bool) -> i32>,
    free_disk: Option<unsafe extern "C" fn(*mut gendisk)>,
    swap_slot_free_notify: Option<unsafe extern "C" fn(*mut block_device, u64)>,
    report_zones: Option<unsafe extern "C" fn(*mut gendisk, u64, u32, *mut blk_report_zones_args) -> i32>,
    devnode: Option<unsafe extern "C" fn(*mut gendisk, *mut u16) -> *mut u8>,
    get_unique_id: Option<unsafe extern "C" fn(*mut gendisk, *mut u8, u32) -> i32>,
    owner: *mut module,
    pr_ops: *const pr_ops,
    alternative_gpt_sector: Option<unsafe extern "C" fn(*mut gendisk, *mut u64) -> i32>,
}


// Largest string for a blockdev identifier
#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
    mark_dead: Option<unsafe extern "C" fn(*mut block_device, bool)>,
    sync: Option<unsafe extern "C" fn(*mut block_device)>,
    freeze: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
    req_list: rq_list,
    need_ts: bool,
    complete: Option<unsafe extern "C" fn(*mut io_comp_batch)>,
    poll_ctx: *mut (),
}


// Maximum contiguous integrity buffer allocation.
#[repr(C)]
#[derive(Clone, Copy)]
struct journal_header_s {
    h_magic: u32,
    h_blocktype: u32,
    h_sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct journal_superblock_s {
    s_header: journal_header_s,
    s_blocksize: u32,
    s_maxlen: u32,
    s_first: u32,
    s_sequence: u32,
    s_start: u32,
    s_errno: u32,
    s_feature_compat: u32,
    s_feature_incompat: u32,
    s_feature_ro_compat: u32,
    s_uuid: [u8; 16],
    s_nr_users: u32,
    s_dynsuper: u32,
    s_max_transaction: u32,
    s_max_trans_data: u32,
    s_checksum_type: u8,
    s_padding2: [u8; 3],
    s_num_fc_blks: u32,
    s_head: u32,
    s_padding: [u32; 40],
    s_checksum: u32,
    s_users: [u8; 768],
}


#[inline(always)]
fn set_buffer_verified(bh: *mut buffer_head) {
if test_bit__3!(25i32, unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) == 0 {
set_bit(25i64, (unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) as *mut u64);
}
}

#[inline(always)]
fn clear_buffer_verified(bh: *mut buffer_head) {
clear_bit(25i64, (unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) as *mut u64);
}

#[inline(always)]
fn buffer_verified(bh: *const buffer_head) -> i32 {
test_bit__2!(25i32, unsafe { core::ptr::addr_of!((*bh).b_state) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct jbd2_inode {
    i_transaction: *mut transaction_s,
    i_next_transaction: *mut transaction_s,
    i_list: list_head,
    i_vfs_inode: *mut inode,
    i_flags: u64,
    i_dirty_start_page: u64,
    i_dirty_end_page: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct jbd2_journal_handle {
    __slate_anon_0: __SlateRecord_jbd2_h_499,
    h_rsv_handle: *mut jbd2_journal_handle,
    h_total_credits: i32,
    h_revoke_credits: i32,
    h_revoke_credits_requested: i32,
    h_ref: i32,
    h_err: i32,
    __slate_bits_0: __slate_bits::__SlateBits807U0,
    h_type: u8,
    h_line_no: u16,
    h_start_jiffies: u64,
    h_requested_credits: u32,
    saved_alloc_context: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct transaction_chp_stats_s {
    cs_chp_time: u64,
    cs_forced_to_close: u32,
    cs_written: u32,
    cs_dropped: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct transaction_s {
    t_journal: *mut journal_s,
    t_tid: u32,
    t_state: u32,
    t_log_start: u64,
    t_nr_buffers: i32,
    t_reserved_list: *mut journal_head,
    t_buffers: *mut journal_head,
    t_forget: *mut journal_head,
    t_checkpoint_list: *mut journal_head,
    t_shadow_list: *mut journal_head,
    t_inode_list: list_head,
    t_max_wait: u64,
    t_start: u64,
    t_requested: u64,
    t_chp_stats: transaction_chp_stats_s,
    t_updates: __SlateRecord_types_h_190,
    t_outstanding_credits: __SlateRecord_types_h_190,
    t_outstanding_revokes: __SlateRecord_types_h_190,
    t_handle_count: __SlateRecord_types_h_190,
    t_cpnext: *mut transaction_s,
    t_cpprev: *mut transaction_s,
    t_expires: u64,
    t_start_time: i64,
    __slate_bits_0: __slate_bits::__SlateBits770U0,
    t_need_data_flush: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct transaction_run_stats_s {
    rs_wait: u64,
    rs_request_delay: u64,
    rs_running: u64,
    rs_locked: u64,
    rs_flushing: u64,
    rs_logging: u64,
    rs_handle_count: u32,
    rs_blocks: u32,
    rs_blocks_logged: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct transaction_stats_s {
    ts_tid: u64,
    ts_requested: u64,
    run: transaction_run_stats_s,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct journal_s {
    j_flags: u64,
    j_errno: i32,
    j_abort_mutex: mutex,
    j_sb_buffer: *mut buffer_head,
    j_superblock: *mut journal_superblock_s,
    j_state_lock: rwlock,
    j_barrier_count: i32,
    j_barrier: mutex,
    j_running_transaction: *mut transaction_s,
    j_committing_transaction: *mut transaction_s,
    j_checkpoint_transactions: *mut transaction_s,
    j_wait_transaction_locked: wait_queue_head,
    j_wait_done_commit: wait_queue_head,
    j_wait_commit: wait_queue_head,
    j_wait_updates: wait_queue_head,
    j_wait_reserved: wait_queue_head,
    j_fc_wait: wait_queue_head,
    j_checkpoint_mutex: mutex,
    j_chkpt_bhs: [*mut buffer_head; 64],
    j_shrinker: *mut shrinker,
    j_checkpoint_jh_count: percpu_counter,
    j_shrink_transaction: *mut transaction_s,
    j_head: u64,
    j_tail: u64,
    j_free: u64,
    j_first: u64,
    j_last: u64,
    j_fc_first: u64,
    j_fc_off: u64,
    j_fc_last: u64,
    j_dev: *mut block_device,
    j_blocksize: i32,
    j_blk_offset: u64,
    j_devname: [u8; 56],
    j_fs_dev: *mut block_device,
    j_fs_dev_wb_err: u32,
    j_total_len: u32,
    j_reserved_credits: __SlateRecord_types_h_190,
    j_list_lock: spinlock,
    j_inode: *mut inode,
    j_tail_sequence: u32,
    j_transaction_sequence: u32,
    j_commit_sequence: u32,
    j_commit_request: u32,
    j_uuid: [u8; 16],
    j_task: *mut task_struct,
    j_max_transaction_buffers: i32,
    j_revoke_records_per_block: i32,
    j_transaction_overhead_buffers: i32,
    j_commit_interval: u64,
    j_commit_timer: timer_list,
    j_revoke_lock: spinlock,
    j_revoke: *mut jbd2_revoke_table_s,
    j_revoke_table: [*mut jbd2_revoke_table_s; 2],
    j_wbuf: *mut *mut buffer_head,
    j_fc_wbuf: *mut *mut buffer_head,
    j_wbufsize: i32,
    j_fc_wbufsize: i32,
    j_last_sync_writer: i32,
    j_average_commit_time: u64,
    j_min_batch_time: u32,
    j_max_batch_time: u32,
    j_commit_callback: Option<unsafe extern "C" fn(*mut journal_s, *mut transaction_s)>,
    j_submit_inode_data_buffers: Option<unsafe extern "C" fn(*mut jbd2_inode) -> i32>,
    j_finish_inode_data_buffers: Option<unsafe extern "C" fn(*mut jbd2_inode) -> i32>,
    j_history_lock: spinlock,
    j_proc_entry: *mut proc_dir_entry,
    j_stats: transaction_stats_s,
    j_failed_commit: u32,
    j_private: *mut (),
    j_csum_seed: u32,
    jbd2_trans_commit_key: lock_class_key,
    j_fc_cleanup_callback: Option<unsafe extern "C" fn(*mut journal_s, i32, u32)>,
    j_fc_replay_callback: Option<unsafe extern "C" fn(*mut journal_s, *mut buffer_head, u32, i32, u32) -> i32>,
    j_bmap: Option<unsafe extern "C" fn(*mut journal_s, *mut u64) -> i32>,
}


// journal feature predicate functions
// Journaling has been aborted for errors.
#[repr(C)]
#[derive(Clone, Copy)]
struct jbd2_buffer_trigger_type {
    t_frozen: Option<unsafe extern "C" fn(*mut jbd2_buffer_trigger_type, *mut buffer_head, *mut (), u64)>,
    t_abort: Option<unsafe extern "C" fn(*mut jbd2_buffer_trigger_type, *mut buffer_head)>,
}


fn is_journal_aborted(journal: *mut journal_s) -> i32 {
((unsafe { (*journal).j_flags }) & 2) as i32
}

fn is_handle_aborted(handle: *mut jbd2_journal_handle) -> i32 {
if ((unsafe { (*handle).__slate_bits_0.__get_h_aborted() }) as i32) != 0 || unsafe { (*handle).__slate_anon_0.h_transaction }.is_null() {
return 1;
}
is_journal_aborted(unsafe { (*(*handle).__slate_anon_0.h_transaction).t_journal })
}

fn dquot_reclaim_block(inode: *mut inode, nr: i64) {
unsafe { dquot_reclaim_space_nodirty(inode, nr << ((*inode).i_blkbits as i32)) };
mark_inode_dirty_sync(inode);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent {
    fe_logical: u64,
    fe_physical: u64,
    fe_length: u64,
    fe_reserved64: [u64; 2],
    fe_flags: u32,
    fe_reserved: [u32; 3],
}


// map extended attribute tree
// request caching of the extents
// flags used in fe_flags:
// Last extent in file.
// Data location unknown.
// Location still pending.
// Sets EXTENT_UNKNOWN.
// Space allocated, but
// no data (i.e. zero).
#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
    fi_flags: u32,
    fi_extents_mapped: u32,
    fi_extents_max: u32,
    fi_extents_start: *mut fiemap_extent,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_batch {
    nr: u8,
    i: u8,
    percpu_pvec_drained: bool,
    folios: [*mut folio; 31],
}


// no blocks allocated, need allocation
// blocks allocated at @addr
// data inline in the inode
#[repr(C)]
#[derive(Clone, Copy)]
struct iomap {
    addr: u64,
    offset: i64,
    length: u64,
    r#type: u16,
    flags: u16,
    bdev: *mut block_device,
    dax_dev: *mut dax_device,
    inline_data: *mut (),
    private: *mut (),
    validity_cookie: u64,
}


// direct I/O
// do not block
#[repr(C)]
#[derive(Clone, Copy)]
struct iomap_ops {
    iomap_begin: Option<unsafe extern "C" fn(*mut inode, i64, i64, u32, *mut iomap, *mut iomap) -> i32>,
    iomap_end: Option<unsafe extern "C" fn(*mut inode, i64, i64, i64, u32, *mut iomap) -> i32>,
    iomap_next: Option<unsafe extern "C" fn(*const iomap_iter, *mut iomap, *mut iomap) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iomap_iter {
    inode: *mut inode,
    pos: i64,
    len: u64,
    iter_start_pos: i64,
    status: i32,
    flags: u32,
    iomap: iomap,
    srcmap: iomap,
    fbatch: *mut folio_batch,
    private: *mut (),
}


fn iomap_length_trim(iter: *const iomap_iter, pos: i64, __UNIQUE_ID_x__603: u64) -> u64 {
let mut end: u64 = ((unsafe { (*iter).iomap.offset }) as u64).wrapping_add(unsafe { (*iter).iomap.length });
if ((unsafe { (*iter).srcmap.r#type }) as i32) != 0 {
let mut __v0: u64;
let __UNIQUE_ID_x__600: u64 = end;
let __UNIQUE_ID_y__601: u64 = ((unsafe { (*iter).srcmap.offset }) as u64).wrapping_add(unsafe { (*iter).srcmap.length });
loop {
if __types_ok!(__UNIQUE_ID_x__600, __UNIQUE_ID_y__601) == 0 {
unsafe { __compiletime_assert_602() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__600 < __UNIQUE_ID_y__601 { __UNIQUE_ID_x__600 } else { __UNIQUE_ID_y__601 };
end = __v0;
}
let mut __v2: u64;
let __UNIQUE_ID_y__604: u64 = end.wrapping_sub(pos as u64);
loop {
if __types_ok!(__UNIQUE_ID_x__603, __UNIQUE_ID_y__604) == 0 {
unsafe { __compiletime_assert_605() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__603 < __UNIQUE_ID_y__604 { __UNIQUE_ID_x__603 } else { __UNIQUE_ID_y__604 };
__v2
}

#[inline(always)]
fn iomap_iter_next(iter: *const iomap_iter, iomap: *mut iomap, srcmap: *mut iomap, begin: Option<unsafe extern "C" fn(*mut inode, i64, i64, u32, *mut iomap, *mut iomap) -> i32>, end: Option<unsafe extern "C" fn(*mut inode, i64, i64, i64, u32, *mut iomap) -> i32>) -> i32 {
let mut ret: i32 = 0;
if (unsafe { (*iomap).length }) != 0 {
if end != None {
// Calculate how far the iter was advanced and the
// original length bytes for end().
let advanced: i64 = unsafe { (*iter).pos }.wrapping_sub(unsafe { (*iter).iter_start_pos });
let len: i64 = iomap_length_trim(iter, unsafe { (*iter).iter_start_pos }, unsafe { (*iter).len }.wrapping_add(advanced as u64)) as i64;
ret = unsafe { end.unwrap()((*iter).inode, (*iter).iter_start_pos, len, advanced, (*iter).flags, iomap) };
}
ret = unsafe { iomap_iter_continue(iter, iomap, srcmap, ret) };
if ret <= 0 {
return ret;
}
}
ret = unsafe { begin.unwrap()((*iter).inode, (*iter).pos, (*iter).len as i64, (*iter).flags, iomap, srcmap) };
if ret < 0 { ret } else { 1 }
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct bgl_lock {
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blockgroup_lock {
    locks: [bgl_lock; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fscrypt_dummy_policy {
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


// data is being allocated
// don't preallocate (for tails)
// goal is meaningful
// blocks already pre-reserved by delayed allocation
// Use blocks from reserved pool
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_allocation_request {
    inode: *mut inode,
    len: u32,
    logical: u32,
    lleft: u32,
    lright: u32,
    goal: u64,
    pleft: u64,
    pright: u64,
    flags: u32,
}


// Logical to physical block mapping, used by ext4_map_blocks()
//
// This structure is used to pass requests into ext4_map_blocks() as
// well as to store the information returned by ext4_map_blocks().  It
// takes less room on the stack than a struct buffer_head.
// This is for use in ext4_map_query_blocks() for a special case where we can
// have a physically and logically contiguous blocks split across two leaf
// nodes instead of a single extent. This is required in case of atomic writes
// to know whether the returned extent is last in leaf. If yes, then lookup for
// next in leaf block in ext4_map_query_blocks_next_in_leaf().
// - This is never going to be added to any buffer head state.
// - We use the next available bit after BH_BITMAP_UPTODATE.
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_map_blocks {
    m_pblk: u64,
    m_lblk: u32,
    m_len: u32,
    m_flags: u32,
    m_seq: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_system_blocks {
    root: rb_root,
    rcu: callback_head,
}


// Flags for ext4_io_end->flags
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_io_end_vec {
    list: list_head,
    offset: i64,
    size: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_io_end {
    list: list_head,
    handle: *mut jbd2_journal_handle,
    inode: *mut inode,
    bio: *mut bio,
    flag: u32,
    count: refcount_struct,
    list_vec: list_head,
}


// Root inode
#[repr(C)]
#[derive(Clone, Copy)]
struct flex_groups {
    free_clusters: __SlateRecord_types_h_197,
    free_inodes: __SlateRecord_types_h_190,
    used_dirs: __SlateRecord_types_h_190,
}


// Besides EXTENT_STATUS_REFERENCED, all these extent type masks
// are exclusive, only one type can be set at a time.
#[repr(C)]
#[derive(Clone, Copy)]
struct extent_status {
    rb_node: rb_node,
    es_lblk: u32,
    es_len: u32,
    es_pblk: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_es_tree {
    root: rb_root,
    cache_es: *mut extent_status,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_es_stats {
    es_stats_shrunk: u64,
    es_stats_cache_hits: percpu_counter,
    es_stats_cache_misses: percpu_counter,
    es_stats_scan_time: u64,
    es_stats_max_scan_time: u64,
    es_stats_all_cnt: percpu_counter,
    es_stats_shk_cnt: percpu_counter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_pending_tree {
    root: rb_root,
}


fn ext4_es_type(es: *mut extent_status) -> u32 {
((unsafe { (*es).es_pblk }) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(5) & 15u64) as u32
}

fn ext4_es_is_unwritten(es: &extent_status) -> i32 {
(ext4_es_type((es as *const extent_status) as *mut extent_status) & 2u32 != 0) as i32
}

#[unsafe(link_section = ".text.slate_distinct.extents.ext4_es_is_delayed")]
extern "C" fn ext4_es_is_delayed(es: *mut extent_status) -> i32 {
(ext4_es_type(es) & 4u32 != 0) as i32
}

fn ext4_es_is_hole(es: &extent_status) -> i32 {
(ext4_es_type((es as *const extent_status) as *mut extent_status) & 8u32 != 0) as i32
}

fn ext4_es_pblock(es: &extent_status) -> u64 {
(unsafe { (*(es as *const extent_status)).es_pblk }) & !(18446744073709551615u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(5))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_stats {
    fc_ineligible_reason_count: [u32; 13],
    fc_num_commits: u64,
    fc_ineligible_commits: u64,
    fc_failed_commits: u64,
    fc_skipped_commits: u64,
    fc_numblks: u64,
    s_fc_avg_commit_time: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_alloc_region {
    lblk: u32,
    pblk: u64,
    ino: i32,
    len: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_replay_state {
    fc_replay_num_tags: i32,
    fc_replay_expected_off: i32,
    fc_current_pass: i32,
    fc_cur_tag: i32,
    fc_crc: i32,
    fc_regions: *mut ext4_fc_alloc_region,
    fc_regions_size: i32,
    fc_regions_used: i32,
    fc_regions_valid: i32,
    fc_modified_inodes: *mut i32,
    fc_modified_inodes_used: i32,
    fc_modified_inodes_size: i32,
}


// Inode flags
// Secure deletion
// Undelete
// Compress file
// Synchronous updates
// Immutable file
// writes to file may only append
// do not dump file
// do not update atime
// Reserved for compression usage...
// One or more compressed clusters
// Don't compress
// nb: was previously EXT2_ECOMPR_FL
// encrypted file
// End compression flags --- maybe not all used
// hash-indexed directory
// AFS directory
// file data should be journaled
// file tail should not be merged
// dirsync behaviour (directories only)
// Top of directory hierarchies
// Set to each huge file
// Inode uses extents
// Verity protected inode
// Inode used for large EA
// Inode has inline data.
// Create with parents projid
// Casefolded directory
// reserved for ext4 lib
// Flags that are appropriate for regular files (all but dir-specific ones).
// Flags that are appropriate for non-directories/regular files.
// Since it's pretty easy to mix up bit numbers and hex values, we use a
// build-time check to make sure that EXT4_XXX_FL is consistent with respect to
// EXT4_INODE_XXX. If all is well, the macros will be dropped, so, it won't cost
// any extra space in the compiled kernel image, otherwise, the build will fail.
// It's important that these values are the same, since we are using
// EXT4_INODE_XXX to test for flag values, but EXT4_XXX_FL must be consistent
// with the values of FS_XXX_FL defined in include/linux/fs.h and the on-disk
// values found in ext2, ext3 and ext4 filesystems, and of course the values
// defined in e2fsprogs.
//
// It's not paranoia if the Murphy's Law really *is* out to get you.  :-)
// Flags used by ext4_map_blocks()
//
// Allocate any needed blocks and/or convert an unwritten
// extent to be an initialized ext4
// Request the creation of an unwritten extent
// Caller is from the delayed allocation writeout path
// finally doing the actual allocation of delayed blocks
// This means that we cannot merge newly allocated extents, and if we
// found an unwritten extent, we need to split it.
// Convert unwritten extent to initialized.
// Eventual metadata allocation (due to growing extent tree)
// should not fail, so try to use reserved blocks for that.
// Don't normalize allocation size (used for fallocate)
// Convert written extents to unwritten
// Write zeros to newly created written extents
// Convert extent to initialized after IO complete
// Atomic write caller needs this to query in the slow path of mixed mapping
// case, when a contiguous extent can be split across two adjacent leaf nodes.
// Look EXT4_MAP_QUERY_LAST_IN_LEAF.
// The bit position of these flags must not overlap with any of the
// EXT4_GET_BLOCKS_*.  They are used by ext4_find_extent(),
// read_extent_tree_block(), ext4_split_extent_at(),
// ext4_ext_insert_extent(), and ext4_ext_create_new_leaf().
// EXT4_EX_NOCACHE is used to indicate that the we shouldn't be
// caching the extents when reading from the extent tree while a
// truncate or punch hole operation is in progress.
// Flags used by ext4_free_blocks
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_inode_info {
    i_data: [u32; 15],
    i_dtime: u32,
    i_file_acl: u64,
    i_block_group: u32,
    i_dir_start_lookup: u32,
    i_flags: u64,
    __slate_anon_6: __SlateRecord_ext4_h_1077,
    __slate_anon_7: __SlateRecord_ext4_h_1086,
    i_fc_dilist: list_head,
    i_fc_list: list_head,
    i_fc_lblk_start: u32,
    i_fc_lblk_len: u32,
    i_fc_snap: *mut ext4_fc_inode_snap,
    i_raw_lock: spinlock,
    i_fc_lock: spinlock,
    i_disksize: i64,
    i_data_sem: rw_semaphore,
    vfs_inode: inode,
    jinode: *mut jbd2_inode,
    i_metadata_bhs: *mut mapping_metadata_bhs,
    i_crtime: timespec64,
    i_prealloc_active: __SlateRecord_types_h_190,
    i_reserved_data_blocks: u32,
    i_prealloc_node: rb_root,
    i_prealloc_lock: rwlock,
    i_es_tree: ext4_es_tree,
    i_es_lock: rwlock,
    i_es_list: list_head,
    i_es_all_nr: u32,
    i_es_shk_nr: u32,
    i_es_shrink_lblk: u32,
    i_es_seq: u64,
    i_last_alloc_group: u32,
    i_pending_tree: ext4_pending_tree,
    i_extra_isize: u16,
    i_inline_off: u16,
    i_inline_size: u16,
    i_reserved_quota: i64,
    i_block_reservation_lock: spinlock,
    i_completed_io_lock: spinlock,
    i_rsv_conversion_list: list_head,
    i_rsv_conversion_work: work_struct,
    i_sync_tid: u32,
    i_datasync_tid: u32,
    i_dquot: [*mut dquot; 3],
    i_csum_seed: u32,
    i_projid: __SlateRecord_projid_h_24,
}


// Mode for data writes:
// Write data to journal
// Some quota option set
// Enable support for dio read nolocking
// Delalloc support
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_super_block {
    s_inodes_count: u32,
    s_blocks_count_lo: u32,
    s_r_blocks_count_lo: u32,
    s_free_blocks_count_lo: u32,
    s_free_inodes_count: u32,
    s_first_data_block: u32,
    s_log_block_size: u32,
    s_log_cluster_size: u32,
    s_blocks_per_group: u32,
    s_clusters_per_group: u32,
    s_inodes_per_group: u32,
    s_mtime: u32,
    s_wtime: u32,
    s_mnt_count: u16,
    s_max_mnt_count: u16,
    s_magic: u16,
    s_state: u16,
    s_errors: u16,
    s_minor_rev_level: u16,
    s_lastcheck: u32,
    s_checkinterval: u32,
    s_creator_os: u32,
    s_rev_level: u32,
    s_def_resuid: u16,
    s_def_resgid: u16,
    s_first_ino: u32,
    s_inode_size: u16,
    s_block_group_nr: u16,
    s_feature_compat: u32,
    s_feature_incompat: u32,
    s_feature_ro_compat: u32,
    s_uuid: [u8; 16],
    s_volume_name: [u8; 16],
    s_last_mounted: [u8; 64],
    s_algorithm_usage_bitmap: u32,
    s_prealloc_blocks: u8,
    s_prealloc_dir_blocks: u8,
    s_reserved_gdt_blocks: u16,
    s_journal_uuid: [u8; 16],
    s_journal_inum: u32,
    s_journal_dev: u32,
    s_last_orphan: u32,
    s_hash_seed: [u32; 4],
    s_def_hash_version: u8,
    s_jnl_backup_type: u8,
    s_desc_size: u16,
    s_default_mount_opts: u32,
    s_first_meta_bg: u32,
    s_mkfs_time: u32,
    s_jnl_blocks: [u32; 17],
    s_blocks_count_hi: u32,
    s_r_blocks_count_hi: u32,
    s_free_blocks_count_hi: u32,
    s_min_extra_isize: u16,
    s_want_extra_isize: u16,
    s_flags: u32,
    s_raid_stride: u16,
    s_mmp_update_interval: u16,
    s_mmp_block: u64,
    s_raid_stripe_width: u32,
    s_log_groups_per_flex: u8,
    s_checksum_type: u8,
    s_encryption_level: u8,
    s_reserved_pad: u8,
    s_kbytes_written: u64,
    s_snapshot_inum: u32,
    s_snapshot_id: u32,
    s_snapshot_r_blocks_count: u64,
    s_snapshot_list: u32,
    s_error_count: u32,
    s_first_error_time: u32,
    s_first_error_ino: u32,
    s_first_error_block: u64,
    s_first_error_func: [u8; 32],
    s_first_error_line: u32,
    s_last_error_time: u32,
    s_last_error_ino: u32,
    s_last_error_line: u32,
    s_last_error_block: u64,
    s_last_error_func: [u8; 32],
    s_mount_opts: [u8; 64],
    s_usr_quota_inum: u32,
    s_grp_quota_inum: u32,
    s_overhead_clusters: u32,
    s_backup_bgs: [u32; 2],
    s_encrypt_algos: [u8; 4],
    s_encrypt_pw_salt: [u8; 16],
    s_lpf_ino: u32,
    s_prj_quota_inum: u32,
    s_checksum_seed: u32,
    s_wtime_hi: u8,
    s_mtime_hi: u8,
    s_mkfs_time_hi: u8,
    s_lastcheck_hi: u8,
    s_first_error_time_hi: u8,
    s_last_error_time_hi: u8,
    s_first_error_errcode: u8,
    s_last_error_errcode: u8,
    s_encoding: u16,
    s_encoding_flags: u16,
    s_orphan_file_inum: u32,
    s_def_resuid_hi: u16,
    s_def_resgid_hi: u16,
    s_reserved: [u32; 93],
    s_checksum: u32,
}


// Number of quota types we support
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_journal_trigger {
    tr_triggers: jbd2_buffer_trigger_type,
    sb: *mut super_block,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_orphan_block {
    ob_free_entries: __SlateRecord_types_h_190,
    ob_bh: *mut buffer_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_orphan_info {
    of_blocks: i32,
    of_csum_seed: u32,
    of_binfo: *mut ext4_orphan_block,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_snap_stats {
    lock_updates_ns_total: __SlateRecord_types_h_197,
    lock_updates_ns_max: __SlateRecord_types_h_197,
    lock_updates_samples: __SlateRecord_types_h_197,
    snap_inodes: __SlateRecord_types_h_197,
    snap_ranges: __SlateRecord_types_h_197,
    snap_fail_es_miss: __SlateRecord_types_h_197,
    snap_fail_es_delayed: __SlateRecord_types_h_197,
    snap_fail_es_other: __SlateRecord_types_h_197,
    snap_fail_inodes_cap: __SlateRecord_types_h_197,
    snap_fail_ranges_cap: __SlateRecord_types_h_197,
    snap_fail_nomem: __SlateRecord_types_h_197,
    snap_fail_inode_loc: __SlateRecord_types_h_197,
    snap_fail_no_snap: __SlateRecord_types_h_197,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct ext4_sb_info {
    s_desc_size: u64,
    s_inodes_per_block: u64,
    s_blocks_per_group: u64,
    s_clusters_per_group: u64,
    s_inodes_per_group: u64,
    s_itb_per_group: u64,
    s_gdb_count: u64,
    s_desc_per_block: u64,
    s_groups_count: u32,
    s_blockfile_groups: u32,
    s_overhead: u64,
    s_cluster_ratio: u32,
    s_cluster_bits: u32,
    s_bitmap_maxbytes: i64,
    s_sbh: *mut buffer_head,
    s_es: *mut ext4_super_block,
    s_group_desc: *mut *mut buffer_head,
    s_mount_opt: u32,
    s_mount_opt2: u32,
    s_mount_flags: u64,
    s_def_mount_opt: u32,
    s_def_mount_opt2: u32,
    s_sb_block: u64,
    s_resv_clusters: __SlateRecord_types_h_197,
    s_resuid: __SlateRecord_uidgid_types_h_9,
    s_resgid: __SlateRecord_uidgid_types_h_13,
    s_mount_state: u16,
    s_pad: u16,
    s_addr_per_block_bits: i32,
    s_desc_per_block_bits: i32,
    s_inode_size: i32,
    s_first_ino: i32,
    s_inode_readahead_blks: u32,
    s_inode_goal: u32,
    s_hash_seed: [u32; 4],
    s_def_hash_version: i32,
    s_hash_unsigned: i32,
    s_freeclusters_counter: percpu_counter,
    s_freeinodes_counter: percpu_counter,
    s_dirs_counter: percpu_counter,
    s_dirtyclusters_counter: percpu_counter,
    s_sra_exceeded_retry_limit: percpu_counter,
    s_blockgroup_lock: *mut blockgroup_lock,
    s_proc: *mut proc_dir_entry,
    s_kobj: kobject,
    s_kobj_unregister: completion,
    s_error_notify_mutex: mutex,
    s_sb: *mut super_block,
    s_mmp_bh: *mut buffer_head,
    s_journal: *mut journal_s,
    s_ext4_flags: u64,
    s_orphan_lock: mutex,
    s_orphan: list_head,
    s_orphan_info: ext4_orphan_info,
    s_commit_interval: u64,
    s_max_batch_time: u32,
    s_min_batch_time: u32,
    s_journal_bdev_file: *mut file,
    s_qf_names: [*mut u8; 3],
    s_jquota_fmt: i32,
    s_want_extra_isize: u32,
    s_system_blks: *mut ext4_system_blocks,
    s_group_info: *mut *mut *mut ext4_group_info,
    s_buddy_cache: *mut inode,
    s_md_lock: spinlock,
    s_mb_offsets: *mut u16,
    s_mb_maxs: *mut u32,
    s_group_info_size: u32,
    s_mb_free_pending: __SlateRecord_types_h_190,
    s_freed_data_list: [list_head; 2],
    s_discard_list: list_head,
    s_discard_work: work_struct,
    s_retry_alloc_pending: __SlateRecord_types_h_190,
    s_mb_avg_fragment_size: *mut xarray,
    s_mb_largest_free_orders: *mut xarray,
    s_stripe: u64,
    s_mb_max_linear_groups: u32,
    s_mb_stream_request: u32,
    s_mb_max_to_scan: u32,
    s_mb_min_to_scan: u32,
    s_mb_stats: u32,
    s_mb_order2_reqs: u32,
    s_mb_group_prealloc: u32,
    s_max_dir_size_kb: u32,
    s_mb_prefetch: u32,
    s_mb_prefetch_limit: u32,
    s_mb_best_avail_max_trim_order: u32,
    s_sb_update_sec: u32,
    s_sb_update_kb: u32,
    s_mb_last_groups: *mut u32,
    s_mb_nr_global_goals: u32,
    s_bal_reqs: __SlateRecord_types_h_190,
    s_bal_success: __SlateRecord_types_h_190,
    s_bal_allocated: __SlateRecord_types_h_190,
    s_bal_ex_scanned: __SlateRecord_types_h_190,
    s_bal_cX_ex_scanned: [__SlateRecord_types_h_190; 5],
    s_bal_groups_scanned: __SlateRecord_types_h_190,
    s_bal_goals: __SlateRecord_types_h_190,
    s_bal_stream_goals: __SlateRecord_types_h_190,
    s_bal_len_goals: __SlateRecord_types_h_190,
    s_bal_breaks: __SlateRecord_types_h_190,
    s_bal_2orders: __SlateRecord_types_h_190,
    s_bal_cX_groups_considered: [__SlateRecord_types_h_197; 5],
    s_bal_cX_hits: [__SlateRecord_types_h_197; 5],
    s_bal_cX_failed: [__SlateRecord_types_h_197; 5],
    s_mb_buddies_generated: __SlateRecord_types_h_190,
    s_mb_generation_time: __SlateRecord_types_h_197,
    s_mb_lost_chunks: __SlateRecord_types_h_190,
    s_mb_preallocated: __SlateRecord_types_h_190,
    s_mb_discarded: __SlateRecord_types_h_190,
    s_lock_busy: __SlateRecord_types_h_190,
    s_locality_groups: *mut ext4_locality_group,
    s_sectors_written_start: u64,
    s_kbytes_written: u64,
    s_extent_max_zeroout_kb: u32,
    s_log_groups_per_flex: u32,
    s_flex_groups: *mut *mut flex_groups,
    s_flex_groups_allocated: u32,
    rsv_conversion_wq: *mut workqueue_struct,
    s_err_report: timer_list,
    s_err_report_sec: u64,
    s_li_request: *mut ext4_li_request,
    s_li_wait_mult: u32,
    s_mmp_tsk: *mut task_struct,
    s_last_trim_minblks: u64,
    s_min_folio_order: u16,
    s_max_folio_order: u16,
    s_csum_seed: u32,
    s_es_shrinker: *mut shrinker,
    s_es_list: list_head,
    s_es_nr_inode: i64,
    s_es_stats: ext4_es_stats,
    s_ea_block_cache: *mut mb_cache,
    s_ea_inode_cache: *mut mb_cache,
    s_ea_inode_to_free: llist_head,
    s_ea_inode_work: delayed_work,
    __slate_pad_136: [u8; 40],
    s_es_lock: spinlock,
    s_journal_triggers: [ext4_journal_trigger; 1],
    s_err_ratelimit_state: ratelimit_state,
    s_warning_ratelimit_state: ratelimit_state,
    s_msg_ratelimit_state: ratelimit_state,
    s_warning_count: __SlateRecord_types_h_190,
    s_msg_count: __SlateRecord_types_h_190,
    s_dummy_enc_policy: fscrypt_dummy_policy,
    s_writepages_rwsem: percpu_rw_semaphore,
    s_daxdev: *mut dax_device,
    s_dax_part_off: u64,
    s_bdev_wb_err: u32,
    s_bdev_wb_lock: spinlock,
    s_error_lock: spinlock,
    s_add_error_count: i32,
    s_first_error_code: i32,
    s_first_error_line: u32,
    s_first_error_ino: u32,
    s_first_error_block: u64,
    s_first_error_func: *const u8,
    s_first_error_time: i64,
    s_last_error_code: i32,
    s_last_error_line: u32,
    s_last_error_ino: u32,
    s_last_error_block: u64,
    s_last_error_func: *const u8,
    s_last_error_time: i64,
    s_sb_upd_work: work_struct,
    s_awu_min: u32,
    s_awu_max: u32,
    s_fc_subtid: __SlateRecord_types_h_190,
    s_fc_q: [list_head; 2],
    s_fc_dentry_q: [list_head; 2],
    s_fc_bytes: u32,
    s_fc_lock: mutex,
    s_fc_bh: *mut buffer_head,
    s_fc_stats: ext4_fc_stats,
    s_fc_snap_stats: ext4_fc_snap_stats,
    s_fc_ineligible_tid: u32,
    s_fc_replay_state: ext4_fc_replay_state,
}


fn EXT4_SB(sb: *mut super_block) -> *mut ext4_sb_info {
(unsafe { (*sb).s_fs_info }) as *mut ext4_sb_info
}

fn EXT4_I(inode: *mut inode) -> *mut ext4_inode_info {
container_of!(inode, ext4_inode_info)
}

fn ext4_test_inode_flag(inode: *mut inode, bit: i32) -> i32 {
test_bit__4!(bit.wrapping_add(0), unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) })
}

fn ext4_test_inode_state(inode: *mut inode, bit: i32) -> i32 {
test_bit__4!(bit.wrapping_add(32), unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) })
}

fn ext4_set_inode_state(inode: *mut inode, bit: i32) {
set_bit(bit.wrapping_add(32) as i64, (unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) }) as *mut u64);
}

// The reason why "FAST_COMMIT" is a compat feature is that, FS becomes
// incompatible only if fast commit blocks are present in the FS. Since we
// clear the journal (and thus the fast commit blocks), we don't mark FS as
// incompatible. We also have a JBD2 incompat feature, which gets set when
// there are fast commit blocks present in the journal.
// Orphan file exists
// METADATA_CSUM also enables group descriptor checksums (GDT_CSUM).  When
// METADATA_CSUM is set, group descriptor checksums use the same algorithm as
// all other data structures' checksums.  However, the METADATA_CSUM and
// GDT_CSUM bits are mutually exclusive.
// Orphan file may be
// non-empty
// Needs recovery
// Journal device
// extents support
// EA in inode
// data in dirent
// >2GB or 3-lvl htree
// data in inode
fn ext4_has_feature_quota(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_ro_compat }) & 256 != 0
}

fn ext4_has_feature_metadata_csum(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_ro_compat }) & 1024 != 0
}

fn ext4_has_feature_extents(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_incompat }) & 64 != 0
}

// Structure of a directory entry
fn ext4_chksum(crc: u32, address: *const (), length: u32) -> u32 {
unsafe { crc32c(crc, address, length as u64) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_iloc {
    bh: *mut buffer_head,
    offset: u64,
    block_group: u32,
}


// htree levels for ext4
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_li_request {
    lr_super: *mut super_block,
    lr_mode: u32,
    lr_first_not_zeroed: u32,
    lr_next_group: u32,
    lr_request: list_head,
    lr_next_sched: u64,
    lr_timeout: u64,
}


fn ext4_get_maxbytes(inode: *mut inode) -> i64 {
if ext4_test_inode_flag(inode, 19i32) != 0 {
return unsafe { (*(*inode).i_sb).s_maxbytes };
}
unsafe { (*EXT4_SB((*inode).i_sb)).s_bitmap_maxbytes }
}

fn ext4_update_i_disksize(inode: *mut inode, newsize: i64) {
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = S_ISREG!(unsafe { (*inode).i_mode }) && inode_is_locked(inode) == 0;
__ret_warn_on = __v1 as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("657: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 657b - ., 3; .popsection\n# {0}", const 657 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_update_i_disksize_2, sym __slate_asm_string_ext4_update_i_disksize_3, const 3595 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("658: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 658b - ., 4; .popsection\n# {0}", const 658 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { down_write(core::ptr::addr_of_mut!((*EXT4_I(inode)).i_data_sem)) };
if newsize > unsafe { (*EXT4_I(inode)).i_disksize } {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_659() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*EXT4_I(inode)).i_disksize, newsize);
if true {
break;
}
}
}
unsafe { up_write(core::ptr::addr_of_mut!((*EXT4_I(inode)).i_data_sem)) };
}

fn ext4_update_inode_size(inode: *mut inode, newsize: i64) -> i32 {
let mut changed: i32 = 0;
if newsize > unsafe { (*inode).i_size } {
i_size_write(inode, newsize);
changed = 1;
}
if newsize > unsafe { (*EXT4_I(inode)).i_disksize } {
ext4_update_i_disksize(inode, newsize);
changed |= 2;
}
changed
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_group_info {
    bb_state: u64,
    bb_free_root: rb_root,
    bb_first_free: i32,
    bb_free: i32,
    bb_fragments: i32,
    bb_avg_fragment_size_order: i32,
    bb_largest_free_order: i32,
    bb_group: u32,
    bb_prealloc_list: list_head,
    alloc_sem: rw_semaphore,
    bb_counters: [i32; 0],
}


fn ext4_quota_capable(sb: *mut super_block) -> bool {
(unsafe { (*EXT4_SB(sb)).s_mount_opt }) & 262144 != 0 || ext4_has_feature_quota(sb)
}

fn ext4_has_inline_data(inode: *mut inode) -> i32 {
let __v0: bool = ext4_test_inode_flag(inode, 28i32) != 0 && (unsafe { (*EXT4_I(inode)).i_inline_off }) != 0;
__v0 as i32
}

// Maximum number of logical blocks in a file; ext4_extent's ee_block is
// __le32.
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_replay_set_iblocks(inode: *mut inode) -> i32 {
let mut __v1: u16 = unsafe { core::mem::zeroed() };
let mut __v0: u16 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__1580: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__1579: u32 = unsafe { core::mem::zeroed() };
let mut map: core::mem::MaybeUninit<ext4_map_blocks> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut ext4_map_blocks = core::ptr::addr_of_mut!(map) as *mut ext4_map_blocks;
let mut cmp2: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u64 = core::ptr::addr_of_mut!(cmp2) as *mut u64;
let mut cmp1: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u64 = core::ptr::addr_of_mut!(cmp1) as *mut u64;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut numblks: i32 = unsafe { core::mem::zeroed() };
let mut end: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u32 = core::ptr::addr_of_mut!(end) as *mut u32;
let mut cur: u32 = unsafe { core::mem::zeroed() };
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut path2: *mut ext4_ext_path = unsafe { core::mem::zeroed() };
let mut path: *mut ext4_ext_path = unsafe { core::mem::zeroed() };
unsafe {
path = core::ptr::null_mut::<ext4_ext_path>();
path2 = core::ptr::null_mut::<ext4_ext_path>();
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
cur = 0;
let __slate_auto_bytes_1 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
numblks = 0;
let __slate_auto_bytes_2 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
ret = 0;
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
// Determin the size of the file first
path = ext4_find_extent(inode, 4294967294u32, core::ptr::null_mut::<ext4_ext_path>(), 1073741824i32);
if IS_ERR(path as *const ()) {
return PTR_ERR(path as *const ()) as i32;
} else {
'__join_0: {
*__slate_slot_22 = (*path.wrapping_offset(((*path).p_depth as i32) as isize)).p_ext;
if !(*__slate_slot_22).is_null() {
*__slate_slot_20 = unsafe { (*(*__slate_slot_22)).ee_block }.wrapping_add(ext4_ext_get_actual_len(*__slate_slot_22) as u32);
// Count the number of data blocks
cur = 0;
while cur < *__slate_slot_20 {
(*__slate_slot_14).m_lblk = cur;
(*__slate_slot_14).m_len = (*__slate_slot_20).wrapping_sub(cur);
ret = ext4_map_blocks(core::ptr::null_mut::<jbd2_journal_handle>(), inode, core::ptr::addr_of_mut!(*__slate_slot_14), 0i32);
if ret < 0 {
break;
} else {
if ret > 0 {
numblks = numblks.wrapping_add(ret);
}
cur = cur.wrapping_add((*__slate_slot_14).m_len);
}
}
// Count the number of extent tree blocks. We do it by looking up
// two successive extents and determining the difference between
// their paths. When path is different for 2 successive extents
// we compare the blocks in the path at each level and increment
// iblocks by total number of differences found.
cur = 0;
ret = skip_hole(inode, &mut (*core::ptr::addr_of_mut!(cur)));
if ret >= 0 {
path = ext4_find_extent(inode, cur, path, 0i32);
if !IS_ERR(path as *const ()) {
numblks = numblks.wrapping_add((*path).p_depth as i32);
'__loop_2: while cur < *__slate_slot_20 {
path = ext4_find_extent(inode, cur, path, 0i32);
if IS_ERR(path as *const ()) {
break;
} else {
*__slate_slot_22 = (*path.wrapping_offset(((*path).p_depth as i32) as isize)).p_ext;
if (*__slate_slot_22).is_null() {
break '__join_0;
} else {
__UNIQUE_ID_x__1579 = cur.wrapping_add(1);
__UNIQUE_ID_y__1580 = unsafe { (*(*__slate_slot_22)).ee_block }.wrapping_add(ext4_ext_get_actual_len(*__slate_slot_22) as u32);
while false {
}
cur = if __UNIQUE_ID_x__1579 > __UNIQUE_ID_y__1580 { __UNIQUE_ID_x__1579 } else { __UNIQUE_ID_y__1580 };
ret = skip_hole(inode, &mut (*core::ptr::addr_of_mut!(cur)));
if ret < 0 {
break;
} else {
path2 = ext4_find_extent(inode, cur, path2, 0i32);
if IS_ERR(path2 as *const ()) {
break;
} else {
*__slate_slot_18 = 0;
loop {
__v0 = (*path).p_depth;
__v1 = (*path2).p_depth;
while false {
}
let __v2: i32 = if (__v0 as i32) > (__v1 as i32) { __v0 as i32 } else { __v1 as i32 };
if *__slate_slot_18 <= __v2 {
*__slate_slot_15 = 0;
*__slate_slot_16 = 0;
if *__slate_slot_18 <= ((*path).p_depth as i32) {
*__slate_slot_16 = if !unsafe { (*path.wrapping_offset(*__slate_slot_18 as isize)).p_bh }.is_null() { (*(*path.wrapping_offset(*__slate_slot_18 as isize)).p_bh).b_blocknr } else { 0 };
}
if *__slate_slot_18 <= ((*path2).p_depth as i32) {
*__slate_slot_15 = if !unsafe { (*path2.wrapping_offset(*__slate_slot_18 as isize)).p_bh }.is_null() { (*(*path2.wrapping_offset(*__slate_slot_18 as isize)).p_bh).b_blocknr } else { 0 };
}
if *__slate_slot_16 != *__slate_slot_15 && *__slate_slot_15 != 0 {
numblks = numblks.wrapping_add(1);
}
let __v4: i32 = *__slate_slot_18;
*__slate_slot_18 = __v4.wrapping_add(1);
} else {
continue '__loop_2;
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
(*inode).i_blocks = (numblks << ((*(*inode).i_sb).s_blocksize_bits as i32).wrapping_sub(9)) as u64;
ext4_mark_inode_dirty__52!(inode);
}
ext4_free_ext_path(path);
ext4_free_ext_path(path2);
return 0;
}
}
}

// Add new method to test whether block and inode bitmaps are properly
// initialized. With uninit_bg reading the block from disk is not enough
// to mark the bitmap uptodate. We need to also zero-out the bitmap
// Ext4 handle operation types -- for logging purposes
fn ext4_handle_valid(handle: *mut jbd2_journal_handle) -> i32 {
(!handle.is_null() && ((unsafe { (*handle).__slate_bits_0.__get_h_invalid() }) as i32) == 0) as i32
}

fn ext4_handle_sync(handle: *mut jbd2_journal_handle) {
if ext4_handle_valid(handle) != 0 {
unsafe {
(*handle).__slate_bits_0.__set_h_sync(1u8);
}
}
}

fn ext4_free_metadata_revoke_credits(sb: *mut super_block, blocks: i32) -> i32 {
// Freeing each metadata block can result in freeing one cluster
(blocks as u32).wrapping_mul(unsafe { (*EXT4_SB(sb)).s_cluster_ratio }) as i32
}

fn ext4_trans_default_revoke_credits(sb: *mut super_block) -> i32 {
ext4_free_metadata_revoke_credits(sb, 8i32)
}

fn __ext4_journal_start(inode: *mut inode, line: u32, r#type: i32, blocks: i32, rsv_blocks: i32, revoke_creds: i32) -> *mut jbd2_journal_handle {
unsafe { __ext4_journal_start_sb(inode, (*inode).i_sb, line, r#type, blocks, rsv_blocks, revoke_creds) }
}

fn ext4_journal_extend(handle: *mut jbd2_journal_handle, nblocks: i32, revoke: i32) -> i32 {
if ext4_handle_valid(handle) != 0 {
return unsafe { jbd2_journal_extend(handle, nblocks, revoke) };
}
0
}

fn ext4_journal_restart(handle: *mut jbd2_journal_handle, nblocks: i32, revoke: i32) -> i32 {
if ext4_handle_valid(handle) != 0 {
return unsafe { jbd2__journal_restart(handle, nblocks, revoke, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32)) };
}
0
}

// Ensure @handle has at least @check_creds credits available. If not,
// transaction will be extended or restarted to contain at least @extend_cred
// credits. Before restarting transaction @fn is executed to allow for cleanup
// before the transaction is restarted.
//
// The return value is < 0 in case of error, 0 in case the handle has enough
// credits or transaction extension succeeded, 1 in case transaction had to be
// restarted.
fn ext4_update_inode_fsync_trans(handle: *mut jbd2_journal_handle, inode: *mut inode, datasync: i32) {
let ei: *mut ext4_inode_info = EXT4_I(inode);
let __v0: bool = ext4_handle_valid(handle) != 0 && is_handle_aborted(handle) == 0;
if __v0 {
unsafe {
(*ei).i_sync_tid = (*(*handle).__slate_anon_0.h_transaction).t_tid;
}
if datasync != 0 {
unsafe {
(*ei).i_datasync_tid = (*(*handle).__slate_anon_0.h_transaction).t_tid;
}
}
}
}

// Ext4 inode journal modes
// journal data mode
// ordered data mode
// writeback data mode
fn ext4_should_journal_data(inode: *mut inode) -> i32 {
(unsafe { ext4_inode_journal_mode(inode) }) & 1
}

fn ext4_free_data_revoke_credits(inode: *mut inode, blocks: i32) -> i32 {
if (unsafe { (*EXT4_SB((*inode).i_sb)).s_mount_opt }) & 3072 == 1024 {
return 0;
}
if ext4_should_journal_data(inode) == 0 {
return 0;
}
// Data blocks in one extent are contiguous, just account for partial
// clusters at extent boundaries
(blocks as u32).wrapping_add(2u32.wrapping_mul(unsafe { (*EXT4_SB((*inode).i_sb)).s_cluster_ratio }.wrapping_sub(1))) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_extent_tail {
    et_checksum: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_extent {
    ee_block: u32,
    ee_len: u16,
    ee_start_hi: u16,
    ee_start_lo: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_extent_idx {
    ei_block: u32,
    ei_leaf_lo: u32,
    ei_leaf_hi: u16,
    ei_unused: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_extent_header {
    eh_magic: u16,
    eh_entries: u16,
    eh_max: u16,
    eh_depth: u16,
    eh_generation: u32,
}


fn find_ext4_extent_tail(eh: *mut ext4_extent_header) -> *mut ext4_extent_tail {
((unsafe { (eh as *const u8).wrapping_offset(EXT4_EXTENT_TAIL_OFFSET!(eh) as isize) }) as *mut ()) as *mut ext4_extent_tail
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_ext_path {
    p_block: u64,
    p_depth: u16,
    p_maxdepth: u16,
    p_ext: *mut ext4_extent,
    p_idx: *mut ext4_extent_idx,
    p_hdr: *mut ext4_extent_header,
    p_bh: *mut buffer_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct partial_cluster {
    pclu: u64,
    lblk: u32,
    state: u32,
}


// EXT_INIT_MAX_LEN is the maximum number of blocks we can have in an
// initialized extent. This is 2^15 and not (2^16 - 1), since we use the
// MSB of ee_len field in the extent datastructure to signify if this
// particular extent is an initialized extent or an unwritten (i.e.
// preallocated).
// EXT_UNWRITTEN_MAX_LEN is the maximum number of blocks we can have in an
// unwritten extent.
// If ee_len is <= 0x8000, it is an initialized extent. Otherwise, it is an
// unwritten one. In other words, if MSB of ee_len is set, it is an
// unwritten extent with only one special scenario when ee_len = 0x8000.
// In this case we can not have an unwritten extent of zero length and
// thus we make it as a special case of initialized extent with 0x8000 length.
// This way we get better extent-to-group alignment for initialized extents.
// Hence, the maximum number of blocks we can have in an *initialized*
// extent is 2^15 (32768) and in an *unwritten* extent is 2^15-1 (32767).
fn ext_inode_hdr(inode: *mut inode) -> *mut ext4_extent_header {
(unsafe { (*EXT4_I(inode)).i_data.as_mut_ptr() as *mut u32 }) as *mut ext4_extent_header
}

fn ext_block_hdr(bh: *mut buffer_head) -> *mut ext4_extent_header {
(unsafe { (*bh).b_data }) as *mut ext4_extent_header
}

fn ext_depth(inode: *mut inode) -> u16 {
unsafe { (*ext_inode_hdr(inode)).eh_depth }
}

fn ext4_ext_mark_unwritten(ext: *mut ext4_extent) {
// We can not have an unwritten extent of zero length!
loop {
if unlikely!((((unsafe { (*ext).ee_len }) as i32) as u64) & 18446744073709518847u64 == 0) != 0 {
loop {
unsafe {
core::arch::asm!("660: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 660b - ., 3; .popsection\n# {0}", const 660 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_mark_unwritten_0, sym __slate_asm_string_ext4_ext_mark_unwritten_1, const 193 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe {
(*ext).ee_len = (((*ext).ee_len as i32) | 32768i32) as u16;
}
}

fn ext4_ext_is_unwritten(ext: *mut ext4_extent) -> i32 {
// Extent with ee_len of 0x8000 is treated as an initialized extent
((((unsafe { (*ext).ee_len }) as i32) as u64) > 32768u64) as i32
}

fn ext4_ext_get_actual_len(ext: *mut ext4_extent) -> i32 {
(if (((unsafe { (*ext).ee_len }) as i32) as u64) <= 32768u64 { ((unsafe { (*ext).ee_len }) as i32) as u64 } else { (((unsafe { (*ext).ee_len }) as i32) as u64).wrapping_sub(32768u64) }) as i32
}

fn ext4_ext_mark_initialized(ext: *mut ext4_extent) {
unsafe {
(*ext).ee_len = ext4_ext_get_actual_len(ext) as u16;
}
}

fn ext4_ext_pblock(ex: *mut ext4_extent) -> u64 {
let mut block: u64 = (unsafe { (*ex).ee_start_lo }) as u64;
block |= ((unsafe { (*ex).ee_start_hi }) as u64) << 31i32 << 1i32;
block
}

fn ext4_idx_pblock(ix: *mut ext4_extent_idx) -> u64 {
let mut block: u64 = (unsafe { (*ix).ei_leaf_lo }) as u64;
block |= ((unsafe { (*ix).ei_leaf_hi }) as u64) << 31i32 << 1i32;
block
}

fn ext4_ext_store_pblock(ex: *mut ext4_extent, pb: u64) {
unsafe {
(*ex).ee_start_lo = (pb & 4294967295) as u32;
}
unsafe {
(*ex).ee_start_hi = (pb >> 31i32 >> 1i32 & 65535) as u16;
}
}

fn ext4_idx_store_pblock(ix: *mut ext4_extent_idx, pb: u64) {
unsafe {
(*ix).ei_leaf_lo = (pb & 4294967295) as u32;
}
unsafe {
(*ix).ei_leaf_hi = (pb >> 31i32 >> 1i32 & 65535) as u16;
}
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
struct xattr_handler {
    name: *const u8,
    prefix: *const u8,
    flags: i32,
    list: Option<unsafe extern "C" fn(*mut dentry) -> bool>,
    get: Option<unsafe extern "C" fn(*const xattr_handler, *mut dentry, *mut inode, *const u8, *mut (), u64) -> i32>,
    set: Option<unsafe extern "C" fn(*const xattr_handler, *mut mnt_idmap, *mut dentry, *mut inode, *const u8, *const (), u64, i32) -> i32>,
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
struct writeback_control {
    nr_to_write: i64,
    pages_skipped: i64,
    range_start: i64,
    range_end: i64,
    sync_mode: u32,
    __slate_bits_0: __slate_bits::__SlateBits548U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
    next: *mut static_call_mod,
    r#mod: *mut module,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
    system: *const u8,
    eval_string: *const u8,
    eval_value: u64,
}


// When a tracepoint is used, it's name is added to the __tracepoint_check
// section. This section is only used at build time to make sure all
// defined tracepoints are used. It is discarded after the build.
// Make sure the alignment of the structure in the __tracepoints section will
// not add unwanted padding between the beginning of the section and the
// structure. Force alignment to the same alignment as the section start.
//
// When lockdep is enabled, we make sure to always test if RCU is
// "watching" regardless if the tracepoint is enabled or not. Tracepoints
// require RCU to be active, and it should always warn at the tracepoint
// site if it is not watching, as it will need to be active when the
// tracepoint is enabled.
fn __do_trace_ext4_fallocate_enter(inode: *mut inode, offset: i64, len: i64, mode: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1035: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1035) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1036: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1036) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1038() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fallocate_enter)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1037: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1037;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, i64, i64, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, i64, i64, i32) -> i32>>(__SCT__tp_func_ext4_fallocate_enter as *const ()) };
unsafe { __v12.unwrap()(__data, inode, offset, len, mode) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fallocate_enter: __SlateAlign16<[u8; 21]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 97u8, 108u8, 108u8, 111u8, 99u8, 97u8, 116u8, 101u8, 95u8, 101u8, 110u8, 116u8, 101u8, 114u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fallocate_enter_1039: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fallocate_enter) }) as *mut ();

fn trace_ext4_fallocate_enter(inode: *mut inode, offset: i64, len: i64, mode: i32) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fallocate_enter.key) }) != 0 {
__do_trace_ext4_fallocate_enter(inode, offset, len, mode);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1040: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_zero_range(inode: *mut inode, offset: i64, len: i64, mode: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1049: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1049) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1050: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1050) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1052() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_zero_range)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1051: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1051;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, i64, i64, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, i64, i64, i32) -> i32>>(__SCT__tp_func_ext4_zero_range as *const ()) };
unsafe { __v12.unwrap()(__data, inode, offset, len, mode) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_zero_range: __SlateAlign16<[u8; 16]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 122u8, 101u8, 114u8, 111u8, 95u8, 114u8, 97u8, 110u8, 103u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_zero_range_1053: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_zero_range) }) as *mut ();

fn trace_ext4_zero_range(inode: *mut inode, offset: i64, len: i64, mode: i32) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_zero_range.key) }) != 0 {
__do_trace_ext4_zero_range(inode, offset, len, mode);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1054: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fallocate_exit(inode: *mut inode, offset: i64, max_blocks: u32, ret: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1056: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1056) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1057: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1057) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1059() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fallocate_exit)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1058: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1058;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, i64, u32, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, i64, u32, i32) -> i32>>(__SCT__tp_func_ext4_fallocate_exit as *const ()) };
unsafe { __v12.unwrap()(__data, inode, offset, max_blocks, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fallocate_exit: __SlateAlign16<[u8; 20]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 97u8, 108u8, 108u8, 111u8, 99u8, 97u8, 116u8, 101u8, 95u8, 101u8, 120u8, 105u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fallocate_exit_1060: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fallocate_exit) }) as *mut ();

fn trace_ext4_fallocate_exit(inode: *mut inode, offset: i64, max_blocks: u32, ret: i32) {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fallocate_exit.key) }) != 0 {
__do_trace_ext4_fallocate_exit(inode, offset, max_blocks, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1061: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_convert_to_initialized_enter(inode: *mut inode, map: *mut ext4_map_blocks, ux: *mut ext4_extent) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1091: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1091) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1092: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1092) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1094() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_convert_to_initialized_enter)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1093: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1093;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, *mut ext4_map_blocks, *mut ext4_extent) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, *mut ext4_map_blocks, *mut ext4_extent) -> i32>>(__SCT__tp_func_ext4_ext_convert_to_initialized_enter as *const ()) };
unsafe { __v12.unwrap()(__data, inode, map, ux) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_convert_to_initialized_enter: __SlateAlign16<[u8; 38]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 99u8, 111u8, 110u8, 118u8, 101u8, 114u8, 116u8, 95u8, 116u8, 111u8, 95u8, 105u8, 110u8, 105u8, 116u8, 105u8, 97u8, 108u8, 105u8, 122u8, 101u8, 100u8, 95u8, 101u8, 110u8, 116u8, 101u8, 114u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_convert_to_initialized_enter_1095: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_convert_to_initialized_enter) }) as *mut ();

fn trace_ext4_ext_convert_to_initialized_enter(inode: *mut inode, map: *mut ext4_map_blocks, ux: *mut ext4_extent) {
if static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_convert_to_initialized_enter.key) }) != 0 {
__do_trace_ext4_ext_convert_to_initialized_enter(inode, map, ux);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1096: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_convert_to_initialized_fastpath(inode: *mut inode, map: *mut ext4_map_blocks, ux: *mut ext4_extent, ix: *mut ext4_extent) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1098: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1098) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1099: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1099) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1101() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_convert_to_initialized_fastpath)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1100: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1100;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, *mut ext4_map_blocks, *mut ext4_extent, *mut ext4_extent) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, *mut ext4_map_blocks, *mut ext4_extent, *mut ext4_extent) -> i32>>(__SCT__tp_func_ext4_ext_convert_to_initialized_fastpath as *const ()) };
unsafe { __v12.unwrap()(__data, inode, map, ux, ix) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_convert_to_initialized_fastpath: __SlateAlign16<[u8; 41]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 99u8, 111u8, 110u8, 118u8, 101u8, 114u8, 116u8, 95u8, 116u8, 111u8, 95u8, 105u8, 110u8, 105u8, 116u8, 105u8, 97u8, 108u8, 105u8, 122u8, 101u8, 100u8, 95u8, 102u8, 97u8, 115u8, 116u8, 112u8, 97u8, 116u8, 104u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_convert_to_initialized_fastpath_1102: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_convert_to_initialized_fastpath) }) as *mut ();

fn trace_ext4_ext_convert_to_initialized_fastpath(inode: *mut inode, map: *mut ext4_map_blocks, ux: *mut ext4_extent, ix: *mut ext4_extent) {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_convert_to_initialized_fastpath.key) }) != 0 {
__do_trace_ext4_ext_convert_to_initialized_fastpath(inode, map, ux, ix);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1103: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_map_blocks_enter(inode: *mut inode, lblk: u32, len: u32, flags: u32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1105: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1105) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1106: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1106) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1108() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_map_blocks_enter)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1107: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1107;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u32, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u32, u32) -> i32>>(__SCT__tp_func_ext4_ext_map_blocks_enter as *const ()) };
unsafe { __v12.unwrap()(__data, inode, lblk, len, flags) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_map_blocks_enter: __SlateAlign16<[u8; 26]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 109u8, 97u8, 112u8, 95u8, 98u8, 108u8, 111u8, 99u8, 107u8, 115u8, 95u8, 101u8, 110u8, 116u8, 101u8, 114u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_map_blocks_enter_1109: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_map_blocks_enter) }) as *mut ();

fn trace_ext4_ext_map_blocks_enter(inode: *mut inode, lblk: u32, len: u32, flags: u32) {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_map_blocks_enter.key) }) != 0 {
__do_trace_ext4_ext_map_blocks_enter(inode, lblk, len, flags);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1110: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_map_blocks_exit(inode: *mut inode, flags: u32, map: *mut ext4_map_blocks, ret: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1119: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1119) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1120: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1120) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1122() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_map_blocks_exit)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1121: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1121;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, u32, *mut ext4_map_blocks, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, u32, *mut ext4_map_blocks, i32) -> i32>>(__SCT__tp_func_ext4_ext_map_blocks_exit as *const ()) };
unsafe { __v12.unwrap()(__data, inode, flags, map, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_map_blocks_exit: __SlateAlign16<[u8; 25]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 109u8, 97u8, 112u8, 95u8, 98u8, 108u8, 111u8, 99u8, 107u8, 115u8, 95u8, 101u8, 120u8, 105u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_map_blocks_exit_1123: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_map_blocks_exit) }) as *mut ();

fn trace_ext4_ext_map_blocks_exit(inode: *mut inode, flags: u32, map: *mut ext4_map_blocks, ret: i32) {
if static_branch_unlikely__11!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_map_blocks_exit.key) }) != 0 {
__do_trace_ext4_ext_map_blocks_exit(inode, flags, map, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1124: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_load_extent(inode: *mut inode, lblk: u32, pblk: u64) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1133: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1133) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1134: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1134) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1136() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_load_extent)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1135: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1135;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u64) -> i32>>(__SCT__tp_func_ext4_ext_load_extent as *const ()) };
unsafe { __v12.unwrap()(__data, inode, lblk, pblk) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_load_extent: __SlateAlign16<[u8; 21]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 108u8, 111u8, 97u8, 100u8, 95u8, 101u8, 120u8, 116u8, 101u8, 110u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_load_extent_1137: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_load_extent) }) as *mut ();

fn trace_ext4_ext_load_extent(inode: *mut inode, lblk: u32, pblk: u64) {
if static_branch_unlikely__8!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_load_extent.key) }) != 0 {
__do_trace_ext4_ext_load_extent(inode, lblk, pblk);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1138: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_handle_unwritten_extents(inode: *mut inode, map: *mut ext4_map_blocks, flags: i32, allocated: u32, newblock: u64) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1182: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1182) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1183: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1183) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1185() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_handle_unwritten_extents)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1184: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1184;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, *mut ext4_map_blocks, i32, u32, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, *mut ext4_map_blocks, i32, u32, u64) -> i32>>(__SCT__tp_func_ext4_ext_handle_unwritten_extents as *const ()) };
unsafe { __v12.unwrap()(__data, inode, map, flags, allocated, newblock) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_handle_unwritten_extents: __SlateAlign16<[u8; 34]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 117u8, 110u8, 119u8, 114u8, 105u8, 116u8, 116u8, 101u8, 110u8, 95u8, 101u8, 120u8, 116u8, 101u8, 110u8, 116u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_handle_unwritten_extents_1186: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_handle_unwritten_extents) }) as *mut ();

fn trace_ext4_ext_handle_unwritten_extents(inode: *mut inode, map: *mut ext4_map_blocks, flags: i32, allocated: u32, newblock: u64) {
if static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_handle_unwritten_extents.key) }) != 0 {
__do_trace_ext4_ext_handle_unwritten_extents(inode, map, flags, allocated, newblock);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1187: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_get_implied_cluster_alloc_exit(sb: *mut super_block, map: *mut ext4_map_blocks, ret: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1189: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1189) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1190: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1190) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1192() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_get_implied_cluster_alloc_exit)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1191: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1191;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut super_block, *mut ext4_map_blocks, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut super_block, *mut ext4_map_blocks, i32) -> i32>>(__SCT__tp_func_ext4_get_implied_cluster_alloc_exit as *const ()) };
unsafe { __v12.unwrap()(__data, sb, map, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_get_implied_cluster_alloc_exit: __SlateAlign16<[u8; 36]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 103u8, 101u8, 116u8, 95u8, 105u8, 109u8, 112u8, 108u8, 105u8, 101u8, 100u8, 95u8, 99u8, 108u8, 117u8, 115u8, 116u8, 101u8, 114u8, 95u8, 97u8, 108u8, 108u8, 111u8, 99u8, 95u8, 101u8, 120u8, 105u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_get_implied_cluster_alloc_exit_1193: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_get_implied_cluster_alloc_exit) }) as *mut ();

fn trace_ext4_get_implied_cluster_alloc_exit(sb: *mut super_block, map: *mut ext4_map_blocks, ret: i32) {
if static_branch_unlikely__12!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_get_implied_cluster_alloc_exit.key) }) != 0 {
__do_trace_ext4_get_implied_cluster_alloc_exit(sb, map, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1194: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_show_extent(inode: *mut inode, lblk: u32, pblk: u64, len: u16) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1196: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1196) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1197: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1197) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1199() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_show_extent)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1198: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1198;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u64, u16) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u64, u16) -> i32>>(__SCT__tp_func_ext4_ext_show_extent as *const ()) };
unsafe { __v12.unwrap()(__data, inode, lblk, pblk, len) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_show_extent: __SlateAlign16<[u8; 21]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 115u8, 104u8, 111u8, 119u8, 95u8, 101u8, 120u8, 116u8, 101u8, 110u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_show_extent_1200: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_show_extent) }) as *mut ();

fn trace_ext4_ext_show_extent(inode: *mut inode, lblk: u32, pblk: u64, len: u16) {
if static_branch_unlikely__10!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_show_extent.key) }) != 0 {
__do_trace_ext4_ext_show_extent(inode, lblk, pblk, len);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1201: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_remove_blocks(inode: *mut inode, ex: *mut ext4_extent, from: u32, to: u64, pc: *mut partial_cluster) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1203: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1203) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1204: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1204) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1206() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_remove_blocks)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1205: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1205;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, *mut ext4_extent, u32, u64, *mut partial_cluster) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, *mut ext4_extent, u32, u64, *mut partial_cluster) -> i32>>(__SCT__tp_func_ext4_remove_blocks as *const ()) };
unsafe { __v12.unwrap()(__data, inode, ex, from, to, pc) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_remove_blocks: __SlateAlign16<[u8; 19]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 114u8, 101u8, 109u8, 111u8, 118u8, 101u8, 95u8, 98u8, 108u8, 111u8, 99u8, 107u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_remove_blocks_1207: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_remove_blocks) }) as *mut ();

fn trace_ext4_remove_blocks(inode: *mut inode, ex: *mut ext4_extent, from: u32, to: u64, pc: *mut partial_cluster) {
if static_branch_unlikely__17!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_remove_blocks.key) }) != 0 {
__do_trace_ext4_remove_blocks(inode, ex, from, to, pc);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1208: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_rm_leaf(inode: *mut inode, start: u32, ex: *mut ext4_extent, pc: *mut partial_cluster) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1210: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1210) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1211: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1211) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1213() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_rm_leaf)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1212: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1212;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, u32, *mut ext4_extent, *mut partial_cluster) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, u32, *mut ext4_extent, *mut partial_cluster) -> i32>>(__SCT__tp_func_ext4_ext_rm_leaf as *const ()) };
unsafe { __v12.unwrap()(__data, inode, start, ex, pc) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_rm_leaf: __SlateAlign16<[u8; 17]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 114u8, 109u8, 95u8, 108u8, 101u8, 97u8, 102u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_rm_leaf_1214: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_rm_leaf) }) as *mut ();

fn trace_ext4_ext_rm_leaf(inode: *mut inode, start: u32, ex: *mut ext4_extent, pc: *mut partial_cluster) {
if static_branch_unlikely__14!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_rm_leaf.key) }) != 0 {
__do_trace_ext4_ext_rm_leaf(inode, start, ex, pc);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1215: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_rm_idx(inode: *mut inode, pblk: u64) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1217: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1217) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1218: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1218) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1220() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_rm_idx)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1219: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1219;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, u64) -> i32>>(__SCT__tp_func_ext4_ext_rm_idx as *const ()) };
unsafe { __v12.unwrap()(__data, inode, pblk) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_rm_idx: __SlateAlign16<[u8; 16]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 114u8, 109u8, 95u8, 105u8, 100u8, 120u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_rm_idx_1221: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_rm_idx) }) as *mut ();

fn trace_ext4_ext_rm_idx(inode: *mut inode, pblk: u64) {
if static_branch_unlikely__15!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_rm_idx.key) }) != 0 {
__do_trace_ext4_ext_rm_idx(inode, pblk);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1222: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_remove_space(inode: *mut inode, start: u32, end: u32, depth: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1224: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1224) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1225: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1225) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1227() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_remove_space)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1226: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1226;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u32, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u32, i32) -> i32>>(__SCT__tp_func_ext4_ext_remove_space as *const ()) };
unsafe { __v12.unwrap()(__data, inode, start, end, depth) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_remove_space: __SlateAlign16<[u8; 22]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 114u8, 101u8, 109u8, 111u8, 118u8, 101u8, 95u8, 115u8, 112u8, 97u8, 99u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_remove_space_1228: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_remove_space) }) as *mut ();

fn trace_ext4_ext_remove_space(inode: *mut inode, start: u32, end: u32, depth: i32) {
if static_branch_unlikely__19!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_remove_space.key) }) != 0 {
__do_trace_ext4_ext_remove_space(inode, start, end, depth);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1229: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_ext_remove_space_done(inode: *mut inode, start: u32, end: u32, depth: i32, pc: *mut partial_cluster, eh_entries: u16) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1231: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1231) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1232: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1232) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1234() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_ext_remove_space_done)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1233: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1233;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u32, i32, *mut partial_cluster, u16) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, u32, u32, i32, *mut partial_cluster, u16) -> i32>>(__SCT__tp_func_ext4_ext_remove_space_done as *const ()) };
unsafe { __v12.unwrap()(__data, inode, start, end, depth, pc, eh_entries) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_ext_remove_space_done: __SlateAlign16<[u8; 27]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 101u8, 120u8, 116u8, 95u8, 114u8, 101u8, 109u8, 111u8, 118u8, 101u8, 95u8, 115u8, 112u8, 97u8, 99u8, 101u8, 95u8, 100u8, 111u8, 110u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_ext_remove_space_done_1235: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_ext_remove_space_done) }) as *mut ();

fn trace_ext4_ext_remove_space_done(inode: *mut inode, start: u32, end: u32, depth: i32, pc: *mut partial_cluster, eh_entries: u16) {
if static_branch_unlikely__18!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_ext_remove_space_done.key) }) != 0 {
__do_trace_ext4_ext_remove_space_done(inode, start, end, depth, pc, eh_entries);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1236: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_collapse_range(inode: *mut inode, offset: i64, len: i64) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1308: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1308) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1309: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1309) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1311() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_collapse_range)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1310: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1310;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, i64, i64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, i64, i64) -> i32>>(__SCT__tp_func_ext4_collapse_range as *const ()) };
unsafe { __v12.unwrap()(__data, inode, offset, len) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_collapse_range: __SlateAlign16<[u8; 20]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 99u8, 111u8, 108u8, 108u8, 97u8, 112u8, 115u8, 101u8, 95u8, 114u8, 97u8, 110u8, 103u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_collapse_range_1312: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_collapse_range) }) as *mut ();

fn trace_ext4_collapse_range(inode: *mut inode, offset: i64, len: i64) {
if static_branch_unlikely__13!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_collapse_range.key) }) != 0 {
__do_trace_ext4_collapse_range(inode, offset, len);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1313: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_insert_range(inode: *mut inode, offset: i64, len: i64) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1315: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1315) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1316: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1316) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1318() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_insert_range)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1317: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1317;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, i64, i64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, i64, i64) -> i32>>(__SCT__tp_func_ext4_insert_range as *const ()) };
unsafe { __v12.unwrap()(__data, inode, offset, len) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_insert_range: __SlateAlign16<[u8; 18]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 105u8, 110u8, 115u8, 101u8, 114u8, 116u8, 95u8, 114u8, 97u8, 110u8, 103u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_insert_range_1319: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_insert_range) }) as *mut ();

fn trace_ext4_insert_range(inode: *mut inode, offset: i64, len: i64) {
if static_branch_unlikely__16!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_insert_range.key) }) != 0 {
__do_trace_ext4_insert_range(inode, offset, len);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1320: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// Extents support for EXT4
//
// TODO:
//   - ext4*_error() should be used in some situations
//   - analyze all BUG()/BUG_ON(), use -EIO where appropriate
//   - smart tree reduction
// used by extent splitting.
// safe to zeroout if split fails 					due to ENOSPC
fn ext4_extent_block_csum(inode: *mut inode, eh: *mut ext4_extent_header) -> u32 {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let csum: u32 = ext4_chksum(unsafe { (*ei).i_csum_seed }, eh as *const (), EXT4_EXTENT_TAIL_OFFSET!(eh) as u32);
csum
}

fn ext4_extent_block_csum_verify(inode: *mut inode, eh: *mut ext4_extent_header) -> i32 {
if !ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb }) {
return 1;
}
let et: *mut ext4_extent_tail = find_ext4_extent_tail(eh);
if (unsafe { (*et).et_checksum }) != ext4_extent_block_csum(inode, eh) {
return 0;
}
1
}

fn ext4_extent_block_csum_set(inode: *mut inode, eh: *mut ext4_extent_header) {
if !ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb }) {
return;
}
let et: *mut ext4_extent_tail = find_ext4_extent_tail(eh);
unsafe {
(*et).et_checksum = ext4_extent_block_csum(inode, eh);
}
}

fn ext4_ext_trunc_restart_fn(inode: *mut inode, dropped: &mut i32) -> i32 {
// Drop i_data_sem to avoid deadlock with ext4_map_blocks.  At this
// moment, get_block can be called only for blocks inside i_size since
// page cache has been already dropped and writes are blocked by
// i_rwsem. So we can safely drop the i_data_sem here.
loop {
if unlikely!(EXT4_JOURNAL!(inode).is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("1510: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1510b - ., 3; .popsection\n# {0}", const 1510 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_trunc_restart_fn_0, sym __slate_asm_string_ext4_ext_trunc_restart_fn_1, const 98 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe { ext4_discard_preallocations(inode) };
unsafe { up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem)) };
unsafe {
*(dropped as *mut i32) = 1;
}
0
}

fn ext4_ext_path_brelse(path: *mut ext4_ext_path) {
brelse(unsafe { (*path).p_bh });
unsafe {
(*path).p_bh = core::ptr::null_mut::<buffer_head>();
}
}

fn ext4_ext_drop_refs(mut path: *mut ext4_ext_path) {
let mut depth: i32 = 0;
let mut i: i32 = 0;
if IS_ERR_OR_NULL(path as *const ()) {
return;
}
depth = (unsafe { (*path).p_depth }) as i32;
while i <= depth {
ext4_ext_path_brelse(path);
i = i.wrapping_add(1);
let __v3: *mut ext4_ext_path = path;
path = __v3.wrapping_offset(1 as isize);
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_free_ext_path(path: *mut ext4_ext_path) {
if IS_ERR_OR_NULL(path as *const ()) {
return;
}
ext4_ext_drop_refs(path);
unsafe { kfree(path as *const ()) };
}

/// Make sure 'handle' has at least 'check_cred' credits. If not, restart
/// transaction with 'restart_cred' credits. The function drops i_data_sem
/// when restarting transaction and gets it after transaction is restarted.
///
/// The function returns 0 on success, 1 if transaction had to be restarted,
/// and < 0 in case of fatal error.
#[unsafe(no_mangle)]
extern "C" fn ext4_datasem_ensure_credits(handle: *mut jbd2_journal_handle, inode: *mut inode, check_cred: i32, restart_cred: i32, revoke_cred: i32) -> i32 {
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut dropped: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
dropped = 0;
err = __ext4_journal_ensure_credits(handle, check_cred, restart_cred, revoke_cred);
if err > 0 {
err = ext4_ext_trunc_restart_fn(inode, &mut (*core::ptr::addr_of_mut!(dropped)));
if err >= 0 {
err = ext4_journal_restart(handle, restart_cred, revoke_cred);
if err == 0 {
err = 1;
}
}
}
*__slate_slot_3 = err;
if dropped != 0 {
down_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
}
return *__slate_slot_3;
}
}

/// could return:
///  - EROFS
///  - ENOMEM
fn ext4_ext_get_access(handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path) -> i32 {
let mut err: i32 = 0;
if !unsafe { (*path).p_bh }.is_null() {
// path points to block
BUFFER_TRACE!();
err = ext4_journal_get_write_access__51!(handle, unsafe { (*inode).i_sb }, unsafe { (*path).p_bh }, 1i32);
// The extent buffer's verified bit will be set again in
// __ext4_ext_dirty(). We could leave an inconsistent
// buffer if the extents updating procudure break off du
// to some error happens, force to check it again.
if err == 0 {
clear_buffer_verified(unsafe { (*path).p_bh });
}
}
// path points to leaf/index in inode body
//
// we use in-core data, no need to protect them
err
}

/// could return:
///  - EROFS
///  - ENOMEM
///  - EIO
#[unsafe(no_mangle)]
extern "C" fn __ext4_ext_dirty(r#where: *const u8, line: u32, handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path) -> i32 {
let mut err: i32 = 0;
while false {
}
let mut __v1: i64;
let __ret_warn_on: i32 = (rwsem_is_locked(unsafe { core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem) }) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1511: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1511b - ., 3; .popsection\n# {0}", const 1511 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___ext4_ext_dirty_3, sym __slate_asm_string___ext4_ext_dirty_4, const 196 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1512: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1512b - ., 4; .popsection\n# {0}", const 1512 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if !unsafe { (*path).p_bh }.is_null() {
ext4_extent_block_csum_set(inode, ext_block_hdr(unsafe { (*path).p_bh }));
// path points to block
err = unsafe { __ext4_handle_dirty_metadata(r#where, line, handle, inode, (*path).p_bh) };
// Extents updating done, re-set verified flag
if err == 0 {
set_buffer_verified(unsafe { (*path).p_bh });
}
} else {
// path points to leaf/index in inode body
err = ext4_mark_inode_dirty__37!(handle, inode);
}
err
}

fn ext4_ext_find_goal(inode: *mut inode, path: *mut ext4_ext_path, block: u32) -> u64 {
if !path.is_null() {
let depth: i32 = (unsafe { (*path).p_depth }) as i32;
// Try to predict block placement assuming that we are
// filling in a file which will eventually be
// non-sparse --- i.e., in the case of libbfd writing
// an ELF object sections out-of-order but in a way
// the eventually results in a contiguous object or
// executable file, or some database extending a table
// space file.  However, this is actually somewhat
// non-ideal if we are writing a sparse file such as
// qemu or KVM writing a raw image file that is going
// to stay fairly sparse, since it will end up
// fragmenting the file system's free space.  Maybe we
// should have some hueristics or some way to allow
// userspace to pass a hint to file system,
// especially if the latter case turns out to be
// common.
let ex: *mut ext4_extent = unsafe { (*path.wrapping_offset(depth as isize)).p_ext };
if !ex.is_null() {
let ext_pblk: u64 = ext4_ext_pblock(ex);
let ext_block: u32 = unsafe { (*ex).ee_block };
if block > ext_block {
return ext_pblk.wrapping_add(block.wrapping_sub(ext_block) as u64);
} else {
return ext_pblk.wrapping_sub(ext_block.wrapping_sub(block) as u64);
}
}
// it looks like index is empty;
// try to find starting block from index itself
if !unsafe { (*path.wrapping_offset(depth as isize)).p_bh }.is_null() {
return unsafe { (*(*path.wrapping_offset(depth as isize)).p_bh).b_blocknr };
}
}
// OK. use inode's group
unsafe { ext4_inode_to_goal_block(inode) }
}

/// Allocation for a meta data block
fn ext4_ext_new_meta_block(handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path, ex: *mut ext4_extent, err: *mut i32, flags: u32) -> u64 {
let goal: u64 = ext4_ext_find_goal(inode, path, unsafe { (*ex).ee_block });
let newblock: u64 = unsafe { ext4_new_meta_blocks(handle, inode, goal, flags, core::ptr::null_mut::<u64>(), err) };
newblock
}

fn ext4_ext_space_block(inode: *mut inode, check: i32) -> i32 {
let size: i32 = (unsafe { (*(*inode).i_sb).s_blocksize }.wrapping_sub(core::mem::size_of::<ext4_extent_header>() as u64) / (core::mem::size_of::<ext4_extent>() as u64)) as i32;
size
}

fn ext4_ext_space_block_idx(inode: *mut inode, check: i32) -> i32 {
let size: i32 = (unsafe { (*(*inode).i_sb).s_blocksize }.wrapping_sub(core::mem::size_of::<ext4_extent_header>() as u64) / (core::mem::size_of::<ext4_extent_idx>() as u64)) as i32;
size
}

fn ext4_ext_space_root(inode: *mut inode, check: i32) -> i32 {
let mut size: i32 = core::mem::size_of::<[u32; 15]>() as i32;
size = (size as u64).wrapping_sub(core::mem::size_of::<ext4_extent_header>() as u64) as i32;
size = ((size as u64) / (core::mem::size_of::<ext4_extent>() as u64)) as i32;
size
}

fn ext4_ext_space_root_idx(inode: *mut inode, check: i32) -> i32 {
let mut size: i32 = core::mem::size_of::<[u32; 15]>() as i32;
size = (size as u64).wrapping_sub(core::mem::size_of::<ext4_extent_header>() as u64) as i32;
size = ((size as u64) / (core::mem::size_of::<ext4_extent_idx>() as u64)) as i32;
size
}

fn ext4_force_split_extent_at(handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path, lblk: u32, nofail: i32) -> *mut ext4_ext_path {
let mut flags: i32 = 1073741824 | 8;
if nofail != 0 {
flags |= 32 | 268435456;
}
ext4_split_extent_at(handle, inode, path, lblk, flags)
}

fn ext4_ext_max_entries(inode: *mut inode, depth: i32) -> i32 {
let mut max: i32 = 0;
if depth == (ext_depth(inode) as i32) {
if depth == 0 {
max = ext4_ext_space_root(inode, 1i32);
} else {
max = ext4_ext_space_root_idx(inode, 1i32);
}
} else {
if depth == 0 {
max = ext4_ext_space_block(inode, 1i32);
} else {
max = ext4_ext_space_block_idx(inode, 1i32);
}
}
max
}

fn ext4_valid_extent(inode: *mut inode, ext: *mut ext4_extent) -> i32 {
let block: u64 = ext4_ext_pblock(ext);
let len: i32 = ext4_ext_get_actual_len(ext);
let lblock: u32 = unsafe { (*ext).ee_block };
// We allow neither:
//  - zero length
//  - overflow/wrap-around
if lblock.wrapping_add(len as u32) <= lblock {
return 0;
}
unsafe { ext4_inode_block_valid(inode, block, len as u32) }
}

fn ext4_valid_extent_idx(inode: *mut inode, ext_idx: *mut ext4_extent_idx) -> i32 {
let block: u64 = ext4_idx_pblock(ext_idx);
unsafe { ext4_inode_block_valid(inode, block, 1u32) }
}

fn ext4_valid_extent_entries(inode: *mut inode, eh: *mut ext4_extent_header, lblk: u32, pblk: &mut u64, depth: i32) -> i32 {
let mut entries: u16 = 0;
let mut lblock: u32 = 0;
let mut cur: u32 = 0;
if ((unsafe { (*eh).eh_entries }) as i32) == 0 {
return 1;
}
entries = unsafe { (*eh).eh_entries };
if depth == 0 {
// leaf entries
let mut ext: *mut ext4_extent = EXT_FIRST_EXTENT!(eh);
// The logical block in the first entry should equal to
// the number in the index block.
if depth != (ext_depth(inode) as i32) && lblk != unsafe { (*ext).ee_block } {
return 0;
}
while entries != 0 {
if ext4_valid_extent(inode, ext) == 0 {
return 0;
}
// Check for overlapping extents
lblock = unsafe { (*ext).ee_block };
if lblock < cur {
unsafe {
*(pblk as *mut u64) = ext4_ext_pblock(ext);
}
return 0;
}
cur = lblock.wrapping_add(ext4_ext_get_actual_len(ext) as u32);
let __v0: *mut ext4_extent = ext;
ext = __v0.wrapping_offset(1 as isize);
entries = (entries as i32).wrapping_sub(1) as u16;
}
} else {
let mut ext_idx: *mut ext4_extent_idx = EXT_FIRST_INDEX!(eh);
// The logical block in the first entry should equal to
// the number in the parent index block.
if depth != (ext_depth(inode) as i32) && lblk != unsafe { (*ext_idx).ei_block } {
return 0;
}
while entries != 0 {
if ext4_valid_extent_idx(inode, ext_idx) == 0 {
return 0;
}
// Check for overlapping index extents
lblock = unsafe { (*ext_idx).ei_block };
if lblock < cur {
unsafe {
*(pblk as *mut u64) = ext4_idx_pblock(ext_idx);
}
return 0;
}
let __v5: *mut ext4_extent_idx = ext_idx;
ext_idx = __v5.wrapping_offset(1 as isize);
entries = (entries as i32).wrapping_sub(1) as u16;
cur = lblock.wrapping_add(1);
}
}
1
}

fn __ext4_ext_check(function: *const u8, line: u32, inode: *mut inode, eh: *mut ext4_extent_header, depth: i32, mut pblk: u64, lblk: u32) -> i32 {
// Verify checksum on non-root extent tree nodes
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut max: i32 = unsafe { core::mem::zeroed() };
let mut error_msg: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *const u8 = core::ptr::addr_of_mut!(error_msg) as *mut *const u8;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
max = 0;
err = 117i32.wrapping_neg();
if unlikely!(((*eh).eh_magic as i32) != 62218) != 0 {
*__slate_slot_3 = c"invalid magic".as_ptr() as *const u8;
} else {
if unlikely!(((*eh).eh_depth as i32) != depth) != 0 {
*__slate_slot_3 = c"unexpected eh_depth".as_ptr() as *const u8;
} else {
if unlikely!(((*eh).eh_max as i32) == 0) != 0 {
*__slate_slot_3 = c"invalid eh_max".as_ptr() as *const u8;
} else {
max = ext4_ext_max_entries(inode, depth);
if unlikely!(((*eh).eh_max as i32) > max) != 0 {
*__slate_slot_3 = c"too large eh_max".as_ptr() as *const u8;
} else {
if unlikely!(((*eh).eh_entries as i32) > ((*eh).eh_max as i32)) != 0 {
*__slate_slot_3 = c"invalid eh_entries".as_ptr() as *const u8;
} else {
if unlikely!(((*eh).eh_entries as i32) == 0 && depth > 0) != 0 {
*__slate_slot_3 = c"eh_entries is 0 but eh_depth is > 0".as_ptr() as *const u8;
} else {
if ext4_valid_extent_entries(inode, eh, lblk, &mut (*core::ptr::addr_of_mut!(pblk)), depth) == 0 {
*__slate_slot_3 = c"invalid extent entries".as_ptr() as *const u8;
} else {
if unlikely!(depth > 32) != 0 {
*__slate_slot_3 = c"too large eh_depth".as_ptr() as *const u8;
} else {
if (ext_depth(inode) as i32) != depth {
__v0 = ext4_extent_block_csum_verify(inode, eh) == 0;
} else {
__v0 = false;
}
if __v0 {
*__slate_slot_3 = c"extent tree corrupted".as_ptr() as *const u8;
err = 74i32.wrapping_neg();
} else {
return 0;
}
}
}
}
}
}
}
}
}
__ext4_error_inode(inode, function, line, 0u64, err.wrapping_neg(), c"pblk %llu bad header/extent: %s - magic %x, entries %u, max %u(%u), depth %u(%u)".as_ptr() as *const u8, pblk, *__slate_slot_3, (*eh).eh_magic as i32, (*eh).eh_entries as i32, (*eh).eh_max as i32, max, (*eh).eh_depth as i32, depth);
return err;
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_ext_check_inode(inode: *mut inode) -> i32 {
ext4_ext_check!(inode, ext_inode_hdr(inode), ext_depth(inode), 0i32)
}

fn ext4_cache_extents(inode: *mut inode, eh: *mut ext4_extent_header) {
let mut ex: *mut ext4_extent = EXT_FIRST_EXTENT!(eh);
let mut prev: u32 = 0;
let mut i: i32 = 0;
while false {
}
i = (unsafe { (*eh).eh_entries }) as i32;
while i > 0 {
let mut status: u32 = 1u32;
let lblk: u32 = unsafe { (*ex).ee_block };
let len: i32 = ext4_ext_get_actual_len(ex);
if prev != 0 && prev != lblk {
unsafe { ext4_es_cache_extent(inode, prev, lblk.wrapping_sub(prev), 18446744073709551615u64, 8u32) };
}
if ext4_ext_is_unwritten(ex) != 0 {
status = 2u32;
}
unsafe { ext4_es_cache_extent(inode, lblk, len as u32, ext4_ext_pblock(ex), status) };
prev = lblk.wrapping_add(len as u32);
i = i.wrapping_sub(1);
let __v4: *mut ext4_extent = ex;
ex = __v4.wrapping_offset(1 as isize);
}
}

fn __read_extent_tree_block(function: *const u8, line: u32, inode: *mut inode, idx: *mut ext4_extent_idx, depth: i32, flags: i32) -> *mut buffer_head {
let mut pblk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(pblk) as *mut u64;
let mut gfp_flags: u32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
gfp_flags = (BIT!(3i32) as u32) | (((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32));
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if flags & 268435456 != 0 {
let __v2: u32 = gfp_flags;
gfp_flags = __v2 | (BIT!(15i32) as u32);
}
*__slate_slot_3 = ext4_idx_pblock(idx);
*__slate_slot_6 = sb_getblk_gfp((*inode).i_sb, *__slate_slot_3, gfp_flags);
if unlikely!((*__slate_slot_6).is_null()) != 0 {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut buffer_head;
} else {
'__join_0: {
if bh_uptodate_or_lock(*__slate_slot_6) == 0 {
trace_ext4_ext_load_extent(inode, *__slate_slot_3 as u32, __slate_intrinsic_ded315013ea6523(const { ((0 as i32) as u32) as i32 }) as u64);
*__slate_slot_5 = ext4_read_bh(*__slate_slot_6, 0u32, None, false);
if *__slate_slot_5 < 0 {
break '__join_0;
}
}
if buffer_verified(*__slate_slot_6 as *const buffer_head) != 0 && flags & 536870912 == 0 {
return *__slate_slot_6;
} else {
*__slate_slot_5 = __ext4_ext_check(function, line, inode, ext_block_hdr(*__slate_slot_6), depth, *__slate_slot_3, (*idx).ei_block);
if *__slate_slot_5 == 0 {
set_buffer_verified(*__slate_slot_6);
// If this is a leaf block, cache all of its entries
if flags & 1073741824 == 0 && depth == 0 {
let eh: *mut ext4_extent_header = ext_block_hdr(*__slate_slot_6);
ext4_cache_extents(inode, eh);
}
return *__slate_slot_6;
}
}
}
put_bh(*__slate_slot_6);
return ERR_PTR(*__slate_slot_5 as i64) as *mut buffer_head;
}
}
}

/// This function is called to cache a file's extent information in the
/// extent status tree
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_precache(inode: *mut inode) -> i32 {
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32;
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut path: *mut ext4_ext_path = unsafe { core::mem::zeroed() };
let mut ei: *mut ext4_inode_info = unsafe { core::mem::zeroed() };
unsafe {
ei = EXT4_I!(inode);
path = core::ptr::null_mut::<ext4_ext_path>();
let __slate_auto_bytes_2 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
i = 0;
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
ret = 0;
if ext4_test_inode_flag(inode, 19i32) == 0 {
return 0;
} else {
// not an extent-mapped inode
ext4_check_map_extents_env(inode);
down_read(core::ptr::addr_of_mut!((*ei).i_data_sem));
*__slate_slot_14 = ext_depth(inode) as i32;
// Don't cache anything if there are no external extent blocks
if *__slate_slot_14 == 0 {
up_read(core::ptr::addr_of_mut!((*ei).i_data_sem));
return ret;
} else {
__obj_size = size_mul(core::mem::size_of::<ext4_ext_path>() as u64, (*__slate_slot_14).wrapping_add(1) as u64);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_7 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_8 = _kzalloc_noprof(__obj_size, 3136u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
alloc_tag_restore!();
} else {
*__slate_slot_8 = _kzalloc_noprof(__obj_size, 3136u32, *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
}
path = *__slate_slot_8 as *mut ext4_ext_path;
if path.is_null() {
up_read(core::ptr::addr_of_mut!((*ei).i_data_sem));
return 12i32.wrapping_neg();
} else {
(*path.wrapping_offset(0 as isize)).p_hdr = ext_inode_hdr(inode);
ret = ext4_ext_check__3!(inode, unsafe { (*path.wrapping_offset(0 as isize)).p_hdr }, *__slate_slot_14, 0i32);
if ret == 0 {
(*path.wrapping_offset(0 as isize)).p_idx = EXT_FIRST_INDEX!(unsafe { (*path.wrapping_offset(0 as isize)).p_hdr });
'__join_1: {
loop {
if i >= 0 {
// If this is a leaf block or we've reached the end of
// the index block, go up
if i == *__slate_slot_14 || (*path.wrapping_offset(i as isize)).p_idx > EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(i as isize)).p_hdr }) {
ext4_ext_path_brelse(path.wrapping_offset(i as isize));
i = i.wrapping_sub(1);
} else {
let __v4: *mut ext4_ext_path = path.wrapping_offset(i as isize);
let __v3: *mut ext4_extent_idx = (*__v4).p_idx;
(*__v4).p_idx = __v3.wrapping_offset(1 as isize);
*__slate_slot_16 = __read_extent_tree_block(c"ext4_ext_precache".as_ptr() as *const u8, 648u32, inode, __v3, (*__slate_slot_14).wrapping_sub(i).wrapping_sub(1), 536870912i32);
if IS_ERR(*__slate_slot_16 as *const ()) {
break;
} else {
i = i.wrapping_add(1);
(*path.wrapping_offset(i as isize)).p_bh = *__slate_slot_16;
(*path.wrapping_offset(i as isize)).p_hdr = ext_block_hdr(*__slate_slot_16);
(*path.wrapping_offset(i as isize)).p_idx = EXT_FIRST_INDEX!(unsafe { (*path.wrapping_offset(i as isize)).p_hdr });
}
}
} else {
break '__join_1;
}
}
ret = PTR_ERR(*__slate_slot_16 as *const ()) as i32;
}
ext4_set_inode_state(inode, 7i32);
}
up_read(core::ptr::addr_of_mut!((*ei).i_data_sem));
ext4_free_ext_path(path);
return ret;
}
}
}
}
}

/// ext4_ext_binsearch_idx:
/// binary search for the closest index of the given block
/// the header must be checked before calling this
fn ext4_ext_binsearch_idx(inode: *mut inode, path: *mut ext4_ext_path, block: u32) {
let eh: *mut ext4_extent_header = unsafe { (*path).p_hdr };
let mut r: *mut ext4_extent_idx = unsafe { core::mem::zeroed() };
let mut l: *mut ext4_extent_idx = unsafe { core::mem::zeroed() };
let mut __v0: i32;
if false {
unsafe { _printk(c"binsearch for %u(idx):  ".as_ptr() as *const u8, block) };
}
__v0 = 0;
l = EXT_FIRST_INDEX!(eh).wrapping_offset(1 as isize);
r = EXT_LAST_INDEX!(eh);
while l <= r {
let m: *mut ext4_extent_idx = l.wrapping_offset((((unsafe { r.offset_from(l as *mut ext4_extent_idx) }) as i64) / 2) as isize);
let mut __v1: i32;
if false {
unsafe { _printk(c"%p(%u):%p(%u):%p(%u) ".as_ptr() as *const u8, l, (*l).ei_block, m, (*m).ei_block, r, (*r).ei_block) };
}
__v1 = 0;
if block < unsafe { (*m).ei_block } {
r = m.wrapping_offset(-(1 as isize));
} else {
l = m.wrapping_offset(1 as isize);
}
}
unsafe {
(*path).p_idx = l.wrapping_offset(-(1 as isize));
}
let mut __v3: i32;
if false {
unsafe { _printk(c"  -> %u->%lld ".as_ptr() as *const u8, (*(*path).p_idx).ei_block, ext4_idx_pblock((*path).p_idx)) };
}
__v3 = 0;
}

/// ext4_ext_binsearch:
/// binary search for closest extent of the given block
/// the header must be checked before calling this
fn ext4_ext_binsearch(inode: *mut inode, path: *mut ext4_ext_path, block: u32) {
let eh: *mut ext4_extent_header = unsafe { (*path).p_hdr };
let mut r: *mut ext4_extent = unsafe { core::mem::zeroed() };
let mut l: *mut ext4_extent = unsafe { core::mem::zeroed() };
if ((unsafe { (*eh).eh_entries }) as i32) == 0 {
// this leaf is empty:
// we get such a leaf in split/add case
return;
}
let mut __v0: i32;
if false {
unsafe { _printk(c"binsearch for %u:  ".as_ptr() as *const u8, block) };
}
__v0 = 0;
l = EXT_FIRST_EXTENT!(eh).wrapping_offset(1 as isize);
r = EXT_LAST_EXTENT!(eh);
while l <= r {
let m: *mut ext4_extent = l.wrapping_offset((((unsafe { r.offset_from(l as *mut ext4_extent) }) as i64) / 2) as isize);
let mut __v1: i32;
if false {
unsafe { _printk(c"%p(%u):%p(%u):%p(%u) ".as_ptr() as *const u8, l, (*l).ee_block, m, (*m).ee_block, r, (*r).ee_block) };
}
__v1 = 0;
if block < unsafe { (*m).ee_block } {
r = m.wrapping_offset(-(1 as isize));
} else {
l = m.wrapping_offset(1 as isize);
}
}
unsafe {
(*path).p_ext = l.wrapping_offset(-(1 as isize));
}
let mut __v3: i32;
if false {
unsafe { _printk(c"  -> %d:%llu:[%d]%d ".as_ptr() as *const u8, (*(*path).p_ext).ee_block, ext4_ext_pblock((*path).p_ext), ext4_ext_is_unwritten((*path).p_ext), ext4_ext_get_actual_len((*path).p_ext)) };
}
__v3 = 0;
}

#[unsafe(no_mangle)]
extern "C" fn ext4_ext_tree_init(handle: *mut jbd2_journal_handle, inode: *mut inode) {
let eh: *mut ext4_extent_header = ext_inode_hdr(inode);
unsafe {
(*eh).eh_depth = 0;
}
unsafe {
(*eh).eh_entries = 0;
}
unsafe {
(*eh).eh_magic = 62218;
}
unsafe {
(*eh).eh_max = ext4_ext_space_root(inode, 0i32) as u16;
}
unsafe {
(*eh).eh_generation = 0;
}
ext4_mark_inode_dirty__40!(handle, inode);
}

#[unsafe(no_mangle)]
extern "C" fn ext4_find_extent(inode: *mut inode, block: u32, mut path: *mut ext4_ext_path, flags: i32) -> *mut ext4_ext_path {
let mut __slate_compound_20: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut gfp_flags: u32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut ppos: i16 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i16> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i16 = core::ptr::addr_of_mut!(i) as *mut i16;
let mut depth: core::mem::MaybeUninit<i16> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i16 = core::ptr::addr_of_mut!(depth) as *mut i16;
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut eh: core::mem::MaybeUninit<*mut ext4_extent_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut ext4_extent_header = core::ptr::addr_of_mut!(eh) as *mut *mut ext4_extent_header;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_5 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 2usize);
ppos = 0;
let __slate_auto_bytes_6 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
gfp_flags = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32);
while false {
}
if flags & 268435456 != 0 {
let __v12: u32 = gfp_flags;
gfp_flags = __v12 | (BIT!(15i32) as u32);
}
*__slate_slot_19 = ext_inode_hdr(inode);
*__slate_slot_17 = ext_depth(inode) as i16;
if (*__slate_slot_17 as i32) < 0 || (*__slate_slot_17 as i32) > 5 {
__ext4_error_inode(inode, c"ext4_find_extent".as_ptr() as *const u8, 904u32, 0u64, 0i32, c"inode has invalid extent depth: %d".as_ptr() as *const u8, *__slate_slot_17 as i32);
*__slate_slot_14 = 117i32.wrapping_neg();
} else {
if !path.is_null() {
ext4_ext_drop_refs(path);
if (*__slate_slot_17 as i32) > ((*path.wrapping_offset(0 as isize)).p_maxdepth as i32) {
kfree(path as *const ());
path = core::ptr::null_mut::<ext4_ext_path>();
}
}
if path.is_null() {
__obj_size = size_mul(core::mem::size_of::<ext4_ext_path>() as u64, (*__slate_slot_17 as i32).wrapping_add(2) as u64);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_5 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_6 = _kzalloc_noprof(__obj_size, gfp_flags, *{
__slate_compound_20 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_20)
});
alloc_tag_restore!();
} else {
*__slate_slot_6 = _kzalloc_noprof(__obj_size, gfp_flags, *{
__slate_compound_22 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_22)
});
}
let __v8: *mut () = *__slate_slot_6;
// account possible depth increase
path = __v8 as *mut ext4_ext_path;
if unlikely!(path.is_null()) != 0 {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut ext4_ext_path;
} else {
(*path.wrapping_offset(0 as isize)).p_maxdepth = (*__slate_slot_17 as i32).wrapping_add(1) as u16;
}
}
(*path.wrapping_offset(0 as isize)).p_hdr = *__slate_slot_19;
(*path.wrapping_offset(0 as isize)).p_bh = core::ptr::null_mut::<buffer_head>();
*__slate_slot_16 = *__slate_slot_17;
if flags & 1073741824 == 0 && (*__slate_slot_17 as i32) == 0 {
ext4_cache_extents(inode, *__slate_slot_19);
}
// walk through the tree
'__join_6: {
while *__slate_slot_16 != 0 {
ext4_ext_binsearch_idx(inode, path.wrapping_offset((ppos as i32) as isize), block);
(*path.wrapping_offset((ppos as i32) as isize)).p_block = ext4_idx_pblock((*path.wrapping_offset((ppos as i32) as isize)).p_idx);
(*path.wrapping_offset((ppos as i32) as isize)).p_depth = *__slate_slot_16 as u16;
(*path.wrapping_offset((ppos as i32) as isize)).p_ext = core::ptr::null_mut::<ext4_extent>();
let __v2: i16 = (*__slate_slot_16 as i32).wrapping_sub(1) as i16;
*__slate_slot_16 = __v2;
*__slate_slot_18 = __read_extent_tree_block(c"ext4_find_extent".as_ptr() as *const u8, 939u32, inode, (*path.wrapping_offset((ppos as i32) as isize)).p_idx, __v2 as i32, flags);
if IS_ERR(*__slate_slot_18 as *const ()) {
break '__join_6;
} else {
*__slate_slot_19 = ext_block_hdr(*__slate_slot_18);
ppos = (ppos as i32).wrapping_add(1) as i16;
(*path.wrapping_offset((ppos as i32) as isize)).p_bh = *__slate_slot_18;
(*path.wrapping_offset((ppos as i32) as isize)).p_hdr = *__slate_slot_19;
}
}
(*path.wrapping_offset((ppos as i32) as isize)).p_depth = *__slate_slot_16 as u16;
(*path.wrapping_offset((ppos as i32) as isize)).p_ext = core::ptr::null_mut::<ext4_extent>();
(*path.wrapping_offset((ppos as i32) as isize)).p_idx = core::ptr::null_mut::<ext4_extent_idx>();
// find extent
ext4_ext_binsearch(inode, path.wrapping_offset((ppos as i32) as isize), block);
// if not an empty leaf
if !unsafe { (*path.wrapping_offset((ppos as i32) as isize)).p_ext }.is_null() {
(*path.wrapping_offset((ppos as i32) as isize)).p_block = ext4_ext_pblock((*path.wrapping_offset((ppos as i32) as isize)).p_ext);
}
return path;
}
*__slate_slot_14 = PTR_ERR(*__slate_slot_18 as *const ()) as i32;
}
ext4_free_ext_path(path);
return ERR_PTR(*__slate_slot_14 as i64) as *mut ext4_ext_path;
}
}

/// ext4_ext_insert_index:
/// insert new index [@logical;@ptr] into the block at @curp;
/// check where to insert: before @curp or after @curp
fn ext4_ext_insert_index(handle: *mut jbd2_journal_handle, inode: *mut inode, curp: *mut ext4_ext_path, logical: i32, ptr: u64) -> i32 {
let mut ix: *mut ext4_extent_idx = unsafe { core::mem::zeroed() };
let mut len: i32 = 0;
let mut err: i32 = ext4_ext_get_access(handle, inode, curp);
if err != 0 {
return err;
}
if unlikely!((logical as u32) == unsafe { (*(*curp).p_idx).ei_block }) != 0 {
unsafe { __ext4_error_inode(inode, c"ext4_ext_insert_index".as_ptr() as *const u8, 989u32, 0u64, 0i32, c"logical %d == ei_block %d!".as_ptr() as *const u8, logical, (*(*curp).p_idx).ei_block) };
return 117i32.wrapping_neg();
}
if unlikely!(((unsafe { (*(*curp).p_hdr).eh_entries }) as i32) >= ((unsafe { (*(*curp).p_hdr).eh_max }) as i32)) != 0 {
unsafe { __ext4_error_inode(inode, c"ext4_ext_insert_index".as_ptr() as *const u8, 998u32, 0u64, 0i32, c"eh_entries %d >= eh_max %d!".as_ptr() as *const u8, (*(*curp).p_hdr).eh_entries as i32, (*(*curp).p_hdr).eh_max as i32) };
return 117i32.wrapping_neg();
}
if (logical as u32) > unsafe { (*(*curp).p_idx).ei_block } {
// insert after
let mut __v0: i32;
if false {
unsafe { _printk(c"insert new index %d after: %llu\n".as_ptr() as *const u8, logical, ptr) };
}
__v0 = 0;
ix = unsafe { (*curp).p_idx }.wrapping_offset(1 as isize);
} else {
// insert before
let mut __v1: i32;
if false {
unsafe { _printk(c"insert new index %d before: %llu\n".as_ptr() as *const u8, logical, ptr) };
}
__v1 = 0;
ix = unsafe { (*curp).p_idx };
}
if unlikely!(ix > EXT_MAX_INDEX!(unsafe { (*curp).p_hdr })) != 0 {
unsafe { __ext4_error_inode(inode, c"ext4_ext_insert_index".as_ptr() as *const u8, 1015u32, 0u64, 0i32, c"ix > EXT_MAX_INDEX!".as_ptr() as *const u8) };
return 117i32.wrapping_neg();
}
len = ((unsafe { EXT_LAST_INDEX!(unsafe { (*curp).p_hdr }).offset_from(ix as *mut ext4_extent_idx) }) as i64).wrapping_add(1) as i32;
loop {
if unlikely!(len < 0) != 0 {
loop {
unsafe {
core::arch::asm!("1513: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1513b - ., 3; .popsection\n# {0}", const 1513 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_insert_index_2, sym __slate_asm_string_ext4_ext_insert_index_3, const 1020 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if len > 0 {
let mut __v7: i32;
if false {
unsafe { _printk(c"insert new index %d: move %d indices from 0x%p to 0x%p\n".as_ptr() as *const u8, logical, len, ix, ix.wrapping_offset(1 as isize)) };
}
__v7 = 0;
unsafe { memmove(ix.wrapping_offset(1 as isize) as *mut (), ix as *const (), (len as u64).wrapping_mul(core::mem::size_of::<ext4_extent_idx>() as u64)) };
}
unsafe {
(*ix).ei_block = logical as u32;
}
ext4_idx_store_pblock(ix, ptr);
le16_add_cpu(unsafe { core::ptr::addr_of_mut!((*(*curp).p_hdr).eh_entries) }, 1u16);
if unlikely!(ix > EXT_LAST_INDEX!(unsafe { (*curp).p_hdr })) != 0 {
unsafe { __ext4_error_inode(inode, c"ext4_ext_insert_index".as_ptr() as *const u8, 1033u32, 0u64, 0i32, c"ix > EXT_LAST_INDEX!".as_ptr() as *const u8) };
return 117i32.wrapping_neg();
}
err = ext4_ext_dirty__13!(handle, inode, curp);
ext4_std_error__15!(unsafe { (*inode).i_sb }, err);
err
}

/// ext4_ext_split:
/// inserts new subtree into the path, using free index entry
/// at depth @at:
/// - allocates all needed blocks (new leaf and all intermediate index blocks)
/// - makes decision where to split
/// - moves remaining extents and index entries (right to the split point)
///   into the newly allocated blocks
/// - initializes subtree
fn ext4_ext_split(handle: *mut jbd2_journal_handle, inode: *mut inode, flags: u32, path: *mut ext4_ext_path, newext: *mut ext4_extent, at: i32) -> i32 {
let mut __slate_compound_50: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_52: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut ext_size: u64 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut gfp_flags: u32 = unsafe { core::mem::zeroed() };
// array of allocated blocks
let mut ablocks: *mut u64 = unsafe { core::mem::zeroed() };
let mut border: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut u32 = core::ptr::addr_of_mut!(border) as *mut u32;
let mut oldblock: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut u64 = core::ptr::addr_of_mut!(oldblock) as *mut u64;
let mut newblock: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut u64 = core::ptr::addr_of_mut!(newblock) as *mut u64;
let mut a: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut i32 = core::ptr::addr_of_mut!(a) as *mut i32;
let mut m: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut i32 = core::ptr::addr_of_mut!(m) as *mut i32;
let mut k: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut i32 = core::ptr::addr_of_mut!(k) as *mut i32;
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut fidx: core::mem::MaybeUninit<*mut ext4_extent_idx> = core::mem::MaybeUninit::uninit();
let __slate_slot_46: *mut *mut ext4_extent_idx = core::ptr::addr_of_mut!(fidx) as *mut *mut ext4_extent_idx;
let mut neh: core::mem::MaybeUninit<*mut ext4_extent_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_47: *mut *mut ext4_extent_header = core::ptr::addr_of_mut!(neh) as *mut *mut ext4_extent_header;
let mut depth: i32 = unsafe { core::mem::zeroed() };
let mut bh: *mut buffer_head = unsafe { core::mem::zeroed() };
unsafe {
{
bh = core::ptr::null_mut::<buffer_head>();
depth = ext_depth(inode) as i32;
let __slate_auto_bytes_2 = __slate_slot_47 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_46 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
i = at;
let __slate_auto_bytes_4 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ablocks = core::ptr::null_mut::<u64>();
gfp_flags = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32);
err = 0;
ext_size = 0;
if flags & 268435456 != 0 {
let __v34: u32 = gfp_flags;
gfp_flags = __v34 | (BIT!(15i32) as u32);
}
}
// make decision: where to split?
//
// FIXME: now decision is simplest: at current extent
// if current leaf will be split, then we should use
// border from split point
if unlikely!((*path.wrapping_offset(depth as isize)).p_ext > EXT_MAX_EXTENT!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr })) != 0 {
__ext4_error_inode(inode, c"ext4_ext_split".as_ptr() as *const u8, 1079u32, 0u64, 0i32, c"p_ext > EXT_MAX_EXTENT!".as_ptr() as *const u8);
return 117i32.wrapping_neg();
} else {
if (*path.wrapping_offset(depth as isize)).p_ext != EXT_MAX_EXTENT!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr }) {
*__slate_slot_39 = (*unsafe { (*path.wrapping_offset(depth as isize)).p_ext }.wrapping_offset(1 as isize)).ee_block;
} else {
*__slate_slot_39 = (*newext).ee_block;
}
// If error occurs, then we break processing
// and mark filesystem read-only. index won't
// be inserted and tree will be in consistent
// state. Next mount will repair buffers too.
__obj_size = size_mul(core::mem::size_of::<u64>() as u64, depth as u64);
let __slate_auto_bytes_1 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_25 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_26 = _kzalloc_noprof(__obj_size, gfp_flags, *{
__slate_compound_50 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_50)
});
alloc_tag_restore!();
} else {
*__slate_slot_26 = _kzalloc_noprof(__obj_size, gfp_flags, *{
__slate_compound_52 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_52)
});
}
let __v28: *mut () = *__slate_slot_26;
// Get array to track all allocated blocks.
// We need this to handle errors and free blocks
// upon them.
ablocks = __v28 as *mut u64;
if ablocks.is_null() {
return 12i32.wrapping_neg();
} else {
// allocate all needed blocks
*__slate_slot_42 = 0;
'__join_10: {
while *__slate_slot_42 < depth.wrapping_sub(at) {
*__slate_slot_41 = ext4_ext_new_meta_block(handle, inode, path, newext, core::ptr::addr_of_mut!(err), flags);
if *__slate_slot_41 == 0 {
break '__join_10;
} else {
*ablocks.wrapping_offset(*__slate_slot_42 as isize) = *__slate_slot_41;
let __v23: i32 = *__slate_slot_42;
*__slate_slot_42 = __v23.wrapping_add(1);
}
}
// initialize new leaf
let __v21: i32 = *__slate_slot_42;
let __v20: i32 = __v21.wrapping_sub(1);
*__slate_slot_42 = __v20;
*__slate_slot_41 = *ablocks.wrapping_offset(__v20 as isize);
if unlikely!(*__slate_slot_41 == 0) != 0 {
__ext4_error_inode(inode, c"ext4_ext_split".as_ptr() as *const u8, 1123u32, 0u64, 0i32, c"newblock == 0!".as_ptr() as *const u8);
err = 117i32.wrapping_neg();
} else {
bh = sb_getblk_gfp((*inode).i_sb, *__slate_slot_41, (BIT!(3i32) as u32) | (((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32)));
if unlikely!(bh.is_null()) != 0 {
err = 12i32.wrapping_neg();
break '__join_10;
} else {
lock_buffer(bh);
err = ext4_journal_get_create_access__5!(handle, unsafe { (*inode).i_sb }, bh, 1i32);
if err == 0 {
*__slate_slot_47 = ext_block_hdr(bh);
(*(*__slate_slot_47)).eh_entries = 0;
(*(*__slate_slot_47)).eh_max = ext4_ext_space_block(inode, 0i32) as u16;
(*(*__slate_slot_47)).eh_magic = 62218;
(*(*__slate_slot_47)).eh_depth = 0;
(*(*__slate_slot_47)).eh_generation = 0;
// move remainder of path[depth] to the new leaf
if unlikely!(((*(*path.wrapping_offset(depth as isize)).p_hdr).eh_entries as i32) != ((*(*path.wrapping_offset(depth as isize)).p_hdr).eh_max as i32)) != 0 {
__ext4_error_inode(inode, c"ext4_ext_split".as_ptr() as *const u8, 1151u32, 0u64, 0i32, c"eh_entries %d != eh_max %d!".as_ptr() as *const u8, (*(*path.wrapping_offset(depth as isize)).p_hdr).eh_entries as i32, (*(*path.wrapping_offset(depth as isize)).p_hdr).eh_max as i32);
err = 117i32.wrapping_neg();
} else {
// start copy from next extent
let __v19: *mut ext4_ext_path = path.wrapping_offset(depth as isize);
let __v18: *mut ext4_extent = (*__v19).p_ext;
(*__v19).p_ext = __v18.wrapping_offset(1 as isize);
*__slate_slot_43 = (EXT_MAX_EXTENT!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr }).offset_from(__v18 as *mut ext4_extent) as i64) as i32;
if *__slate_slot_43 != 0 {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_16 = EXT_FIRST_EXTENT!(*__slate_slot_47);
memmove(*__slate_slot_16 as *mut (), (*path.wrapping_offset(depth as isize)).p_ext as *const (), (core::mem::size_of::<ext4_extent>() as u64).wrapping_mul(*__slate_slot_43 as u64));
le16_add_cpu(core::ptr::addr_of_mut!((*(*__slate_slot_47)).eh_entries), *__slate_slot_43 as u16);
}
// zero out unused area in the extent block
ext_size = (core::mem::size_of::<ext4_extent_header>() as u64).wrapping_add((core::mem::size_of::<ext4_extent>() as u64).wrapping_mul(((*(*__slate_slot_47)).eh_entries as i32) as u64));
memset(unsafe { (*bh).b_data }.wrapping_offset(ext_size as isize) as *mut (), 0i32, unsafe { (*(*inode).i_sb).s_blocksize }.wrapping_sub(ext_size));
ext4_extent_block_csum_set(inode, *__slate_slot_47);
set_buffer_uptodate(bh);
unlock_buffer(bh);
err = ext4_handle_dirty_metadata__50!(handle, inode, bh);
if err == 0 {
brelse(bh);
bh = core::ptr::null_mut::<buffer_head>();
// correct old leaf
if *__slate_slot_43 != 0 {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(depth as isize));
if err != 0 {
break '__join_10;
} else {
le16_add_cpu(core::ptr::addr_of_mut!((*(*path.wrapping_offset(depth as isize)).p_hdr).eh_entries), (*__slate_slot_43).wrapping_neg() as u16);
err = ext4_ext_dirty__16!(handle, inode, path.wrapping_offset(depth as isize));
if err != 0 {
break '__join_10;
}
}
}
// create intermediate indexes
*__slate_slot_44 = depth.wrapping_sub(at).wrapping_sub(1);
if unlikely!(*__slate_slot_44 < 0) != 0 {
__ext4_error_inode(inode, c"ext4_ext_split".as_ptr() as *const u8, 1194u32, 0u64, 0i32, c"k %d < 0!".as_ptr() as *const u8, *__slate_slot_44);
err = 117i32.wrapping_neg();
} else {
if *__slate_slot_44 != 0 {
}
// insert new index into current index block
//
// current depth stored in i var
i = depth.wrapping_sub(1);
'__join_23: {
'__join_20: {
loop {
let __v3: i32 = *__slate_slot_44;
*__slate_slot_44 = __v3.wrapping_sub(1);
if __v3 != 0 {
*__slate_slot_40 = *__slate_slot_41;
let mut __v14: i32 = *__slate_slot_42;
let __v13: i32 = __v14.wrapping_sub(1);
*__slate_slot_42 = __v13;
*__slate_slot_41 = *ablocks.wrapping_offset(__v13 as isize);
bh = sb_getblk((*inode).i_sb, *__slate_slot_41);
if unlikely!(bh.is_null()) != 0 {
break '__join_23;
} else {
lock_buffer(bh);
err = ext4_journal_get_create_access__6!(handle, unsafe { (*inode).i_sb }, bh, 1i32);
if err != 0 {
break '__join_10;
} else {
*__slate_slot_47 = ext_block_hdr(bh);
(*(*__slate_slot_47)).eh_entries = 1;
(*(*__slate_slot_47)).eh_magic = 62218;
(*(*__slate_slot_47)).eh_max = ext4_ext_space_block_idx(inode, 0i32) as u16;
(*(*__slate_slot_47)).eh_depth = depth.wrapping_sub(i) as u16;
(*(*__slate_slot_47)).eh_generation = 0;
*__slate_slot_46 = EXT_FIRST_INDEX!(*__slate_slot_47);
(*(*__slate_slot_46)).ei_block = *__slate_slot_39;
ext4_idx_store_pblock(*__slate_slot_46, *__slate_slot_40);
// move remainder of path[i] to the new index block
if unlikely!(EXT_MAX_INDEX!(unsafe { (*path.wrapping_offset(i as isize)).p_hdr }) != EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(i as isize)).p_hdr })) != 0 {
break '__join_20;
} else {
// start copy indexes
let __v11: *mut ext4_ext_path = path.wrapping_offset(i as isize);
let __v10: *mut ext4_extent_idx = (*__v11).p_idx;
(*__v11).p_idx = __v10.wrapping_offset(1 as isize);
*__slate_slot_43 = (EXT_MAX_INDEX!(unsafe { (*path.wrapping_offset(i as isize)).p_hdr }).offset_from(__v10 as *mut ext4_extent_idx) as i64) as i32;
if *__slate_slot_43 != 0 {
let __v7: *mut ext4_extent_idx = *__slate_slot_46;
let __v6: *mut ext4_extent_idx = __v7.wrapping_offset(1 as isize);
*__slate_slot_46 = __v6;
memmove(__v6 as *mut (), (*path.wrapping_offset(i as isize)).p_idx as *const (), (core::mem::size_of::<ext4_extent_idx>() as u64).wrapping_mul(*__slate_slot_43 as u64));
le16_add_cpu(core::ptr::addr_of_mut!((*(*__slate_slot_47)).eh_entries), *__slate_slot_43 as u16);
}
// zero out unused area in the extent block
ext_size = (core::mem::size_of::<ext4_extent_header>() as u64).wrapping_add((core::mem::size_of::<ext4_extent>() as u64).wrapping_mul(((*(*__slate_slot_47)).eh_entries as i32) as u64));
memset(unsafe { (*bh).b_data }.wrapping_offset(ext_size as isize) as *mut (), 0i32, unsafe { (*(*inode).i_sb).s_blocksize }.wrapping_sub(ext_size));
ext4_extent_block_csum_set(inode, *__slate_slot_47);
set_buffer_uptodate(bh);
unlock_buffer(bh);
err = ext4_handle_dirty_metadata__52!(handle, inode, bh);
if err != 0 {
break '__join_10;
} else {
brelse(bh);
bh = core::ptr::null_mut::<buffer_head>();
// correct old index
if *__slate_slot_43 != 0 {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(i as isize));
if err != 0 {
break '__join_10;
} else {
le16_add_cpu(core::ptr::addr_of_mut!((*(*path.wrapping_offset(i as isize)).p_hdr).eh_entries), (*__slate_slot_43).wrapping_neg() as u16);
err = ext4_ext_dirty__15!(handle, inode, path.wrapping_offset(i as isize));
if err != 0 {
break '__join_10;
}
}
}
i = i.wrapping_sub(1);
}
}
}
}
} else {
break;
}
}
// insert new index
err = ext4_ext_insert_index(handle, inode, path.wrapping_offset(at as isize), *__slate_slot_39 as i32, *__slate_slot_41);
break '__join_10;
}
__ext4_error_inode(inode, c"ext4_ext_split".as_ptr() as *const u8, 1236u32, 0u64, 0i32, c"EXT_MAX_INDEX != EXT_LAST_INDEX ee_block %d!".as_ptr() as *const u8, (*(*path.wrapping_offset(i as isize)).p_ext).ee_block);
err = 117i32.wrapping_neg();
break '__join_10;
}
err = 12i32.wrapping_neg();
}
}
}
}
}
}
}
if !bh.is_null() {
if buffer_locked(bh as *const buffer_head) != 0 {
unlock_buffer(bh);
}
brelse(bh);
}
if err != 0 {
// free all allocated blocks in error case
i = 0;
while i < depth {
if *ablocks.wrapping_offset(i as isize) != 0 {
ext4_free_blocks(handle, inode, core::ptr::null_mut::<buffer_head>(), *ablocks.wrapping_offset(i as isize), 1u64, 1i32);
}
i = i.wrapping_add(1);
}
}
kfree(ablocks as *const ());
return err;
}
}
}
}

/// ext4_ext_grow_indepth:
/// implements tree growing procedure:
/// - allocates new block
/// - moves top-level data (index block or leaf) into the new block
/// - initializes new top-level, creating index that points to the
///   just created block
fn ext4_ext_grow_indepth(handle: *mut jbd2_journal_handle, inode: *mut inode, mut flags: u32) -> i32 {
let mut ext_size: u64 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut es: *mut ext4_super_block = unsafe { core::mem::zeroed() };
let mut goal: u64 = unsafe { core::mem::zeroed() };
let mut newblock: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(newblock) as *mut u64;
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut neh: core::mem::MaybeUninit<*mut ext4_extent_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut ext4_extent_header = core::ptr::addr_of_mut!(neh) as *mut *mut ext4_extent_header;
unsafe {
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
goal = 0;
es = (*EXT4_SB((*inode).i_sb)).s_es;
err = 0;
ext_size = 0;
// Try to prepend new index to old one
if ext_depth(inode) != 0 {
goal = ext4_idx_pblock(EXT_FIRST_INDEX!(ext_inode_hdr(inode)));
}
if goal > ((*es).s_first_data_block as u64) {
flags |= 512;
goal = goal.wrapping_sub(1);
} else {
goal = ext4_inode_to_goal_block(inode);
}
*__slate_slot_9 = ext4_new_meta_blocks(handle, inode, goal, flags, core::ptr::null_mut::<u64>(), core::ptr::addr_of_mut!(err));
if *__slate_slot_9 == 0 {
return err;
} else {
*__slate_slot_10 = sb_getblk_gfp((*inode).i_sb, *__slate_slot_9, (BIT!(3i32) as u32) | (((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32)));
if unlikely!((*__slate_slot_10).is_null()) != 0 {
return 12i32.wrapping_neg();
} else {
lock_buffer(*__slate_slot_10);
err = ext4_journal_get_create_access__7!(handle, unsafe { (*inode).i_sb }, *__slate_slot_10, 1i32);
if err != 0 {
unlock_buffer(*__slate_slot_10);
} else {
ext_size = core::mem::size_of::<[u32; 15]>() as u64;
// move top-level index/leaf into new block
memmove((*(*__slate_slot_10)).b_data as *mut (), ((*EXT4_I!(inode)).i_data.as_mut_ptr() as *mut u32) as *const (), ext_size);
// zero out unused area in the extent block
memset(unsafe { (*(*__slate_slot_10)).b_data }.wrapping_offset(ext_size as isize) as *mut (), 0i32, unsafe { (*(*inode).i_sb).s_blocksize }.wrapping_sub(ext_size));
// set size of new block
*__slate_slot_11 = ext_block_hdr(*__slate_slot_10);
// old root could have indexes or leaves
// so calculate e_max right way
if ext_depth(inode) != 0 {
(*(*__slate_slot_11)).eh_max = ext4_ext_space_block_idx(inode, 0i32) as u16;
} else {
(*(*__slate_slot_11)).eh_max = ext4_ext_space_block(inode, 0i32) as u16;
}
(*(*__slate_slot_11)).eh_magic = 62218;
ext4_extent_block_csum_set(inode, *__slate_slot_11);
set_buffer_uptodate(*__slate_slot_10);
set_buffer_verified(*__slate_slot_10);
unlock_buffer(*__slate_slot_10);
err = ext4_handle_dirty_metadata__16!(handle, inode, *__slate_slot_10);
if err == 0 {
// Update top-level index: num,max,pointer
*__slate_slot_11 = ext_inode_hdr(inode);
(*(*__slate_slot_11)).eh_entries = 1;
ext4_idx_store_pblock(EXT_FIRST_INDEX!(*__slate_slot_11), *__slate_slot_9);
if ((*(*__slate_slot_11)).eh_depth as i32) == 0 {
// Root extent block becomes index block
(*(*__slate_slot_11)).eh_max = ext4_ext_space_root_idx(inode, 0i32) as u16;
(*EXT_FIRST_INDEX!(*__slate_slot_11)).ei_block = (*EXT_FIRST_EXTENT!(*__slate_slot_11)).ee_block;
}
le16_add_cpu(core::ptr::addr_of_mut!((*(*__slate_slot_11)).eh_depth), 1u16);
err = ext4_mark_inode_dirty__65!(handle, inode);
}
}
brelse(*__slate_slot_10);
return err;
}
}
}
}

/// ext4_ext_create_new_leaf:
/// finds empty index and adds new leaf.
/// if no free index is found, then it requests in-depth growing.
fn ext4_ext_create_new_leaf(handle: *mut jbd2_journal_handle, inode: *mut inode, mb_flags: u32, gb_flags: u32, mut path: *mut ext4_ext_path, newext: *mut ext4_extent) -> *mut ext4_ext_path {
let mut ee_block: u32 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32;
let mut curp: core::mem::MaybeUninit<*mut ext4_ext_path> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut ext4_ext_path = core::ptr::addr_of_mut!(curp) as *mut *mut ext4_ext_path;
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
err = 0;
ee_block = (*newext).ee_block;
'__join_0: {
'__join_8: {
loop {
let __v4: i32 = ext_depth(inode) as i32;
*__slate_slot_8 = __v4;
*__slate_slot_7 = __v4;
// walk up to the tree and look for free index entry
*__slate_slot_9 = path.wrapping_offset(*__slate_slot_8 as isize);
while *__slate_slot_7 > 0 && !EXT_HAS_FREE_INDEX!(*__slate_slot_9) {
let __v3: i32 = *__slate_slot_7;
*__slate_slot_7 = __v3.wrapping_sub(1);
let __v1: *mut ext4_ext_path = *__slate_slot_9;
*__slate_slot_9 = __v1.wrapping_offset(-(1 as isize));
}
// we use already allocated block for index block,
// so subsequent data blocks should be contiguous
if EXT_HAS_FREE_INDEX!(*__slate_slot_9) {
break '__join_8;
} else {
// tree is full, time to grow in depth
err = ext4_ext_grow_indepth(handle, inode, mb_flags);
if err != 0 {
break '__join_0;
} else {
// refill path
path = ext4_find_extent(inode, ee_block, path, gb_flags as i32);
if IS_ERR(path as *const ()) {
return path;
} else {
// only first (depth 0 -> 1) produces free space;
// in all other cases we have to split the grown tree
*__slate_slot_8 = ext_depth(inode) as i32;
if ((*(*path.wrapping_offset(*__slate_slot_8 as isize)).p_hdr).eh_entries as i32) == ((*(*path.wrapping_offset(*__slate_slot_8 as isize)).p_hdr).eh_max as i32) {
// now we need to split
} else {
break;
}
}
}
}
}
return path;
}
// if we found index with free entry, then use that
// entry: create all needed subtree and add new leaf
err = ext4_ext_split(handle, inode, mb_flags, path, newext, *__slate_slot_7);
if err == 0 {
// refill path
path = ext4_find_extent(inode, ee_block, path, gb_flags as i32);
return path;
}
}
ext4_free_ext_path(path);
return ERR_PTR(err as i64) as *mut ext4_ext_path;
}
}

/// search the closest allocated block to the left for *logical
/// and returns it at @logical + it's physical address at @phys
/// if *logical is the smallest allocated block, the function
/// returns 0 at @phys
/// return value contains 0 (success) or error code
fn ext4_ext_search_left(inode: *mut inode, path: *mut ext4_ext_path, logical: *mut u32, phys: *mut u64) -> i32 {
let mut depth: i32 = 0;
let mut ee_len: i32 = 0;
if unlikely!(path.is_null()) != 0 {
unsafe { __ext4_error_inode(inode, c"ext4_ext_search_left".as_ptr() as *const u8, 1476u32, 0u64, 0i32, c"path == NULL *logical %d!".as_ptr() as *const u8, *logical) };
return 117i32.wrapping_neg();
}
depth = (unsafe { (*path).p_depth }) as i32;
unsafe {
*phys = 0;
}
if depth == 0 && unsafe { (*path).p_ext }.is_null() {
return 0;
}
// usually extent in the path covers blocks smaller
// then *logical, but it can be that extent is the
// first one in the file
let ex: *mut ext4_extent = unsafe { (*path.wrapping_offset(depth as isize)).p_ext };
ee_len = ext4_ext_get_actual_len(ex);
if (unsafe { *logical }) < unsafe { (*ex).ee_block } {
if unlikely!(EXT_FIRST_EXTENT!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr }) != ex) != 0 {
unsafe { __ext4_error_inode(inode, c"ext4_ext_search_left".as_ptr() as *const u8, 1495u32, 0u64, 0i32, c"EXT_FIRST_EXTENT != ex *logical %d ee_block %d!".as_ptr() as *const u8, *logical, (*ex).ee_block) };
return 117i32.wrapping_neg();
}
loop {
let __v1: i32 = depth.wrapping_sub(1);
depth = __v1;
if __v1 < 0 {
break;
}
let ix: *mut ext4_extent_idx = unsafe { (*path.wrapping_offset(depth as isize)).p_idx };
if unlikely!(ix != EXT_FIRST_INDEX!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr })) != 0 {
unsafe { __ext4_error_inode(inode, c"ext4_ext_search_left".as_ptr() as *const u8, 1505u32, 0u64, 0i32, c"ix (%d) != EXT_FIRST_INDEX (%d) (depth %d)!".as_ptr() as *const u8, if !ix.is_null() { (*ix).ei_block } else { 0u32 }, (*EXT_FIRST_INDEX!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr })).ei_block, depth) };
return 117i32.wrapping_neg();
}
}
return 0;
}
if unlikely!((unsafe { *logical }) < unsafe { (*ex).ee_block }.wrapping_add(ee_len as u32)) != 0 {
unsafe { __ext4_error_inode(inode, c"ext4_ext_search_left".as_ptr() as *const u8, 1515u32, 0u64, 0i32, c"logical %d < ee_block %d + ee_len %d!".as_ptr() as *const u8, *logical, (*ex).ee_block, ee_len) };
return 117i32.wrapping_neg();
}
unsafe {
*logical = unsafe { (*ex).ee_block }.wrapping_add(ee_len as u32).wrapping_sub(1);
}
unsafe {
*phys = ext4_ext_pblock(ex).wrapping_add(ee_len as u64).wrapping_sub(1);
}
0
}

/// Search the closest allocated block to the right for *logical
/// and returns it at @logical + it's physical address at @phys.
/// If not exists, return 0 and @phys is set to 0. We will return
/// 1 which means we found an allocated block and ret_ex is valid.
/// Or return a (< 0) error code.
fn ext4_ext_search_right(inode: *mut inode, path: *mut ext4_ext_path, logical: *mut u32, phys: *mut u64, ret_ex: *mut ext4_extent, flags: i32) -> i32 {
let mut ee_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(ee_len) as *mut i32; // Note, NOT eh_depth; depth from top of tree
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut ix: core::mem::MaybeUninit<*mut ext4_extent_idx> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut ext4_extent_idx = core::ptr::addr_of_mut!(ix) as *mut *mut ext4_extent_idx;
let mut eh: core::mem::MaybeUninit<*mut ext4_extent_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut ext4_extent_header = core::ptr::addr_of_mut!(eh) as *mut *mut ext4_extent_header;
let mut bh: *mut buffer_head = unsafe { core::mem::zeroed() };
unsafe {
bh = core::ptr::null_mut::<buffer_head>();
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if unlikely!(path.is_null()) != 0 {
__ext4_error_inode(inode, c"ext4_ext_search_right".as_ptr() as *const u8, 1544u32, 0u64, 0i32, c"path == NULL *logical %d!".as_ptr() as *const u8, *logical);
return 117i32.wrapping_neg();
} else {
*__slate_slot_11 = (*path).p_depth as i32;
*phys = 0;
if *__slate_slot_11 == 0 && unsafe { (*path).p_ext }.is_null() {
return 0;
} else {
'__join_4: {
// usually extent in the path covers blocks smaller
// then *logical, but it can be that extent is the
// first one in the file
*__slate_slot_12 = (*path.wrapping_offset(*__slate_slot_11 as isize)).p_ext;
*__slate_slot_10 = ext4_ext_get_actual_len(*__slate_slot_12);
if *logical < (*(*__slate_slot_12)).ee_block {
if unlikely!(EXT_FIRST_EXTENT!(unsafe { (*path.wrapping_offset(*__slate_slot_11 as isize)).p_hdr }) != *__slate_slot_12) != 0 {
__ext4_error_inode(inode, c"ext4_ext_search_right".as_ptr() as *const u8, 1563u32, 0u64, 0i32, c"first_extent(path[%d].p_hdr) != ex".as_ptr() as *const u8, *__slate_slot_11);
return 117i32.wrapping_neg();
} else {
loop {
let __v9: i32 = *__slate_slot_11;
let __v8: i32 = __v9.wrapping_sub(1);
*__slate_slot_11 = __v8;
if __v8 >= 0 {
*__slate_slot_13 = (*path.wrapping_offset(*__slate_slot_11 as isize)).p_idx;
if unlikely!(*__slate_slot_13 != EXT_FIRST_INDEX!(unsafe { (*path.wrapping_offset(*__slate_slot_11 as isize)).p_hdr })) != 0 {
break;
}
} else {
break '__join_4;
}
}
__ext4_error_inode(inode, c"ext4_ext_search_right".as_ptr() as *const u8, 1571u32, 0u64, 0i32, c"ix != EXT_FIRST_INDEX *logical %d!".as_ptr() as *const u8, *logical);
return 117i32.wrapping_neg();
}
} else {
if unlikely!(*logical < unsafe { (*(*__slate_slot_12)).ee_block }.wrapping_add(*__slate_slot_10 as u32)) != 0 {
__ext4_error_inode(inode, c"ext4_ext_search_right".as_ptr() as *const u8, 1581u32, 0u64, 0i32, c"logical %d < ee_block %d + ee_len %d!".as_ptr() as *const u8, *logical, (*(*__slate_slot_12)).ee_block, *__slate_slot_10);
return 117i32.wrapping_neg();
} else {
if *__slate_slot_12 != EXT_LAST_EXTENT!(unsafe { (*path.wrapping_offset(*__slate_slot_11 as isize)).p_hdr }) {
// next allocated block in this leaf
let __v7: *mut ext4_extent = *__slate_slot_12;
*__slate_slot_12 = __v7.wrapping_offset(1 as isize);
} else {
// go up and search for index to the right
'__join_12: {
loop {
let __v5: i32 = *__slate_slot_11;
let __v4: i32 = __v5.wrapping_sub(1);
*__slate_slot_11 = __v4;
if __v4 >= 0 {
*__slate_slot_13 = (*path.wrapping_offset(*__slate_slot_11 as isize)).p_idx;
if *__slate_slot_13 != EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(*__slate_slot_11 as isize)).p_hdr }) {
break '__join_12;
}
} else {
break;
}
}
// we've gone up to the root and found no index to the right
return 0;
}
let __v3: *mut ext4_extent_idx = *__slate_slot_13;
*__slate_slot_13 = __v3.wrapping_offset(1 as isize);
// we've found index to the right, let's
// follow it and find the closest allocated
// block to the right
loop {
let __v1: i32 = *__slate_slot_11;
let __v0: i32 = __v1.wrapping_add(1);
*__slate_slot_11 = __v0;
if __v0 < ((*path).p_depth as i32) {
// subtract from p_depth to get proper eh_depth
bh = read_extent_tree_block!(inode, *__slate_slot_13, ((*path).p_depth as i32).wrapping_sub(*__slate_slot_11), flags);
if IS_ERR(bh as *const ()) {
return PTR_ERR(bh as *const ()) as i32;
} else {
*__slate_slot_14 = ext_block_hdr(bh);
*__slate_slot_13 = EXT_FIRST_INDEX!(*__slate_slot_14);
put_bh(bh);
}
} else {
break;
}
}
bh = read_extent_tree_block__2!(inode, *__slate_slot_13, ((*path).p_depth as i32).wrapping_sub(*__slate_slot_11), flags);
if IS_ERR(bh as *const ()) {
return PTR_ERR(bh as *const ()) as i32;
} else {
*__slate_slot_14 = ext_block_hdr(bh);
*__slate_slot_12 = EXT_FIRST_EXTENT!(*__slate_slot_14);
}
}
}
}
}
*logical = (*(*__slate_slot_12)).ee_block;
*phys = ext4_ext_pblock(*__slate_slot_12);
if !ret_ex.is_null() {
*ret_ex = *(*__slate_slot_12);
}
if !bh.is_null() {
put_bh(bh);
}
return 1;
}
}
}
}

/// ext4_ext_next_allocated_block:
/// returns allocated block in subsequent extent or EXT_MAX_BLOCKS.
/// NOTE: it considers block number from index entry as
/// allocated block. Thus, index entries have to be consistent
/// with leaves.
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_next_allocated_block(path: *mut ext4_ext_path) -> u32 {
let mut depth: i32 = 0;
loop {
if unlikely!(path.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("1514: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1514b - ., 3; .popsection\n# {0}", const 1514 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_next_allocated_block_0, sym __slate_asm_string_ext4_ext_next_allocated_block_1, const 1644 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
depth = (unsafe { (*path).p_depth }) as i32;
if depth == 0 && unsafe { (*path).p_ext }.is_null() {
return 4294967295;
}
while depth >= 0 {
let p: *mut ext4_ext_path = path.wrapping_offset(depth as isize);
if depth == ((unsafe { (*path).p_depth }) as i32) {
// leaf
if !unsafe { (*p).p_ext }.is_null() && (unsafe { (*p).p_ext }) != EXT_LAST_EXTENT!(unsafe { (*p).p_hdr }) {
return unsafe { (*unsafe { (*p).p_ext }.wrapping_offset(1 as isize)).ee_block };
}
} else {
// index
if (unsafe { (*p).p_idx }) != EXT_LAST_INDEX!(unsafe { (*p).p_hdr }) {
return unsafe { (*unsafe { (*p).p_idx }.wrapping_offset(1 as isize)).ei_block };
}
}
depth = depth.wrapping_sub(1);
}
4294967295
}

/// ext4_ext_next_leaf_block:
/// returns first allocated block from next leaf or EXT_MAX_BLOCKS
fn ext4_ext_next_leaf_block(path: *mut ext4_ext_path) -> u32 {
let mut depth: i32 = 0;
loop {
if unlikely!(path.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("1515: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1515b - ., 3; .popsection\n# {0}", const 1515 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_next_leaf_block_0, sym __slate_asm_string_ext4_ext_next_leaf_block_1, const 1676 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
depth = (unsafe { (*path).p_depth }) as i32;
// zero-tree has no leaf blocks at all
if depth == 0 {
return 4294967295;
}
// go to index block
depth = depth.wrapping_sub(1);
while depth >= 0 {
if (unsafe { (*path.wrapping_offset(depth as isize)).p_idx }) != EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr }) {
return unsafe { (*unsafe { (*path.wrapping_offset(depth as isize)).p_idx }.wrapping_offset(1 as isize)).ei_block };
}
depth = depth.wrapping_sub(1);
}
4294967295
}

/// ext4_ext_correct_indexes:
/// if leaf gets modified and modified extent is first in the leaf,
/// then we have to correct all indexes above.
/// TODO: do we need to correct tree in all cases?
fn ext4_ext_correct_indexes(handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path) -> i32 {
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut k: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(k) as *mut i32;
let mut border: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(border) as *mut u32;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut depth: i32 = unsafe { core::mem::zeroed() };
let mut eh: core::mem::MaybeUninit<*mut ext4_extent_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut ext4_extent_header = core::ptr::addr_of_mut!(eh) as *mut *mut ext4_extent_header;
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
depth = ext_depth(inode) as i32;
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
err = 0;
*__slate_slot_9 = (*path.wrapping_offset(depth as isize)).p_hdr;
*__slate_slot_7 = (*path.wrapping_offset(depth as isize)).p_ext;
if unlikely!((*__slate_slot_7).is_null() || (*__slate_slot_9).is_null()) != 0 {
__ext4_error_inode(inode, c"ext4_ext_correct_indexes".as_ptr() as *const u8, 1717u32, 0u64, 0i32, c"ex %p == NULL or eh %p == NULL".as_ptr() as *const u8, *__slate_slot_7, *__slate_slot_9);
return 117i32.wrapping_neg();
} else {
if depth == 0 {
// there is no tree at all
return 0;
} else {
if *__slate_slot_7 != EXT_FIRST_EXTENT!(*__slate_slot_9) {
// we correct tree if first leaf got modified only
return 0;
} else {
// TODO: we need correction if border is smaller than current one
*__slate_slot_5 = depth.wrapping_sub(1);
*__slate_slot_6 = (*(*path.wrapping_offset(depth as isize)).p_ext).ee_block;
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_5 as isize));
if err != 0 {
return err;
} else {
if unlikely!((*path.wrapping_offset(*__slate_slot_5 as isize)).p_idx > EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(*__slate_slot_5 as isize)).p_hdr })) != 0 {
__ext4_error_inode(inode, c"ext4_ext_correct_indexes".as_ptr() as *const u8, 1743u32, 0u64, 0i32, c"path[%d].p_idx %p > EXT_LAST_INDEX %p".as_ptr() as *const u8, *__slate_slot_5, (*path.wrapping_offset(*__slate_slot_5 as isize)).p_idx, EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(*__slate_slot_5 as isize)).p_hdr }));
return 117i32.wrapping_neg();
} else {
(*(*path.wrapping_offset(*__slate_slot_5 as isize)).p_idx).ei_block = *__slate_slot_6;
err = ext4_ext_dirty__17!(handle, inode, path.wrapping_offset(*__slate_slot_5 as isize));
if err != 0 {
return err;
} else {
'__join_3: {
'__join_1: {
loop {
let __v3: i32 = *__slate_slot_5;
*__slate_slot_5 = __v3.wrapping_sub(1);
if __v3 != 0 {
// change all left-side indexes
if (*path.wrapping_offset((*__slate_slot_5).wrapping_add(1) as isize)).p_idx != EXT_FIRST_INDEX!(unsafe { (*path.wrapping_offset((*__slate_slot_5).wrapping_add(1) as isize)).p_hdr }) {
break '__join_3;
} else {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_5 as isize));
if err != 0 {
break '__join_1;
} else {
if unlikely!((*path.wrapping_offset(*__slate_slot_5 as isize)).p_idx > EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(*__slate_slot_5 as isize)).p_hdr })) != 0 {
break;
} else {
(*(*path.wrapping_offset(*__slate_slot_5 as isize)).p_idx).ei_block = *__slate_slot_6;
err = ext4_ext_dirty__18!(handle, inode, path.wrapping_offset(*__slate_slot_5 as isize));
if err != 0 {
break '__join_1;
}
}
}
}
} else {
break '__join_3;
}
}
__ext4_error_inode(inode, c"ext4_ext_correct_indexes".as_ptr() as *const u8, 1762u32, 0u64, 0i32, c"path[%d].p_idx %p > EXT_LAST_INDEX %p".as_ptr() as *const u8, *__slate_slot_5, (*path.wrapping_offset(*__slate_slot_5 as isize)).p_idx, EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(*__slate_slot_5 as isize)).p_hdr }));
err = 117i32.wrapping_neg();
}
loop {
let __v1: i32 = *__slate_slot_5;
let __v0: i32 = __v1.wrapping_add(1);
*__slate_slot_5 = __v0;
if __v0 < depth {
clear_buffer_verified((*path.wrapping_offset(*__slate_slot_5 as isize)).p_bh);
} else {
break;
}
}
// The path[k].p_bh is either unmodified or with no verified bit
// set (see ext4_ext_get_access()). So just clear the verified bit
// of the successfully modified extents buffers, which will force
// these extents to be checked to avoid using inconsistent data.
return err;
}
return 0;
}
}
}
}
}
}
}
}

fn ext4_can_extents_be_merged(inode: *mut inode, ex1: *mut ext4_extent, ex2: *mut ext4_extent) -> i32 {
let mut ext1_ee_len: u16 = 0;
let mut ext2_ee_len: u16 = 0;
if ext4_ext_is_unwritten(ex1) != ext4_ext_is_unwritten(ex2) {
return 0;
}
ext1_ee_len = ext4_ext_get_actual_len(ex1) as u16;
ext2_ee_len = ext4_ext_get_actual_len(ex2) as u16;
if unsafe { (*ex1).ee_block }.wrapping_add((ext1_ee_len as i32) as u32) != unsafe { (*ex2).ee_block } {
return 0;
}
if ((ext1_ee_len as i32).wrapping_add(ext2_ee_len as i32) as u64) > 32768u64 {
return 0;
}
if ext4_ext_is_unwritten(ex1) != 0 && ((ext1_ee_len as i32).wrapping_add(ext2_ee_len as i32) as u64) > 32767u64 {
return 0;
}
if ext4_ext_pblock(ex1).wrapping_add((ext1_ee_len as i32) as u64) == ext4_ext_pblock(ex2) {
return 1;
}
0
}

/// This function tries to merge the "ex" extent to the next extent in the tree.
/// It always tries to merge towards right. If you want to merge towards
/// left, pass "ex - 1" as argument instead of "ex".
/// Returns 0 if the extents (ex and ex+1) were _not_ merged and returns
/// 1 if they got merged.
fn ext4_ext_try_to_merge_right(inode: *mut inode, path: *mut ext4_ext_path, ex: *mut ext4_extent) -> i32 {
let mut eh: *mut ext4_extent_header = unsafe { core::mem::zeroed() };
let depth: u32 = ext_depth(inode) as u32;
let mut len: u32 = 0;
let mut merge_done: i32 = 0;
let mut unwritten: i32 = 0;
loop {
if unlikely!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("1516: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1516b - ., 3; .popsection\n# {0}", const 1516 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_try_to_merge_right_0, sym __slate_asm_string_ext4_ext_try_to_merge_right_1, const 1834 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
eh = unsafe { (*path.wrapping_offset(depth as isize)).p_hdr };
while ex < EXT_LAST_EXTENT!(eh) {
if ext4_can_extents_be_merged(inode, ex, ex.wrapping_offset(1 as isize)) == 0 {
break;
}
// merge with next extent!
unwritten = ext4_ext_is_unwritten(ex);
unsafe {
(*ex).ee_len = ext4_ext_get_actual_len(ex).wrapping_add(ext4_ext_get_actual_len(ex.wrapping_offset(1 as isize))) as u16;
}
if unwritten != 0 {
ext4_ext_mark_unwritten(ex);
}
if ex.wrapping_offset(1 as isize) < EXT_LAST_EXTENT!(eh) {
len = (((unsafe { EXT_LAST_EXTENT!(eh).offset_from(ex as *mut ext4_extent) }) as i64).wrapping_sub(1) as u64).wrapping_mul(core::mem::size_of::<ext4_extent>() as u64) as u32;
unsafe { memmove(ex.wrapping_offset(1 as isize) as *mut (), ex.wrapping_offset(2 as isize) as *const (), len as u64) };
}
le16_add_cpu(unsafe { core::ptr::addr_of_mut!((*eh).eh_entries) }, 1i32.wrapping_neg() as u16);
merge_done = 1;
let mut __v6: i64;
let __ret_warn_on: i32 = (((unsafe { (*eh).eh_entries }) as i32) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1517: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1517b - ., 3; .popsection\n# {0}", const 1517 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_ext_try_to_merge_right_7, sym __slate_asm_string_ext4_ext_try_to_merge_right_8, const 1854 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1518: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1518b - ., 4; .popsection\n# {0}", const 1518 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
if (unsafe { (*eh).eh_entries }) == 0 {
unsafe { __ext4_error_inode(inode, c"ext4_ext_try_to_merge_right".as_ptr() as *const u8, 1856u32, 0u64, 0i32, c"eh->eh_entries = 0!".as_ptr() as *const u8) };
}
}
merge_done
}

/// This function does a very simple check to see if we can collapse
/// an extent tree with a single extent tree leaf block into the inode.
fn ext4_ext_try_to_merge_up(handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path) {
let mut s: u64 = 0;
let max_root: u32 = ext4_ext_space_root(inode, 0i32) as u32;
let mut blk: u64 = 0;
if ((unsafe { (*path.wrapping_offset(0 as isize)).p_depth }) as i32) != 1 || ((unsafe { (*(*path.wrapping_offset(0 as isize)).p_hdr).eh_entries }) as i32) != 1 || (((unsafe { (*(*path.wrapping_offset(1 as isize)).p_hdr).eh_entries }) as i32) as u32) > max_root {
return;
}
// We need to modify the block allocation bitmap and the block
// group descriptor to release the extent tree block.  If we
// can't get the journal credits, give up.
if ext4_journal_extend(handle, 2i32, ext4_free_metadata_revoke_credits(unsafe { (*inode).i_sb }, 1i32)) != 0 {
return;
}
// Copy the extent data up to the inode
blk = ext4_idx_pblock(unsafe { (*path.wrapping_offset(0 as isize)).p_idx });
s = (((unsafe { (*(*path.wrapping_offset(1 as isize)).p_hdr).eh_entries }) as i32) as u64).wrapping_mul(core::mem::size_of::<ext4_extent_idx>() as u64);
s = s.wrapping_add(core::mem::size_of::<ext4_extent_header>() as u64);
unsafe {
(*path.wrapping_offset(1 as isize)).p_maxdepth = (*path.wrapping_offset(0 as isize)).p_maxdepth;
}
unsafe { memcpy((*path.wrapping_offset(0 as isize)).p_hdr as *mut (), (*path.wrapping_offset(1 as isize)).p_hdr as *const (), s) };
unsafe {
(*path.wrapping_offset(0 as isize)).p_depth = 0;
}
unsafe {
(*path.wrapping_offset(0 as isize)).p_ext = EXT_FIRST_EXTENT!(unsafe { (*path.wrapping_offset(0 as isize)).p_hdr }).wrapping_offset((unsafe { (*path.wrapping_offset(1 as isize)).p_ext }.offset_from(EXT_FIRST_EXTENT!(unsafe { (*path.wrapping_offset(1 as isize)).p_hdr }) as *mut ext4_extent) as i64) as isize);
}
unsafe {
(*(*path.wrapping_offset(0 as isize)).p_hdr).eh_max = max_root as u16;
}
ext4_ext_path_brelse(path.wrapping_offset(1 as isize));
unsafe { ext4_free_blocks(handle, inode, core::ptr::null_mut::<buffer_head>(), blk, 1u64, 3i32) };
}

/// This function tries to merge the @ex extent to neighbours in the tree, then
/// tries to collapse the extent tree into the inode.
fn ext4_ext_try_to_merge(handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path, ex: *mut ext4_extent) {
let depth: u32 = ext_depth(inode) as u32;
let mut merge_done: i32 = 0;
loop {
if unlikely!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("1519: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1519b - ., 3; .popsection\n# {0}", const 1519 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_try_to_merge_0, sym __slate_asm_string_ext4_ext_try_to_merge_1, const 1922 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
let eh: *mut ext4_extent_header = unsafe { (*path.wrapping_offset(depth as isize)).p_hdr };
if ex > EXT_FIRST_EXTENT!(eh) {
merge_done = ext4_ext_try_to_merge_right(inode, path, ex.wrapping_offset(-(1 as isize)));
}
if merge_done == 0 {
ext4_ext_try_to_merge_right(inode, path, ex);
}
ext4_ext_try_to_merge_up(handle, inode, path);
}

/// check if a portion of the "newext" extent overlaps with an
/// existing extent.
///
/// If there is an overlap discovered, it updates the length of the newext
/// such that there will be no overlap, and then returns 1.
/// If there is no overlap found, it returns 0.
fn ext4_ext_check_overlap(sbi: *mut ext4_sb_info, inode: *mut inode, newext: *mut ext4_extent, path: *mut ext4_ext_path) -> u32 {
let mut ret: u32 = unsafe { core::mem::zeroed() };
let mut len1: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(len1) as *mut u32;
let mut depth: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(depth) as *mut u32;
let mut b2: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(b2) as *mut u32;
let mut b1: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(b1) as *mut u32;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = 0;
*__slate_slot_4 = (*newext).ee_block;
*__slate_slot_1 = ext4_ext_get_actual_len(newext) as u32;
*__slate_slot_2 = ext_depth(inode) as u32;
if !unsafe { (*path.wrapping_offset(*__slate_slot_2 as isize)).p_ext }.is_null() {
*__slate_slot_3 = EXT4_LBLK_CMASK!(sbi, unsafe { (*(*path.wrapping_offset(*__slate_slot_2 as isize)).p_ext).ee_block });
// get the next allocated block if the extent in the path
// is before the requested block(s)
if *__slate_slot_3 < *__slate_slot_4 {
*__slate_slot_3 = ext4_ext_next_allocated_block(path);
if *__slate_slot_3 == 4294967295 {
break '__join_0;
} else {
*__slate_slot_3 = EXT4_LBLK_CMASK!(sbi, *__slate_slot_3);
}
}
// check for wrap through zero on extent logical start block
if (*__slate_slot_4).wrapping_add(*__slate_slot_1) < *__slate_slot_4 {
*__slate_slot_1 = 4294967295u32.wrapping_sub(*__slate_slot_4);
(*newext).ee_len = *__slate_slot_1 as u16;
ret = 1;
}
// check for overlap
if (*__slate_slot_4).wrapping_add(*__slate_slot_1) > *__slate_slot_3 {
(*newext).ee_len = (*__slate_slot_3).wrapping_sub(*__slate_slot_4) as u16;
ret = 1;
}
}
}
return ret;
}
}

/// ext4_ext_insert_extent:
/// tries to merge requested extent into the existing extent or
/// inserts requested extent as new one into the tree,
/// creating new leaf in the no-space case.
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_insert_extent(handle: *mut jbd2_journal_handle, inode: *mut inode, mut path: *mut ext4_ext_path, newext: *mut ext4_extent, gb_flags: i32) -> *mut ext4_ext_path {
let mut npath: core::mem::MaybeUninit<*mut ext4_ext_path> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut ext4_ext_path = core::ptr::addr_of_mut!(npath) as *mut *mut ext4_ext_path;
let mut __v18: bool = unsafe { core::mem::zeroed() };
// Try to see whether we should rather test the extent on
// right from ex, or from the left of ex. This is because
// ext4_find_extent() can return either extent on the
// left, or on the right from the searched position. This
// will make merging more effective.
let mut __v19: bool = unsafe { core::mem::zeroed() };
let mut unwritten: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i32 = core::ptr::addr_of_mut!(unwritten) as *mut i32;
let mut mb_flags: i32 = unsafe { core::mem::zeroed() };
let mut next: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u32 = core::ptr::addr_of_mut!(next) as *mut u32;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut i32 = core::ptr::addr_of_mut!(len) as *mut i32;
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32; // nearest extent
let mut nearex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut ext4_extent = core::ptr::addr_of_mut!(nearex) as *mut *mut ext4_extent;
let mut fex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut ext4_extent = core::ptr::addr_of_mut!(fex) as *mut *mut ext4_extent;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut eh: core::mem::MaybeUninit<*mut ext4_extent_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut *mut ext4_extent_header = core::ptr::addr_of_mut!(eh) as *mut *mut ext4_extent_header;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
err = 0;
let __slate_auto_bytes_7 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
mb_flags = 0;
let __slate_auto_bytes_8 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
while false {
}
if gb_flags & 4 != 0 {
mb_flags |= 1024;
}
'__join_0: {
if unlikely!(ext4_ext_get_actual_len(newext) == 0) != 0 {
__ext4_error_inode(inode, c"ext4_ext_insert_extent".as_ptr() as *const u8, 2009u32, 0u64, 0i32, c"ext4_ext_get_actual_len(newext) == 0".as_ptr() as *const u8);
err = 117i32.wrapping_neg();
} else {
*__slate_slot_27 = ext_depth(inode) as i32;
*__slate_slot_30 = (*path.wrapping_offset(*__slate_slot_27 as isize)).p_ext;
*__slate_slot_31 = (*path.wrapping_offset(*__slate_slot_27 as isize)).p_hdr;
if unlikely!(unsafe { (*path.wrapping_offset(*__slate_slot_27 as isize)).p_hdr }.is_null()) != 0 {
__ext4_error_inode(inode, c"ext4_ext_insert_extent".as_ptr() as *const u8, 2017u32, 0u64, 0i32, c"path[%d].p_hdr == NULL".as_ptr() as *const u8, *__slate_slot_27);
err = 117i32.wrapping_neg();
} else {
'__join_5: {
// try to insert block into found extent and return
if !(*__slate_slot_30).is_null() && gb_flags & 8 == 0 {
if *__slate_slot_30 < EXT_LAST_EXTENT!(*__slate_slot_31) {
__v19 = unsafe { (*(*__slate_slot_30)).ee_block }.wrapping_add(ext4_ext_get_actual_len(*__slate_slot_30) as u32) < (*newext).ee_block;
} else {
__v19 = false;
}
if __v19 {
let __v15: *mut ext4_extent = *__slate_slot_30;
*__slate_slot_30 = __v15.wrapping_offset(1 as isize);
} else {
if *__slate_slot_30 > EXT_FIRST_EXTENT!(*__slate_slot_31) {
__v18 = unsafe { (*newext).ee_block }.wrapping_add(ext4_ext_get_actual_len(newext) as u32) < (*(*__slate_slot_30)).ee_block;
} else {
__v18 = false;
}
if __v18 {
let __v17: *mut ext4_extent = *__slate_slot_30;
*__slate_slot_30 = __v17.wrapping_offset(-(1 as isize));
}
// Try to append newex to the ex
if ext4_can_extents_be_merged(inode, *__slate_slot_30, newext) != 0 {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_27 as isize));
if err != 0 {
break '__join_0;
} else {
*__slate_slot_22 = ext4_ext_is_unwritten(*__slate_slot_30);
(*(*__slate_slot_30)).ee_len = ext4_ext_get_actual_len(*__slate_slot_30).wrapping_add(ext4_ext_get_actual_len(newext)) as u16;
if *__slate_slot_22 != 0 {
ext4_ext_mark_unwritten(*__slate_slot_30);
}
*__slate_slot_28 = *__slate_slot_30;
break '__join_5;
}
}
}
if ext4_can_extents_be_merged(inode, newext, *__slate_slot_30) != 0 {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_27 as isize));
if err != 0 {
break '__join_0;
} else {
*__slate_slot_22 = ext4_ext_is_unwritten(*__slate_slot_30);
(*(*__slate_slot_30)).ee_block = (*newext).ee_block;
ext4_ext_store_pblock(*__slate_slot_30, ext4_ext_pblock(newext));
(*(*__slate_slot_30)).ee_len = ext4_ext_get_actual_len(*__slate_slot_30).wrapping_add(ext4_ext_get_actual_len(newext)) as u16;
if *__slate_slot_22 != 0 {
ext4_ext_mark_unwritten(*__slate_slot_30);
}
*__slate_slot_28 = *__slate_slot_30;
break '__join_5;
}
}
// Try to prepend newex to the ex
}
'__join_19: {
*__slate_slot_27 = ext_depth(inode) as i32;
*__slate_slot_31 = (*path.wrapping_offset(*__slate_slot_27 as isize)).p_hdr;
if ((*(*__slate_slot_31)).eh_entries as i32) >= ((*(*__slate_slot_31)).eh_max as i32) {
// probably next leaf has space for us?
*__slate_slot_29 = EXT_LAST_EXTENT!(*__slate_slot_31);
*__slate_slot_24 = 4294967295;
if (*newext).ee_block > (*(*__slate_slot_29)).ee_block {
*__slate_slot_24 = ext4_ext_next_leaf_block(path);
}
'__join_24: {
if *__slate_slot_24 != 4294967295 {
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_11 = ext4_find_extent(inode, *__slate_slot_24, core::ptr::null_mut::<ext4_ext_path>(), gb_flags);
if IS_ERR(*__slate_slot_11 as *const ()) {
err = PTR_ERR(*__slate_slot_11 as *const ()) as i32;
break '__join_0;
} else {
'__join_30: {
loop {
if unlikely!(((*(*__slate_slot_11)).p_depth as i32) != ((*path).p_depth as i32)) != 0 {
break '__join_30;
} else {
if true {
break;
}
}
}
*__slate_slot_31 = (*(*__slate_slot_11).wrapping_offset(*__slate_slot_27 as isize)).p_hdr;
if ((*(*__slate_slot_31)).eh_entries as i32) < ((*(*__slate_slot_31)).eh_max as i32) {
ext4_free_ext_path(path);
path = *__slate_slot_11;
break '__join_19;
} else {
ext4_free_ext_path(*__slate_slot_11);
break '__join_24;
}
}
core::arch::asm!("1520: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1520b - ., 3; .popsection\n# {0}", const 1520 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_insert_extent_34, sym __slate_asm_string_ext4_ext_insert_extent_35, const 2115 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}
// There is no free space in the found leaf.
// We're gonna add a new leaf in the tree.
if gb_flags & 32 != 0 {
mb_flags |= 8192;
}
path = ext4_ext_create_new_leaf(handle, inode, mb_flags as u32, gb_flags as u32, path, newext);
if IS_ERR(path as *const ()) {
return path;
} else {
*__slate_slot_27 = ext_depth(inode) as i32;
*__slate_slot_31 = (*path.wrapping_offset(*__slate_slot_27 as isize)).p_hdr;
}
}
}
*__slate_slot_28 = (*path.wrapping_offset(*__slate_slot_27 as isize)).p_ext;
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_27 as isize));
if err != 0 {
break '__join_0;
} else {
if (*__slate_slot_28).is_null() {
// there is no extent in this leaf, create first one
*__slate_slot_28 = EXT_FIRST_EXTENT!(*__slate_slot_31);
} else {
'__join_9: {
if (*newext).ee_block > (*(*__slate_slot_28)).ee_block {
// Insert after
let __v3: *mut ext4_extent = *__slate_slot_28;
*__slate_slot_28 = __v3.wrapping_offset(1 as isize);
} else {
// Insert before
'__join_14: {
loop {
if unlikely!((*newext).ee_block == (*(*__slate_slot_28)).ee_block) != 0 {
break '__join_14;
} else {
if true {
break;
}
}
}
break '__join_9;
}
core::arch::asm!("1521: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1521b - ., 3; .popsection\n# {0}", const 1521 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_insert_extent_32, sym __slate_asm_string_ext4_ext_insert_extent_33, const 2171 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
*__slate_slot_26 = (EXT_LAST_EXTENT!(*__slate_slot_31).offset_from(*__slate_slot_28 as *mut ext4_extent) as i64).wrapping_add(1) as i32;
if *__slate_slot_26 > 0 {
memmove((*__slate_slot_28).wrapping_offset(1 as isize) as *mut (), *__slate_slot_28 as *const (), (*__slate_slot_26 as u64).wrapping_mul(core::mem::size_of::<ext4_extent>() as u64));
}
}
le16_add_cpu(core::ptr::addr_of_mut!((*(*__slate_slot_31)).eh_entries), 1u16);
(*path.wrapping_offset(*__slate_slot_27 as isize)).p_ext = *__slate_slot_28;
(*(*__slate_slot_28)).ee_block = (*newext).ee_block;
ext4_ext_store_pblock(*__slate_slot_28, ext4_ext_pblock(newext));
(*(*__slate_slot_28)).ee_len = (*newext).ee_len;
}
}
if gb_flags & 8 == 0 {
ext4_ext_try_to_merge(handle, inode, path, *__slate_slot_28);
}
// try to merge extents
// time to correct all indexes above
err = ext4_ext_correct_indexes(handle, inode, path);
if err == 0 {
err = ext4_ext_dirty__14!(handle, inode, path.wrapping_offset(((*path).p_depth as i32) as isize));
if err == 0 {
return path;
}
}
}
}
}
ext4_free_ext_path(path);
return ERR_PTR(err as i64) as *mut ext4_ext_path;
}
}

fn ext4_fill_es_cache_info(inode: *mut inode, mut block: u32, num: u32, fieinfo: *mut fiemap_extent_info) -> i32 {
let mut next: u32 = 0;
let end: u32 = block.wrapping_add(num).wrapping_sub(1);
let mut es: extent_status = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(es) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let blksize_bits: u8 = unsafe { (*(*inode).i_sb).s_blocksize_bits };
let mut flags: u32 = 0;
let mut err: i32 = 0;
while block <= end {
next = 0;
flags = 0;
if (unsafe { ext4_es_lookup_extent(inode, block, core::ptr::addr_of_mut!(next), core::ptr::addr_of_mut!(es), core::ptr::null_mut::<u64>()) }) == 0 {
break;
}
if ext4_es_is_unwritten(unsafe { &(*core::ptr::addr_of_mut!(es)) }) != 0 {
flags |= 2048;
}
if ext4_es_is_delayed(core::ptr::addr_of_mut!(es)) != 0 {
flags |= 6u32;
}
if ext4_es_is_hole(unsafe { &(*core::ptr::addr_of_mut!(es)) }) != 0 {
flags |= 134217728;
}
if next == 0 {
flags |= 1;
}
if flags & 134217732u32 != 0 {
es.es_pblk = 0;
} else {
es.es_pblk = ext4_es_pblock(unsafe { &(*core::ptr::addr_of_mut!(es)) });
}
err = unsafe { fiemap_fill_next_extent(fieinfo, (es.es_lblk as u64) << (blksize_bits as i32), es.es_pblk << (blksize_bits as i32), (es.es_len as u64) << (blksize_bits as i32), flags) };
if next == 0 {
break;
}
block = next;
if err < 0 {
return err;
}
if err == 1 {
return 0;
}
}
0
}

/// ext4_ext_find_hole - find hole around given block according to the given path
/// @inode:	inode we lookup in
/// @path:	path in extent tree to @lblk
/// @lblk:	pointer to logical block around which we want to determine hole
///
/// Determine hole length (and start if easily possible) around given logical
/// block. We don't try too hard to find the beginning of the hole but @path
/// actually points to extent before @lblk, we provide it.
///
/// The function returns the length of a hole starting at @lblk. We update @lblk
/// to the beginning of the hole if we managed to find it.
fn ext4_ext_find_hole(inode: *mut inode, path: *mut ext4_ext_path, lblk: *mut u32) -> u32 {
let depth: i32 = ext_depth(inode) as i32;
let mut len: u32 = 0;
let ex: *mut ext4_extent = unsafe { (*path.wrapping_offset(depth as isize)).p_ext };
if ex.is_null() {
// there is no extent yet, so gap is [0;-]
unsafe {
*lblk = 0;
}
len = 4294967295;
} else {
if (unsafe { *lblk }) < unsafe { (*ex).ee_block } {
len = unsafe { (*ex).ee_block }.wrapping_sub(unsafe { *lblk });
} else {
if (unsafe { *lblk }) >= unsafe { (*ex).ee_block }.wrapping_add(ext4_ext_get_actual_len(ex) as u32) {
let mut next: u32 = 0;
unsafe {
*lblk = unsafe { (*ex).ee_block }.wrapping_add(ext4_ext_get_actual_len(ex) as u32);
}
next = ext4_ext_next_allocated_block(path);
loop {
if unlikely!(next == unsafe { *lblk }) != 0 {
loop {
unsafe {
core::arch::asm!("1522: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1522b - ., 3; .popsection\n# {0}", const 1522 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_find_hole_0, sym __slate_asm_string_ext4_ext_find_hole_1, const 2301 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
len = next.wrapping_sub(unsafe { *lblk });
} else {
loop {
unsafe {
core::arch::asm!("1523: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1523b - ., 3; .popsection\n# {0}", const 1523 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_find_hole_5, sym __slate_asm_string_ext4_ext_find_hole_6, const 2304 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
}
len
}

/// ext4_ext_rm_idx:
/// removes index from the index block.
fn ext4_ext_rm_idx(handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path, depth: i32) -> i32 {
let mut len: i32 = unsafe { core::mem::zeroed() };
let mut k: i32 = unsafe { core::mem::zeroed() };
let mut leaf: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(leaf) as *mut u64;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
k = depth.wrapping_sub(1);
// free index block
*__slate_slot_9 = ext4_idx_pblock((*path.wrapping_offset(k as isize)).p_idx);
if unlikely!(((*(*path.wrapping_offset(k as isize)).p_hdr).eh_entries as i32) == 0) != 0 {
__ext4_error_inode(inode, c"ext4_ext_rm_idx".as_ptr() as *const u8, 2323u32, 0u64, 0i32, c"path[%d].p_hdr->eh_entries == 0".as_ptr() as *const u8, k);
return 117i32.wrapping_neg();
} else {
*__slate_slot_10 = ext4_ext_get_access(handle, inode, path.wrapping_offset(k as isize));
if *__slate_slot_10 != 0 {
return *__slate_slot_10;
} else {
if (*path.wrapping_offset(k as isize)).p_idx != EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(k as isize)).p_hdr }) {
len = (EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(k as isize)).p_hdr }).offset_from((*path.wrapping_offset(k as isize)).p_idx as *mut ext4_extent_idx) as i64) as i32;
len = (len as u64).wrapping_mul(core::mem::size_of::<ext4_extent_idx>() as u64) as i32;
memmove((*path.wrapping_offset(k as isize)).p_idx as *mut (), unsafe { (*path.wrapping_offset(k as isize)).p_idx }.wrapping_offset(1 as isize) as *const (), len as u64);
}
le16_add_cpu(core::ptr::addr_of_mut!((*(*path.wrapping_offset(k as isize)).p_hdr).eh_entries), 1i32.wrapping_neg() as u16);
*__slate_slot_10 = ext4_ext_dirty__12!(handle, inode, path.wrapping_offset(k as isize));
if *__slate_slot_10 != 0 {
return *__slate_slot_10;
} else {
trace_ext4_ext_rm_idx(inode, *__slate_slot_9);
ext4_free_blocks(handle, inode, core::ptr::null_mut::<buffer_head>(), *__slate_slot_9, 1u64, 3i32);
'__join_3: {
loop {
let __v2: i32 = k.wrapping_sub(1);
k = __v2;
if __v2 >= 0 {
if (*path.wrapping_offset(k.wrapping_add(1) as isize)).p_idx != EXT_FIRST_INDEX!(unsafe { (*path.wrapping_offset(k.wrapping_add(1) as isize)).p_hdr }) {
break '__join_3;
} else {
*__slate_slot_10 = ext4_ext_get_access(handle, inode, path.wrapping_offset(k as isize));
if *__slate_slot_10 != 0 {
break;
} else {
(*(*path.wrapping_offset(k as isize)).p_idx).ei_block = (*(*path.wrapping_offset(k.wrapping_add(1) as isize)).p_idx).ei_block;
*__slate_slot_10 = ext4_ext_dirty__7!(handle, inode, path.wrapping_offset(k as isize));
if *__slate_slot_10 != 0 {
break;
}
}
}
} else {
break '__join_3;
}
}
loop {
let __v0: i32 = k.wrapping_add(1);
k = __v0;
if __v0 < depth {
clear_buffer_verified((*path.wrapping_offset(k as isize)).p_bh);
} else {
break;
}
}
// The path[k].p_bh is either unmodified or with no verified bit
// set (see ext4_ext_get_access()). So just clear the verified bit
// of the successfully modified extents buffers, which will force
// these extents to be checked to avoid using inconsistent data.
return *__slate_slot_10;
}
return 0;
}
}
}
}
}

/// ext4_ext_calc_credits_for_single_extent:
/// This routine returns max. credits that needed to insert an extent
/// to the extent tree.
/// When pass the actual path, the caller should calculate credits
/// under i_data_sem.
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_calc_credits_for_single_extent(inode: *mut inode, nrblocks: i32, path: *mut ext4_ext_path) -> i32 {
if !path.is_null() {
let depth: i32 = ext_depth(inode) as i32;
let mut ret: i32 = 0;
// probably there is space in leaf?
if ((unsafe { (*(*path.wrapping_offset(depth as isize)).p_hdr).eh_entries }) as i32) < ((unsafe { (*(*path.wrapping_offset(depth as isize)).p_hdr).eh_max }) as i32) {
// There are some space in the leaf tree, no
// need to account for leaf block credit
//
// bitmaps and block group descriptor blocks
// and other metadata blocks still need to be
// accounted.
//
// 1 bitmap, 1 block group descriptor
ret = 2u32.wrapping_add(EXT4_META_TRANS_BLOCKS!(unsafe { (*inode).i_sb })) as i32;
return ret;
}
}
unsafe { ext4_chunk_trans_blocks(inode, nrblocks) }
}

/// How many index/leaf blocks need to change/allocate to add @extents extents?
///
/// If we add a single extent, then in the worse case, each tree level
/// index/leaf need to be changed in case of the tree split.
///
/// If more extents are inserted, they could cause the whole tree split more
/// than once, but this is really rare.
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_index_trans_blocks(inode: *mut inode, extents: i32) -> i32 {
let mut index: i32 = 0;
// If we are converting the inline data, only one is needed here.
if ext4_has_inline_data(inode) != 0 {
return 1;
}
// Extent tree can change between the time we estimate credits and
// the time we actually modify the tree. Assume the worst case.
if extents <= 1 {
index = 10i32.wrapping_add(extents);
} else {
let ext_max: i32 = ext4_ext_space_block(inode, 0i32);
index = 15i32;
// Modified extents need not start at the beginning of the
// leaf. Already two extents may need two leaf block
// modifications...
index = index.wrapping_add(extents.wrapping_add(ext_max).wrapping_sub(1).wrapping_add(ext_max).wrapping_sub(1) / ext_max);
}
index
}

fn get_default_free_blocks_flags(inode: *mut inode) -> i32 {
let __v0: bool = S_ISDIR!(unsafe { (*inode).i_mode }) || S_ISLNK!(unsafe { (*inode).i_mode }) || ext4_test_inode_flag(inode, 21i32) != 0;
if __v0 {
return 1 | 2;
} else {
if ext4_should_journal_data(inode) != 0 {
return 2;
}
}
0
}

/// ext4_rereserve_cluster - increment the reserved cluster count when
///                          freeing a cluster with a pending reservation
///
/// @inode - file containing the cluster
/// @lblk - logical block in cluster to be reserved
///
/// Increments the reserved cluster count and adjusts quota in a bigalloc
/// file system when freeing a partial cluster containing at least one
/// delayed and unwritten block.  A partial cluster meeting that
/// requirement will have a pending reservation.  If so, the
/// RERESERVE_CLUSTER flag is used when calling ext4_free_blocks() to
/// defer reserved and allocated space accounting to a subsequent call
/// to this function.
fn ext4_rereserve_cluster(inode: *mut inode, lblk: u32) {
let sbi: *mut ext4_sb_info = EXT4_SB(unsafe { (*inode).i_sb });
let ei: *mut ext4_inode_info = EXT4_I!(inode);
dquot_reclaim_block(inode, EXT4_C2B!(sbi, 1i32) as i64);
spin_lock(unsafe { core::ptr::addr_of_mut!((*ei).i_block_reservation_lock) });
let __v2: u32 = unsafe { (*ei).i_reserved_data_blocks };
unsafe {
(*ei).i_reserved_data_blocks = __v2.wrapping_add(1);
}
percpu_counter_add(unsafe { core::ptr::addr_of_mut!((*sbi).s_dirtyclusters_counter) }, 1i64);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ei).i_block_reservation_lock) });
percpu_counter_add(unsafe { core::ptr::addr_of_mut!((*sbi).s_freeclusters_counter) }, 1i64);
unsafe { ext4_remove_pending(inode, lblk) };
}

fn ext4_remove_blocks(handle: *mut jbd2_journal_handle, inode: *mut inode, ex: *mut ext4_extent, partial: *mut partial_cluster, from: u32, to: u32) -> i32 {
let sbi: *mut ext4_sb_info = EXT4_SB(unsafe { (*inode).i_sb });
let ee_len: u16 = ext4_ext_get_actual_len(ex) as u16;
let mut last_pblk: u64 = 0;
let mut pblk: u64 = 0;
let mut num: u32 = 0;
let mut flags: i32 = 0;
// only extent tail removal is allowed
if from < unsafe { (*ex).ee_block } || to != unsafe { (*ex).ee_block }.wrapping_add((ee_len as i32) as u32).wrapping_sub(1) {
unsafe { __ext4_error((*sbi).s_sb, c"ext4_remove_blocks".as_ptr() as *const u8, 2502u32, false, 0i32, 0u64, c"strange request: removal(2) %u-%u from %u:%u".as_ptr() as *const u8, from, to, (*ex).ee_block, ee_len as i32) };
return 0;
}
trace_ext4_remove_blocks(inode, ex, from, to as u64, partial);
// if we have a partial cluster, and it's different from the
// cluster of the last block in the extent, we free it
last_pblk = ext4_ext_pblock(ex).wrapping_add((ee_len as i32) as u64).wrapping_sub(1);
if ((unsafe { (*partial).state }) as u32) != 0 && (unsafe { (*partial).pclu }) != EXT4_B2C!(sbi, last_pblk) {
if ((unsafe { (*partial).state }) as u32) == 1 {
flags = get_default_free_blocks_flags(inode);
if unsafe { ext4_is_pending(inode, (*partial).lblk) } {
flags |= 64;
}
unsafe { ext4_free_blocks(handle, inode, core::ptr::null_mut::<buffer_head>(), EXT4_C2B!(sbi, unsafe { (*partial).pclu }), (*sbi).s_cluster_ratio as u64, flags) };
if flags & 64 != 0 {
ext4_rereserve_cluster(inode, unsafe { (*partial).lblk });
}
}
unsafe {
(*partial).state = 0;
}
}
num = unsafe { (*ex).ee_block }.wrapping_add((ee_len as i32) as u32).wrapping_sub(from);
pblk = ext4_ext_pblock(ex).wrapping_add((ee_len as i32) as u64).wrapping_sub(num as u64);
// We free the partial cluster at the end of the extent (if any),
// unless the cluster is used by another extent (partial_cluster
// state is nofree).  If a partial cluster exists here, it must be
// shared with the last block in the extent.
flags = get_default_free_blocks_flags(inode);
// partial, left end cluster aligned, right end unaligned
if EXT4_LBLK_COFF!(sbi, to) != unsafe { (*sbi).s_cluster_ratio }.wrapping_sub(1) && EXT4_LBLK_CMASK!(sbi, to) >= from && ((unsafe { (*partial).state }) as u32) != 2 {
if unsafe { ext4_is_pending(inode, to) } {
flags |= 64;
}
unsafe { ext4_free_blocks(handle, inode, core::ptr::null_mut::<buffer_head>(), EXT4_PBLK_CMASK!(sbi, last_pblk), (*sbi).s_cluster_ratio as u64, flags) };
if flags & 64 != 0 {
ext4_rereserve_cluster(inode, to);
}
unsafe {
(*partial).state = 0;
}
flags = get_default_free_blocks_flags(inode);
}
flags |= 32;
// For bigalloc file systems, we never free a partial cluster
// at the beginning of the extent.  Instead, we check to see if we
// need to free it on a subsequent call to ext4_remove_blocks,
// or at the end of ext4_ext_rm_leaf or ext4_ext_remove_space.
flags |= 16;
unsafe { ext4_free_blocks(handle, inode, core::ptr::null_mut::<buffer_head>(), pblk, num as u64, flags) };
// reset the partial cluster if we've freed past it
if ((unsafe { (*partial).state }) as u32) != 0 && (unsafe { (*partial).pclu }) != EXT4_B2C!(sbi, pblk) {
unsafe {
(*partial).state = 0;
}
}
// If we've freed the entire extent but the beginning is not left
// cluster aligned and is not marked as ineligible for freeing we
// record the partial cluster at the beginning of the extent.  It
// wasn't freed by the preceding ext4_free_blocks() call, and we
// need to look farther to the left to determine if it's to be freed
// (not shared with another extent). Else, reset the partial
// cluster - we're either  done freeing or the beginning of the
// extent is left cluster aligned.
if EXT4_LBLK_COFF!(sbi, from) != 0 && num == ((ee_len as i32) as u32) {
if ((unsafe { (*partial).state }) as u32) == 0 {
unsafe {
(*partial).pclu = EXT4_B2C!(sbi, pblk);
}
unsafe {
(*partial).lblk = from;
}
unsafe {
(*partial).state = 1;
}
}
} else {
unsafe {
(*partial).state = 0;
}
}
0
}

// ext4_ext_rm_leaf() Removes the extents associated with the
// blocks appearing between "start" and "end".  Both "start"
// and "end" must appear in the same extent or EIO is returned.
//
// @handle: The journal handle
// @inode:  The files inode
// @path:   The path to the leaf
// @partial_cluster: The cluster which we'll have to free if all extents
//                   has been released from it.  However, if this value is
//                   negative, it's a cluster just to the right of the
//                   punched region and it must not be freed.
// @start:  The first block to remove
// @end:   The last block to remove
fn ext4_ext_rm_leaf(handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path, partial: *mut partial_cluster, start: u32, end: u32) -> i32 {
let mut flags: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__1528: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__1527: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__1525: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__1524: u32 = unsafe { core::mem::zeroed() };
let mut pblk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u64 = core::ptr::addr_of_mut!(pblk) as *mut u64;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut unwritten: u32 = unsafe { core::mem::zeroed() };
let mut ex_ee_len: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u16 = core::ptr::addr_of_mut!(ex_ee_len) as *mut u16;
let mut ex_ee_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(ex_ee_block) as *mut u32;
let mut num: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(num) as *mut u32;
let mut b: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut u32 = core::ptr::addr_of_mut!(b) as *mut u32;
let mut a: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut u32 = core::ptr::addr_of_mut!(a) as *mut u32;
let mut eh: core::mem::MaybeUninit<*mut ext4_extent_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut *mut ext4_extent_header = core::ptr::addr_of_mut!(eh) as *mut *mut ext4_extent_header;
let mut revoke_credits: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut i32 = core::ptr::addr_of_mut!(revoke_credits) as *mut i32;
let mut credits: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut i32 = core::ptr::addr_of_mut!(credits) as *mut i32;
let mut depth: i32 = unsafe { core::mem::zeroed() };
let mut correct_index: i32 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
unsafe {
sbi = EXT4_SB((*inode).i_sb);
err = 0;
correct_index = 0;
depth = ext_depth(inode) as i32;
let __slate_auto_bytes = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 2usize);
unwritten = 0;
let __slate_auto_bytes_8 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// the header must be checked already in ext4_ext_remove_space()
if unsafe { (*path.wrapping_offset(depth as isize)).p_hdr }.is_null() {
(*path.wrapping_offset(depth as isize)).p_hdr = ext_block_hdr((*path.wrapping_offset(depth as isize)).p_bh);
}
*__slate_slot_31 = (*path.wrapping_offset(depth as isize)).p_hdr;
if unlikely!(unsafe { (*path.wrapping_offset(depth as isize)).p_hdr }.is_null()) != 0 {
__ext4_error_inode(inode, c"ext4_ext_rm_leaf".as_ptr() as *const u8, 2645u32, 0u64, 0i32, c"path[%d].p_hdr == NULL".as_ptr() as *const u8, depth);
return 117i32.wrapping_neg();
} else {
// find where to start removing
*__slate_slot_24 = (*path.wrapping_offset(depth as isize)).p_ext;
if (*__slate_slot_24).is_null() {
*__slate_slot_24 = EXT_LAST_EXTENT!(*__slate_slot_31);
}
*__slate_slot_27 = (*(*__slate_slot_24)).ee_block;
*__slate_slot_26 = ext4_ext_get_actual_len(*__slate_slot_24) as u16;
trace_ext4_ext_rm_leaf(inode, start, *__slate_slot_24, partial);
'__join_0: {
'__join_33: {
'__join_26: {
while *__slate_slot_24 >= EXT_FIRST_EXTENT!(*__slate_slot_31) && (*__slate_slot_27).wrapping_add((*__slate_slot_26 as i32) as u32) > start {
if ext4_ext_is_unwritten(*__slate_slot_24) != 0 {
unwritten = 1;
} else {
unwritten = 0;
}
(*path.wrapping_offset(depth as isize)).p_ext = *__slate_slot_24;
__UNIQUE_ID_x__1524 = *__slate_slot_27;
__UNIQUE_ID_y__1525 = start;
while false {
}
*__slate_slot_30 = if __UNIQUE_ID_x__1524 > __UNIQUE_ID_y__1525 { __UNIQUE_ID_x__1524 } else { __UNIQUE_ID_y__1525 };
__UNIQUE_ID_x__1527 = (*__slate_slot_27).wrapping_add((*__slate_slot_26 as i32) as u32).wrapping_sub(1);
__UNIQUE_ID_y__1528 = end;
while false {
}
*__slate_slot_29 = if __UNIQUE_ID_x__1527 < __UNIQUE_ID_y__1528 { __UNIQUE_ID_x__1527 } else { __UNIQUE_ID_y__1528 };
// If this extent is beyond the end of the hole, skip it
if end < *__slate_slot_27 {
// We're going to skip this extent and move to another,
// so note that its first cluster is in use to avoid
// freeing it when removing blocks.  Eventually, the
// right edge of the truncated/punched region will
// be just to the left.
if (*sbi).s_cluster_ratio > 1 {
*__slate_slot_23 = ext4_ext_pblock(*__slate_slot_24);
(*partial).pclu = EXT4_B2C!(sbi, *__slate_slot_23);
(*partial).state = 2;
}
let __v13: *mut ext4_extent = *__slate_slot_24;
*__slate_slot_24 = __v13.wrapping_offset(-(1 as isize));
*__slate_slot_27 = (*(*__slate_slot_24)).ee_block;
*__slate_slot_26 = ext4_ext_get_actual_len(*__slate_slot_24) as u16;
} else {
if *__slate_slot_29 != (*__slate_slot_27).wrapping_add((*__slate_slot_26 as i32) as u32).wrapping_sub(1) {
break '__join_33;
} else {
if *__slate_slot_30 != *__slate_slot_27 {
// remove tail of the extent
*__slate_slot_28 = (*__slate_slot_30).wrapping_sub(*__slate_slot_27);
} else {
// remove whole extent: excellent!
*__slate_slot_28 = 0;
}
// 3 for leaf, sb, and inode plus 2 (bmap and group
// descriptor) for each block group; assume two block
// groups plus ex_ee_len/blocks_per_block_group for
// the worst case
*__slate_slot_33 = 7u64.wrapping_add(2u64.wrapping_mul(((*__slate_slot_26 as i32) as u64) / EXT4_BLOCKS_PER_GROUP!(unsafe { (*inode).i_sb }))) as i32;
if *__slate_slot_24 == EXT_FIRST_EXTENT!(*__slate_slot_31) {
correct_index = 1;
let __v11: i32 = (ext_depth(inode) as i32).wrapping_add(1);
let __v10: i32 = *__slate_slot_33;
*__slate_slot_33 = __v10.wrapping_add(__v11);
}
let __v8: i32 = EXT4_MAXQUOTAS_TRANS_BLOCKS!(unsafe { (*inode).i_sb });
let __v7: i32 = *__slate_slot_33;
*__slate_slot_33 = __v7.wrapping_add(__v8);
// We may end up freeing some index blocks and data from the
// punched range. Note that partial clusters are accounted for
// by ext4_free_data_revoke_credits().
*__slate_slot_32 = ext4_free_metadata_revoke_credits((*inode).i_sb, ext_depth(inode) as i32).wrapping_add(ext4_free_data_revoke_credits(inode, (*__slate_slot_29).wrapping_sub(*__slate_slot_30).wrapping_add(1) as i32));
err = ext4_datasem_ensure_credits(handle, inode, *__slate_slot_33, *__slate_slot_33, *__slate_slot_32);
if err != 0 {
break '__join_26;
} else {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(depth as isize));
if err != 0 {
break '__join_0;
} else {
err = ext4_remove_blocks(handle, inode, *__slate_slot_24, partial, *__slate_slot_30, *__slate_slot_29);
if err != 0 {
break '__join_0;
} else {
if *__slate_slot_28 == 0 {
ext4_ext_store_pblock(*__slate_slot_24, 0u64);
}
// this extent is removed; mark slot entirely unused
(*(*__slate_slot_24)).ee_len = *__slate_slot_28 as u16;
// Do not mark unwritten if all the blocks in the
// extent have been removed.
if unwritten != 0 && *__slate_slot_28 != 0 {
ext4_ext_mark_unwritten(*__slate_slot_24);
}
// If the extent was completely released,
// we need to remove it from the leaf
if *__slate_slot_28 == 0 {
if end != 4294967294u32 {
// For hole punching, we need to scoot all the
// extents up when an extent is removed so that
// we dont have blank extents in the middle
memmove(*__slate_slot_24 as *mut (), (*__slate_slot_24).wrapping_offset(1 as isize) as *const (), ((EXT_LAST_EXTENT!(*__slate_slot_31).offset_from(*__slate_slot_24 as *mut ext4_extent) as i64) as u64).wrapping_mul(core::mem::size_of::<ext4_extent>() as u64));
// Now get rid of the one at the end
memset(EXT_LAST_EXTENT!(*__slate_slot_31) as *mut (), 0i32, core::mem::size_of::<ext4_extent>() as u64);
}
le16_add_cpu(core::ptr::addr_of_mut!((*(*__slate_slot_31)).eh_entries), 1i32.wrapping_neg() as u16);
}
err = ext4_ext_dirty__21!(handle, inode, path.wrapping_offset(depth as isize));
if err != 0 {
break '__join_0;
} else {
let __v4: *mut ext4_extent = *__slate_slot_24;
*__slate_slot_24 = __v4.wrapping_offset(-(1 as isize));
*__slate_slot_27 = (*(*__slate_slot_24)).ee_block;
*__slate_slot_26 = ext4_ext_get_actual_len(*__slate_slot_24) as u16;
}
}
}
}
}
}
}
if correct_index != 0 && (*(*__slate_slot_31)).eh_entries != 0 {
err = ext4_ext_correct_indexes(handle, inode, path);
}
// If there's a partial cluster and at least one extent remains in
// the leaf, free the partial cluster if it isn't shared with the
// current extent.  If it is shared with the current extent
// we reset the partial cluster because we've reached the start of the
// truncated/punched region and we're done removing blocks.
if ((*partial).state as u32) == 1 && *__slate_slot_24 >= EXT_FIRST_EXTENT!(*__slate_slot_31) {
*__slate_slot_23 = ext4_ext_pblock(*__slate_slot_24).wrapping_add((*__slate_slot_26 as i32) as u64).wrapping_sub(1);
if (*partial).pclu != EXT4_B2C!(sbi, *__slate_slot_23) {
flags = get_default_free_blocks_flags(inode);
if ext4_is_pending(inode, (*partial).lblk) {
flags |= 64;
}
ext4_free_blocks(handle, inode, core::ptr::null_mut::<buffer_head>(), EXT4_C2B!(sbi, unsafe { (*partial).pclu }), (*sbi).s_cluster_ratio as u64, flags);
if flags & 64 != 0 {
ext4_rereserve_cluster(inode, (*partial).lblk);
}
}
(*partial).state = 0;
}
// if this leaf is free, then we should
// remove it from index block above
if err == 0 && ((*(*__slate_slot_31)).eh_entries as i32) == 0 && !unsafe { (*path.wrapping_offset(depth as isize)).p_bh }.is_null() {
err = ext4_ext_rm_idx(handle, inode, path, depth);
break '__join_0;
} else {
break '__join_0;
}
}
if err > 0 {
err = 11i32.wrapping_neg();
break '__join_0;
} else {
break '__join_0;
}
}
__ext4_error_inode(inode, c"ext4_ext_rm_leaf".as_ptr() as *const u8, 2698u32, 0u64, 0i32, c"can not handle truncate %u:%u on extent %u:%u".as_ptr() as *const u8, start, end, *__slate_slot_27, (*__slate_slot_27).wrapping_add((*__slate_slot_26 as i32) as u32).wrapping_sub(1));
err = 117i32.wrapping_neg();
}
return err;
}
}
}

/// ext4_ext_more_to_rm:
/// returns 1 if current index has to be freed (even partial)
fn ext4_ext_more_to_rm(path: *mut ext4_ext_path) -> i32 {
loop {
if unlikely!(unsafe { (*path).p_idx }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("1530: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1530b - ., 3; .popsection\n# {0}", const 1530 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_more_to_rm_0, sym __slate_asm_string_ext4_ext_more_to_rm_1, const 2831 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if (unsafe { (*path).p_idx }) < EXT_FIRST_INDEX!(unsafe { (*path).p_hdr }) {
return 0;
}
// if truncate on deeper level happened, it wasn't partial,
// so we have to consider current index for truncation
if (((unsafe { (*(*path).p_hdr).eh_entries }) as i32) as u64) == unsafe { (*path).p_block } {
return 0;
}
1
}

#[unsafe(no_mangle)]
extern "C" fn ext4_ext_remove_space(inode: *mut inode, start: u32, end: u32) -> i32 {
let mut __slate_compound_49: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_51: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut flags: i32 = unsafe { core::mem::zeroed() };
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut k: i32 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut pblk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut u64 = core::ptr::addr_of_mut!(pblk) as *mut u64;
let mut lblk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u32 = core::ptr::addr_of_mut!(lblk) as *mut u32;
let mut ex_end: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u32 = core::ptr::addr_of_mut!(ex_end) as *mut u32;
let mut ee_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut u32 = core::ptr::addr_of_mut!(ee_block) as *mut u32;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut flags_1: i32 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
let mut partial: core::mem::MaybeUninit<partial_cluster> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut partial_cluster = core::ptr::addr_of_mut!(partial) as *mut partial_cluster;
let mut path: *mut ext4_ext_path = unsafe { core::mem::zeroed() };
let mut depth: i32 = unsafe { core::mem::zeroed() };
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
unsafe {
sbi = EXT4_SB((*inode).i_sb);
depth = ext_depth(inode) as i32;
path = core::ptr::null_mut::<ext4_ext_path>();
let __slate_auto_bytes_6 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_7 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
i = 0;
err = 0;
flags_1 = 1073741824 | 268435456;
(*__slate_slot_42).pclu = 0;
(*__slate_slot_42).lblk = 0;
(*__slate_slot_42).state = 0;
// probably first extent we're gonna free will be last in block
*__slate_slot_41 = ext4_journal_start_with_revoke!(inode, 5i32, depth.wrapping_add(1), ext4_free_metadata_revoke_credits((*inode).i_sb, depth));
if IS_ERR(*__slate_slot_41 as *const ()) {
return PTR_ERR(*__slate_slot_41 as *const ()) as i32;
} else {
loop {
'__join_1: {
trace_ext4_ext_remove_space(inode, start, end, depth);
// Check if we are removing extents inside the extent tree. If that
// is the case, we are going to punch a hole inside the extent tree
// so we have to check whether we need to split the extent covering
// the last block to remove so we can easily remove the part of it
// in ext4_ext_rm_leaf().
if end < 4294967294u32 {
let __slate_auto_bytes_2 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// find extent for or closest extent to this block
path = ext4_find_extent(inode, end, core::ptr::null_mut::<ext4_ext_path>(), flags_1);
if IS_ERR(path as *const ()) {
ext4_journal_stop__79!(*__slate_slot_41);
return PTR_ERR(path as *const ()) as i32;
} else {
depth = ext_depth(inode) as i32;
// Leaf not may not exist only if inode has no blocks at all
*__slate_slot_36 = (*path.wrapping_offset(depth as isize)).p_ext;
if (*__slate_slot_36).is_null() {
if depth != 0 {
__ext4_error_inode(inode, c"ext4_ext_remove_space".as_ptr() as *const u8, 2897u32, 0u64, 0i32, c"path[%d].p_hdr == NULL".as_ptr() as *const u8, depth);
err = 117i32.wrapping_neg();
break '__join_1;
} else {
break '__join_1;
}
} else {
*__slate_slot_35 = (*(*__slate_slot_36)).ee_block;
*__slate_slot_34 = (*__slate_slot_35).wrapping_add(ext4_ext_get_actual_len(*__slate_slot_36) as u32).wrapping_sub(1);
// See if the last block is inside the extent, if so split
// the extent at 'end' block so we can easily remove the
// tail of the first part of the split extent in
// ext4_ext_rm_leaf().
if end >= *__slate_slot_35 && end < *__slate_slot_34 {
// If we're going to split the extent, note that
// the cluster containing the block after 'end' is
// in use to avoid freeing it when removing blocks.
if (*sbi).s_cluster_ratio > 1 {
*__slate_slot_32 = ext4_ext_pblock(*__slate_slot_36).wrapping_add(end as u64).wrapping_sub(*__slate_slot_35 as u64).wrapping_add(1);
(*__slate_slot_42).pclu = EXT4_B2C!(sbi, *__slate_slot_32);
(*__slate_slot_42).state = 2;
}
// Split the extent in two so that 'end' is the last
// block in the first new extent. Also we should not
// fail removing space due to ENOSPC so try to use
// reserved block if that happens.
path = ext4_force_split_extent_at(*__slate_slot_41, inode, path, end.wrapping_add(1), 1i32);
if IS_ERR(path as *const ()) {
err = PTR_ERR(path as *const ()) as i32;
break '__join_1;
}
} else {
if (*sbi).s_cluster_ratio > 1 && end >= *__slate_slot_34 && ((*__slate_slot_42).state as u32) == 0 {
// If we're punching, there's an extent to the right.
// If the partial cluster hasn't been set, set it to
// that extent's first cluster and its state to nofree
// so it won't be freed should it contain blocks to be
// removed. If it's already set (tofree/nofree), we're
// retrying and keep the original partial cluster info
// so a cluster marked tofree as a result of earlier
// extent removal is not lost.
*__slate_slot_33 = (*__slate_slot_34).wrapping_add(1);
err = ext4_ext_search_right(inode, path, core::ptr::addr_of_mut!(*__slate_slot_33), core::ptr::addr_of_mut!(*__slate_slot_32), core::ptr::null_mut::<ext4_extent>(), flags_1);
if err < 0 {
break '__join_1;
} else {
if *__slate_slot_32 != 0 {
(*__slate_slot_42).pclu = EXT4_B2C!(sbi, *__slate_slot_32);
(*__slate_slot_42).state = 2;
}
}
}
}
}
}
}
// We start scanning from right side, freeing all the blocks
// after i_size and walking into the tree depth-wise.
depth = ext_depth(inode) as i32;
if !path.is_null() {
let __v23: i32 = depth;
i = __v23;
k = __v23;
loop {
let __v21: i32 = k.wrapping_sub(1);
k = __v21;
if __v21 > 0 {
(*path.wrapping_offset(k as isize)).p_block = ((*(*path.wrapping_offset(k as isize)).p_hdr).eh_entries as i32).wrapping_add(1) as u64;
} else {
break;
}
}
} else {
__obj_size = size_mul(core::mem::size_of::<ext4_ext_path>() as u64, depth.wrapping_add(1) as u64);
let __slate_auto_bytes_1 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_26 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_27 = _kzalloc_noprof(__obj_size, 35904u32, *{
__slate_compound_49 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_49)
});
alloc_tag_restore!();
} else {
*__slate_slot_27 = _kzalloc_noprof(__obj_size, 35904u32, *{
__slate_compound_51 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_51)
});
}
path = *__slate_slot_27 as *mut ext4_ext_path;
let __v25: u16 = depth as u16;
(*path.wrapping_offset(0 as isize)).p_depth = __v25;
(*path.wrapping_offset(0 as isize)).p_maxdepth = __v25;
(*path.wrapping_offset(0 as isize)).p_hdr = ext_inode_hdr(inode);
i = 0;
if ext4_ext_check__2!(inode, unsafe { (*path.wrapping_offset(0 as isize)).p_hdr }, depth, 0i32) != 0 {
err = 117i32.wrapping_neg();
break '__join_1;
}
}
err = 0;
'__join_10: {
'__join_19: {
loop {
if i >= 0 && err == 0 {
if i == depth {
// this is leaf block
err = ext4_ext_rm_leaf(*__slate_slot_41, inode, path, core::ptr::addr_of_mut!(*__slate_slot_42), start, end);
// root level has p_bh == NULL, brelse() eats this
ext4_ext_path_brelse(path.wrapping_offset(i as isize));
i = i.wrapping_sub(1);
} else {
// this is index block
if unsafe { (*path.wrapping_offset(i as isize)).p_hdr }.is_null() {
(*path.wrapping_offset(i as isize)).p_hdr = ext_block_hdr((*path.wrapping_offset(i as isize)).p_bh);
}
if unsafe { (*path.wrapping_offset(i as isize)).p_idx }.is_null() {
// this level hasn't been touched yet
(*path.wrapping_offset(i as isize)).p_idx = EXT_LAST_INDEX!(unsafe { (*path.wrapping_offset(i as isize)).p_hdr });
(*path.wrapping_offset(i as isize)).p_block = ((*(*path.wrapping_offset(i as isize)).p_hdr).eh_entries as i32).wrapping_add(1) as u64;
} else {
// we were already here, see at next index
let __v17: *mut ext4_ext_path = path.wrapping_offset(i as isize);
let __v16: *mut ext4_extent_idx = (*__v17).p_idx;
(*__v17).p_idx = __v16.wrapping_offset(-(1 as isize));
}
if ext4_ext_more_to_rm(path.wrapping_offset(i as isize)) != 0 {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// go to the next level
memset(path.wrapping_offset(i as isize).wrapping_offset(1 as isize) as *mut (), 0i32, core::mem::size_of::<ext4_ext_path>() as u64);
*__slate_slot_9 = read_extent_tree_block__3!(inode, unsafe { (*path.wrapping_offset(i as isize)).p_idx }, depth.wrapping_sub(i).wrapping_sub(1), flags_1);
if IS_ERR(*__slate_slot_9 as *const ()) {
break '__join_19;
} else {
// Yield here to deal with large extent trees.
// Should be a no-op if we did IO above.
cond_resched__8!();
__ret_warn_on = (i.wrapping_add(1) > depth) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("1531: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1531b - ., 3; .popsection\n# {0}", const 1531 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_ext_remove_space_46, sym __slate_asm_string_ext4_ext_remove_space_47, const 3032 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1532: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1532b - ., 4; .popsection\n# {0}", const 1532 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if __ret_warn_on != 0 {
break;
} else {
(*path.wrapping_offset(i.wrapping_add(1) as isize)).p_bh = *__slate_slot_9;
// save actual number of indexes since this
// number is changed at the next iteration
(*path.wrapping_offset(i as isize)).p_block = (*(*path.wrapping_offset(i as isize)).p_hdr).eh_entries as u64;
i = i.wrapping_add(1);
}
}
} else {
// we finished processing this index, go up
if ((*(*path.wrapping_offset(i as isize)).p_hdr).eh_entries as i32) == 0 && i > 0 {
// index is empty, remove it;
// handle must be already prepared by the
// truncatei_leaf()
err = ext4_ext_rm_idx(*__slate_slot_41, inode, path, i);
}
// root level has p_bh == NULL, brelse() eats this
ext4_ext_path_brelse(path.wrapping_offset(i as isize));
i = i.wrapping_sub(1);
}
}
} else {
break '__join_10;
}
}
err = 117i32.wrapping_neg();
break '__join_10;
}
// should we reset i_size?
err = PTR_ERR(*__slate_slot_9 as *const ()) as i32;
}
trace_ext4_ext_remove_space_done(inode, start, end, depth, core::ptr::addr_of_mut!(*__slate_slot_42), (*(*path).p_hdr).eh_entries);
// if there's a partial cluster and we have removed the first extent
// in the file, then we also free the partial cluster, if any
if ((*__slate_slot_42).state as u32) == 1 && err == 0 {
flags = get_default_free_blocks_flags(inode);
if ext4_is_pending(inode, (*__slate_slot_42).lblk) {
flags |= 64;
}
ext4_free_blocks(*__slate_slot_41, inode, core::ptr::null_mut::<buffer_head>(), EXT4_C2B!(sbi, (*__slate_slot_42).pclu), (*sbi).s_cluster_ratio as u64, flags);
if flags & 64 != 0 {
ext4_rereserve_cluster(inode, (*__slate_slot_42).lblk);
}
(*__slate_slot_42).state = 0;
}
// TODO: flexible tree reduction should be here
if ((*(*path).p_hdr).eh_entries as i32) == 0 {
// truncate to zero freed all the tree,
// so we need to correct eh_depth
err = ext4_ext_get_access(*__slate_slot_41, inode, path);
if err == 0 {
(*ext_inode_hdr(inode)).eh_depth = 0;
(*ext_inode_hdr(inode)).eh_max = ext4_ext_space_root(inode, 0i32) as u16;
err = ext4_ext_dirty__11!(*__slate_slot_41, inode, path);
}
}
}
ext4_free_ext_path(path);
path = core::ptr::null_mut::<ext4_ext_path>();
if err != 11i32.wrapping_neg() {
break;
}
}
ext4_journal_stop__4!(*__slate_slot_41);
return err;
}
}
}

/// called at mount time
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_init(sb: *mut super_block) {
// possible initialization would be here
if ext4_has_feature_extents(sb) {
}
}

/// called at umount time
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_release(sb: *mut super_block) {
if !ext4_has_feature_extents(sb) {
return;
}
}

fn ext4_zeroout_es(inode: *mut inode, ex: *mut ext4_extent) {
let ee_block: u32 = unsafe { (*ex).ee_block };
let mut ee_pblock: u64 = 0;
let ee_len: u32 = ext4_ext_get_actual_len(ex) as u32;
ee_pblock = ext4_ext_pblock(ex);
if ee_len == 0 {
return;
}
unsafe { ext4_es_insert_extent(inode, ee_block, ee_len, ee_pblock, 1u32, false) };
}

/// FIXME!! we need to try to merge to left or right after zero-out
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_zeroout(inode: *mut inode, ex: *mut ext4_extent) -> i32 {
let mut ee_pblock: u64 = 0;
let mut ee_len: u32 = 0;
while false {
}
ee_len = ext4_ext_get_actual_len(ex) as u32;
ee_pblock = ext4_ext_pblock(ex);
unsafe { ext4_issue_zeroout(inode, (*ex).ee_block, ee_pblock, ee_len) }
}

/// ext4_split_extent_at() splits an extent at given block.
///
/// @handle: the journal handle
/// @inode: the file inode
/// @path: the path to the extent
/// @split: the logical block where the extent is splitted.
/// @flags: flags used to insert new extent to extent tree.
///
///
/// Splits extent [a, b] into two extents [a, @split) and [@split, b], states
/// of which are same as the original extent. No conversion is performed.
///
/// Return an extent path pointer on success, or an error pointer on failure. On
/// failure, the extent is restored to original state.
fn ext4_split_extent_at(handle: *mut jbd2_journal_handle, inode: *mut inode, mut path: *mut ext4_ext_path, split: u32, mut flags: i32) -> *mut ext4_ext_path {
let mut __v0: *mut () = unsafe { core::mem::zeroed() };
let mut is_unwrit: i32 = unsafe { core::mem::zeroed() };
let mut insert_err: i32 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut depth: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(depth) as *mut u32;
let mut ee_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(ee_len) as *mut u32;
let mut ex2: *mut ext4_extent = unsafe { core::mem::zeroed() };
let mut orig_ex: core::mem::MaybeUninit<ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut ext4_extent = core::ptr::addr_of_mut!(orig_ex) as *mut ext4_extent;
let mut newex: core::mem::MaybeUninit<ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut ext4_extent = core::ptr::addr_of_mut!(newex) as *mut ext4_extent;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut ee_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(ee_block) as *mut u32;
let mut newblock: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(newblock) as *mut u64;
unsafe {
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 12usize);
let __slate_auto_bytes_4 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 12usize);
ex2 = core::ptr::null_mut::<ext4_extent>();
let __slate_auto_bytes_5 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
err = 0;
insert_err = 0;
is_unwrit = 0;
// Do not cache extents that are in the process of being modified.
flags |= 1073741824;
*__slate_slot_7 = ext_depth(inode) as u32;
*__slate_slot_12 = (*path.wrapping_offset(*__slate_slot_7 as isize)).p_ext;
*__slate_slot_13 = (*(*__slate_slot_12)).ee_block;
*__slate_slot_8 = ext4_ext_get_actual_len(*__slate_slot_12) as u32;
*__slate_slot_14 = (split.wrapping_sub(*__slate_slot_13) as u64).wrapping_add(ext4_ext_pblock(*__slate_slot_12));
is_unwrit = ext4_ext_is_unwritten(*__slate_slot_12);
'__join_26: {
loop {
if unlikely!(split < *__slate_slot_13 || split >= (*__slate_slot_13).wrapping_add(*__slate_slot_8)) != 0 {
break '__join_26;
} else {
if true {
break;
}
}
}
'__join_2: {
'__join_7: {
// No split needed
if split != *__slate_slot_13 {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_7 as isize));
if err == 0 {
// case a
memcpy(core::ptr::addr_of_mut!(*__slate_slot_10) as *mut (), *__slate_slot_12 as *const (), core::mem::size_of::<ext4_extent>() as u64);
(*(*__slate_slot_12)).ee_len = split.wrapping_sub(*__slate_slot_13) as u16;
if is_unwrit != 0 {
ext4_ext_mark_unwritten(*__slate_slot_12);
}
// path may lead to new leaf, not to original leaf any more
// after ext4_ext_insert_extent() returns,
err = ext4_ext_dirty__3!(handle, inode, path.wrapping_offset(*__slate_slot_7 as isize));
if err == 0 {
ex2 = core::ptr::addr_of_mut!(*__slate_slot_11);
(*ex2).ee_block = split;
(*ex2).ee_len = (*__slate_slot_8).wrapping_sub(split.wrapping_sub(*__slate_slot_13)) as u16;
ext4_ext_store_pblock(ex2, *__slate_slot_14);
if is_unwrit != 0 {
ext4_ext_mark_unwritten(ex2);
}
path = ext4_ext_insert_extent(handle, inode, path, core::ptr::addr_of_mut!(*__slate_slot_11), flags);
if !IS_ERR(path as *const ()) {
return path;
} else {
insert_err = PTR_ERR(path as *const ()) as i32;
err = 0;
if insert_err != 28i32.wrapping_neg() && insert_err != 122i32.wrapping_neg() && insert_err != 12i32.wrapping_neg() {
break '__join_2;
} else {
// Get a new path to try to zeroout or fix the extent length.
// Using EXT4_EX_NOFAIL guarantees that ext4_find_extent()
// will not return -ENOMEM, otherwise -ENOMEM will cause a
// retry in do_writepages(), and a WARN_ON may be triggered
// in ext4_da_update_reserve_space() due to an incorrect
// ee_len causing the i_reserved_data_blocks exception.
path = ext4_find_extent(inode, *__slate_slot_13, core::ptr::null_mut::<ext4_ext_path>(), flags | 268435456);
if IS_ERR(path as *const ()) {
__ext4_error_inode(inode, c"ext4_split_extent_at".as_ptr() as *const u8, 3280u32, 0u64, 0i32, c"Failed split extent on %u, err %pe".as_ptr() as *const u8, split, path);
break '__join_2;
} else {
*__slate_slot_7 = ext_depth(inode) as u32;
*__slate_slot_12 = (*path.wrapping_offset(*__slate_slot_7 as isize)).p_ext;
if (*__slate_slot_12).is_null() {
__ext4_error_inode(inode, c"ext4_split_extent_at".as_ptr() as *const u8, 3289u32, 0u64, 0i32, c"bad extent address lblock: %lu, depth: %d pblock %llu".as_ptr() as *const u8, *__slate_slot_13 as u64, *__slate_slot_7, (*path.wrapping_offset(*__slate_slot_7 as isize)).p_block);
err = 117i32.wrapping_neg();
break '__join_7;
} else {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_7 as isize));
if err != 0 {
break '__join_7;
}
}
}
}
}
}
(*(*__slate_slot_12)).ee_len = (*__slate_slot_10).ee_len;
err = ext4_ext_dirty__5!(handle, inode, path.wrapping_offset(((*path).p_depth as i32) as isize));
}
}
}
if err != 0 || insert_err != 0 {
ext4_free_ext_path(path);
if err != 0 {
__v0 = ERR_PTR(err as i64);
} else {
__v0 = ERR_PTR(insert_err as i64);
}
path = __v0 as *mut ext4_ext_path;
}
}
if IS_ERR(path as *const ()) {
ext4_es_remove_extent(inode, *__slate_slot_13, *__slate_slot_8);
}
// Remove all remaining potentially stale extents.
return path;
}
core::arch::asm!("1533: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1533b - ., 3; .popsection\n# {0}", const 1533 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_split_extent_at_15, sym __slate_asm_string_ext4_split_extent_at_16, const 3226 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

fn ext4_split_extent_zeroout(handle: *mut jbd2_journal_handle, inode: *mut inode, path: *mut ext4_ext_path, map: *mut ext4_map_blocks, flags: i32) -> i32 {
let mut ee_len: u32 = 0;
let depth: u32 = ext_depth(inode) as u32;
let mut ee_block: u32 = 0;
let mut lblk: u64 = 0;
let mut pblk: u64 = 0;
let mut len: u64 = 0;
let mut is_unwrit: i32 = 0;
let mut err: i32 = 0;
let ex: *mut ext4_extent = unsafe { (*path.wrapping_offset(depth as isize)).p_ext };
ee_block = unsafe { (*ex).ee_block };
ee_len = ext4_ext_get_actual_len(ex) as u32;
is_unwrit = ext4_ext_is_unwritten(ex);
if flags & 16 != 0 {
// EXT4_GET_BLOCKS_CONVERT: Caller wants the range specified by
// map to be initialized. Zeroout everything except the map
// range.
let map_end: i64 = ((unsafe { (*map).m_lblk }) as i64).wrapping_add((unsafe { (*map).m_len }) as i64);
let ex_end: i64 = (ee_block as i64).wrapping_add(ee_len as i64);
if is_unwrit == 0 {
return 22i32.wrapping_neg();
}
// Shouldn't happen. Just exit
// zeroout left
if (unsafe { (*map).m_lblk }) > ee_block {
lblk = ee_block as u64;
len = unsafe { (*map).m_lblk }.wrapping_sub(ee_block) as u64;
pblk = ext4_ext_pblock(ex);
err = unsafe { ext4_issue_zeroout(inode, lblk as u32, pblk, len as u32) };
if err != 0 {
return err;
}
// ZEROOUT failed, just return original error
}
// zeroout right
if map_end < ex_end {
lblk = map_end as u64;
len = ex_end.wrapping_sub(map_end) as u64;
pblk = ext4_ext_pblock(ex).wrapping_add(map_end.wrapping_sub(ee_block as i64) as u64);
err = unsafe { ext4_issue_zeroout(inode, lblk as u32, pblk, len as u32) };
if err != 0 {
return err;
}
// ZEROOUT failed, just return original error
}
} else {
if flags & 256 != 0 {
// EXT4_GET_BLOCKS_CONVERT_UNWRITTEN: Caller wants the
// range specified by map to be marked unwritten.
// Zeroout the map range leaving rest as it is.
if is_unwrit != 0 {
return 22i32.wrapping_neg();
}
// Shouldn't happen. Just exit
lblk = (unsafe { (*map).m_lblk }) as u64;
len = (unsafe { (*map).m_len }) as u64;
pblk = ext4_ext_pblock(ex).wrapping_add(unsafe { (*map).m_lblk }.wrapping_sub(ee_block) as u64);
err = unsafe { ext4_issue_zeroout(inode, lblk as u32, pblk, len as u32) };
if err != 0 {
return err;
}
// ZEROOUT failed, just return original error
} else {
// We no longer perform unwritten to unwritten splits in IO paths.
// Hence this should not happen.
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1534: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1534b - ., 3; .popsection\n# {0}", const 1534 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_split_extent_zeroout_1, sym __slate_asm_string_ext4_split_extent_zeroout_2, const 3389 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1535: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1535b - ., 4; .popsection\n# {0}", const 1535 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
return 22i32.wrapping_neg();
}
}
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(depth as isize));
if err != 0 {
return err;
}
ext4_ext_mark_initialized(ex);
err = ext4_ext_dirty__6!(handle, inode, path.wrapping_offset(depth as isize));
if err != 0 {
return err;
}
0
}

/// ext4_split_extent() splits an extent and mark extent which is covered
/// by @map as split_flags indicates
///
/// It may result in splitting the extent into multiple extents (up to three)
/// There are three possibilities:
///   a> There is no split required
///   b> Splits in two extents: Split is happening at either end of the extent
///   c> Splits in three extents: Somone is splitting in middle of the extent
fn ext4_split_extent(handle: *mut jbd2_journal_handle, inode: *mut inode, mut path: *mut ext4_ext_path, map: *mut ext4_map_blocks, split_flag: i32, mut flags: i32, allocated: *mut u32, did_zeroout: *mut bool) -> *mut ext4_ext_path {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut orig_err: i32 = unsafe { core::mem::zeroed() };
let mut orig_unwritten: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(orig_unwritten) as *mut i32;
let mut unwritten: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(unwritten) as *mut i32;
let mut depth: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(depth) as *mut u32;
let mut orig_ee_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(orig_ee_len) as *mut u32;
let mut ee_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(ee_len) as *mut u32;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut orig_ee_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(orig_ee_block) as *mut u32;
let mut ee_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u32 = core::ptr::addr_of_mut!(ee_block) as *mut u32;
unsafe {
'__join_6: {
'__join_19: {
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
orig_err = 0;
*__slate_slot_9 = ext_depth(inode) as u32;
*__slate_slot_12 = (*path.wrapping_offset(*__slate_slot_9 as isize)).p_ext;
*__slate_slot_14 = (*(*__slate_slot_12)).ee_block;
*__slate_slot_11 = ext4_ext_get_actual_len(*__slate_slot_12) as u32;
*__slate_slot_8 = ext4_ext_is_unwritten(*__slate_slot_12);
*__slate_slot_13 = *__slate_slot_14;
*__slate_slot_10 = *__slate_slot_11;
*__slate_slot_7 = *__slate_slot_8;
// Do not cache extents that are in the process of being modified.
flags |= 1073741824;
if unsafe { (*map).m_lblk }.wrapping_add((*map).m_len) < (*__slate_slot_14).wrapping_add(*__slate_slot_11) {
path = ext4_split_extent_at(handle, inode, path, unsafe { (*map).m_lblk }.wrapping_add((*map).m_len), flags);
if IS_ERR(path as *const ()) {
break '__join_19;
} else {
// Update path is required because previous ext4_split_extent_at
// may result in split of original leaf or extent zeroout.
path = ext4_find_extent(inode, (*map).m_lblk, path, flags);
if IS_ERR(path as *const ()) {
break '__join_19;
} else {
*__slate_slot_9 = ext_depth(inode) as u32;
*__slate_slot_12 = (*path.wrapping_offset(*__slate_slot_9 as isize)).p_ext;
if (*__slate_slot_12).is_null() {
__ext4_error_inode(inode, c"ext4_split_extent".as_ptr() as *const u8, 3461u32, 0u64, 0i32, c"unexpected hole at %lu".as_ptr() as *const u8, (*map).m_lblk as u64);
ext4_free_ext_path(path);
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut ext4_ext_path;
} else {
// extent would have changed so update original values
*__slate_slot_13 = (*(*__slate_slot_12)).ee_block;
*__slate_slot_10 = ext4_ext_get_actual_len(*__slate_slot_12) as u32;
*__slate_slot_7 = ext4_ext_is_unwritten(*__slate_slot_12);
}
}
}
}
if (*map).m_lblk >= *__slate_slot_14 {
path = ext4_split_extent_at(handle, inode, path, (*map).m_lblk, flags);
if !IS_ERR(path as *const ()) {
break '__join_6;
}
} else {
break '__join_6;
}
}
'__join_0: {
orig_err = PTR_ERR(path as *const ()) as i32;
// There was an error in splitting the extent. So instead, just zeroout
// unwritten portions and convert it to initialized as a last resort. If
// there is any failure here we just return the original error
if !(orig_err != 28i32.wrapping_neg() && orig_err != 122i32.wrapping_neg() && orig_err != 12i32.wrapping_neg()) {
// we can't zeroout? just return the original err
if split_flag & 1 != 0 {
if flags & 256 != 0 {
let max_zeroout_blks: i32 = ((*EXT4_SB((*inode).i_sb)).s_extent_max_zeroout_kb >> ((*(*inode).i_sb).s_blocksize_bits as i32).wrapping_sub(10)) as i32;
if (*map).m_len > (max_zeroout_blks as u32) {
break '__join_0;
}
}
path = ext4_find_extent(inode, (*map).m_lblk, core::ptr::null_mut::<ext4_ext_path>(), flags);
if !IS_ERR(path as *const ()) {
*__slate_slot_9 = ext_depth(inode) as u32;
*__slate_slot_12 = (*path.wrapping_offset(*__slate_slot_9 as isize)).p_ext;
*__slate_slot_14 = (*(*__slate_slot_12)).ee_block;
*__slate_slot_11 = ext4_ext_get_actual_len(*__slate_slot_12) as u32;
*__slate_slot_8 = ext4_ext_is_unwritten(*__slate_slot_12);
__ret_warn_on = (*__slate_slot_14 != *__slate_slot_13 || *__slate_slot_11 != *__slate_slot_10 || *__slate_slot_8 != *__slate_slot_7) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("1536: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1536b - ., 3; .popsection\n# {0}", const 1536 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_split_extent_15, sym __slate_asm_string_ext4_split_extent_16, const 3517 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1537: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1537b - ., 4; .popsection\n# {0}", const 1537 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// extent to zeroout should have been unchanged but its not
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 == 0 {
if ext4_split_extent_zeroout(handle, inode, path, map, flags) == 0 {
// zeroout succeeded
if !did_zeroout.is_null() {
*did_zeroout = true;
break '__join_6;
} else {
break '__join_6;
}
}
}
ext4_free_ext_path(path);
}
}
}
}
return ERR_PTR(orig_err as i64) as *mut ext4_ext_path;
}
if !allocated.is_null() {
if unsafe { (*map).m_lblk }.wrapping_add((*map).m_len) > (*__slate_slot_14).wrapping_add(*__slate_slot_11) {
*allocated = (*__slate_slot_11).wrapping_sub(unsafe { (*map).m_lblk }.wrapping_sub(*__slate_slot_14));
} else {
*allocated = (*map).m_len;
}
}
return path;
}
}

/// This function is called by ext4_ext_map_blocks() if someone tries to write
/// to an unwritten extent. It may result in splitting the unwritten
/// extent into multiple extents (up to three - one initialized and two
/// unwritten).
/// There are three possibilities:
///   a> There is no split required: Entire extent should be initialized
///   b> Splits in two extents: Write is happening at either end of the extent
///   c> Splits in three extents: Somone is writing in middle of the extent
///
/// Pre-conditions:
///  - The extent pointed to by 'path' is unwritten.
///  - The extent pointed to by 'path' contains a superset
///    of the logical span [map->m_lblk, map->m_lblk + map->m_len).
///
/// Post-conditions on success:
///  - the returned value is the number of blocks beyond map->l_lblk
///    that are allocated and initialized.
///    It is guaranteed to be >= map->m_len.
fn ext4_ext_convert_to_initialized(handle: *mut jbd2_journal_handle, inode: *mut inode, map: *mut ext4_map_blocks, mut path: *mut ext4_ext_path, flags: i32, allocated: *mut u32) -> *mut ext4_ext_path {
let mut prev_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(prev_len) as *mut u32;
let mut ee_pblk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(ee_pblk) as *mut u64;
let mut prev_pblk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(prev_pblk) as *mut u64;
let mut prev_lblk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(prev_lblk) as *mut u32;
let mut next_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(next_len) as *mut u32;
let mut ee_pblk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(ee_pblk) as *mut u64;
let mut next_pblk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u64 = core::ptr::addr_of_mut!(next_pblk) as *mut u64;
// See if we can merge right
let mut next_lblk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(next_lblk) as *mut u32;
let mut max_zeroout: u32 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut map_len: u32 = unsafe { core::mem::zeroed() };
let mut depth: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u32 = core::ptr::addr_of_mut!(depth) as *mut u32;
let mut ee_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u32 = core::ptr::addr_of_mut!(ee_len) as *mut u32;
let mut eof_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u32 = core::ptr::addr_of_mut!(eof_block) as *mut u32;
let mut ee_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u32 = core::ptr::addr_of_mut!(ee_block) as *mut u32;
let mut abut_ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut ext4_extent = core::ptr::addr_of_mut!(abut_ex) as *mut *mut ext4_extent;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut zero_ex2: core::mem::MaybeUninit<ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut ext4_extent = core::ptr::addr_of_mut!(zero_ex2) as *mut ext4_extent;
let mut zero_ex1: core::mem::MaybeUninit<ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut ext4_extent = core::ptr::addr_of_mut!(zero_ex1) as *mut ext4_extent;
let mut split_map: core::mem::MaybeUninit<ext4_map_blocks> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut ext4_map_blocks = core::ptr::addr_of_mut!(split_map) as *mut ext4_map_blocks;
let mut eh: core::mem::MaybeUninit<*mut ext4_extent_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut ext4_extent_header = core::ptr::addr_of_mut!(eh) as *mut *mut ext4_extent_header;
let mut sbi: core::mem::MaybeUninit<*mut ext4_sb_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut *mut ext4_sb_info = core::ptr::addr_of_mut!(sbi) as *mut *mut ext4_sb_info;
unsafe {
{
let __slate_auto_bytes_6 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_9 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 12usize);
let __slate_auto_bytes_10 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 12usize);
let __slate_auto_bytes_11 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_12 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_13 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_14 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_15 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_15.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
map_len = (*map).m_len;
err = 0;
max_zeroout = 0;
*__slate_slot_27 = EXT4_SB((*inode).i_sb);
*__slate_slot_19 = (((*EXT4_I!(inode)).i_disksize as u64).wrapping_add((*(*inode).i_sb).s_blocksize).wrapping_sub(1) >> ((*(*inode).i_sb).s_blocksize_bits as i32)) as u32;
if *__slate_slot_19 < unsafe { (*map).m_lblk }.wrapping_add(map_len) {
*__slate_slot_19 = unsafe { (*map).m_lblk }.wrapping_add(map_len);
}
}
*__slate_slot_17 = ext_depth(inode) as u32;
*__slate_slot_26 = (*path.wrapping_offset(*__slate_slot_17 as isize)).p_hdr;
*__slate_slot_22 = (*path.wrapping_offset(*__slate_slot_17 as isize)).p_ext;
*__slate_slot_20 = (*(*__slate_slot_22)).ee_block;
*__slate_slot_18 = ext4_ext_get_actual_len(*__slate_slot_22) as u32;
(*__slate_slot_24).ee_len = 0;
(*__slate_slot_23).ee_len = 0;
trace_ext4_ext_convert_to_initialized_enter(inode, map, *__slate_slot_22);
// Pre-conditions
'__join_36: {
loop {
if unlikely!(ext4_ext_is_unwritten(*__slate_slot_22) == 0) != 0 {
break '__join_36;
} else {
if true {
break;
}
}
}
'__join_32: {
loop {
if unlikely!(in_range__3!(unsafe { (*map).m_lblk }, *__slate_slot_20, *__slate_slot_18) == 0) != 0 {
break '__join_32;
} else {
if true {
break;
}
}
}
'__join_0: {
// Attempt to transfer newly initialized blocks from the currently
// unwritten extent to its neighbor. This is much cheaper
// than an insertion followed by a merge as those involve costly
// memmove() calls. Transferring to the left is the common case in
// steady state for workloads doing fallocate(FALLOC_FL_KEEP_SIZE)
// followed by append writes.
//
// Limitations of the current logic:
//  - L1: we do not deal with writes covering the whole extent.
//    This would require removing the extent if the transfer
//    is possible.
//  - L2: we only attempt to merge with an extent stored in the
//    same extent tree node.
*allocated = 0;
if (*map).m_lblk == *__slate_slot_20 && map_len < *__slate_slot_18 && *__slate_slot_22 > EXT_FIRST_EXTENT!(*__slate_slot_26) {
// L2
let __slate_auto_bytes_16 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_16.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_21 = (*__slate_slot_22).wrapping_offset(-(1 as isize));
*__slate_slot_8 = (*(*__slate_slot_21)).ee_block;
*__slate_slot_5 = ext4_ext_get_actual_len(*__slate_slot_21) as u32;
*__slate_slot_7 = ext4_ext_pblock(*__slate_slot_21);
*__slate_slot_6 = ext4_ext_pblock(*__slate_slot_22);
// A transfer of blocks from 'ex' to 'abut_ex' is allowed
// upon those conditions:
// - C1: abut_ex is initialized,
// - C2: abut_ex is logically abutting ex,
// - C3: abut_ex is physically abutting ex,
// - C4: abut_ex can receive the additional blocks without
//   overflowing the (initialized) length limit.
if ext4_ext_is_unwritten(*__slate_slot_21) == 0 && (*__slate_slot_8).wrapping_add(*__slate_slot_5) == *__slate_slot_20 && (*__slate_slot_7).wrapping_add(*__slate_slot_5 as u64) == *__slate_slot_6 && (*__slate_slot_5 as u64) < 32768u64.wrapping_sub(map_len as u64) {
// C4
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_17 as isize));
if err != 0 {
break '__join_0;
} else {
trace_ext4_ext_convert_to_initialized_fastpath(inode, map, *__slate_slot_22, *__slate_slot_21);
// Shift the start of ex by 'map_len' blocks
(*(*__slate_slot_22)).ee_block = (*__slate_slot_20).wrapping_add(map_len);
ext4_ext_store_pblock(*__slate_slot_22, (*__slate_slot_6).wrapping_add(map_len as u64));
(*(*__slate_slot_22)).ee_len = (*__slate_slot_18).wrapping_sub(map_len) as u16;
ext4_ext_mark_unwritten(*__slate_slot_22); // Restore the flag
// Extend abut_ex by 'map_len' blocks
(*(*__slate_slot_21)).ee_len = (*__slate_slot_5).wrapping_add(map_len) as u16;
// Result: number of initialized blocks past m_lblk
*allocated = map_len;
}
}
// C1
// C2
// C3
} else {
if unsafe { (*map).m_lblk }.wrapping_add(map_len) == (*__slate_slot_20).wrapping_add(*__slate_slot_18) && map_len < *__slate_slot_18 && *__slate_slot_22 < EXT_LAST_EXTENT!(*__slate_slot_26) {
// L2
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_21 = (*__slate_slot_22).wrapping_offset(1 as isize);
*__slate_slot_12 = (*(*__slate_slot_21)).ee_block;
*__slate_slot_9 = ext4_ext_get_actual_len(*__slate_slot_21) as u32;
*__slate_slot_11 = ext4_ext_pblock(*__slate_slot_21);
*__slate_slot_10 = ext4_ext_pblock(*__slate_slot_22);
// A transfer of blocks from 'ex' to 'abut_ex' is allowed
// upon those conditions:
// - C1: abut_ex is initialized,
// - C2: abut_ex is logically abutting ex,
// - C3: abut_ex is physically abutting ex,
// - C4: abut_ex can receive the additional blocks without
//   overflowing the (initialized) length limit.
if ext4_ext_is_unwritten(*__slate_slot_21) == 0 && unsafe { (*map).m_lblk }.wrapping_add(map_len) == *__slate_slot_12 && (*__slate_slot_10).wrapping_add(*__slate_slot_18 as u64) == *__slate_slot_11 && (*__slate_slot_9 as u64) < 32768u64.wrapping_sub(map_len as u64) {
// C4
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_17 as isize));
if err != 0 {
break '__join_0;
} else {
trace_ext4_ext_convert_to_initialized_fastpath(inode, map, *__slate_slot_22, *__slate_slot_21);
// Shift the start of abut_ex by 'map_len' blocks
(*(*__slate_slot_21)).ee_block = (*__slate_slot_12).wrapping_sub(map_len);
ext4_ext_store_pblock(*__slate_slot_21, (*__slate_slot_11).wrapping_sub(map_len as u64));
(*(*__slate_slot_22)).ee_len = (*__slate_slot_18).wrapping_sub(map_len) as u16;
ext4_ext_mark_unwritten(*__slate_slot_22); // Restore the flag
// Extend abut_ex by 'map_len' blocks
(*(*__slate_slot_21)).ee_len = (*__slate_slot_9).wrapping_add(map_len) as u16;
// Result: number of initialized blocks past m_lblk
*allocated = map_len;
}
}
// C1
// C2
// C3
}
}
// See if we can merge left
// L1
// L1
if *allocated != 0 {
// Mark the block containing both extents as dirty
err = ext4_ext_dirty__2!(handle, inode, path.wrapping_offset(*__slate_slot_17 as isize));
// Update path to point to the right extent
(*path.wrapping_offset(*__slate_slot_17 as isize)).p_ext = *__slate_slot_21;
if err != 0 {
break '__join_0;
}
} else {
*allocated = (*__slate_slot_18).wrapping_sub(unsafe { (*map).m_lblk }.wrapping_sub(*__slate_slot_20));
let __ret_warn_on: i32 = ((*map).m_lblk < *__slate_slot_20) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("1540: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1540b - ., 3; .popsection\n# {0}", const 1540 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_ext_convert_to_initialized_28, sym __slate_asm_string_ext4_ext_convert_to_initialized_29, const 3723 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1541: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1541b - ., 4; .popsection\n# {0}", const 1541 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// It is safe to convert extent to initialized via explicit
// zeroout only if extent is fully inside i_size or new_size.
if (*__slate_slot_20).wrapping_add(*__slate_slot_18) <= *__slate_slot_19 {
max_zeroout = (*(*__slate_slot_27)).s_extent_max_zeroout_kb >> ((*(*inode).i_sb).s_blocksize_bits as i32).wrapping_sub(10);
}
'__join_3: {
// five cases:
// 1. split the extent into three extents.
// 2. split the extent into two extents, zeroout the head of the first
//    extent.
// 3. split the extent into two extents, zeroout the tail of the second
//    extent.
// 4. split the extent into two extents with out zeroout.
// 5. no splitting needed, just possibly zeroout the head and / or the
//    tail of the extent.
(*__slate_slot_25).m_lblk = (*map).m_lblk;
(*__slate_slot_25).m_len = (*map).m_len;
if max_zeroout != 0 && *allocated > (*__slate_slot_25).m_len {
if *allocated <= max_zeroout {
// case 3 or 5
(*__slate_slot_24).ee_block = (*__slate_slot_25).m_lblk.wrapping_add((*__slate_slot_25).m_len);
(*__slate_slot_24).ee_len = unsafe { *allocated }.wrapping_sub((*__slate_slot_25).m_len) as u16;
ext4_ext_store_pblock(core::ptr::addr_of_mut!(*__slate_slot_24), ext4_ext_pblock(*__slate_slot_22).wrapping_add((*__slate_slot_25).m_lblk as u64).wrapping_add((*__slate_slot_25).m_len as u64).wrapping_sub(*__slate_slot_20 as u64));
err = ext4_ext_zeroout(inode, core::ptr::addr_of_mut!(*__slate_slot_24));
if err != 0 {
break '__join_3;
} else {
(*__slate_slot_25).m_len = *allocated;
}
}
if (*__slate_slot_25).m_lblk.wrapping_sub(*__slate_slot_20).wrapping_add((*__slate_slot_25).m_len) < max_zeroout {
// case 2 or 5
if (*__slate_slot_25).m_lblk != *__slate_slot_20 {
(*__slate_slot_23).ee_block = (*(*__slate_slot_22)).ee_block;
(*__slate_slot_23).ee_len = (*__slate_slot_25).m_lblk.wrapping_sub(*__slate_slot_20) as u16;
ext4_ext_store_pblock(core::ptr::addr_of_mut!(*__slate_slot_23), ext4_ext_pblock(*__slate_slot_22));
err = ext4_ext_zeroout(inode, core::ptr::addr_of_mut!(*__slate_slot_23));
if err != 0 {
break '__join_3;
}
}
let __v1: u32 = (*__slate_slot_25).m_len;
(*__slate_slot_25).m_len = __v1.wrapping_add((*__slate_slot_25).m_lblk.wrapping_sub(*__slate_slot_20));
(*__slate_slot_25).m_lblk = *__slate_slot_20;
*allocated = (*map).m_len;
}
}
}
path = ext4_split_convert_extents(handle, inode, core::ptr::addr_of_mut!(*__slate_slot_25), path, flags | 16, core::ptr::null_mut::<u32>());
if IS_ERR(path as *const ()) {
return path;
}
}
ext4_zeroout_es(inode, core::ptr::addr_of_mut!(*__slate_slot_24)); // If we have gotten a failure, don't zero out status tree
ext4_zeroout_es(inode, core::ptr::addr_of_mut!(*__slate_slot_23));
return path;
}
ext4_free_ext_path(path);
return ERR_PTR(err as i64) as *mut ext4_ext_path;
}
core::arch::asm!("1539: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1539b - ., 3; .popsection\n# {0}", const 1539 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_convert_to_initialized_30, sym __slate_asm_string_ext4_ext_convert_to_initialized_31, const 3600 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("1538: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1538b - ., 3; .popsection\n# {0}", const 1538 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_ext_convert_to_initialized_32, sym __slate_asm_string_ext4_ext_convert_to_initialized_33, const 3599 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

/// This function is called by ext4_ext_map_blocks() from
/// ext4_get_blocks_dio_write() when DIO to write
/// to an unwritten extent.
///
/// Writing to an unwritten extent may result in splitting the unwritten
/// extent into multiple initialized/unwritten extents (up to three)
/// There are three possibilities:
///   a> There is no split required: Entire extent should be unwritten
///   b> Splits in two extents: Write is happening at either end of the extent
///   c> Splits in three extents: Somone is writing in middle of the extent
///
/// This works the same way in the case of initialized -> unwritten conversion.
///
/// One of more index blocks maybe needed if the extent tree grow after
/// the unwritten extent split. To prevent ENOSPC occur at the IO
/// complete, we need to split the unwritten extent before DIO submit
/// the IO. The unwritten extent called at this time will be split
/// into three unwritten extent(at most). After IO complete, the part
/// being filled will be convert to initialized by the end_io callback function
/// via ext4_convert_unwritten_extents().
///
/// The size of unwritten extent to be written is passed to the caller via the
/// allocated pointer. Return an extent path pointer on success, or an error
/// pointer on failure.
fn ext4_split_convert_extents(handle: *mut jbd2_journal_handle, inode: *mut inode, map: *mut ext4_map_blocks, mut path: *mut ext4_ext_path, flags: i32, allocated: *mut u32) -> *mut ext4_ext_path {
let mut did_zeroout: bool = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32;
let mut split_flag: i32 = unsafe { core::mem::zeroed() };
let mut ee_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(ee_len) as *mut u32;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut ee_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(ee_block) as *mut u32;
let mut eof_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(eof_block) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
split_flag = 0;
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
err = 0;
did_zeroout = false;
*__slate_slot_10 = (((*EXT4_I!(inode)).i_disksize as u64).wrapping_add((*(*inode).i_sb).s_blocksize).wrapping_sub(1) >> ((*(*inode).i_sb).s_blocksize_bits as i32)) as u32;
if *__slate_slot_10 < unsafe { (*map).m_lblk }.wrapping_add((*map).m_len) {
*__slate_slot_10 = unsafe { (*map).m_lblk }.wrapping_add((*map).m_len);
}
*__slate_slot_5 = ext_depth(inode) as i32;
*__slate_slot_8 = (*path.wrapping_offset(*__slate_slot_5 as isize)).p_ext;
*__slate_slot_9 = (*(*__slate_slot_8)).ee_block;
*__slate_slot_7 = ext4_ext_get_actual_len(*__slate_slot_8) as u32;
// No split needed
if !(*__slate_slot_9 == (*map).m_lblk && *__slate_slot_7 == (*map).m_len) {
// It is only safe to convert extent to initialized via explicit
// zeroout only if extent is fully inside i_size or new_size.
split_flag |= if (*__slate_slot_9).wrapping_add(*__slate_slot_7) <= *__slate_slot_10 { 1 } else { 0 };
// pass SPLIT_NOMERGE explicitly so we don't end up merging extents we
// just split.
path = ext4_split_extent(handle, inode, path, map, split_flag, flags | 8, allocated, core::ptr::addr_of_mut!(did_zeroout));
if IS_ERR(path as *const ()) {
return path;
}
}
path = ext4_find_extent(inode, (*map).m_lblk, path, flags);
if IS_ERR(path as *const ()) {
return path;
} else {
'__join_2: {
*__slate_slot_5 = ext_depth(inode) as i32;
*__slate_slot_8 = (*path.wrapping_offset(*__slate_slot_5 as isize)).p_ext;
// Conversion is already handled in case of zeroout
if !did_zeroout {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_5 as isize));
if err != 0 {
break '__join_2;
} else {
if flags & 16 != 0 {
ext4_ext_mark_initialized(*__slate_slot_8);
} else {
if flags & 256 != 0 {
ext4_ext_mark_unwritten(*__slate_slot_8);
}
}
if flags & 8 == 0 {
ext4_ext_try_to_merge(handle, inode, path, *__slate_slot_8);
}
// note: ext4_ext_correct_indexes() isn't needed here because
// borders are not changed
err = ext4_ext_dirty__10!(handle, inode, path.wrapping_offset(*__slate_slot_5 as isize));
if err != 0 {
break '__join_2;
}
}
}
// Lets update the extent status tree after conversion
if flags & 1073741824 == 0 {
ext4_es_insert_extent(inode, (*(*__slate_slot_8)).ee_block, ext4_ext_get_actual_len(*__slate_slot_8) as u32, ext4_ext_pblock(*__slate_slot_8), (if ext4_ext_is_unwritten(*__slate_slot_8) != 0 { 2i32 } else { 1i32 }) as u32, false);
}
}
if err != 0 {
ext4_free_ext_path(path);
return ERR_PTR(err as i64) as *mut ext4_ext_path;
} else {
return path;
}
}
}
}

fn ext4_convert_unwritten_extents_endio(handle: *mut jbd2_journal_handle, inode: *mut inode, map: *mut ext4_map_blocks, path: *mut ext4_ext_path, flags: i32) -> *mut ext4_ext_path {
let mut ee_block: u32 = 0;
let mut ee_len: u32 = 0;
let depth: i32 = ext_depth(inode) as i32;
let ex: *mut ext4_extent = unsafe { (*path.wrapping_offset(depth as isize)).p_ext };
ee_block = unsafe { (*ex).ee_block };
ee_len = ext4_ext_get_actual_len(ex) as u32;
let mut __v0: i32;
__v0 = 0;
ext4_split_convert_extents(handle, inode, map, path, flags, core::ptr::null_mut::<u32>())
}

fn convert_initialized_extent(handle: *mut jbd2_journal_handle, inode: *mut inode, map: *mut ext4_map_blocks, mut path: *mut ext4_ext_path, flags: i32, allocated: &mut u32) -> *mut ext4_ext_path {
let mut ex: *mut ext4_extent = unsafe { core::mem::zeroed() };
let mut ee_block: u32 = 0;
let mut ee_len: u32 = 0;
let mut depth: i32 = 0;
// Make sure that the extent is no bigger than we support with
// unwritten extent
if ((unsafe { (*map).m_len }) as u64) > 32767u64 {
unsafe {
(*map).m_len = 16383u32;
}
}
depth = ext_depth(inode) as i32;
ex = unsafe { (*path.wrapping_offset(depth as isize)).p_ext };
ee_block = unsafe { (*ex).ee_block };
ee_len = ext4_ext_get_actual_len(ex) as u32;
let mut __v0: i32;
__v0 = 0;
path = ext4_split_convert_extents(handle, inode, map, path, flags, core::ptr::null_mut::<u32>());
if IS_ERR(path as *const ()) {
return path;
}
ext4_update_inode_fsync_trans(handle, inode, 1i32);
// The extent might be initialized in case of zeroout.
path = ext4_find_extent(inode, unsafe { (*map).m_lblk }, path, flags);
if IS_ERR(path as *const ()) {
return path;
}
depth = ext_depth(inode) as i32;
ex = unsafe { (*path.wrapping_offset(depth as isize)).p_ext };
if ext4_ext_is_unwritten(ex) != 0 {
let __v2: u32 = unsafe { (*map).m_flags };
let __v3: u32 = ((__v2 as u64) | BIT!(11i32)) as u32;
unsafe {
(*map).m_flags = __v3;
}
} else {
let __v5: u32 = unsafe { (*map).m_flags };
let __v6: u32 = ((__v5 as u64) | BIT!(4i32)) as u32;
unsafe {
(*map).m_flags = __v6;
}
}
if (unsafe { *(allocated as *mut u32) }) > unsafe { (*map).m_len } {
unsafe {
*(allocated as *mut u32) = (*map).m_len;
}
}
unsafe {
(*map).m_len = *(allocated as *mut u32);
}
path
}

fn ext4_ext_handle_unwritten_extents(handle: *mut jbd2_journal_handle, inode: *mut inode, map: *mut ext4_map_blocks, mut path: *mut ext4_ext_path, mut flags: i32, allocated: *mut u32, newblock: u64) -> *mut ext4_ext_path {
let mut err: i32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_2: {
err = 0;
// When writing into unwritten space, we should not fail to
// allocate metadata blocks for the new extent block if needed.
flags |= 32;
trace_ext4_ext_handle_unwritten_extents(inode, map, flags, *allocated, newblock);
// IO end_io complete, convert the filled extent to written
if flags & 16 != 0 {
path = ext4_convert_unwritten_extents_endio(handle, inode, map, path, flags);
if IS_ERR(path as *const ()) {
return path;
} else {
ext4_update_inode_fsync_trans(handle, inode, 1i32);
}
} else {
// buffered IO cases
//
// repeat fallocate creation request
// we already have an unwritten extent
if flags & 2 != 0 {
let __v10: u32 = (*map).m_flags;
let __v9: u32 = ((__v10 as u64) | BIT!(11i32)) as u32;
(*map).m_flags = __v9;
} else {
// buffered READ or buffered write_begin() lookup
if flags & 1 == 0 {
// We have blocks reserved already.  We
// return allocated blocks so that delalloc
// won't do block reservation for us.  But
// the buffer head will be unmapped so that
// a read from the block returns 0s.
let __v7: u32 = (*map).m_flags;
let __v6: u32 = ((__v7 as u64) | BIT!(11i32)) as u32;
(*map).m_flags = __v6;
break '__join_2;
} else {
// Default case when (flags & EXT4_GET_BLOCKS_CREATE) == 1.
// For buffered writes, at writepage time, etc.  Convert a
// discovered unwritten extent to written.
path = ext4_ext_convert_to_initialized(handle, inode, map, path, flags, allocated);
if IS_ERR(path as *const ()) {
return path;
} else {
ext4_update_inode_fsync_trans(handle, inode, 1i32);
// shouldn't get a 0 allocated when converting an unwritten extent
// unless m_len is 0 (bug) or extent has been corrupted
if unlikely!(*allocated == 0) != 0 {
__ext4_error_inode(inode, c"ext4_ext_handle_unwritten_extents".as_ptr() as *const u8, 4069u32, 0u64, 0i32, c"unexpected allocated == 0, m_len = %u".as_ptr() as *const u8, (*map).m_len);
err = 117i32.wrapping_neg();
ext4_free_ext_path(path);
return ERR_PTR(err as i64) as *mut ext4_ext_path;
} else {
let __v4: u32 = (*map).m_flags;
let __v3: u32 = ((__v4 as u64) | BIT!(5i32)) as u32;
(*map).m_flags = __v3;
}
}
}
}
}
let __v1: u32 = (*map).m_flags;
let __v0: u32 = ((__v1 as u64) | BIT!(4i32)) as u32;
(*map).m_flags = __v0;
}
(*map).m_pblk = newblock;
if *allocated > (*map).m_len {
*allocated = (*map).m_len;
}
(*map).m_len = *allocated;
return path;
}
}

// get_implied_cluster_alloc - check to see if the requested
// allocation (in the map structure) overlaps with a cluster already
// allocated in an extent.
// 	@sb	The filesystem superblock structure
// 	@map	The requested lblk->pblk mapping
// 	@ex	The extent structure which might contain an implied
// 			cluster allocation
//
// This function is called by ext4_ext_map_blocks() after we failed to
// find blocks that were already in the inode's extent tree.  Hence,
// we know that the beginning of the requested region cannot overlap
// the extent from the inode's extent tree.  There are three cases we
// want to catch.  The first is this case:
//
// 		 |--- cluster # N--|
//    |--- extent ---|	|---- requested region ---|
// 			|==========|
//
// The second case that we need to test for is this one:
//
//   |--------- cluster # N ----------------|
// 	   |--- requested region --|   |------- extent ----|
// 	   |=======================|
//
// The third case is when the requested region lies between two extents
// within the same cluster:
//          |------------- cluster # N-------------|
// |----- ex -----|                  |---- ex_right ----|
//                  |------ requested region ------|
//                  |================|
//
// In each of the above cases, we need to set the map->m_pblk and
// map->m_len so it corresponds to the return the extent labelled as
// "|====|" from cluster #N, since it is already in use for data in
// cluster EXT4_B2C(sbi, map->m_lblk).	We will then return 1 to
// signal to ext4_ext_map_blocks() that map->m_pblk should be treated
// as a new "allocated" block region.  Otherwise, we will return 0 and
// ext4_ext_map_blocks() will then allocate one or more new clusters
// by calling ext4_mb_new_blocks().
fn get_implied_cluster_alloc(sb: *mut super_block, map: *mut ext4_map_blocks, ex: *mut ext4_extent, path: *mut ext4_ext_path) -> i32 {
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let c_offset: u32 = EXT4_LBLK_COFF!(sbi, unsafe { (*map).m_lblk });
let mut ex_cluster_start: u32 = 0;
let mut ex_cluster_end: u32 = 0;
let mut rr_cluster_start: u32 = 0;
let ee_block: u32 = unsafe { (*ex).ee_block };
let mut ee_start: u64 = ext4_ext_pblock(ex);
let ee_len: u16 = ext4_ext_get_actual_len(ex) as u16;
// The extent passed in that we are trying to match
ex_cluster_start = EXT4_B2C!(sbi, ee_block);
ex_cluster_end = EXT4_B2C!(sbi, ee_block.wrapping_add((ee_len as i32) as u32).wrapping_sub(1));
// The requested region passed into ext4_map_blocks()
rr_cluster_start = EXT4_B2C!(sbi, unsafe { (*map).m_lblk });
if rr_cluster_start == ex_cluster_end || rr_cluster_start == ex_cluster_start {
if rr_cluster_start == ex_cluster_end {
ee_start = ee_start.wrapping_add((ee_len as i32).wrapping_sub(1) as u64);
}
unsafe {
(*map).m_pblk = EXT4_PBLK_CMASK!(sbi, ee_start).wrapping_add(c_offset as u64);
}
let mut __v2: u32;
let __UNIQUE_ID_x__1542: u32 = unsafe { (*map).m_len };
let __UNIQUE_ID_y__1543: u32 = unsafe { (*sbi).s_cluster_ratio }.wrapping_sub(c_offset);
while false {
}
__v2 = if __UNIQUE_ID_x__1542 < __UNIQUE_ID_y__1543 { __UNIQUE_ID_x__1542 } else { __UNIQUE_ID_y__1543 };
unsafe {
(*map).m_len = __v2;
}
// Check for and handle this case:
//
//   |--------- cluster # N-------------|
// 		       |------- extent ----|
// 	   |--- requested region ---|
// 	   |===========|
if (unsafe { (*map).m_lblk }) < ee_block {
let mut __v4: u32;
let __UNIQUE_ID_x__1545: u32 = unsafe { (*map).m_len };
let __UNIQUE_ID_y__1546: u32 = ee_block.wrapping_sub(unsafe { (*map).m_lblk });
while false {
}
__v4 = if __UNIQUE_ID_x__1545 < __UNIQUE_ID_y__1546 { __UNIQUE_ID_x__1545 } else { __UNIQUE_ID_y__1546 };
unsafe {
(*map).m_len = __v4;
}
}
// Check for the case where there is already another allocated
// block to the right of 'ex' but before the end of the cluster.
//
//          |------------- cluster # N-------------|
// |----- ex -----|                  |---- ex_right ----|
//                  |------ requested region ------|
//                  |================|
if (unsafe { (*map).m_lblk }) > ee_block {
let next: u32 = ext4_ext_next_allocated_block(path);
let mut __v6: u32;
let __UNIQUE_ID_x__1548: u32 = unsafe { (*map).m_len };
let __UNIQUE_ID_y__1549: u32 = next.wrapping_sub(unsafe { (*map).m_lblk });
while false {
}
__v6 = if __UNIQUE_ID_x__1548 < __UNIQUE_ID_y__1549 { __UNIQUE_ID_x__1548 } else { __UNIQUE_ID_y__1549 };
unsafe {
(*map).m_len = __v6;
}
}
trace_ext4_get_implied_cluster_alloc_exit(sb, map, 1i32);
return 1;
}
trace_ext4_get_implied_cluster_alloc_exit(sb, map, 0i32);
0
}

// Determine hole length around the given logical block, first try to
// locate and expand the hole from the given @path, and then adjust it
// if it's partially or completely converted to delayed extents, insert
// it into the extent cache tree if it's indeed a hole, finally return
// the length of the determined extent.
fn ext4_ext_determine_insert_hole(inode: *mut inode, path: *mut ext4_ext_path, lblk: u32) -> u32 {
let mut __UNIQUE_ID_y__1552: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__1551: u32 = unsafe { core::mem::zeroed() };
let mut es: core::mem::MaybeUninit<extent_status> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut extent_status = core::ptr::addr_of_mut!(es) as *mut extent_status;
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32;
let mut hole_start: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(hole_start) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
*__slate_slot_10 = lblk;
*__slate_slot_9 = ext4_ext_find_hole(inode, path, core::ptr::addr_of_mut!(*__slate_slot_10));
'__join_2: {
loop {
ext4_es_find_extent_range(inode, Some(ext4_es_is_delayed), *__slate_slot_10, (*__slate_slot_10).wrapping_add(*__slate_slot_9).wrapping_sub(1), core::ptr::addr_of_mut!(*__slate_slot_8));
if (*__slate_slot_8).es_len == 0 {
break '__join_2;
} else {
// There's a delalloc extent in the hole, handle it if the delalloc
// extent is in front of, behind and straddle the queried range.
if lblk >= (*__slate_slot_8).es_lblk.wrapping_add((*__slate_slot_8).es_len) {
// The delalloc extent is in front of the queried range,
// find again from the queried start block.
let __v4: u32 = *__slate_slot_9;
*__slate_slot_9 = __v4.wrapping_sub(lblk.wrapping_sub(*__slate_slot_10));
*__slate_slot_10 = lblk;
} else {
break;
}
}
}
if in_range!(lblk, (*__slate_slot_8).es_lblk, (*__slate_slot_8).es_len) != 0 {
// The delalloc extent containing lblk, it must have been
// added after ext4_map_blocks() checked the extent status
// tree so we are not holding i_rwsem and delalloc info is
// only stabilized by i_data_sem we are going to release
// soon. Don't modify the extent status tree and report
// extent as a hole, just adjust the length to the delalloc
// extent's after lblk.
*__slate_slot_9 = (*__slate_slot_8).es_lblk.wrapping_add((*__slate_slot_8).es_len).wrapping_sub(lblk);
return *__slate_slot_9;
} else {
__UNIQUE_ID_x__1551 = (*__slate_slot_8).es_lblk.wrapping_sub(*__slate_slot_10);
__UNIQUE_ID_y__1552 = *__slate_slot_9;
while false {
}
// The delalloc extent is partially or completely behind
// the queried range, update hole length until the
// beginning of the delalloc extent.
*__slate_slot_9 = if __UNIQUE_ID_x__1551 < __UNIQUE_ID_y__1552 { __UNIQUE_ID_x__1551 } else { __UNIQUE_ID_y__1552 };
}
}
// Put just found gap into cache to speed up subsequent requests
ext4_es_cache_extent(inode, *__slate_slot_10, *__slate_slot_9, 18446744073709551615u64, 8u32);
// Update hole_len to reflect hole size after lblk
if *__slate_slot_10 != lblk {
let __v1: u32 = *__slate_slot_9;
*__slate_slot_9 = __v1.wrapping_sub(lblk.wrapping_sub(*__slate_slot_10));
}
return *__slate_slot_9;
}
}

/// Block allocation/map/preallocation routine for extents based files
///
///
/// Need to be called with
/// down_read(&EXT4_I(inode)->i_data_sem) if not allocating file system block
/// (ie, flags is zero). Otherwise down_write(&EXT4_I(inode)->i_data_sem)
///
/// return > 0, number of blocks already mapped/allocated
///          if flags doesn't contain EXT4_GET_BLOCKS_CREATE and these are pre-allocated blocks
///          	buffer head is unmapped
///          otherwise blocks are mapped
///
/// return = 0, if plain look up failed (blocks have not been allocated)
///          buffer head is unmapped
///
/// return < 0, error case.
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_map_blocks(handle: *mut jbd2_journal_handle, inode: *mut inode, map: *mut ext4_map_blocks, flags: i32) -> i32 {
let mut fb_flags: i32 = unsafe { core::mem::zeroed() };
// Check if the extent after searching to the right implies a
// cluster we can use.
let mut __v25: bool = unsafe { core::mem::zeroed() };
// If we are doing bigalloc, check to see if the extent returned
// by ext4_find_extent() implies a cluster we can use.
let mut __v27: bool = unsafe { core::mem::zeroed() };
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32;
let mut ee_len: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u16 = core::ptr::addr_of_mut!(ee_len) as *mut u16;
let mut ee_start: u64 = unsafe { core::mem::zeroed() };
let mut ee_block: u32 = unsafe { core::mem::zeroed() };
let mut cluster_offset: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut u32 = core::ptr::addr_of_mut!(cluster_offset) as *mut u32;
let mut ar: core::mem::MaybeUninit<ext4_allocation_request> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut ext4_allocation_request = core::ptr::addr_of_mut!(ar) as *mut ext4_allocation_request;
let mut allocated_clusters: u32 = unsafe { core::mem::zeroed() };
let mut offset: u32 = unsafe { core::mem::zeroed() };
let mut allocated: u32 = unsafe { core::mem::zeroed() };
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut pblk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut u64 = core::ptr::addr_of_mut!(pblk) as *mut u64;
let mut newblock: u64 = unsafe { core::mem::zeroed() };
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
let mut ex2: core::mem::MaybeUninit<ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_47: *mut ext4_extent = core::ptr::addr_of_mut!(ex2) as *mut ext4_extent;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut newex: core::mem::MaybeUninit<ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_49: *mut ext4_extent = core::ptr::addr_of_mut!(newex) as *mut ext4_extent;
let mut path: *mut ext4_ext_path = unsafe { core::mem::zeroed() };
unsafe {
'__join_7: {
path = core::ptr::null_mut::<ext4_ext_path>();
let __slate_auto_bytes_1 = __slate_slot_49 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 12usize);
let __slate_auto_bytes_2 = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_47 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 12usize);
sbi = EXT4_SB((*inode).i_sb);
newblock = 0;
let __slate_auto_bytes_4 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
err = 0;
let __slate_auto_bytes_5 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
allocated = 0;
offset = 0;
allocated_clusters = 0;
let __slate_auto_bytes_6 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 56usize);
let __slate_auto_bytes_7 = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
trace_ext4_ext_map_blocks_enter(inode, (*map).m_lblk, (*map).m_len, flags as u32);
// find extent for this block
path = ext4_find_extent(inode, (*map).m_lblk, core::ptr::null_mut::<ext4_ext_path>(), flags);
if IS_ERR(path as *const ()) {
err = PTR_ERR(path as *const ()) as i32;
} else {
*__slate_slot_42 = ext_depth(inode) as i32;
// consistent leaf must not be empty;
// this situation is possible, though, _during_ tree modification;
// this is why assert can't be put in ext4_find_extent()
if unlikely!(unsafe { (*path.wrapping_offset(*__slate_slot_42 as isize)).p_ext }.is_null() && *__slate_slot_42 != 0) != 0 {
__ext4_error_inode(inode, c"ext4_ext_map_blocks".as_ptr() as *const u8, 4311u32, 0u64, 0i32, c"bad extent address lblock: %lu, depth: %d pblock %lld".as_ptr() as *const u8, (*map).m_lblk as u64, *__slate_slot_42, (*path.wrapping_offset(*__slate_slot_42 as isize)).p_block);
err = 117i32.wrapping_neg();
} else {
*__slate_slot_48 = (*path.wrapping_offset(*__slate_slot_42 as isize)).p_ext;
if !(*__slate_slot_48).is_null() {
ee_block = (*(*__slate_slot_48)).ee_block;
ee_start = ext4_ext_pblock(*__slate_slot_48);
let __slate_auto_bytes = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
// unwritten extents are treated as holes, except that
// we split out initialized portions during a write.
*__slate_slot_33 = ext4_ext_get_actual_len(*__slate_slot_48) as u16;
trace_ext4_ext_show_extent(inode, ee_block, ee_start, *__slate_slot_33);
// if found extent covers block, simply return it
if in_range__5!(unsafe { (*map).m_lblk }, ee_block, *__slate_slot_33) != 0 {
newblock = (unsafe { (*map).m_lblk }.wrapping_sub(ee_block) as u64).wrapping_add(ee_start);
// number of remaining blocks in the extent
allocated = ((*__slate_slot_33 as i32) as u32).wrapping_sub(unsafe { (*map).m_lblk }.wrapping_sub(ee_block));
// If the extent is initialized check whether the
// caller wants to convert it to unwritten.
if ext4_ext_is_unwritten(*__slate_slot_48) == 0 && flags & 256 != 0 {
path = convert_initialized_extent(handle, inode, map, path, flags, &mut (*core::ptr::addr_of_mut!(allocated)));
if IS_ERR(path as *const ()) {
err = PTR_ERR(path as *const ()) as i32;
break '__join_7;
} else {
break '__join_7;
}
} else {
if ext4_ext_is_unwritten(*__slate_slot_48) == 0 {
let __v30: u32 = (*map).m_flags;
let __v29: u32 = ((__v30 as u64) | BIT!(4i32)) as u32;
(*map).m_flags = __v29;
(*map).m_pblk = newblock;
if allocated > (*map).m_len {
allocated = (*map).m_len;
}
(*map).m_len = allocated;
break '__join_7;
} else {
path = ext4_ext_handle_unwritten_extents(handle, inode, map, path, flags, core::ptr::addr_of_mut!(allocated), newblock);
if IS_ERR(path as *const ()) {
err = PTR_ERR(path as *const ()) as i32;
break '__join_7;
} else {
break '__join_7;
}
}
}
}
}
// requested block isn't allocated yet;
// we couldn't try to create block if flags doesn't contain EXT4_GET_BLOCKS_CREATE
if flags & 1 == 0 {
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_28 = ext4_ext_determine_insert_hole(inode, path, (*map).m_lblk);
(*map).m_pblk = 0;
(*map).m_len = min_t!(u32, unsafe { (*map).m_len }, *__slate_slot_28);
} else {
// Okay, we need to do block allocation.
(*__slate_slot_49).ee_block = (*map).m_lblk;
*__slate_slot_37 = EXT4_LBLK_COFF!(sbi, unsafe { (*map).m_lblk });
if *__slate_slot_37 != 0 && !(*__slate_slot_48).is_null() {
__v27 = get_implied_cluster_alloc((*inode).i_sb, map, *__slate_slot_48, path) != 0;
} else {
__v27 = false;
}
if __v27 {
let __v26: u32 = (*map).m_len;
allocated = __v26;
(*__slate_slot_38).len = __v26;
newblock = (*map).m_pblk;
} else {
// find neighbour allocated blocks
(*__slate_slot_38).lleft = (*map).m_lblk;
err = ext4_ext_search_left(inode, path, core::ptr::addr_of_mut!((*__slate_slot_38).lleft), core::ptr::addr_of_mut!((*__slate_slot_38).pleft));
if err != 0 {
break '__join_7;
} else {
(*__slate_slot_38).lright = (*map).m_lblk;
err = ext4_ext_search_right(inode, path, core::ptr::addr_of_mut!((*__slate_slot_38).lright), core::ptr::addr_of_mut!((*__slate_slot_38).pright), core::ptr::addr_of_mut!(*__slate_slot_47), flags);
if err < 0 {
break '__join_7;
} else {
if (*sbi).s_cluster_ratio > 1 && err != 0 {
__v25 = get_implied_cluster_alloc((*inode).i_sb, map, core::ptr::addr_of_mut!(*__slate_slot_47), path) != 0;
} else {
__v25 = false;
}
if __v25 {
let __v24: u32 = (*map).m_len;
allocated = __v24;
(*__slate_slot_38).len = __v24;
newblock = (*map).m_pblk;
err = 0;
} else {
// See if request is beyond maximum number of blocks we can have in
// a single extent. For an initialized extent this limit is
// EXT_INIT_MAX_LEN and for an unwritten extent this limit is
// EXT_UNWRITTEN_MAX_LEN.
if ((*map).m_len as u64) > 32768u64 && flags & 2 == 0 {
(*map).m_len = 32768u32;
} else {
if ((*map).m_len as u64) > 32767u64 && flags & 2 != 0 {
(*map).m_len = 32767u32;
}
}
// Check if we can really insert (m_lblk)::(m_lblk + m_len) extent
(*__slate_slot_49).ee_len = (*map).m_len as u16;
err = ext4_ext_check_overlap(sbi, inode, core::ptr::addr_of_mut!(*__slate_slot_49), path) as i32;
if err != 0 {
allocated = ext4_ext_get_actual_len(core::ptr::addr_of_mut!(*__slate_slot_49)) as u32;
} else {
allocated = (*map).m_len;
}
// allocate new block
(*__slate_slot_38).inode = inode;
(*__slate_slot_38).goal = ext4_ext_find_goal(inode, path, (*map).m_lblk);
(*__slate_slot_38).logical = (*map).m_lblk;
// We calculate the offset from the beginning of the cluster
// for the logical block number, since when we allocate a
// physical cluster, the physical block should start at the
// same offset from the beginning of the cluster.  This is
// needed so that future calls to get_implied_cluster_alloc()
// work correctly.
offset = EXT4_LBLK_COFF!(sbi, unsafe { (*map).m_lblk });
(*__slate_slot_38).len = EXT4_NUM_B2C__2!(sbi, offset.wrapping_add(allocated));
let __v23: u64 = (*__slate_slot_38).goal;
(*__slate_slot_38).goal = __v23.wrapping_sub(offset as u64);
let __v21: u32 = (*__slate_slot_38).logical;
(*__slate_slot_38).logical = __v21.wrapping_sub(offset);
if S_ISREG!(unsafe { (*inode).i_mode }) {
(*__slate_slot_38).flags = 32;
} else {
(*__slate_slot_38).flags = 0;
}
// disable in-core preallocation for non-regular files
if flags & 64 != 0 {
(*__slate_slot_38).flags |= 64;
}
if flags & 4 != 0 {
(*__slate_slot_38).flags |= 1024;
}
if flags & 32 != 0 {
(*__slate_slot_38).flags |= 8192;
}
newblock = ext4_mb_new_blocks(handle, core::ptr::addr_of_mut!(*__slate_slot_38), core::ptr::addr_of_mut!(err));
if newblock == 0 {
break '__join_7;
} else {
allocated_clusters = (*__slate_slot_38).len;
(*__slate_slot_38).len = EXT4_C2B!(sbi, (*__slate_slot_38).len).wrapping_sub(offset);
if (*__slate_slot_38).len > allocated {
(*__slate_slot_38).len = allocated;
}
}
}
}
}
}
*__slate_slot_44 = newblock.wrapping_add(offset as u64); // try to insert new extent into found leaf and return
ext4_ext_store_pblock(core::ptr::addr_of_mut!(*__slate_slot_49), *__slate_slot_44);
(*__slate_slot_49).ee_len = (*__slate_slot_38).len as u16;
// Mark unwritten
if flags & 2 != 0 {
ext4_ext_mark_unwritten(core::ptr::addr_of_mut!(*__slate_slot_49));
let __v11: u32 = (*map).m_flags;
let __v10: u32 = ((__v11 as u64) | BIT!(11i32)) as u32;
(*map).m_flags = __v10;
}
path = ext4_ext_insert_extent(handle, inode, path, core::ptr::addr_of_mut!(*__slate_slot_49), flags);
if IS_ERR(path as *const ()) {
err = PTR_ERR(path as *const ()) as i32;
// Gracefully handle out of space conditions. If the filesystem
// is inconsistent, we'll just leak allocated blocks to avoid
// causing even more damage.
if allocated_clusters != 0 && (err == 122i32.wrapping_neg() || err == 28i32.wrapping_neg()) {
fb_flags = 0;
// free data blocks we just allocated.
// not a good idea to call discard here directly,
// but otherwise we'd need to call it every free().
ext4_discard_preallocations(inode);
if flags & 4 != 0 {
fb_flags = 8;
}
ext4_free_blocks(handle, inode, core::ptr::null_mut::<buffer_head>(), newblock, EXT4_C2B!(sbi, allocated_clusters) as u64, fb_flags);
}
} else {
// Cache the extent and update transaction to commit on fdatasync only
// when it is _not_ an unwritten extent.
if flags & 2 == 0 {
ext4_update_inode_fsync_trans(handle, inode, 1i32);
} else {
ext4_update_inode_fsync_trans(handle, inode, 0i32);
}
let __v7: u32 = (*map).m_flags;
let __v6: u32 = ((__v7 as u64) | (BIT!(5i32) | BIT!(4i32))) as u32;
(*map).m_flags = __v6;
(*map).m_pblk = *__slate_slot_44;
(*map).m_len = (*__slate_slot_38).len;
allocated = (*map).m_len;
}
}
}
}
}
if flags & 4096 != 0 {
let __ret_warn_on: i32 = (flags & 1 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("1556: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1556b - ., 3; .popsection\n# {0}", const 1556 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_ext_map_blocks_51, sym __slate_asm_string_ext4_ext_map_blocks_52, const 4539 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1557: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1557b - ., 4; .popsection\n# {0}", const 1557 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if err == 0 && !(*__slate_slot_48).is_null() && *__slate_slot_48 == EXT_LAST_EXTENT!(unsafe { (*path.wrapping_offset(*__slate_slot_42 as isize)).p_hdr }) {
let __v1: u32 = (*map).m_flags;
let __v0: u32 = ((__v1 as u64) | BIT!(27i32)) as u32;
(*map).m_flags = __v0;
}
}
// We never use EXT4_GET_BLOCKS_QUERY_LAST_IN_LEAF with CREATE flag.
// So we know that the depth used here is correct, since there was no
// block allocation done if EXT4_GET_BLOCKS_QUERY_LAST_IN_LEAF is set.
// If tomorrow we start using this QUERY flag with CREATE, then we will
// need to re-calculate the depth as it might have changed due to block
// allocation.
ext4_free_ext_path(path);
trace_ext4_ext_map_blocks_exit(inode, flags as u32, map, (if err != 0 { err as u32 } else { allocated }) as i32);
return (if err != 0 { err as u32 } else { allocated }) as i32;
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_ext_truncate(handle: *mut jbd2_journal_handle, inode: *mut inode) -> i32 {
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut last_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(last_block) as *mut u32;
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
unsafe {
sb = (*inode).i_sb;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
err = 0;
// TODO: optimization is possible here.
// Probably we need not scan at all,
// because page truncation is enough.
// we have to know where to truncate from in crash case
(*EXT4_I!(inode)).i_disksize = (*inode).i_size;
err = ext4_mark_inode_dirty__42!(handle, inode);
if err != 0 {
return err;
} else {
*__slate_slot_1 = (((*inode).i_size as u64).wrapping_add((*sb).s_blocksize).wrapping_sub(1) >> (EXT4_BLOCK_SIZE_BITS!(sb) as i32)) as u32;
ext4_es_remove_extent(inode, *__slate_slot_1, 4294967295u32.wrapping_sub(*__slate_slot_1));
loop {
err = ext4_ext_remove_space(inode, *__slate_slot_1, 4294967294u32);
if err == 12i32.wrapping_neg() {
memalloc_retry_wait((BIT!(5i32) as u32) | (BIT!(11i32) as u32));
} else {
break;
}
}
return err;
}
}
}

/// Pre-allocate blocks for the range [@offset, @offset + @len). Allocated
/// blocks are marked as unwritten by default. If EXT4_GET_BLOCKS_ZERO is
/// set, the allocated blocks are zeroed on disk and their extents are
/// converted to written state.
///
/// When @new_size is nonzero, the caller intends to extend the file, and
/// the file size should be updated to the end of the allocated blocks.
///
/// Allocation may partially succeed due to some non-fatal issues. In that
/// case, i_disksize (and i_size) is advanced up to the successfully
/// processed portion of the range.
///
/// Return 0 on success, or a negative error code on failure or partial
/// failure.
fn ext4_alloc_file_blocks(file: *mut file, offset: i64, len: i64, new_size: i64, mut flags: i32) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut converted: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(converted) as *mut u32;
let mut orphan: bool = unsafe { core::mem::zeroed() };
let mut alloc_zero: bool = unsafe { core::mem::zeroed() };
let mut blkbits: u32 = unsafe { core::mem::zeroed() };
let mut old_size: i64 = unsafe { core::mem::zeroed() };
let mut epos: i64 = unsafe { core::mem::zeroed() };
let mut credits: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(credits) as *mut u32;
let mut map: core::mem::MaybeUninit<ext4_map_blocks> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut ext4_map_blocks = core::ptr::addr_of_mut!(map) as *mut ext4_map_blocks;
let mut len_lblk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u32 = core::ptr::addr_of_mut!(len_lblk) as *mut u32;
let mut depth: i32 = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut ret3: i32 = unsafe { core::mem::zeroed() };
let mut ret2: i32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
unsafe {
inode = file_inode(file as *const file);
let __slate_auto_bytes_1 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
ret = 0;
ret2 = 0;
ret3 = 0;
retries = 0;
depth = 0;
let __slate_auto_bytes_2 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_4 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
epos = 0;
old_size = i_size_read(inode as *const inode);
blkbits = (*inode).i_blkbits as u32;
alloc_zero = false;
orphan = false;
'__join_50: {
loop {
if unlikely!(ext4_test_inode_flag(inode, 19i32) == 0) != 0 {
break '__join_50;
} else {
if true {
break;
}
}
}
(*__slate_slot_23).m_lblk = (offset >> blkbits) as u32;
let __v16: u32 = EXT4_MAX_BLOCKS!(len, offset, blkbits) as u32;
*__slate_slot_24 = __v16;
(*__slate_slot_23).m_len = __v16;
// Don't normalize the request if it can fit in one extent so
// that it doesn't get unnecessarily split into multiple
// extents.
if (*__slate_slot_24 as u64) <= 32767u64 {
flags |= 64;
}
// Do the actual write zero during a running journal transaction
// costs a lot. First allocate an unwritten extent and then
// convert it to written after zeroing it out.
if flags & 512 != 0 {
let __v12: i32 = flags & !512;
flags = __v12;
flags |= 2;
alloc_zero = true;
}
// credits to insert 1 extent into extent tree
*__slate_slot_22 = ext4_chunk_trans_blocks(inode, *__slate_slot_24 as i32) as u32;
depth = ext_depth(inode) as i32;
// Zero to the end of the block containing i_size
if new_size > old_size {
ret = ext4_block_zero_eof(inode, old_size, 9223372036854775807i64);
if ret != 0 {
return ret;
}
}
'__loop_19: loop {
'__join_18: {
if *__slate_slot_24 != 0 {
// Recalculate credits when extent tree depth changes.
if depth != (ext_depth(inode) as i32) {
*__slate_slot_22 = ext4_chunk_trans_blocks(inode, *__slate_slot_24 as i32) as u32;
depth = ext_depth(inode) as i32;
}
*__slate_slot_30 = ext4_journal_start__45!(inode, 3i32, *__slate_slot_22);
if IS_ERR(*__slate_slot_30 as *const ()) {
ret = PTR_ERR(*__slate_slot_30 as *const ()) as i32;
} else {
ret = ext4_map_blocks(*__slate_slot_30, inode, core::ptr::addr_of_mut!(*__slate_slot_23), flags);
if ret <= 0 {
ext4_mark_inode_dirty__31!(*__slate_slot_30, inode);
ext4_journal_stop__12!(*__slate_slot_30);
} else {
ext4_update_inode_fsync_trans(*__slate_slot_30, inode, 1i32);
ret = ext4_journal_stop__73!(*__slate_slot_30);
if unlikely__2!(ret) == 0 {
// allow a full retry cycle for any remaining allocations
retries = 0;
if alloc_zero && ((*__slate_slot_23).m_flags as u64) & (BIT!(4i32) | BIT!(11i32)) != 0 {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __v5: i64 = new_size;
let __ret_warn_on: i32 = (((*__slate_slot_23).m_lblk.wrapping_add((*__slate_slot_23).m_len) as i64) > (if __v5 != 0 { __v5 } else { old_size }.wrapping_sub(1) | __round_mask!(i_blocksize(inode as *const inode))).wrapping_add(1) >> ((*inode).i_blkbits as i32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("1559: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1559b - ., 3; .popsection\n# {0}", const 1559 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_alloc_file_blocks_33, sym __slate_asm_string_ext4_alloc_file_blocks_34, const 4690 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1560: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1560b - ., 4; .popsection\n# {0}", const 1560 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
ret = ext4_issue_zeroout(inode, (*__slate_slot_23).m_lblk, (*__slate_slot_23).m_pblk, (*__slate_slot_23).m_len);
if unlikely__2!(ret) != 0 {
break '__join_18;
} else {
*__slate_slot_30 = ext4_journal_start__25!(inode, 3i32, *__slate_slot_22);
if IS_ERR(*__slate_slot_30 as *const ()) {
ret = PTR_ERR(*__slate_slot_30 as *const ()) as i32;
break '__join_18;
} else {
ret = ext4_convert_unwritten_extents(*__slate_slot_30, inode, ((*__slate_slot_23).m_lblk as i64) << blkbits, ((*__slate_slot_23).m_len as i64) << blkbits, core::ptr::addr_of_mut!(*__slate_slot_8));
if ret != 0 {
(*__slate_slot_23).m_len = *__slate_slot_8;
}
// If blocks beyond i_disksize are converted, add
// the inode to the orphan list and advance the epos.
if new_size != 0 && *__slate_slot_8 != 0 {
ret2 = ext4_orphan_add(*__slate_slot_30, inode);
ret = if ret != 0 { ret } else { ret2 };
orphan = true;
}
ret3 = ext4_journal_stop__72!(*__slate_slot_30);
ret = if ret != 0 { ret } else { ret3 };
}
}
}
let __v3: u32 = (*__slate_slot_23).m_lblk;
(*__slate_slot_23).m_lblk = __v3.wrapping_add((*__slate_slot_23).m_len);
let __v1: u32 = (*__slate_slot_24).wrapping_sub((*__slate_slot_23).m_len);
*__slate_slot_24 = __v1;
(*__slate_slot_23).m_len = __v1;
epos = EXT4_LBLK_TO_B!(inode, (*__slate_slot_23).m_lblk);
if ret == 0 {
continue '__loop_19;
}
}
}
}
}
}
if ret == 28i32.wrapping_neg() {
__v0 = ext4_should_retry_alloc((*inode).i_sb, core::ptr::addr_of_mut!(retries)) != 0;
} else {
__v0 = false;
}
if !__v0 {
break;
}
}
if epos == 0 || new_size == 0 {
return ret;
} else {
// Allocate blocks, update the file size to match the size of the
// already successfully allocated blocks.
if epos > new_size {
epos = new_size;
}
*__slate_slot_30 = ext4_journal_start__17!(inode, 0i32, 2i32);
if IS_ERR(*__slate_slot_30 as *const ()) {
// The conversion has successfully completed. Not much to
// do with the error here so just cleanup the orphan list
// and hope for the best.
if orphan && (*inode).__slate_anon_12.i_nlink != 0 {
ext4_orphan_del(core::ptr::null_mut::<jbd2_journal_handle>(), inode);
}
ret2 = PTR_ERR(*__slate_slot_30 as *const ()) as i32;
} else {
ext4_update_inode_size(inode, epos);
if orphan && (*inode).__slate_anon_12.i_nlink != 0 {
ext4_orphan_del(*__slate_slot_30, inode);
}
ret2 = ext4_mark_inode_dirty__5!(*__slate_slot_30, inode);
ext4_update_inode_fsync_trans(*__slate_slot_30, inode, 1i32);
ret3 = ext4_journal_stop__56!(*__slate_slot_30);
ret2 = if ret3 != 0 { ret3 } else { ret2 };
if epos > old_size {
pagecache_isize_extended(inode, old_size, epos);
}
}
if ret2 != 0 {
ext4_std_error__45!(unsafe { (*inode).i_sb }, ret2);
}
return if ret != 0 { ret } else { ret2 };
}
}
core::arch::asm!("1558: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1558b - ., 3; .popsection\n# {0}", const 1558 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_alloc_file_blocks_35, sym __slate_asm_string_ext4_alloc_file_blocks_36, const 4614 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

fn ext4_insert_range(file: *mut file, offset: i64, len: i64) -> i32 {
let mut __UNIQUE_ID_y__1570: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__1569: u32 = unsafe { core::mem::zeroed() };
let mut start: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i64 = core::ptr::addr_of_mut!(start) as *mut i64;
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut ee_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(ee_len) as *mut u32;
let mut credits: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u32 = core::ptr::addr_of_mut!(credits) as *mut u32;
let mut ee_start_lblk: u32 = unsafe { core::mem::zeroed() };
let mut len_lblk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u32 = core::ptr::addr_of_mut!(len_lblk) as *mut u32;
let mut start_lblk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u32 = core::ptr::addr_of_mut!(start_lblk) as *mut u32;
let mut extent: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut ext4_extent = core::ptr::addr_of_mut!(extent) as *mut *mut ext4_extent;
let mut path: core::mem::MaybeUninit<*mut ext4_ext_path> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut ext4_ext_path = core::ptr::addr_of_mut!(path) as *mut *mut ext4_ext_path;
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
let mut mapping: *mut address_space = unsafe { core::mem::zeroed() };
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
unsafe {
inode = file_inode(file as *const file);
sb = (*inode).i_sb;
mapping = (*inode).i_mapping;
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
ee_start_lblk = 0;
let __slate_auto_bytes_5 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
trace_ext4_insert_range(inode, offset, len);
let __ret_warn_on: i32 = (inode_is_locked(inode) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("1567: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1567b - ., 3; .popsection\n# {0}", const 1567 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_insert_range_31, sym __slate_asm_string_ext4_insert_range_32, const 5752 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1568: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1568b - ., 4; .popsection\n# {0}", const 1568 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// Currently just for extent based files
if ext4_test_inode_flag(inode, 19i32) == 0 {
return 95i32.wrapping_neg();
} else {
// Insert range works only on fs cluster size aligned regions.
if !IS_ALIGNED!(offset | len, EXT4_CLUSTER_SIZE!(sb)) {
return 22i32.wrapping_neg();
} else {
// Offset must be less than i_size
if offset >= (*inode).i_size {
return 22i32.wrapping_neg();
} else {
// Check whether the maximum file size would be exceeded
if len > unsafe { (*(*inode).i_sb).s_maxbytes }.wrapping_sub((*inode).i_size) {
return 27i32.wrapping_neg();
} else {
// Write out all dirty pages. Need to round down to align start offset
// to page size boundary for page size > block size.
*__slate_slot_12 = round_down!(offset, _AC!() << 12i32);
*__slate_slot_14 = filemap_write_and_wait_range(mapping, *__slate_slot_12, 9223372036854775807i64);
if *__slate_slot_14 != 0 {
return *__slate_slot_14;
} else {
truncate_pagecache(inode, *__slate_slot_12);
*__slate_slot_16 = ext4_chunk_trans_extent(inode, 0i32) as u32;
*__slate_slot_22 = ext4_journal_start__7!(inode, 5i32, *__slate_slot_16);
if IS_ERR(*__slate_slot_22 as *const ()) {
return PTR_ERR(*__slate_slot_22 as *const ()) as i32;
} else {
'__join_0: {
ext4_fc_mark_ineligible(sb, 7i32, *__slate_slot_22);
// Expand file to avoid data loss if there is error while shifting
let __v7: i64 = (*inode).i_size;
(*inode).i_size = __v7.wrapping_add(len);
let __v5: *mut ext4_inode_info = EXT4_I!(inode);
let __v4: i64 = (*__v5).i_disksize;
(*__v5).i_disksize = __v4.wrapping_add(len);
*__slate_slot_14 = ext4_mark_inode_dirty__11!(*__slate_slot_22, inode);
if *__slate_slot_14 == 0 {
*__slate_slot_19 = (offset >> ((*inode).i_blkbits as i32)) as u32;
*__slate_slot_18 = (len >> ((*inode).i_blkbits as i32)) as u32;
ext4_check_map_extents_env(inode);
down_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
ext4_discard_preallocations(inode);
*__slate_slot_21 = ext4_find_extent(inode, *__slate_slot_19, core::ptr::null_mut::<ext4_ext_path>(), 0i32);
if IS_ERR(*__slate_slot_21 as *const ()) {
up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
*__slate_slot_14 = PTR_ERR(*__slate_slot_21 as *const ()) as i32;
} else {
*__slate_slot_13 = ext_depth(inode) as i32;
*__slate_slot_20 = (*(*__slate_slot_21).wrapping_offset(*__slate_slot_13 as isize)).p_ext;
if !(*__slate_slot_20).is_null() {
ee_start_lblk = (*(*__slate_slot_20)).ee_block;
*__slate_slot_15 = ext4_ext_get_actual_len(*__slate_slot_20) as u32;
// If start_lblk is not the starting block of extent, split
// the extent @start_lblk
if *__slate_slot_19 > ee_start_lblk && *__slate_slot_19 < ee_start_lblk.wrapping_add(*__slate_slot_15) {
*__slate_slot_21 = ext4_split_extent_at(*__slate_slot_22, inode, *__slate_slot_21, *__slate_slot_19, 1073741864i32);
}
if IS_ERR(*__slate_slot_21 as *const ()) {
up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
*__slate_slot_14 = PTR_ERR(*__slate_slot_21 as *const ()) as i32;
break '__join_0;
}
}
ext4_free_ext_path(*__slate_slot_21);
ext4_es_remove_extent(inode, *__slate_slot_19, 4294967295u32.wrapping_sub(*__slate_slot_19));
__UNIQUE_ID_x__1569 = ee_start_lblk;
__UNIQUE_ID_y__1570 = *__slate_slot_19;
while false {
}
// if start_lblk lies in a hole which is at start of file, use
// ee_start_lblk to shift extents
let __v2: u32 = if __UNIQUE_ID_x__1569 > __UNIQUE_ID_y__1570 { __UNIQUE_ID_x__1569 } else { __UNIQUE_ID_y__1570 };
*__slate_slot_14 = ext4_ext_shift_extents(inode, *__slate_slot_22, __v2, *__slate_slot_18, 1u32);
up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
if *__slate_slot_14 == 0 {
ext4_update_inode_fsync_trans(*__slate_slot_22, inode, 1i32);
if (*file).f_flags & 1052672u32 != 0 || IS_SYNC!(inode) {
ext4_handle_sync(*__slate_slot_22);
}
}
}
}
}
ext4_journal_stop__55!(*__slate_slot_22);
return *__slate_slot_14;
}
}
}
}
}
}
}
}

fn ext4_zero_range(file: *mut file, offset: i64, len: i64, mode: i32) -> i64 {
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(flags) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut partial_zeroed: u32 = unsafe { core::mem::zeroed() };
let mut blocksize: u32 = unsafe { core::mem::zeroed() };
let mut end: i64 = unsafe { core::mem::zeroed() };
let mut new_size: i64 = unsafe { core::mem::zeroed() };
let mut align_end: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i64 = core::ptr::addr_of_mut!(align_end) as *mut i64;
let mut align_start: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i64 = core::ptr::addr_of_mut!(align_start) as *mut i64;
let mut handle: *mut jbd2_journal_handle = unsafe { core::mem::zeroed() };
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
unsafe {
inode = file_inode(file as *const file);
handle = core::ptr::null_mut::<jbd2_journal_handle>();
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
new_size = 0;
end = offset.wrapping_add(len);
blocksize = i_blocksize(inode as *const inode);
partial_zeroed = 0;
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
trace_ext4_zero_range(inode, offset, len, mode);
let __ret_warn_on: i32 = (inode_is_locked(inode) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("1561: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1561b - ., 3; .popsection\n# {0}", const 1561 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_zero_range_19, sym __slate_asm_string_ext4_zero_range_20, const 4792 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1562: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1562b - ., 4; .popsection\n# {0}", const 1562 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// Indirect files do not support unwritten extents
if ext4_test_inode_flag(inode, 19i32) == 0 {
return 95i32.wrapping_neg() as i64;
} else {
if mode & 1 == 0 {
if end > (*inode).i_size {
__v2 = true;
} else {
__v2 = end > (*EXT4_I!(inode)).i_disksize;
}
__v3 = __v2;
} else {
__v3 = false;
}
if __v3 {
new_size = end;
*__slate_slot_8 = inode_newsize_ok(inode as *const inode, new_size);
if *__slate_slot_8 != 0 {
return *__slate_slot_8 as i64;
}
}
*__slate_slot_7 = 2 | 1;
// Preallocate the range including the unaligned edges, and zero
// out partial blocks if they already contain data.
if !IS_ALIGNED!(offset | end, blocksize) {
*__slate_slot_8 = ext4_alloc_file_blocks(file, offset, len, new_size, *__slate_slot_7);
if *__slate_slot_8 == 0 {
*__slate_slot_8 = ext4_zero_partial_blocks(inode, offset, len, core::ptr::addr_of_mut!(partial_zeroed));
}
if *__slate_slot_8 != 0 {
return *__slate_slot_8 as i64;
}
}
*__slate_slot_8 = ext4_update_disksize_before_punch(inode, offset, len);
if *__slate_slot_8 != 0 {
return *__slate_slot_8 as i64;
} else {
// Now release the pages and zero block aligned part of pages
*__slate_slot_8 = ext4_truncate_page_cache_block_range(inode, offset, end);
if *__slate_slot_8 != 0 {
return *__slate_slot_8 as i64;
} else {
// Zero range excluding the unaligned edges
*__slate_slot_14 = round_up!(offset, blocksize);
*__slate_slot_13 = round_down__2!(end, blocksize);
// In WRITE_ZEROES mode, edges that were not partial-zeroed (clean
// unwritten or hole) must be allocated and zeroed as whole blocks.
// Expand the aligned range outward to cover them.
if mode & 128 != 0 {
if !IS_ALIGNED!(offset, blocksize) && partial_zeroed & 1 == 0 {
*__slate_slot_14 = round_down__2!(offset, blocksize);
}
if !IS_ALIGNED!(end, blocksize) && partial_zeroed & 2 == 0 {
*__slate_slot_13 = round_up!(end, blocksize);
}
}
if *__slate_slot_13 > *__slate_slot_14 {
if mode & 128 != 0 {
*__slate_slot_7 = 1 | 512 | 1073741824;
} else {
*__slate_slot_7 |= 256 | 1073741824;
}
*__slate_slot_8 = ext4_alloc_file_blocks(file, *__slate_slot_14, (*__slate_slot_13).wrapping_sub(*__slate_slot_14), new_size, *__slate_slot_7);
if *__slate_slot_8 != 0 {
return *__slate_slot_8 as i64;
}
}
// Finish zeroing out if it doesn't contain partial block
if IS_ALIGNED!(offset | end, blocksize) {
return *__slate_slot_8 as i64;
} else {
// In FALLOC_FL_WRITE_ZEROES mode, edges that have been partially
// zeroed must be written back to ensure the entire zeroed range
// is converted to the written state. In SYNC mode, writeback is
// also required to persist the zeroed data to disk.
if partial_zeroed != 0 && (mode & 128 != 0 || (*file).f_flags & 1052672u32 != 0 || IS_SYNC!(inode)) {
*__slate_slot_8 = filemap_write_and_wait_range((*inode).i_mapping, offset, end.wrapping_sub(1));
if *__slate_slot_8 != 0 {
return *__slate_slot_8 as i64;
}
}
handle = ext4_journal_start__10!(inode, 0i32, 1i32);
if IS_ERR(handle as *const ()) {
*__slate_slot_8 = PTR_ERR(handle as *const ()) as i32;
ext4_std_error__32!(unsafe { (*inode).i_sb }, *__slate_slot_8);
return *__slate_slot_8 as i64;
} else {
if new_size != 0 {
ext4_update_inode_size(inode, new_size);
}
*__slate_slot_8 = ext4_mark_inode_dirty__15!(handle, inode);
if unlikely__2!(*__slate_slot_8) == 0 {
ext4_update_inode_fsync_trans(handle, inode, 1i32);
if (*file).f_flags & 1052672u32 != 0 || IS_SYNC!(inode) {
ext4_handle_sync(handle);
}
}
ext4_journal_stop__92!(handle);
return *__slate_slot_8 as i64;
}
}
}
}
}
}
}

fn ext4_do_fallocate(file: *mut file, offset: i64, len: i64, mode: i32) -> i64 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut new_size: i64 = unsafe { core::mem::zeroed() };
let mut end: i64 = unsafe { core::mem::zeroed() };
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
unsafe {
inode = file_inode(file as *const file);
end = offset.wrapping_add(len);
new_size = 0;
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
trace_ext4_fallocate_enter(inode, offset, len, mode);
let __ret_warn_on: i32 = (inode_is_locked(inode) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("1563: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1563b - ., 3; .popsection\n# {0}", const 1563 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_do_fallocate_12, sym __slate_asm_string_ext4_do_fallocate_13, const 4910 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1564: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1564b - ., 4; .popsection\n# {0}", const 1564 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
'__join_0: {
// We only support preallocation for extent-based files only.
if ext4_test_inode_flag(inode, 19i32) == 0 {
*__slate_slot_6 = 95i32.wrapping_neg();
} else {
if mode & 1 == 0 {
if end > (*inode).i_size {
__v1 = true;
} else {
__v1 = end > (*EXT4_I!(inode)).i_disksize;
}
__v2 = __v1;
} else {
__v2 = false;
}
if __v2 {
new_size = end;
*__slate_slot_6 = inode_newsize_ok(inode as *const inode, new_size);
if *__slate_slot_6 != 0 {
break '__join_0;
}
}
*__slate_slot_6 = ext4_alloc_file_blocks(file, offset, len, new_size, 3i32);
if *__slate_slot_6 == 0 {
if (*file).f_flags & 1052672u32 != 0 || IS_SYNC!(inode) {
__v0 = !unsafe { (*EXT4_SB((*inode).i_sb)).s_journal }.is_null();
} else {
__v0 = false;
}
if __v0 {
*__slate_slot_6 = ext4_fc_commit((*EXT4_SB((*inode).i_sb)).s_journal, (*EXT4_I!(inode)).i_sync_tid);
}
}
}
}
trace_ext4_fallocate_exit(inode, offset, EXT4_MAX_BLOCKS__2!(len, offset, unsafe { (*inode).i_blkbits }) as u32, *__slate_slot_6);
return *__slate_slot_6 as i64;
}
}

/// preallocate space for a file. This implements ext4's fallocate file
/// operation, which gets called from sys_fallocate system call.
/// For block-mapped files, posix_fallocate should fall back to the method
/// of writing zeroes to the required new blocks (the same behavior which is
/// expected for file systems which do not support fallocate() system call).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.extents.ext4_fallocate")]
extern "C" fn ext4_fallocate(file: *mut file, mode: i32, offset: i64, len: i64) -> i64 {
// Don't allow writing zeroes if the underlying device does not
// enable the unmap write zeroes operation.
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut mapping: *mut address_space = unsafe { core::mem::zeroed() };
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
unsafe {
inode = file_inode(file as *const file);
mapping = (*file).f_mapping;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Encrypted inodes can't handle collapse range or insert
// range since we would need to re-encrypt blocks with a
// different IV or XTS tweak (which are based on the logical
// block number).
if IS_ENCRYPTED!(inode) != 0 && mode & (8 | 32) != 0 {
return 95i32.wrapping_neg() as i64;
} else {
if mode & 128 != 0 {
__v0 = bdev_write_zeroes_unmap_sectors((*(*inode).i_sb).s_bdev) == 0;
} else {
__v0 = false;
}
if __v0 {
return 95i32.wrapping_neg() as i64;
} else {
// Return error if mode is not supported
if mode & !(1 | 2 | 16 | 8 | 32 | 128) != 0 {
return 95i32.wrapping_neg() as i64;
} else {
inode_lock(inode);
*__slate_slot_1 = ext4_convert_inline_data(inode);
if *__slate_slot_1 == 0 {
// Wait all existing dio workers, newcomers will block on i_rwsem
inode_dio_wait(inode);
*__slate_slot_1 = file_modified(file);
if *__slate_slot_1 == 0 {
if mode & (0 | 2 | 8 | 16 | 32 | 64 | 128) == 0 {
*__slate_slot_1 = ext4_do_fallocate(file, offset, len, mode) as i32;
} else {
'__join_1: {
// Follow-up operations will drop page cache, hold invalidate lock
// to prevent page faults from reinstantiating pages we have
// released from page cache.
filemap_invalidate_lock(mapping);
*__slate_slot_1 = ext4_break_layouts(inode);
if *__slate_slot_1 == 0 {
let __t0: i32 = mode & (0 | 2 | 8 | 16 | 32 | 64 | 128);
if __t0 == 2 {
*__slate_slot_1 = ext4_punch_hole(file, offset, len);
break '__join_1;
} else {
if __t0 == 8 {
*__slate_slot_1 = ext4_collapse_range(file, offset, len);
break '__join_1;
} else {
if __t0 == 32 {
*__slate_slot_1 = ext4_insert_range(file, offset, len);
break '__join_1;
} else {
if __t0 != 16 {
if __t0 != 128 {
*__slate_slot_1 = 95i32.wrapping_neg();
break '__join_1;
}
}
}
}
}
*__slate_slot_1 = ext4_zero_range(file, offset, len, mode) as i32;
}
}
filemap_invalidate_unlock(mapping);
}
}
}
inode_unlock(inode);
return *__slate_slot_1 as i64;
}
}
}
}
}

/// This function converts a range of blocks to written extents. The caller of
/// this function will pass the start offset and the size. all unwritten extents
/// within this range will be converted to written extents.
///
/// This function is called from the direct IO end io call back function for
/// atomic writes, to convert the unwritten extents after IO is completed.
///
/// Note that the requirement for atomic writes is that all conversion should
/// happen atomically in a single fs journal transaction. We mainly only allocate
/// unwritten extents either on a hole on a pre-exiting unwritten extent range in
/// ext4_map_blocks_atomic_write(). The only case where we can have multiple
/// unwritten extents in a range [offset, offset+len) is when there is a split
/// unwritten extent between two leaf nodes which was cached in extent status
/// cache during ext4_iomap_alloc() time. That will allow
/// ext4_map_blocks_atomic_write() to return the unwritten extent range w/o going
/// into the slow path. That means we might need a loop for conversion of this
/// unwritten extent split across leaf block within a single journal transaction.
/// Split extents across leaf nodes is a rare case, but let's still handle that
/// to meet the requirements of multi-fsblock atomic writes.
///
/// Returns 0 on success.
#[unsafe(no_mangle)]
extern "C" fn ext4_convert_unwritten_extents_atomic(mut handle: *mut jbd2_journal_handle, inode: *mut inode, offset: i64, len: i64) -> i32 {
let mut max_blocks: u32 = 0;
let mut ret: i32 = 0;
let mut ret2: i32 = 0;
let mut ret3: i32 = 0;
let mut map: ext4_map_blocks = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(map) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let blkbits: u32 = (unsafe { (*inode).i_blkbits }) as u32;
let mut credits: u32 = 0;
let flags: i32 = 16 | (2 | 1) | 1024 | 1073741824;
map.m_lblk = (offset >> blkbits) as u32;
max_blocks = EXT4_MAX_BLOCKS!(len, offset, blkbits) as u32;
if handle.is_null() {
// TODO: An optimization can be added later by having an extent
// status flag e.g. EXTENT_STATUS_SPLIT_LEAF. If we query that
// it can tell if the extent in the cache is a split extent.
// But for now let's assume pextents as 2 always.
credits = (unsafe { ext4_meta_trans_blocks(inode, max_blocks as i32, 2i32, 0i32) }) as u32;
}
if credits != 0 {
handle = ext4_journal_start__24!(inode, 3i32, credits);
if IS_ERR(handle as *const ()) {
ret = PTR_ERR(handle as *const ()) as i32;
return ret;
}
}
while ret >= 0 && (ret as u32) < max_blocks {
let __v0: u32 = map.m_lblk;
map.m_lblk = __v0.wrapping_add(ret as u32);
let __v3: u32 = max_blocks.wrapping_sub(ret as u32);
max_blocks = __v3;
map.m_len = __v3;
ret = unsafe { ext4_map_blocks(handle, inode, core::ptr::addr_of_mut!(map), flags) };
if (ret as u32) != max_blocks {
unsafe { __ext4_msg((*inode).i_sb, c"\x016".as_ptr() as *const u8, c"inode #%llu: block %u: len %u: split block mapping found for atomic write, ret = %d".as_ptr() as *const u8, (*inode).i_ino, map.m_lblk, map.m_len, ret) };
}
if ret <= 0 {
break;
}
}
ret2 = ext4_mark_inode_dirty__35!(handle, inode);
if credits != 0 {
ret3 = ext4_journal_stop__87!(handle);
if unlikely__2!(ret3) != 0 {
ret2 = ret3;
}
}
if ret <= 0 || ret2 != 0 {
unsafe { __ext4_warning((*inode).i_sb, c"ext4_convert_unwritten_extents_atomic".as_ptr() as *const u8, 5113u32, c"inode #%llu: block %u: len %u: returned %d or %d".as_ptr() as *const u8, (*inode).i_ino, map.m_lblk, map.m_len, ret, ret2) };
}
if ret > 0 { ret2 } else { ret }
}

/// This function convert a range of blocks to written extents
/// The caller of this function will pass the start offset and the size.
/// all unwritten extents within this range will be converted to
/// written extents.
///
/// This function is called from the direct/buffered I/O end io call back
/// function and FALLOC_FL_WRITE_ZEROES, to convert the fallocated
/// unwritten extents after data I/O is completed.
///
/// Returns 0 on full success, or a negative error code on partial
/// success or failure. The number of blocks converted is returned via
/// @converted.
#[unsafe(no_mangle)]
extern "C" fn ext4_convert_unwritten_extents(mut handle: *mut jbd2_journal_handle, inode: *mut inode, offset: i64, len: i64, converted: *mut u32) -> i32 {
let mut max_blocks: u32 = 0;
let mut conv_blocks: u32 = 0;
let mut ret: i32 = 0;
let mut ret2: i32 = 0;
let mut ret3: i32 = 0;
let mut map: ext4_map_blocks = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(map) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let blkbits: u32 = (unsafe { (*inode).i_blkbits }) as u32;
let mut credits: u32 = 0;
map.m_lblk = (offset >> blkbits) as u32;
let __v0: u32 = EXT4_MAX_BLOCKS!(len, offset, blkbits) as u32;
max_blocks = __v0;
map.m_len = __v0;
if handle.is_null() {
// credits to insert 1 extent into extent tree
credits = (unsafe { ext4_chunk_trans_blocks(inode, max_blocks as i32) }) as u32;
}
while max_blocks != 0 {
if credits != 0 {
handle = ext4_journal_start__5!(inode, 3i32, credits);
if IS_ERR(handle as *const ()) {
ret = PTR_ERR(handle as *const ()) as i32;
break;
}
}
// Do not cache any unrelated extents, as it does not hold the
// i_rwsem or invalidate_lock, which could corrupt the extent
// status tree.
ret = unsafe { ext4_map_blocks(handle, inode, core::ptr::addr_of_mut!(map), 1073742867i32) };
if ret <= 0 {
unsafe { __ext4_warning((*inode).i_sb, c"ext4_convert_unwritten_extents".as_ptr() as *const u8, 5172u32, c"inode #%llu: block %u: len %u: ext4_map_blocks returned %d".as_ptr() as *const u8, (*inode).i_ino, map.m_lblk, map.m_len, ret) };
if unlikely!(ret == 0) != 0 {
ret = 22i32.wrapping_neg();
}
} else {
conv_blocks = conv_blocks.wrapping_add(map.m_len);
}
ret2 = ext4_mark_inode_dirty__51!(handle, inode);
if credits != 0 {
ret3 = ext4_journal_stop__27!(handle);
if unlikely__2!(ret3) != 0 {
ret2 = ret3;
}
}
ret = if ret < 0 { ret } else { ret2 };
if ret != 0 {
break;
}
let __v4: u32 = map.m_lblk;
map.m_lblk = __v4.wrapping_add(map.m_len);
let __v7: u32 = max_blocks.wrapping_sub(map.m_len);
max_blocks = __v7;
map.m_len = __v7;
}
// Converted some or all blocks successfully?
if !converted.is_null() {
unsafe {
*converted = conv_blocks;
}
}
ret
}

#[unsafe(no_mangle)]
extern "C" fn ext4_convert_unwritten_io_end_vec(mut handle: *mut jbd2_journal_handle, io_end: *mut ext4_io_end) -> i32 {
let mut ret: i32 = 0;
let mut err: i32 = 0;
let mut io_end_vec: *mut ext4_io_end_vec = unsafe { core::mem::zeroed() };
// This is somewhat ugly but the idea is clear: When transaction is
// reserved, everything goes into it. Otherwise we rather start several
// smaller transactions for conversion of each extent separately.
if !handle.is_null() {
handle = ext4_journal_start_reserved!(handle, 11i32);
if IS_ERR(handle as *const ()) {
return PTR_ERR(handle as *const ()) as i32;
}
}
io_end_vec = list_first_entry__6!(unsafe { core::ptr::addr_of_mut!((*io_end).list_vec) });
while list_entry_is_head!(io_end_vec, unsafe { core::ptr::addr_of_mut!((*io_end).list_vec) }, list) == 0 {
ret = ext4_convert_unwritten_extents(handle, unsafe { (*io_end).inode }, unsafe { (*io_end_vec).offset }, unsafe { (*io_end_vec).size }, core::ptr::null_mut::<u32>());
if ret != 0 {
break;
}
io_end_vec = list_next_entry__5!(io_end_vec, list);
}
if !handle.is_null() {
err = ext4_journal_stop__17!(handle);
}
if ret < 0 { ret } else { err }
}

fn ext4_iomap_xattr_fiemap(inode: *mut inode, iomap: *mut iomap) -> i32 {
// offset of xattr in inode
let mut offset: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(offset) as *mut i32;
let mut iloc: core::mem::MaybeUninit<ext4_iloc> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut ext4_iloc = core::ptr::addr_of_mut!(iloc) as *mut ext4_iloc;
let mut iomap_type: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u16 = core::ptr::addr_of_mut!(iomap_type) as *mut u16;
let mut error: i32 = unsafe { core::mem::zeroed() };
let mut blockbits: i32 = unsafe { core::mem::zeroed() };
let mut length: u64 = unsafe { core::mem::zeroed() };
let mut physical: u64 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
physical = 0;
length = 0;
blockbits = (*(*inode).i_sb).s_blocksize_bits as i32;
error = 0;
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 2usize);
// in-inode?
if ext4_test_inode_state(inode, 1i32) != 0 {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
error = ext4_get_inode_loc(inode, core::ptr::addr_of_mut!(*__slate_slot_3));
if error != 0 {
return error;
} else {
physical = (*(*__slate_slot_3).bh).b_blocknr << blockbits;
*__slate_slot_2 = 128i32.wrapping_add((*EXT4_I!(inode)).i_extra_isize as i32);
physical = physical.wrapping_add(*__slate_slot_2 as u64);
length = unsafe { (*EXT4_SB((*inode).i_sb)).s_inode_size }.wrapping_sub(*__slate_slot_2) as u64;
brelse((*__slate_slot_3).bh);
*__slate_slot_4 = 4;
}
} else {
if (*EXT4_I!(inode)).i_file_acl != 0 {
// external block
physical = (*EXT4_I!(inode)).i_file_acl << blockbits;
length = (*(*inode).i_sb).s_blocksize;
*__slate_slot_4 = 2;
} else {
// no in-inode or external block for xattr, so return -ENOENT
error = 2i32.wrapping_neg();
break '__join_0;
}
}
(*iomap).addr = physical;
(*iomap).offset = 0;
(*iomap).length = length;
(*iomap).r#type = *__slate_slot_4;
(*iomap).flags = 0;
}
return error;
}
}

#[unsafe(link_section = ".text.slate_distinct.extents.ext4_iomap_xattr_begin")]
extern "C" fn ext4_iomap_xattr_begin(inode: *mut inode, offset: i64, length: i64, flags: u32, iomap: *mut iomap, srcmap: *mut iomap) -> i32 {
let mut error: i32 = ext4_iomap_xattr_fiemap(inode, iomap);
if error == 0 && (offset as u64) >= unsafe { (*iomap).length } {
error = 2i32.wrapping_neg();
}
error
}

#[unsafe(link_section = ".text.slate_distinct.extents.ext4_iomap_xattr_next")]
extern "C" fn ext4_iomap_xattr_next(iter: *const iomap_iter, iomap: *mut iomap, srcmap: *mut iomap) -> i32 {
iomap_iter_next(iter, iomap, srcmap, Some(ext4_iomap_xattr_begin), None)
}

static ext4_iomap_xattr_ops: __SlateConst<iomap_ops> = __SlateConst({
let mut __t0: iomap_ops = unsafe { core::mem::zeroed() };
__t0.iomap_next = Some(ext4_iomap_xattr_next);
__t0
});

fn ext4_fiemap_check_ranges(inode: *mut inode, start: u64, len: &mut u64) -> i32 {
let maxbytes: u64 = ext4_get_maxbytes(inode) as u64;
if (unsafe { *(len as *mut u64) }) == 0 {
return 22i32.wrapping_neg();
}
if start > maxbytes {
return 27i32.wrapping_neg();
}
// Shrink request scope to what the fs can actually handle.
if (unsafe { *(len as *mut u64) }) > maxbytes || maxbytes.wrapping_sub(unsafe { *(len as *mut u64) }) < start {
unsafe {
*(len as *mut u64) = maxbytes.wrapping_sub(start);
}
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.extents.ext4_fiemap")]
extern "C" fn ext4_fiemap(inode: *mut inode, fieinfo: *mut fiemap_extent_info, start: u64, mut len: u64) -> i32 {
let mut error: i32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
error = 0;
inode_lock_shared(inode);
if (*fieinfo).fi_flags & 4 != 0 {
error = ext4_ext_precache(inode);
if error != 0 {
break '__join_0;
} else {
let __v3: u32 = (*fieinfo).fi_flags & 4294967291u32;
(*fieinfo).fi_flags = __v3;
}
}
// For bitmap files the maximum size limit could be smaller than
// s_maxbytes, so check len here manually instead of just relying on the
// generic check.
error = ext4_fiemap_check_ranges(inode, start, &mut (*core::ptr::addr_of_mut!(len)));
if error == 0 {
if (*fieinfo).fi_flags & 2 != 0 {
let __v0: u32 = (*fieinfo).fi_flags & 4294967293u32;
(*fieinfo).fi_flags = __v0;
error = iomap_fiemap(inode, fieinfo, start, len, core::ptr::addr_of!(ext4_iomap_xattr_ops.0));
} else {
error = iomap_fiemap(inode, fieinfo, start, len, core::ptr::addr_of!(ext4_iomap_report_ops));
}
}
}
inode_unlock_shared(inode);
return error;
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_get_es_cache(inode: *mut inode, fieinfo: *mut fiemap_extent_info, start: u64, mut len: u64) -> i32 {
let mut start_blk: u32 = 0;
let mut len_blks: u32 = 0;
let mut last_blk: u64 = 0;
let mut error: i32 = 0;
if ext4_has_inline_data(inode) != 0 {
let mut has_inline: i32 = 0;
unsafe { down_read(core::ptr::addr_of_mut!((*EXT4_I!(inode)).__slate_anon_6.xattr_sem)) };
has_inline = ext4_has_inline_data(inode);
unsafe { up_read(core::ptr::addr_of_mut!((*EXT4_I!(inode)).__slate_anon_6.xattr_sem)) };
if has_inline != 0 {
return 0;
}
}
if (unsafe { (*fieinfo).fi_flags }) & 4 != 0 {
inode_lock_shared(inode);
error = ext4_ext_precache(inode);
inode_unlock_shared(inode);
if error != 0 {
return error;
}
let __v4: u32 = (unsafe { (*fieinfo).fi_flags }) & 4294967291u32;
unsafe {
(*fieinfo).fi_flags = __v4;
}
}
error = unsafe { fiemap_prep(inode, fieinfo, start, core::ptr::addr_of_mut!(len), 0u32) };
if error != 0 {
return error;
}
error = ext4_fiemap_check_ranges(inode, start, unsafe { &mut (*core::ptr::addr_of_mut!(len)) });
if error != 0 {
return error;
}
start_blk = (start >> ((unsafe { (*(*inode).i_sb).s_blocksize_bits }) as i32)) as u32;
last_blk = start.wrapping_add(len).wrapping_sub(1) >> ((unsafe { (*(*inode).i_sb).s_blocksize_bits }) as i32);
if last_blk >= 4294967295 {
last_blk = 4294967294u64;
}
len_blks = (last_blk as u32).wrapping_sub(start_blk).wrapping_add(1);
// Walk the extent tree gathering extent information
// and pushing extents back to the user.
ext4_fill_es_cache_info(inode, start_blk, len_blks, fieinfo)
}

/// ext4_ext_shift_path_extents:
/// Shift the extents of a path structure lying between path[depth].p_ext
/// and EXT_LAST_EXTENT(path[depth].p_hdr), by @shift blocks. @SHIFT tells
/// if it is right shift or left shift operation.
fn ext4_ext_shift_path_extents(path: *mut ext4_ext_path, shift: u32, inode: *mut inode, handle: *mut jbd2_journal_handle, SHIFT: u32) -> i32 {
// Try to merge to the left.
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut restart_credits: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(restart_credits) as *mut i32;
let mut credits: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(credits) as *mut i32;
let mut update: bool = unsafe { core::mem::zeroed() };
let mut ex_last: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex_last) as *mut *mut ext4_extent;
let mut ex_start: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex_start) as *mut *mut ext4_extent;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
err = 0;
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
update = false;
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_17 = (*path).p_depth as i32;
'__join_0: {
loop {
if *__slate_slot_17 >= 0 {
if *__slate_slot_17 == ((*path).p_depth as i32) {
*__slate_slot_15 = (*path.wrapping_offset(*__slate_slot_17 as isize)).p_ext;
if (*__slate_slot_15).is_null() {
return 117i32.wrapping_neg();
} else {
*__slate_slot_14 = EXT_LAST_EXTENT!(unsafe { (*path.wrapping_offset(*__slate_slot_17 as isize)).p_hdr });
// leaf + sb + inode
*__slate_slot_12 = 3;
if *__slate_slot_15 == EXT_FIRST_EXTENT!(unsafe { (*path.wrapping_offset(*__slate_slot_17 as isize)).p_hdr }) {
update = true;
// extent tree + sb + inode
*__slate_slot_12 = (*__slate_slot_17).wrapping_add(2);
}
*__slate_slot_11 = ext4_chunk_trans_extent(inode, 0i32);
err = ext4_datasem_ensure_credits(handle, inode, *__slate_slot_12, *__slate_slot_11, 0i32);
if err != 0 {
break;
} else {
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_17 as isize));
if err != 0 {
break '__join_0;
} else {
while *__slate_slot_15 <= *__slate_slot_14 {
if (SHIFT as u32) == 0 {
le32_add_cpu(core::ptr::addr_of_mut!((*(*__slate_slot_15)).ee_block), shift.wrapping_neg());
if *__slate_slot_15 > EXT_FIRST_EXTENT!(unsafe { (*path.wrapping_offset(*__slate_slot_17 as isize)).p_hdr }) {
__v8 = ext4_ext_try_to_merge_right(inode, path, (*__slate_slot_15).wrapping_offset(-(1 as isize))) != 0;
} else {
__v8 = false;
}
if __v8 {
let __v5: *mut ext4_extent = *__slate_slot_14;
*__slate_slot_14 = __v5.wrapping_offset(-(1 as isize));
} else {
let __v7: *mut ext4_extent = *__slate_slot_15;
*__slate_slot_15 = __v7.wrapping_offset(1 as isize);
}
} else {
le32_add_cpu(core::ptr::addr_of_mut!((*(*__slate_slot_14)).ee_block), shift);
ext4_ext_try_to_merge_right(inode, path, *__slate_slot_14);
let __v10: *mut ext4_extent = *__slate_slot_14;
*__slate_slot_14 = __v10.wrapping_offset(-(1 as isize));
}
}
err = ext4_ext_dirty__20!(handle, inode, path.wrapping_offset(*__slate_slot_17 as isize));
if err != 0 {
break '__join_0;
} else {
let __v3: i32 = *__slate_slot_17;
let __v2: i32 = __v3.wrapping_sub(1);
*__slate_slot_17 = __v2;
if __v2 < 0 || !update {
break '__join_0;
}
}
}
}
}
}
// Update index too
err = ext4_ext_get_access(handle, inode, path.wrapping_offset(*__slate_slot_17 as isize));
if err != 0 {
break '__join_0;
} else {
if (SHIFT as u32) == 0 {
le32_add_cpu(core::ptr::addr_of_mut!((*(*path.wrapping_offset(*__slate_slot_17 as isize)).p_idx).ei_block), shift.wrapping_neg());
} else {
le32_add_cpu(core::ptr::addr_of_mut!((*(*path.wrapping_offset(*__slate_slot_17 as isize)).p_idx).ei_block), shift);
}
err = ext4_ext_dirty__4!(handle, inode, path.wrapping_offset(*__slate_slot_17 as isize));
if err != 0 {
break '__join_0;
} else {
// we are done if current index is not a starting index
if (*path.wrapping_offset(*__slate_slot_17 as isize)).p_idx != EXT_FIRST_INDEX!(unsafe { (*path.wrapping_offset(*__slate_slot_17 as isize)).p_hdr }) {
break '__join_0;
} else {
let __v1: i32 = *__slate_slot_17;
*__slate_slot_17 = __v1.wrapping_sub(1);
}
}
}
} else {
break '__join_0;
}
}
if err > 0 {
err = 11i32.wrapping_neg();
}
}
return err;
}
}

/// ext4_ext_shift_extents:
/// All the extents which lies in the range from @start to the last allocated
/// block for the @inode are shifted either towards left or right (depending
/// upon @SHIFT) by @shift blocks.
/// On success, 0 is returned, error otherwise.
fn ext4_ext_shift_extents(inode: *mut inode, handle: *mut jbd2_journal_handle, mut start: u32, shift: u32, SHIFT: u32) -> i32 {
let mut tmp: u32 = unsafe { core::mem::zeroed() };
let mut ex_end: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(ex_end) as *mut u32;
let mut ex_start: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(ex_start) as *mut u32;
let mut iterator: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut u32 = core::ptr::addr_of_mut!(iterator) as *mut *mut u32;
let mut stop: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(stop) as *mut u32;
let mut extent: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut ext4_extent = core::ptr::addr_of_mut!(extent) as *mut *mut ext4_extent;
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut path: core::mem::MaybeUninit<*mut ext4_ext_path> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut ext4_ext_path = core::ptr::addr_of_mut!(path) as *mut *mut ext4_ext_path;
unsafe {
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
tmp = 4294967295;
// Let path point to the last extent
*__slate_slot_15 = ext4_find_extent(inode, 4294967294u32, core::ptr::null_mut::<ext4_ext_path>(), 1073741824i32);
if IS_ERR(*__slate_slot_15 as *const ()) {
return PTR_ERR(*__slate_slot_15 as *const ()) as i32;
} else {
'__join_0: {
*__slate_slot_13 = (*(*__slate_slot_15)).p_depth as i32;
*__slate_slot_12 = (*(*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize)).p_ext;
if !(*__slate_slot_12).is_null() {
*__slate_slot_11 = (*(*__slate_slot_12)).ee_block;
// For left shifts, make sure the hole on the left is big enough to
// accommodate the shift.  For right shifts, make sure the last extent
// won't be shifted beyond EXT_MAX_BLOCKS.
if (SHIFT as u32) == 0 {
*__slate_slot_15 = ext4_find_extent(inode, start.wrapping_sub(1), *__slate_slot_15, 1073741824i32);
if IS_ERR(*__slate_slot_15 as *const ()) {
return PTR_ERR(*__slate_slot_15 as *const ()) as i32;
} else {
*__slate_slot_13 = (*(*__slate_slot_15)).p_depth as i32;
*__slate_slot_12 = (*(*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize)).p_ext;
if !(*__slate_slot_12).is_null() {
*__slate_slot_9 = (*(*__slate_slot_12)).ee_block;
*__slate_slot_8 = unsafe { (*(*__slate_slot_12)).ee_block }.wrapping_add(ext4_ext_get_actual_len(*__slate_slot_12) as u32);
} else {
*__slate_slot_9 = 0;
*__slate_slot_8 = 0;
}
if start == *__slate_slot_9 && shift > *__slate_slot_9 || shift > start.wrapping_sub(*__slate_slot_8) {
ret = 22i32.wrapping_neg();
break '__join_0;
}
}
} else {
if shift > 4294967295u32.wrapping_sub((*__slate_slot_11).wrapping_add(ext4_ext_get_actual_len(*__slate_slot_12) as u32)) {
ret = 22i32.wrapping_neg();
break '__join_0;
}
}
// In case of left shift, iterator points to start and it is increased
// till we reach stop. In case of right shift, iterator points to stop
// and it is decreased till we reach start.
'__loop_29: loop {
ret = 0;
if (SHIFT as u32) == 0 {
*__slate_slot_10 = core::ptr::addr_of_mut!(start);
} else {
*__slate_slot_10 = core::ptr::addr_of_mut!(*__slate_slot_11);
}
if tmp != 4294967295 {
*(*__slate_slot_10) = tmp;
}
// Its safe to start updating extents.  Start and stop are unsigned, so
// in case of right shift if extent with 0 block is reached, iterator
// becomes NULL to indicate the end of the loop.
loop {
if !(*__slate_slot_10).is_null() && start <= *__slate_slot_11 {
*__slate_slot_15 = ext4_find_extent(inode, *(*__slate_slot_10), *__slate_slot_15, 1073741824i32);
if IS_ERR(*__slate_slot_15 as *const ()) {
return PTR_ERR(*__slate_slot_15 as *const ()) as i32;
} else {
*__slate_slot_13 = (*(*__slate_slot_15)).p_depth as i32;
*__slate_slot_12 = (*(*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize)).p_ext;
if (*__slate_slot_12).is_null() {
break '__loop_29;
} else {
if (SHIFT as u32) == 0 && *(*__slate_slot_10) > (*(*__slate_slot_12)).ee_block {
// Hole, move to the next extent
if *__slate_slot_12 < EXT_LAST_EXTENT!(unsafe { (*(*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize)).p_hdr }) {
let __v6: *mut ext4_ext_path = (*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize);
let __v5: *mut ext4_extent = (*__v6).p_ext;
(*__v6).p_ext = __v5.wrapping_offset(1 as isize);
} else {
*(*__slate_slot_10) = ext4_ext_next_allocated_block(*__slate_slot_15);
continue;
}
}
tmp = *(*__slate_slot_10);
if (SHIFT as u32) == 0 {
*__slate_slot_12 = EXT_LAST_EXTENT!(unsafe { (*(*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize)).p_hdr });
*(*__slate_slot_10) = unsafe { (*(*__slate_slot_12)).ee_block }.wrapping_add(ext4_ext_get_actual_len(*__slate_slot_12) as u32);
} else {
*__slate_slot_12 = EXT_FIRST_EXTENT!(unsafe { (*(*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize)).p_hdr });
if (*(*__slate_slot_12)).ee_block > start {
*(*__slate_slot_10) = unsafe { (*(*__slate_slot_12)).ee_block }.wrapping_sub(1);
} else {
if (*(*__slate_slot_12)).ee_block == start {
*__slate_slot_10 = core::ptr::null_mut::<u32>();
} else {
*__slate_slot_12 = EXT_LAST_EXTENT!(unsafe { (*(*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize)).p_hdr });
while (*(*__slate_slot_12)).ee_block >= start {
let __v3: *mut ext4_extent = *__slate_slot_12;
*__slate_slot_12 = __v3.wrapping_offset(-(1 as isize));
}
if *__slate_slot_12 == EXT_LAST_EXTENT!(unsafe { (*(*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize)).p_hdr }) {
break '__join_0;
} else {
let __v1: *mut ext4_extent = *__slate_slot_12;
*__slate_slot_12 = __v1.wrapping_offset(1 as isize);
*__slate_slot_10 = core::ptr::null_mut::<u32>();
}
}
}
(*(*__slate_slot_15).wrapping_offset(*__slate_slot_13 as isize)).p_ext = *__slate_slot_12;
}
ret = ext4_ext_shift_path_extents(*__slate_slot_15, shift, inode, handle, SHIFT);
// iterator can be NULL which means we should break
if ret == 11i32.wrapping_neg() {
continue '__loop_29;
} else {
if ret != 0 {
break '__join_0;
}
}
}
}
} else {
break '__join_0;
}
}
}
__ext4_error_inode(inode, c"ext4_ext_shift_extents".as_ptr() as *const u8, 5580u32, 0u64, 0i32, c"unexpected hole at %lu".as_ptr() as *const u8, *(*__slate_slot_10) as u64);
ret = 117i32.wrapping_neg();
}
}
ext4_free_ext_path(*__slate_slot_15);
return ret;
}
}
}

/// ext4_collapse_range:
/// This implements the fallocate's collapse range functionality for ext4
/// Returns: 0 and non-zero on error.
fn ext4_collapse_range(file: *mut file, offset: i64, len: i64) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut new_size: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i64 = core::ptr::addr_of_mut!(new_size) as *mut i64;
let mut start: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i64 = core::ptr::addr_of_mut!(start) as *mut i64;
let mut credits: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(credits) as *mut u32;
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
let mut end_lblk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(end_lblk) as *mut u32;
let mut start_lblk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(start_lblk) as *mut u32;
let mut end: i64 = unsafe { core::mem::zeroed() };
let mut mapping: *mut address_space = unsafe { core::mem::zeroed() };
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
unsafe {
inode = file_inode(file as *const file);
sb = (*inode).i_sb;
mapping = (*inode).i_mapping;
end = offset.wrapping_add(len);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
trace_ext4_collapse_range(inode, offset, len);
let __ret_warn_on: i32 = (inode_is_locked(inode) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("1565: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1565b - ., 3; .popsection\n# {0}", const 1565 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_collapse_range_19, sym __slate_asm_string_ext4_collapse_range_20, const 5650 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1566: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1566b - ., 4; .popsection\n# {0}", const 1566 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// Currently just for extent based files
if ext4_test_inode_flag(inode, 19i32) == 0 {
return 95i32.wrapping_neg();
} else {
// Collapse range works only on fs cluster size aligned regions.
if !IS_ALIGNED!(offset | len, EXT4_CLUSTER_SIZE!(sb)) {
return 22i32.wrapping_neg();
} else {
// There is no need to overlap collapse range with EOF, in which case
// it is effectively a truncate operation
if end >= (*inode).i_size {
return 22i32.wrapping_neg();
} else {
// Write tail of the last page before removed range and data that
// will be shifted since they will get removed from the page cache
// below. We are also protected from pages becoming dirty by
// i_rwsem and invalidate_lock.
// Need to round down offset to be aligned with page size boundary
// for page size > block size.
*__slate_slot_5 = round_down!(offset, _AC!() << 12i32);
*__slate_slot_3 = filemap_write_and_wait_range(mapping, *__slate_slot_5, offset);
if *__slate_slot_3 == 0 {
*__slate_slot_3 = filemap_write_and_wait_range(mapping, end, 9223372036854775807i64);
}
if *__slate_slot_3 != 0 {
return *__slate_slot_3;
} else {
truncate_pagecache(inode, *__slate_slot_5);
*__slate_slot_6 = ext4_chunk_trans_extent(inode, 0i32) as u32;
*__slate_slot_7 = ext4_journal_start__13!(inode, 5i32, *__slate_slot_6);
if IS_ERR(*__slate_slot_7 as *const ()) {
return PTR_ERR(*__slate_slot_7 as *const ()) as i32;
} else {
ext4_fc_mark_ineligible(sb, 7i32, *__slate_slot_7);
*__slate_slot_9 = (offset >> ((*inode).i_blkbits as i32)) as u32;
*__slate_slot_8 = (offset.wrapping_add(len) >> ((*inode).i_blkbits as i32)) as u32;
ext4_check_map_extents_env(inode);
down_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
ext4_discard_preallocations(inode);
ext4_es_remove_extent(inode, *__slate_slot_9, 4294967295u32.wrapping_sub(*__slate_slot_9));
*__slate_slot_3 = ext4_ext_remove_space(inode, *__slate_slot_9, (*__slate_slot_8).wrapping_sub(1));
if *__slate_slot_3 != 0 {
up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
} else {
ext4_discard_preallocations(inode);
*__slate_slot_3 = ext4_ext_shift_extents(inode, *__slate_slot_7, *__slate_slot_8, (*__slate_slot_8).wrapping_sub(*__slate_slot_9), 0u32);
if *__slate_slot_3 != 0 {
up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
} else {
*__slate_slot_4 = unsafe { (*inode).i_size }.wrapping_sub(len);
i_size_write(inode, *__slate_slot_4);
(*EXT4_I!(inode)).i_disksize = *__slate_slot_4;
up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
*__slate_slot_3 = ext4_mark_inode_dirty__3!(*__slate_slot_7, inode);
if *__slate_slot_3 == 0 {
ext4_update_inode_fsync_trans(*__slate_slot_7, inode, 1i32);
if (*file).f_flags & 1052672u32 != 0 || IS_SYNC!(inode) {
ext4_handle_sync(*__slate_slot_7);
}
}
}
}
ext4_journal_stop__14!(*__slate_slot_7);
return *__slate_slot_3;
}
}
}
}
}
}
}

// ext4_insert_range:
// This function implements the FALLOC_FL_INSERT_RANGE flag of fallocate.
// The data blocks starting from @offset to the EOF are shifted by @len
// towards right to create a hole in the @inode. Inode size is increased
// by len bytes.
// Returns 0 on success, error otherwise.
/// ext4_swap_extents() - Swap extents between two inodes
/// @handle: handle for this transaction
/// @inode1:	First inode
/// @inode2:	Second inode
/// @lblk1:	Start block for first inode
/// @lblk2:	Start block for second inode
/// @count:	Number of blocks to swap
/// @unwritten: Mark second inode's extents as unwritten after swap
/// @erp:	Pointer to save error value
///
/// This helper routine does exactly what is promise "swap extents". All other
/// stuff such as page-cache locking consistency, bh mapping consistency or
/// extent's data copying must be performed by caller.
/// Locking:
/// 		i_rwsem is held for both inodes
/// 		i_data_sem is locked for write for both inodes
/// Assumptions:
/// 		All pages from requested range are locked for both inodes
#[unsafe(no_mangle)]
extern "C" fn ext4_swap_extents(handle: *mut jbd2_journal_handle, inode1: *mut inode, inode2: *mut inode, mut lblk1: u32, mut lblk2: u32, mut count: u32, unwritten: i32, erp: *mut i32) -> i32 {
let mut next2: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u32 = core::ptr::addr_of_mut!(next2) as *mut u32;
let mut next1: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(next1) as *mut u32;
// Hole handling
let mut __v16: bool = unsafe { core::mem::zeroed() };
let mut split: i32 = unsafe { core::mem::zeroed() };
let mut len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(len) as *mut i32;
let mut e2_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(e2_len) as *mut i32;
let mut e1_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(e1_len) as *mut i32;
let mut e2_blk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u32 = core::ptr::addr_of_mut!(e2_blk) as *mut u32;
let mut e1_blk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(e1_blk) as *mut u32;
let mut tmp_ex: core::mem::MaybeUninit<ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut ext4_extent = core::ptr::addr_of_mut!(tmp_ex) as *mut ext4_extent;
let mut ex2: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex2) as *mut *mut ext4_extent;
let mut ex1: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex1) as *mut *mut ext4_extent;
let mut replaced_count: i32 = unsafe { core::mem::zeroed() };
let mut path2: *mut ext4_ext_path = unsafe { core::mem::zeroed() };
let mut path1: *mut ext4_ext_path = unsafe { core::mem::zeroed() };
unsafe {
path1 = core::ptr::null_mut::<ext4_ext_path>();
path2 = core::ptr::null_mut::<ext4_ext_path>();
replaced_count = 0;
'__join_67: {
loop {
if unlikely!(rwsem_is_locked(core::ptr::addr_of_mut!((*EXT4_I!(inode1)).i_data_sem)) == 0) != 0 {
break '__join_67;
} else {
if true {
break;
}
}
}
'__join_63: {
loop {
if unlikely!(rwsem_is_locked(core::ptr::addr_of_mut!((*EXT4_I!(inode2)).i_data_sem)) == 0) != 0 {
break '__join_63;
} else {
if true {
break;
}
}
}
'__join_59: {
loop {
if unlikely!(inode_is_locked(inode1) == 0) != 0 {
break '__join_59;
} else {
if true {
break;
}
}
}
'__join_55: {
loop {
if unlikely!(inode_is_locked(inode2) == 0) != 0 {
break '__join_55;
} else {
if true {
break;
}
}
}
ext4_es_remove_extent(inode1, lblk1, count);
ext4_es_remove_extent(inode2, lblk2, count);
'__join_0: {
'__join_50: {
'__join_48: {
'__join_30: {
'__join_27: {
'__join_19: {
'__join_16: {
'__loop_1: loop {
if count != 0 {
let __slate_auto_bytes_1 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 12usize);
let __slate_auto_bytes_4 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
split = 0;
path1 = ext4_find_extent(inode1, lblk1, path1, 1073741824i32);
if IS_ERR(path1 as *const ()) {
break '__join_50;
} else {
path2 = ext4_find_extent(inode2, lblk2, path2, 1073741824i32);
if IS_ERR(path2 as *const ()) {
break '__join_48;
} else {
*__slate_slot_25 = (*path1.wrapping_offset(((*path1).p_depth as i32) as isize)).p_ext;
*__slate_slot_24 = (*path2.wrapping_offset(((*path2).p_depth as i32) as isize)).p_ext;
// Do we have something to swap ?
if unlikely!((*__slate_slot_24).is_null() || (*__slate_slot_25).is_null()) != 0 {
break '__join_0;
} else {
*__slate_slot_22 = (*(*__slate_slot_25)).ee_block;
*__slate_slot_21 = (*(*__slate_slot_24)).ee_block;
*__slate_slot_20 = ext4_ext_get_actual_len(*__slate_slot_25);
*__slate_slot_19 = ext4_ext_get_actual_len(*__slate_slot_24);
if in_range__2!(lblk1, *__slate_slot_22, *__slate_slot_20) == 0 {
__v16 = true;
} else {
__v16 = in_range__2!(lblk2, *__slate_slot_21, *__slate_slot_19) == 0;
}
if __v16 {
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// if hole after extent, then go to next extent
*__slate_slot_15 = ext4_ext_next_allocated_block(path1);
*__slate_slot_14 = ext4_ext_next_allocated_block(path2);
// If hole before extent, then shift to that extent
if *__slate_slot_22 > lblk1 {
*__slate_slot_15 = *__slate_slot_22;
}
if *__slate_slot_21 > lblk2 {
*__slate_slot_14 = *__slate_slot_21;
}
// Do we have something to swap
if *__slate_slot_15 == 4294967295 || *__slate_slot_14 == 4294967295 {
break '__join_0;
} else {
// Move to the rightest boundary
*__slate_slot_18 = (*__slate_slot_15).wrapping_sub(lblk1) as i32;
if (*__slate_slot_18 as u32) < (*__slate_slot_14).wrapping_sub(lblk2) {
*__slate_slot_18 = (*__slate_slot_14).wrapping_sub(lblk2) as i32;
}
if (*__slate_slot_18 as u32) > count {
*__slate_slot_18 = count as i32;
}
lblk1 = lblk1.wrapping_add(*__slate_slot_18 as u32);
lblk2 = lblk2.wrapping_add(*__slate_slot_18 as u32);
count = count.wrapping_sub(*__slate_slot_18 as u32);
}
} else {
// Prepare left boundary
if *__slate_slot_22 < lblk1 {
split = 1;
path1 = ext4_force_split_extent_at(handle, inode1, path1, lblk1, 0i32);
if IS_ERR(path1 as *const ()) {
break '__join_30;
}
}
if *__slate_slot_21 < lblk2 {
split = 1;
path2 = ext4_force_split_extent_at(handle, inode2, path2, lblk2, 0i32);
if IS_ERR(path2 as *const ()) {
break '__join_27;
}
}
// ext4_split_extent_at() may result in leaf extent split,
// path must to be revalidated.
if split == 0 {
// Prepare right boundary
*__slate_slot_18 = count as i32;
if (*__slate_slot_18 as u32) > (*__slate_slot_22).wrapping_add(*__slate_slot_20 as u32).wrapping_sub(lblk1) {
*__slate_slot_18 = (*__slate_slot_22).wrapping_add(*__slate_slot_20 as u32).wrapping_sub(lblk1) as i32;
}
if (*__slate_slot_18 as u32) > (*__slate_slot_21).wrapping_add(*__slate_slot_19 as u32).wrapping_sub(lblk2) {
*__slate_slot_18 = (*__slate_slot_21).wrapping_add(*__slate_slot_19 as u32).wrapping_sub(lblk2) as i32;
}
if *__slate_slot_18 != *__slate_slot_20 {
split = 1;
path1 = ext4_force_split_extent_at(handle, inode1, path1, lblk1.wrapping_add(*__slate_slot_18 as u32), 0i32);
if IS_ERR(path1 as *const ()) {
break '__join_19;
}
}
if *__slate_slot_18 != *__slate_slot_19 {
split = 1;
path2 = ext4_force_split_extent_at(handle, inode2, path2, lblk2.wrapping_add(*__slate_slot_18 as u32), 0i32);
if IS_ERR(path2 as *const ()) {
break '__join_16;
}
}
// ext4_split_extent_at() may result in leaf extent split,
// path must to be revalidated.
if split == 0 {
loop {
if unlikely!(*__slate_slot_19 != *__slate_slot_20) != 0 {
break '__loop_1;
} else {
if true {
break;
}
}
}
*erp = ext4_ext_get_access(handle, inode1, path1.wrapping_offset(((*path1).p_depth as i32) as isize));
if unlikely__2!(unsafe { *erp }) != 0 {
break '__join_0;
} else {
*erp = ext4_ext_get_access(handle, inode2, path2.wrapping_offset(((*path2).p_depth as i32) as isize));
if unlikely__2!(unsafe { *erp }) != 0 {
break '__join_0;
} else {
// Both extents are fully inside boundaries. Swap it now
*__slate_slot_23 = *(*__slate_slot_25);
ext4_ext_store_pblock(*__slate_slot_25, ext4_ext_pblock(*__slate_slot_24));
ext4_ext_store_pblock(*__slate_slot_24, ext4_ext_pblock(core::ptr::addr_of_mut!(*__slate_slot_23)));
(*(*__slate_slot_25)).ee_len = *__slate_slot_19 as u16;
(*(*__slate_slot_24)).ee_len = *__slate_slot_20 as u16;
if unwritten != 0 {
ext4_ext_mark_unwritten(*__slate_slot_24);
}
if ext4_ext_is_unwritten(core::ptr::addr_of_mut!(*__slate_slot_23)) != 0 {
ext4_ext_mark_unwritten(*__slate_slot_25);
}
ext4_ext_try_to_merge(handle, inode2, path2, *__slate_slot_24);
ext4_ext_try_to_merge(handle, inode1, path1, *__slate_slot_25);
*erp = ext4_ext_dirty__9!(handle, inode2, path2.wrapping_offset(((*path2).p_depth as i32) as isize));
if unlikely__2!(unsafe { *erp }) != 0 {
break '__join_0;
} else {
*erp = ext4_ext_dirty!(handle, inode1, path1.wrapping_offset(((*path1).p_depth as i32) as isize));
// Looks scarry ah..? second inode already points to new blocks,
// and it was successfully dirtied. But luckily error may happen
// only due to journal error, so full transaction will be
// aborted anyway.
if unlikely__2!(unsafe { *erp }) != 0 {
break '__join_0;
} else {
lblk1 = lblk1.wrapping_add(*__slate_slot_18 as u32);
lblk2 = lblk2.wrapping_add(*__slate_slot_18 as u32);
replaced_count = replaced_count.wrapping_add(*__slate_slot_18);
count = count.wrapping_sub(*__slate_slot_18 as u32);
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
} else {
break '__join_0;
}
}
core::arch::asm!("1576: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1576b - ., 3; .popsection\n# {0}", const 1576 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_swap_extents_29, sym __slate_asm_string_ext4_swap_extents_30, const 6000 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
*erp = PTR_ERR(path2 as *const ()) as i32;
break '__join_0;
}
*erp = PTR_ERR(path1 as *const ()) as i32;
break '__join_0;
}
*erp = PTR_ERR(path2 as *const ()) as i32;
break '__join_0;
}
*erp = PTR_ERR(path1 as *const ()) as i32;
break '__join_0;
}
*erp = PTR_ERR(path2 as *const ()) as i32;
break '__join_0;
}
*erp = PTR_ERR(path1 as *const ()) as i32;
}
ext4_free_ext_path(path1);
ext4_free_ext_path(path2);
return replaced_count;
}
core::arch::asm!("1575: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1575b - ., 3; .popsection\n# {0}", const 1575 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_swap_extents_31, sym __slate_asm_string_ext4_swap_extents_32, const 5886 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("1574: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1574b - ., 3; .popsection\n# {0}", const 1574 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_swap_extents_33, sym __slate_asm_string_ext4_swap_extents_34, const 5885 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("1573: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1573b - ., 3; .popsection\n# {0}", const 1573 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_swap_extents_35, sym __slate_asm_string_ext4_swap_extents_36, const 5884 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("1572: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1572b - ., 3; .popsection\n# {0}", const 1572 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_swap_extents_38, sym __slate_asm_string_ext4_swap_extents_39, const 5883 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

/// ext4_clu_mapped - determine whether any block in a logical cluster has
///                   been mapped to a physical cluster
///
/// @inode - file containing the logical cluster
/// @lclu - logical cluster of interest
///
/// Returns 1 if any block in the logical cluster is mapped, signifying
/// that a physical cluster has been allocated for it.  Otherwise,
/// returns 0.  Can also return negative error codes.  Derived from
/// ext4_ext_map_blocks().
#[unsafe(no_mangle)]
extern "C" fn ext4_clu_mapped(inode: *mut inode, lclu: u32) -> i32 {
// if data can be stored inline, the logical cluster isn't
// mapped - no physical clusters have been allocated, and the
// file has no extents
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut last_lclu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(last_lclu) as *mut u32;
let mut first_lclu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(first_lclu) as *mut u32;
let mut first_lblk: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(first_lblk) as *mut u32;
let mut extent: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut ext4_extent = core::ptr::addr_of_mut!(extent) as *mut *mut ext4_extent;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut mapped: i32 = unsafe { core::mem::zeroed() };
let mut depth: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(depth) as *mut i32;
let mut path: core::mem::MaybeUninit<*mut ext4_ext_path> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut ext4_ext_path = core::ptr::addr_of_mut!(path) as *mut *mut ext4_ext_path;
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
unsafe {
sbi = EXT4_SB((*inode).i_sb);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
mapped = 0;
err = 0;
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if ext4_test_inode_state(inode, 6i32) != 0 {
__v0 = true;
} else {
__v0 = ext4_has_inline_data(inode) != 0;
}
if __v0 {
return 0;
} else {
// search for the extent closest to the first block in the cluster
*__slate_slot_8 = ext4_find_extent(inode, EXT4_C2B!(sbi, lclu), core::ptr::null_mut::<ext4_ext_path>(), 0i32);
if IS_ERR(*__slate_slot_8 as *const ()) {
return PTR_ERR(*__slate_slot_8 as *const ()) as i32;
} else {
*__slate_slot_7 = ext_depth(inode) as i32;
// A consistent leaf must not be empty.  This situation is possible,
// though, _during_ tree modification, and it's why an assert can't
// be put in ext4_find_extent().
if unlikely!(unsafe { (*(*__slate_slot_8).wrapping_offset(*__slate_slot_7 as isize)).p_ext }.is_null() && *__slate_slot_7 != 0) != 0 {
__ext4_error_inode(inode, c"ext4_clu_mapped".as_ptr() as *const u8, 6093u32, 0u64, 0i32, c"bad extent address - lblock: %lu, depth: %d, pblock: %lld".as_ptr() as *const u8, EXT4_C2B!(sbi, lclu) as u64, *__slate_slot_7, (*(*__slate_slot_8).wrapping_offset(*__slate_slot_7 as isize)).p_block);
err = 117i32.wrapping_neg();
} else {
*__slate_slot_4 = (*(*__slate_slot_8).wrapping_offset(*__slate_slot_7 as isize)).p_ext;
// can't be mapped if the extent tree is empty
if !(*__slate_slot_4).is_null() {
*__slate_slot_3 = (*(*__slate_slot_4)).ee_block;
*__slate_slot_2 = EXT4_B2C!(sbi, *__slate_slot_3);
// Three possible outcomes at this point - found extent spanning
// the target cluster, to the left of the target cluster, or to the
// right of the target cluster.  The first two cases are handled here.
// The last case indicates the target cluster is not mapped.
if lclu >= *__slate_slot_2 {
*__slate_slot_1 = EXT4_B2C!(sbi, (*__slate_slot_3).wrapping_add(ext4_ext_get_actual_len(*__slate_slot_4) as u32).wrapping_sub(1));
if lclu <= *__slate_slot_1 {
mapped = 1;
} else {
*__slate_slot_3 = ext4_ext_next_allocated_block(*__slate_slot_8);
*__slate_slot_2 = EXT4_B2C!(sbi, *__slate_slot_3);
if lclu == *__slate_slot_2 {
mapped = 1;
}
}
}
}
}
ext4_free_ext_path(*__slate_slot_8);
return if err != 0 { err } else { mapped };
}
}
}
}

/// Updates physical block address and unwritten status of extent
/// starting at lblk start and of len. If such an extent doesn't exist,
/// this function splits the extent tree appropriately to create an
/// extent like this.  This function is called in the fast commit
/// replay path.  Returns 0 on success and error on failure.
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_replay_update_ex(inode: *mut inode, start: u32, len: i32, unwritten: i32, pblk: u64) -> i32 {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut path: core::mem::MaybeUninit<*mut ext4_ext_path> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut ext4_ext_path = core::ptr::addr_of_mut!(path) as *mut *mut ext4_ext_path;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_6 = ext4_find_extent(inode, start, core::ptr::null_mut::<ext4_ext_path>(), 0i32);
if IS_ERR(*__slate_slot_6 as *const ()) {
return PTR_ERR(*__slate_slot_6 as *const ()) as i32;
} else {
'__join_0: {
*__slate_slot_5 = (*(*__slate_slot_6).wrapping_offset(((*(*__slate_slot_6)).p_depth as i32) as isize)).p_ext;
if (*__slate_slot_5).is_null() {
*__slate_slot_4 = 117i32.wrapping_neg();
} else {
if (*(*__slate_slot_5)).ee_block != start {
__v3 = true;
} else {
__v3 = ext4_ext_get_actual_len(*__slate_slot_5) != len;
}
if __v3 {
// We need to split this extent to match our extent first
down_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
*__slate_slot_6 = ext4_force_split_extent_at(core::ptr::null_mut::<jbd2_journal_handle>(), inode, *__slate_slot_6, start, 1i32);
up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
if IS_ERR(*__slate_slot_6 as *const ()) {
*__slate_slot_4 = PTR_ERR(*__slate_slot_6 as *const ()) as i32;
break '__join_0;
} else {
*__slate_slot_6 = ext4_find_extent(inode, start, *__slate_slot_6, 0i32);
if IS_ERR(*__slate_slot_6 as *const ()) {
return PTR_ERR(*__slate_slot_6 as *const ()) as i32;
} else {
*__slate_slot_5 = (*(*__slate_slot_6).wrapping_offset(((*(*__slate_slot_6)).p_depth as i32) as isize)).p_ext;
let __ret_warn_on: i32 = ((*(*__slate_slot_5)).ee_block != start) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("1577: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1577b - ., 3; .popsection\n# {0}", const 1577 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_ext_replay_update_ex_11, sym __slate_asm_string_ext4_ext_replay_update_ex_12, const 6171 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1578: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1578b - ., 4; .popsection\n# {0}", const 1578 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if ext4_ext_get_actual_len(*__slate_slot_5) != len {
down_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
*__slate_slot_6 = ext4_force_split_extent_at(core::ptr::null_mut::<jbd2_journal_handle>(), inode, *__slate_slot_6, start.wrapping_add(len as u32), 1i32);
up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
if IS_ERR(*__slate_slot_6 as *const ()) {
*__slate_slot_4 = PTR_ERR(*__slate_slot_6 as *const ()) as i32;
break '__join_0;
} else {
*__slate_slot_6 = ext4_find_extent(inode, start, *__slate_slot_6, 0i32);
if IS_ERR(*__slate_slot_6 as *const ()) {
return PTR_ERR(*__slate_slot_6 as *const ()) as i32;
} else {
*__slate_slot_5 = (*(*__slate_slot_6).wrapping_offset(((*(*__slate_slot_6)).p_depth as i32) as isize)).p_ext;
}
}
}
}
}
}
if unwritten != 0 {
ext4_ext_mark_unwritten(*__slate_slot_5);
} else {
ext4_ext_mark_initialized(*__slate_slot_5);
}
ext4_ext_store_pblock(*__slate_slot_5, pblk);
down_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
*__slate_slot_4 = ext4_ext_dirty__8!(inode, (*__slate_slot_6).wrapping_offset(((*(*__slate_slot_6)).p_depth as i32) as isize));
up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem));
}
}
ext4_free_ext_path(*__slate_slot_6);
ext4_mark_inode_dirty__4!(inode);
return *__slate_slot_4;
}
}
}

/// Try to shrink the extent tree
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_replay_shrink_inode(inode: *mut inode, end: u32) {
let mut path: *mut ext4_ext_path = core::ptr::null_mut::<ext4_ext_path>();
let mut old_cur: u32 = 0;
let mut cur: u32 = 0;
while cur < end {
path = ext4_find_extent(inode, cur, core::ptr::null_mut::<ext4_ext_path>(), 0i32);
if IS_ERR(path as *const ()) {
return;
}
let ex: *mut ext4_extent = unsafe { (*path.wrapping_offset(((*path).p_depth as i32) as isize)).p_ext };
if ex.is_null() {
ext4_free_ext_path(path);
ext4_mark_inode_dirty__21!(inode);
return;
}
old_cur = cur;
cur = unsafe { (*ex).ee_block }.wrapping_add(ext4_ext_get_actual_len(ex) as u32);
if cur <= old_cur {
cur = old_cur.wrapping_add(1);
}
ext4_ext_try_to_merge(core::ptr::null_mut::<jbd2_journal_handle>(), inode, path, ex);
unsafe { down_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem)) };
ext4_ext_dirty__19!(inode, path.wrapping_offset(((unsafe { (*path).p_depth }) as i32) as isize));
unsafe { up_write(core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_data_sem)) };
ext4_mark_inode_dirty__27!(inode);
ext4_free_ext_path(path);
}
}

/// Check if *cur is a hole and if it is, skip it
fn skip_hole(inode: *mut inode, cur: &mut u32) -> i32 {
let mut ret: i32 = 0;
let mut map: ext4_map_blocks = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(map) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
map.m_lblk = unsafe { *(cur as *mut u32) };
map.m_len = ((unsafe { (*inode).i_size }) >> ((unsafe { (*(*inode).i_sb).s_blocksize_bits }) as i32)).wrapping_sub((unsafe { *(cur as *mut u32) }) as i64) as u32;
ret = unsafe { ext4_map_blocks(core::ptr::null_mut::<jbd2_journal_handle>(), inode, core::ptr::addr_of_mut!(map), 0i32) };
if ret < 0 {
return ret;
}
if ret != 0 {
return 0;
}
unsafe {
*(cur as *mut u32) = unsafe { *(cur as *mut u32) }.wrapping_add(map.m_len);
}
0
}

// Count number of blocks used by this inode and update i_blocks
#[unsafe(no_mangle)]
extern "C" fn ext4_ext_clear_bb(inode: *mut inode) -> i32 {
let mut map: core::mem::MaybeUninit<ext4_map_blocks> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut ext4_map_blocks = core::ptr::addr_of_mut!(map) as *mut ext4_map_blocks;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut j: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(j) as *mut i32;
let mut end: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(end) as *mut u32;
let mut cur: u32 = unsafe { core::mem::zeroed() };
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut path: *mut ext4_ext_path = unsafe { core::mem::zeroed() };
unsafe {
path = core::ptr::null_mut::<ext4_ext_path>();
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
cur = 0;
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
ret = 0;
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
if ext4_test_inode_flag(inode, 28i32) != 0 {
return 0;
} else {
// Determin the size of the file first
path = ext4_find_extent(inode, 4294967294u32, core::ptr::null_mut::<ext4_ext_path>(), 1073741824i32);
if IS_ERR(path as *const ()) {
return PTR_ERR(path as *const ()) as i32;
} else {
*__slate_slot_7 = (*path.wrapping_offset(((*path).p_depth as i32) as isize)).p_ext;
if !(*__slate_slot_7).is_null() {
*__slate_slot_5 = unsafe { (*(*__slate_slot_7)).ee_block }.wrapping_add(ext4_ext_get_actual_len(*__slate_slot_7) as u32);
cur = 0;
while cur < *__slate_slot_5 {
(*__slate_slot_2).m_lblk = cur;
(*__slate_slot_2).m_len = (*__slate_slot_5).wrapping_sub(cur);
ret = ext4_map_blocks(core::ptr::null_mut::<jbd2_journal_handle>(), inode, core::ptr::addr_of_mut!(*__slate_slot_2), 0i32);
if ret < 0 {
break;
} else {
if ret > 0 {
path = ext4_find_extent(inode, (*__slate_slot_2).m_lblk, path, 0i32);
if !IS_ERR(path as *const ()) {
*__slate_slot_4 = 0;
while *__slate_slot_4 < ((*path).p_depth as i32) {
ext4_mb_mark_bb((*inode).i_sb, (*path.wrapping_offset(*__slate_slot_4 as isize)).p_block, 1i32, false);
ext4_fc_record_regions((*inode).i_sb, (*inode).i_ino as i32, 0u32, (*path.wrapping_offset(*__slate_slot_4 as isize)).p_block, 1i32, 1i32);
let __v1: i32 = *__slate_slot_4;
*__slate_slot_4 = __v1.wrapping_add(1);
}
} else {
path = core::ptr::null_mut::<ext4_ext_path>();
}
ext4_mb_mark_bb((*inode).i_sb, (*__slate_slot_2).m_pblk, (*__slate_slot_2).m_len as i32, false);
ext4_fc_record_regions((*inode).i_sb, (*inode).i_ino as i32, (*__slate_slot_2).m_lblk, (*__slate_slot_2).m_pblk, (*__slate_slot_2).m_len as i32, 1i32);
}
cur = cur.wrapping_add((*__slate_slot_2).m_len);
}
}
}
ext4_free_ext_path(path);
return 0;
}
}
}
}

static __slate_asm_string___ext4_ext_dirty_3: [u8; 1] = *b"\0";

static __slate_asm_string___ext4_ext_dirty_4: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_ext4_alloc_file_blocks_33: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_alloc_file_blocks_34: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_alloc_file_blocks_35: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_alloc_file_blocks_36: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_collapse_range_19: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_collapse_range_20: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_do_fallocate_12: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_do_fallocate_13: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_convert_to_initialized_28: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_convert_to_initialized_29: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_convert_to_initialized_30: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_convert_to_initialized_31: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_convert_to_initialized_32: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_convert_to_initialized_33: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_find_hole_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_find_hole_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_find_hole_5: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_find_hole_6: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_insert_extent_32: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_insert_extent_33: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_insert_extent_34: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_insert_extent_35: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_insert_index_2: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_insert_index_3: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_map_blocks_51: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_map_blocks_52: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_mark_unwritten_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_mark_unwritten_1: [u8; 52] = *b"/home/takashi/c-corpus/linux/fs/ext4/ext4_extents.h\0";

static __slate_asm_string_ext4_ext_more_to_rm_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_more_to_rm_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_next_allocated_block_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_next_allocated_block_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_next_leaf_block_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_next_leaf_block_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_remove_space_46: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_remove_space_47: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_replay_update_ex_11: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_replay_update_ex_12: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_trunc_restart_fn_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_trunc_restart_fn_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_try_to_merge_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_try_to_merge_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_try_to_merge_right_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_try_to_merge_right_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_ext_try_to_merge_right_7: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_try_to_merge_right_8: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_insert_range_31: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_insert_range_32: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_split_extent_15: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_split_extent_16: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_split_extent_at_15: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_split_extent_at_16: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_split_extent_zeroout_1: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_split_extent_zeroout_2: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_swap_extents_29: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_swap_extents_30: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_swap_extents_31: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_swap_extents_32: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_swap_extents_33: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_swap_extents_34: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_swap_extents_35: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_swap_extents_36: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_swap_extents_38: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_swap_extents_39: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_ext4_update_i_disksize_2: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_update_i_disksize_3: [u8; 44] = *b"/home/takashi/c-corpus/linux/fs/ext4/ext4.h\0";

static __slate_asm_string_ext4_zero_range_19: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_zero_range_20: [u8; 47] = *b"/home/takashi/c-corpus/linux/fs/ext4/extents.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
