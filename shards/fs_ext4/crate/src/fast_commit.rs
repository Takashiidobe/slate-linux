
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
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
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
//! Defines x86 CPU feature bits
//! This counts to 15. Any more, it will return 16th argument.
//! Prevent speculative execution past this barrier.
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
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
//! Sign Flag
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
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
//! tell ->d_revalidate() to trust no cache
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Locality group:
//!   we try to group all related changes together
//!   so that writeback can flush/allocate them together as well
//!   Size of lg_prealloc_list hash is determined by MB_DEFAULT_GROUP_PREALLOC
//!   (512). We store prealloc space into the hash based on the pa_free blocks
//!   order value.ie, fls(pa_free)-1;
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
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
// An exchange-type operation, which takes a value and a pointer, and
// returns the old value.
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// Device or resource busy
// File exists
// Invalid argument
// Text file busy
// No space left on device
// Read-only file system
// Bad CRC detected
// Value too large for defined data type
// Operation not supported on transport endpoint
// Operation already in progress
// Operation now in progress
// Stale file handle
// Filesystem is corrupted
// Operation Canceled
// Memory page has hardware error
// Operation is not supported
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
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
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
// Generate SD flag bits
// Implementation specific max size
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

macro_rules! EXT4_FIRST_INO {
($s:expr) => {
unsafe { (*EXT4_SB($s)).s_first_ino }
};
}

macro_rules! EXT4_INODE_SIZE {
($s:expr) => {
unsafe { (*EXT4_SB($s)).s_inode_size }
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

macro_rules! IOPRIO_BAD_VALUE {
($val:expr, $max:expr) => {
($val) < 0 || ($val) >= $max
};
}

macro_rules! IOPRIO_PRIO_CLASS {
($ioprio:expr) => {
($ioprio) >> 13i32 & 7i32
};
}

macro_rules! IOPRIO_PRIO_VALUE {
($prioclass:expr, $priolevel:expr) => {
ioprio_value($prioclass, $priolevel, 0i32)
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

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PRIO_TO_NICE {
($prio:expr) => {
$prio.wrapping_sub(100i32.wrapping_add(19i32.wrapping_sub(20i32.wrapping_neg()).wrapping_add(1) / 2))
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

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
};
}

macro_rules! __SLAB_FLAG_BIT {
($nr:expr) => {
1u32 << $nr
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

macro_rules! __assume_ctx_lock {
($var:expr) => {
loop {
$var;
if true {
break;
}
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

macro_rules! __must_be_array__12 {
() => {
core::mem::size_of::<__SlateRecord_fast_commit_c_1119_4>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__16 {
() => {
((const { core::mem::size_of::<*mut jbd2_inode>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut jbd2_inode>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut jbd2_inode>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut jbd2_inode>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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
macro_rules! __round_mask__3 {
($y:expr) => {
$y.wrapping_sub(1) as u32
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
macro_rules! __sign_use__3 {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
};
}

/// Atomic operations are already serializing on x86
macro_rules! __smp_mb__before_atomic {
() => {
while false {
}
};
}

macro_rules! __types_ok__4 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy)
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
macro_rules! container_of__9 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__60!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__16 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__59!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__18 {
($ptr:expr) => {
{
let mut __v: *mut ext4_fc_range;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__23!() as isize)) }) as *mut ()) as *mut ext4_fc_range;
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
macro_rules! container_of__27 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__67!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__30 {
($ptr:expr) => {
{
let mut __v: *mut ext4_fc_dentry_update;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__24!() as isize)) }) as *mut ()) as *mut ext4_fc_dentry_update;
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
let mut __v: *mut ext4_inode_info;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__58!() as isize)) }) as *mut ()) as *mut ext4_inode_info;
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
macro_rules! container_of__36 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__51!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! ext4_handle_dirty_metadata__39 {
($bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_fc_replay_inode".as_ptr() as *const u8, 1981u32, core::ptr::null_mut::<jbd2_journal_handle>(), core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__43 {
($bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_fc_replay_inode".as_ptr() as *const u8, 2012u32, core::ptr::null_mut::<jbd2_journal_handle>(), core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_iget {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_unlink".as_ptr() as *const u8, 1818u32) }
};
}

macro_rules! ext4_iget__2 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_create".as_ptr() as *const u8, 2062u32) }
};
}

macro_rules! ext4_iget__6 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_inode".as_ptr() as *const u8, 1992u32) }
};
}

macro_rules! ext4_iget__7 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_set_bitmaps_and_counters".as_ptr() as *const u8, 2329u32) }
};
}

macro_rules! ext4_iget__8 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_link_internal".as_ptr() as *const u8, 1842u32) }
};
}

macro_rules! ext4_iget__9 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_inode".as_ptr() as *const u8, 1942u32) }
};
}

macro_rules! ext4_iget__10 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_del_range".as_ptr() as *const u8, 2271u32) }
};
}

macro_rules! ext4_iget__12 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_link".as_ptr() as *const u8, 1881u32) }
};
}

macro_rules! ext4_iget__15 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_add_range".as_ptr() as *const u8, 2148u32) }
};
}

macro_rules! ext4_iget__17 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_create".as_ptr() as *const u8, 2049u32) }
};
}

macro_rules! ext4_iget__23 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_fc_replay_unlink".as_ptr() as *const u8, 1810u32) }
};
}

macro_rules! ext4_mark_inode_dirty__61 {
($__i:expr) => {
unsafe { __ext4_mark_inode_dirty(core::ptr::null_mut::<jbd2_journal_handle>(), $__i, c"ext4_fc_replay_create".as_ptr() as *const u8, 2078u32) };
};
}

macro_rules! ext4_mark_inode_dirty__74 {
($__i:expr) => {
unsafe { __ext4_mark_inode_dirty(core::ptr::null_mut::<jbd2_journal_handle>(), $__i, c"ext4_fc_replay_del_range".as_ptr() as *const u8, 2310u32) };
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
macro_rules! flex_array_size {
($count:expr) => {
size_mul($count, (core::mem::size_of::<u8>() as u64).wrapping_add(__must_be_array__12!() as u64))
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
macro_rules! in_range__4 {
($val:expr, $start:expr, $len:expr) => {
{
let __v: i32 = if (core::mem::size_of::<u64>() as u64) | (core::mem::size_of::<i32>() as u64) | (core::mem::size_of::<u64>() as u64) <= (core::mem::size_of::<u32>() as u64) {
in_range32(($val) as u32, ($start) as u32, ($len) as u32) as i32
} else {
in_range64($val, $start, ($len) as u64) as i32
};
__v
}
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
(1i32.wrapping_neg() as u32) < 1
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

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

macro_rules! krealloc_node {
($_o:expr, $_s:expr, $_f:expr, $_n:expr) => {
krealloc_node_align!($_o, $_s, 1i32, $_f, $_n)
};
}

macro_rules! krealloc_node_align {
($p:expr, $new_size:expr, $align:expr, $flags:expr, $nid:expr) => {
alloc_hooks!(unsafe { krealloc_node_align_noprof(($p) as *const (), $new_size, ($align) as u64, $flags, $nid) })
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

macro_rules! likely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__13 {
($ptr:expr, $type:ty) => {
container_of__36!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__14 {
($ptr:expr, $type:ty) => {
container_of__16!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__15 {
($ptr:expr, $type:ty) => {
container_of__27!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__16 {
($ptr:expr, $type:ty) => {
container_of__9!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__8 {
($ptr:expr) => {
container_of__32!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__9 {
($ptr:expr) => {
container_of__30!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__11 {
($ptr:expr) => {
container_of__18!($ptr)
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
($ptr:expr, $type:ty) => {
list_entry__13!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__9 {
($ptr:expr, $type:ty) => {
list_entry__14!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry {
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
macro_rules! list_first_entry__5 {
($ptr:expr) => {
list_entry__9!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__7 {
($ptr:expr) => {
list_entry__11!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__12 {
($ptr:expr, $type:ty) => {
list_entry__16!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__13 {
($ptr:expr, $type:ty) => {
list_entry__15!(unsafe { (*$ptr).next }, $type)
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__4 {
($pos:expr, $member:ident) => {
list_entry__9!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry__8!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__6 {
($pos:expr, $member:ident) => {
list_entry__11!(unsafe { (*$pos).$member.next })
};
}

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
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

macro_rules! offsetof__8 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, i_block) as u64
};
}

macro_rules! offsetof__22 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, i_generation) as u64
};
}

macro_rules! offsetof__23 {
() => {
core::mem::offset_of!(ext4_fc_range, list) as u64
};
}

macro_rules! offsetof__24 {
() => {
core::mem::offset_of!(ext4_fc_dentry_update, fcd_list) as u64
};
}

macro_rules! offsetof__27 {
() => {
4u64
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, vfs_inode) as u64
};
}

macro_rules! offsetof__6 {
() => {
8u64
};
}

macro_rules! offsetof__43 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, fc_crc) as u64
};
}

macro_rules! offsetof__51 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, i_fc_dilist) as u64
};
}

macro_rules! offsetof__58 {
() => {
core::mem::offset_of!(ext4_inode_info, i_fc_list) as u64
};
}

macro_rules! offsetof__59 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, i_fc_list) as u64
};
}

macro_rules! offsetof__60 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, fcd_dilist) as u64
};
}

macro_rules! offsetof__67 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, fcd_list) as u64
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

macro_rules! read_lock {
($lock:expr) => {
unsafe { _raw_read_lock($lock) };
};
}

macro_rules! read_unlock {
($lock:expr) => {
unsafe { _raw_read_unlock($lock) };
};
}

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up__3 {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask__3!($y)).wrapping_add(1)
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

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size {
($count:expr) => {
size_add(24u64, flex_array_size!($count))
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
macro_rules! struct_size_t {
($count:expr) => {
struct_size!($count)
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
    static mut cpuinfo_op: seq_operations;
    static mut cpu_number: i32;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut dquot_operations: dquot_operations;
    static mut kstat: kernel_stat;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_ext4_fc_replay_scan: static_call_key;
    static mut __tracepoint_ext4_fc_replay_scan: tracepoint;
    static mut __SCK__tp_func_ext4_fc_replay: static_call_key;
    static mut __tracepoint_ext4_fc_replay: tracepoint;
    static mut __SCK__tp_func_ext4_fc_commit_start: static_call_key;
    static mut __tracepoint_ext4_fc_commit_start: tracepoint;
    static mut __SCK__tp_func_ext4_fc_commit_stop: static_call_key;
    static mut __tracepoint_ext4_fc_commit_stop: tracepoint;
    static mut __SCK__tp_func_ext4_fc_lock_updates: static_call_key;
    static mut __tracepoint_ext4_fc_lock_updates: tracepoint;
    static mut __SCK__tp_func_ext4_fc_stats: static_call_key;
    static mut __tracepoint_ext4_fc_stats: tracepoint;
    static mut __SCK__tp_func_ext4_fc_track_create: static_call_key;
    static mut __tracepoint_ext4_fc_track_create: tracepoint;
    static mut __SCK__tp_func_ext4_fc_track_link: static_call_key;
    static mut __tracepoint_ext4_fc_track_link: tracepoint;
    static mut __SCK__tp_func_ext4_fc_track_unlink: static_call_key;
    static mut __tracepoint_ext4_fc_track_unlink: tracepoint;
    static mut __SCK__tp_func_ext4_fc_track_inode: static_call_key;
    static mut __tracepoint_ext4_fc_track_inode: tracepoint;
    static mut __SCK__tp_func_ext4_fc_track_range: static_call_key;
    static mut __tracepoint_ext4_fc_track_range: tracepoint;
    static mut __SCK__tp_func_ext4_fc_cleanup: static_call_key;
    static mut __tracepoint_ext4_fc_cleanup: tracepoint;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __cmpxchg_wrong_size();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_10();
    fn ____wrong_branch_error() -> bool;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_read_lock(arg0: *mut rwlock);
    fn _raw_read_unlock(arg0: *mut rwlock);
    fn schedule();
    fn prepare_to_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32);
    fn finish_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn rcu_is_watching() -> bool;
    fn down_write(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn ktime_get() -> i64;
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn take_dentry_name_snapshot(arg0: *mut name_snapshot, arg1: *mut dentry);
    fn release_dentry_name_snapshot(arg0: *mut name_snapshot);
    fn wake_up_bit(arg0: *mut u64, arg1: i32);
    fn bit_waitqueue(arg0: *mut u64, arg1: i32) -> *mut wait_queue_head;
    fn wake_bit_function(arg0: *mut wait_queue_entry, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn set_task_ioprio(arg0: *mut task_struct, arg1: i32) -> i32;
    fn __compiletime_assert_384();
    fn __kmem_cache_create_args(arg0: *const u8, arg1: u32, arg2: *mut kmem_cache_args, arg3: u32) -> *mut kmem_cache;
    fn kmem_cache_destroy(arg0: *mut kmem_cache);
    fn krealloc_node_align_noprof(arg0: *const (), arg1: u64, arg2: u64, arg3: u32, arg4: i32) -> *mut ();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_390();
    fn __builtin_unreachable();
    fn kmem_cache_alloc_noprof(arg0: *mut kmem_cache, arg1: u32) -> *mut ();
    fn kmem_cache_free(arg0: *mut kmem_cache, arg1: *mut ());
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kvmalloc_node_noprof(arg0: u64, arg1: u64, arg2: u32, arg3: i32) -> *mut ();
    fn kvfree(arg0: *const ());
    fn __compiletime_assert_399();
    fn __compiletime_assert_400();
    fn set_nlink(arg0: *mut inode, arg1: u32);
    fn iput(arg0: *mut inode);
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn blk_start_plug(arg0: *mut blk_plug);
    fn blk_finish_plug(arg0: *mut blk_plug);
    fn blkdev_issue_flush(arg0: *mut block_device) -> i32;
    fn bh_end_write(arg0: *mut bio);
    fn __brelse(arg0: *mut buffer_head);
    fn __lock_buffer(arg0: *mut buffer_head);
    fn sync_dirty_buffer(arg0: *mut buffer_head) -> i32;
    fn bh_submit(arg0: *mut buffer_head, arg1: u32, arg2: Option<unsafe extern "C" fn(*mut bio)>);
    fn crc32c(arg0: u32, arg1: *const (), arg2: u64) -> u32;
    fn jbd2_journal_lock_updates(arg0: *mut journal_s);
    fn jbd2_journal_unlock_updates(arg0: *mut journal_s);
    fn jbd2_complete_transaction(arg0: *mut journal_s, arg1: u32) -> i32;
    fn jbd2_fc_begin_commit(arg0: *mut journal_s, arg1: u32) -> i32;
    fn jbd2_fc_end_commit(arg0: *mut journal_s) -> i32;
    fn jbd2_fc_end_commit_fallback(arg0: *mut journal_s) -> i32;
    fn jbd2_fc_get_buf(arg0: *mut journal_s, arg1: *mut *mut buffer_head) -> i32;
    fn jbd2_submit_inode_data(arg0: *mut journal_s, arg1: *mut jbd2_inode) -> i32;
    fn jbd2_wait_inode_data(arg0: *mut journal_s, arg1: *mut jbd2_inode) -> i32;
    fn jbd2_fc_wait_bufs(arg0: *mut journal_s, arg1: i32) -> i32;
    fn jbd2_fc_release_bufs(arg0: *mut journal_s);
    fn ext4_es_lookup_extent(arg0: *mut inode, arg1: u32, arg2: *mut u32, arg3: *mut extent_status, arg4: *mut u64) -> i32;
    fn ext4_mark_inode_used(arg0: *mut super_block, arg1: i32) -> i32;
    fn ext4_mb_mark_bb(arg0: *mut super_block, arg1: u64, arg2: i32, arg3: bool);
    fn ext4_inode_csum_set(arg0: *mut inode, arg1: *mut ext4_inode, arg2: *mut ext4_inode_info);
    fn __ext4_iget(arg0: *mut super_block, arg1: u64, arg2: u32, arg3: *const u8, arg4: u32) -> *mut inode;
    fn ext4_get_inode_loc_noio(arg0: *mut inode, arg1: *mut ext4_iloc) -> i32;
    fn ext4_get_fc_inode_loc(arg0: *mut super_block, arg1: u64, arg2: *mut ext4_iloc) -> i32;
    fn ext4_reset_inode_seed(arg0: *mut inode);
    fn ext4_init_new_dir(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut inode) -> i32;
    fn __ext4_unlink(arg0: *mut inode, arg1: *const qstr, arg2: *mut inode, arg3: *mut dentry) -> i32;
    fn __ext4_link(arg0: *mut inode, arg1: *mut inode, arg2: *const qstr, arg3: *mut dentry) -> i32;
    fn ext4_ext_remove_space(arg0: *mut inode, arg1: u32, arg2: u32) -> i32;
    fn ext4_map_blocks(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut ext4_map_blocks, arg3: i32) -> i32;
    fn ext4_ext_insert_extent(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut ext4_ext_path, arg3: *mut ext4_extent, arg4: i32) -> *mut ext4_ext_path;
    fn ext4_find_extent(arg0: *mut inode, arg1: u32, arg2: *mut ext4_ext_path, arg3: i32) -> *mut ext4_ext_path;
    fn ext4_free_ext_path(arg0: *mut ext4_ext_path);
    fn ext4_ext_replay_shrink_inode(arg0: *mut inode, arg1: u32);
    fn ext4_ext_replay_set_iblocks(arg0: *mut inode) -> i32;
    fn ext4_ext_replay_update_ex(arg0: *mut inode, arg1: u32, arg2: i32, arg3: i32, arg4: u64) -> i32;
    fn ext4_ext_clear_bb(arg0: *mut inode) -> i32;
    fn __ext4_mark_inode_dirty(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *const u8, arg3: u32) -> i32;
    fn __ext4_handle_dirty_metadata(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle, arg3: *mut inode, arg4: *mut buffer_head) -> i32;
    fn ext4_inode_journal_mode(arg0: *mut inode) -> i32;
    fn seq_write(arg0: *mut seq_file, arg1: *const (), arg2: u64) -> i32;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn seq_putc(arg0: *mut seq_file, arg1: u8);
    fn __seq_puts(arg0: *mut seq_file, arg1: *const u8);
    #[link_name = "strlen"]
    fn __builtin_strlen(arg0: *const u8) -> u64;
    fn __traceiter_ext4_fc_replay_scan(arg0: *mut (), arg1: *mut super_block, arg2: i32, arg3: i32) -> i32;
    fn __SCT__tp_func_ext4_fc_replay_scan(arg0: *mut (), arg1: *mut super_block, arg2: i32, arg3: i32) -> i32;
    fn __compiletime_assert_1434();
    fn __traceiter_ext4_fc_replay(arg0: *mut (), arg1: *mut super_block, arg2: i32, arg3: i32, arg4: i32, arg5: i32) -> i32;
    fn __SCT__tp_func_ext4_fc_replay(arg0: *mut (), arg1: *mut super_block, arg2: i32, arg3: i32, arg4: i32, arg5: i32) -> i32;
    fn __compiletime_assert_1441();
    fn __traceiter_ext4_fc_commit_start(arg0: *mut (), arg1: *mut super_block, arg2: u32) -> i32;
    fn __SCT__tp_func_ext4_fc_commit_start(arg0: *mut (), arg1: *mut super_block, arg2: u32) -> i32;
    fn __compiletime_assert_1448();
    fn __traceiter_ext4_fc_commit_stop(arg0: *mut (), arg1: *mut super_block, arg2: i32, arg3: i32, arg4: u32) -> i32;
    fn __SCT__tp_func_ext4_fc_commit_stop(arg0: *mut (), arg1: *mut super_block, arg2: i32, arg3: i32, arg4: u32) -> i32;
    fn __compiletime_assert_1455();
    fn __traceiter_ext4_fc_lock_updates(arg0: *mut (), arg1: *mut super_block, arg2: u32, arg3: u64, arg4: u32, arg5: u32, arg6: i32, arg7: i32) -> i32;
    fn __SCT__tp_func_ext4_fc_lock_updates(arg0: *mut (), arg1: *mut super_block, arg2: u32, arg3: u64, arg4: u32, arg5: u32, arg6: i32, arg7: i32) -> i32;
    fn __compiletime_assert_1462();
    fn __traceiter_ext4_fc_stats(arg0: *mut (), arg1: *mut super_block) -> i32;
    fn __SCT__tp_func_ext4_fc_stats(arg0: *mut (), arg1: *mut super_block) -> i32;
    fn __compiletime_assert_1469();
    fn __traceiter_ext4_fc_track_create(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: *mut dentry, arg4: i32) -> i32;
    fn __SCT__tp_func_ext4_fc_track_create(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: *mut dentry, arg4: i32) -> i32;
    fn __compiletime_assert_1476();
    fn __traceiter_ext4_fc_track_link(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: *mut dentry, arg4: i32) -> i32;
    fn __SCT__tp_func_ext4_fc_track_link(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: *mut dentry, arg4: i32) -> i32;
    fn __compiletime_assert_1483();
    fn __traceiter_ext4_fc_track_unlink(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: *mut dentry, arg4: i32) -> i32;
    fn __SCT__tp_func_ext4_fc_track_unlink(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: *mut dentry, arg4: i32) -> i32;
    fn __compiletime_assert_1490();
    fn __traceiter_ext4_fc_track_inode(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: i32) -> i32;
    fn __SCT__tp_func_ext4_fc_track_inode(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: i32) -> i32;
    fn __compiletime_assert_1497();
    fn __traceiter_ext4_fc_track_range(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: i64, arg4: i64, arg5: i32) -> i32;
    fn __SCT__tp_func_ext4_fc_track_range(arg0: *mut (), arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: i64, arg4: i64, arg5: i32) -> i32;
    fn __compiletime_assert_1504();
    fn __traceiter_ext4_fc_cleanup(arg0: *mut (), arg1: *mut journal_s, arg2: i32, arg3: u32) -> i32;
    fn __SCT__tp_func_ext4_fc_cleanup(arg0: *mut (), arg1: *mut journal_s, arg2: i32, arg3: u32) -> i32;
    fn __compiletime_assert_1511();
    fn __compiletime_assert_1545();
    fn __compiletime_assert_1548();
    fn __compiletime_assert_1549();
    fn __compiletime_assert_1550();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
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
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_add(mut i: i64, mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock addq {1:r}, ({0:r})\n# {2}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u64, in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic64_cmpxchg(v: *mut __SlateRecord_types_h_197, mut __old: i64, mut __new: i64) -> i64 {
let mut __v0: i64;
let mut __ret: i64 = 0;
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u64;
core::arch::asm!("lock cmpxchgb {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t0 as u64) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t1: u64;
core::arch::asm!("lock cmpxchgw {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t1 as u64) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t2: u64;
core::arch::asm!("lock cmpxchgl {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t2 as u64) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t3: u64;
core::arch::asm!("lock cmpxchgq {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t3 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v0 = __ret;
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic64_add(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic64_cmpxchg(v: *mut __SlateRecord_types_h_197, old: i64, new: i64) -> i64 {
arch_atomic64_cmpxchg(v, old, new)
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
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
core::arch::asm!("8: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 8b - ., 3; .popsection\n# {0}", const 8 as i32, options(nostack, att_syntax));
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
core::arch::asm!("9: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 9b - ., 4; .popsection\n# {0}", const 9 as i32, options(nostack, att_syntax));
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic64_read(v)
}

#[inline(always)]
fn atomic64_add(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic64_add(i, v);
}

#[inline(always)]
fn atomic64_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic64_inc(v);
}

#[inline(always)]
fn atomic64_cmpxchg(v: *mut __SlateRecord_types_h_197, old: i64, new: i64) -> i64 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic64_cmpxchg(v, old, new)
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


fn div_u64_rem(dividend: u64, divisor: u32, remainder: &mut u32) -> u64 {
unsafe {
*(remainder as *mut u32) = (dividend % (divisor as u64)) as u32;
}
dividend / (divisor as u64)
}

fn div64_u64(dividend: u64, divisor: u64) -> u64 {
dividend / divisor
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


fn div_u64(dividend: u64, divisor: u32) -> u64 {
let mut remainder: u32 = 0;
div_u64_rem(dividend, divisor, unsafe { &mut (*core::ptr::addr_of_mut!(remainder)) })
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
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
unsafe { __compiletime_assert_10() };
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
struct mutex_waiter {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
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

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
// @n: parameter
//
// constant-capable log of base 2 calculation
// - this can be used to initialise global variables from constant data, hence
// the massive ternary operator construction
//
// selects the appropriately-sized optimised version depending on sizeof(n)
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
struct hrtimer_clock_base {
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


// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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
    __slate_bits_0: __slate_bits::__SlateBits133U0,
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
fn IS_ERR_OR_NULL(ptr: *const ()) -> bool {
unlikely!(ptr.is_null()) != 0 || IS_ERR_VALUE!(ptr as u64) != 0
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
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_replay_scan, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_replay, const 0 as u8, label {
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
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_commit_start, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_commit_stop, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_lock_updates, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_stats, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_track_create, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_track_link, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_track_unlink, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_track_inode, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_track_range, const 0 as u8, label {
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
fn arch_static_branch__const12(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_fc_cleanup, const 0 as u8, label {
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
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
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
    __slate_bits_0: __slate_bits::__SlateBits140U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
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
struct rseq {
}


// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// min_t - return minimum of two values, using the specified type
// @type: data type to use
// @x: first value
// @y: second value
// max_t - return maximum of two values, using the specified type
// @type: data type to use
// @x: first value
// @y: second value
fn in_range64(val: u64, start: u64, len: u64) -> bool {
val.wrapping_sub(start) < len
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


fn in_range32(val: u32, start: u32, len: u32) -> bool {
val.wrapping_sub(start) < len
}

// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
static mut __UNIQUE_ID_addressable___SCK__might_resched_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_14() };
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
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_15() };
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
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_16() };
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
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_19() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


fn list_del_init(entry: *mut list_head) {
__list_del_entry(entry);
INIT_LIST_HEAD(entry);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_20() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
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


fn __list_splice(list: *const list_head, prev: *mut list_head, next: *mut list_head) {
let first: *mut list_head = unsafe { (*list).next };
let last: *mut list_head = unsafe { (*list).prev };
unsafe {
(*first).prev = prev;
}
unsafe {
(*prev).next = first;
}
unsafe {
(*last).next = next;
}
unsafe {
(*next).prev = last;
}
}

fn list_splice_init(list: *mut list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, head, unsafe { (*head).next });
INIT_LIST_HEAD(list);
}
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


fn list_splice_tail_init(list: *mut list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, unsafe { (*head).prev }, head);
INIT_LIST_HEAD(list);
}
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


#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
    __slate_bits_0: __slate_bits::__SlateBits141U0,
    pkru: u32,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
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
struct vm_struct {
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
struct mem_cgroup {
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
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits227U0,
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

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
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
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mm_cid_pcpu {
    cid: u32,
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
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
struct kmem_cache {
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
    __slate_bits_0: __slate_bits::__SlateBits290U0,
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
struct mnt_idmap {
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
// I am kswapd
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
fn task_nice(p: *const task_struct) -> i32 {
PRIO_TO_NICE!(unsafe { (*p).static_prio })
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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
struct completion {
    done: u32,
    wait: swait_queue_head,
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
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
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
struct tty_audit_buf {
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
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
struct fs_context {
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
fn memalloc_flags_save(flags: u32) -> u32 {
let oldflags: u32 = !unsafe { (*get_current()).flags } & flags;
let __v0: *mut task_struct = get_current();
unsafe {
(*__v0).flags = (*__v0).flags | flags;
}
oldflags
}

fn memalloc_flags_restore(flags: u32) {
let __v0: *mut task_struct = get_current();
let __v2: u32 = (unsafe { (*__v0).flags }) & !flags;
unsafe {
(*__v0).flags = __v2;
}
}

fn memalloc_nofs_save() -> u32 {
memalloc_flags_save(262144u32)
}

fn memalloc_nofs_restore(flags: u32) {
memalloc_flags_restore(flags);
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
struct poll_table_struct {
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
struct posix_acl {
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
union key_payload {
    rcu_data0: *mut (),
    data: [*mut (); 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct rhash_head {
    next: *mut rhash_head,
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits293U0,
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
struct wake_irq {
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
struct dev_pm_qos {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
// VFS does not apply umask
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
struct module_param_attrs {
}


fn d_inode(dentry: *const dentry) -> *mut inode {
unsafe { (*dentry).d_inode }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct name_snapshot {
    name: qstr,
    inline_name: shortname_store,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_bit_key {
    flags: *mut u64,
    bit_nr: i32,
    timeout: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_bit_queue_entry {
    key: wait_bit_key,
    wq_entry: wait_queue_entry,
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


fn rt_or_dl_task_policy(tsk: *mut task_struct) -> bool {
let policy: i32 = (unsafe { (*tsk).policy }) as i32;
if policy == 1 || policy == 2 {
return true;
}
if policy == 6 {
return true;
}
false
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
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
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


// The 10 bits between the priority class and the priority level are used to
// optionally define I/O hints for any combination of I/O priority class and
// level. Depending on the kernel configuration, I/O scheduler being used and
// the target I/O device being used, hints can influence how I/Os are processed
// without affecting the I/O scheduling ordering defined by the I/O priority
// class and level.
#[inline(always)]
fn ioprio_value(prioclass: i32, priolevel: i32, priohint: i32) -> u16 {
if IOPRIO_BAD_VALUE!(prioclass, 8i32) || IOPRIO_BAD_VALUE!(priolevel, 8i32) || IOPRIO_BAD_VALUE!(priohint, 1024i32) {
return 57344u16;
}
(prioclass << 13i32 | priohint << 3i32 | priolevel) as u16
}

fn task_nice_ioprio(task: *mut task_struct) -> i32 {
task_nice(task as *const task_struct).wrapping_add(20) / 5
}

fn task_nice_ioclass(task: *mut task_struct) -> i32 {
if (unsafe { (*task).policy }) == 5 {
return 3;
} else {
if rt_or_dl_task_policy(task) {
return 1;
} else {
return 2;
}
}
}

fn __get_task_ioprio(p: *mut task_struct) -> i32 {
let ioc: *mut io_context = unsafe { (*p).io_context };
let mut prio: i32 = 0;
if ioc.is_null() {
return IOPRIO_PRIO_VALUE!(task_nice_ioclass(p), task_nice_ioprio(p)) as i32;
}
if p != get_current() {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*p).alloc_lock) });
}
prio = (unsafe { (*ioc).ioprio }) as i32;
if IOPRIO_PRIO_CLASS!(prio) == 0 {
prio = IOPRIO_PRIO_VALUE!(task_nice_ioclass(p), task_nice_ioprio(p)) as i32;
}
prio
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


fn get_current_ioprio() -> i32 {
__get_task_ioprio(get_current())
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
struct disk_stats {
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
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
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
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
#[repr(C)]
#[derive(Clone, Copy)]
struct request {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits587U0,
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


// Please use this macro to create slab caches. Simply specify the
// name of the structure and maybe some flags that are listed above.
//
// The alignment of the struct determines object alignment. If you
// f.e. add ____cacheline_aligned_in_smp to the struct declaration
// then the objects will be properly aligned in SMP configurations.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_buffer_head_h_62 {
    b_page: *mut page,
    b_folio: *mut folio,
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
// kmalloc_flex - Allocate a single instance of the given flexible structure
// @VAR_OR_TYPE: Variable or type to allocate (with its flex array).
// @FAM: The name of the flexible array member of the structure.
// @COUNT: How many flexible array member elements are desired.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to @VAR_OR_TYPE on success, NULL on
// failure. If @FAM has been annotated with __counted_by(), the allocation
// will immediately fail if @COUNT is larger than what the type of the
// struct's counter variable can represent.
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


// XArray tags, for tagging dirty and writeback pages in the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
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
struct proc_dir_entry {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ext4_h_806 {
    linux1: __SlateRecord_ext4_h_807,
    hurd1: __SlateRecord_ext4_h_810,
    masix1: __SlateRecord_ext4_h_813,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_807 {
    l_i_version: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_810 {
    h_i_translator: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_813 {
    m_i_reserved1: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ext4_h_822 {
    linux2: __SlateRecord_ext4_h_823,
    hurd2: __SlateRecord_ext4_h_831,
    masix2: __SlateRecord_ext4_h_838,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_823 {
    l_i_blocks_high: u16,
    l_i_file_acl_high: u16,
    l_i_uid_high: u16,
    l_i_gid_high: u16,
    l_i_checksum_lo: u16,
    l_i_reserved: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_831 {
    h_i_reserved1: u16,
    h_i_mode_high: u16,
    h_i_uid_high: u16,
    h_i_gid_high: u16,
    h_i_author: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_838 {
    h_i_reserved1: u16,
    m_i_file_acl_high: u16,
    m_i_reserved2: [u32; 2],
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
    __slate_bits_0: __slate_bits::__SlateBits665U0,
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
    __slate_bits_0: __slate_bits::__SlateBits602U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits691U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits691U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits691U2,
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
    __slate_bits_3: __slate_bits::__SlateBits691U3,
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
struct mb_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dax_device {
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
struct cgroup_taskset {
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
struct swap_cluster_info {
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
struct swap_ops {
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
struct __SlateRecord_fast_commit_c_1119_4 {
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


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits550U0 {
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
    pub struct __SlateBits587U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits290U0 {
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
    pub struct __SlateBits424U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits665U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits925U0 {
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
    pub struct __SlateBits293U0 {
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
    pub struct __SlateBits691U0 {
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
    pub struct __SlateBits688U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits691U1 {
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
    pub struct __SlateBits691U2 {
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
    pub struct __SlateBits691U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits133U0 {
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
    pub struct __SlateBits141U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits602U0 {
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
    pub struct __SlateBits140U0 {
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
    pub struct __SlateBits227U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits810U0 {
        #[bits(1)]
        pub t_synchronous_commit: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits814U0 {
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
    __slate_bits_0: __slate_bits::__SlateBits688U0,
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
fn set_buffer_dirty(bh: *mut buffer_head) {
if test_bit__3!(1i32, unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) == 0 {
set_bit(1i64, (unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) as *mut u64);
}
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

fn brelse(bh: *mut buffer_head) {
if !bh.is_null() {
unsafe { __brelse(bh) };
}
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
    __slate_bits_0: __slate_bits::__SlateBits814U0,
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
    __slate_bits_0: __slate_bits::__SlateBits810U0,
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
// Journal high priority write IO operation flags
// Journaling has been aborted for errors.
// Fast commit is ongoing
// Full commit is ongoing
#[repr(C)]
#[derive(Clone, Copy)]
struct jbd2_buffer_trigger_type {
    t_frozen: Option<unsafe extern "C" fn(*mut jbd2_buffer_trigger_type, *mut buffer_head, *mut (), u64)>,
    t_abort: Option<unsafe extern "C" fn(*mut jbd2_buffer_trigger_type, *mut buffer_head)>,
}


fn tid_gt(x: u32, y: u32) -> i32 {
let difference: i32 = x.wrapping_sub(y) as i32;
(difference > 0) as i32
}

fn tid_geq(x: u32, y: u32) -> i32 {
let difference: i32 = x.wrapping_sub(y) as i32;
(difference >= 0) as i32
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
struct fiemap_extent {
    fe_logical: u64,
    fe_physical: u64,
    fe_length: u64,
    fe_reserved64: [u64; 2],
    fe_flags: u32,
    fe_reserved: [u32; 3],
}


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
// Root inode
// Translate a cluster number to a block number
#[repr(C)]
#[derive(Clone, Copy)]
struct flex_groups {
    free_clusters: __SlateRecord_types_h_197,
    free_inodes: __SlateRecord_types_h_190,
    used_dirs: __SlateRecord_types_h_190,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_inode {
    i_mode: u16,
    i_uid: u16,
    i_size_lo: u32,
    i_atime: u32,
    i_ctime: u32,
    i_mtime: u32,
    i_dtime: u32,
    i_gid: u16,
    i_links_count: u16,
    i_blocks_lo: u32,
    i_flags: u32,
    osd1: __SlateRecord_ext4_h_806,
    i_block: [u32; 15],
    i_generation: u32,
    i_file_acl_lo: u32,
    i_size_high: u32,
    i_obso_faddr: u32,
    osd2: __SlateRecord_ext4_h_822,
    i_extra_isize: u16,
    i_checksum_hi: u16,
    i_ctime_extra: u32,
    i_mtime_extra: u32,
    i_atime_extra: u32,
    i_crtime: u32,
    i_crtime_extra: u32,
    i_version_hi: u32,
    i_projid: u32,
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


fn ext4_es_type(es: &extent_status) -> u32 {
((unsafe { (*(es as *const extent_status)).es_pblk }) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(5) & 15u64) as u32
}

fn ext4_es_is_written(es: &extent_status) -> i32 {
(ext4_es_type(es) & 1u32 != 0) as i32
}

fn ext4_es_is_unwritten(es: &extent_status) -> i32 {
(ext4_es_type(es) & 2u32 != 0) as i32
}

fn ext4_es_is_delayed(es: &extent_status) -> i32 {
(ext4_es_type(es) & 4u32 != 0) as i32
}

fn ext4_es_is_hole(es: &extent_status) -> i32 {
(ext4_es_type(es) & 8u32 != 0) as i32
}

fn ext4_es_pblock(es: &extent_status) -> u64 {
(unsafe { (*(es as *const extent_status)).es_pblk }) & !(18446744073709551615u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(5))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_tl {
    fc_tag: u16,
    fc_len: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_head {
    fc_features: u32,
    fc_tid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_add_range {
    fc_ino: u32,
    fc_ex: [u8; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_del_range {
    fc_ino: u32,
    fc_lblk: u32,
    fc_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_dentry_info {
    fc_parent_ino: u32,
    fc_ino: u32,
    fc_dname: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_inode {
    fc_ino: u32,
    fc_raw_inode: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_tail {
    fc_tid: u32,
    fc_crc: u32,
}


// Tag base length
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_dentry_update {
    fcd_op: i32,
    fcd_parent: i32,
    fcd_ino: i32,
    fcd_name: name_snapshot,
    fcd_list: list_head,
    fcd_dilist: list_head,
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


fn tag2str(tag: u16) -> *const u8 {
match tag as i32 {
4 => {
return c"ADD_ENTRY".as_ptr() as *const u8;
}
5 => {
return c"DEL_ENTRY".as_ptr() as *const u8;
}
1 => {
return c"ADD_RANGE".as_ptr() as *const u8;
}
3 => {
return c"CREAT_DENTRY".as_ptr() as *const u8;
}
2 => {
return c"DEL_RANGE".as_ptr() as *const u8;
}
6 => {
return c"INODE".as_ptr() as *const u8;
}
7 => {
return c"PAD".as_ptr() as *const u8;
}
8 => {
return c"TAIL".as_ptr() as *const u8;
}
9 => {
return c"HEAD".as_ptr() as *const u8;
}
_ => {
return c"ERROR".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

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


// Fast commit replay ongoing
// Mode for data writes:
// Write data to journal
// Use block barriers
// Some quota option set
// Enable support for dio read nolocking
// Delalloc support
// Journal fast commit
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

fn ext4_fc_lock(sb: *mut super_block) -> i32 {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_fc_lock)) };
memalloc_nofs_save() as i32
}

fn ext4_fc_unlock(sb: *mut super_block, ctx: i32) {
memalloc_nofs_restore(ctx as u32);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_fc_lock)) };
}

fn ext4_set_mount_flag(sb: *mut super_block, bit: i32) {
set_bit(bit as i64, (unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_mount_flags) }) as *mut u64);
}

fn ext4_clear_mount_flag(sb: *mut super_block, bit: i32) {
clear_bit(bit as i64, (unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_mount_flags) }) as *mut u64);
}

fn ext4_test_mount_flag(sb: *mut super_block, bit: i32) -> i32 {
test_bit__4!(bit, unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_mount_flags) })
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

fn ext4_clear_inode_state(inode: *mut inode, bit: i32) {
clear_bit(bit.wrapping_add(32) as i64, (unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) }) as *mut u64);
}

fn ext4_inode_state_wait_word(inode: *mut inode) -> *mut u64 {
unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) }
}

fn ext4_inode_state_wait_bit(bit: i32) -> i32 {
bit.wrapping_add(32)
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


fn ext4_raw_inode(iloc: &ext4_iloc) -> *mut ext4_inode {
unsafe { (*(*(iloc as *const ext4_iloc)).bh).b_data }.wrapping_offset((unsafe { (*(iloc as *const ext4_iloc)).offset }) as isize) as *mut ext4_inode
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


fn ext4_has_inline_data(inode: *mut inode) -> i32 {
let __v0: bool = ext4_test_inode_flag(inode, 28i32) != 0 && (unsafe { (*EXT4_I(inode)).i_inline_off }) != 0;
__v0 as i32
}

// Maximum number of logical blocks in a file; ext4_extent's ee_block is
// __le32.
// Add new method to test whether block and inode bitmaps are properly
// initialized. With uninit_bg reading the block from disk is not enough
// to mark the bitmap uptodate. We need to also zero-out the bitmap
// Ensure @handle has at least @check_creds credits available. If not,
// transaction will be extended or restarted to contain at least @extend_cred
// credits. Before restarting transaction @fn is executed to allow for cleanup
// before the transaction is restarted.
//
// The return value is < 0 in case of error, 0 in case the handle has enough
// credits or transaction extension succeeded, 1 in case transaction had to be
// restarted.
// Ext4 inode journal modes
// journal data mode
// ordered data mode
// writeback data mode
fn ext4_should_journal_data(inode: *mut inode) -> i32 {
(unsafe { ext4_inode_journal_mode(inode) }) & 1
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
fn ext4_ext_mark_unwritten(ext: *mut ext4_extent) {
// We can not have an unwritten extent of zero length!
loop {
if unlikely!((((unsafe { (*ext).ee_len }) as i32) as u64) & 18446744073709518847u64 == 0) != 0 {
loop {
unsafe {
core::arch::asm!("654: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 654b - ., 3; .popsection\n# {0}", const 654 as i32, options(nostack, att_syntax));
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

fn ext4_ext_store_pblock(ex: *mut ext4_extent, pb: u64) {
unsafe {
(*ex).ee_start_lo = (pb & 4294967295) as u32;
}
unsafe {
(*ex).ee_start_hi = (pb >> 31i32 >> 1i32 & 65535) as u16;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
    users: i32,
    seq: seqlock,
    umask: i32,
    in_exec: i32,
    root: path,
    pwd: path,
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


#[inline(always)]
fn seq_puts(m: *mut seq_file, s: *const u8) {
if !unsafe { __slate_intrinsic_70b71f185ebb8c4a(*s) } {
unsafe { __seq_puts(m, s) };
} else {
if (unsafe { *s.wrapping_offset(0 as isize) }) != 0 && (unsafe { *s.wrapping_offset(1 as isize) }) == 0 {
unsafe { seq_putc(m, *s.wrapping_offset(0 as isize)) };
} else {
unsafe { seq_write(m, s as *const (), __builtin_strlen(s)) };
}
}
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
    __slate_bits_0: __slate_bits::__SlateBits925U0,
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
    __slate_bits_0: __slate_bits::__SlateBits550U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
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
struct ext4_locality_group {
    lg_mutex: mutex,
    lg_prealloc_list: [list_head; 10],
    lg_prealloc_lock: spinlock,
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
fn __do_trace_ext4_fc_replay_scan(sb: *mut super_block, error: i32, off: i32) {
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
let mut __UNIQUE_ID_guard_1431: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1431) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1432: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1432) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1434() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_replay_scan)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1433: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1433;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut super_block, i32, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut super_block, i32, i32) -> i32>>(__SCT__tp_func_ext4_fc_replay_scan as *const ()) };
unsafe { __v12.unwrap()(__data, sb, error, off) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_replay_scan: __SlateAlign16<[u8; 20]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 114u8, 101u8, 112u8, 108u8, 97u8, 121u8, 95u8, 115u8, 99u8, 97u8, 110u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_replay_scan_1435: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_replay_scan) }) as *mut ();

fn trace_ext4_fc_replay_scan(sb: *mut super_block, error: i32, off: i32) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_replay_scan.key) }) != 0 {
__do_trace_ext4_fc_replay_scan(sb, error, off);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1436: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_replay(sb: *mut super_block, tag: i32, ino: i32, priv1: i32, priv2: i32) {
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
let mut __UNIQUE_ID_guard_1438: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1438) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1439: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1439) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1441() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_replay)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1440: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1440;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut super_block, i32, i32, i32, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut super_block, i32, i32, i32, i32) -> i32>>(__SCT__tp_func_ext4_fc_replay as *const ()) };
unsafe { __v12.unwrap()(__data, sb, tag, ino, priv1, priv2) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_replay: [u8; 15] = [101, 120, 116, 52, 95, 102, 99, 95, 114, 101, 112, 108, 97, 121, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_replay_1442: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_replay) }) as *mut ();

fn trace_ext4_fc_replay(sb: *mut super_block, tag: i32, ino: i32, priv1: i32, priv2: i32) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_replay.key) }) != 0 {
__do_trace_ext4_fc_replay(sb, tag, ino, priv1, priv2);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1443: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_commit_start(sb: *mut super_block, commit_tid: u32) {
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
let mut __UNIQUE_ID_guard_1445: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1445) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1446: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1446) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1448() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_commit_start)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1447: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1447;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut super_block, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut super_block, u32) -> i32>>(__SCT__tp_func_ext4_fc_commit_start as *const ()) };
unsafe { __v12.unwrap()(__data, sb, commit_tid) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_commit_start: __SlateAlign16<[u8; 21]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 99u8, 111u8, 109u8, 109u8, 105u8, 116u8, 95u8, 115u8, 116u8, 97u8, 114u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_commit_start_1449: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_commit_start) }) as *mut ();

fn trace_ext4_fc_commit_start(sb: *mut super_block, commit_tid: u32) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_commit_start.key) }) != 0 {
__do_trace_ext4_fc_commit_start(sb, commit_tid);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1450: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_commit_stop(sb: *mut super_block, nblks: i32, reason: i32, commit_tid: u32) {
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
let mut __UNIQUE_ID_guard_1452: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1452) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1453: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1453) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1455() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_commit_stop)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1454: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1454;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut super_block, i32, i32, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut super_block, i32, i32, u32) -> i32>>(__SCT__tp_func_ext4_fc_commit_stop as *const ()) };
unsafe { __v12.unwrap()(__data, sb, nblks, reason, commit_tid) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_commit_stop: __SlateAlign16<[u8; 20]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 99u8, 111u8, 109u8, 109u8, 105u8, 116u8, 95u8, 115u8, 116u8, 111u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_commit_stop_1456: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_commit_stop) }) as *mut ();

fn trace_ext4_fc_commit_stop(sb: *mut super_block, nblks: i32, reason: i32, commit_tid: u32) {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_commit_stop.key) }) != 0 {
__do_trace_ext4_fc_commit_stop(sb, nblks, reason, commit_tid);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1457: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __trace_ext4_fc_lock_updates_enabled() -> bool {
static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_lock_updates.key) }) != 0
}

fn trace_ext4_fc_lock_updates_enabled() -> bool {
__trace_ext4_fc_lock_updates_enabled()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1458: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_lock_updates(sb: *mut super_block, commit_tid: u32, locked_ns: u64, nr_inodes: u32, nr_ranges: u32, err: i32, snap_err: i32) {
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
let mut __UNIQUE_ID_guard_1459: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1459) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1460: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1460) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1462() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_lock_updates)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1461: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1461;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut super_block, u32, u64, u32, u32, i32, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut super_block, u32, u64, u32, u32, i32, i32) -> i32>>(__SCT__tp_func_ext4_fc_lock_updates as *const ()) };
unsafe { __v12.unwrap()(__data, sb, commit_tid, locked_ns, nr_inodes, nr_ranges, err, snap_err) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_lock_updates: __SlateAlign16<[u8; 21]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 108u8, 111u8, 99u8, 107u8, 95u8, 117u8, 112u8, 100u8, 97u8, 116u8, 101u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_lock_updates_1463: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_lock_updates) }) as *mut ();

fn trace_call__ext4_fc_lock_updates(sb: *mut super_block, commit_tid: u32, locked_ns: u64, nr_inodes: u32, nr_ranges: u32, err: i32, snap_err: i32) {
__do_trace_ext4_fc_lock_updates(sb, commit_tid, locked_ns, nr_inodes, nr_ranges, err, snap_err);
}

fn __do_trace_ext4_fc_stats(sb: *mut super_block) {
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
let mut __UNIQUE_ID_guard_1466: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1466) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1467: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1467) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1469() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_stats)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1468: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1468;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut super_block) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut super_block) -> i32>>(__SCT__tp_func_ext4_fc_stats as *const ()) };
unsafe { __v12.unwrap()(__data, sb) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_stats: [u8; 14] = [101, 120, 116, 52, 95, 102, 99, 95, 115, 116, 97, 116, 115, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_stats_1470: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_stats) }) as *mut ();

fn trace_ext4_fc_stats(sb: *mut super_block) {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_stats.key) }) != 0 {
__do_trace_ext4_fc_stats(sb);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1471: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_track_create(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry, ret: i32) {
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
let mut __UNIQUE_ID_guard_1473: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1473) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1474: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1474) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1476() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_create)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1475: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1475;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, *mut dentry, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, *mut dentry, i32) -> i32>>(__SCT__tp_func_ext4_fc_track_create as *const ()) };
unsafe { __v12.unwrap()(__data, handle, inode, dentry, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_track_create: __SlateAlign16<[u8; 21]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 116u8, 114u8, 97u8, 99u8, 107u8, 95u8, 99u8, 114u8, 101u8, 97u8, 116u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_track_create_1477: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_track_create) }) as *mut ();

fn trace_ext4_fc_track_create(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry, ret: i32) {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_create.key) }) != 0 {
__do_trace_ext4_fc_track_create(handle, inode, dentry, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1478: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_track_link(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry, ret: i32) {
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
let mut __UNIQUE_ID_guard_1480: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1480) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1481: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1481) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1483() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_link)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1482: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1482;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, *mut dentry, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, *mut dentry, i32) -> i32>>(__SCT__tp_func_ext4_fc_track_link as *const ()) };
unsafe { __v12.unwrap()(__data, handle, inode, dentry, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_track_link: __SlateAlign16<[u8; 19]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 116u8, 114u8, 97u8, 99u8, 107u8, 95u8, 108u8, 105u8, 110u8, 107u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_track_link_1484: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_track_link) }) as *mut ();

fn trace_ext4_fc_track_link(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry, ret: i32) {
if static_branch_unlikely__11!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_link.key) }) != 0 {
__do_trace_ext4_fc_track_link(handle, inode, dentry, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1485: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_track_unlink(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry, ret: i32) {
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
let mut __UNIQUE_ID_guard_1487: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1487) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1488: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1488) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1490() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_unlink)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1489: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1489;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, *mut dentry, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, *mut dentry, i32) -> i32>>(__SCT__tp_func_ext4_fc_track_unlink as *const ()) };
unsafe { __v12.unwrap()(__data, handle, inode, dentry, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_track_unlink: __SlateAlign16<[u8; 21]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 116u8, 114u8, 97u8, 99u8, 107u8, 95u8, 117u8, 110u8, 108u8, 105u8, 110u8, 107u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_track_unlink_1491: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_track_unlink) }) as *mut ();

fn trace_ext4_fc_track_unlink(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry, ret: i32) {
if static_branch_unlikely__8!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_unlink.key) }) != 0 {
__do_trace_ext4_fc_track_unlink(handle, inode, dentry, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1492: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_track_inode(handle: *mut jbd2_journal_handle, inode: *mut inode, ret: i32) {
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
let mut __UNIQUE_ID_guard_1494: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1494) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1495: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1495) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1497() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_inode)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1496: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1496;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, i32) -> i32>>(__SCT__tp_func_ext4_fc_track_inode as *const ()) };
unsafe { __v12.unwrap()(__data, handle, inode, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_track_inode: __SlateAlign16<[u8; 20]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 116u8, 114u8, 97u8, 99u8, 107u8, 95u8, 105u8, 110u8, 111u8, 100u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_track_inode_1498: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_track_inode) }) as *mut ();

fn trace_ext4_fc_track_inode(handle: *mut jbd2_journal_handle, inode: *mut inode, ret: i32) {
if static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_inode.key) }) != 0 {
__do_trace_ext4_fc_track_inode(handle, inode, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1499: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_track_range(handle: *mut jbd2_journal_handle, inode: *mut inode, start: i64, end: i64, ret: i32) {
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
let mut __UNIQUE_ID_guard_1501: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1501) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1502: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1502) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1504() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_range)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1503: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1503;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, i64, i64, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut jbd2_journal_handle, *mut inode, i64, i64, i32) -> i32>>(__SCT__tp_func_ext4_fc_track_range as *const ()) };
unsafe { __v12.unwrap()(__data, handle, inode, start, end, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_track_range: __SlateAlign16<[u8; 20]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 116u8, 114u8, 97u8, 99u8, 107u8, 95u8, 114u8, 97u8, 110u8, 103u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_track_range_1505: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_track_range) }) as *mut ();

fn trace_ext4_fc_track_range(handle: *mut jbd2_journal_handle, inode: *mut inode, start: i64, end: i64, ret: i32) {
if static_branch_unlikely__12!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_track_range.key) }) != 0 {
__do_trace_ext4_fc_track_range(handle, inode, start, end, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1506: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_fc_cleanup(journal: *mut journal_s, full: i32, tid: u32) {
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
let mut __UNIQUE_ID_guard_1508: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1508) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1509: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1509) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1511() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_fc_cleanup)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1510: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1510;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut journal_s, i32, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut journal_s, i32, u32) -> i32>>(__SCT__tp_func_ext4_fc_cleanup as *const ()) };
unsafe { __v12.unwrap()(__data, journal, full, tid) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_fc_cleanup: __SlateAlign16<[u8; 16]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 99u8, 95u8, 99u8, 108u8, 101u8, 97u8, 110u8, 117u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_fc_cleanup_1512: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_fc_cleanup) }) as *mut ();

fn trace_ext4_fc_cleanup(journal: *mut journal_s, full: i32, tid: u32) {
if static_branch_unlikely__10!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_fc_cleanup.key) }) != 0 {
__do_trace_ext4_fc_cleanup(journal, full, tid);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1513: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// fs/ext4/fast_commit.c
//
// Written by Harshad Shirwadkar <harshadshirwadkar@gmail.com>
//
// Ext4 fast commits routines.
// Ext4 Fast Commits
//
// Ext4 fast commits implement fine grained journalling for Ext4.
//
// Fast commits are organized as a log of tag-length-value (TLV) structs. (See
// struct ext4_fc_tl). Each TLV contains some delta that is replayed TLV by
// TLV during the recovery phase. For the scenarios for which we currently
// don't have replay code, fast commit falls back to full commits.
// Fast commits record delta in one of the following three categories.
//
// (A) Directory entry updates:
//
// - EXT4_FC_TAG_UNLINK		- records directory entry unlink
// - EXT4_FC_TAG_LINK		- records directory entry link
// - EXT4_FC_TAG_CREAT		- records inode and directory entry creation
//
// (B) File specific data range updates:
//
// - EXT4_FC_TAG_ADD_RANGE	- records addition of new blocks to an inode
// - EXT4_FC_TAG_DEL_RANGE	- records deletion of blocks from an inode
//
// (C) Inode metadata (mtime / ctime etc):
//
// - EXT4_FC_TAG_INODE		- record the inode that should be replayed
// 				  during recovery. Note that iblocks field is
// 				  not replayed and instead derived during
// 				  replay.
// Commit Operation
// With fast commits, we maintain all the directory entry operations in the
// order in which they are issued in an in-memory queue. This queue is flushed
// to disk during the commit operation. We also maintain a list of inodes
// that need to be committed during a fast commit in another in memory queue of
// inodes. During the commit operation, we commit in the following order:
//
// [1] Prepare all the inodes to write out their data by setting
//     "EXT4_STATE_FC_FLUSHING_DATA". This ensures that inode cannot be
//     deleted while it is being flushed.
// [2] Flush data buffers to disk and clear "EXT4_STATE_FC_FLUSHING_DATA"
//     state.
// [3] Lock the journal by calling jbd2_journal_lock_updates(). This ensures
//     that all the existing handles finish and no new handles can start.
// [4] Mark all the fast commit eligible inodes as undergoing fast commit by
//     setting "EXT4_STATE_FC_COMMITTING" state, and snapshot the inode state
//     needed for log writing.
// [5] Unlock the journal by calling jbd2_journal_unlock_updates(). This allows
//     starting of new handles. Updates to inodes being fast committed are
//     tracked for requeue rather than blocking.
// [6] Commit all the directory entry updates in the fast commit space.
// [7] Commit all the changed inodes in the fast commit space.
// [8] Write tail tag (this tag ensures the atomicity, please read the following
//     section for more details).
// [9] Clear "EXT4_STATE_FC_COMMITTING" and wake up waiters in
//     ext4_fc_cleanup().
//
// All the inode updates must be enclosed within jbd2_journal_start()
// and jbd2_journal_stop() similar to JBD2 journaling.
//
// Fast Commit Ineligibility
//
// Not all operations are supported by fast commits today (e.g extended
// attributes). Fast commit ineligibility is marked by calling
// ext4_fc_mark_ineligible(): This makes next fast commit operation to fall back
// to full commit.
//
// Atomicity of commits
// In order to guarantee atomicity during the commit operation, fast commit
// uses "EXT4_FC_TAG_TAIL" tag that marks a fast commit as complete. Tail
// tag contains CRC of the contents and TID of the transaction after which
// this fast commit should be applied. Recovery code replays fast commit
// logs only if there's at least 1 valid tail present. For every fast commit
// operation, there is 1 tail. This means, we may end up with multiple tails
// in the fast commit space. Here's an example:
//
// - Create a new file A and remove existing file B
// - fsync()
// - Append contents to file A
// - Truncate file A
// - fsync()
//
// The fast commit space at the end of above operations would look like this:
//      [HEAD] [CREAT A] [UNLINK B] [TAIL] [ADD_RANGE A] [DEL_RANGE A] [TAIL]
//             |<---  Fast Commit 1   --->|<---      Fast Commit 2     ---->|
//
// Replay code should thus check for all the valid tails in the FC area.
//
// Fast Commit Replay Idempotence
//
// Fast commits tags are idempotent in nature provided the recovery code follows
// certain rules. The guiding principle that the commit path follows while
// committing is that it stores the result of a particular operation instead of
// storing the procedure.
//
// Let's consider this rename operation: 'mv /a /b'. Let's assume dirent '/a'
// was associated with inode 10. During fast commit, instead of storing this
// operation as a procedure "rename a to b", we store the resulting file system
// state as a "series" of outcomes:
//
// - Link dirent b to inode 10
// - Unlink dirent a
// - Inode <10> with valid refcount
//
// Now when recovery code runs, it needs "enforce" this state on the file
// system. This is what guarantees idempotence of fast commit replay.
//
// Let's take an example of a procedure that is not idempotent and see how fast
// commits make it idempotent. Consider following sequence of operations:
//
//     rm A;    mv B A;    read A
//  (x)     (y)        (z)
//
// (x), (y) and (z) are the points at which we can crash. If we store this
// sequence of operations as is then the replay is not idempotent. Let's say
// while in replay, we crash at (z). During the second replay, file A (which was
// actually created as a result of "mv B A" operation) would get deleted. Thus,
// file named A would be absent when we try to read A. So, this sequence of
// operations is not idempotent. However, as mentioned above, instead of storing
// the procedure fast commits store the outcome of each procedure. Thus the fast
// commit log for above procedure would be as follows:
//
// (Let's assume dirent A was linked to inode 10 and dirent B was linked to
// inode 11 before the replay)
//
//    [Unlink A]   [Link A to inode 11]   [Unlink B]   [Inode 11]
// (w)          (x)                    (y)          (z)
//
// If we crash at (z), we will have file A linked to inode 11. During the second
// replay, we will remove file A (inode 11). But we will create it back and make
// it point to inode 11. We won't find B, so we'll just skip that step. At this
// point, the refcount for inode 11 is not reliable, but that gets fixed by the
// replay of last inode 11 tag. Crashes at points (w), (x) and (y) get handled
// similarly. Thus, by converting a non-idempotent procedure into a series of
// idempotent outcomes, fast commits ensured idempotence during the replay.
//
// Locking
// sbi->s_fc_lock protects the fast commit inodes queue and the fast commit
// dentry queue. ei->i_fc_lock protects the fast commit related info in a given
// inode. Most of the code avoids acquiring both the locks, but if one must do
// that then sbi->s_fc_lock must be acquired before ei->i_fc_lock.
//
// TODOs
//
// 0) Fast commit replay path hardening: Fast commit replay code should use
//    journal handles to make sure all the updates it does during the replay
//    path are atomic. With that if we crash during fast commit replay, after
//    trying to do recovery again, we will find a file system where fast commit
//    area is invalid (because new full commit would be found). In order to deal
//    with that, fast commit replay code should ensure that the "FC_REPLAY"
//    superblock state is persisted before starting the replay, so that after
//    the crash, fast commit recovery code can look at that flag and perform
//    fast commit recovery even if that area is invalidated by later full
//    commits.
//
// 1) Handle more ineligible cases.
//
// 2) Change ext4_fc_commit() to lookup logical to physical mapping using extent
//    status tree. This would get rid of the need to call ext4_fc_track_inode()
//    before acquiring i_data_sem. To do that we would need to ensure that
//    modified extents from the extent status tree are not evicted from memory.
static mut ext4_fc_dentry_cachep: *mut kmem_cache = unsafe { core::mem::zeroed() };

static mut ext4_fc_range_cachep: *mut kmem_cache = unsafe { core::mem::zeroed() };

// Avoid spending unbounded time/memory snapshotting highly fragmented files
// under jbd2_journal_lock_updates(). If we exceed this limit, fall back to
// full commit.
fn ext4_fc_set_snap_err(snap_err: *mut i32, err: i32) {
if !snap_err.is_null() && (unsafe { *snap_err }) == 0 {
unsafe {
*snap_err = err;
}
}
}

fn ext4_fc_reset_inode(inode: *mut inode) {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
unsafe {
(*ei).i_fc_lblk_start = 0;
}
unsafe {
(*ei).i_fc_lblk_len = 0;
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_fc_init_inode(inode: *mut inode) {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
ext4_fc_reset_inode(inode);
ext4_clear_inode_state(inode, 10i32);
ext4_clear_inode_state(inode, 13i32);
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_dilist) });
unsafe {
(*ei).i_fc_snap = core::ptr::null_mut::<ext4_fc_inode_snap>();
}
}

fn ext4_fc_disabled(sb: *mut super_block) -> bool {
(unsafe { (*EXT4_SB(sb)).s_mount_opt2 }) & 16 == 0 || ((unsafe { (*EXT4_SB(sb)).s_mount_state }) as i32) & 32 != 0
}

fn ext4_fc_eligible(sb: *mut super_block) -> bool {
!ext4_fc_disabled(sb) && ext4_test_mount_flag(sb, 1i32) == 0
}

/// Wait for an inode fast-commit state bit to clear while dropping the
/// fast-commit lock around schedule().
fn ext4_fc_wait_inode_state(inode: *mut inode, bit: i32, alloc_ctx: &mut i32) {
let wait_word: *mut u64 = ext4_inode_state_wait_word(inode);
let wait_bit: i32 = ext4_inode_state_wait_bit(bit);
'__slate_break_0: while ext4_test_inode_state(inode, bit) != 0 {
let mut wait: wait_bit_queue_entry = unsafe { core::mem::zeroed() };
wait = wait_bit_queue_entry { key: {
let mut __t0: wait_bit_key = unsafe { core::mem::zeroed() };
__t0.flags = wait_word;
__t0.bit_nr = wait_bit;
__t0
}, wq_entry: {
let mut __t1: wait_queue_entry = unsafe { core::mem::zeroed() };
__t1.private = get_current() as *mut ();
__t1.func = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>>(wake_bit_function as *const ()) };
__t1.entry = list_head { next: core::ptr::addr_of_mut!(wait.wq_entry.entry), prev: core::ptr::addr_of_mut!(wait.wq_entry.entry) };
__t1
} };
let wq: *mut wait_queue_head = unsafe { bit_waitqueue(wait_word, wait_bit) };
unsafe { prepare_to_wait(wq, core::ptr::addr_of_mut!(wait.wq_entry), 2i32) };
if ext4_test_inode_state(inode, bit) != 0 {
ext4_fc_unlock(unsafe { (*inode).i_sb }, unsafe { *(alloc_ctx as *mut i32) });
unsafe { schedule() };
unsafe {
*(alloc_ctx as *mut i32) = ext4_fc_lock((*inode).i_sb);
}
}
unsafe { finish_wait(wq, core::ptr::addr_of_mut!(wait.wq_entry)) };
}
}

fn ext4_fc_wake_inode_state(inode: *mut inode, bit: i32) {
unsafe { wake_up_bit(ext4_inode_state_wait_word(inode), ext4_inode_state_wait_bit(bit)) };
}

fn ext4_fc_snap_stats_update_max(stat: *mut __SlateRecord_types_h_197, value: u64) {
let mut old: u64 = atomic64_read(stat as *const __SlateRecord_types_h_197) as u64;
while value > old {
let prev: u64 = atomic64_cmpxchg(stat, old as i64, value as i64) as u64;
if prev == old {
break;
}
old = prev;
}
}

/// Remove inode from fast commit list. If the inode is being committed
/// we wait until inode commit is done.
#[unsafe(no_mangle)]
extern "C" fn ext4_fc_del(inode: *mut inode) {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let mut fc_dentry: *mut ext4_fc_dentry_update = unsafe { core::mem::zeroed() };
let mut alloc_ctx: i32 = 0;
if ext4_fc_disabled(unsafe { (*inode).i_sb }) {
return;
}
alloc_ctx = ext4_fc_lock(unsafe { (*inode).i_sb });
let __v1: bool = list_empty((unsafe { core::ptr::addr_of_mut!((*ei).i_fc_list) }) as *const list_head) != 0 && list_empty((unsafe { core::ptr::addr_of_mut!((*ei).i_fc_dilist) }) as *const list_head) != 0;
if __v1 {
ext4_fc_free_inode_snap(inode);
ext4_fc_unlock(unsafe { (*inode).i_sb }, alloc_ctx);
return;
}
// Wait for ongoing fast commit to finish. We cannot remove the inode
// from fast commit lists while it is being committed. If we wake from
// FC_FLUSHING_DATA, re-check FC_COMMITTING before deleting because the
// commit thread sets FC_COMMITTING only after clearing FLUSHING_DATA.
loop {
ext4_fc_wait_inode_state(inode, 10i32, unsafe { &mut (*core::ptr::addr_of_mut!(alloc_ctx)) });
if ext4_test_inode_state(inode, 11i32) == 0 {
break;
}
ext4_fc_wait_inode_state(inode, 11i32, unsafe { &mut (*core::ptr::addr_of_mut!(alloc_ctx)) });
}
ext4_fc_free_inode_snap(inode);
list_del_init(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_list) });
// Since this inode is getting removed, let's also remove all FC dentry
// create references, since it is not needed to log it anyways.
if list_empty((unsafe { core::ptr::addr_of_mut!((*ei).i_fc_dilist) }) as *const list_head) != 0 {
ext4_fc_unlock(unsafe { (*inode).i_sb }, alloc_ctx);
return;
}
fc_dentry = list_first_entry__12!(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_dilist) }, ext4_fc_dentry_update);
let mut __v4: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*fc_dentry).fcd_op }) != 3) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1535: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1535b - ., 3; .popsection\n# {0}", const 1535 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_fc_del_5, sym __slate_asm_string_ext4_fc_del_6, const 333 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1536: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1536b - ., 4; .popsection\n# {0}", const 1536 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_list) });
list_del_init(unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_dilist) });
let mut __v9: i64;
let __ret_warn_on: i32 = (list_empty((unsafe { core::ptr::addr_of_mut!((*ei).i_fc_dilist) }) as *const list_head) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1537: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1537b - ., 3; .popsection\n# {0}", const 1537 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_fc_del_10, sym __slate_asm_string_ext4_fc_del_11, const 337 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1538: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1538b - ., 4; .popsection\n# {0}", const 1538 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v9 = (__ret_warn_on != 0) as i64;
ext4_fc_unlock(unsafe { (*inode).i_sb }, alloc_ctx);
unsafe { release_dentry_name_snapshot(core::ptr::addr_of_mut!((*fc_dentry).fcd_name)) };
unsafe { kmem_cache_free(ext4_fc_dentry_cachep, fc_dentry as *mut ()) };
}

/// Mark file system as fast commit ineligible, and record latest
/// ineligible transaction tid. This means until the recorded
/// transaction, commit operation would result in a full jbd2 commit.
#[unsafe(no_mangle)]
extern "C" fn ext4_fc_mark_ineligible(sb: *mut super_block, reason: i32, handle: *mut jbd2_journal_handle) {
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let mut tid: u32 = 0;
let mut has_transaction: bool = true;
let mut is_ineligible: bool = false;
let mut alloc_ctx: i32 = 0;
if ext4_fc_disabled(sb) {
return;
}
if !IS_ERR_OR_NULL(handle as *const ()) {
tid = unsafe { (*(*handle).__slate_anon_0.h_transaction).t_tid };
} else {
read_lock!(unsafe { core::ptr::addr_of_mut!((*(*sbi).s_journal).j_state_lock) });
if !unsafe { (*(*sbi).s_journal).j_running_transaction }.is_null() {
tid = unsafe { (*(*(*sbi).s_journal).j_running_transaction).t_tid };
} else {
has_transaction = false;
}
read_unlock!(unsafe { core::ptr::addr_of_mut!((*(*sbi).s_journal).j_state_lock) });
}
alloc_ctx = ext4_fc_lock(sb);
is_ineligible = ext4_test_mount_flag(sb, 1i32) != 0;
let __v0: bool = has_transaction && (!is_ineligible || tid_gt(tid, unsafe { (*sbi).s_fc_ineligible_tid }) != 0);
if __v0 {
unsafe {
(*sbi).s_fc_ineligible_tid = tid;
}
}
ext4_set_mount_flag(sb, 1i32);
ext4_fc_unlock(sb, alloc_ctx);
let mut __v2: i64;
let __ret_warn_on: i32 = (reason >= 13) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1539: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1539b - ., 3; .popsection\n# {0}", const 1539 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_fc_mark_ineligible_3, sym __slate_asm_string_ext4_fc_mark_ineligible_4, const 376 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1540: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1540b - ., 4; .popsection\n# {0}", const 1540 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
let __v7: *mut u32 = unsafe { (*sbi).s_fc_stats.fc_ineligible_reason_count.as_mut_ptr() as *mut u32 }.wrapping_offset(reason as isize);
let __v8: u32 = unsafe { *__v7 };
unsafe {
*__v7 = __v8.wrapping_add(1);
}
}

/// Generic fast commit tracking function. If this is the first time this we are
/// called after a full commit, we initialize fast commit fields and then call
/// __fc_track_fn() with update = 0. If we have already been called after a full
/// commit, we pass update = 1. Based on that, the track function can determine
/// if it needs to track a field for the first time or if it needs to just
/// update the previously tracked value.
///
/// If enqueue is set, this function enqueues the inode in fast commit list.
fn ext4_fc_track_template(handle: *mut jbd2_journal_handle, inode: *mut inode, __fc_track_fn: Option<unsafe extern "C" fn(*mut jbd2_journal_handle, *mut inode, *mut (), bool) -> i32>, args: *mut (), enqueue: i32) -> i32 {
let mut update: bool = false;
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let sbi: *mut ext4_sb_info = EXT4_SB(unsafe { (*inode).i_sb });
let tid: u32 = unsafe { (*(*handle).__slate_anon_0.h_transaction).t_tid };
let mut alloc_ctx: i32 = 0;
let mut ret: i32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
if ext4_test_inode_state(inode, 10i32) != 0 {
ext4_set_inode_state(inode, 13i32);
}
if tid == unsafe { (*ei).i_sync_tid } {
update = true;
} else {
ext4_fc_reset_inode(inode);
unsafe {
(*ei).i_sync_tid = tid;
}
}
ret = unsafe { __fc_track_fn.unwrap()(handle, inode, args, update) };
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
if enqueue == 0 {
return ret;
}
alloc_ctx = ext4_fc_lock(unsafe { (*inode).i_sb });
if list_empty((unsafe { core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_fc_list) }) as *const list_head) != 0 {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*EXT4_I!(inode)).i_fc_list) }, if (unsafe { (*(*sbi).s_journal).j_flags }) & 512 != 0 || (unsafe { (*(*sbi).s_journal).j_flags }) & 256 != 0 { unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(1 as isize) } else { unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize) });
}
ext4_fc_unlock(unsafe { (*inode).i_sb }, alloc_ctx);
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __track_dentry_update_args {
    dentry: *mut dentry,
    op: i32,
}


/// __track_fn for directory entry updates. Called with ei->i_fc_lock.
#[unsafe(link_section = ".text.slate_distinct.fast_commit.__track_dentry_update")]
extern "C" fn __track_dentry_update(handle: *mut jbd2_journal_handle, inode: *mut inode, arg: *mut (), update: bool) -> i32 {
let mut node: *mut ext4_fc_dentry_update = unsafe { core::mem::zeroed() };
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let dentry: *mut dentry = unsafe { (*(arg as *mut __track_dentry_update_args)).dentry };
let dir: *mut inode = unsafe { (*(*dentry).d_parent).d_inode };
let sb: *mut super_block = unsafe { (*inode).i_sb };
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let mut alloc_ctx: i32 = 0;
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
if IS_ENCRYPTED!(dir) != 0 {
ext4_fc_mark_ineligible(sb, 9i32, handle);
spin_lock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
return 95i32.wrapping_neg();
}
node = alloc_hooks!(unsafe { kmem_cache_alloc_noprof(ext4_fc_dentry_cachep, 3136u32) }) as *mut ext4_fc_dentry_update;
if node.is_null() {
ext4_fc_mark_ineligible(sb, 3i32, handle);
spin_lock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
return 12i32.wrapping_neg();
}
unsafe {
(*node).fcd_op = (*(arg as *mut __track_dentry_update_args)).op;
}
unsafe {
(*node).fcd_parent = (*dir).i_ino as i32;
}
unsafe {
(*node).fcd_ino = (*inode).i_ino as i32;
}
unsafe { take_dentry_name_snapshot(core::ptr::addr_of_mut!((*node).fcd_name), dentry) };
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*node).fcd_dilist) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*node).fcd_list) });
alloc_ctx = ext4_fc_lock(sb);
if (unsafe { (*(*sbi).s_journal).j_flags }) & 512 != 0 || (unsafe { (*(*sbi).s_journal).j_flags }) & 256 != 0 {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*node).fcd_list) }, unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(1 as isize));
} else {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*node).fcd_list) }, unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize));
}
// This helps us keep a track of all fc_dentry updates which is part of
// this ext4 inode. So in case the inode is getting unlinked, before
// even we get a chance to fsync, we could remove all fc_dentry
// references while evicting the inode in ext4_fc_del().
// Also with this, we don't need to loop over all the inodes in
// sbi->s_fc_q to get the corresponding inode in
// ext4_fc_commit_dentry_updates().
if (unsafe { (*(arg as *mut __track_dentry_update_args)).op }) == 3 {
let mut __v6: i64;
let __ret_warn_on: i32 = (list_empty((unsafe { core::ptr::addr_of_mut!((*ei).i_fc_dilist) }) as *const list_head) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1541: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1541b - ., 3; .popsection\n# {0}", const 1541 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___track_dentry_update_7, sym __slate_asm_string___track_dentry_update_8, const 488 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1542: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1542b - ., 4; .popsection\n# {0}", const 1542 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
list_add_tail(unsafe { core::ptr::addr_of_mut!((*node).fcd_dilist) }, unsafe { core::ptr::addr_of_mut!((*ei).i_fc_dilist) });
}
ext4_fc_unlock(sb, alloc_ctx);
spin_lock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
0
}

#[unsafe(no_mangle)]
extern "C" fn __ext4_fc_track_unlink(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry) {
let mut args: __track_dentry_update_args = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(args) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut ret: i32 = 0;
args.dentry = dentry;
args.op = 5;
ret = ext4_fc_track_template(handle, inode, Some(__track_dentry_update), core::ptr::addr_of_mut!(args) as *mut (), 0i32);
trace_ext4_fc_track_unlink(handle, inode, dentry, ret);
}

#[unsafe(no_mangle)]
extern "C" fn ext4_fc_track_unlink(handle: *mut jbd2_journal_handle, dentry: *mut dentry) {
let inode: *mut inode = d_inode(dentry as *const dentry);
if ext4_fc_eligible(unsafe { (*inode).i_sb }) {
__ext4_fc_track_unlink(handle, inode, dentry);
}
}

#[unsafe(no_mangle)]
extern "C" fn __ext4_fc_track_link(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry) {
let mut args: __track_dentry_update_args = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(args) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut ret: i32 = 0;
args.dentry = dentry;
args.op = 4;
ret = ext4_fc_track_template(handle, inode, Some(__track_dentry_update), core::ptr::addr_of_mut!(args) as *mut (), 0i32);
trace_ext4_fc_track_link(handle, inode, dentry, ret);
}

#[unsafe(no_mangle)]
extern "C" fn ext4_fc_track_link(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry) {
if ext4_fc_eligible(unsafe { (*inode).i_sb }) {
__ext4_fc_track_link(handle, inode, dentry);
}
}

#[unsafe(no_mangle)]
extern "C" fn __ext4_fc_track_create(handle: *mut jbd2_journal_handle, inode: *mut inode, dentry: *mut dentry) {
let mut args: __track_dentry_update_args = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(args) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut ret: i32 = 0;
args.dentry = dentry;
args.op = 3;
ret = ext4_fc_track_template(handle, inode, Some(__track_dentry_update), core::ptr::addr_of_mut!(args) as *mut (), 0i32);
trace_ext4_fc_track_create(handle, inode, dentry, ret);
}

#[unsafe(no_mangle)]
extern "C" fn ext4_fc_track_create(handle: *mut jbd2_journal_handle, dentry: *mut dentry) {
let inode: *mut inode = d_inode(dentry as *const dentry);
if ext4_fc_eligible(unsafe { (*inode).i_sb }) {
__ext4_fc_track_create(handle, inode, dentry);
}
}

/// __track_fn for inode tracking
#[unsafe(link_section = ".text.slate_distinct.fast_commit.__track_inode")]
extern "C" fn __track_inode(handle: *mut jbd2_journal_handle, inode: *mut inode, arg: *mut (), update: bool) -> i32 {
if update {
return 17i32.wrapping_neg();
}
unsafe {
(*EXT4_I!(inode)).i_fc_lblk_len = 0;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ext4_fc_track_inode(handle: *mut jbd2_journal_handle, inode: *mut inode) {
let mut ret: i32 = 0;
if S_ISDIR!(unsafe { (*inode).i_mode }) {
return;
}
if ext4_should_journal_data(inode) != 0 {
ext4_fc_mark_ineligible(unsafe { (*inode).i_sb }, 8i32, handle);
return;
}
if !ext4_fc_eligible(unsafe { (*inode).i_sb }) {
return;
}
// Fast commit snapshots inode state at commit time, so there's no need
// to wait for EXT4_STATE_FC_COMMITTING here. If the inode is already
// on the commit queue, ext4_fc_cleanup() will requeue it for the new
// transaction once the current commit finishes.
// From this point on, this inode will not be committed either
// by fast or full commit as long as the handle is open.
ret = ext4_fc_track_template(handle, inode, Some(__track_inode), core::ptr::null_mut::<()>(), 1i32);
trace_ext4_fc_track_inode(handle, inode, ret);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __track_range_args {
    start: u32,
    end: u32,
}


// __track_fn for tracking data updates
#[unsafe(link_section = ".text.slate_distinct.fast_commit.__track_range")]
extern "C" fn __track_range(handle: *mut jbd2_journal_handle, inode: *mut inode, arg: *mut (), update: bool) -> i32 {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let mut oldstart: u32 = 0;
if (unsafe { (*inode).i_ino }) < (EXT4_FIRST_INO!(unsafe { (*inode).i_sb }) as u64) {
let mut __v1: i32;
if false {
unsafe { _printk(c"Special inode %llu being modified\n".as_ptr() as *const u8, (*inode).i_ino) };
}
__v1 = 0;
return 125i32.wrapping_neg();
}
oldstart = unsafe { (*ei).i_fc_lblk_start };
if update && (unsafe { (*ei).i_fc_lblk_len }) > 0 {
let mut __v2: u32;
let __UNIQUE_ID_x__1543: u32 = unsafe { (*ei).i_fc_lblk_start };
let __UNIQUE_ID_y__1544: u32 = unsafe { (*(arg as *mut __track_range_args)).start };
loop {
if __types_ok__4!(__UNIQUE_ID_x__1543, __UNIQUE_ID_y__1544) == 0 {
unsafe { __compiletime_assert_1545() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__1543 < __UNIQUE_ID_y__1544 { __UNIQUE_ID_x__1543 } else { __UNIQUE_ID_y__1544 };
unsafe {
(*ei).i_fc_lblk_start = __v2;
}
let mut __v4: u32;
let __UNIQUE_ID_x__1546: u32 = oldstart.wrapping_add(unsafe { (*ei).i_fc_lblk_len }).wrapping_sub(1);
let __UNIQUE_ID_y__1547: u32 = unsafe { (*(arg as *mut __track_range_args)).end };
loop {
if __types_ok__4!(__UNIQUE_ID_x__1546, __UNIQUE_ID_y__1547) == 0 {
unsafe { __compiletime_assert_1548() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__1546 > __UNIQUE_ID_y__1547 { __UNIQUE_ID_x__1546 } else { __UNIQUE_ID_y__1547 };
unsafe {
(*ei).i_fc_lblk_len = __v4.wrapping_sub((*ei).i_fc_lblk_start).wrapping_add(1);
}
} else {
unsafe {
(*ei).i_fc_lblk_start = (*(arg as *mut __track_range_args)).start;
}
unsafe {
(*ei).i_fc_lblk_len = unsafe { (*(arg as *mut __track_range_args)).end }.wrapping_sub((*(arg as *mut __track_range_args)).start).wrapping_add(1);
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ext4_fc_track_range(handle: *mut jbd2_journal_handle, inode: *mut inode, start: u32, end: u32) {
let mut args: __track_range_args = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(args) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut ret: i32 = 0;
if S_ISDIR!(unsafe { (*inode).i_mode }) {
return;
}
if !ext4_fc_eligible(unsafe { (*inode).i_sb }) {
return;
}
if ext4_has_inline_data(inode) != 0 {
ext4_fc_mark_ineligible(unsafe { (*inode).i_sb }, 0i32, handle);
return;
}
args.start = start;
args.end = end;
ret = ext4_fc_track_template(handle, inode, Some(__track_range), core::ptr::addr_of_mut!(args) as *mut (), 1i32);
trace_ext4_fc_track_range(handle, inode, start as i64, end as i64, ret);
}

fn ext4_fc_submit_bh(sb: *mut super_block, is_tail: bool) {
let mut write_flags: u32 = 38912u32;
let bh: *mut buffer_head = unsafe { (*EXT4_SB(sb)).s_fc_bh };
// Add REQ_FUA | REQ_PREFLUSH only its tail
if (unsafe { (*EXT4_SB(sb)).s_mount_opt }) & 131072 != 0 && is_tail {
write_flags |= 393216u32;
}
lock_buffer(bh);
set_buffer_dirty(bh);
set_buffer_uptodate(bh);
unsafe { bh_submit(bh, 1 | write_flags, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bio)>>(bh_end_write as *const ()) }) };
unsafe {
(*EXT4_SB(sb)).s_fc_bh = core::ptr::null_mut::<buffer_head>();
}
}

// Ext4 commit path routines
/// Allocate len bytes on a fast commit buffer.
///
/// During the commit time this function is used to manage fast commit
/// block space. We don't split a fast commit log onto different
/// blocks. So this function makes sure that if there's not enough space
/// on the current block, the remaining space in the current block is
/// marked as unused by adding EXT4_FC_TAG_PAD tag. In that case,
/// new block is from jbd2 and CRC is updated to reflect the padding
/// we added.
fn ext4_fc_reserve_space(sb: *mut super_block, len: i32, crc: &mut u32) -> *mut u8 {
let mut tl: ext4_fc_tl = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tl) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let mut bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let bsize: i32 = unsafe { (*(*sbi).s_journal).j_blocksize };
let mut ret: i32 = 0;
let off: i32 = ((unsafe { (*sbi).s_fc_bytes }) % (bsize as u32)) as i32;
let mut remaining: i32 = 0;
// If 'len' is too long to fit in any block alongside a PAD tlv, then we
// cannot fulfill the request.
if (len as u64) > (bsize as u64).wrapping_sub(core::mem::size_of::<ext4_fc_tl>() as u64) {
return core::ptr::null_mut::<u8>();
}
if unsafe { (*sbi).s_fc_bh }.is_null() {
ret = unsafe { jbd2_fc_get_buf((*EXT4_SB(sb)).s_journal, core::ptr::addr_of_mut!(bh)) };
if ret != 0 {
return core::ptr::null_mut::<u8>();
}
unsafe {
(*sbi).s_fc_bh = bh;
}
}
let dst: *mut u8 = unsafe { (*(*sbi).s_fc_bh).b_data }.wrapping_offset(off as isize) as *mut u8;
// Allocate the bytes in the current block if we can do so while still
// leaving enough space for a PAD tlv.
remaining = (bsize as u64).wrapping_sub(core::mem::size_of::<ext4_fc_tl>() as u64).wrapping_sub(off as u64) as i32;
if len <= remaining {
let __v1: u32 = unsafe { (*sbi).s_fc_bytes };
unsafe {
(*sbi).s_fc_bytes = __v1.wrapping_add(len as u32);
}
return dst;
}
// Else, terminate the current block with a PAD tlv, then allocate a new
// block and allocate the bytes at the start of that new block.
tl.fc_tag = 7;
tl.fc_len = remaining as u16;
unsafe { memcpy(dst as *mut (), core::ptr::addr_of_mut!(tl) as *const (), core::mem::size_of::<ext4_fc_tl>() as u64) };
unsafe { memset(dst.wrapping_offset(core::mem::size_of::<ext4_fc_tl>() as isize) as *mut (), 0i32, remaining as u64) };
unsafe {
*(crc as *mut u32) = ext4_chksum(*(crc as *mut u32), (*(*sbi).s_fc_bh).b_data as *const (), bsize as u32);
}
ext4_fc_submit_bh(sb, false);
ret = unsafe { jbd2_fc_get_buf((*EXT4_SB(sb)).s_journal, core::ptr::addr_of_mut!(bh)) };
if ret != 0 {
return core::ptr::null_mut::<u8>();
}
unsafe {
(*sbi).s_fc_bh = bh;
}
let __v4: u32 = unsafe { (*sbi).s_fc_bytes };
unsafe {
(*sbi).s_fc_bytes = __v4.wrapping_add(bsize.wrapping_sub(off).wrapping_add(len) as u32);
}
(unsafe { (*(*sbi).s_fc_bh).b_data }) as *mut u8
}

/// Complete a fast commit by writing tail tag.
///
/// Writing tail tag marks the end of a fast commit. In order to guarantee
/// atomicity, after writing tail tag, even if there's space remaining
/// in the block, next commit shouldn't use it. That's why tail tag
/// has the length as that of the remaining space on the block.
fn ext4_fc_write_tail(sb: *mut super_block, mut crc: u32) -> i32 {
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let mut tl: ext4_fc_tl = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tl) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut tail: ext4_fc_tail = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tail) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut off: i32 = 0;
let bsize: i32 = unsafe { (*(*sbi).s_journal).j_blocksize };
let mut dst: *mut u8 = unsafe { core::mem::zeroed() };
// ext4_fc_reserve_space takes care of allocating an extra block if
// there's no enough space on this block for accommodating this tail.
dst = ext4_fc_reserve_space(sb, (core::mem::size_of::<ext4_fc_tl>() as u64).wrapping_add(core::mem::size_of::<ext4_fc_tail>() as u64) as i32, unsafe { &mut (*core::ptr::addr_of_mut!(crc)) });
if dst.is_null() {
return 28i32.wrapping_neg();
}
off = ((unsafe { (*sbi).s_fc_bytes }) % (bsize as u32)) as i32;
tl.fc_tag = 8;
tl.fc_len = (bsize.wrapping_sub(off) as u64).wrapping_add(core::mem::size_of::<ext4_fc_tail>() as u64) as u16;
unsafe {
(*sbi).s_fc_bytes = round_up__3!(unsafe { (*sbi).s_fc_bytes }, bsize);
}
unsafe { memcpy(dst as *mut (), core::ptr::addr_of_mut!(tl) as *const (), core::mem::size_of::<ext4_fc_tl>() as u64) };
let __v0: *mut u8 = dst;
dst = __v0.wrapping_offset(core::mem::size_of::<ext4_fc_tl>() as isize);
tail.fc_tid = unsafe { (*(*(*sbi).s_journal).j_running_transaction).t_tid };
unsafe { memcpy(dst as *mut (), core::ptr::addr_of_mut!(tail.fc_tid) as *const (), core::mem::size_of::<u32>() as u64) };
let __v2: *mut u8 = dst;
dst = __v2.wrapping_offset(core::mem::size_of::<u32>() as isize);
crc = ext4_chksum(crc, (unsafe { (*(*sbi).s_fc_bh).b_data }) as *const (), ((unsafe { dst.offset_from((*(*sbi).s_fc_bh).b_data as *mut u8) }) as i64) as u32);
tail.fc_crc = crc;
unsafe { memcpy(dst as *mut (), core::ptr::addr_of_mut!(tail.fc_crc) as *const (), core::mem::size_of::<u32>() as u64) };
let __v4: *mut u8 = dst;
dst = __v4.wrapping_offset(core::mem::size_of::<u32>() as isize);
unsafe { memset(dst as *mut (), 0i32, bsize.wrapping_sub(off) as u64) }; // Don't leak uninitialized memory.
ext4_fc_submit_bh(sb, true);
0
}

/// Adds tag, length, value and updates CRC. Returns true if tlv was added.
/// Returns false if there's not enough space.
fn ext4_fc_add_tlv(sb: *mut super_block, tag: u16, len: u16, val: *mut u8, crc: &mut u32) -> bool {
let mut tl: ext4_fc_tl = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tl) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let dst: *mut u8 = ext4_fc_reserve_space(sb, (core::mem::size_of::<ext4_fc_tl>() as u64).wrapping_add((len as i32) as u64) as i32, crc);
if dst.is_null() {
return false;
}
tl.fc_tag = tag;
tl.fc_len = len;
unsafe { memcpy(dst as *mut (), core::ptr::addr_of_mut!(tl) as *const (), core::mem::size_of::<ext4_fc_tl>() as u64) };
unsafe { memcpy(dst.wrapping_offset(core::mem::size_of::<ext4_fc_tl>() as isize) as *mut (), val as *const (), len as u64) };
true
}

/// Same as above, but adds dentry tlv.
fn ext4_fc_add_dentry_tlv(sb: *mut super_block, crc: &mut u32, fc_dentry: *mut ext4_fc_dentry_update) -> bool {
let mut fcd: ext4_fc_dentry_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fcd) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut tl: ext4_fc_tl = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tl) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let dlen: i32 = (unsafe { (*fc_dentry).fcd_name.name.__slate_anon_0.__slate_anon_0.len }) as i32;
let mut dst: *mut u8 = ext4_fc_reserve_space(sb, (core::mem::size_of::<ext4_fc_tl>() as u64).wrapping_add(8).wrapping_add(dlen as u64) as i32, crc);
if dst.is_null() {
return false;
}
fcd.fc_parent_ino = (unsafe { (*fc_dentry).fcd_parent }) as u32;
fcd.fc_ino = (unsafe { (*fc_dentry).fcd_ino }) as u32;
tl.fc_tag = (unsafe { (*fc_dentry).fcd_op }) as u16;
tl.fc_len = 8u64.wrapping_add(dlen as u64) as u16;
unsafe { memcpy(dst as *mut (), core::ptr::addr_of_mut!(tl) as *const (), core::mem::size_of::<ext4_fc_tl>() as u64) };
let __v0: *mut u8 = dst;
dst = __v0.wrapping_offset(core::mem::size_of::<ext4_fc_tl>() as isize);
unsafe { memcpy(dst as *mut (), core::ptr::addr_of_mut!(fcd) as *const (), 8u64) };
let __v2: *mut u8 = dst;
dst = __v2.wrapping_offset(8 as isize);
unsafe { memcpy(dst as *mut (), (*fc_dentry).fcd_name.name.name as *const (), dlen as u64) };
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_range {
    list: list_head,
    tag: u16,
    lblk: u32,
    len: u32,
    pblk: u64,
    unwritten: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_inode_snap {
    data_list: list_head,
    inode_len: u32,
    inode_buf: [u8; 0],
}


/// Writes inode in the fast commit space under TLV with tag @tag.
/// Returns 0 on success, error on failure.
fn ext4_fc_write_inode(inode: *mut inode, crc: &mut u32) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut inode_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(inode_len) as *mut i32;
let mut src: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut u8 = core::ptr::addr_of_mut!(src) as *mut *mut u8;
let mut dst: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut u8 = core::ptr::addr_of_mut!(dst) as *mut *mut u8;
let mut tl: core::mem::MaybeUninit<ext4_fc_tl> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut ext4_fc_tl = core::ptr::addr_of_mut!(tl) as *mut ext4_fc_tl;
let mut fc_inode: core::mem::MaybeUninit<ext4_fc_inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut ext4_fc_inode = core::ptr::addr_of_mut!(fc_inode) as *mut ext4_fc_inode;
let mut stats: *mut ext4_fc_snap_stats = unsafe { core::mem::zeroed() };
let mut snap: *mut ext4_fc_inode_snap = unsafe { core::mem::zeroed() };
unsafe {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
snap = (*ei).i_fc_snap;
stats = core::ptr::addr_of_mut!((*EXT4_SB((*inode).i_sb)).s_fc_snap_stats);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if snap.is_null() {
atomic64_inc(core::ptr::addr_of_mut!((*stats).snap_fail_no_snap));
return 125i32.wrapping_neg();
} else {
*__slate_slot_6 = core::ptr::addr_of_mut!((*snap).inode_buf) as *mut u8;
*__slate_slot_5 = (*snap).inode_len as i32;
if (*__slate_slot_6).is_null() || *__slate_slot_5 == 0 {
atomic64_inc(core::ptr::addr_of_mut!((*stats).snap_fail_no_snap));
return 125i32.wrapping_neg();
} else {
(*__slate_slot_9).fc_ino = (*inode).i_ino as u32;
(*__slate_slot_8).fc_tag = 6;
(*__slate_slot_8).fc_len = (*__slate_slot_5 as u64).wrapping_add(core::mem::size_of::<u32>() as u64) as u16;
*__slate_slot_4 = 125i32.wrapping_neg();
*__slate_slot_7 = ext4_fc_reserve_space((*inode).i_sb, (core::mem::size_of::<ext4_fc_tl>() as u64).wrapping_add(*__slate_slot_5 as u64).wrapping_add(core::mem::size_of::<u32>() as u64) as i32, crc);
if !(*__slate_slot_7).is_null() {
memcpy(*__slate_slot_7 as *mut (), core::ptr::addr_of_mut!(*__slate_slot_8) as *const (), core::mem::size_of::<ext4_fc_tl>() as u64);
let __v3: *mut u8 = *__slate_slot_7;
*__slate_slot_7 = __v3.wrapping_offset(core::mem::size_of::<ext4_fc_tl>() as isize);
memcpy(*__slate_slot_7 as *mut (), core::ptr::addr_of_mut!(*__slate_slot_9) as *const (), 4u64);
let __v1: *mut u8 = *__slate_slot_7;
*__slate_slot_7 = __v1.wrapping_offset(4 as isize);
memcpy(*__slate_slot_7 as *mut (), *__slate_slot_6 as *const (), *__slate_slot_5 as u64);
*__slate_slot_4 = 0;
}
return *__slate_slot_4;
}
}
}
}

/// Writes updated data ranges for the inode in question. Updates CRC.
/// Returns 0 on success, error otherwise.
fn ext4_fc_write_inode_data(inode: *mut inode, crc: &mut u32) -> i32 {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let snap: *mut ext4_fc_inode_snap = unsafe { (*ei).i_fc_snap };
let stats: *mut ext4_fc_snap_stats = unsafe { core::ptr::addr_of_mut!((*EXT4_SB((*inode).i_sb)).s_fc_snap_stats) };
let mut fc_ext: ext4_fc_add_range = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fc_ext) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut lrange: ext4_fc_del_range = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(lrange) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
let mut range: *mut ext4_fc_range = unsafe { core::mem::zeroed() };
if snap.is_null() {
atomic64_inc(unsafe { core::ptr::addr_of_mut!((*stats).snap_fail_no_snap) });
return 125i32.wrapping_neg();
}
range = list_first_entry__7!(unsafe { core::ptr::addr_of_mut!((*snap).data_list) });
while list_entry_is_head!(range, unsafe { core::ptr::addr_of_mut!((*snap).data_list) }, list) == 0 {
if ((unsafe { (*range).tag }) as i32) == 2 {
lrange.fc_ino = (unsafe { (*inode).i_ino }) as u32;
lrange.fc_lblk = unsafe { (*range).lblk };
lrange.fc_len = unsafe { (*range).len };
if !ext4_fc_add_tlv(unsafe { (*inode).i_sb }, 2u16, core::mem::size_of::<ext4_fc_del_range>() as u16, core::ptr::addr_of_mut!(lrange) as *mut u8, crc) {
return 28i32.wrapping_neg();
}
} else {
fc_ext.fc_ino = (unsafe { (*inode).i_ino }) as u32;
let ex: *mut ext4_extent = core::ptr::addr_of_mut!(fc_ext.fc_ex) as *mut ext4_extent;
unsafe {
(*ex).ee_block = (*range).lblk;
}
unsafe {
(*ex).ee_len = (*range).len as u16;
}
ext4_ext_store_pblock(ex, unsafe { (*range).pblk });
if unsafe { (*range).unwritten } {
ext4_ext_mark_unwritten(ex);
} else {
ext4_ext_mark_initialized(ex);
}
if !ext4_fc_add_tlv(unsafe { (*inode).i_sb }, 1u16, core::mem::size_of::<ext4_fc_add_range>() as u16, core::ptr::addr_of_mut!(fc_ext) as *mut u8, crc) {
return 28i32.wrapping_neg();
}
}
range = list_next_entry__6!(range, list);
}
0
}

fn ext4_fc_free_ranges(head: *mut list_head) {
let mut range: *mut ext4_fc_range = unsafe { core::mem::zeroed() };
let mut range_n: *mut ext4_fc_range = unsafe { core::mem::zeroed() };
range = list_first_entry__7!(head);
range_n = list_next_entry__6!(range, list);
while list_entry_is_head!(range, head, list) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*range).list) });
unsafe { kmem_cache_free(ext4_fc_range_cachep, range as *mut ()) };
range = range_n;
range_n = list_next_entry__6!(range_n, list);
}
}

fn ext4_fc_free_inode_snap(inode: *mut inode) {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let snap: *mut ext4_fc_inode_snap = unsafe { (*ei).i_fc_snap };
if snap.is_null() {
return;
}
ext4_fc_free_ranges(unsafe { core::ptr::addr_of_mut!((*snap).data_list) });
unsafe { kfree(snap as *const ()) };
unsafe {
(*ei).i_fc_snap = core::ptr::null_mut::<ext4_fc_inode_snap>();
}
}

fn ext4_fc_snapshot_inode_data(inode: *mut inode, ranges: *mut list_head, nr_ranges_total: u32, nr_rangesp: *mut u32, snap_err: *mut i32) -> i32 {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let stats: *mut ext4_fc_snap_stats = unsafe { core::ptr::addr_of_mut!((*EXT4_SB((*inode).i_sb)).s_fc_snap_stats) };
let mut start_lblk: u32 = 0;
let mut end_lblk: u32 = 0;
let mut cur_lblk: u32 = 0;
let mut nr_ranges: u32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
if (unsafe { (*ei).i_fc_lblk_len }) == 0 {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
if !nr_rangesp.is_null() {
unsafe {
*nr_rangesp = 0;
}
}
return 0;
}
start_lblk = unsafe { (*ei).i_fc_lblk_start };
end_lblk = unsafe { (*ei).i_fc_lblk_start }.wrapping_add(unsafe { (*ei).i_fc_lblk_len }).wrapping_sub(1);
unsafe {
(*ei).i_fc_lblk_len = 0;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
cur_lblk = start_lblk;
let mut __v1: i32;
if false {
unsafe { _printk(c"snapshot data ranges %u-%u for inode %llu\n".as_ptr() as *const u8, start_lblk, end_lblk, (*inode).i_ino) };
}
__v1 = 0;
while cur_lblk <= end_lblk {
let mut es: extent_status = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(es) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut range: *mut ext4_fc_range = unsafe { core::mem::zeroed() };
let mut len: u32 = 0;
let remaining: u64 = (end_lblk as u64).wrapping_sub(cur_lblk as u64).wrapping_add(1);
if (unsafe { ext4_es_lookup_extent(inode, cur_lblk, core::ptr::null_mut::<u32>(), core::ptr::addr_of_mut!(es), core::ptr::null_mut::<u64>()) }) == 0 {
atomic64_inc(unsafe { core::ptr::addr_of_mut!((*stats).snap_fail_es_miss) });
ext4_fc_set_snap_err(snap_err, 1i32);
return 11i32.wrapping_neg();
}
if ext4_es_is_delayed(unsafe { &(*core::ptr::addr_of_mut!(es)) }) != 0 {
atomic64_inc(unsafe { core::ptr::addr_of_mut!((*stats).snap_fail_es_delayed) });
ext4_fc_set_snap_err(snap_err, 2i32);
return 11i32.wrapping_neg();
}
len = es.es_len.wrapping_sub(cur_lblk.wrapping_sub(es.es_lblk));
if (len as u64) > remaining {
len = remaining as u32;
}
if len == 0 {
cur_lblk = cur_lblk.wrapping_add(1);
} else {
if nr_ranges_total.wrapping_add(nr_ranges) >= 2048 {
atomic64_inc(unsafe { core::ptr::addr_of_mut!((*stats).snap_fail_ranges_cap) });
ext4_fc_set_snap_err(snap_err, 5i32);
return 7i32.wrapping_neg();
}
range = alloc_hooks!(unsafe { kmem_cache_alloc_noprof(ext4_fc_range_cachep, 3136u32) }) as *mut ext4_fc_range;
if range.is_null() {
atomic64_inc(unsafe { core::ptr::addr_of_mut!((*stats).snap_fail_nomem) });
ext4_fc_set_snap_err(snap_err, 6i32);
return 12i32.wrapping_neg();
}
nr_ranges = nr_ranges.wrapping_add(1);
unsafe {
(*range).lblk = cur_lblk;
}
unsafe {
(*range).len = len;
}
unsafe {
(*range).pblk = 0;
}
unsafe {
(*range).unwritten = false;
}
if ext4_es_is_hole(unsafe { &(*core::ptr::addr_of_mut!(es)) }) != 0 {
unsafe {
(*range).tag = 2;
}
} else {
let __v12: bool = ext4_es_is_written(unsafe { &(*core::ptr::addr_of_mut!(es)) }) != 0 || ext4_es_is_unwritten(unsafe { &(*core::ptr::addr_of_mut!(es)) }) != 0;
if __v12 {
let mut max: u32 = 0;
unsafe {
(*range).tag = 1;
}
unsafe {
(*range).pblk = ext4_es_pblock(&(*core::ptr::addr_of_mut!(es))).wrapping_add(cur_lblk.wrapping_sub(es.es_lblk) as u64);
}
unsafe {
(*range).unwritten = ext4_es_is_unwritten(&(*core::ptr::addr_of_mut!(es))) != 0;
}
max = (if unsafe { (*range).unwritten } { 32767u64 } else { 32768u64 }) as u32;
if (unsafe { (*range).len }) > max {
unsafe {
(*range).len = max;
}
}
} else {
unsafe { kmem_cache_free(ext4_fc_range_cachep, range as *mut ()) };
atomic64_inc(unsafe { core::ptr::addr_of_mut!((*stats).snap_fail_es_other) });
ext4_fc_set_snap_err(snap_err, 3i32);
return 11i32.wrapping_neg();
}
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*range).list) });
list_add_tail(unsafe { core::ptr::addr_of_mut!((*range).list) }, ranges);
if ((unsafe { (*range).len }) as u64) > (end_lblk as u64).wrapping_sub(cur_lblk as u64) {
break;
}
cur_lblk = cur_lblk.wrapping_add(unsafe { (*range).len });
}
}
if !nr_rangesp.is_null() {
unsafe {
*nr_rangesp = nr_ranges;
}
}
0
}

fn ext4_fc_snapshot_inode(inode: *mut inode, nr_ranges_total: u32, nr_rangesp: *mut u32, snap_err: *mut i32) -> i32 {
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: u64 = unsafe { core::mem::zeroed() };
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let stats: *mut ext4_fc_snap_stats = unsafe { core::ptr::addr_of_mut!((*EXT4_SB((*inode).i_sb)).s_fc_snap_stats) };
let mut snap: *mut ext4_fc_inode_snap = unsafe { core::mem::zeroed() };
let mut inode_len: i32 = 128;
let mut iloc: ext4_iloc = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iloc) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut ranges: list_head = unsafe { core::mem::zeroed() };
ranges = list_head { next: core::ptr::addr_of_mut!(ranges), prev: core::ptr::addr_of_mut!(ranges) };
let mut nr_ranges: u32 = 0;
let mut ret: i32 = unsafe { ext4_get_inode_loc_noio(inode, core::ptr::addr_of_mut!(iloc)) };
let mut alloc_ctx: i32 = 0;
if ret != 0 {
atomic64_inc(unsafe { core::ptr::addr_of_mut!((*stats).snap_fail_inode_loc) });
ext4_fc_set_snap_err(snap_err, 7i32);
return ret;
}
if ext4_test_inode_flag(inode, 28i32) != 0 {
inode_len = EXT4_INODE_SIZE!(unsafe { (*inode).i_sb });
} else {
if EXT4_INODE_SIZE!(unsafe { (*inode).i_sb }) > 128 {
inode_len = inode_len.wrapping_add((unsafe { (*ei).i_extra_isize }) as i32);
}
}
let mut __v3: *mut ext4_fc_inode_snap;
let __count: u64 = inode_len as u64;
let __obj_size: u64 = struct_size_t!(__count);
let mut __v4: *mut ();
let mut __v5: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3136u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3136u32, unsafe { *{
__slate_compound_8 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_8)
} });
}
__v5 = _res;
__v4 = __v5;
let __obj_ptr: *mut ext4_fc_inode_snap = __v4 as *mut ext4_fc_inode_snap;
if !__obj_ptr.is_null() {
let mut __v9: u64;
let __v10: u64 = __count;
unsafe {
*core::ptr::addr_of_mut!(*{
__slate_compound_11 = 0u64;
core::ptr::addr_of_mut!(__slate_compound_11)
}) = __v10;
}
__v9 = __v10;
}
__v3 = __obj_ptr;
snap = __v3;
if snap.is_null() {
atomic64_inc(unsafe { core::ptr::addr_of_mut!((*stats).snap_fail_nomem) });
ext4_fc_set_snap_err(snap_err, 6i32);
brelse(iloc.bh);
return 12i32.wrapping_neg();
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*snap).data_list) });
unsafe {
(*snap).inode_len = inode_len as u32;
}
unsafe { memcpy((core::ptr::addr_of_mut!((*snap).inode_buf) as *mut u8) as *mut (), (ext4_raw_inode(&(*core::ptr::addr_of_mut!(iloc))) as *mut u8) as *const (), inode_len as u64) };
brelse(iloc.bh);
ret = ext4_fc_snapshot_inode_data(inode, core::ptr::addr_of_mut!(ranges), nr_ranges_total, core::ptr::addr_of_mut!(nr_ranges), snap_err);
if ret != 0 {
unsafe { kfree(snap as *const ()) };
ext4_fc_free_ranges(core::ptr::addr_of_mut!(ranges));
return ret;
}
alloc_ctx = ext4_fc_lock(unsafe { (*inode).i_sb });
ext4_fc_free_inode_snap(inode);
unsafe {
(*ei).i_fc_snap = snap;
}
list_splice_tail_init(core::ptr::addr_of_mut!(ranges), unsafe { core::ptr::addr_of_mut!((*snap).data_list) });
ext4_fc_unlock(unsafe { (*inode).i_sb }, alloc_ctx);
atomic64_inc(unsafe { core::ptr::addr_of_mut!((*stats).snap_inodes) });
atomic64_add(nr_ranges as i64, unsafe { core::ptr::addr_of_mut!((*stats).snap_ranges) });
if !nr_rangesp.is_null() {
unsafe {
*nr_rangesp = nr_ranges;
}
}
0
}

// Flushes data of all the inodes in the commit queue.
fn ext4_fc_flush_data(journal: *mut journal_s) -> i32 {
let sb: *mut super_block = (unsafe { (*journal).j_private }) as *mut super_block;
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let mut ei: *mut ext4_inode_info = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
ei = list_first_entry!(unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize));
while list_entry_is_head!(ei, unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize), i_fc_list) == 0 {
let mut __v1: *mut jbd2_inode;
loop {
if !(__native_word__16!() || ((const { core::mem::size_of::<*mut jbd2_inode>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1549() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ei).jinode) as *const *mut jbd2_inode) };
ret = unsafe { jbd2_submit_inode_data(journal, __v1) };
if ret != 0 {
return ret;
}
ei = list_next_entry!(ei, i_fc_list);
}
ei = list_first_entry!(unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize));
while list_entry_is_head!(ei, unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize), i_fc_list) == 0 {
let mut __v6: *mut jbd2_inode;
loop {
if !(__native_word__16!() || ((const { core::mem::size_of::<*mut jbd2_inode>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1550() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ei).jinode) as *const *mut jbd2_inode) };
ret = unsafe { jbd2_wait_inode_data(journal, __v6) };
if ret != 0 {
return ret;
}
ei = list_next_entry!(ei, i_fc_list);
}
0
}

/// Commit all the directory entry updates
fn ext4_fc_commit_dentry_updates(journal: *mut journal_s, crc: &mut u32) -> i32 {
let sb: *mut super_block = (unsafe { (*journal).j_private }) as *mut super_block;
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let mut fc_dentry: *mut ext4_fc_dentry_update = unsafe { core::mem::zeroed() };
let mut fc_dentry_n: *mut ext4_fc_dentry_update = unsafe { core::mem::zeroed() };
let mut ei: *mut ext4_inode_info = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
if list_empty(unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize) as *const list_head) != 0 {
return 0;
}
fc_dentry = list_first_entry__5!(unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize));
fc_dentry_n = list_next_entry__4!(fc_dentry, fcd_list);
while list_entry_is_head!(fc_dentry, unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize), fcd_list) == 0 {
if (unsafe { (*fc_dentry).fcd_op }) != 3 {
if !ext4_fc_add_dentry_tlv(sb, crc, fc_dentry) {
return 28i32.wrapping_neg();
}
} else {
// With fcd_dilist we need not loop in sbi->s_fc_q to get the
// corresponding inode. Also, the corresponding inode could have been
// deleted, in which case, we don't need to do anything.
if list_empty((unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_dilist) }) as *const list_head) == 0 {
// For EXT4_FC_TAG_CREAT, fcd_dilist is linked on the created
// inode's i_fc_dilist list (kept singular), so we can recover the
// inode through it.
ei = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_dilist) }, ext4_inode_info);
let inode: *mut inode = unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) };
let mut __v5: i64;
let __ret_warn_on: i32 = ((unsafe { (*inode).i_ino }) != ((unsafe { (*fc_dentry).fcd_ino }) as u64)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1551: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1551b - ., 3; .popsection\n# {0}", const 1551 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_fc_commit_dentry_updates_6, sym __slate_asm_string_ext4_fc_commit_dentry_updates_7, const 1210 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1552: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1552b - ., 4; .popsection\n# {0}", const 1552 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
// We first write the inode and then the create dirent. This
// allows the recovery code to create an unnamed inode first
// and then link it to a directory entry. This allows us
// to use namei.c routines almost as is and simplifies
// the recovery code.
ret = ext4_fc_write_inode(inode, crc);
if ret != 0 {
return ret;
}
ret = ext4_fc_write_inode_data(inode, crc);
if ret != 0 {
return ret;
}
if !ext4_fc_add_dentry_tlv(sb, crc, fc_dentry) {
return 28i32.wrapping_neg();
}
}
}
fc_dentry = fc_dentry_n;
fc_dentry_n = list_next_entry__4!(fc_dentry_n, fcd_list);
}
0
}

fn ext4_fc_snapshot_inodes(journal: *mut journal_s, inodes: *mut *mut inode, inodes_size: u32, nr_inodesp: *mut u32, nr_rangesp: *mut u32, snap_err: *mut i32) -> i32 {
let mut inode_ranges: u32 = unsafe { core::mem::zeroed() };
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut ei: core::mem::MaybeUninit<*mut ext4_inode_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut ext4_inode_info = core::ptr::addr_of_mut!(ei) as *mut *mut ext4_inode_info;
let mut alloc_ctx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(alloc_ctx) as *mut i32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut nr_ranges: u32 = unsafe { core::mem::zeroed() };
let mut idx: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u32 = core::ptr::addr_of_mut!(idx) as *mut u32;
let mut i: u32 = unsafe { core::mem::zeroed() };
let mut fc_dentry: core::mem::MaybeUninit<*mut ext4_fc_dentry_update> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut ext4_fc_dentry_update = core::ptr::addr_of_mut!(fc_dentry) as *mut *mut ext4_fc_dentry_update;
let mut iter: core::mem::MaybeUninit<*mut ext4_inode_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut ext4_inode_info = core::ptr::addr_of_mut!(iter) as *mut *mut ext4_inode_info;
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
unsafe {
sb = (*journal).j_private as *mut super_block;
sbi = EXT4_SB(sb);
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
i = 0;
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
nr_ranges = 0;
ret = 0;
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
*__slate_slot_11 = ext4_fc_lock(sb);
*__slate_slot_17 = list_first_entry!(((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize));
'__join_10: {
'__join_22: {
while list_entry_is_head!(*__slate_slot_17, ((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize), i_fc_list) == 0 {
if i >= inodes_size {
break '__join_22;
} else {
let __v10: u32 = i;
i = __v10.wrapping_add(1);
*inodes.wrapping_offset(__v10 as isize) = core::ptr::addr_of_mut!((*(*__slate_slot_17)).vfs_inode);
*__slate_slot_17 = list_next_entry!(*__slate_slot_17, i_fc_list);
}
}
*__slate_slot_16 = list_first_entry__5!(((*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize));
loop {
if list_entry_is_head!(*__slate_slot_16, ((*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize), fcd_list) == 0 {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if (*(*__slate_slot_16)).fcd_op == 3 {
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_16)).fcd_dilist) as *const list_head) == 0 {
// See the comment in ext4_fc_commit_dentry_updates().
*__slate_slot_8 = list_first_entry__2!(core::ptr::addr_of_mut!((*(*__slate_slot_16)).fcd_dilist), ext4_inode_info);
*__slate_slot_7 = core::ptr::addr_of_mut!((*(*__slate_slot_8)).vfs_inode);
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_8)).i_fc_list) as *const list_head) != 0 {
if i >= inodes_size {
break;
} else {
// Create-only inodes may only be referenced via fcd_dilist and
// not appear on s_fc_q[MAIN]. They may hit the last iput while
// we are snapshotting, but inode eviction calls ext4_fc_del(),
// which waits for FC_COMMITTING to clear. Mark them FC_COMMITTING
// so the inode stays pinned and the snapshot stays valid until
// ext4_fc_cleanup().
ext4_set_inode_state(*__slate_slot_7, 10i32);
let __v6: u32 = i;
i = __v6.wrapping_add(1);
*inodes.wrapping_offset(__v6 as isize) = *__slate_slot_7;
}
}
}
}
*__slate_slot_16 = list_next_entry__4!(*__slate_slot_16, fcd_list);
} else {
break '__join_10;
}
}
atomic64_inc(core::ptr::addr_of_mut!((*sbi).s_fc_snap_stats.snap_fail_inodes_cap));
ext4_fc_set_snap_err(snap_err, 4i32);
ret = 7i32.wrapping_neg();
break '__join_10;
}
atomic64_inc(core::ptr::addr_of_mut!((*sbi).s_fc_snap_stats.snap_fail_inodes_cap));
ext4_fc_set_snap_err(snap_err, 4i32);
ret = 7i32.wrapping_neg();
}
ext4_fc_unlock(sb, *__slate_slot_11);
if ret != 0 {
return ret;
} else {
*__slate_slot_14 = 0;
while *__slate_slot_14 < i {
inode_ranges = 0;
ret = ext4_fc_snapshot_inode(*inodes.wrapping_offset(*__slate_slot_14 as isize), nr_ranges, core::ptr::addr_of_mut!(inode_ranges), snap_err);
if ret != 0 {
break;
} else {
nr_ranges = nr_ranges.wrapping_add(inode_ranges);
let __v1: u32 = *__slate_slot_14;
*__slate_slot_14 = __v1.wrapping_add(1);
}
}
if !nr_inodesp.is_null() {
*nr_inodesp = *__slate_slot_14;
}
if !nr_rangesp.is_null() {
*nr_rangesp = nr_ranges;
}
return ret;
}
}
}

fn ext4_fc_perform_commit(journal: *mut journal_s, commit_tid: u32) -> i32 {
let mut locked_ns: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u64 = core::ptr::addr_of_mut!(locked_ns) as *mut u64;
let mut lock_start: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i64 = core::ptr::addr_of_mut!(lock_start) as *mut i64;
let mut alloc_ctx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(alloc_ctx) as *mut i32;
let mut crc: u32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut plug: core::mem::MaybeUninit<blk_plug> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut blk_plug = core::ptr::addr_of_mut!(plug) as *mut blk_plug;
let mut snap_err: i32 = unsafe { core::mem::zeroed() };
let mut snap_ranges: u32 = unsafe { core::mem::zeroed() };
let mut snap_inodes: u32 = unsafe { core::mem::zeroed() };
let mut inodes_size: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(inodes_size) as *mut u32;
let mut inodes: core::mem::MaybeUninit<*mut *mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut *mut inode = core::ptr::addr_of_mut!(inodes) as *mut *mut *mut inode;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut head: core::mem::MaybeUninit<ext4_fc_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut ext4_fc_head = core::ptr::addr_of_mut!(head) as *mut ext4_fc_head;
let mut iter: core::mem::MaybeUninit<*mut ext4_inode_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut ext4_inode_info = core::ptr::addr_of_mut!(iter) as *mut *mut ext4_inode_info;
let mut snap_stats: *mut ext4_fc_snap_stats = unsafe { core::mem::zeroed() };
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
unsafe {
sb = (*journal).j_private as *mut super_block;
sbi = EXT4_SB(sb);
snap_stats = core::ptr::addr_of_mut!((*sbi).s_fc_snap_stats);
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
snap_inodes = 0;
snap_ranges = 0;
snap_err = 0;
let __slate_auto_bytes_5 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 64usize);
ret = 0;
crc = 0;
let __slate_auto_bytes_6 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Step 1: Mark all inodes on s_fc_q[MAIN] with
// EXT4_STATE_FC_FLUSHING_DATA. This prevents these inodes from being
// freed until the data flush is over.
*__slate_slot_2 = ext4_fc_lock(sb);
*__slate_slot_13 = list_first_entry!(((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize));
while list_entry_is_head!(*__slate_slot_13, ((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize), i_fc_list) == 0 {
ext4_set_inode_state(core::ptr::addr_of_mut!((*(*__slate_slot_13)).vfs_inode), 11i32);
*__slate_slot_13 = list_next_entry!(*__slate_slot_13, i_fc_list);
}
ext4_fc_unlock(sb, *__slate_slot_2);
// Step 2: Flush data for all the eligible inodes.
ret = ext4_fc_flush_data(journal);
// Step 3: Clear EXT4_STATE_FC_FLUSHING_DATA flag, before returning
// any error from step 2. This ensures that waiters waiting on
// EXT4_STATE_FC_FLUSHING_DATA can resume.
*__slate_slot_2 = ext4_fc_lock(sb);
*__slate_slot_13 = list_first_entry!(((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize));
while list_entry_is_head!(*__slate_slot_13, ((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize), i_fc_list) == 0 {
ext4_clear_inode_state(core::ptr::addr_of_mut!((*(*__slate_slot_13)).vfs_inode), 11i32);
ext4_fc_wake_inode_state(core::ptr::addr_of_mut!((*(*__slate_slot_13)).vfs_inode), 11i32);
*__slate_slot_13 = list_next_entry!(*__slate_slot_13, i_fc_list);
}
// Make sure clearing of EXT4_STATE_FC_FLUSHING_DATA is visible before
// the waiter checks the bit. Pairs with implicit barrier in
// prepare_to_wait() in ext4_fc_del().
loop {
kcsan_mb!();
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
if true {
break;
}
}
ext4_fc_unlock(sb, *__slate_slot_2);
// If we encountered error in Step 2, return it now after clearing
// EXT4_STATE_FC_FLUSHING_DATA bit.
if ret != 0 {
return ret;
} else {
ret = ext4_fc_alloc_snapshot_inodes(sb, &mut (*core::ptr::addr_of_mut!(*__slate_slot_10)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_9)));
if ret != 0 {
if ret == 7i32.wrapping_neg() {
atomic64_inc(core::ptr::addr_of_mut!((*snap_stats).snap_fail_inodes_cap));
} else {
if ret == 12i32.wrapping_neg() {
atomic64_inc(core::ptr::addr_of_mut!((*snap_stats).snap_fail_nomem));
}
}
return ret;
} else {
// Step 4: Mark all inodes as being committed.
jbd2_journal_lock_updates(journal);
*__slate_slot_1 = ktime_get();
// The journal is now locked. No more handles can start and all the
// previous handles are now drained. Snapshotting happens in this
// window so log writing can consume only stable snapshots without
// doing logical-to-physical mapping.
*__slate_slot_2 = ext4_fc_lock(sb);
*__slate_slot_13 = list_first_entry!(((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize));
while list_entry_is_head!(*__slate_slot_13, ((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize), i_fc_list) == 0 {
ext4_set_inode_state(core::ptr::addr_of_mut!((*(*__slate_slot_13)).vfs_inode), 10i32);
*__slate_slot_13 = list_next_entry!(*__slate_slot_13, i_fc_list);
}
ext4_fc_unlock(sb, *__slate_slot_2);
ret = ext4_fc_snapshot_inodes(journal, *__slate_slot_10, *__slate_slot_9, core::ptr::addr_of_mut!(snap_inodes), core::ptr::addr_of_mut!(snap_ranges), core::ptr::addr_of_mut!(snap_err));
jbd2_journal_unlock_updates(journal);
*__slate_slot_0 = ktime_to_ns(ktime_sub!(ktime_get(), *__slate_slot_1)) as u64;
atomic64_add(*__slate_slot_0 as i64, core::ptr::addr_of_mut!((*snap_stats).lock_updates_ns_total));
atomic64_inc(core::ptr::addr_of_mut!((*snap_stats).lock_updates_samples));
ext4_fc_snap_stats_update_max(core::ptr::addr_of_mut!((*snap_stats).lock_updates_ns_max), *__slate_slot_0);
if trace_ext4_fc_lock_updates_enabled() {
trace_call__ext4_fc_lock_updates(sb, commit_tid, *__slate_slot_0, snap_inodes, snap_ranges, ret, snap_err);
}
kvfree(*__slate_slot_10 as *const ());
if ret != 0 {
return ret;
} else {
// Step 5: If file system device is different from journal device,
// issue a cache flush before we start writing fast commit blocks.
if (*journal).j_fs_dev != (*journal).j_dev {
blkdev_issue_flush((*journal).j_fs_dev);
}
'__join_0: {
blk_start_plug(core::ptr::addr_of_mut!(*__slate_slot_5));
*__slate_slot_2 = ext4_fc_lock(sb);
// Step 6: Write fast commit blocks to disk.
if (*sbi).s_fc_bytes == 0 {
// Step 6.1: Add a head tag only if this is the first fast
// commit in this TID.
(*__slate_slot_12).fc_features = 0;
(*__slate_slot_12).fc_tid = (*(*(*sbi).s_journal).j_running_transaction).t_tid;
if !ext4_fc_add_tlv(sb, 9u16, core::mem::size_of::<ext4_fc_head>() as u16, core::ptr::addr_of_mut!(*__slate_slot_12) as *mut u8, &mut (*core::ptr::addr_of_mut!(crc))) {
ret = 28i32.wrapping_neg();
break '__join_0;
}
}
// Step 6.2: Now write all the dentry updates.
ret = ext4_fc_commit_dentry_updates(journal, &mut (*core::ptr::addr_of_mut!(crc)));
if ret == 0 {
// Step 6.3: Now write all the changed inodes to disk.
*__slate_slot_13 = list_first_entry!(((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize));
while list_entry_is_head!(*__slate_slot_13, ((*sbi).s_fc_q.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize), i_fc_list) == 0 {
*__slate_slot_11 = core::ptr::addr_of_mut!((*(*__slate_slot_13)).vfs_inode);
if ext4_test_inode_state(*__slate_slot_11, 10i32) != 0 {
ret = ext4_fc_write_inode_data(*__slate_slot_11, &mut (*core::ptr::addr_of_mut!(crc)));
if ret != 0 {
break '__join_0;
} else {
ret = ext4_fc_write_inode(*__slate_slot_11, &mut (*core::ptr::addr_of_mut!(crc)));
if ret != 0 {
break '__join_0;
}
}
}
*__slate_slot_13 = list_next_entry!(*__slate_slot_13, i_fc_list);
}
// Step 6.4: Finally write tail tag to conclude this fast commit.
ret = ext4_fc_write_tail(sb, crc);
}
}
ext4_fc_unlock(sb, *__slate_slot_2);
blk_finish_plug(core::ptr::addr_of_mut!(*__slate_slot_5));
return ret;
}
}
}
}
}

fn ext4_fc_count_snapshot_inodes(sb: *mut super_block) -> u32 {
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let mut iter: *mut ext4_inode_info = unsafe { core::mem::zeroed() };
let mut fc_dentry: *mut ext4_fc_dentry_update = unsafe { core::mem::zeroed() };
let mut nr_inodes: u32 = 0;
let alloc_ctx: i32 = ext4_fc_lock(sb);
iter = list_first_entry!(unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize));
while list_entry_is_head!(iter, unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize), i_fc_list) == 0 {
nr_inodes = nr_inodes.wrapping_add(1);
iter = list_next_entry!(iter, i_fc_list);
}
fc_dentry = list_first_entry__5!(unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize));
while list_entry_is_head!(fc_dentry, unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize), fcd_list) == 0 {
let mut ei: *mut ext4_inode_info = unsafe { core::mem::zeroed() };
if (unsafe { (*fc_dentry).fcd_op }) == 3 {
if list_empty((unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_dilist) }) as *const list_head) == 0 {
// See the comment in ext4_fc_commit_dentry_updates().
ei = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_dilist) }, ext4_inode_info);
if list_empty((unsafe { core::ptr::addr_of_mut!((*ei).i_fc_list) }) as *const list_head) != 0 {
nr_inodes = nr_inodes.wrapping_add(1);
}
}
}
fc_dentry = list_next_entry__4!(fc_dentry, fcd_list);
}
ext4_fc_unlock(sb, alloc_ctx);
nr_inodes
}

fn ext4_fc_alloc_snapshot_inodes(sb: *mut super_block, inodesp: &mut *mut *mut inode, nr_inodesp: &mut u32) -> i32 {
let nr_inodes: u32 = ext4_fc_count_snapshot_inodes(sb);
unsafe {
*(inodesp as *mut *mut *mut inode) = core::ptr::null_mut::<*mut inode>();
}
unsafe {
*(nr_inodesp as *mut u32) = 0;
}
if nr_inodes == 0 {
return 0;
}
if nr_inodes > 1024 {
return 7i32.wrapping_neg();
}
let mut __v0: *mut *mut inode;
let __obj_size: u64 = size_mul(core::mem::size_of::<*mut inode>() as u64, nr_inodes as u64);
__v0 = kvmalloc_node!(__obj_size, 3392u32, 1i32.wrapping_neg()) as *mut *mut inode;
let inodes: *mut *mut inode = __v0;
if inodes.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
*(inodesp as *mut *mut *mut inode) = inodes;
}
unsafe {
*(nr_inodesp as *mut u32) = nr_inodes;
}
0
}

fn ext4_fc_update_stats(sb: *mut super_block, status: i32, commit_time: u64, nblks: i32, commit_tid: u32) {
let stats: *mut ext4_fc_stats = unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_fc_stats) };
let mut __v0: i32;
__v0 = 0;
if status == 0 {
let __v2: u64 = unsafe { (*stats).fc_num_commits };
unsafe {
(*stats).fc_num_commits = __v2.wrapping_add(1);
}
let __v5: u64 = unsafe { (*stats).fc_numblks };
unsafe {
(*stats).fc_numblks = __v5.wrapping_add(nblks as u64);
}
if likely__2!(unsafe { (*stats).s_fc_avg_commit_time }) != 0 {
unsafe {
(*stats).s_fc_avg_commit_time = commit_time.wrapping_add(unsafe { (*stats).s_fc_avg_commit_time }.wrapping_mul(3)) / 4;
}
} else {
unsafe {
(*stats).s_fc_avg_commit_time = commit_time;
}
}
} else {
if status == 3 || status == 1 {
if status == 3 {
let __v8: u64 = unsafe { (*stats).fc_failed_commits };
unsafe {
(*stats).fc_failed_commits = __v8.wrapping_add(1);
}
}
let __v11: u64 = unsafe { (*stats).fc_ineligible_commits };
unsafe {
(*stats).fc_ineligible_commits = __v11.wrapping_add(1);
}
} else {
let __v14: u64 = unsafe { (*stats).fc_skipped_commits };
unsafe {
(*stats).fc_skipped_commits = __v14.wrapping_add(1);
}
}
}
trace_ext4_fc_commit_stop(sb, nblks, status, commit_tid);
}

/// The main commit entry point. Performs a fast commit for transaction
/// commit_tid if needed. If it's not possible to perform a fast commit
/// due to various reasons, we fall back to full commit. Returns 0
/// on success, error otherwise.
#[unsafe(no_mangle)]
extern "C" fn ext4_fc_commit(journal: *mut journal_s, commit_tid: u32) -> i32 {
// There was an ongoing commit, check if we need to restart
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut journal_ioprio: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(journal_ioprio) as *mut i32;
let mut old_ioprio: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(old_ioprio) as *mut i32;
let mut commit_time: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i64 = core::ptr::addr_of_mut!(commit_time) as *mut i64;
let mut start_time: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i64 = core::ptr::addr_of_mut!(start_time) as *mut i64;
let mut fc_bufs_before: i32 = unsafe { core::mem::zeroed() };
let mut status: i32 = unsafe { core::mem::zeroed() };
let mut subtid: i32 = unsafe { core::mem::zeroed() };
let mut bsize: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut nblks: i32 = unsafe { core::mem::zeroed() };
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
unsafe {
sb = (*journal).j_private as *mut super_block;
sbi = EXT4_SB(sb);
nblks = 0;
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
bsize = (*journal).j_blocksize;
subtid = atomic_read(core::ptr::addr_of_mut!((*sbi).s_fc_subtid) as *const __SlateRecord_types_h_190);
status = 0;
fc_bufs_before = 0;
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if (*EXT4_SB(sb)).s_mount_opt2 & 16 == 0 {
return jbd2_complete_transaction(journal, commit_tid);
} else {
trace_ext4_fc_commit_start(sb, commit_tid);
*__slate_slot_4 = ktime_get();
*__slate_slot_2 = get_current_ioprio();
loop {
*__slate_slot_9 = jbd2_fc_begin_commit(journal, commit_tid);
if *__slate_slot_9 == 114i32.wrapping_neg() {
if atomic_read(core::ptr::addr_of_mut!((*sbi).s_fc_subtid) as *const __SlateRecord_types_h_190) <= subtid {
__v0 = tid_gt(commit_tid, (*journal).j_commit_sequence) != 0;
} else {
__v0 = false;
}
if !__v0 {
ext4_fc_update_stats(sb, 2i32, 0u64, 0i32, commit_tid);
return 0;
}
} else {
break;
}
}
if *__slate_slot_9 != 0 {
// Commit couldn't start. Just update stats and perform a
// full commit.
ext4_fc_update_stats(sb, 3i32, 0u64, 0i32, commit_tid);
return jbd2_complete_transaction(journal, commit_tid);
} else {
// After establishing journal barrier via jbd2_fc_begin_commit(), check
// if we are fast commit ineligible.
if ext4_test_mount_flag(sb, 1i32) != 0 {
status = 1;
} else {
// Now that we know that this thread is going to do a fast commit,
// elevate the priority to match that of the journal thread.
if !unsafe { (*(*journal).j_task).io_context }.is_null() {
*__slate_slot_1 = (*(*(*(*sbi).s_journal).j_task).io_context).ioprio as i32;
} else {
*__slate_slot_1 = IOPRIO_PRIO_VALUE!(2i32, 3i32) as i32;
}
set_task_ioprio(get_current(), *__slate_slot_1);
fc_bufs_before = (unsafe { (*sbi).s_fc_bytes }.wrapping_add(bsize as u32).wrapping_sub(1) / (bsize as u32)) as i32;
*__slate_slot_9 = ext4_fc_perform_commit(journal, commit_tid);
if *__slate_slot_9 < 0 {
if *__slate_slot_9 == 11i32.wrapping_neg() || *__slate_slot_9 == 7i32.wrapping_neg() || *__slate_slot_9 == 125i32.wrapping_neg() {
status = 1;
} else {
status = 3;
}
} else {
nblks = (unsafe { (*sbi).s_fc_bytes }.wrapping_add(bsize as u32).wrapping_sub(1) / (bsize as u32)).wrapping_sub(fc_bufs_before as u32) as i32;
*__slate_slot_9 = jbd2_fc_wait_bufs(journal, nblks);
if *__slate_slot_9 < 0 {
status = 3;
} else {
atomic_inc(core::ptr::addr_of_mut!((*sbi).s_fc_subtid));
*__slate_slot_9 = jbd2_fc_end_commit(journal);
set_task_ioprio(get_current(), *__slate_slot_2);
// weight the commit time higher than the average time so we
// don't react too strongly to vast changes in the commit time
*__slate_slot_3 = ktime_to_ns(ktime_sub!(ktime_get(), *__slate_slot_4));
ext4_fc_update_stats(sb, status, *__slate_slot_3 as u64, nblks, commit_tid);
return *__slate_slot_9;
}
}
}
set_task_ioprio(get_current(), *__slate_slot_2);
*__slate_slot_9 = jbd2_fc_end_commit_fallback(journal);
ext4_fc_update_stats(sb, status, 0u64, 0i32, commit_tid);
return *__slate_slot_9;
}
}
}
}

/// Fast commit cleanup routine. This is called after every fast commit and
/// full commit. full is true if we are called after a full commit.
#[unsafe(link_section = ".text.slate_distinct.fast_commit.ext4_fc_cleanup")]
extern "C" fn ext4_fc_cleanup(journal: *mut journal_s, full: i32, tid: u32) {
let sb: *mut super_block = (unsafe { (*journal).j_private }) as *mut super_block;
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let mut ei: *mut ext4_inode_info = unsafe { core::mem::zeroed() };
let mut fc_dentry: *mut ext4_fc_dentry_update = unsafe { core::mem::zeroed() };
let mut alloc_ctx: i32 = 0;
if full != 0 && !unsafe { (*sbi).s_fc_bh }.is_null() {
unsafe {
(*sbi).s_fc_bh = core::ptr::null_mut::<buffer_head>();
}
}
trace_ext4_fc_cleanup(journal, full, tid);
unsafe { jbd2_fc_release_bufs(journal) };
alloc_ctx = ext4_fc_lock(sb);
while list_empty(unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize) as *const list_head) == 0 {
let mut requeue: bool = false;
ei = list_first_entry__9!(unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize), ext4_inode_info);
list_del_init(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_list) });
ext4_fc_free_inode_snap(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) });
spin_lock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
if full != 0 {
requeue = tid_geq(tid, unsafe { (*ei).i_sync_tid }) == 0;
} else {
requeue = ext4_test_inode_state(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) }, 13i32) != 0;
}
if !requeue {
ext4_fc_reset_inode(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) });
}
ext4_clear_inode_state(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) }, 13i32);
ext4_clear_inode_state(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) }, 10i32);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
if requeue {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_list) }, unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(1 as isize));
}
// Make sure clearing of EXT4_STATE_FC_COMMITTING is
// visible before we send the wakeup. Pairs with implicit
// barrier in prepare_to_wait() in ext4_fc_del().
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
ext4_fc_wake_inode_state(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) }, 10i32);
}
while list_empty(unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize) as *const list_head) == 0 {
fc_dentry = list_first_entry__13!(unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize), ext4_fc_dentry_update);
list_del_init(unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_list) });
let __v5: bool = (unsafe { (*fc_dentry).fcd_op }) == 3 && list_empty((unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_dilist) }) as *const list_head) == 0;
if __v5 {
// See the comment in ext4_fc_commit_dentry_updates().
ei = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_dilist) }, ext4_inode_info);
ext4_fc_free_inode_snap(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) });
spin_lock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
ext4_clear_inode_state(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) }, 13i32);
ext4_clear_inode_state(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) }, 10i32);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ei).i_fc_lock) });
// Make sure clearing of EXT4_STATE_FC_COMMITTING is
// visible before we send the wakeup. Pairs with
// implicit barrier in prepare_to_wait() in
// ext4_fc_del().
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
ext4_fc_wake_inode_state(unsafe { core::ptr::addr_of_mut!((*ei).vfs_inode) }, 10i32);
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*fc_dentry).fcd_dilist) });
unsafe { release_dentry_name_snapshot(core::ptr::addr_of_mut!((*fc_dentry).fcd_name)) };
unsafe { kmem_cache_free(ext4_fc_dentry_cachep, fc_dentry as *mut ()) };
}
list_splice_init(unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(1 as isize), unsafe { (*sbi).s_fc_dentry_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize));
list_splice_init(unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(1 as isize), unsafe { (*sbi).s_fc_q.as_mut_ptr() as *mut list_head }.wrapping_offset(0 as isize));
if tid_geq(tid, unsafe { (*sbi).s_fc_ineligible_tid }) != 0 {
unsafe {
(*sbi).s_fc_ineligible_tid = 0;
}
ext4_clear_mount_flag(sb, 1i32);
}
if full != 0 {
unsafe {
(*sbi).s_fc_bytes = 0;
}
}
ext4_fc_unlock(sb, alloc_ctx);
trace_ext4_fc_stats(sb);
}

// Ext4 Replay Path Routines
/// Helper struct for dentry replay routines
#[repr(C)]
#[derive(Clone, Copy)]
struct dentry_info_args {
    parent_ino: i32,
    dname_len: i32,
    ino: i32,
    inode_len: i32,
    dname: *mut u8,
}


/// Same as struct ext4_fc_tl, but uses native endianness fields
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_tl_mem {
    fc_tag: u16,
    fc_len: u16,
}


fn tl_to_darg(darg: &mut dentry_info_args, tl: &ext4_fc_tl_mem, val: *mut u8) {
let mut fcd: ext4_fc_dentry_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fcd) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
unsafe { memcpy(core::ptr::addr_of_mut!(fcd) as *mut (), val as *const (), 8u64) };
unsafe {
(*(darg as *mut dentry_info_args)).parent_ino = fcd.fc_parent_ino as i32;
}
unsafe {
(*(darg as *mut dentry_info_args)).ino = fcd.fc_ino as i32;
}
unsafe {
(*(darg as *mut dentry_info_args)).dname = val.wrapping_offset(offsetof__6!() as isize) as *mut u8;
}
unsafe {
(*(darg as *mut dentry_info_args)).dname_len = (((*(tl as *const ext4_fc_tl_mem)).fc_len as i32) as u64).wrapping_sub(8) as i32;
}
}

fn ext4_fc_get_tl(tl: &mut ext4_fc_tl_mem, val: *mut u8) {
let mut tl_disk: ext4_fc_tl = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tl_disk) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
unsafe { memcpy(core::ptr::addr_of_mut!(tl_disk) as *mut (), val as *const (), core::mem::size_of::<ext4_fc_tl>() as u64) };
unsafe {
(*(tl as *mut ext4_fc_tl_mem)).fc_len = tl_disk.fc_len;
}
unsafe {
(*(tl as *mut ext4_fc_tl_mem)).fc_tag = tl_disk.fc_tag;
}
}

/// Unlink replay function
fn ext4_fc_replay_unlink(sb: *mut super_block, tl: &ext4_fc_tl_mem, val: *mut u8) -> i32 {
let mut entry: qstr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut darg: dentry_info_args = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(darg) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut ret: i32 = 0;
tl_to_darg(unsafe { &mut (*core::ptr::addr_of_mut!(darg)) }, tl, val);
trace_ext4_fc_replay(sb, 5i32, darg.ino, darg.parent_ino, darg.dname_len);
entry.name = darg.dname as *const u8;
unsafe {
entry.__slate_anon_0.__slate_anon_0.len = darg.dname_len as u32;
}
let inode: *mut inode = ext4_iget__23!(sb, darg.ino, 0i32);
if IS_ERR(inode as *const ()) {
let mut __v0: i32;
__v0 = 0;
return 0;
}
let old_parent: *mut inode = ext4_iget!(sb, darg.parent_ino, 0i32);
if IS_ERR(old_parent as *const ()) {
let mut __v1: i32;
__v1 = 0;
unsafe { iput(inode) };
return 0;
}
ret = unsafe { __ext4_unlink(old_parent, core::ptr::addr_of_mut!(entry) as *const qstr, inode, core::ptr::null_mut::<dentry>()) };
// -ENOENT ok coz it might not exist anymore.
if ret == 2i32.wrapping_neg() {
ret = 0;
}
unsafe { iput(old_parent) };
unsafe { iput(inode) };
ret
}

fn ext4_fc_replay_link_internal(sb: *mut super_block, darg: *mut dentry_info_args, inode: *mut inode) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut qstr_dname: core::mem::MaybeUninit<qstr> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut qstr = core::ptr::addr_of_mut!(qstr_dname) as *mut qstr;
let mut dir: *mut inode = unsafe { core::mem::zeroed() };
unsafe {
dir = core::ptr::null_mut::<inode>();
core::ptr::write(__slate_slot_3, qstr { __slate_anon_0: {
let mut __t0: __SlateRecord_dcache_h_50 = core::mem::zeroed();
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_dcache_h_51 = core::mem::zeroed();
__t1.len = (*darg).dname_len as u32;
__t1
};
__t0
}, name: (*darg).dname as *const u8 });
ret = 0;
dir = ext4_iget__8!(sb, unsafe { (*darg).parent_ino }, 0i32);
if IS_ERR(dir as *const ()) {
dir = core::ptr::null_mut::<inode>();
} else {
ret = __ext4_link(dir, inode, core::ptr::addr_of_mut!(*__slate_slot_3) as *const qstr, core::ptr::null_mut::<dentry>());
// It's possible that link already existed since data blocks
// for the dir in question got persisted before we crashed OR
// we replayed this tag and crashed before the entire replay
// could complete.
if !(ret != 0 && ret != 17i32.wrapping_neg()) {
ret = 0;
}
}
if !dir.is_null() {
iput(dir);
}
return ret;
}
}

/// Link replay function
fn ext4_fc_replay_link(sb: *mut super_block, tl: &ext4_fc_tl_mem, val: *mut u8) -> i32 {
let mut darg: dentry_info_args = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(darg) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut ret: i32 = 0;
tl_to_darg(unsafe { &mut (*core::ptr::addr_of_mut!(darg)) }, tl, val);
trace_ext4_fc_replay(sb, 4i32, darg.ino, darg.parent_ino, darg.dname_len);
let inode: *mut inode = ext4_iget__12!(sb, darg.ino, 0i32);
if IS_ERR(inode as *const ()) {
let mut __v0: i32;
__v0 = 0;
return 0;
}
ret = ext4_fc_replay_link_internal(sb, core::ptr::addr_of_mut!(darg), inode);
unsafe { iput(inode) };
ret
}

/// Record all the modified inodes during replay. We use this later to setup
/// block bitmaps correctly.
fn ext4_fc_record_modified_inode(sb: *mut super_block, ino: i32) -> i32 {
let mut i: i32 = 0;
let state: *mut ext4_fc_replay_state = unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_fc_replay_state) };
while i < unsafe { (*state).fc_modified_inodes_used } {
if (unsafe { *unsafe { (*state).fc_modified_inodes }.wrapping_offset(i as isize) }) == ino {
return 0;
}
i = i.wrapping_add(1);
}
if (unsafe { (*state).fc_modified_inodes_used }) == unsafe { (*state).fc_modified_inodes_size } {
let fc_modified_inodes: *mut i32 = krealloc_node!(unsafe { (*state).fc_modified_inodes }, (core::mem::size_of::<i32>() as u64).wrapping_mul(unsafe { (*state).fc_modified_inodes_size }.wrapping_add(4) as u64), 3264u32, 1i32.wrapping_neg()) as *mut i32;
if fc_modified_inodes.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*state).fc_modified_inodes = fc_modified_inodes;
}
let __v9: i32 = unsafe { (*state).fc_modified_inodes_size };
unsafe {
(*state).fc_modified_inodes_size = __v9.wrapping_add(4);
}
}
let __v12: i32 = unsafe { (*state).fc_modified_inodes_used };
unsafe {
(*state).fc_modified_inodes_used = __v12.wrapping_add(1);
}
unsafe {
*unsafe { (*state).fc_modified_inodes }.wrapping_offset(__v12 as isize) = ino;
}
0
}

/// Inode replay function
fn ext4_fc_replay_inode(sb: *mut super_block, tl: &ext4_fc_tl_mem, val: *mut u8) -> i32 {
let mut off_gen: u64 = unsafe { core::mem::zeroed() };
let mut eh: core::mem::MaybeUninit<*mut ext4_extent_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut ext4_extent_header = core::ptr::addr_of_mut!(eh) as *mut *mut ext4_extent_header;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut ino: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(ino) as *mut i32;
let mut inode_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(inode_len) as *mut i32;
let mut iloc: core::mem::MaybeUninit<ext4_iloc> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut ext4_iloc = core::ptr::addr_of_mut!(iloc) as *mut ext4_iloc;
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
let mut raw_fc_inode: core::mem::MaybeUninit<*mut ext4_inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut ext4_inode = core::ptr::addr_of_mut!(raw_fc_inode) as *mut *mut ext4_inode;
let mut raw_inode: core::mem::MaybeUninit<*mut ext4_inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut ext4_inode = core::ptr::addr_of_mut!(raw_inode) as *mut *mut ext4_inode;
let mut fc_inode: core::mem::MaybeUninit<ext4_fc_inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut ext4_fc_inode = core::ptr::addr_of_mut!(fc_inode) as *mut ext4_fc_inode;
unsafe {
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
inode = core::ptr::null_mut::<inode>();
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 24usize);
let __slate_auto_bytes_4 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let tag: i32 = (*(tl as *const ext4_fc_tl_mem)).fc_tag as i32;
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
off_gen = offsetof__22!(ext4_inode);
memcpy(core::ptr::addr_of_mut!(*__slate_slot_11) as *mut (), val as *const (), 4u64);
*__slate_slot_5 = (*__slate_slot_11).fc_ino as i32;
trace_ext4_fc_replay(sb, tag, *__slate_slot_5, 0i32, 0i32);
inode = ext4_iget__9!(sb, *__slate_slot_5, 0i32);
if !IS_ERR(inode as *const ()) {
ext4_ext_clear_bb(inode);
iput(inode);
}
inode = core::ptr::null_mut::<inode>();
*__slate_slot_4 = ext4_fc_record_modified_inode(sb, *__slate_slot_5);
if *__slate_slot_4 == 0 {
*__slate_slot_9 = val.wrapping_offset(offsetof__27!() as isize) as *mut ext4_inode;
*__slate_slot_4 = ext4_get_fc_inode_loc(sb, *__slate_slot_5 as u64, core::ptr::addr_of_mut!(*__slate_slot_7));
if *__slate_slot_4 == 0 {
*__slate_slot_6 = (((*(tl as *const ext4_fc_tl_mem)).fc_len as i32) as u64).wrapping_sub(4) as i32;
*__slate_slot_10 = ext4_raw_inode(&(*core::ptr::addr_of_mut!(*__slate_slot_7)));
memcpy(*__slate_slot_10 as *mut (), *__slate_slot_9 as *const (), offsetof__8!(ext4_inode));
memcpy((*__slate_slot_10 as *mut u8).wrapping_offset(off_gen as isize) as *mut (), (*__slate_slot_9 as *mut u8).wrapping_offset(off_gen as isize) as *const (), (*__slate_slot_6 as u64).wrapping_sub(off_gen));
if (*(*__slate_slot_10)).i_flags & 524288 != 0 {
*__slate_slot_2 = ((*(*__slate_slot_10)).i_block.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) as *mut ext4_extent_header;
if ((*(*__slate_slot_2)).eh_magic as i32) != 62218 {
memset(*__slate_slot_2 as *mut (), 0i32, core::mem::size_of::<ext4_extent_header>() as u64);
(*(*__slate_slot_2)).eh_magic = 62218;
(*(*__slate_slot_2)).eh_max = ((core::mem::size_of::<[u32; 15]>() as u64).wrapping_sub(core::mem::size_of::<ext4_extent_header>() as u64) / (core::mem::size_of::<ext4_extent>() as u64)) as u16;
}
} else {
if (*(*__slate_slot_10)).i_flags & 268435456 != 0 {
memcpy(((*(*__slate_slot_10)).i_block.as_mut_ptr() as *mut u32) as *mut (), ((*(*__slate_slot_9)).i_block.as_mut_ptr() as *mut u32) as *const (), core::mem::size_of::<[u32; 15]>() as u64);
}
}
// Immediately update the inode on disk.
*__slate_slot_4 = ext4_handle_dirty_metadata__39!((*__slate_slot_7).bh);
if *__slate_slot_4 == 0 {
*__slate_slot_4 = sync_dirty_buffer((*__slate_slot_7).bh);
if *__slate_slot_4 == 0 {
*__slate_slot_4 = ext4_mark_inode_used(sb, *__slate_slot_5);
if *__slate_slot_4 == 0 {
// Given that we just wrote the inode on disk, this SHOULD succeed.
inode = ext4_iget__6!(sb, *__slate_slot_5, 0i32);
if IS_ERR(inode as *const ()) {
inode = core::ptr::null_mut::<inode>();
*__slate_slot_4 = 117i32.wrapping_neg();
} else {
// Our allocator could have made different decisions than before
// crashing. This should be fixed but until then, we calculate
// the number of blocks the inode.
if ext4_test_inode_flag(inode, 28i32) == 0 {
ext4_ext_replay_set_iblocks(inode);
}
(*inode).i_generation = (*ext4_raw_inode(&(*core::ptr::addr_of_mut!(*__slate_slot_7)))).i_generation;
ext4_reset_inode_seed(inode);
ext4_inode_csum_set(inode, ext4_raw_inode(&(*core::ptr::addr_of_mut!(*__slate_slot_7))), EXT4_I!(inode));
*__slate_slot_4 = ext4_handle_dirty_metadata__43!((*__slate_slot_7).bh);
sync_dirty_buffer((*__slate_slot_7).bh);
}
}
}
}
brelse((*__slate_slot_7).bh);
}
}
iput(inode);
if *__slate_slot_4 == 0 {
blkdev_issue_flush((*sb).s_bdev);
}
return *__slate_slot_4;
}
}

/// Dentry create replay function.
///
/// EXT4_FC_TAG_CREAT is preceded by EXT4_FC_TAG_INODE_FULL. Which means, the
/// inode for which we are trying to create a dentry here, should already have
/// been replayed before we start here.
fn ext4_fc_replay_create(sb: *mut super_block, tl: &ext4_fc_tl_mem, val: *mut u8) -> i32 {
let mut darg: core::mem::MaybeUninit<dentry_info_args> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut dentry_info_args = core::ptr::addr_of_mut!(darg) as *mut dentry_info_args;
let mut dir: *mut inode = unsafe { core::mem::zeroed() };
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
ret = 0;
inode = core::ptr::null_mut::<inode>();
dir = core::ptr::null_mut::<inode>();
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
tl_to_darg(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2)), tl, val);
trace_ext4_fc_replay(sb, 3i32, (*__slate_slot_2).ino, (*__slate_slot_2).parent_ino, (*__slate_slot_2).dname_len);
// This takes care of update group descriptor and other metadata
ret = ext4_mark_inode_used(sb, (*__slate_slot_2).ino);
if ret == 0 {
inode = ext4_iget__17!(sb, (*__slate_slot_2).ino, 0i32);
if IS_ERR(inode as *const ()) {
inode = core::ptr::null_mut::<inode>();
ret = 22i32.wrapping_neg();
} else {
if S_ISDIR!(unsafe { (*inode).i_mode }) {
// If we are creating a directory, we need to make sure that the
// dot and dot dot dirents are setup properly.
dir = ext4_iget__2!(sb, (*__slate_slot_2).parent_ino, 0i32);
if IS_ERR(dir as *const ()) {
break '__join_0;
} else {
ret = ext4_init_new_dir(core::ptr::null_mut::<jbd2_journal_handle>(), dir, inode);
iput(dir);
if ret != 0 {
ret = 0;
break '__join_0;
}
}
}
ret = ext4_fc_replay_link_internal(sb, core::ptr::addr_of_mut!(*__slate_slot_2), inode);
if ret == 0 {
set_nlink(inode, 1u32);
ext4_mark_inode_dirty__61!(inode);
}
}
}
}
iput(inode);
return ret;
}
}

/// Record physical disk regions which are in use as per fast commit area,
/// and used by inodes during replay phase. Our simple replay phase
/// allocator excludes these regions from allocation.
#[unsafe(no_mangle)]
extern "C" fn ext4_fc_record_regions(sb: *mut super_block, ino: i32, lblk: u32, pblk: u64, len: i32, replay: i32) -> i32 {
let state: *mut ext4_fc_replay_state = unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_fc_replay_state) };
// during replay phase, the fc_regions_valid may not same as
// fc_regions_used, update it when do new additions.
if replay != 0 && (unsafe { (*state).fc_regions_used }) != unsafe { (*state).fc_regions_valid } {
unsafe {
(*state).fc_regions_used = (*state).fc_regions_valid;
}
}
if (unsafe { (*state).fc_regions_used }) == unsafe { (*state).fc_regions_size } {
let fc_regions: *mut ext4_fc_alloc_region = krealloc_node!(unsafe { (*state).fc_regions }, (core::mem::size_of::<ext4_fc_alloc_region>() as u64).wrapping_mul(unsafe { (*state).fc_regions_size }.wrapping_add(4) as u64), 3264u32, 1i32.wrapping_neg()) as *mut ext4_fc_alloc_region;
if fc_regions.is_null() {
return 12i32.wrapping_neg();
}
let __v6: i32 = unsafe { (*state).fc_regions_size };
unsafe {
(*state).fc_regions_size = __v6.wrapping_add(4);
}
unsafe {
(*state).fc_regions = fc_regions;
}
}
let __v9: i32 = unsafe { (*state).fc_regions_used };
unsafe {
(*state).fc_regions_used = __v9.wrapping_add(1);
}
let region: *mut ext4_fc_alloc_region = unsafe { (*state).fc_regions }.wrapping_offset(__v9 as isize);
unsafe {
(*region).ino = ino;
}
unsafe {
(*region).lblk = lblk;
}
unsafe {
(*region).pblk = pblk;
}
unsafe {
(*region).len = len;
}
if replay != 0 {
let __v12: i32 = unsafe { (*state).fc_regions_valid };
unsafe {
(*state).fc_regions_valid = __v12.wrapping_add(1);
}
}
0
}

/// Replay add range tag
fn ext4_fc_replay_add_range(sb: *mut super_block, val: *mut u8) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut path: *mut ext4_ext_path = unsafe { core::mem::zeroed() };
let mut map: core::mem::MaybeUninit<ext4_map_blocks> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut ext4_map_blocks = core::ptr::addr_of_mut!(map) as *mut ext4_map_blocks;
let mut start_pblk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(start_pblk) as *mut u64;
let mut len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(len) as *mut i32;
let mut remaining: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(remaining) as *mut i32;
let mut cur: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(cur) as *mut u32;
let mut start: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u32 = core::ptr::addr_of_mut!(start) as *mut u32;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut newex: core::mem::MaybeUninit<ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut ext4_extent = core::ptr::addr_of_mut!(newex) as *mut ext4_extent;
let mut fc_add_ex: core::mem::MaybeUninit<ext4_fc_add_range> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut ext4_fc_add_range = core::ptr::addr_of_mut!(fc_add_ex) as *mut ext4_fc_add_range;
unsafe {
let __slate_auto_bytes = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 12usize);
let __slate_auto_bytes_2 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 32usize);
path = core::ptr::null_mut::<ext4_ext_path>();
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
memcpy(core::ptr::addr_of_mut!(*__slate_slot_18) as *mut (), val as *const (), core::mem::size_of::<ext4_fc_add_range>() as u64);
*__slate_slot_16 = core::ptr::addr_of_mut!((*__slate_slot_18).fc_ex) as *mut ext4_extent;
trace_ext4_fc_replay(sb, 1i32, (*__slate_slot_18).fc_ino as i32, (*(*__slate_slot_16)).ee_block as i32, ext4_ext_get_actual_len(*__slate_slot_16));
*__slate_slot_15 = ext4_iget__15!(sb, (*__slate_slot_18).fc_ino, 0i32);
if IS_ERR(*__slate_slot_15 as *const ()) {
return 0;
} else {
'__join_0: {
*__slate_slot_7 = ext4_fc_record_modified_inode(sb, (*(*__slate_slot_15)).i_ino as i32);
if *__slate_slot_7 == 0 {
*__slate_slot_14 = (*(*__slate_slot_16)).ee_block;
*__slate_slot_10 = ext4_ext_pblock(*__slate_slot_16);
*__slate_slot_11 = ext4_ext_get_actual_len(*__slate_slot_16);
*__slate_slot_13 = *__slate_slot_14;
*__slate_slot_12 = *__slate_slot_11;
'__join_13: {
'__join_9: {
while *__slate_slot_12 > 0 {
(*__slate_slot_9).m_lblk = *__slate_slot_13;
(*__slate_slot_9).m_len = *__slate_slot_12 as u32;
(*__slate_slot_9).m_pblk = 0;
*__slate_slot_7 = ext4_map_blocks(core::ptr::null_mut::<jbd2_journal_handle>(), *__slate_slot_15, core::ptr::addr_of_mut!(*__slate_slot_9), 0i32);
if *__slate_slot_7 < 0 {
break '__join_0;
} else {
if *__slate_slot_7 == 0 {
// Range is not mapped
path = ext4_find_extent(*__slate_slot_15, *__slate_slot_13, path, 0i32);
if IS_ERR(path as *const ()) {
break '__join_13;
} else {
memset(core::ptr::addr_of_mut!(*__slate_slot_17) as *mut (), 0i32, core::mem::size_of::<ext4_extent>() as u64);
(*__slate_slot_17).ee_block = *__slate_slot_13;
ext4_ext_store_pblock(core::ptr::addr_of_mut!(*__slate_slot_17), (*__slate_slot_10).wrapping_add(*__slate_slot_13 as u64).wrapping_sub(*__slate_slot_14 as u64));
(*__slate_slot_17).ee_len = (*__slate_slot_9).m_len as u16;
if ext4_ext_is_unwritten(*__slate_slot_16) != 0 {
ext4_ext_mark_unwritten(core::ptr::addr_of_mut!(*__slate_slot_17));
}
down_write(core::ptr::addr_of_mut!((*EXT4_I!(*__slate_slot_15)).i_data_sem));
path = ext4_ext_insert_extent(core::ptr::null_mut::<jbd2_journal_handle>(), *__slate_slot_15, path, core::ptr::addr_of_mut!(*__slate_slot_17), 0i32);
up_write(core::ptr::addr_of_mut!((*EXT4_I!(*__slate_slot_15)).i_data_sem));
if IS_ERR(path as *const ()) {
break '__join_9;
}
}
} else {
if (*__slate_slot_10).wrapping_add(*__slate_slot_13 as u64).wrapping_sub(*__slate_slot_14 as u64) != (*__slate_slot_9).m_pblk {
// Logical to physical mapping changed. This can happen
// if this range was removed and then reallocated to
// map to new physical blocks during a fast commit.
*__slate_slot_7 = ext4_ext_replay_update_ex(*__slate_slot_15, *__slate_slot_13, (*__slate_slot_9).m_len as i32, ext4_ext_is_unwritten(*__slate_slot_16), (*__slate_slot_10).wrapping_add(*__slate_slot_13 as u64).wrapping_sub(*__slate_slot_14 as u64));
if *__slate_slot_7 != 0 {
break '__join_0;
} else {
// Mark the old blocks as free since they aren't used
// anymore. We maintain an array of all the modified
// inodes. In case these blocks are still used at either
// a different logical range in the same inode or in
// some different inode, we will mark them as allocated
// at the end of the FC replay using our array of
// modified inodes.
ext4_mb_mark_bb((*(*__slate_slot_15)).i_sb, (*__slate_slot_9).m_pblk, (*__slate_slot_9).m_len as i32, false);
}
} else {
// Range is mapped and needs a state change
*__slate_slot_7 = ext4_ext_replay_update_ex(*__slate_slot_15, *__slate_slot_13, (*__slate_slot_9).m_len as i32, ext4_ext_is_unwritten(*__slate_slot_16), (*__slate_slot_9).m_pblk);
if *__slate_slot_7 != 0 {
break '__join_0;
} else {
// We may have split the extent tree while toggling the state.
// Try to shrink the extent tree now.
ext4_ext_replay_shrink_inode(*__slate_slot_15, (*__slate_slot_14).wrapping_add(*__slate_slot_11 as u32));
}
}
}
let __v3: u32 = *__slate_slot_13;
*__slate_slot_13 = __v3.wrapping_add((*__slate_slot_9).m_len);
*__slate_slot_12 = (*__slate_slot_12 as u32).wrapping_sub((*__slate_slot_9).m_len) as i32;
}
}
ext4_ext_replay_shrink_inode(*__slate_slot_15, (i_size_read(*__slate_slot_15 as *const inode) >> ((*sb).s_blocksize_bits as i32)) as u32);
*__slate_slot_7 = 0;
break '__join_0;
}
*__slate_slot_7 = PTR_ERR(path as *const ()) as i32;
path = core::ptr::null_mut::<ext4_ext_path>();
break '__join_0;
}
*__slate_slot_7 = PTR_ERR(path as *const ()) as i32;
path = core::ptr::null_mut::<ext4_ext_path>();
}
}
ext4_free_ext_path(path);
iput(*__slate_slot_15);
return *__slate_slot_7;
}
}
}

/// Replay DEL_RANGE tag
fn ext4_fc_replay_del_range(sb: *mut super_block, val: *mut u8) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut remaining: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(remaining) as *mut u32;
let mut cur: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(cur) as *mut u32;
let mut map: core::mem::MaybeUninit<ext4_map_blocks> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut ext4_map_blocks = core::ptr::addr_of_mut!(map) as *mut ext4_map_blocks;
let mut lrange: core::mem::MaybeUninit<ext4_fc_del_range> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut ext4_fc_del_range = core::ptr::addr_of_mut!(lrange) as *mut ext4_fc_del_range;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
unsafe {
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 12usize);
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
memcpy(core::ptr::addr_of_mut!(*__slate_slot_14) as *mut (), val as *const (), core::mem::size_of::<ext4_fc_del_range>() as u64);
*__slate_slot_12 = (*__slate_slot_14).fc_lblk;
*__slate_slot_11 = (*__slate_slot_14).fc_len;
trace_ext4_fc_replay(sb, 2i32, (*__slate_slot_14).fc_ino as i32, *__slate_slot_12 as i32, *__slate_slot_11 as i32);
*__slate_slot_15 = ext4_iget__10!(sb, (*__slate_slot_14).fc_ino, 0i32);
if IS_ERR(*__slate_slot_15 as *const ()) {
return 0;
} else {
'__join_0: {
*__slate_slot_10 = ext4_fc_record_modified_inode(sb, (*(*__slate_slot_15)).i_ino as i32);
if *__slate_slot_10 == 0 {
while *__slate_slot_11 > 0 {
(*__slate_slot_13).m_lblk = *__slate_slot_12;
(*__slate_slot_13).m_len = *__slate_slot_11;
*__slate_slot_10 = ext4_map_blocks(core::ptr::null_mut::<jbd2_journal_handle>(), *__slate_slot_15, core::ptr::addr_of_mut!(*__slate_slot_13), 0i32);
if *__slate_slot_10 < 0 {
break '__join_0;
} else {
if *__slate_slot_10 > 0 {
let __v3: u32 = *__slate_slot_11;
*__slate_slot_11 = __v3.wrapping_sub(*__slate_slot_10 as u32);
let __v1: u32 = *__slate_slot_12;
*__slate_slot_12 = __v1.wrapping_add(*__slate_slot_10 as u32);
ext4_mb_mark_bb((*(*__slate_slot_15)).i_sb, (*__slate_slot_13).m_pblk, (*__slate_slot_13).m_len as i32, false);
} else {
let __v7: u32 = *__slate_slot_11;
*__slate_slot_11 = __v7.wrapping_sub((*__slate_slot_13).m_len);
let __v5: u32 = *__slate_slot_12;
*__slate_slot_12 = __v5.wrapping_add((*__slate_slot_13).m_len);
}
}
}
down_write(core::ptr::addr_of_mut!((*EXT4_I!(*__slate_slot_15)).i_data_sem));
*__slate_slot_10 = ext4_ext_remove_space(*__slate_slot_15, (*__slate_slot_14).fc_lblk, (*__slate_slot_14).fc_lblk.wrapping_add((*__slate_slot_14).fc_len).wrapping_sub(1));
up_write(core::ptr::addr_of_mut!((*EXT4_I!(*__slate_slot_15)).i_data_sem));
if *__slate_slot_10 == 0 {
ext4_ext_replay_shrink_inode(*__slate_slot_15, (i_size_read(*__slate_slot_15 as *const inode) >> ((*sb).s_blocksize_bits as i32)) as u32);
ext4_mark_inode_dirty__74!(*__slate_slot_15);
*__slate_slot_10 = 0;
}
}
}
iput(*__slate_slot_15);
return *__slate_slot_10;
}
}
}

fn ext4_fc_set_bitmaps_and_counters(sb: *mut super_block) {
let mut path: *mut ext4_ext_path = core::ptr::null_mut::<ext4_ext_path>();
let mut map: ext4_map_blocks = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(map) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut i: i32 = 0;
let mut ret: i32 = 0;
let mut j: i32 = 0;
let mut cur: u32 = 0;
let mut end: u32 = 0;
let state: *mut ext4_fc_replay_state = unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_fc_replay_state) };
while i < unsafe { (*state).fc_modified_inodes_used } {
let inode: *mut inode = ext4_iget__7!(sb, unsafe { *unsafe { (*state).fc_modified_inodes }.wrapping_offset(i as isize) }, 0i32);
if IS_ERR(inode as *const ()) {
let mut __v0: i32;
if false {
unsafe { _printk(c"Inode %d not found.".as_ptr() as *const u8, *unsafe { (*state).fc_modified_inodes }.wrapping_offset(i as isize)) };
}
__v0 = 0;
} else {
cur = 0;
end = 4294967295;
if ext4_test_inode_flag(inode, 28i32) != 0 {
unsafe { iput(inode) };
} else {
while cur < end {
map.m_lblk = cur;
map.m_len = end.wrapping_sub(cur);
ret = unsafe { ext4_map_blocks(core::ptr::null_mut::<jbd2_journal_handle>(), inode, core::ptr::addr_of_mut!(map), 0i32) };
if ret < 0 {
break;
}
if ret > 0 {
path = unsafe { ext4_find_extent(inode, map.m_lblk, path, 0i32) };
if !IS_ERR(path as *const ()) {
j = 0;
while j < ((unsafe { (*path).p_depth }) as i32) {
unsafe { ext4_mb_mark_bb((*inode).i_sb, (*path.wrapping_offset(j as isize)).p_block, 1i32, true) };
j = j.wrapping_add(1);
}
} else {
path = core::ptr::null_mut::<ext4_ext_path>();
}
cur = cur.wrapping_add(ret as u32);
unsafe { ext4_mb_mark_bb((*inode).i_sb, map.m_pblk, map.m_len as i32, true) };
} else {
cur = cur.wrapping_add(if map.m_len != 0 { map.m_len } else { 1 });
}
}
unsafe { iput(inode) };
}
}
i = i.wrapping_add(1);
}
unsafe { ext4_free_ext_path(path) };
}

/// Check if block is in excluded regions for block allocation. The simple
/// allocator that runs during replay phase is calls this function to see
/// if it is okay to use a block.
#[unsafe(no_mangle)]
extern "C" fn ext4_fc_replay_check_excluded(sb: *mut super_block, blk: u64) -> bool {
let mut i: i32 = 0;
let state: *mut ext4_fc_replay_state = unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_fc_replay_state) };
while i < unsafe { (*state).fc_regions_valid } {
if !((unsafe { (*unsafe { (*state).fc_regions }.wrapping_offset(i as isize)).ino }) == 0 || (unsafe { (*unsafe { (*state).fc_regions }.wrapping_offset(i as isize)).len }) == 0) {
if in_range__4!(blk, unsafe { (*unsafe { (*state).fc_regions }.wrapping_offset(i as isize)).pblk }, unsafe { (*unsafe { (*state).fc_regions }.wrapping_offset(i as isize)).len }) != 0 {
return true;
}
}
i = i.wrapping_add(1);
}
false
}

/// Cleanup function called after replay
#[unsafe(no_mangle)]
extern "C" fn ext4_fc_replay_cleanup(sb: *mut super_block) {
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let __v2: u16 = (((unsafe { (*sbi).s_mount_state }) as i32) & !32) as u16;
unsafe {
(*sbi).s_mount_state = __v2;
}
unsafe { kfree((*sbi).s_fc_replay_state.fc_regions as *const ()) };
unsafe { kfree((*sbi).s_fc_replay_state.fc_modified_inodes as *const ()) };
}

fn ext4_fc_value_len_isvalid(sbi: *mut ext4_sb_info, tag: i32, mut len: i32) -> bool {
match tag {
1 => {
return (len as u64) == (core::mem::size_of::<ext4_fc_add_range>() as u64);
}
2 => {
return (len as u64) == (core::mem::size_of::<ext4_fc_del_range>() as u64);
}
3..=5 => {
len = (len as u64).wrapping_sub(8) as i32;
return len >= 1 && len <= 255;
}
6 => {
len = (len as u64).wrapping_sub(4) as i32;
return len >= 128 && len <= unsafe { (*sbi).s_inode_size };
}
7 => {
return true; // padding can have any length
}
8 => {
return (len as u64) >= (core::mem::size_of::<ext4_fc_tail>() as u64);
}
9 => {
return (len as u64) == (core::mem::size_of::<ext4_fc_head>() as u64);
}
_ => {
}
}
false
}

/// Recovery Scan phase handler
///
/// This function is called during the scan phase and is responsible
/// for doing following things:
/// - Make sure the fast commit area has valid tags for replay
/// - Count number of tags that need to be replayed by the replay handler
/// - Verify CRC
/// - Create a list of excluded blocks for allocation during replay phase
///
/// This function returns JBD2_FC_REPLAY_CONTINUE to indicate that SCAN is
/// incomplete and JBD2 should send more blocks. It returns JBD2_FC_REPLAY_STOP
/// to indicate that scan has finished and JBD2 can now start replay phase.
/// It returns a negative error to indicate that there was an error. At the end
/// of a successful scan phase, sbi->s_fc_replay_state.fc_replay_num_tags is set
/// to indicate the number of tags that need to replayed during the replay phase.
fn ext4_fc_replay_scan(journal: *mut journal_s, bh: *mut buffer_head, off: i32, expected_tid: u32) -> i32 {
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut ex: core::mem::MaybeUninit<*mut ext4_extent> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut ext4_extent = core::ptr::addr_of_mut!(ex) as *mut *mut ext4_extent;
let mut head: core::mem::MaybeUninit<ext4_fc_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut ext4_fc_head = core::ptr::addr_of_mut!(head) as *mut ext4_fc_head;
let mut val: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut u8 = core::ptr::addr_of_mut!(val) as *mut *mut u8;
let mut cur: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut u8 = core::ptr::addr_of_mut!(cur) as *mut *mut u8;
let mut end: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut u8 = core::ptr::addr_of_mut!(end) as *mut *mut u8;
let mut start: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut u8 = core::ptr::addr_of_mut!(start) as *mut *mut u8;
let mut tail: core::mem::MaybeUninit<ext4_fc_tail> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut ext4_fc_tail = core::ptr::addr_of_mut!(tail) as *mut ext4_fc_tail;
let mut tl: core::mem::MaybeUninit<ext4_fc_tl_mem> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut ext4_fc_tl_mem = core::ptr::addr_of_mut!(tl) as *mut ext4_fc_tl_mem;
let mut ext: core::mem::MaybeUninit<ext4_fc_add_range> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut ext4_fc_add_range = core::ptr::addr_of_mut!(ext) as *mut ext4_fc_add_range;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut state: core::mem::MaybeUninit<*mut ext4_fc_replay_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut ext4_fc_replay_state = core::ptr::addr_of_mut!(state) as *mut *mut ext4_fc_replay_state;
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
unsafe {
sb = (*journal).j_private as *mut super_block;
sbi = EXT4_SB(sb);
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 1;
let __slate_auto_bytes_1 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_2 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_24 = core::ptr::addr_of_mut!((*sbi).s_fc_replay_state);
*__slate_slot_19 = (*bh).b_data as *mut u8;
*__slate_slot_18 = (*__slate_slot_19).wrapping_offset((*journal).j_blocksize as isize);
if (*(*__slate_slot_24)).fc_replay_expected_off == 0 {
(*(*__slate_slot_24)).fc_cur_tag = 0;
(*(*__slate_slot_24)).fc_replay_num_tags = 0;
(*(*__slate_slot_24)).fc_crc = 0;
(*(*__slate_slot_24)).fc_regions = core::ptr::null_mut::<ext4_fc_alloc_region>();
(*(*__slate_slot_24)).fc_regions_size = 0;
(*(*__slate_slot_24)).fc_regions_used = 0;
(*(*__slate_slot_24)).fc_regions_valid = 0;
// Check if we can stop early
if ((*(*__slate_slot_19 as *mut ext4_fc_tl)).fc_tag as i32) != 9 {
return 0;
}
}
'__join_0: {
if off != (*(*__slate_slot_24)).fc_replay_expected_off {
ret = 117i32.wrapping_neg();
} else {
let __v13: *mut ext4_fc_replay_state = *__slate_slot_24;
let __v12: i32 = (*__v13).fc_replay_expected_off;
(*__v13).fc_replay_expected_off = __v12.wrapping_add(1);
*__slate_slot_17 = *__slate_slot_19;
loop {
if *__slate_slot_17 <= (*__slate_slot_18).wrapping_offset(-(core::mem::size_of::<ext4_fc_tl>() as isize)) {
ext4_fc_get_tl(&mut (*core::ptr::addr_of_mut!(*__slate_slot_21)), *__slate_slot_17);
*__slate_slot_16 = (*__slate_slot_17).wrapping_offset(core::mem::size_of::<ext4_fc_tl>() as isize);
if (((*__slate_slot_21).fc_len as i32) as i64) > ((*__slate_slot_18).offset_from(*__slate_slot_16 as *mut u8) as i64) {
__v10 = true;
} else {
__v10 = !ext4_fc_value_len_isvalid(sbi, (*__slate_slot_21).fc_tag as i32, (*__slate_slot_21).fc_len as i32);
}
if __v10 {
break;
} else {
'__join_3: {
let __t0: i32 = (*__slate_slot_21).fc_tag as i32;
if __t0 == 1 {
memcpy(core::ptr::addr_of_mut!(*__slate_slot_22) as *mut (), *__slate_slot_16 as *const (), core::mem::size_of::<ext4_fc_add_range>() as u64);
*__slate_slot_14 = core::ptr::addr_of_mut!((*__slate_slot_22).fc_ex) as *mut ext4_extent;
ret = ext4_fc_record_regions(sb, (*__slate_slot_22).fc_ino as i32, (*(*__slate_slot_14)).ee_block, ext4_ext_pblock(*__slate_slot_14), ext4_ext_get_actual_len(*__slate_slot_14), 0i32);
if ret < 0 {
break '__join_3;
} else {
ret = 1;
}
} else {
if __t0 != 2 {
if __t0 != 4 {
if __t0 != 5 {
if __t0 != 3 {
if __t0 != 6 {
if __t0 != 7 {
if __t0 == 8 {
let __v5: *mut ext4_fc_replay_state = *__slate_slot_24;
let __v4: i32 = (*__v5).fc_cur_tag;
(*__v5).fc_cur_tag = __v4.wrapping_add(1);
memcpy(core::ptr::addr_of_mut!(*__slate_slot_20) as *mut (), *__slate_slot_16 as *const (), core::mem::size_of::<ext4_fc_tail>() as u64);
(*(*__slate_slot_24)).fc_crc = ext4_chksum((*(*__slate_slot_24)).fc_crc as u32, *__slate_slot_17 as *const (), (core::mem::size_of::<ext4_fc_tl>() as u64).wrapping_add(offsetof__43!(ext4_fc_tail)) as u32) as i32;
if (*__slate_slot_20).fc_tid == expected_tid && (*__slate_slot_20).fc_crc == ((*(*__slate_slot_24)).fc_crc as u32) {
(*(*__slate_slot_24)).fc_replay_num_tags = (*(*__slate_slot_24)).fc_cur_tag;
(*(*__slate_slot_24)).fc_regions_valid = (*(*__slate_slot_24)).fc_regions_used;
} else {
ret = if (*(*__slate_slot_24)).fc_replay_num_tags != 0 { 0 } else { 74i32.wrapping_neg() };
}
(*(*__slate_slot_24)).fc_crc = 0;
break '__join_3;
} else {
if __t0 == 9 {
memcpy(core::ptr::addr_of_mut!(*__slate_slot_15) as *mut (), *__slate_slot_16 as *const (), core::mem::size_of::<ext4_fc_head>() as u64);
if (*__slate_slot_15).fc_features & 4294967295u32 != 0 {
ret = 95i32.wrapping_neg();
break '__join_3;
} else {
if (*__slate_slot_15).fc_tid != expected_tid {
ret = 0;
break '__join_3;
} else {
let __v2: *mut ext4_fc_replay_state = *__slate_slot_24;
let __v1: i32 = (*__v2).fc_cur_tag;
(*__v2).fc_cur_tag = __v1.wrapping_add(1);
(*(*__slate_slot_24)).fc_crc = ext4_chksum((*(*__slate_slot_24)).fc_crc as u32, *__slate_slot_17 as *const (), (core::mem::size_of::<ext4_fc_tl>() as u64).wrapping_add(((*__slate_slot_21).fc_len as i32) as u64) as u32) as i32;
break '__join_3;
}
}
} else {
ret = if (*(*__slate_slot_24)).fc_replay_num_tags != 0 { 0 } else { 125i32.wrapping_neg() };
break '__join_3;
}
}
}
}
}
}
}
}
}
let __v8: *mut ext4_fc_replay_state = *__slate_slot_24;
let __v7: i32 = (*__v8).fc_cur_tag;
(*__v8).fc_cur_tag = __v7.wrapping_add(1);
(*(*__slate_slot_24)).fc_crc = ext4_chksum((*(*__slate_slot_24)).fc_crc as u32, *__slate_slot_17 as *const (), (core::mem::size_of::<ext4_fc_tl>() as u64).wrapping_add(((*__slate_slot_21).fc_len as i32) as u64) as u32) as i32;
}
if ret < 0 || ret == 0 {
break '__join_0;
} else {
*__slate_slot_17 = (*__slate_slot_17).wrapping_offset(core::mem::size_of::<ext4_fc_tl>() as isize).wrapping_offset(((*__slate_slot_21).fc_len as i32) as isize);
}
}
} else {
break '__join_0;
}
}
ret = if (*(*__slate_slot_24)).fc_replay_num_tags != 0 { 0 } else { 125i32.wrapping_neg() };
}
}
trace_ext4_fc_replay_scan(sb, ret, off);
return ret;
}
}

/// Main recovery path entry point.
/// The meaning of return codes is similar as above.
#[unsafe(link_section = ".text.slate_distinct.fast_commit.ext4_fc_replay")]
extern "C" fn ext4_fc_replay(journal: *mut journal_s, bh: *mut buffer_head, pass: u32, off: i32, expected_tid: u32) -> i32 {
let sb: *mut super_block = (unsafe { (*journal).j_private }) as *mut super_block;
let sbi: *mut ext4_sb_info = EXT4_SB(sb);
let mut tl: ext4_fc_tl_mem = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tl) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut cur: *mut u8 = unsafe { core::mem::zeroed() };
let mut ret: i32 = 1;
let state: *mut ext4_fc_replay_state = unsafe { core::ptr::addr_of_mut!((*sbi).s_fc_replay_state) };
let mut tail: ext4_fc_tail = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tail) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
if (pass as u32) == 0 {
unsafe {
(*state).fc_current_pass = 0;
}
return ext4_fc_replay_scan(journal, bh, off, expected_tid);
}
if ((unsafe { (*state).fc_current_pass }) as u32) != (pass as u32) {
unsafe {
(*state).fc_current_pass = (pass as u32) as i32;
}
unsafe {
(*sbi).s_mount_state = (((*sbi).s_mount_state as i32) | 32) as u16;
}
}
if (unsafe { (*sbi).s_fc_replay_state.fc_replay_num_tags }) == 0 {
let mut __v3: i32;
__v3 = 0;
ext4_fc_set_bitmaps_and_counters(sb);
return 0;
}
let start: *mut u8 = (unsafe { (*bh).b_data }) as *mut u8;
let end: *mut u8 = start.wrapping_offset((unsafe { (*journal).j_blocksize }) as isize);
cur = start;
while cur <= end.wrapping_offset(-(core::mem::size_of::<ext4_fc_tl>() as isize)) {
ext4_fc_get_tl(unsafe { &mut (*core::ptr::addr_of_mut!(tl)) }, cur);
let val: *mut u8 = cur.wrapping_offset(core::mem::size_of::<ext4_fc_tl>() as isize);
if (unsafe { (*state).fc_replay_num_tags }) == 0 {
ret = 0;
ext4_fc_set_bitmaps_and_counters(sb);
break;
}
let mut __v5: i32;
__v5 = 0;
let __v7: i32 = unsafe { (*state).fc_replay_num_tags };
unsafe {
(*state).fc_replay_num_tags = __v7.wrapping_sub(1);
}
match tl.fc_tag as i32 {
4 => {
ret = ext4_fc_replay_link(sb, unsafe { &(*core::ptr::addr_of_mut!(tl)) }, val);
}
5 => {
ret = ext4_fc_replay_unlink(sb, unsafe { &(*core::ptr::addr_of_mut!(tl)) }, val);
}
1 => {
ret = ext4_fc_replay_add_range(sb, val);
}
3 => {
ret = ext4_fc_replay_create(sb, unsafe { &(*core::ptr::addr_of_mut!(tl)) }, val);
}
2 => {
ret = ext4_fc_replay_del_range(sb, val);
}
6 => {
ret = ext4_fc_replay_inode(sb, unsafe { &(*core::ptr::addr_of_mut!(tl)) }, val);
}
7 => {
trace_ext4_fc_replay(sb, 7i32, 0i32, tl.fc_len as i32, 0i32);
}
8 => {
trace_ext4_fc_replay(sb, 8i32, 0i32, tl.fc_len as i32, 0i32);
unsafe { memcpy(core::ptr::addr_of_mut!(tail) as *mut (), val as *const (), core::mem::size_of::<ext4_fc_tail>() as u64) };
let mut __v10: i64;
let __ret_warn_on: i32 = (tail.fc_tid != expected_tid) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1553: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1553b - ., 3; .popsection\n# {0}", const 1553 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_fc_replay_11, sym __slate_asm_string_ext4_fc_replay_12, const 2646 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1554: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1554b - ., 4; .popsection\n# {0}", const 1554 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v10 = (__ret_warn_on != 0) as i64;
}
9 => {
}
_ => {
trace_ext4_fc_replay(sb, tl.fc_tag as i32, 0i32, tl.fc_len as i32, 0i32);
ret = 125i32.wrapping_neg();
}
}
if ret < 0 {
break;
}
ret = 1;
cur = cur.wrapping_offset(core::mem::size_of::<ext4_fc_tl>() as isize).wrapping_offset((tl.fc_len as i32) as isize);
}
ret
}

#[unsafe(no_mangle)]
extern "C" fn ext4_fc_init(sb: *mut super_block, journal: *mut journal_s) {
// We set replay callback even if fast commit disabled because we may
// could still have fast commit blocks that need to be replayed even if
// fast commit has now been turned off.
unsafe {
(*journal).j_fc_replay_callback = Some(ext4_fc_replay);
}
if (unsafe { (*EXT4_SB(sb)).s_mount_opt2 }) & 16 == 0 {
return;
}
unsafe {
(*journal).j_fc_cleanup_callback = Some(ext4_fc_cleanup);
}
}

static fc_ineligible_reasons: __SlateAlign16<[*const u8; 13]> = __SlateAlign16([(b"Extended attributes changed\0".as_ptr() as *mut u8) as *const u8, (b"Cross rename\0".as_ptr() as *mut u8) as *const u8, (b"Journal flag changed\0".as_ptr() as *mut u8) as *const u8, (b"Insufficient memory\0".as_ptr() as *mut u8) as *const u8, (b"Swap boot\0".as_ptr() as *mut u8) as *const u8, (b"Resize\0".as_ptr() as *mut u8) as *const u8, (b"Dir renamed\0".as_ptr() as *mut u8) as *const u8, (b"Falloc range op\0".as_ptr() as *mut u8) as *const u8, (b"Data journalling\0".as_ptr() as *mut u8) as *const u8, (b"Encrypted filename\0".as_ptr() as *mut u8) as *const u8, (b"Inode format migration\0".as_ptr() as *mut u8) as *const u8, (b"fs-verity enable\0".as_ptr() as *mut u8) as *const u8, (b"Move extents\0".as_ptr() as *mut u8) as *const u8]);

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.fast_commit.ext4_fc_info_show")]
extern "C" fn ext4_fc_info_show(seq: *mut seq_file, v: *mut ()) -> i32 {
let sbi: *mut ext4_sb_info = EXT4_SB((unsafe { (*seq).private }) as *mut super_block);
let stats: *mut ext4_fc_stats = unsafe { core::ptr::addr_of_mut!((*sbi).s_fc_stats) };
let snap_stats: *mut ext4_fc_snap_stats = unsafe { core::ptr::addr_of_mut!((*sbi).s_fc_snap_stats) };
let mut lock_avg_ns: u64 = 0;
let mut lock_updates_samples: u64 = 0;
let mut lock_updates_ns_total: u64 = 0;
let mut lock_updates_ns_max: u64 = 0;
if v != (1i32 as *mut ()) {
return 0;
}
lock_updates_samples = atomic64_read((unsafe { core::ptr::addr_of_mut!((*snap_stats).lock_updates_samples) }) as *const __SlateRecord_types_h_197) as u64;
lock_updates_ns_total = atomic64_read((unsafe { core::ptr::addr_of_mut!((*snap_stats).lock_updates_ns_total) }) as *const __SlateRecord_types_h_197) as u64;
lock_updates_ns_max = atomic64_read((unsafe { core::ptr::addr_of_mut!((*snap_stats).lock_updates_ns_max) }) as *const __SlateRecord_types_h_197) as u64;
if lock_updates_samples != 0 {
lock_avg_ns = div64_u64(lock_updates_ns_total, lock_updates_samples);
}
unsafe { seq_printf(seq, c"fc stats:\n%ld commits\n%ld ineligible\n%ld numblks\n%lluus avg_commit_time\n".as_ptr() as *const u8, (*stats).fc_num_commits, (*stats).fc_ineligible_commits, (*stats).fc_numblks, div_u64((*stats).s_fc_avg_commit_time, 1000u32)) };
seq_puts(seq, c"Ineligible reasons:\n".as_ptr() as *const u8);
for i in 0i32..13i32 {
unsafe { seq_printf(seq, c"\"%s\":\t%d\n".as_ptr() as *const u8, fc_ineligible_reasons.0[i as usize], *((*stats).fc_ineligible_reason_count.as_mut_ptr() as *mut u32).wrapping_offset(i as isize)) };
}
unsafe { seq_printf(seq, c"Snapshot stats:\n%llu inodes\n%llu ranges\n%lluus lock_updates_avg\n%lluus lock_updates_max\n".as_ptr() as *const u8, atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_inodes) as *const __SlateRecord_types_h_197), atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_ranges) as *const __SlateRecord_types_h_197), div_u64(lock_avg_ns, 1000u32), div_u64(lock_updates_ns_max, 1000u32)) };
unsafe { seq_printf(seq, c"Snapshot failures:\n%llu es_miss\n%llu es_delayed\n%llu es_other\n%llu inodes_cap\n%llu ranges_cap\n%llu nomem\n%llu inode_loc\n%llu no_snap\n".as_ptr() as *const u8, atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_fail_es_miss) as *const __SlateRecord_types_h_197), atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_fail_es_delayed) as *const __SlateRecord_types_h_197), atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_fail_es_other) as *const __SlateRecord_types_h_197), atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_fail_inodes_cap) as *const __SlateRecord_types_h_197), atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_fail_ranges_cap) as *const __SlateRecord_types_h_197), atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_fail_nomem) as *const __SlateRecord_types_h_197), atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_fail_inode_loc) as *const __SlateRecord_types_h_197), atomic64_read(core::ptr::addr_of_mut!((*snap_stats).snap_fail_no_snap) as *const __SlateRecord_types_h_197)) };
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn ext4_fc_init_dentry_cache() -> i32 {
let mut __slate_compound_0: kmem_cache_args = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: kmem_cache_args = unsafe { core::mem::zeroed() };
unsafe {
ext4_fc_dentry_cachep = __kmem_cache_create_args(c"ext4_fc_dentry_update".as_ptr() as *const u8, core::mem::size_of::<ext4_fc_dentry_update>() as u32, core::ptr::addr_of_mut!(*{
__slate_compound_0 = {
let mut __t0: kmem_cache_args = core::mem::zeroed();
__t0.align = core::mem::align_of::<ext4_fc_dentry_update>() as u32;
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
}), __SLAB_FLAG_BIT!(14i32));
}
if unsafe { ext4_fc_dentry_cachep }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
ext4_fc_range_cachep = __kmem_cache_create_args(c"ext4_fc_range".as_ptr() as *const u8, core::mem::size_of::<ext4_fc_range>() as u32, core::ptr::addr_of_mut!(*{
__slate_compound_1 = {
let mut __t1: kmem_cache_args = core::mem::zeroed();
__t1.align = core::mem::align_of::<ext4_fc_range>() as u32;
__t1
};
core::ptr::addr_of_mut!(__slate_compound_1)
}), __SLAB_FLAG_BIT!(14i32));
}
if unsafe { ext4_fc_range_cachep }.is_null() {
unsafe { kmem_cache_destroy(ext4_fc_dentry_cachep) };
return 12i32.wrapping_neg();
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ext4_fc_destroy_dentry_cache() {
unsafe { kmem_cache_destroy(ext4_fc_range_cachep) };
unsafe { kmem_cache_destroy(ext4_fc_dentry_cachep) };
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string___track_dentry_update_7: [u8; 1] = *b"\0";

static __slate_asm_string___track_dentry_update_8: [u8; 51] = *b"/home/takashi/c-corpus/linux/fs/ext4/fast_commit.c\0";

static __slate_asm_string_ext4_ext_mark_unwritten_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_ext_mark_unwritten_1: [u8; 52] = *b"/home/takashi/c-corpus/linux/fs/ext4/ext4_extents.h\0";

static __slate_asm_string_ext4_fc_commit_dentry_updates_6: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_fc_commit_dentry_updates_7: [u8; 51] = *b"/home/takashi/c-corpus/linux/fs/ext4/fast_commit.c\0";

static __slate_asm_string_ext4_fc_del_10: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_fc_del_11: [u8; 51] = *b"/home/takashi/c-corpus/linux/fs/ext4/fast_commit.c\0";

static __slate_asm_string_ext4_fc_del_5: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_fc_del_6: [u8; 51] = *b"/home/takashi/c-corpus/linux/fs/ext4/fast_commit.c\0";

static __slate_asm_string_ext4_fc_mark_ineligible_3: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_fc_mark_ineligible_4: [u8; 51] = *b"/home/takashi/c-corpus/linux/fs/ext4/fast_commit.c\0";

static __slate_asm_string_ext4_fc_replay_11: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_fc_replay_12: [u8; 51] = *b"/home/takashi/c-corpus/linux/fs/ext4/fast_commit.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
