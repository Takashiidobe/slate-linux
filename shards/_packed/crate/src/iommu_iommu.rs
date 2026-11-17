
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
//! Copyright (C) 2007-2008 Advanced Micro Devices, Inc.
//! Author: Joerg Roedel <jroedel@suse.de>
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
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! # chars in a path name including nul
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
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
//! This counts to 15. Any more, it will return 16th argument.
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Defines x86 CPU feature bits
//! Compile time versions of __arch_hweightN()
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
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
//! upper_32_bits - return bits 32-63 of a number
//! @n: the number we're accessing
//!
//! A basic shift-right of a 64- or 32-bit quantity.  Use this to suppress
//! the "right shift count >= width of type" warning when that quantity is
//! 32-bits.
//! restart by calling sys_restart_syscall
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! signal pending
//! If !LOCKDEP, the value is meaningless
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
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
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! Convenience macros for the sake of set_current_state:
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
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Permission denied
// Bad address
// Device or resource busy
// File exists
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
// pr_info - Print an info-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_INFO loglevel. It uses pr_fmt() to
// generate the format string.
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
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
// lower_32_bits - return bits 0-31 of a number
// @n: the number we're accessing
// Driver requests probe retry
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// No static inline type checking - see Subtlety (1) above.
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
// N 32-bit words worth of info
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// dev_WARN*() acts like dev_printk(), but with the key difference of
// using WARN/WARN_ONCE to include file/line information and a backtrace.
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
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
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__13 {
() => {
((core::mem::size_of::<[*const bus_type; 2]>() as u64) / (core::mem::size_of::<*const bus_type>() as u64)).wrapping_add(__must_be_array__21!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__23 {
() => {
((core::mem::size_of::<[*const bus_type; 2]>() as u64) / (core::mem::size_of::<*const bus_type>() as u64)).wrapping_add(__must_be_array__35!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__37 {
() => {
((core::mem::size_of::<[*const bus_type; 2]>() as u64) / (core::mem::size_of::<*const bus_type>() as u64)).wrapping_add(__must_be_array__23!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__41 {
() => {
((core::mem::size_of::<[*const bus_type; 2]>() as u64) / (core::mem::size_of::<*const bus_type>() as u64)).wrapping_add(__must_be_array__22!() as u64)
};
}

macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
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

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! GENMASK_INPUT_CHECK__22 {
() => {
core::mem::size_of::<__SlateRecord_iommu_c_2569>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__27 {
() => {
core::mem::size_of::<__SlateRecord_iommu_c_2585>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__50 {
() => {
core::mem::size_of::<__SlateRecord_iommu_c_2573>() as i32
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
macro_rules! GENMASK_TYPE__15 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__22!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
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
macro_rules! GENMASK_TYPE__52 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__50!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
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
macro_rules! GENMASK_TYPE__54 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__27!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK__7 {
($h:expr, $l:expr) => {
GENMASK_TYPE__15!(u64, $h, $l)
};
}

macro_rules! GENMASK__44 {
($h:expr, $l:expr) => {
GENMASK_TYPE__54!(u64, $h, $l)
};
}

macro_rules! GENMASK__46 {
($h:expr, $l:expr) => {
GENMASK_TYPE__52!(u64, $h, $l)
};
}

macro_rules! IS_ALIGNED {
($x:expr, $a:expr) => {
($x) & (($a) as u64).wrapping_sub(1) == 0
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

macro_rules! PCI_SLOT__2 {
($devfn:expr) => {
($devfn) >> 3i32 & 31
};
}

macro_rules! PFN_PHYS {
($x:expr) => {
(($x) as u64) << 12i32
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
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

macro_rules! __WRITE_ONCE__5 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut bool, $val) };
if true {
break;
}
}
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

macro_rules! __cmp_once__2 {
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

/// __ffs - find first set bit in word
/// @word: The word to search
///
/// Undefined if no bit exists, so code should check against 0 first.
macro_rules! __ffs {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ($word.trailing_zeros() as i32) as u64 } else { variable__ffs($word) }
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

macro_rules! __must_be_array__21 {
() => {
core::mem::size_of::<__SlateRecord_iommu_c_264>() as i32
};
}

macro_rules! __must_be_array__22 {
() => {
core::mem::size_of::<__SlateRecord_iommu_c_258>() as i32
};
}

macro_rules! __must_be_array__23 {
() => {
core::mem::size_of::<__SlateRecord_iommu_c_306>() as i32
};
}

macro_rules! __must_be_array__24 {
() => {
core::mem::size_of::<__SlateRecord_iommu_c_3090_4>() as i32
};
}

macro_rules! __must_be_array__35 {
() => {
core::mem::size_of::<__SlateRecord_iommu_c_318>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__11 {
() => {
((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
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
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! __ratelimit__15 {
($state:expr) => {
unsafe { ___ratelimit($state, c"iommu_get_default_domain_type".as_ptr() as *const u8) }
};
}

macro_rules! __ratelimit__20 {
($state:expr) => {
unsafe { ___ratelimit($state, c"iommu_get_def_domain_type".as_ptr() as *const u8) }
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
macro_rules! __verify_pcpu_ptr__3 {
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__96 {
($ptr:expr) => {
{
let mut __v: *mut group_device;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__62!() as isize)) }) as *mut ()) as *mut group_device;
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
macro_rules! container_of__99 {
($ptr:expr) => {
{
let mut __v: *mut iommu_group;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__125!() as isize)) }) as *mut ()) as *mut iommu_group;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__15!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__104 {
($ptr:expr) => {
{
let mut __v: *const iommu_device;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__104!() as isize)) }) as *mut ()) as *const iommu_device;
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
let mut __v: *mut iommu_resv_region;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__30!() as isize)) }) as *mut ()) as *mut iommu_resv_region;
__v
}
};
}

macro_rules! dev_fwnode {
($dev:expr) => {
unsafe { __dev_fwnode($dev) }
};
}

macro_rules! dev_is_pci {
($d:expr) => {
(unsafe { (*$d).bus }) == unsafe { core::ptr::addr_of!(pci_bus_type) }
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
size_mul($count, (core::mem::size_of::<u32>() as u64).wrapping_add(__must_be_array__24!() as u64))
};
}

/// Macro to get the container device of a MC device
macro_rules! fsl_mc_cont_dev {
($_dev:expr) => {
if fsl_mc_is_cont_dev!($_dev) != 0 { $_dev } else { unsafe { (*$_dev).parent } }
};
}

/// Macro to check if a device is a container device
macro_rules! fsl_mc_is_cont_dev {
($_dev:expr) => {
((unsafe { (*to_fsl_mc_device!($_dev)).flags }) as i32) & 1
};
}

macro_rules! fwnode_has_op {
($fwnode:expr, $op:ident) => {
!IS_ERR_OR_NULL(($fwnode) as *const ()) && !unsafe { (*$fwnode).ops }.is_null() && (unsafe { (*(*$fwnode).ops).$op }) != None
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

macro_rules! likely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__4 {
($ptr:expr) => {
container_of__34!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr, $type:ty) => {
container_of__4!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__33 {
($ptr:expr) => {
container_of__96!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__52 {
($ptr:expr) => {
container_of__99!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__60 {
($ptr:expr) => {
container_of__104!($ptr)
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

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head__5 {
($pos:expr, $head:expr, $member:ident) => {
list_is_head(unsafe { core::ptr::addr_of!((*$pos).$member) }, ($head) as *const list_head)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__44 {
($ptr:expr) => {
list_entry__52!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__49 {
($ptr:expr) => {
list_entry__60!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__3 {
($ptr:expr) => {
list_entry__33!(unsafe { (*$ptr).next })
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
list_entry__4!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__64 {
($ptr:expr, $type:ty) => {
list_entry!(unsafe { (*$ptr).next }, $type)
};
}

/// list_last_entry - get the last element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_last_entry__3 {
($ptr:expr) => {
list_entry__4!(unsafe { (*$ptr).prev })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__51 {
($pos:expr, $member:ident) => {
list_entry__60!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__38 {
($pos:expr, $member:ident) => {
list_entry__52!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__6 {
($pos:expr, $member:ident) => {
list_entry__4!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__4 {
($pos:expr, $member:ident) => {
list_entry__33!(unsafe { (*$pos).$member.next })
};
}

/// list_prev_entry - get the prev element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_prev_entry__3 {
($pos:expr, $member:ident) => {
list_entry__4!(unsafe { (*$pos).$member.prev })
};
}

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

macro_rules! lockdep_assert_not_held {
($l:expr) => {
loop {
$l;
if true {
break;
}
}
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

macro_rules! might_sleep_if {
($cond:expr) => {
loop {
if $cond {
might_sleep!();
}
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
macro_rules! min_t__3 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__2!($type, $x, $y)
};
}

macro_rules! offsetof__3 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__30 {
() => {
core::mem::offset_of!(iommu_resv_region, list) as u64
};
}

macro_rules! offsetof__62 {
() => {
core::mem::offset_of!(group_device, list) as u64
};
}

macro_rules! offsetof__4 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, kobj) as u64
};
}

macro_rules! offsetof__104 {
() => {
core::mem::offset_of!(iommu_device, list) as u64
};
}

macro_rules! offsetof__23 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, attr) as u64
};
}

macro_rules! offsetof__125 {
() => {
core::mem::offset_of!(iommu_group, entry) as u64
};
}

macro_rules! offsetof__6 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, list) as u64
};
}

macro_rules! offsetof__15 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, domain) as u64
};
}

macro_rules! page_to_phys {
($page:expr) => {
PFN_PHYS!(((unsafe { $page.offset_from(vmemmap_base as *mut page) }) as i64) as u64)
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

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size__8 {
($count:expr) => {
size_add(16u64, flex_array_size!($count))
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
macro_rules! struct_size_t__8 {
($count:expr) => {
struct_size__8!($count)
};
}

macro_rules! test_bit__4 {
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

macro_rules! to_fsl_mc_device {
($_dev:expr) => {
{
let mut __v: *mut fsl_mc_device;
__v = ((unsafe { ((($_dev) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!(fsl_mc_device) as isize)) }) as *mut ()) as *mut fsl_mc_device;
__v
}
};
}

macro_rules! to_iommu_group {
($_kobj:expr) => {
{
let mut __v: *mut iommu_group;
__v = ((unsafe { ((($_kobj) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!(iommu_group) as isize)) }) as *mut ()) as *mut iommu_group;
__v
}
};
}

macro_rules! to_iommu_group_attr {
($_attr:expr) => {
{
let mut __v: *mut iommu_group_attribute;
__v = ((unsafe { ((($_attr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__23!(iommu_group_attribute) as isize)) }) as *mut ()) as *mut iommu_group_attribute;
__v
}
};
}

macro_rules! to_pci_dev {
($n:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!(pci_dev) as isize)) }) as *mut ()) as *mut pci_dev;
__v
}
};
}

macro_rules! type_max {
() => {
__type_max!()
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

macro_rules! xa_lock {
($xa:expr) => {
spin_lock(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

macro_rules! xa_unlock {
($xa:expr) => {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __cpu_online_mask: cpumask;
    static mut vmemmap_base: u64;
    static mut cpuinfo_op: seq_operations;
    static mut cpu_number: i32;
    static mut kernel_kobj: *mut kobject;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut kstat: kernel_stat;
    static mut pci_bus_type: bus_type;
    static mut platform_bus_type: bus_type;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_add_device_to_group: static_call_key;
    static mut __tracepoint_add_device_to_group: tracepoint;
    static mut __SCK__tp_func_remove_device_from_group: static_call_key;
    static mut __tracepoint_remove_device_from_group: tracepoint;
    static mut __SCK__tp_func_attach_device_to_domain: static_call_key;
    static mut __tracepoint_attach_device_to_domain: tracepoint;
    static mut __SCK__tp_func_map: static_call_key;
    static mut __tracepoint_map: tracepoint;
    static mut __SCK__tp_func_unmap: static_call_key;
    static mut __tracepoint_unmap: tracepoint;
    static mut __SCK__tp_func_io_page_fault: static_call_key;
    static mut __tracepoint_io_page_fault: tracepoint;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn kstrtobool(arg0: *const u8, arg1: *mut bool) -> i32;
    fn kasprintf(arg0: u32, arg1: *const u8, ...) -> *mut u8;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __compiletime_assert_7();
    fn __compiletime_assert_8();
    fn __compiletime_assert_11();
    fn __compiletime_assert_12();
    fn ____wrong_branch_error() -> bool;
    fn kstrdup(arg0: *const u8, arg1: u32) -> *mut u8;
    fn sysfs_streq(arg0: *const u8, arg1: *const u8) -> bool;
    fn irq_to_desc(arg0: u32) -> *mut irq_desc;
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn rcu_is_watching() -> bool;
    fn __compiletime_assert_123();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn __mmdrop(arg0: *mut mm_struct);
    fn xa_load(arg0: *mut xarray, arg1: u64) -> *mut ();
    fn xa_store(arg0: *mut xarray, arg1: u64, arg2: *mut (), arg3: u32) -> *mut ();
    fn xa_erase(arg0: *mut xarray, arg1: u64) -> *mut ();
    fn xa_find(arg0: *mut xarray, arg1: *mut u64, arg2: u64, arg3: u32) -> *mut ();
    fn xa_find_after(arg0: *mut xarray, arg1: *mut u64, arg2: u64, arg3: u32) -> *mut ();
    fn __xa_cmpxchg(arg0: *mut xarray, arg1: u64, arg2: *mut (), arg3: *mut (), arg4: u32) -> *mut ();
    fn __xa_insert(arg0: *mut xarray, arg1: u64, arg2: *mut (), arg3: u32) -> i32;
    fn ida_alloc_range(arg0: *mut ida, arg1: u32, arg2: u32, arg3: u32) -> i32;
    fn ida_free(arg0: *mut ida, arg1: u32);
    fn sysfs_create_file_ns(arg0: *mut kobject, arg1: *const attribute, arg2: *const ns_common) -> i32;
    fn sysfs_remove_file_ns(arg0: *mut kobject, arg1: *const attribute, arg2: *const ns_common);
    fn sysfs_create_link(arg0: *mut kobject, arg1: *mut kobject, arg2: *const u8) -> i32;
    fn sysfs_create_link_nowarn(arg0: *mut kobject, arg1: *mut kobject, arg2: *const u8) -> i32;
    fn sysfs_remove_link(arg0: *mut kobject, arg1: *const u8);
    fn sysfs_emit(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn sysfs_emit_at(arg0: *mut u8, arg1: i32, arg2: *const u8, ...) -> i32;
    fn kobject_init_and_add(arg0: *mut kobject, arg1: *const kobj_type, arg2: *mut kobject, arg3: *const u8, ...) -> i32;
    fn kobject_create_and_add(arg0: *const u8, arg1: *mut kobject) -> *mut kobject;
    fn kobject_get(arg0: *mut kobject) -> *mut kobject;
    fn kobject_put(arg0: *mut kobject);
    fn kset_create_and_add(arg0: *const u8, arg1: *const kset_uevent_ops, arg2: *mut kobject) -> *mut kset;
    fn bus_for_each_dev(arg0: *const bus_type, arg1: *mut device, arg2: *mut (), arg3: Option<unsafe extern "C" fn(*mut device, *mut ()) -> i32>) -> i32;
    fn bus_register_notifier(arg0: *const bus_type, arg1: *mut notifier_block) -> i32;
    fn capable(arg0: i32) -> bool;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn krealloc_node_align_noprof(arg0: *const (), arg1: u64, arg2: u64, arg3: u32, arg4: i32) -> *mut ();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_415();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn is_module_address(arg0: u64) -> bool;
    fn try_module_get(arg0: *mut module) -> bool;
    fn module_put(arg0: *mut module);
    fn driver_deferred_probe_check_state(arg0: *mut device) -> i32;
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn __dev_fwnode_const(arg0: *const device) -> *const fwnode_handle;
    fn __dev_fwnode(arg0: *mut device) -> *mut fwnode_handle;
    fn device_property_read_u32_array(arg0: *const device, arg1: *const u8, arg2: *mut u32, arg3: u64) -> i32;
    fn fwnode_handle_get(arg0: *mut fwnode_handle) -> *mut fwnode_handle;
    fn iommu_device_link(arg0: *mut iommu_device, arg1: *mut device) -> i32;
    fn iommu_device_unlink(arg0: *mut iommu_device, arg1: *mut device);
    fn pci_dev_put(arg0: *mut pci_dev);
    fn pci_get_device(arg0: u32, arg1: u32, arg2: *mut pci_dev) -> *mut pci_dev;
    fn pci_acs_enabled(arg0: *mut pci_dev, arg1: u16) -> bool;
    fn pci_acs_path_enabled(arg0: *mut pci_dev, arg1: *mut pci_dev, arg2: u16) -> bool;
    fn pci_devs_are_dma_aliases(arg0: *mut pci_dev, arg1: *mut pci_dev) -> bool;
    fn pci_for_each_dma_alias(arg0: *mut pci_dev, arg1: Option<unsafe extern "C" fn(*mut pci_dev, u16, *mut ()) -> i32>, arg2: *mut ()) -> i32;
    fn pci_ats_supported(arg0: *mut pci_dev) -> bool;
    fn pci_max_pasids(arg0: *mut pci_dev) -> i32;
    fn msi_device_has_isolated_msi(arg0: *mut device) -> bool;
    fn __traceiter_add_device_to_group(arg0: *mut (), arg1: i32, arg2: *mut device) -> i32;
    fn __SCT__tp_func_add_device_to_group(arg0: *mut (), arg1: i32, arg2: *mut device) -> i32;
    fn __compiletime_assert_652();
    fn __traceiter_remove_device_from_group(arg0: *mut (), arg1: i32, arg2: *mut device) -> i32;
    fn __SCT__tp_func_remove_device_from_group(arg0: *mut (), arg1: i32, arg2: *mut device) -> i32;
    fn __compiletime_assert_659();
    fn __traceiter_attach_device_to_domain(arg0: *mut (), arg1: *mut device) -> i32;
    fn __SCT__tp_func_attach_device_to_domain(arg0: *mut (), arg1: *mut device) -> i32;
    fn __compiletime_assert_666();
    fn __traceiter_map(arg0: *mut (), arg1: u64, arg2: u64, arg3: u64) -> i32;
    fn __SCT__tp_func_map(arg0: *mut (), arg1: u64, arg2: u64, arg3: u64) -> i32;
    fn __compiletime_assert_673();
    fn __traceiter_unmap(arg0: *mut (), arg1: u64, arg2: u64, arg3: u64) -> i32;
    fn __SCT__tp_func_unmap(arg0: *mut (), arg1: u64, arg2: u64, arg3: u64) -> i32;
    fn __compiletime_assert_680();
    fn __traceiter_io_page_fault(arg0: *mut (), arg1: *mut device, arg2: u64, arg3: i32) -> i32;
    fn __SCT__tp_func_io_page_fault(arg0: *mut (), arg1: *mut device, arg2: u64, arg3: i32) -> i32;
    fn __compiletime_assert_687();
    fn iommu_setup_dma_ops(arg0: *mut device, arg1: *mut iommu_domain);
    fn iommu_get_dma_cookie(arg0: *mut iommu_domain) -> i32;
    fn iommu_put_dma_cookie(arg0: *mut iommu_domain);
    fn iommu_put_msi_cookie(arg0: *mut iommu_domain);
    fn iommu_dma_init_fq(arg0: *mut iommu_domain) -> i32;
    fn __compiletime_assert_696();
    fn __compiletime_assert_711();
    fn __compiletime_assert_790();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
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
#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn IS_ERR_OR_NULL(ptr: *const ()) -> bool {
unlikely!(ptr.is_null()) != 0 || IS_ERR_VALUE!(ptr as u64) != 0
}

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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// NOTE: @fn is as per module_param, not __setup!
// I.e., @fn returns 0 for no error or non-zero for error
// (possibly @fn returns a -errno value, but it does not matter).
// Emits warning if @fn returns non-zero.
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
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
struct static_key_mod {
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
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
fn __fls(mut word: u64) -> u64 {
if unsafe { __slate_intrinsic_f217c4720f04f6db(word) } {
return 63i32.wrapping_sub(word.leading_zeros() as i32) as u64;
}
unsafe {
let __t0: u64;
core::arch::asm!("bsr {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
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
core::arch::asm!("2: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 2b - ., 3; .popsection\n# {0}", const 2 as i32, options(nostack, att_syntax));
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
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 4; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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


#[inline(always)]
fn test_and_set_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit(nr, addr)
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_4: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_5() };
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

// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_6() };
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
unsafe { __compiletime_assert_7() };
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
unsafe { __compiletime_assert_8() };
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
unsafe { __compiletime_assert_11() };
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
struct perf_event {
}


fn list_del_init(entry: *mut list_head) {
__list_del_entry(entry);
INIT_LIST_HEAD(entry);
}

fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_12() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

fn __list_splice(list: &list_head, prev: *mut list_head, next: *mut list_head) {
let first: *mut list_head = unsafe { (*(list as *const list_head)).next };
let last: *mut list_head = unsafe { (*(list as *const list_head)).prev };
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

fn list_splice(list: &list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list, head, unsafe { (*head).next });
}
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
fn list_count_nodes(head: *mut list_head) -> u64 {
let mut pos: *mut list_head = unsafe { core::mem::zeroed() };
let mut count: u64 = 0;
pos = unsafe { (*head).next };
while list_is_head(pos as *const list_head, head as *const list_head) == 0 {
count = count.wrapping_add(1);
pos = unsafe { (*pos).next };
}
count
}

// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry_reverse - iterate backwards over list of given type.
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
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
fn arch_atomic_dec_and_test(mut v: *mut __SlateRecord_types_h_190) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock decl ({0:r})\nsete {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), lateout(reg_byte) __t0, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
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
fn atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_and_test(v)
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_add_device_to_group, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_remove_device_from_group, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_attach_device_to_domain, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_map, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_unmap, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_io_page_fault, const 0 as u8, label {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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
test_bit__4!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
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
struct llist_head {
    first: *mut llist_node,
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
struct llist_node {
    next: *mut llist_node,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits159U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits160U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits161U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
}


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mutex_h_253 {
    lock: *mut mutex,
}


#[inline(always)]
fn class_mutex_destructor(_T: *mut __SlateRecord_mutex_h_253) {
unsafe { mutex_unlock((*_T).lock) };
}

#[inline(always)]
fn class_mutex_constructor(l: *mut mutex) -> __SlateRecord_mutex_h_253 {
let mut _t: __SlateRecord_mutex_h_253 = __SlateRecord_mutex_h_253 { lock: l };
let _T: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(_t);
unsafe { mutex_lock((*_T).lock) };
return _t;
}

#[inline(always)]
fn __class_mutex_cleanup_ctx(_T: *mut *mut __SlateRecord_mutex_h_253) {
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
    __slate_bits_0: __slate_bits::__SlateBits172U0,
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
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
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
struct rq {
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
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
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
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
struct xol_area {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct mem_cgroup {
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
    __slate_bits_0: __slate_bits::__SlateBits295U0,
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


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
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
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
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


fn mem_alloc_profiling_enabled() -> bool {
false
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


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_123() };
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
__verify_pcpu_ptr__3!();
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
__verify_pcpu_ptr__2!();
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
struct kernfs_root {
}


fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__3!();
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
__verify_pcpu_ptr__2!();
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


// Don't care
// Suits me
// Don't call further
#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
struct wake_irq {
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
struct dev_pm_qos {
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
    __slate_bits_0: __slate_bits::__SlateBits427U0,
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
fn gfpflags_allow_blocking(gfp_flags: u32) -> bool {
gfp_flags & (BIT!(10i32) as u32) != 0
}

// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


fn mmdrop(mm: *mut mm_struct) {
// The implicit full barrier implied by atomic_dec_and_test() is
// required by the membarrier system call before returning to
// user-space, after storing to rq->curr.
if unlikely!(atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.__slate_anon_0.mm_count) })) != 0 {
unsafe { __mmdrop(mm) };
}
}

fn fs_reclaim_acquire(gfp_mask: u32) {
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


fn fs_reclaim_release(gfp_mask: u32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


fn might_alloc(gfp_mask: u32) {
fs_reclaim_acquire(gfp_mask);
fs_reclaim_release(gfp_mask);
if (unsafe { (*get_current()).flags }) & 2048 != 0 {
return;
}
might_sleep_if!(gfpflags_allow_blocking(gfp_mask));
}

fn xa_tag_pointer(p: *mut (), tag: u64) -> *mut () {
((p as u64) | tag) as *mut ()
}

fn xa_untag_pointer(entry: *mut ()) -> *mut () {
((entry as u64) & !3) as *mut ()
}

fn xa_pointer_tag(entry: *mut ()) -> u32 {
((entry as u64) & 3) as u32
}

fn xa_mk_internal(v: u64) -> *mut () {
(v << 2i32 | 2) as *mut ()
}

fn xa_is_internal(entry: *const ()) -> bool {
(entry as u64) & 3 == 2
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


fn xa_is_err(entry: *const ()) -> bool {
let __v0: bool = xa_is_internal(entry) && entry >= (xa_mk_internal(4095i32.wrapping_neg() as u64) as *const ());
__v0
}

fn xa_err(entry: *mut ()) -> i32 {
// xa_to_internal() would not do sign extension.
if xa_is_err(entry as *const ()) {
return ((entry as i64) >> 2i32) as i32;
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


fn xa_init_flags(xa: *mut xarray, flags: u32) {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*xa).xa_lock) });
unsafe {
*core::ptr::addr_of_mut!((*xa).xa_lock) = *{
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
(*xa).xa_flags = flags;
}
unsafe {
(*xa).xa_head = core::ptr::null_mut::<()>();
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


fn xa_init(xa: *mut xarray) {
xa_init_flags(xa, 0u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


fn xa_empty(xa: *const xarray) -> bool {
unsafe { (*xa).xa_head }.is_null()
}

// xa_for_each_range() - Iterate over a portion of an XArray.
// @xa: XArray.
// @index: Index of @entry.
// @entry: Entry retrieved from array.
// @start: First index to retrieve from array.
// @last: Last index to retrieve from array.
//
// During the iteration, @entry will have the value of the entry stored
// in @xa at @index.  You may modify @index during the iteration if you
// want to skip or reprocess indices.  It is safe to modify the array
// during the iteration.  At the end of the iteration, @entry will be set
// to NULL and @index will have a value less than or equal to max.
//
// xa_for_each_range() is O(n.log(n)) while xas_for_each() is O(n).  You have
// to handle your own locking with xas_for_each(), and if you have to unlock
// after each iteration, it will also end up being O(n.log(n)).
// xa_for_each_range() will spin if it hits a retry entry; if you intend to
// see retry entries, you should use the xas_for_each() iterator instead.
// The xas_for_each() iterator will expand into more inline code than
// xa_for_each_range().
//
// Context: Any context.  Takes and releases the RCU lock.
// xa_for_each_start() - Iterate over a portion of an XArray.
// @xa: XArray.
// @index: Index of @entry.
// @entry: Entry retrieved from array.
// @start: First index to retrieve from array.
//
// During the iteration, @entry will have the value of the entry stored
// in @xa at @index.  You may modify @index during the iteration if you
// want to skip or reprocess indices.  It is safe to modify the array
// during the iteration.  At the end of the iteration, @entry will be set
// to NULL and @index will have a value less than or equal to max.
//
// xa_for_each_start() is O(n.log(n)) while xas_for_each() is O(n).  You have
// to handle your own locking with xas_for_each(), and if you have to unlock
// after each iteration, it will also end up being O(n.log(n)).
// xa_for_each_start() will spin if it hits a retry entry; if you intend to
// see retry entries, you should use the xas_for_each() iterator instead.
// The xas_for_each() iterator will expand into more inline code than
// xa_for_each_start().
//
// Context: Any context.  Takes and releases the RCU lock.
fn xa_cmpxchg(xa: *mut xarray, index: u64, old: *mut (), entry: *mut (), gfp: u32) -> *mut () {
might_alloc(gfp);
xa_lock!(xa);
let curr: *mut () = unsafe { __xa_cmpxchg(xa, index, old, entry, gfp) };
xa_unlock!(xa);
curr
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


fn xa_insert(xa: *mut xarray, index: u64, entry: *mut (), gfp: u32) -> i32 {
let mut err: i32 = 0;
might_alloc(gfp);
xa_lock!(xa);
err = unsafe { __xa_insert(xa, index, entry, gfp) };
xa_unlock!(xa);
err
}

fn xa_reserve(xa: *mut xarray, index: u64, gfp: u32) -> i32 {
xa_err(xa_cmpxchg(xa, index, core::ptr::null_mut::<()>(), xa_mk_internal(257u64), gfp))
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


fn xa_release(xa: *mut xarray, index: u64) {
xa_cmpxchg(xa, index, xa_mk_internal(257u64), core::ptr::null_mut::<()>(), 0u32);
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
struct idr {
    idr_rt: xarray,
    idr_base: u32,
    idr_next: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


// The IDR API does not expose the tagging functionality of the radix tree
// to users.  Use tag 0 to track whether a node has free space below it.
// Set the IDR flag and the IDR_FREE tag
#[repr(C)]
#[derive(Clone, Copy)]
struct ida {
    xa: xarray,
}


fn ida_alloc(ida: *mut ida, gfp: u32) -> i32 {
unsafe { ida_alloc_range(ida, 0u32, 4294967295u32, gfp) }
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
struct tty_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits480U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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


fn sysfs_create_file(kobj: *mut kobject, attr: *const attribute) -> i32 {
unsafe { sysfs_create_file_ns(kobj, attr, core::ptr::null::<ns_common>()) }
}

fn sysfs_remove_file(kobj: *mut kobject, attr: *const attribute) {
unsafe { sysfs_remove_file_ns(kobj, attr, core::ptr::null::<ns_common>()) };
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
    __slate_bits_0: __slate_bits::__SlateBits482U0,
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
    __slate_bits_0: __slate_bits::__SlateBits507U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits507U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits507U2,
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
    __slate_bits_3: __slate_bits::__SlateBits507U3,
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
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct ctl_dir {
    header: ctl_table_header,
    root: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
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
struct posix_acl {
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
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_private {
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
    __slate_bits_0: __slate_bits::__SlateBits298U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
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
struct cgroup_taskset {
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
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
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
    __slate_bits_0: __slate_bits::__SlateBits653U0,
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
struct iova_bitmap {
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
// All kzalloc aliases for kmalloc_(obj|objs|flex).
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
struct iommu_dma_cookie {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dma_msi_cookie {
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
struct iommufd_ctx {
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
struct iommufd_access {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_device {
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
struct iommufd_hwpt_paging {
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
struct fsl_mc_resource_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
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
union __SlateRecord_mc_h_341 {
    mutex: mutex,
    spinlock: raw_spinlock,
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


fn dev_clear_dma_iommu(dev: *mut device) {
clear_bit(2i64, (unsafe { (*dev).flags.as_mut_ptr() as *mut u64 }) as *mut u64);
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
    __slate_bits_0: __slate_bits::__SlateBits504U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hw_irq_h_82 {
    ioapic: ioapic_alloc_info,
    uv: uv_alloc_info,
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
union __SlateRecord_msi_h_26 {
    __slate_anon_0: __SlateRecord_msi_h_26_2,
    dmar_subhandle: u32,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_26_2 {
    __slate_bits_0: __slate_bits::__SlateBits942U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_map {
    __slate_anon_0: __SlateRecord_user_namespace_h_26,
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
    __slate_bits_0: __slate_bits::__SlateBits946U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_3 {
    __slate_bits_0: __slate_bits::__SlateBits947U0,
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
union __SlateRecord_msi_h_109 {
    msi_mask: u32,
    msix_ctrl: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits956U0,
    default_irq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_122 {
    mask_pos: u8,
    mask_base: *mut (),
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
union __SlateRecord_msi_h_202 {
    pci: pci_msi_desc,
    data: msi_desc_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_desc {
    mask: cpumask,
    __slate_bits_0: __slate_bits::__SlateBits784U0,
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
struct __SlateRecord_iommu_c_258 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_264 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_306 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_318 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1029 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1029_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1029_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1029_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1029_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1029_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1031 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1031_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1031_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1031_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1031_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1031_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1034 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1034_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1034_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1034_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1034_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_1034_6 {
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
struct __SlateRecord_iommu_c_2569 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_2573 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_2585 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_3090_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_3122_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_c_3122_12 {
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
    pub struct __SlateBits622U0 {
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
    pub struct __SlateBits653U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits295U0 {
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
    pub struct __SlateBits427U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits480U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits793U0 {
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
    pub struct __SlateBits298U0 {
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
    pub struct __SlateBits159U0 {
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
    pub struct __SlateBits161U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits507U0 {
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
    pub struct __SlateBits504U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits507U1 {
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
    pub struct __SlateBits507U2 {
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
    pub struct __SlateBits507U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits946U0 {
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
    pub struct __SlateBits947U0 {
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
    pub struct __SlateBits949U0 {
        #[bits(8)]
        pub reserved: u32,
        #[bits(24)]
        pub destid_8_31: u32,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits942U0 {
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
    pub struct __SlateBits784U0 {
        #[bits(1)]
        pub is_managed: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits956U0 {
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
    pub struct __SlateBits926U0 {
        #[bits(1)]
        pub is_level: u32,
        #[bits(1)]
        pub active_low: u32,
        #[bits(1)]
        pub valid: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits839U0 {
        #[bits(1)]
        pub user_pasid_table: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits740U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits482U0 {
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
    pub struct __SlateBits160U0 {
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
    pub struct __SlateBits172U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits456U0 {
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
    pub struct __SlateBits888U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits827U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits827U1 {
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
    pub struct __SlateBits827U2 {
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
    pub struct __SlateBits827U3 {
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
    pub struct __SlateBits827U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits793U0,
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
    __slate_bits_0: __slate_bits::__SlateBits622U0,
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
struct scatterlist {
    page_link: u64,
    offset: u32,
    length: u32,
    dma_address: u64,
    dma_length: u32,
    dma_flags: u32,
}


// These macros should be used after a dma_map_sg call has been done
// to get bus addresses of each of the SG entries and their lengths.
// You should only work with the number of sg entries dma_map_sg
// returns, or alternatively stop on the first sg_dma_len(sg) which
// is 0.
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

fn sg_page(sg: *mut scatterlist) -> *mut page {
((unsafe { (*sg).page_link }) & !(1 | 2)) as *mut page
}

fn sg_dma_is_bus_address(sg: *mut scatterlist) -> bool {
(unsafe { (*sg).dma_flags }) & 1u32 != 0
}

fn sg_phys(sg: *mut scatterlist) -> u64 {
page_to_phys!(sg_page(sg)).wrapping_add((unsafe { (*sg).offset }) as u64)
}

fn fwnode_handle_put(fwnode: *mut fwnode_handle) {
loop {
if fwnode_has_op!(fwnode, put) {
unsafe { unsafe { (*(*fwnode).ops).put }.unwrap()(fwnode) };
}
if true {
break;
}
}
}

fn device_property_read_u32(dev: *const device, propname: *const u8, val: *mut u32) -> i32 {
unsafe { device_property_read_u32_array(dev, propname, val, 1u64) }
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


// Domain feature flags
// Support for iommu_map/unmap
// Domain for use in DMA-API
// implementation
// This are the possible domain-types
//
// 	IOMMU_DOMAIN_BLOCKED	- All DMA is blocked, can be used to isolate
// 				  devices
// 	IOMMU_DOMAIN_IDENTITY	- DMA addresses are system physical addresses
// 	IOMMU_DOMAIN_UNMANAGED	- DMA mappings managed by IOMMU-API user, used
// 				  for VMs
// 	IOMMU_DOMAIN_DMA	- Internally used for DMA-API implementations.
// 				  This flag allows IOMMU drivers to implement
// 				  certain optimizations for these domains
// 	IOMMU_DOMAIN_DMA_FQ	- As above, but definitely using batched TLB
// 				  invalidation.
// 	IOMMU_DOMAIN_SVA	- DMA addresses are shared process addresses
// 				  represented by mm_struct's.
// 	IOMMU_DOMAIN_PLATFORM	- Legacy domain for drivers that do their own
// 				  dma_api stuff. Do not use in new drivers.
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


fn iommu_is_dma_domain(domain: *mut iommu_domain) -> bool {
(unsafe { (*domain).r#type }) & 2u32 != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_resv_region {
    list: list_head,
    start: u64,
    length: u64,
    prot: i32,
    r#type: u32,
    free: Option<unsafe extern "C" fn(*mut device, *mut iommu_resv_region)>,
}


// Reserved for DMA w/o PASID
// starting range for allocation
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
    __slate_bits_0: __slate_bits::__SlateBits839U0,
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
    __slate_bits_0: __slate_bits::__SlateBits740U0,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_device_register")]
extern "C" fn iommu_device_register(iommu: *mut iommu_device, ops: *const iommu_ops, hwdev: *mut device) -> i32 {
let mut err: i32 = 0;
// We need to be able to take module references appropriately
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { is_module_address(ops as u64) }) && unsafe { (*ops).owner }.is_null()) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("694: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 694b - ., 3; .popsection\n# {0}", const 694 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_device_register_1, sym __slate_asm_string_iommu_device_register_2, const 295 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("695: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 695b - ., 4; .popsection\n# {0}", const 695 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*iommu).ops = ops;
}
if !hwdev.is_null() {
unsafe {
(*iommu).fwnode = dev_fwnode!(hwdev);
}
}
spin_lock(unsafe { core::ptr::addr_of_mut!(iommu_device_lock) });
list_add_tail(unsafe { core::ptr::addr_of_mut!((*iommu).list) }, unsafe { core::ptr::addr_of_mut!(iommu_device_list) });
spin_unlock(unsafe { core::ptr::addr_of_mut!(iommu_device_lock) });
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__37!() && err == 0 {
err = bus_iommu_probe(unsafe { iommu_buses.0[i as usize] });
i = i.wrapping_add(1);
}
if err != 0 {
iommu_device_unregister(iommu);
} else {
loop {
while false {
}
__WRITE_ONCE__5!((*iommu).ready, true);
if true {
break;
}
}
}
err
}

fn iommu_iotlb_gather_init(gather: *mut iommu_iotlb_gather) {
let mut __slate_compound_1: iommu_pages_list = unsafe { core::mem::zeroed() };
let mut __slate_compound_0: iommu_iotlb_gather = unsafe { core::mem::zeroed() };
unsafe {
*gather = *{
__slate_compound_0 = {
let mut __t0: iommu_iotlb_gather = core::mem::zeroed();
__t0.start = 18446744073709551615u64;
__t0.freelist = *{
__slate_compound_1 = iommu_pages_list { pages: list_head { next: core::ptr::addr_of_mut!((*gather).freelist.pages), prev: core::ptr::addr_of_mut!((*gather).freelist.pages) } };
core::ptr::addr_of_mut!(__slate_compound_1)
};
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
}

fn iommu_paging_domain_alloc(dev: *mut device) -> *mut iommu_domain {
iommu_paging_domain_alloc_flags(dev, 0u32)
}

fn iommu_iotlb_sync(domain: *mut iommu_domain, iotlb_gather: *mut iommu_iotlb_gather) {
if (unsafe { (*(*domain).ops).iotlb_sync }) != None && likely!((unsafe { (*iotlb_gather).start }) < unsafe { (*iotlb_gather).end }) != 0 {
unsafe { unsafe { (*(*domain).ops).iotlb_sync }.unwrap()(domain, iotlb_gather) };
}
iommu_iotlb_gather_init(iotlb_gather);
}

fn iommu_iotlb_gather_add_range(gather: *mut iommu_iotlb_gather, iova: u64, size: u64) {
let end: u64 = iova.wrapping_add(size).wrapping_sub(1);
if (unsafe { (*gather).start }) > iova {
unsafe {
(*gather).start = iova;
}
}
if (unsafe { (*gather).end }) < end {
unsafe {
(*gather).end = end;
}
}
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


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_fwspec_init")]
extern "C" fn iommu_fwspec_init(dev: *mut device, iommu_fwnode: *mut fwnode_handle) -> i32 {
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: u64 = unsafe { core::mem::zeroed() };
let iommu: *const iommu_device = iommu_from_fwnode(iommu_fwnode as *const fwnode_handle);
let mut fwspec: *mut iommu_fwspec = dev_iommu_fwspec_get(dev);
if iommu.is_null() {
return unsafe { driver_deferred_probe_check_state(dev) };
}
let __v0: bool = if unsafe { (*dev).iommu }.is_null() {
let mut __v1: bool;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_790() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*iommu).ready) as *const bool) };
!__v1
} else {
false
};
if __v0 {
return 517i32.wrapping_neg();
}
if !fwspec.is_null() {
return if (unsafe { (*iommu).ops }) == iommu_fwspec_ops(fwspec) { 0 } else { 22i32.wrapping_neg() };
}
if dev_iommu_get(dev).is_null() {
return 12i32.wrapping_neg();
}
// Preallocate for the overwhelmingly common case of 1 ID
let mut __v3: *mut iommu_fwspec;
let __count: u64 = 1;
let __obj_size: u64 = struct_size_t__8!(__count);
let mut __v4: *mut ();
let mut __v5: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_8 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_8)
} });
}
__v5 = _res;
__v4 = __v5;
let __obj_ptr: *mut iommu_fwspec = __v4 as *mut iommu_fwspec;
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
fwspec = __v3;
if fwspec.is_null() {
return 12i32.wrapping_neg();
}
unsafe { fwnode_handle_get(iommu_fwnode) };
unsafe {
(*fwspec).iommu_fwnode = iommu_fwnode;
}
dev_iommu_fwspec_set(dev, fwspec);
0
}

fn dev_iommu_fwspec_get(dev: *mut device) -> *mut iommu_fwspec {
if !unsafe { (*dev).iommu }.is_null() {
return unsafe { (*(*dev).iommu).fwspec };
} else {
return core::ptr::null_mut::<iommu_fwspec>();
}
}

fn dev_iommu_fwspec_set(dev: *mut device, fwspec: *mut iommu_fwspec) {
unsafe {
(*(*dev).iommu).fwspec = fwspec;
}
}

fn iommu_debugfs_setup() {
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
    __slate_bits_0: __slate_bits::__SlateBits888U0,
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
    __slate_bits_0: __slate_bits::__SlateBits827U0,
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
    __slate_bits_1: __slate_bits::__SlateBits827U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits827U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits827U3,
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
    __slate_bits_4: __slate_bits::__SlateBits827U4,
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
    __slate_bits_0: __slate_bits::__SlateBits456U0,
}


fn pci_is_root_bus(pbus: *mut pci_bus) -> bool {
unsafe { (*pbus).parent }.is_null()
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
struct fsl_mc_resource {
    r#type: u32,
    id: i32,
    data: *mut (),
    parent_pool: *mut fsl_mc_resource_pool,
    node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsl_mc_device_irq {
    virq: u32,
    mc_dev: *mut fsl_mc_device,
    dev_irq_index: u8,
    resource: fsl_mc_resource,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsl_mc_obj_desc {
    r#type: [u8; 16],
    id: i32,
    vendor: u16,
    ver_major: u16,
    ver_minor: u16,
    irq_count: u8,
    region_count: u8,
    state: u32,
    label: [u8; 16],
    flags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsl_mc_device {
    dev: device,
    dma_mask: u64,
    flags: u16,
    icid: u32,
    mc_handle: u16,
    mc_io: *mut fsl_mc_io,
    obj_desc: fsl_mc_obj_desc,
    regions: *mut resource,
    irqs: *mut *mut fsl_mc_device_irq,
    resource: *mut fsl_mc_resource,
    consumer_link: *mut device_link,
}


// High priority flag
// Command completion flag
#[repr(C)]
#[derive(Clone, Copy)]
struct fsl_mc_io {
    dev: *mut device,
    flags: u16,
    portal_size: u32,
    portal_phys_addr: u64,
    portal_virt_addr: *mut (),
    dpmcp_dev: *mut fsl_mc_device,
    __slate_anon_6: __SlateRecord_mc_h_341,
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
struct irqstat {
    cnt: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits926U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uv_alloc_info {
    limit: i32,
    blade: i32,
    offset: u64,
    name: *mut u8,
}


core::arch::global_asm!(".section\t\".initcall4.init\", \"a\"\t\t\n__initcall__kmod_iommu__692_271_iommu_subsys_init4:\t\t\t\n.long\t{0} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_device_register: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_device_unregister: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dev_iommu_priv_set: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_get_group_resv_regions: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_alloc: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_get_iommudata: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_set_iommudata: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_set_name: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_add_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_remove_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_for_each_dev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_get: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_ref_get: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_put: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_id: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_generic_device_group: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_generic_single_device_group: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_device_group: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_fsl_mc_device_group: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_device_iommu_capable: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_has_isolated_msi: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_set_fault_handler: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_paging_domain_alloc_flags: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_domain_free: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_attach_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_detach_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_get_domain_for_dev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_driver_get_domain_for_dev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_attach_group: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_detach_group: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_iova_to_phys: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_map: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_unmap: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_unmap_fast: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_map_sg: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_report_iommu_fault: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section\t\".initcall1.init\", \"a\"\t\t\n__initcall__kmod_iommu__783_2962_iommu_init1:\t\t\t\n.long\t{37} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_set_pgtable_quirks: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_get_resv_regions: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_put_resv_regions: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_alloc_resv_region: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_default_passthrough: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_fwspec_init: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_fwspec_add_ids: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {44} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_claim_dma_owner: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {45} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_device_claim_dma_owner: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {46} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_release_dma_owner: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {47} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_device_release_dma_owner: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {48} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_group_dma_owner_claimed: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {49} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_attach_device_pasid: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {50} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_replace_device_pasid: ; .asciz \"GPL\" ; .ascii \"IOMMUFD_INTERNAL\" \"\\0\" ; .balign 8 ; .quad {51} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_detach_device_pasid: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {52} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_alloc_global_pasid: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {53} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_free_global_pasid: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {54} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_attach_handle_get: ; .asciz \"GPL\" ; .ascii \"IOMMUFD_INTERNAL\" \"\\0\" ; .balign 8 ; .quad {55} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_attach_group_handle: ; .asciz \"GPL\" ; .ascii \"IOMMUFD_INTERNAL\" \"\\0\" ; .balign 8 ; .quad {56} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_detach_group_handle: ; .asciz \"GPL\" ; .ascii \"IOMMUFD_INTERNAL\" \"\\0\" ; .balign 8 ; .quad {57} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_iommu_replace_group_handle: ; .asciz \"GPL\" ; .ascii \"IOMMUFD_INTERNAL\" \"\\0\" ; .balign 8 ; .quad {58} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_dev_reset_iommu_prepare: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {59} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_dev_reset_iommu_done: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {60} ; .previous", sym iommu_subsys_init, sym iommu_device_register, sym iommu_device_unregister, sym dev_iommu_priv_set, sym iommu_get_group_resv_regions, sym iommu_group_alloc, sym iommu_group_get_iommudata, sym iommu_group_set_iommudata, sym iommu_group_set_name, sym iommu_group_add_device, sym iommu_group_remove_device, sym iommu_group_for_each_dev, sym iommu_group_get, sym iommu_group_ref_get, sym iommu_group_put, sym iommu_group_id, sym generic_device_group, sym generic_single_device_group, sym pci_device_group, sym fsl_mc_device_group, sym device_iommu_capable, sym iommu_group_has_isolated_msi, sym iommu_set_fault_handler, sym iommu_paging_domain_alloc_flags, sym iommu_domain_free, sym iommu_attach_device, sym iommu_detach_device, sym iommu_get_domain_for_dev, sym iommu_driver_get_domain_for_dev, sym iommu_attach_group, sym iommu_detach_group, sym iommu_iova_to_phys, sym iommu_map, sym iommu_unmap, sym iommu_unmap_fast, sym iommu_map_sg, sym report_iommu_fault, sym iommu_init, sym iommu_set_pgtable_quirks, sym iommu_get_resv_regions, sym iommu_put_resv_regions, sym iommu_alloc_resv_region, sym iommu_default_passthrough, sym iommu_fwspec_init, sym iommu_fwspec_add_ids, sym iommu_group_claim_dma_owner, sym iommu_device_claim_dma_owner, sym iommu_group_release_dma_owner, sym iommu_device_release_dma_owner, sym iommu_group_dma_owner_claimed, sym iommu_attach_device_pasid, sym iommu_replace_device_pasid, sym iommu_detach_device_pasid, sym iommu_alloc_global_pasid, sym iommu_free_global_pasid, sym iommu_attach_handle_get, sym iommu_attach_group_handle, sym iommu_detach_group_handle, sym iommu_replace_group_handle, sym pci_dev_reset_iommu_prepare, sym pci_dev_reset_iommu_done, options(att_syntax));

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
    __slate_bits_0: __slate_bits::__SlateBits949U0,
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
fn __do_trace_add_device_to_group(group_id: i32, dev: *mut device) {
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
let mut __UNIQUE_ID_guard_649: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_649) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_650: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_650) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_652() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_add_device_to_group)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_651: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_651;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), i32, *mut device) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32, *mut device) -> i32>>(__SCT__tp_func_add_device_to_group as *const ()) };
unsafe { __v12.unwrap()(__data, group_id, dev) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_add_device_to_group: __SlateAlign16<[u8; 20]> = __SlateAlign16([97u8, 100u8, 100u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 95u8, 116u8, 111u8, 95u8, 103u8, 114u8, 111u8, 117u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_add_device_to_group_653: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_add_device_to_group) }) as *mut ();

fn trace_add_device_to_group(group_id: i32, dev: *mut device) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_add_device_to_group.key) }) != 0 {
__do_trace_add_device_to_group(group_id, dev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_654: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_remove_device_from_group(group_id: i32, dev: *mut device) {
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
let mut __UNIQUE_ID_guard_656: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_656) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_657: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_657) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_659() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_remove_device_from_group)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_658: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_658;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), i32, *mut device) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32, *mut device) -> i32>>(__SCT__tp_func_remove_device_from_group as *const ()) };
unsafe { __v12.unwrap()(__data, group_id, dev) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_remove_device_from_group: __SlateAlign16<[u8; 25]> = __SlateAlign16([114u8, 101u8, 109u8, 111u8, 118u8, 101u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 95u8, 102u8, 114u8, 111u8, 109u8, 95u8, 103u8, 114u8, 111u8, 117u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_remove_device_from_group_660: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_remove_device_from_group) }) as *mut ();

fn trace_remove_device_from_group(group_id: i32, dev: *mut device) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_remove_device_from_group.key) }) != 0 {
__do_trace_remove_device_from_group(group_id, dev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_661: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_attach_device_to_domain(dev: *mut device) {
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
let mut __UNIQUE_ID_guard_663: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_663) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_664: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_664) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_666() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_attach_device_to_domain)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_665: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_665;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut device) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut device) -> i32>>(__SCT__tp_func_attach_device_to_domain as *const ()) };
unsafe { __v12.unwrap()(__data, dev) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_attach_device_to_domain: __SlateAlign16<[u8; 24]> = __SlateAlign16([97u8, 116u8, 116u8, 97u8, 99u8, 104u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 95u8, 116u8, 111u8, 95u8, 100u8, 111u8, 109u8, 97u8, 105u8, 110u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_attach_device_to_domain_667: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_attach_device_to_domain) }) as *mut ();

fn trace_attach_device_to_domain(dev: *mut device) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_attach_device_to_domain.key) }) != 0 {
__do_trace_attach_device_to_domain(dev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_668: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_map(iova: u64, paddr: u64, size: u64) {
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
let mut __UNIQUE_ID_guard_670: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_670) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_671: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_671) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_673() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_map)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_672: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_672;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), u64, u64, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), u64, u64, u64) -> i32>>(__SCT__tp_func_map as *const ()) };
unsafe { __v12.unwrap()(__data, iova, paddr, size) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_map: [u8; 4] = [109, 97, 112, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_map_674: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_map) }) as *mut ();

fn trace_map(iova: u64, paddr: u64, size: u64) {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_map.key) }) != 0 {
__do_trace_map(iova, paddr, size);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_675: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_unmap(iova: u64, size: u64, unmapped_size: u64) {
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
let mut __UNIQUE_ID_guard_677: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_677) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_678: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_678) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_680() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_unmap)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_679: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_679;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), u64, u64, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), u64, u64, u64) -> i32>>(__SCT__tp_func_unmap as *const ()) };
unsafe { __v12.unwrap()(__data, iova, size, unmapped_size) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_unmap: [u8; 6] = [117, 110, 109, 97, 112, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_unmap_681: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_unmap) }) as *mut ();

fn trace_unmap(iova: u64, size: u64, unmapped_size: u64) {
if static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(__tracepoint_unmap.key) }) != 0 {
__do_trace_unmap(iova, size, unmapped_size);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_682: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_io_page_fault(dev: *mut device, iova: u64, flags: i32) {
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
let mut __UNIQUE_ID_guard_684: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_684) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_685: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_685) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_687() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_io_page_fault)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_686: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_686;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut device, u64, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut device, u64, i32) -> i32>>(__SCT__tp_func_io_page_fault as *const ()) };
unsafe { __v12.unwrap()(__data, dev, iova, flags) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_io_page_fault: [u8; 14] = [105, 111, 95, 112, 97, 103, 101, 95, 102, 97, 117, 108, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_io_page_fault_688: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_io_page_fault) }) as *mut ();

fn trace_io_page_fault(dev: *mut device, iova: u64, flags: i32) {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(__tracepoint_io_page_fault.key) }) != 0 {
__do_trace_io_page_fault(dev, iova, flags);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_689: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct pt_iommu {
    domain: iommu_domain,
    ops: *const pt_iommu_ops,
    driver_ops: *const pt_iommu_driver_ops,
    nid: i32,
    iommu_device: *mut device,
}


fn iommupt_from_domain(domain: *mut iommu_domain) -> *mut pt_iommu {
if IS_ENABLED__2!() == 0 || !unsafe { (*domain).is_iommupt } {
return core::ptr::null_mut::<pt_iommu>();
}
container_of__6!(domain, pt_iommu)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pt_iommu_info {
    pgsize_bitmap: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pt_iommu_ops {
    map_range: Option<unsafe extern "C" fn(*mut pt_iommu, u64, u64, u64, u32, u32, *mut u64) -> i32>,
    unmap_range: Option<unsafe extern "C" fn(*mut pt_iommu, u64, u64, *mut iommu_iotlb_gather) -> u64>,
    set_dirty: Option<unsafe extern "C" fn(*mut pt_iommu, u64) -> i32>,
    get_info: Option<unsafe extern "C" fn(*mut pt_iommu, *mut pt_iommu_info)>,
    deinit: Option<unsafe extern "C" fn(*mut pt_iommu)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pt_iommu_driver_ops {
    change_top: Option<unsafe extern "C" fn(*mut pt_iommu, u64, u32)>,
    get_top_lock: Option<unsafe extern "C" fn(*mut pt_iommu) -> *mut spinlock>,
}


// Generate the exported function signatures from iommu_pt.h
// amdv1_mock is used by the iommufd selftest
fn dev_iommu_ops(dev: *mut device) -> *const iommu_ops {
// Assume that valid ops must be installed if iommu_probe_device()
// has succeeded. The device ops are essentially for internal use
// within the IOMMU subsystem itself, so we should be able to trust
// ourselves not to misuse the helper.
unsafe { (*(*(*dev).iommu).iommu_dev).ops }
}

fn iommu_fwspec_ops(fwspec: *mut iommu_fwspec) -> *const iommu_ops {
iommu_ops_from_fwnode((if !fwspec.is_null() { unsafe { (*fwspec).iommu_fwnode } } else { core::ptr::null_mut::<fwnode_handle>() }) as *const fwnode_handle)
}

fn iommu_debug_map(domain: *mut iommu_domain, phys: u64, size: u64) {
}

fn iommu_debug_unmap_begin(domain: *mut iommu_domain, iova: u64, size: u64) {
}

fn iommu_debug_unmap_end(domain: *mut iommu_domain, iova: u64, size: u64, unmapped: u64) {
}

fn iommu_debug_init() {
}

static mut iommu_group_kset: *mut kset = unsafe { core::mem::zeroed() };

static mut iommu_group_ida: ida = ida { xa: xarray { xa_lock: spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} }, xa_flags: 16777221u32, xa_head: core::ptr::null_mut::<()>() } };

static mut iommu_global_pasid_ida: ida = ida { xa: xarray { xa_lock: spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} }, xa_flags: 16777221u32, xa_head: core::ptr::null_mut::<()>() } };

#[unsafe(link_section = ".data..read_mostly")]
static mut iommu_def_domain_type: u32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".data..read_mostly")]
static mut iommu_dma_strict: bool = 0i32 != 0i32;

#[unsafe(link_section = ".data..read_mostly")]
static mut iommu_cmd_line: u32 = unsafe { core::mem::zeroed() };

// Tags used with xa_tag_pointer() in group->pasid_array
#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group {
    kobj: kobject,
    devices_kobj: *mut kobject,
    devices: list_head,
    pasid_array: xarray,
    mutex: mutex,
    iommu_data: *mut (),
    iommu_data_release: Option<unsafe extern "C" fn(*mut ())>,
    name: *mut u8,
    id: i32,
    default_domain: *mut iommu_domain,
    blocking_domain: *mut iommu_domain,
    domain: *mut iommu_domain,
    entry: list_head,
    owner_cnt: u32,
    /// Number of devices in the group undergoing or awaiting recovery.
    /// If non-zero, concurrent domain attachments are rejected.
    recovery_cnt: u32,
    owner: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct group_device {
    list: list_head,
    dev: *mut device,
    name: *mut u8,
    /// Device is blocked for a pending recovery while its group->domain is
    /// retained. This can happen when:
    ///  - Device is undergoing a reset
    blocked: bool,
    reset_depth: u32,
}


// Iterate over each struct group_device in a struct iommu_group
fn __dev_to_gdev(dev: *mut device) -> *mut group_device {
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
let mut gdev: *mut group_device = unsafe { core::mem::zeroed() };
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
gdev = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(gdev, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
if (unsafe { (*gdev).dev }) == dev {
return gdev;
}
gdev = list_next_entry__4!(gdev, list);
}
core::ptr::null_mut::<group_device>()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group_attribute {
    attr: attribute,
    show: Option<unsafe extern "C" fn(*mut iommu_group, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut iommu_group, *const u8, u64) -> i64>,
}


static iommu_group_resv_type_string: __SlateAlign16<[*const u8; 5]> = __SlateAlign16([(b"direct\0".as_ptr() as *mut u8) as *const u8, (b"direct-relaxable\0".as_ptr() as *mut u8) as *const u8, (b"reserved\0".as_ptr() as *mut u8) as *const u8, (b"msi\0".as_ptr() as *mut u8) as *const u8, (b"msi\0".as_ptr() as *mut u8) as *const u8]);

fn __iommu_group_set_domain(group: *mut iommu_group, new_domain: *mut iommu_domain) -> i32 {
__iommu_group_set_domain_internal(group, new_domain, 0u32)
}

fn __iommu_group_set_domain_nofail(group: *mut iommu_group, new_domain: *mut iommu_domain) {
let mut __v0: i64;
let __ret_warn_on: i32 = (__iommu_group_set_domain_internal(group, new_domain, 1u32) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("690: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 690b - ., 3; .popsection\n# {0}", const 690 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_group_set_domain_nofail_1, sym __slate_asm_string___iommu_group_set_domain_nofail_2, const 158 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("691: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 691b - ., 4; .popsection\n# {0}", const 691 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

fn iommu_create_device_direct_mappings(domain: *mut iommu_domain, dev: *mut device) -> i32 {
let mut phys_addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(phys_addr) as *mut u64;
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut map_size: u64 = unsafe { core::mem::zeroed() };
let mut addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(addr) as *mut u64;
let mut end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u64 = core::ptr::addr_of_mut!(end) as *mut u64;
let mut start: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(start) as *mut u64;
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut pg_size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(pg_size) as *mut u64;
let mut mappings: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut list_head = core::ptr::addr_of_mut!(mappings) as *mut list_head;
let mut entry: core::mem::MaybeUninit<*mut iommu_resv_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut iommu_resv_region = core::ptr::addr_of_mut!(entry) as *mut *mut iommu_resv_region;
unsafe {
{
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_15, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_15), prev: core::ptr::addr_of_mut!(*__slate_slot_15) });
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
*__slate_slot_14 = if (*domain).pgsize_bitmap != 0 { 1u64 << __ffs!(unsafe { (*domain).pgsize_bitmap }) } else { 0 };
__ret_warn_on = (iommu_is_dma_domain(domain) && *__slate_slot_14 == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("721: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 721b - ., 3; .popsection\n# {0}", const 721 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_create_device_direct_mappings_19, sym __slate_asm_string_iommu_create_device_direct_mappings_20, const 1210 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("722: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 722b - ., 4; .popsection\n# {0}", const 722 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
let __v12: i64 = (__ret_warn_on != 0) as i64;
if __v12 != 0 {
return 22i32.wrapping_neg();
} else {
iommu_get_resv_regions(dev, core::ptr::addr_of_mut!(*__slate_slot_15));
// We need to consider overlapping regions for different devices
*__slate_slot_16 = list_first_entry__7!(core::ptr::addr_of_mut!(*__slate_slot_15));
'__loop_1: while list_entry_is_head!(*__slate_slot_16, core::ptr::addr_of_mut!(*__slate_slot_15), list) == 0 {
{
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
map_size = 0;
if ((*(*__slate_slot_16)).r#type as u32) == 0 {
(*(*dev).iommu).__slate_bits_0.__set_require_direct(1u32);
}
}
if ((*(*__slate_slot_16)).r#type as u32) != 0 && ((*(*__slate_slot_16)).r#type as u32) != 1 {
__v5 = true;
} else {
__v5 = !iommu_is_dma_domain(domain);
}
if !__v5 {
*__slate_slot_9 = ALIGN!(unsafe { (*(*__slate_slot_16)).start }, *__slate_slot_14);
*__slate_slot_8 = ALIGN!(unsafe { (*(*__slate_slot_16)).start }.wrapping_add((*(*__slate_slot_16)).length), *__slate_slot_14);
*__slate_slot_7 = *__slate_slot_9;
while *__slate_slot_7 <= *__slate_slot_8 {
'__join_4: {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if *__slate_slot_7 != *__slate_slot_8 {
// Return address by iommu_iova_to_phys for 0 is
// ambiguous. Offset to address 1 if addr is 0.
*__slate_slot_4 = iommu_iova_to_phys(domain, if *__slate_slot_7 != 0 { *__slate_slot_7 } else { 1u64 });
if *__slate_slot_4 == 0 {
map_size = map_size.wrapping_add(*__slate_slot_14);
break '__join_4;
}
}
if map_size != 0 {
ret = iommu_map(domain, (*__slate_slot_7).wrapping_sub(map_size), (*__slate_slot_7).wrapping_sub(map_size), map_size, (*(*__slate_slot_16)).prot, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if ret != 0 {
break '__loop_1;
} else {
map_size = 0;
}
}
}
let __v1: u64 = *__slate_slot_7;
*__slate_slot_7 = __v1.wrapping_add(*__slate_slot_14);
}
}
*__slate_slot_16 = list_next_entry__6!(*__slate_slot_16, list);
}
iommu_put_resv_regions(dev, core::ptr::addr_of_mut!(*__slate_slot_15));
return ret;
}
}
}

static mut iommu_device_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(iommu_device_list) }, prev: unsafe { core::ptr::addr_of_mut!(iommu_device_list) } };

static mut iommu_device_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

static iommu_buses: __SlateAlign16<[*const bus_type; 2]> = __SlateAlign16([unsafe { core::ptr::addr_of!(platform_bus_type) }, unsafe { core::ptr::addr_of!(pci_bus_type) }]);

/// Use a function instead of an array here because the domain-type is a
/// bit-field, so an array would waste memory.
fn iommu_domain_type_str(t: u32) -> *const u8 {
match t {
0 => {
return c"Blocked".as_ptr() as *const u8;
}
4 => {
return c"Passthrough".as_ptr() as *const u8;
}
1 => {
return c"Unmanaged".as_ptr() as *const u8;
}
3 | 11 => {
return c"Translated".as_ptr() as *const u8;
}
32 => {
return c"Platform".as_ptr() as *const u8;
}
_ => {
return c"Unknown".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(link_section = ".init.text")]
extern "C" fn iommu_subsys_init() -> i32 {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
if ((unsafe { iommu_cmd_line }) as u64) & BIT!(0i32) == 0 {
if IS_ENABLED!() != 0 {
iommu_set_default_passthrough(false);
} else {
iommu_set_default_translated(false);
}
let __v0: bool = iommu_default_passthrough() && cc_platform_has(0u32);
if __v0 {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x016iommu: Memory encryption detected - Disabling default IOMMU Passthrough\n".as_ptr() as *const u8) };
iommu_set_default_translated(false);
}
}
if !iommu_default_passthrough() && !unsafe { iommu_dma_strict } {
unsafe {
iommu_def_domain_type = 11u32;
}
}
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x016iommu: Default domain type: %s%s\n".as_ptr() as *const u8, iommu_domain_type_str(iommu_def_domain_type), if (iommu_cmd_line as u64) & BIT!(0i32) != 0 { c" (set via kernel command line)".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
if !iommu_default_passthrough() {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x016iommu: DMA domain TLB invalidation policy: %s mode%s\n".as_ptr() as *const u8, if iommu_dma_strict { c"strict".as_ptr() as *mut u8 } else { c"lazy".as_ptr() as *mut u8 }, if (iommu_cmd_line as u64) & BIT!(1i32) != 0 { c" (set via kernel command line)".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}
let mut __v7: *mut notifier_block;
let __obj_size: u64 = size_mul(core::mem::size_of::<notifier_block>() as u64, ARRAY_SIZE__41!());
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
__v7 = __v8 as *mut notifier_block;
let nb: *mut notifier_block = __v7;
if nb.is_null() {
return 12i32.wrapping_neg();
}
iommu_debug_init();
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__13!() {
unsafe {
(*nb.wrapping_offset(i as isize)).notifier_call = Some(iommu_bus_notifier);
}
unsafe { bus_register_notifier(iommu_buses.0[i as usize], nb.wrapping_offset(i as isize)) };
i = i.wrapping_add(1);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_subsys_init_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(iommu_subsys_init)) };

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.remove_iommu_group")]
extern "C" fn remove_iommu_group(dev: *mut device, data: *mut ()) -> i32 {
if !unsafe { (*dev).iommu }.is_null() && (unsafe { (*(*dev).iommu).iommu_dev }) == (data as *mut iommu_device) {
iommu_release_device(dev);
}
0
}

// iommu_device_register() - Register an IOMMU hardware instance
// @iommu: IOMMU handle for the instance
// @ops:   IOMMU ops to associate with the instance
// @hwdev: (optional) actual instance device, used for fwnode lookup
//
// Return: 0 on success, or an error.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_device_register_697: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_device, *const iommu_ops, *mut device) -> i32>, *mut ()>(Some(iommu_device_register)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_device_unregister")]
extern "C" fn iommu_device_unregister(iommu: *mut iommu_device) {
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__23!() {
unsafe { bus_for_each_dev(iommu_buses.0[i as usize], core::ptr::null_mut::<device>(), iommu as *mut (), Some(remove_iommu_group)) };
i = i.wrapping_add(1);
}
spin_lock(unsafe { core::ptr::addr_of_mut!(iommu_device_lock) });
list_del(unsafe { core::ptr::addr_of_mut!((*iommu).list) });
spin_unlock(unsafe { core::ptr::addr_of_mut!(iommu_device_lock) });
// Pairs with the alloc in generic_single_device_group()
iommu_group_put(unsafe { (*iommu).singleton_group });
unsafe {
(*iommu).singleton_group = core::ptr::null_mut::<iommu_group>();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_device_unregister_698: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_device)>, *mut ()>(Some(iommu_device_unregister)) };

fn dev_iommu_get(dev: *mut device) -> *mut dev_iommu {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut param: *mut dev_iommu = unsafe { (*dev).iommu };
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(iommu_probe_device_lock) });
if !param.is_null() {
return param;
}
let mut __v1: *mut dev_iommu;
let __obj_size: u64 = size_mul(core::mem::size_of::<dev_iommu>() as u64, 1u64);
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
__v1 = __v2 as *mut dev_iommu;
param = __v1;
if param.is_null() {
return core::ptr::null_mut::<dev_iommu>();
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*param).lock) }, c"&param->lock".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__412) });
if true {
break;
}
}
unsafe {
(*dev).iommu = param;
}
param
}

static mut __key__412: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
extern "C" fn dev_iommu_free(dev: *mut device) {
let param: *mut dev_iommu = unsafe { (*dev).iommu };
unsafe {
(*dev).iommu = core::ptr::null_mut::<dev_iommu>();
}
if !unsafe { (*param).fwspec }.is_null() {
fwnode_handle_put(unsafe { (*(*param).fwspec).iommu_fwnode });
unsafe { kfree((*param).fwspec as *const ()) };
}
unsafe { kfree(param as *const ()) };
}

/// Internal equivalent of device_iommu_mapped() for when we care that a device
/// actually has API ops, and don't want false positives from VFIO-only groups.
fn dev_has_iommu(dev: *mut device) -> bool {
!unsafe { (*dev).iommu }.is_null() && !unsafe { (*(*dev).iommu).iommu_dev }.is_null()
}

fn dev_iommu_get_max_pasids(dev: *mut device) -> u32 {
let mut max_pasids: u32 = 0;
let mut bits: u32 = 0;
let mut ret: i32 = 0;
if dev_is_pci!(dev) {
ret = unsafe { pci_max_pasids(to_pci_dev!(dev)) };
if ret > 0 {
max_pasids = ret as u32;
}
} else {
ret = device_property_read_u32(dev as *const device, c"pasid-num-bits".as_ptr() as *const u8, core::ptr::addr_of_mut!(bits));
if ret == 0 {
max_pasids = (1u64 << bits) as u32;
}
}
min_t__3!(u32, max_pasids, unsafe { (*(*(*dev).iommu).iommu_dev).max_pasids })
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.dev_iommu_priv_set")]
extern "C" fn dev_iommu_priv_set(dev: *mut device, r#priv: *mut ()) {
// FSL_PAMU does something weird
if IS_ENABLED!() == 0 {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(iommu_probe_device_lock) });
}
unsafe {
(*(*dev).iommu).r#priv = r#priv;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dev_iommu_priv_set_701: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, *mut ())>, *mut ()>(Some(dev_iommu_priv_set)) };

/// Init the dev->iommu and dev->iommu_group in the struct device and get the
/// driver probed
fn iommu_init_device(dev: *mut device) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut group: core::mem::MaybeUninit<*mut iommu_group> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut iommu_group = core::ptr::addr_of_mut!(group) as *mut *mut iommu_group;
let mut iommu_dev: core::mem::MaybeUninit<*mut iommu_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut iommu_device = core::ptr::addr_of_mut!(iommu_dev) as *mut *mut iommu_device;
let mut ops: core::mem::MaybeUninit<*const iommu_ops> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *const iommu_ops = core::ptr::addr_of_mut!(ops) as *mut *const iommu_ops;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if dev_iommu_get(dev).is_null() {
return 12i32.wrapping_neg();
} else {
// For FDT-based systems and ACPI IORT/VIOT, the common firmware parsing
// is buried in the bus dma_configure path. Properly unpicking that is
// still a big job, so for now just invoke the whole thing. The device
// already having a driver bound means dma_configure has already run and
// found no IOMMU to wait for, so there's no point calling it again.
if unsafe { (*(*dev).iommu).fwspec }.is_null() && unsafe { (*dev).driver }.is_null() && (*(*dev).bus).dma_configure != None {
mutex_unlock(core::ptr::addr_of_mut!(iommu_probe_device_lock));
unsafe { (*(*dev).bus).dma_configure }.unwrap()(dev);
mutex_lock(core::ptr::addr_of_mut!(iommu_probe_device_lock));
// If another instance finished the job for us, skip it
if unsafe { (*dev).iommu }.is_null() || !unsafe { (*dev).iommu_group }.is_null() {
return 19i32.wrapping_neg();
}
}
// At this point, relevant devices either now have a fwspec which will
// match ops registered with a non-NULL fwnode, or we can reasonably
// assume that only one of Intel, AMD, s390, PAMU or legacy SMMUv2 can
// be present, and that any of their registered instances has suitable
// ops for probing, and thus cheekily co-opt the same mechanism.
*__slate_slot_6 = iommu_fwspec_ops((*(*dev).iommu).fwspec);
if (*__slate_slot_6).is_null() {
*__slate_slot_3 = 19i32.wrapping_neg();
} else {
if !try_module_get((*(*__slate_slot_6)).owner) {
*__slate_slot_3 = 22i32.wrapping_neg();
} else {
*__slate_slot_5 = unsafe { (*(*__slate_slot_6)).probe_device }.unwrap()(dev);
if IS_ERR(*__slate_slot_5 as *const ()) {
*__slate_slot_3 = PTR_ERR(*__slate_slot_5 as *const ()) as i32;
} else {
(*(*dev).iommu).iommu_dev = *__slate_slot_5;
*__slate_slot_3 = iommu_device_link(*__slate_slot_5, dev);
if *__slate_slot_3 == 0 {
*__slate_slot_4 = unsafe { (*(*__slate_slot_6)).device_group }.unwrap()(dev);
__ret_warn_on = (*__slate_slot_4).is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("702: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 702b - ., 3; .popsection\n# {0}", const 702 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_init_device_7, sym __slate_asm_string_iommu_init_device_8, const 523 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("703: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 703b - ., 4; .popsection\n# {0}", const 703 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
*__slate_slot_4 = ERR_PTR(22i32.wrapping_neg() as i64) as *mut iommu_group;
}
if IS_ERR(*__slate_slot_4 as *const ()) {
*__slate_slot_3 = PTR_ERR(*__slate_slot_4 as *const ()) as i32;
iommu_device_unlink(*__slate_slot_5, dev);
} else {
(*dev).iommu_group = *__slate_slot_4;
(*(*dev).iommu).max_pasids = dev_iommu_get_max_pasids(dev);
if (*(*__slate_slot_6)).is_attach_deferred != None {
let __t0 = unsafe { (*(*__slate_slot_6)).is_attach_deferred }.unwrap()(dev) as u32;
(*(*dev).iommu).__slate_bits_0.__set_attach_deferred(__t0);
}
return 0;
}
}
if (*(*__slate_slot_6)).release_device != None {
unsafe { (*(*__slate_slot_6)).release_device }.unwrap()(dev);
}
}
module_put((*(*__slate_slot_6)).owner);
}
}
(*(*dev).iommu).iommu_dev = core::ptr::null_mut::<iommu_device>();
dev_iommu_free(dev);
return *__slate_slot_3;
}
}
}

fn iommu_deinit_device(dev: *mut device) {
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
let ops: *const iommu_ops = dev_iommu_ops(dev);
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
unsafe { iommu_device_unlink((*(*dev).iommu).iommu_dev, dev) };
// release_device() must stop using any attached domain on the device.
// If there are still other devices in the group, they are not affected
// by this callback.
//
// If the iommu driver provides release_domain, the core code ensures
// that domain is attached prior to calling release_device. Drivers can
// use this to enforce a translation on the idle iommu. Typically, the
// global static blocked_domain is a good choice.
//
// Otherwise, the iommu driver must set the device to either an identity
// or a blocking translation in release_device() and stop using any
// domain pointer, as it is going to be freed.
//
// Regardless, if a delayed attach never occurred, then the release
// should still avoid touching any hardware configuration either.
if ((unsafe { (*(*dev).iommu).__slate_bits_0.__get_attach_deferred() }) as i32) == 0 && !unsafe { (*ops).release_domain }.is_null() {
let mut release_domain: *mut iommu_domain = unsafe { (*ops).release_domain };
// If the device requires direct mappings then it should not
// be parked on a BLOCKED domain during release as that would
// break the direct mappings.
if ((unsafe { (*(*dev).iommu).__slate_bits_0.__get_require_direct() }) as i32) != 0 && !unsafe { (*ops).identity_domain }.is_null() && release_domain == unsafe { (*ops).blocked_domain } {
release_domain = unsafe { (*ops).identity_domain };
}
unsafe { unsafe { (*(*release_domain).ops).attach_dev }.unwrap()(release_domain, dev, (*group).domain) };
}
if (unsafe { (*ops).release_device }) != None {
unsafe { unsafe { (*ops).release_device }.unwrap()(dev) };
}
// If this is the last driver to use the group then we must free the
// domains before we do the module_put().
if list_empty((unsafe { core::ptr::addr_of_mut!((*group).devices) }) as *const list_head) != 0 {
if !unsafe { (*group).default_domain }.is_null() {
iommu_domain_free(unsafe { (*group).default_domain });
unsafe {
(*group).default_domain = core::ptr::null_mut::<iommu_domain>();
}
}
if !unsafe { (*group).blocking_domain }.is_null() {
iommu_domain_free(unsafe { (*group).blocking_domain });
unsafe {
(*group).blocking_domain = core::ptr::null_mut::<iommu_domain>();
}
}
unsafe {
(*group).domain = core::ptr::null_mut::<iommu_domain>();
}
}
// Caller must put iommu_group
unsafe {
(*dev).iommu_group = core::ptr::null_mut::<iommu_group>();
}
unsafe { module_put((*ops).owner) };
dev_iommu_free(dev);
if IS_ENABLED__2!() != 0 {
dev_clear_dma_iommu(dev);
}
}

fn pasid_array_entry_to_domain(entry: *mut ()) -> *mut iommu_domain {
if xa_pointer_tag(entry) == 0 {
return xa_untag_pointer(entry) as *mut iommu_domain;
}
unsafe { (*(xa_untag_pointer(entry) as *mut iommu_attach_handle)).domain }
}

#[unsafe(no_mangle)]
static mut iommu_probe_device_lock: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

fn __iommu_probe_device(dev: *mut device, group_list: *mut list_head) -> i32 {
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v5: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v5) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut gdev: core::mem::MaybeUninit<*mut group_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut group_device = core::ptr::addr_of_mut!(gdev) as *mut *mut group_device;
let mut group: core::mem::MaybeUninit<*mut iommu_group> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut iommu_group = core::ptr::addr_of_mut!(group) as *mut *mut iommu_group;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
// Serialise to avoid races between IOMMU drivers registering in
// parallel and/or the "replay" calls from ACPI/OF code via client
// driver probe. Once the latter have been cleaned up we should
// probably be able to use device_lock() here to minimise the scope,
// but for now enforcing a simple global ordering is fine.
lockdep_assert_held!(core::ptr::addr_of_mut!(iommu_probe_device_lock));
// Device is probed already if in a group
if !unsafe { (*dev).iommu_group }.is_null() {
return 0;
} else {
*__slate_slot_9 = iommu_init_device(dev);
if *__slate_slot_9 != 0 {
return *__slate_slot_9;
} else {
// And if we do now see any replay calls, they would indicate someone
// misusing the dma_configure path outside bus code.
if !unsafe { (*dev).driver }.is_null() {
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"%s %s: late IOMMU probe at driver bind, something fishy here!\n".as_ptr() as *const u8, dev_driver_string(dev as *const device), dev_name(dev as *const device));
*__slate_slot_5 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string___iommu_probe_device_14, sym __slate_asm_string___iommu_probe_device_15, const 654 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_3 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_5).unwrap()(*__slate_slot_3, dev_driver_string(dev as *const device), dev_name(dev as *const device));
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
}
*__slate_slot_11 = (*dev).iommu_group;
*__slate_slot_10 = iommu_group_alloc_device(*__slate_slot_11, dev);
mutex_lock(core::ptr::addr_of_mut!((*(*__slate_slot_11)).mutex));
if IS_ERR(*__slate_slot_10 as *const ()) {
*__slate_slot_9 = PTR_ERR(*__slate_slot_10 as *const ()) as i32;
} else {
// The gdev must be in the list before calling
// iommu_setup_default_domain()
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_10)).list), core::ptr::addr_of_mut!((*(*__slate_slot_11)).devices));
let __ret_warn_on: i32 = (!unsafe { (*(*__slate_slot_11)).default_domain }.is_null() && unsafe { (*(*__slate_slot_11)).domain }.is_null()) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("705: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 705b - ., 3; .popsection\n# {0}", const 705 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_probe_device_12, sym __slate_asm_string___iommu_probe_device_13, const 669 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("706: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 706b - ., 4; .popsection\n# {0}", const 706 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if !unsafe { (*(*__slate_slot_11)).default_domain }.is_null() {
iommu_create_device_direct_mappings((*(*__slate_slot_11)).default_domain, dev);
}
'__join_1: {
if !unsafe { (*(*__slate_slot_11)).domain }.is_null() {
*__slate_slot_9 = __iommu_device_set_domain(*__slate_slot_11, dev, (*(*__slate_slot_11)).domain, core::ptr::null_mut::<iommu_domain>(), 0u32);
if *__slate_slot_9 != 0 {
break '__join_1;
}
} else {
if unsafe { (*(*__slate_slot_11)).default_domain }.is_null() && group_list.is_null() {
*__slate_slot_9 = iommu_setup_default_domain(*__slate_slot_11, 0i32);
if *__slate_slot_9 != 0 {
break '__join_1;
}
} else {
if unsafe { (*(*__slate_slot_11)).default_domain }.is_null() {
// With a group_list argument we defer the default_domain setup
// to the caller by providing a de-duplicated list of groups
// that need further setup.
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_11)).entry) as *const list_head) != 0 {
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_11)).entry), group_list);
}
}
}
}
if !unsafe { (*(*__slate_slot_11)).default_domain }.is_null() {
iommu_setup_dma_ops(dev, (*(*__slate_slot_11)).default_domain);
}
mutex_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_11)).mutex));
return 0;
}
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_10)).list));
__iommu_group_free_device(*__slate_slot_11, *__slate_slot_10);
}
iommu_deinit_device(dev);
mutex_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_11)).mutex));
iommu_group_put(*__slate_slot_11);
return *__slate_slot_9;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_704: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn iommu_probe_device(dev: *mut device) -> i32 {
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!(iommu_probe_device_lock)) };
ret = __iommu_probe_device(dev, core::ptr::null_mut::<list_head>());
unsafe { mutex_unlock(core::ptr::addr_of_mut!(iommu_probe_device_lock)) };
if ret != 0 {
return ret;
}
let ops: *const iommu_ops = dev_iommu_ops(dev);
if (unsafe { (*ops).probe_finalize }) != None {
unsafe { unsafe { (*ops).probe_finalize }.unwrap()(dev) };
}
0
}

fn __iommu_group_free_device(group: *mut iommu_group, grp_dev: *mut group_device) {
let dev: *mut device = unsafe { (*grp_dev).dev };
unsafe { sysfs_remove_link((*group).devices_kobj, (*grp_dev).name as *const u8) };
unsafe { sysfs_remove_link(core::ptr::addr_of_mut!((*dev).kobj), c"iommu_group".as_ptr() as *const u8) };
trace_remove_device_from_group(unsafe { (*group).id }, dev);
// If the group has become empty then ownership must have been
// released, and the current domain must be set back to NULL or
// the default domain.
if list_empty((unsafe { core::ptr::addr_of_mut!((*group).devices) }) as *const list_head) != 0 {
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*group).owner_cnt }) != 0 || (unsafe { (*group).domain }) != unsafe { (*group).default_domain }) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("707: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 707b - ., 3; .popsection\n# {0}", const 707 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_group_free_device_1, sym __slate_asm_string___iommu_group_free_device_2, const 744 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("708: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 708b - ., 4; .popsection\n# {0}", const 708 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
unsafe { kfree((*grp_dev).name as *const ()) };
unsafe { kfree(grp_dev as *const ()) };
}

/// Remove the iommu_group from the struct device.
fn __iommu_group_remove_device(dev: *mut device) {
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
let mut device: *mut group_device = unsafe { core::mem::zeroed() };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
device = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(device, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
if (unsafe { (*device).dev }) == dev {
list_del(unsafe { core::ptr::addr_of_mut!((*device).list) });
__iommu_group_free_device(group, device);
if dev_has_iommu(dev) {
iommu_deinit_device(dev);
} else {
unsafe {
(*dev).iommu_group = core::ptr::null_mut::<iommu_group>();
}
}
break;
}
device = list_next_entry__4!(device, list);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
// Pairs with the get in iommu_init_device() or
// iommu_group_add_device()
iommu_group_put(group);
}

fn iommu_release_device(dev: *mut device) {
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
if !group.is_null() {
__iommu_group_remove_device(dev);
}
// Free any fwspec if no iommu_driver was ever attached
if !unsafe { (*dev).iommu }.is_null() {
dev_iommu_free(dev);
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn iommu_set_def_domain_type(str: *mut u8) -> i32 {
let mut pt: bool = false;
let ret: i32 = unsafe { kstrtobool(str as *const u8, core::ptr::addr_of_mut!(pt)) };
if ret != 0 {
return ret;
}
if pt {
iommu_set_default_passthrough(true);
} else {
iommu_set_default_translated(true);
}
0
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_iommu_set_def_domain_type: [u8; 18] = [105, 111, 109, 109, 117, 46, 112, 97, 115, 115, 116, 104, 114, 111, 117, 103, 104, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_iommu_set_def_domain_type: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_iommu_set_def_domain_type) as *const u8 };
__t0.setup_func = Some(iommu_set_def_domain_type);
__t0.early = 1i32;
__t0
};

#[unsafe(link_section = ".init.text")]
extern "C" fn iommu_dma_setup(str: *mut u8) -> i32 {
let ret: i32 = unsafe { kstrtobool(str as *const u8, core::ptr::addr_of_mut!(iommu_dma_strict)) };
if ret == 0 {
let __v0: u32 = unsafe { iommu_cmd_line };
unsafe {
iommu_cmd_line = ((__v0 as u64) | BIT!(1i32)) as u32;
}
}
ret
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_iommu_dma_setup: [u8; 13] = [105, 111, 109, 109, 117, 46, 115, 116, 114, 105, 99, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_iommu_dma_setup: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_iommu_dma_setup) as *const u8 };
__t0.setup_func = Some(iommu_dma_setup);
__t0.early = 1i32;
__t0
};

#[unsafe(no_mangle)]
extern "C" fn iommu_set_dma_strict() {
unsafe {
iommu_dma_strict = true;
}
if (unsafe { iommu_def_domain_type }) == 11u32 {
unsafe {
iommu_def_domain_type = 3u32;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_attr_show")]
extern "C" fn iommu_group_attr_show(kobj: *mut kobject, __attr: *mut attribute, buf: *mut u8) -> i64 {
let attr: *mut iommu_group_attribute = to_iommu_group_attr!(__attr);
let group: *mut iommu_group = to_iommu_group!(kobj);
let mut ret: i64 = 5i32.wrapping_neg() as i64;
if (unsafe { (*attr).show }) != None {
ret = unsafe { unsafe { (*attr).show }.unwrap()(group, buf) };
}
ret
}

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_attr_store")]
extern "C" fn iommu_group_attr_store(kobj: *mut kobject, __attr: *mut attribute, buf: *const u8, count: u64) -> i64 {
let attr: *mut iommu_group_attribute = to_iommu_group_attr!(__attr);
let group: *mut iommu_group = to_iommu_group!(kobj);
let mut ret: i64 = 5i32.wrapping_neg() as i64;
if (unsafe { (*attr).store }) != None {
ret = unsafe { unsafe { (*attr).store }.unwrap()(group, buf, count) };
}
ret
}

static iommu_group_sysfs_ops: __SlateConst<sysfs_ops> = __SlateConst(sysfs_ops { show: Some(iommu_group_attr_show), store: Some(iommu_group_attr_store) });

fn iommu_group_create_file(group: *mut iommu_group, attr: *mut iommu_group_attribute) -> i32 {
sysfs_create_file(unsafe { core::ptr::addr_of_mut!((*group).kobj) }, (unsafe { core::ptr::addr_of_mut!((*attr).attr) }) as *const attribute)
}

fn iommu_group_remove_file(group: *mut iommu_group, attr: *mut iommu_group_attribute) {
sysfs_remove_file(unsafe { core::ptr::addr_of_mut!((*group).kobj) }, (unsafe { core::ptr::addr_of_mut!((*attr).attr) }) as *const attribute);
}

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_show_name")]
extern "C" fn iommu_group_show_name(group: *mut iommu_group, buf: *mut u8) -> i64 {
(unsafe { sysfs_emit(buf, c"%s\n".as_ptr() as *const u8, (*group).name) }) as i64
}

// iommu_insert_resv_region - Insert a new region in the
// list of reserved regions.
// @new: new region to insert
// @regions: list of regions
//
// Elements are sorted by start address and overlapping segments
// of the same type are merged.
fn iommu_insert_resv_region(new: *mut iommu_resv_region, regions: *mut list_head) -> i32 {
let mut __UNIQUE_ID_y__710: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__709: u64 = unsafe { core::mem::zeroed() };
let mut iter_end: u64 = unsafe { core::mem::zeroed() };
let mut top_end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(top_end) as *mut u64;
let mut stack: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut list_head = core::ptr::addr_of_mut!(stack) as *mut list_head;
let mut top: core::mem::MaybeUninit<*mut iommu_resv_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut iommu_resv_region = core::ptr::addr_of_mut!(top) as *mut *mut iommu_resv_region;
let mut nr: core::mem::MaybeUninit<*mut iommu_resv_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut iommu_resv_region = core::ptr::addr_of_mut!(nr) as *mut *mut iommu_resv_region;
let mut tmp: core::mem::MaybeUninit<*mut iommu_resv_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut iommu_resv_region = core::ptr::addr_of_mut!(tmp) as *mut *mut iommu_resv_region;
let mut iter: core::mem::MaybeUninit<*mut iommu_resv_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut iommu_resv_region = core::ptr::addr_of_mut!(iter) as *mut *mut iommu_resv_region;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_7, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_7), prev: core::ptr::addr_of_mut!(*__slate_slot_7) });
*__slate_slot_9 = iommu_alloc_resv_region((*new).start, (*new).length, (*new).prot, (*new).r#type, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if (*__slate_slot_9).is_null() {
return 12i32.wrapping_neg();
} else {
// First add the new element based on start address sorting
*__slate_slot_11 = list_first_entry__7!(regions);
while list_entry_is_head!(*__slate_slot_11, regions, list) == 0 {
if (*(*__slate_slot_9)).start < (*(*__slate_slot_11)).start || (*(*__slate_slot_9)).start == (*(*__slate_slot_11)).start && ((*(*__slate_slot_9)).r#type as u32) <= ((*(*__slate_slot_11)).r#type as u32) {
break;
} else {
*__slate_slot_11 = list_next_entry__6!(*__slate_slot_11, list);
}
}
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_9)).list), core::ptr::addr_of_mut!((*(*__slate_slot_11)).list));
// Merge overlapping segments of type nr->type in @regions, if any
*__slate_slot_11 = list_first_entry__7!(regions);
let __v6: *mut iommu_resv_region = list_next_entry__6!(*__slate_slot_11, list);
*__slate_slot_10 = __v6;
while list_entry_is_head!(*__slate_slot_11, regions, list) == 0 {
'__join_2: {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
iter_end = unsafe { (*(*__slate_slot_11)).start }.wrapping_add((*(*__slate_slot_11)).length).wrapping_sub(1);
// no merge needed on elements of different types than @new
if ((*(*__slate_slot_11)).r#type as u32) != ((*new).r#type as u32) {
list_move_tail(core::ptr::addr_of_mut!((*(*__slate_slot_11)).list), core::ptr::addr_of_mut!(*__slate_slot_7));
} else {
// look for the last stack element of same type as @iter
*__slate_slot_8 = list_last_entry__3!(core::ptr::addr_of_mut!(*__slate_slot_7));
'__join_7: {
while list_entry_is_head!(*__slate_slot_8, core::ptr::addr_of_mut!(*__slate_slot_7), list) == 0 {
if ((*(*__slate_slot_8)).r#type as u32) == ((*(*__slate_slot_11)).r#type as u32) {
break '__join_7;
} else {
*__slate_slot_8 = list_prev_entry__3!(*__slate_slot_8, list);
}
}
list_move_tail(core::ptr::addr_of_mut!((*(*__slate_slot_11)).list), core::ptr::addr_of_mut!(*__slate_slot_7));
break '__join_2;
}
*__slate_slot_5 = unsafe { (*(*__slate_slot_8)).start }.wrapping_add((*(*__slate_slot_8)).length).wrapping_sub(1);
if (*(*__slate_slot_11)).start > (*__slate_slot_5).wrapping_add(1) {
list_move_tail(core::ptr::addr_of_mut!((*(*__slate_slot_11)).list), core::ptr::addr_of_mut!(*__slate_slot_7));
} else {
__UNIQUE_ID_x__709 = *__slate_slot_5;
__UNIQUE_ID_y__710 = iter_end;
while false {
}
let __v3: u64 = if __UNIQUE_ID_x__709 > __UNIQUE_ID_y__710 { __UNIQUE_ID_x__709 } else { __UNIQUE_ID_y__710 };
(*(*__slate_slot_8)).length = __v3.wrapping_sub((*(*__slate_slot_8)).start).wrapping_add(1);
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_11)).list));
kfree(*__slate_slot_11 as *const ());
}
}
}
*__slate_slot_11 = *__slate_slot_10;
let __v0: *mut iommu_resv_region = list_next_entry__6!(*__slate_slot_10, list);
*__slate_slot_10 = __v0;
}
list_splice(&(*(core::ptr::addr_of_mut!(*__slate_slot_7) as *const list_head)), regions);
return 0;
}
}
}

fn iommu_insert_device_resv_regions(dev_resv_regions: *mut list_head, group_resv_regions: *mut list_head) -> i32 {
let mut entry: *mut iommu_resv_region = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
entry = list_first_entry__7!(dev_resv_regions);
while list_entry_is_head!(entry, dev_resv_regions, list) == 0 {
ret = iommu_insert_resv_region(entry, group_resv_regions);
if ret != 0 {
break;
}
entry = list_next_entry__6!(entry, list);
}
ret
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_get_group_resv_regions")]
extern "C" fn iommu_get_group_resv_regions(group: *mut iommu_group, head: *mut list_head) -> i32 {
let mut device: *mut group_device = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
device = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(device, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
let mut dev_resv_regions: list_head = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(dev_resv_regions) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
// Non-API groups still expose reserved_regions in sysfs,
// so filter out calls that get here that way.
if !dev_has_iommu(unsafe { (*device).dev }) {
break;
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!(dev_resv_regions));
iommu_get_resv_regions(unsafe { (*device).dev }, core::ptr::addr_of_mut!(dev_resv_regions));
ret = iommu_insert_device_resv_regions(core::ptr::addr_of_mut!(dev_resv_regions), head);
iommu_put_resv_regions(unsafe { (*device).dev }, core::ptr::addr_of_mut!(dev_resv_regions));
if ret != 0 {
break;
}
device = list_next_entry__4!(device, list);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_get_group_resv_regions_712: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group, *mut list_head) -> i32>, *mut ()>(Some(iommu_get_group_resv_regions)) };

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_show_resv_regions")]
extern "C" fn iommu_group_show_resv_regions(group: *mut iommu_group, buf: *mut u8) -> i64 {
let mut region: *mut iommu_resv_region = unsafe { core::mem::zeroed() };
let mut next: *mut iommu_resv_region = unsafe { core::mem::zeroed() };
let mut group_resv_regions: list_head = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(group_resv_regions) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut offset: i32 = 0;
INIT_LIST_HEAD(core::ptr::addr_of_mut!(group_resv_regions));
iommu_get_group_resv_regions(group, core::ptr::addr_of_mut!(group_resv_regions));
region = list_first_entry__7!(core::ptr::addr_of_mut!(group_resv_regions));
next = list_next_entry__6!(region, list);
while list_entry_is_head!(region, core::ptr::addr_of_mut!(group_resv_regions), list) == 0 {
let __v3: i32 = unsafe { sysfs_emit_at(buf, offset, c"0x%016llx 0x%016llx %s\n".as_ptr() as *const u8, (*region).start as i64, unsafe { (*region).start }.wrapping_add((*region).length).wrapping_sub(1) as i64, iommu_group_resv_type_string.0[((*region).r#type as u32) as usize]) };
offset = offset.wrapping_add(__v3);
unsafe { kfree(region as *const ()) };
region = next;
next = list_next_entry__6!(next, list);
}
offset as i64
}

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_show_type")]
extern "C" fn iommu_group_show_type(group: *mut iommu_group, buf: *mut u8) -> i64 {
let mut r#type: *mut u8 = c"unknown".as_ptr() as *mut u8;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
if !unsafe { (*group).default_domain }.is_null() {
match unsafe { (*(*group).default_domain).r#type } {
0 => {
r#type = c"blocked".as_ptr() as *mut u8;
}
4 => {
r#type = c"identity".as_ptr() as *mut u8;
}
1 => {
r#type = c"unmanaged".as_ptr() as *mut u8;
}
3 => {
r#type = c"DMA".as_ptr() as *mut u8;
}
11 => {
r#type = c"DMA-FQ".as_ptr() as *mut u8;
}
_ => {
}
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
(unsafe { sysfs_emit(buf, c"%s\n".as_ptr() as *const u8, r#type) }) as i64
}

static mut iommu_group_attr_name: iommu_group_attribute = iommu_group_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"name\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_iommu_c_1029>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1029_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1029_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1029_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1029_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1029_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, show: Some(iommu_group_show_name), store: None };

static mut iommu_group_attr_reserved_regions: iommu_group_attribute = iommu_group_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"reserved_regions\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_iommu_c_1031>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1031_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1031_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1031_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1031_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1031_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, show: Some(iommu_group_show_resv_regions), store: None };

static mut iommu_group_attr_type: iommu_group_attribute = iommu_group_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"type\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_iommu_c_1034>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1034_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1034_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1034_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1034_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_iommu_c_1034_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, show: Some(iommu_group_show_type), store: Some(iommu_group_store_type) };

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_release")]
extern "C" fn iommu_group_release(kobj: *mut kobject) {
let group: *mut iommu_group = to_iommu_group!(kobj);
let mut __v1: i32;
if false {
unsafe { _printk(c"\x017iommu: Releasing group %d\n".as_ptr() as *const u8, (*group).id) };
}
__v1 = 0;
if (unsafe { (*group).iommu_data_release }) != None {
unsafe { unsafe { (*group).iommu_data_release }.unwrap()((*group).iommu_data) };
}
unsafe { ida_free(core::ptr::addr_of_mut!(iommu_group_ida), (*group).id as u32) };
// Domains are free'd by iommu_deinit_device()
let mut __v2: i64;
{
let __ret_warn_on: i32 = !unsafe { (*group).default_domain }.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("713: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 713b - ., 3; .popsection\n# {0}", const 713 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_group_release_3, sym __slate_asm_string_iommu_group_release_4, const 1049 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("714: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 714b - ., 4; .popsection\n# {0}", const 714 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
let mut __v7: i64;
let __ret_warn_on: i32 = !unsafe { (*group).blocking_domain }.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("715: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 715b - ., 3; .popsection\n# {0}", const 715 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_group_release_8, sym __slate_asm_string_iommu_group_release_9, const 1050 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("716: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 716b - ., 4; .popsection\n# {0}", const 716 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
unsafe { kfree((*group).name as *const ()) };
unsafe { kfree(group as *const ()) };
}

static iommu_group_ktype: __SlateConst<kobj_type> = __SlateConst({
let mut __t0: kobj_type = unsafe { core::mem::zeroed() };
__t0.release = Some(iommu_group_release);
__t0.sysfs_ops = unsafe { core::ptr::addr_of!(iommu_group_sysfs_ops.0) };
__t0
});

/// iommu_group_alloc - Allocate a new group
///
/// This function is called by an iommu driver to allocate a new iommu
/// group.  The iommu group represents the minimum granularity of the iommu.
/// Upon successful return, the caller holds a reference to the supplied
/// group in order to hold the group until devices are added.  Use
/// iommu_group_put() to release this extra reference count, allowing the
/// group to be automatically reclaimed once it has no devices or external
/// references.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_alloc")]
extern "C" fn iommu_group_alloc() -> *mut iommu_group {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let mut __v0: *mut iommu_group;
let __obj_size: u64 = size_mul(core::mem::size_of::<iommu_group>() as u64, 1u64);
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
__v0 = __v1 as *mut iommu_group;
group = __v0;
if group.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut iommu_group;
}
unsafe {
(*group).kobj.kset = iommu_group_kset;
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*group).mutex) }, c"&group->mutex".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__1082) });
if true {
break;
}
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*group).devices) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*group).entry) });
xa_init(unsafe { core::ptr::addr_of_mut!((*group).pasid_array) });
ret = ida_alloc(unsafe { core::ptr::addr_of_mut!(iommu_group_ida) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if ret < 0 {
unsafe { kfree(group as *const ()) };
return ERR_PTR(ret as i64) as *mut iommu_group;
}
unsafe {
(*group).id = ret;
}
ret = unsafe { kobject_init_and_add(core::ptr::addr_of_mut!((*group).kobj), core::ptr::addr_of!(iommu_group_ktype.0), core::ptr::null_mut::<kobject>(), c"%d".as_ptr() as *const u8, (*group).id) };
if ret != 0 {
unsafe { kobject_put(core::ptr::addr_of_mut!((*group).kobj)) };
return ERR_PTR(ret as i64) as *mut iommu_group;
}
unsafe {
(*group).devices_kobj = kobject_create_and_add(c"devices".as_ptr() as *const u8, core::ptr::addr_of_mut!((*group).kobj));
}
if unsafe { (*group).devices_kobj }.is_null() {
unsafe { kobject_put(core::ptr::addr_of_mut!((*group).kobj)) }; // triggers .release & free
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut iommu_group;
}
// The devices_kobj holds a reference on the group kobject, so
// as long as that exists so will the group.  We can therefore
// use the devices_kobj for reference counting.
unsafe { kobject_put(core::ptr::addr_of_mut!((*group).kobj)) };
ret = iommu_group_create_file(group, unsafe { core::ptr::addr_of_mut!(iommu_group_attr_reserved_regions) });
if ret != 0 {
unsafe { kobject_put((*group).devices_kobj) };
return ERR_PTR(ret as i64) as *mut iommu_group;
}
ret = iommu_group_create_file(group, unsafe { core::ptr::addr_of_mut!(iommu_group_attr_type) });
if ret != 0 {
unsafe { kobject_put((*group).devices_kobj) };
return ERR_PTR(ret as i64) as *mut iommu_group;
}
let mut __v7: i32;
if false {
unsafe { _printk(c"\x017iommu: Allocated group %d\n".as_ptr() as *const u8, (*group).id) };
}
__v7 = 0;
group
}

static mut __key__1082: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_alloc_717: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> *mut iommu_group>, *mut ()>(Some(iommu_group_alloc)) };

/// iommu_group_get_iommudata - retrieve iommu_data registered for a group
/// @group: the group
///
/// iommu drivers can store data in the group for use when doing iommu
/// operations.  This function provides a way to retrieve it.  Caller
/// should hold a group reference.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_get_iommudata")]
extern "C" fn iommu_group_get_iommudata(group: *mut iommu_group) -> *mut () {
unsafe { (*group).iommu_data }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_get_iommudata_718: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group) -> *mut ()>, *mut ()>(Some(iommu_group_get_iommudata)) };

/// iommu_group_set_iommudata - set iommu_data for a group
/// @group: the group
/// @iommu_data: new data
/// @release: release function for iommu_data
///
/// iommu drivers can store data in the group for use when doing iommu
/// operations.  This function provides a way to set the data after
/// the group has been allocated.  Caller should hold a group reference.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_set_iommudata")]
extern "C" fn iommu_group_set_iommudata(group: *mut iommu_group, iommu_data: *mut (), release: Option<unsafe extern "C" fn(*mut ())>) {
unsafe {
(*group).iommu_data = iommu_data;
}
unsafe {
(*group).iommu_data_release = release;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_set_iommudata_719: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group, *mut (), Option<unsafe extern "C" fn(*mut ())>)>, *mut ()>(Some(iommu_group_set_iommudata)) };

/// iommu_group_set_name - set name for a group
/// @group: the group
/// @name: name
///
/// Allow iommu driver to set a name for a group.  When set it will
/// appear in a name attribute file under the group in sysfs.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_set_name")]
extern "C" fn iommu_group_set_name(group: *mut iommu_group, name: *const u8) -> i32 {
let mut ret: i32 = 0;
if !unsafe { (*group).name }.is_null() {
iommu_group_remove_file(group, unsafe { core::ptr::addr_of_mut!(iommu_group_attr_name) });
unsafe { kfree((*group).name as *const ()) };
unsafe {
(*group).name = core::ptr::null_mut::<u8>();
}
if name.is_null() {
return 0;
}
}
unsafe {
(*group).name = kstrdup(name, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
}
if unsafe { (*group).name }.is_null() {
return 12i32.wrapping_neg();
}
ret = iommu_group_create_file(group, unsafe { core::ptr::addr_of_mut!(iommu_group_attr_name) });
if ret != 0 {
unsafe { kfree((*group).name as *const ()) };
unsafe {
(*group).name = core::ptr::null_mut::<u8>();
}
return ret;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_set_name_720: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group, *const u8) -> i32>, *mut ()>(Some(iommu_group_set_name)) };

/// This is undone by __iommu_group_free_device()
fn iommu_group_alloc_device(group: *mut iommu_group, dev: *mut device) -> *mut group_device {
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut device: core::mem::MaybeUninit<*mut group_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut group_device = core::ptr::addr_of_mut!(device) as *mut *mut group_device;
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
i = 0;
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
__obj_size = size_mul(core::mem::size_of::<group_device>() as u64, 1u64);
let __slate_auto_bytes_3 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_3 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
alloc_tag_restore!();
} else {
*__slate_slot_3 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
}
*__slate_slot_8 = *__slate_slot_3 as *mut group_device;
if (*__slate_slot_8).is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut group_device;
} else {
(*(*__slate_slot_8)).dev = dev;
*__slate_slot_10 = sysfs_create_link(core::ptr::addr_of_mut!((*dev).kobj), core::ptr::addr_of_mut!((*group).kobj), c"iommu_group".as_ptr() as *const u8);
if *__slate_slot_10 == 0 {
(*(*__slate_slot_8)).name = kasprintf(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), c"%s".as_ptr() as *const u8, kobject_name(core::ptr::addr_of_mut!((*dev).kobj) as *const kobject));
'__join_3: {
'__join_11: {
'__join_4: {
loop {
if unsafe { (*(*__slate_slot_8)).name }.is_null() {
break '__join_11;
} else {
*__slate_slot_10 = sysfs_create_link_nowarn((*group).devices_kobj, core::ptr::addr_of_mut!((*dev).kobj), (*(*__slate_slot_8)).name as *const u8);
if *__slate_slot_10 != 0 {
if *__slate_slot_10 == 17i32.wrapping_neg() && i >= 0 {
// Account for the slim chance of collision
// and append an instance to the name.
kfree((*(*__slate_slot_8)).name as *const ());
let __v1: i32 = i;
i = __v1.wrapping_add(1);
(*(*__slate_slot_8)).name = kasprintf(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), c"%s.%d".as_ptr() as *const u8, kobject_name(core::ptr::addr_of_mut!((*dev).kobj) as *const kobject), __v1);
} else {
break '__join_4;
}
} else {
break;
}
}
}
trace_add_device_to_group((*group).id, dev);
while false {
}
_dev_info(dev as *const device, c"Adding to iommu group %d\n".as_ptr() as *const u8, (*group).id);
return *__slate_slot_8;
}
kfree((*(*__slate_slot_8)).name as *const ());
break '__join_3;
}
*__slate_slot_10 = 12i32.wrapping_neg();
}
sysfs_remove_link(core::ptr::addr_of_mut!((*dev).kobj), c"iommu_group".as_ptr() as *const u8);
}
kfree(*__slate_slot_8 as *const ());
while false {
}
_dev_err(dev as *const device, c"Failed to add to iommu group %d: %d\n".as_ptr() as *const u8, (*group).id, *__slate_slot_10);
return ERR_PTR(*__slate_slot_10 as i64) as *mut group_device;
}
}
}

/// iommu_group_add_device - add a device to an iommu group
/// @group: the group into which to add the device (reference should be held)
/// @dev: the device
///
/// This function is called by an iommu driver to add a device into a
/// group.  Adding a device increments the group reference count.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_add_device")]
extern "C" fn iommu_group_add_device(group: *mut iommu_group, dev: *mut device) -> i32 {
let mut gdev: *mut group_device = unsafe { core::mem::zeroed() };
gdev = iommu_group_alloc_device(group, dev);
if IS_ERR(gdev as *const ()) {
return PTR_ERR(gdev as *const ()) as i32;
}
iommu_group_ref_get(group);
unsafe {
(*dev).iommu_group = group;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
list_add_tail(unsafe { core::ptr::addr_of_mut!((*gdev).list) }, unsafe { core::ptr::addr_of_mut!((*group).devices) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_add_device_723: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group, *mut device) -> i32>, *mut ()>(Some(iommu_group_add_device)) };

/// iommu_group_remove_device - remove a device from its current group
/// @dev: device to be removed
///
/// This function is called by an iommu driver to remove the device from
/// its current group.  This decrements the iommu group reference count.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_remove_device")]
extern "C" fn iommu_group_remove_device(dev: *mut device) {
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
if group.is_null() {
return;
}
while false {
}
unsafe { _dev_info(dev as *const device, c"Removing from iommu group %d\n".as_ptr() as *const u8, (*group).id) };
__iommu_group_remove_device(dev);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_remove_device_724: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device)>, *mut ()>(Some(iommu_group_remove_device)) };

fn iommu_group_first_dev(group: *mut iommu_group) -> *mut device {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
unsafe { (*list_first_entry__64!(core::ptr::addr_of_mut!((*group).devices), group_device)).dev }
}

/// iommu_group_for_each_dev - iterate over each device in the group
/// @group: the group
/// @data: caller opaque data to be passed to callback function
/// @fn: caller supplied callback function
///
/// This function is called by group users to iterate over group devices.
/// Callers should hold a reference count to the group during callback.
/// The group->mutex is held across callbacks, which will block calls to
/// iommu_group_add/remove_device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_for_each_dev")]
extern "C" fn iommu_group_for_each_dev(group: *mut iommu_group, data: *mut (), r#fn: Option<unsafe extern "C" fn(*mut device, *mut ()) -> i32>) -> i32 {
let mut device: *mut group_device = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
device = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(device, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
ret = unsafe { r#fn.unwrap()((*device).dev, data) };
if ret != 0 {
break;
}
device = list_next_entry__4!(device, list);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_for_each_dev_725: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group, *mut (), Option<unsafe extern "C" fn(*mut device, *mut ()) -> i32>) -> i32>, *mut ()>(Some(iommu_group_for_each_dev)) };

/// iommu_group_get - Return the group for a device and increment reference
/// @dev: get the group that this device belongs to
///
/// This function is called by iommu drivers and users to get the group
/// for the specified device.  If found, the group is returned and the group
/// reference in incremented, else NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_get")]
extern "C" fn iommu_group_get(dev: *mut device) -> *mut iommu_group {
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
if !group.is_null() {
unsafe { kobject_get((*group).devices_kobj) };
}
group
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_get_726: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device) -> *mut iommu_group>, *mut ()>(Some(iommu_group_get)) };

/// iommu_group_ref_get - Increment reference on a group
/// @group: the group to use, must not be NULL
///
/// This function is called by iommu drivers to take additional references on an
/// existing group.  Returns the given group for convenience.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_ref_get")]
extern "C" fn iommu_group_ref_get(group: *mut iommu_group) -> *mut iommu_group {
unsafe { kobject_get((*group).devices_kobj) };
group
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_ref_get_727: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group) -> *mut iommu_group>, *mut ()>(Some(iommu_group_ref_get)) };

/// iommu_group_put - Decrement group reference
/// @group: the group to use
///
/// This function is called by iommu drivers and users to release the
/// iommu group.  Once the reference count is zero, the group is released.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_put")]
extern "C" fn iommu_group_put(group: *mut iommu_group) {
if !group.is_null() {
unsafe { kobject_put((*group).devices_kobj) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_put_728: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group)>, *mut ()>(Some(iommu_group_put)) };

/// iommu_group_id - Return ID for a group
/// @group: the group to ID
///
/// Return the unique ID for the group matching the sysfs group number.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_id")]
extern "C" fn iommu_group_id(group: *mut iommu_group) -> i32 {
unsafe { (*group).id }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_id_729: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group) -> i32>, *mut ()>(Some(iommu_group_id)) };

// To consider a PCI device isolated, we require ACS to support Source
// Validation, Request Redirection, Completer Redirection, and Upstream
// Forwarding.  This effectively means that devices cannot spoof their
// requester ID, requests and completions cannot be redirected, and all
// transactions are forwarded upstream, even as it passes through a
// bridge where the target device is downstream.
/// For multifunction devices which are not isolated from each other, find
/// all the other non-isolated functions and look for existing groups.  For
/// each function, we also need to look for aliases to or from other devices
/// that may already have a group.
fn get_pci_function_alias_group(pdev: *mut pci_dev, devfns: *mut u64) -> *mut iommu_group {
let mut tmp: *mut pci_dev = core::ptr::null_mut::<pci_dev>();
let __v0: bool = ((unsafe { (*pdev).__slate_bits_3.__get_multifunction() }) as i32) == 0 || unsafe { pci_acs_enabled(pdev, 29u16) };
if __v0 {
return core::ptr::null_mut::<iommu_group>();
}
loop {
let __v3: *mut pci_dev = unsafe { pci_get_device(4294967295u32, 4294967295u32, tmp) };
tmp = __v3;
if __v3.is_null() {
break;
}
let __v1: bool = tmp == pdev || (unsafe { (*tmp).bus }) != unsafe { (*pdev).bus } || PCI_SLOT__2!(unsafe { (*tmp).devfn }) != PCI_SLOT__2!(unsafe { (*pdev).devfn }) || unsafe { pci_acs_enabled(tmp, 29u16) };
if !__v1 {
let group: *mut iommu_group = get_pci_alias_group(tmp, devfns);
if !group.is_null() {
unsafe { pci_dev_put(tmp) };
return group;
}
}
}
core::ptr::null_mut::<iommu_group>()
}

/// Look for aliases to or from the given device for existing groups. DMA
/// aliases are only supported on the same bus, therefore the search
/// space is quite small (especially since we're really only looking at pcie
/// device, and therefore only expect multiple slots on the root complex or
/// downstream switch ports).  It's conceivable though that a pair of
/// multifunction devices could have aliases between them that would cause a
/// loop.  To prevent this, we use a bitmap to track where we've been.
fn get_pci_alias_group(pdev: *mut pci_dev, devfns: *mut u64) -> *mut iommu_group {
let mut tmp: *mut pci_dev = core::ptr::null_mut::<pci_dev>();
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
if test_and_set_bit(((unsafe { (*pdev).devfn }) & 255) as i64, devfns as *mut u64) {
return core::ptr::null_mut::<iommu_group>();
}
group = iommu_group_get(unsafe { core::ptr::addr_of_mut!((*pdev).dev) });
if !group.is_null() {
return group;
}
loop {
let __v1: *mut pci_dev = unsafe { pci_get_device(4294967295u32, 4294967295u32, tmp) };
tmp = __v1;
if __v1.is_null() {
break;
}
if !(tmp == pdev || (unsafe { (*tmp).bus }) != unsafe { (*pdev).bus }) {
// We alias them or they alias us
if unsafe { pci_devs_are_dma_aliases(pdev, tmp) } {
group = get_pci_alias_group(tmp, devfns);
if !group.is_null() {
unsafe { pci_dev_put(tmp) };
return group;
}
group = get_pci_function_alias_group(tmp, devfns);
if !group.is_null() {
unsafe { pci_dev_put(tmp) };
return group;
}
}
}
}
core::ptr::null_mut::<iommu_group>()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct group_for_pci_data {
    pdev: *mut pci_dev,
    group: *mut iommu_group,
}


/// DMA alias iterator callback, return the last seen device.  Stop and return
/// the IOMMU group if we find one along the way.
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.get_pci_alias_or_group")]
extern "C" fn get_pci_alias_or_group(pdev: *mut pci_dev, alias: u16, opaque: *mut ()) -> i32 {
unsafe {
(*(opaque as *mut group_for_pci_data)).pdev = pdev;
}
unsafe {
(*(opaque as *mut group_for_pci_data)).group = iommu_group_get(core::ptr::addr_of_mut!((*pdev).dev));
}
!unsafe { (*(opaque as *mut group_for_pci_data)).group }.is_null() as i32
}

/// Generic device_group call-back function. It just allocates one
/// iommu-group per device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.generic_device_group")]
extern "C" fn generic_device_group(dev: *mut device) -> *mut iommu_group {
iommu_group_alloc()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_generic_device_group_730: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device) -> *mut iommu_group>, *mut ()>(Some(generic_device_group)) };

/// Generic device_group call-back function. It just allocates one
/// iommu-group per iommu driver instance shared by every device
/// probed by that iommu driver.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.generic_single_device_group")]
extern "C" fn generic_single_device_group(dev: *mut device) -> *mut iommu_group {
let iommu: *mut iommu_device = unsafe { (*(*dev).iommu).iommu_dev };
if unsafe { (*iommu).singleton_group }.is_null() {
let group: *mut iommu_group = iommu_group_alloc();
if IS_ERR(group as *const ()) {
return group;
}
unsafe {
(*iommu).singleton_group = group;
}
}
iommu_group_ref_get(unsafe { (*iommu).singleton_group })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_generic_single_device_group_731: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device) -> *mut iommu_group>, *mut ()>(Some(generic_single_device_group)) };

/// Use standard PCI bus topology, isolation features, and DMA alias quirks
/// to find or create an IOMMU group for a device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.pci_device_group")]
extern "C" fn pci_device_group(dev: *mut device) -> *mut iommu_group {
let mut pdev: *mut pci_dev = to_pci_dev!(dev);
let mut data: group_for_pci_data = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(data) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut bus: *mut pci_bus = unsafe { core::mem::zeroed() };
let mut group: *mut iommu_group = core::ptr::null_mut::<iommu_group>();
let mut devfns: __SlateAlign16<[u64; 4]> = __SlateAlign16({
let mut __t0: [u64; 4] = unsafe { core::mem::zeroed() };
__t0[0] = 0u64;
__t0
});
let mut __v1: i64;
let __ret_warn_on: i32 = !(!(!dev_is_pci!(dev))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("732: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 732b - ., 3; .popsection\n# {0}", const 732 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_device_group_2, sym __slate_asm_string_pci_device_group_3, const 1631 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("733: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 733b - ., 4; .popsection\n# {0}", const 733 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut iommu_group;
}
// Find the upstream DMA alias for the device.  A device must not
// be aliased due to topology in order to have its own IOMMU group.
// If we find an alias along the way that already belongs to a
// group, use it.
if (unsafe { pci_for_each_dma_alias(pdev, Some(get_pci_alias_or_group), core::ptr::addr_of_mut!(data) as *mut ()) }) != 0 {
return data.group;
}
pdev = data.pdev;
// Continue upstream from the point of minimum IOMMU granularity
// due to aliases to the point where devices are protected from
// peer-to-peer DMA by PCI ACS.  Again, if we find an existing
// group, use it.
bus = unsafe { (*pdev).bus };
while !pci_is_root_bus(bus) {
if !unsafe { (*bus).self_ }.is_null() {
if unsafe { pci_acs_path_enabled((*bus).self_, core::ptr::null_mut::<pci_dev>(), 29u16) } {
break;
}
pdev = unsafe { (*bus).self_ };
group = iommu_group_get(unsafe { core::ptr::addr_of_mut!((*pdev).dev) });
if !group.is_null() {
return group;
}
}
bus = unsafe { (*bus).parent };
}
// Look for existing groups on device aliases.  If we alias another
// device or another device aliases us, use the same group.
group = get_pci_alias_group(pdev, devfns.0.as_mut_ptr() as *mut u64);
if !group.is_null() {
return group;
}
// Look for existing groups on non-isolated functions on the same
// slot and aliases of those funcions, if any.  No need to clear
// the search bitmap, the tested devfns are still valid.
group = get_pci_function_alias_group(pdev, devfns.0.as_mut_ptr() as *mut u64);
if !group.is_null() {
return group;
}
// No shared group found, allocate new
iommu_group_alloc()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_device_group_734: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device) -> *mut iommu_group>, *mut ()>(Some(pci_device_group)) };

/// Get the IOMMU group for device on fsl-mc bus
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.fsl_mc_device_group")]
extern "C" fn fsl_mc_device_group(dev: *mut device) -> *mut iommu_group {
let cont_dev: *mut device = fsl_mc_cont_dev!(dev);
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
group = iommu_group_get(cont_dev);
if group.is_null() {
group = iommu_group_alloc();
}
group
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_fsl_mc_device_group_735: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device) -> *mut iommu_group>, *mut ()>(Some(fsl_mc_device_group)) };

fn __iommu_alloc_identity_domain(dev: *mut device) -> *mut iommu_domain {
let ops: *const iommu_ops = dev_iommu_ops(dev);
let mut domain: *mut iommu_domain = unsafe { core::mem::zeroed() };
if !unsafe { (*ops).identity_domain }.is_null() {
return unsafe { (*ops).identity_domain };
}
if (unsafe { (*ops).domain_alloc_identity }) != None {
domain = unsafe { unsafe { (*ops).domain_alloc_identity }.unwrap()(dev) };
if IS_ERR(domain as *const ()) {
return domain;
}
} else {
return ERR_PTR(95i32.wrapping_neg() as i64) as *mut iommu_domain;
}
iommu_domain_init(domain, 4u32, ops);
domain
}

fn __iommu_group_alloc_default_domain(group: *mut iommu_group, req_type: i32) -> *mut iommu_domain {
let dev: *mut device = iommu_group_first_dev(group);
let mut dom: *mut iommu_domain = unsafe { core::mem::zeroed() };
if !unsafe { (*group).default_domain }.is_null() && (unsafe { (*(*group).default_domain).r#type }) == (req_type as u32) {
return unsafe { (*group).default_domain };
}
// When allocating the DMA API domain assume that the driver is going to
// use PASID and make sure the RID's domain is PASID compatible.
if (req_type as u32) & 1u32 != 0 {
dom = __iommu_paging_domain_alloc_flags(dev, req_type as u32, (if (unsafe { (*(*dev).iommu).max_pasids }) != 0 { 8i32 } else { 0i32 }) as u32);
// If driver does not support PASID feature then
// try to allocate non-PASID domain
if PTR_ERR(dom as *const ()) == (95i32.wrapping_neg() as i64) {
dom = __iommu_paging_domain_alloc_flags(dev, req_type as u32, 0u32);
}
return dom;
}
if (req_type as u32) == 4u32 {
return __iommu_alloc_identity_domain(dev);
}
ERR_PTR(22i32.wrapping_neg() as i64) as *mut iommu_domain
}

/// req_type of 0 means "auto" which means to select a domain based on
/// iommu_def_domain_type or what the driver actually supports.
fn iommu_group_alloc_default_domain(group: *mut iommu_group, req_type: i32) -> *mut iommu_domain {
let ops: *const iommu_ops = dev_iommu_ops(iommu_group_first_dev(group));
let mut dom: *mut iommu_domain = unsafe { core::mem::zeroed() };
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
// Allow legacy drivers to specify the domain that will be the default
// domain. This should always be either an IDENTITY/BLOCKED/PLATFORM
// domain. Do not use in new drivers.
if !unsafe { (*ops).default_domain }.is_null() {
if (req_type as u32) != unsafe { (*(*ops).default_domain).r#type } {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut iommu_domain;
}
return unsafe { (*ops).default_domain };
}
if req_type != 0 {
return __iommu_group_alloc_default_domain(group, req_type);
}
// The driver gave no guidance on what type to use, try the default
dom = __iommu_group_alloc_default_domain(group, (unsafe { iommu_def_domain_type }) as i32);
if !IS_ERR(dom as *const ()) {
return dom;
}
// Otherwise IDENTITY and DMA_FQ defaults will try DMA
if (unsafe { iommu_def_domain_type }) == 3u32 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut iommu_domain;
}
dom = __iommu_group_alloc_default_domain(group, 3i32);
if IS_ERR(dom as *const ()) {
return dom;
}
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014iommu: Failed to allocate default IOMMU domain of type %u for group %s - Falling back to IOMMU_DOMAIN_DMA".as_ptr() as *const u8, iommu_def_domain_type, (*group).name) };
dom
}

#[unsafe(no_mangle)]
extern "C" fn iommu_group_default_domain(group: *mut iommu_group) -> *mut iommu_domain {
unsafe { (*group).default_domain }
}

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.probe_iommu_group")]
extern "C" fn probe_iommu_group(dev: *mut device, data: *mut ()) -> i32 {
let group_list: *mut list_head = data as *mut list_head;
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!(iommu_probe_device_lock)) };
ret = __iommu_probe_device(dev, group_list);
unsafe { mutex_unlock(core::ptr::addr_of_mut!(iommu_probe_device_lock)) };
if ret == 19i32.wrapping_neg() {
ret = 0;
}
ret
}

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_bus_notifier")]
extern "C" fn iommu_bus_notifier(nb: *mut notifier_block, action: u64, data: *mut ()) -> i32 {
let dev: *mut device = data as *mut device;
if action == 0 {
let ret: i32 = iommu_probe_device(dev);
return if ret != 0 { 0 } else { 1 };
} else {
if action == 2 {
iommu_release_device(dev);
return 1;
}
}
0
}

/// Combine the driver's chosen def_domain_type across all the devices in a
/// group. Drivers must give a consistent result.
fn iommu_get_def_domain_type(group: *mut iommu_group, dev: *mut device, cur_type: i32) -> i32 {
let ops: *const iommu_ops = dev_iommu_ops(dev);
let mut r#type: i32 = 0;
if !unsafe { (*ops).default_domain }.is_null() {
// Drivers that declare a global static default_domain will
// always choose that.
r#type = (unsafe { (*(*ops).default_domain).r#type }) as i32;
} else {
if (unsafe { (*ops).def_domain_type }) != None {
r#type = unsafe { unsafe { (*ops).def_domain_type }.unwrap()(dev) };
} else {
return cur_type;
}
}
if r#type == 0 || cur_type == r#type {
return cur_type;
}
if cur_type == 0 {
return r#type;
}
loop {
if __ratelimit__20!(unsafe { core::ptr::addr_of_mut!(_rs__1860) }) != 0 {
while false {
}
unsafe { _dev_err(dev as *const device, c"IOMMU driver error, requesting conflicting def_domain_type, %s and %s, for devices in group %u.\n".as_ptr() as *const u8, iommu_domain_type_str(cur_type as u32), iommu_domain_type_str(r#type as u32), (*group).id) };
}
if true {
break;
}
}
// Try to recover, drivers are allowed to force IDENTITY or DMA, IDENTITY
// takes precedence.
if (r#type as u32) == 4u32 {
return r#type;
}
cur_type
}

static mut _rs__1860: ratelimit_state = {
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

/// A target_type of 0 will select the best domain type. 0 can be returned in
/// this case meaning the global default should be used.
fn iommu_get_default_domain_type(group: *mut iommu_group, target_type: i32) -> i32 {
let mut untrusted: *mut device = core::ptr::null_mut::<device>();
let mut gdev: *mut group_device = unsafe { core::mem::zeroed() };
let mut driver_type: i32 = 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
// ARM32 drivers supporting CONFIG_ARM_DMA_USE_IOMMU can declare an
// identity_domain and it will automatically become their default
// domain. Later on ARM_DMA_USE_IOMMU will install its UNMANAGED domain.
// Override the selection to IDENTITY.
if IS_ENABLED!() != 0 {
driver_type = 4i32;
}
gdev = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(gdev, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
driver_type = iommu_get_def_domain_type(group, unsafe { (*gdev).dev }, driver_type);
let __v2: bool = dev_is_pci!(unsafe { (*gdev).dev }) && ((unsafe { (*to_pci_dev!(unsafe { (*gdev).dev })).__slate_bits_3.__get_untrusted() }) as i32) != 0;
if __v2 {
// No ARM32 using systems will set untrusted, it cannot
// work.
let mut __v4: i64;
let __ret_warn_on: i32 = (IS_ENABLED!() != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("736: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 736b - ., 3; .popsection\n# {0}", const 736 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_get_default_domain_type_5, sym __slate_asm_string_iommu_get_default_domain_type_6, const 1909 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("737: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 737b - ., 4; .popsection\n# {0}", const 737 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
if __v4 != 0 {
return 1i32.wrapping_neg();
}
untrusted = unsafe { (*gdev).dev };
}
gdev = list_next_entry__4!(gdev, list);
}
// If the common dma ops are not selected in kconfig then we cannot use
// IOMMU_DOMAIN_DMA at all. Force IDENTITY if nothing else has been
// selected.
if IS_ENABLED__2!() == 0 {
let mut __v11: i64;
let __ret_warn_on: i32 = ((driver_type as u32) == 3u32) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("738: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 738b - ., 3; .popsection\n# {0}", const 738 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_get_default_domain_type_12, sym __slate_asm_string_iommu_get_default_domain_type_13, const 1921 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("739: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 739b - ., 4; .popsection\n# {0}", const 739 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v11 = (__ret_warn_on != 0) as i64;
if __v11 != 0 {
return 1i32.wrapping_neg();
}
if driver_type == 0 {
driver_type = 4i32;
}
}
if !untrusted.is_null() {
if driver_type != 0 && (driver_type as u32) != 3u32 {
loop {
if __ratelimit__15!(unsafe { core::ptr::addr_of_mut!(_rs__1929) }) != 0 {
while false {
}
unsafe { _dev_err(untrusted as *const device, c"Device is not trusted, but driver is overriding group %u to %s, refusing to probe.\n".as_ptr() as *const u8, (*group).id, iommu_domain_type_str(driver_type as u32)) };
}
if true {
break;
}
}
return 1i32.wrapping_neg();
}
driver_type = 3i32;
}
if target_type != 0 {
if driver_type != 0 && target_type != driver_type {
return 1i32.wrapping_neg();
}
return target_type;
}
driver_type
}

static mut _rs__1929: ratelimit_state = {
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

fn iommu_group_do_probe_finalize(dev: *mut device) {
let ops: *const iommu_ops = dev_iommu_ops(dev);
if (unsafe { (*ops).probe_finalize }) != None {
unsafe { unsafe { (*ops).probe_finalize }.unwrap()(dev) };
}
}

fn bus_iommu_probe(bus: *const bus_type) -> i32 {
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
let mut next: *mut iommu_group = unsafe { core::mem::zeroed() };
let mut group_list: list_head = unsafe { core::mem::zeroed() };
group_list = list_head { next: core::ptr::addr_of_mut!(group_list), prev: core::ptr::addr_of_mut!(group_list) };
let mut ret: i32 = unsafe { bus_for_each_dev(bus, core::ptr::null_mut::<device>(), core::ptr::addr_of_mut!(group_list) as *mut (), Some(probe_iommu_group)) };
if ret != 0 {
return ret;
}
group = list_first_entry__44!(core::ptr::addr_of_mut!(group_list));
next = list_next_entry__38!(group, entry);
while list_entry_is_head!(group, core::ptr::addr_of_mut!(group_list), entry) == 0 {
let mut gdev: *mut group_device = unsafe { core::mem::zeroed() };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
// Remove item from the list
list_del_init(unsafe { core::ptr::addr_of_mut!((*group).entry) });
// We go to the trouble of deferred default domain creation so
// that the cross-group default domain type and the setup of the
// IOMMU_RESV_DIRECT will work correctly in non-hotpug scenarios.
ret = iommu_setup_default_domain(group, 0i32);
if ret != 0 {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
return ret;
}
gdev = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(gdev, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
unsafe { iommu_setup_dma_ops((*gdev).dev, (*group).default_domain) };
gdev = list_next_entry__4!(gdev, list);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
// FIXME: Mis-locked because the ops->probe_finalize() call-back
// of some IOMMU drivers calls arm_iommu_attach_device() which
// in-turn might call back into IOMMU core code, where it tries
// to take group->mutex, resulting in a deadlock.
gdev = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(gdev, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
iommu_group_do_probe_finalize(unsafe { (*gdev).dev });
gdev = list_next_entry__4!(gdev, list);
}
group = next;
next = list_next_entry__38!(next, entry);
}
0
}

/// device_iommu_capable() - check for a general IOMMU capability
/// @dev: device to which the capability would be relevant, if available
/// @cap: IOMMU capability
///
/// Return: true if an IOMMU is present and supports the given capability
/// for the given device, otherwise false.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.device_iommu_capable")]
extern "C" fn device_iommu_capable(dev: *mut device, cap: u32) -> bool {
if !dev_has_iommu(dev) {
return false;
}
let ops: *const iommu_ops = dev_iommu_ops(dev);
if (unsafe { (*ops).capable }) == None {
return false;
}
unsafe { unsafe { (*ops).capable }.unwrap()(dev, cap) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_device_iommu_capable_740: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, u32) -> bool>, *mut ()>(Some(device_iommu_capable)) };

/// iommu_group_has_isolated_msi() - Compute msi_device_has_isolated_msi()
///       for a group
/// @group: Group to query
///
/// IOMMU groups should not have differing values of
/// msi_device_has_isolated_msi() for devices in a group. However nothing
/// directly prevents this, so ensure mistakes don't result in isolation failures
/// by checking that all the devices are the same.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_has_isolated_msi")]
extern "C" fn iommu_group_has_isolated_msi(group: *mut iommu_group) -> bool {
let mut group_dev: *mut group_device = unsafe { core::mem::zeroed() };
let mut ret: bool = true;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
group_dev = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(group_dev, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
let __v1: bool = unsafe { msi_device_has_isolated_msi((*group_dev).dev) };
ret = (ret as i32) & (__v1 as i32) != 0;
group_dev = list_next_entry__4!(group_dev, list);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_has_isolated_msi_741: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group) -> bool>, *mut ()>(Some(iommu_group_has_isolated_msi)) };

/// iommu_set_fault_handler() - set a fault handler for an iommu domain
/// @domain: iommu domain
/// @handler: fault handler
/// @token: user data, will be passed back to the fault handler
///
/// This function should be used by IOMMU users which want to be notified
/// whenever an IOMMU fault happens.
///
/// The fault handler itself should return 0 on success, and an appropriate
/// error code otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_set_fault_handler")]
extern "C" fn iommu_set_fault_handler(domain: *mut iommu_domain, handler: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u64, i32, *mut ()) -> i32>, token: *mut ()) {
let mut __v0: i64;
let __ret_warn_on: i32 = (domain.is_null() || ((unsafe { (*domain).cookie_type }) as u32) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("742: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 742b - ., 3; .popsection\n# {0}", const 742 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_set_fault_handler_1, sym __slate_asm_string_iommu_set_fault_handler_2, const 2061 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("743: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 743b - ., 4; .popsection\n# {0}", const 743 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return;
}
unsafe {
(*domain).cookie_type = 3;
}
unsafe {
(*domain).__slate_anon_9.__slate_anon_3.handler = handler;
}
unsafe {
(*domain).__slate_anon_9.__slate_anon_3.handler_token = token;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_set_fault_handler_744: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u64, i32, *mut ()) -> i32>, *mut ())>, *mut ()>(Some(iommu_set_fault_handler)) };

fn iommu_domain_init(domain: *mut iommu_domain, r#type: u32, ops: *const iommu_ops) {
unsafe {
(*domain).r#type = r#type;
}
unsafe {
(*domain).owner = ops;
}
if unsafe { (*domain).ops }.is_null() {
unsafe {
(*domain).ops = (*ops).default_domain_ops;
}
}
}

fn __iommu_paging_domain_alloc_flags(dev: *mut device, r#type: u32, flags: u32) -> *mut iommu_domain {
let mut domain: *mut iommu_domain = unsafe { core::mem::zeroed() };
if !dev_has_iommu(dev) {
return ERR_PTR(19i32.wrapping_neg() as i64) as *mut iommu_domain;
}
let ops: *const iommu_ops = dev_iommu_ops(dev);
if (unsafe { (*ops).domain_alloc_paging }) != None && flags == 0 {
domain = unsafe { unsafe { (*ops).domain_alloc_paging }.unwrap()(dev) };
} else {
if (unsafe { (*ops).domain_alloc_paging_flags }) != None {
domain = unsafe { unsafe { (*ops).domain_alloc_paging_flags }.unwrap()(dev, flags, core::ptr::null::<iommu_user_data>()) };
} else {
return ERR_PTR(95i32.wrapping_neg() as i64) as *mut iommu_domain;
}
}
if IS_ERR(domain as *const ()) {
return domain;
}
if domain.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut iommu_domain;
}
iommu_domain_init(domain, r#type, ops);
domain
}

/// iommu_paging_domain_alloc_flags() - Allocate a paging domain
/// @dev: device for which the domain is allocated
/// @flags: Bitmap of iommufd_hwpt_alloc_flags
///
/// Allocate a paging domain which will be managed by a kernel driver. Return
/// allocated domain if successful, or an ERR pointer for failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_paging_domain_alloc_flags")]
extern "C" fn iommu_paging_domain_alloc_flags(dev: *mut device, flags: u32) -> *mut iommu_domain {
__iommu_paging_domain_alloc_flags(dev, 1u32, flags)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_paging_domain_alloc_flags_745: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, u32) -> *mut iommu_domain>, *mut ()>(Some(iommu_paging_domain_alloc_flags)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_domain_free")]
extern "C" fn iommu_domain_free(domain: *mut iommu_domain) {
match (unsafe { (*domain).cookie_type }) as u32 {
1 => {
unsafe { iommu_put_dma_cookie(domain) };
}
2 => {
unsafe { iommu_put_msi_cookie(domain) };
}
4 => {
mmdrop(unsafe { (*domain).__slate_anon_9.__slate_anon_4.mm });
}
_ => {
}
}
if (unsafe { (*(*domain).ops).free }) != None {
unsafe { unsafe { (*(*domain).ops).free }.unwrap()(domain) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_domain_free_746: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain)>, *mut ()>(Some(iommu_domain_free)) };

/// Put the group's domain back to the appropriate core-owned domain - either the
/// standard kernel-mode DMA configuration or an all-DMA-blocked domain.
fn __iommu_group_set_core_domain(group: *mut iommu_group) {
let mut new_domain: *mut iommu_domain = unsafe { core::mem::zeroed() };
if !unsafe { (*group).owner }.is_null() {
new_domain = unsafe { (*group).blocking_domain };
} else {
new_domain = unsafe { (*group).default_domain };
}
__iommu_group_set_domain_nofail(group, new_domain);
}

fn __iommu_attach_device(domain: *mut iommu_domain, dev: *mut device, old: *mut iommu_domain) -> i32 {
let mut ret: i32 = 0;
if unlikely!((unsafe { (*(*domain).ops).attach_dev }) == None) != 0 {
return 19i32.wrapping_neg();
}
ret = unsafe { unsafe { (*(*domain).ops).attach_dev }.unwrap()(domain, dev, old) };
if ret != 0 {
return ret;
}
unsafe {
(*(*dev).iommu).__slate_bits_0.__set_attach_deferred(0u32);
}
trace_attach_device_to_domain(dev);
0
}

/// iommu_attach_device - Attach an IOMMU domain to a device
/// @domain: IOMMU domain to attach
/// @dev: Device that will be attached
///
/// Returns 0 on success and error code on failure
///
/// Note that EINVAL can be treated as a soft failure, indicating
/// that certain configuration of the domain is incompatible with
/// the device. In this case attaching a different domain to the
/// device may succeed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_attach_device")]
extern "C" fn iommu_attach_device(domain: *mut iommu_domain, dev: *mut device) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
// Caller must be a probed driver on dev
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
unsafe {
group = (*dev).iommu_group;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if group.is_null() {
return 19i32.wrapping_neg();
} else {
// Lock the group to make sure the device-count doesn't
// change while we are attaching
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
*__slate_slot_0 = 22i32.wrapping_neg();
if list_count_nodes(core::ptr::addr_of_mut!((*group).devices)) == 1 {
*__slate_slot_0 = __iommu_attach_group(domain, group);
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
return *__slate_slot_0;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_attach_device_747: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut device) -> i32>, *mut ()>(Some(iommu_attach_device)) };

#[unsafe(no_mangle)]
extern "C" fn iommu_deferred_attach(dev: *mut device, domain: *mut iommu_domain) -> i32 {
// This is called on the dma mapping fast path so avoid locking. This is
// racy, but we have an expectation that the driver will setup its DMAs
// inside probe while being single threaded to avoid racing.
if unsafe { (*dev).iommu }.is_null() || ((unsafe { (*(*dev).iommu).__slate_bits_0.__get_attach_deferred() }) as i32) == 0 {
return 0;
}
let mut __UNIQUE_ID_guard_748: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*(*dev).iommu_group).mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_748) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_749: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*(*dev).iommu_group).mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_749) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let gdev: *mut group_device = __dev_to_gdev(dev);
let mut __v2: i64;
let __ret_warn_on: i32 = gdev.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("750: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 750b - ., 3; .popsection\n# {0}", const 750 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_deferred_attach_3, sym __slate_asm_string_iommu_deferred_attach_4, const 2232 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("751: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 751b - ., 4; .popsection\n# {0}", const 751 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return 19i32.wrapping_neg();
}
// This is a concurrent attach during device recovery. Reject it until
// pci_dev_reset_iommu_done() attaches the device to group->domain.
//
// Note that this might fail the iommu_dma_map(). But there's nothing
// more we can do here.
if unsafe { (*gdev).blocked } {
return 16i32.wrapping_neg();
}
__iommu_attach_device(domain, dev, core::ptr::null_mut::<iommu_domain>())
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_detach_device")]
extern "C" fn iommu_detach_device(domain: *mut iommu_domain, dev: *mut device) {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
// Caller must be a probed driver on dev
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
unsafe {
group = (*dev).iommu_group;
if group.is_null() {
return;
} else {
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
__ret_warn_on_1 = (domain != (*group).domain) as i32;
if unlikely__2!(__ret_warn_on_1) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("752: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 752b - ., 3; .popsection\n# {0}", const 752 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_detach_device_10, sym __slate_asm_string_iommu_detach_device_11, const 2256 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("753: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 753b - ., 4; .popsection\n# {0}", const 753 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v6: i64 = (__ret_warn_on_1 != 0) as i64;
if __v6 != 0 {
__v3 = true;
} else {
__ret_warn_on = (list_count_nodes(core::ptr::addr_of_mut!((*group).devices)) != 1) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("754: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 754b - ., 3; .popsection\n# {0}", const 754 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_detach_device_8, sym __slate_asm_string_iommu_detach_device_9, const 2257 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("755: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 755b - ., 4; .popsection\n# {0}", const 755 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
__v3 = __ret_warn_on != 0;
}
if !__v3 {
__iommu_group_set_core_domain(group);
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_detach_device_756: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut device)>, *mut ()>(Some(iommu_detach_device)) };

/// iommu_get_domain_for_dev() - Return the DMA API domain pointer
/// @dev: Device to query
///
/// This function can be called within a driver bound to dev. The returned
/// pointer is valid for the lifetime of the bound driver.
///
/// It should not be called by drivers with driver_managed_dma = true.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_get_domain_for_dev")]
extern "C" fn iommu_get_domain_for_dev(dev: *mut device) -> *mut iommu_domain {
// Caller must be a probed driver on dev
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
if group.is_null() {
return core::ptr::null_mut::<iommu_domain>();
}
lockdep_assert_not_held!(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
unsafe { (*group).domain }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_get_domain_for_dev_757: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device) -> *mut iommu_domain>, *mut ()>(Some(iommu_get_domain_for_dev)) };

/// iommu_driver_get_domain_for_dev() - Return the driver-level domain pointer
/// @dev: Device to query
///
/// This function can be called by an iommu driver that wants to get the physical
/// domain within an iommu callback function where group->mutex is held.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_driver_get_domain_for_dev")]
extern "C" fn iommu_driver_get_domain_for_dev(dev: *mut device) -> *mut iommu_domain {
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
let gdev: *mut group_device = __dev_to_gdev(dev);
let mut __v1: i64;
let __ret_warn_on: i32 = gdev.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("758: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 758b - ., 3; .popsection\n# {0}", const 758 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_driver_get_domain_for_dev_2, sym __slate_asm_string_iommu_driver_get_domain_for_dev_3, const 2304 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("759: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 759b - ., 4; .popsection\n# {0}", const 759 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
return core::ptr::null_mut::<iommu_domain>();
}
// Driver handles the low-level __iommu_attach_device(), including the
// one invoked by pci_dev_reset_iommu_done() re-attaching the device to
// the cached group->domain. In this case, the driver must get the old
// domain from group->blocking_domain rather than group->domain. This
// prevents it from re-attaching the device from group->domain (old) to
// group->domain (new).
if unsafe { (*gdev).blocked } {
return unsafe { (*group).blocking_domain };
}
unsafe { (*group).domain }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_driver_get_domain_for_dev_760: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device) -> *mut iommu_domain>, *mut ()>(Some(iommu_driver_get_domain_for_dev)) };

/// For IOMMU_DOMAIN_DMA implementations which already provide their own
/// guarantees that the group and its default domain are valid and correct.
#[unsafe(no_mangle)]
extern "C" fn iommu_get_dma_domain(dev: *mut device) -> *mut iommu_domain {
unsafe { (*(*dev).iommu_group).default_domain }
}

fn iommu_make_pasid_array_entry(domain: *mut iommu_domain, handle: *mut iommu_attach_handle) -> *mut () {
if !handle.is_null() {
unsafe {
(*handle).domain = domain;
}
return xa_tag_pointer(handle as *mut (), 1u64);
}
xa_tag_pointer(domain as *mut (), 0u64)
}

fn domain_iommu_ops_compatible(ops: *const iommu_ops, domain: *mut iommu_domain) -> bool {
if (unsafe { (*domain).owner }) == ops {
return true;
}
// For static domains, owner isn't set.
if domain == unsafe { (*ops).blocked_domain } || domain == unsafe { (*ops).identity_domain } {
return true;
}
false
}

fn __iommu_attach_group(domain: *mut iommu_domain, group: *mut iommu_group) -> i32 {
if !unsafe { (*group).domain }.is_null() && (unsafe { (*group).domain }) != unsafe { (*group).default_domain } && (unsafe { (*group).domain }) != unsafe { (*group).blocking_domain } {
return 16i32.wrapping_neg();
}
let dev: *mut device = iommu_group_first_dev(group);
let __v0: bool = !dev_has_iommu(dev) || !domain_iommu_ops_compatible(dev_iommu_ops(dev), domain);
if __v0 {
return 22i32.wrapping_neg();
}
__iommu_group_set_domain(group, domain)
}

/// iommu_attach_group - Attach an IOMMU domain to an IOMMU group
/// @domain: IOMMU domain to attach
/// @group: IOMMU group that will be attached
///
/// Returns 0 on success and error code on failure
///
/// Note that EINVAL can be treated as a soft failure, indicating
/// that certain configuration of the domain is incompatible with
/// the group. In this case attaching a different domain to the
/// group may succeed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_attach_group")]
extern "C" fn iommu_attach_group(domain: *mut iommu_domain, group: *mut iommu_group) -> i32 {
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
ret = __iommu_attach_group(domain, group);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_attach_group_761: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_group) -> i32>, *mut ()>(Some(iommu_attach_group)) };

fn __iommu_device_set_domain(group: *mut iommu_group, dev: *mut device, new_domain: *mut iommu_domain, old_domain: *mut iommu_domain, flags: u32) -> i32 {
let mut ret: i32 = 0;
// If the device requires IOMMU_RESV_DIRECT then we cannot allow
// the blocking domain to be attached as it does not contain the
// required 1:1 mapping. This test effectively excludes the device
// being used with iommu_group_claim_dma_owner() which will block
// vfio and iommufd as well.
if ((unsafe { (*(*dev).iommu).__slate_bits_0.__get_require_direct() }) as i32) != 0 && ((unsafe { (*new_domain).r#type }) == 0 || new_domain == unsafe { (*group).blocking_domain }) {
while false {
}
unsafe { _dev_warn(dev as *const device, c"Firmware has requested this device have a 1:1 IOMMU mapping, rejecting configuring the device without a 1:1 mapping. Contact your platform vendor.\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
if ((unsafe { (*(*dev).iommu).__slate_bits_0.__get_attach_deferred() }) as i32) != 0 {
if new_domain == unsafe { (*group).default_domain } {
return 0;
}
unsafe {
(*(*dev).iommu).__slate_bits_0.__set_attach_deferred(0u32);
}
}
ret = __iommu_attach_device(new_domain, dev, old_domain);
if ret != 0 {
// If we have a blocking domain then try to attach that in hopes
// of avoiding a UAF. Modern drivers should implement blocking
// domains as global statics that cannot fail.
if flags & 1 != 0 && !unsafe { (*group).blocking_domain }.is_null() && (unsafe { (*group).blocking_domain }) != new_domain {
__iommu_attach_device(unsafe { (*group).blocking_domain }, dev, old_domain);
}
return ret;
}
0
}

/// If 0 is returned the group's domain is new_domain. If an error is returned
/// then the group's domain will be set back to the existing domain unless
/// IOMMU_SET_DOMAIN_MUST_SUCCEED, otherwise an error is returned and the group's
/// domains is left inconsistent. This is a driver bug to fail attach with a
/// previously good domain. We try to avoid a kernel UAF because of this.
///
/// IOMMU groups are really the natural working unit of the IOMMU, but the IOMMU
/// API works on domains and devices.  Bridge that gap by iterating over the
/// devices in a group.  Ideally we'd have a single device which represents the
/// requestor ID of the group, but we also allow IOMMU drivers to create policy
/// defined minimum sets, where the physical hardware may be able to distiguish
/// members, but we wish to group them at a higher level (ex. untrusted
/// multi-function PCI devices).  Thus we attach each device.
fn __iommu_group_set_domain_internal(group: *mut iommu_group, new_domain: *mut iommu_domain, flags: u32) -> i32 {
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(result) as *mut i32;
let mut gdev: core::mem::MaybeUninit<*mut group_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut group_device = core::ptr::addr_of_mut!(gdev) as *mut *mut group_device;
let mut last_gdev: core::mem::MaybeUninit<*mut group_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut group_device = core::ptr::addr_of_mut!(last_gdev) as *mut *mut group_device;
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
lockdep_assert_held!(core::ptr::addr_of_mut!((*group).mutex));
if (*group).domain == new_domain {
return 0;
} else {
__ret_warn_on_1 = new_domain.is_null() as i32;
if unlikely__2!(__ret_warn_on_1) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("762: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 762b - ., 3; .popsection\n# {0}", const 762 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_group_set_domain_internal_16, sym __slate_asm_string___iommu_group_set_domain_internal_17, const 2471 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("763: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 763b - ., 4; .popsection\n# {0}", const 763 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v5: i64 = (__ret_warn_on_1 != 0) as i64;
if __v5 != 0 {
return 22i32.wrapping_neg();
} else {
// This is a concurrent attach during device recovery. Reject it until
// pci_dev_reset_iommu_done() attaches the device to group->domain, if
// IOMMU_SET_DOMAIN_MUST_SUCCEED is not set.
if (*group).recovery_cnt != 0 && flags & 1 == 0 {
return 16i32.wrapping_neg();
} else {
// Changing the domain is done by calling attach_dev() on the new
// domain. This switch does not have to be atomic and DMA can be
// discarded during the transition. DMA must only be able to access
// either new_domain or group->domain, never something else.
*__slate_slot_7 = 0;
*__slate_slot_8 = list_first_entry__3!(core::ptr::addr_of_mut!((*group).devices));
'__join_10: {
while list_entry_is_head!(*__slate_slot_8, core::ptr::addr_of_mut!((*group).devices), list) == 0 {
// Device under recovery is attached to group->blocking_domain.
// Don't change that. pci_dev_reset_iommu_done() will re-attach
// its domain to the updated group->domain, after the recovery.
if !(*(*__slate_slot_8)).blocked {
*__slate_slot_6 = __iommu_device_set_domain(group, (*(*__slate_slot_8)).dev, new_domain, (*group).domain, flags);
if *__slate_slot_6 != 0 {
*__slate_slot_7 = *__slate_slot_6;
// Keep trying the other devices in the group. If a
// driver fails attach to an otherwise good domain, and
// does not support blocking domains, it should at least
// drop its reference on the current domain so we don't
// UAF.
if flags & 1 == 0 {
break '__join_10;
}
}
}
*__slate_slot_8 = list_next_entry__4!(*__slate_slot_8, list);
}
(*group).domain = new_domain;
return *__slate_slot_7;
}
*__slate_slot_9 = *__slate_slot_8;
// This is called in error unwind paths. A well behaved driver should
// always allow us to attach to a domain that was already attached.
*__slate_slot_8 = list_first_entry__3!(core::ptr::addr_of_mut!((*group).devices));
while list_entry_is_head!(*__slate_slot_8, core::ptr::addr_of_mut!((*group).devices), list) == 0 {
// No need to revert the last gdev that failed to set domain
if *__slate_slot_8 == *__slate_slot_9 {
break;
} else {
// A NULL domain can happen only for first probe, in which case
// we leave group->domain as NULL and let release clean
// everything up.
if !unsafe { (*group).domain }.is_null() {
let __ret_warn_on: i32 = (__iommu_device_set_domain(group, (*(*__slate_slot_8)).dev, (*group).domain, new_domain, 1u32) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("764: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 764b - ., 3; .popsection\n# {0}", const 764 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_group_set_domain_internal_11, sym __slate_asm_string___iommu_group_set_domain_internal_12, const 2534 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("765: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 765b - ., 4; .popsection\n# {0}", const 765 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
*__slate_slot_8 = list_next_entry__4!(*__slate_slot_8, list);
}
}
return *__slate_slot_6;
}
}
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_detach_group")]
extern "C" fn iommu_detach_group(domain: *mut iommu_domain, group: *mut iommu_group) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
__iommu_group_set_core_domain(group);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_detach_group_766: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_group)>, *mut ()>(Some(iommu_detach_group)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_iova_to_phys")]
extern "C" fn iommu_iova_to_phys(domain: *mut iommu_domain, iova: u64) -> u64 {
if (unsafe { (*domain).r#type }) == 4u32 {
return iova;
}
if (unsafe { (*domain).r#type }) == 0 {
return 0;
}
unsafe { unsafe { (*(*domain).ops).iova_to_phys }.unwrap()(domain, iova) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_iova_to_phys_767: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, u64) -> u64>, *mut ()>(Some(iommu_iova_to_phys)) };

fn iommu_pgsize(domain: *mut iommu_domain, iova: u64, paddr: u64, mut size: u64, count: *mut u64) -> u64 {
let mut addr_merge: u64 = unsafe { core::mem::zeroed() };
let mut offset_end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(offset_end) as *mut u64;
let mut pgsize_next: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(pgsize_next) as *mut u64;
let mut pgsize: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(pgsize) as *mut u64;
let mut offset: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(offset) as *mut u64;
let mut pgsizes: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(pgsizes) as *mut u64;
let mut pgsize_idx_next: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(pgsize_idx_next) as *mut u32;
let mut pgsize_idx: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(pgsize_idx) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
addr_merge = paddr | iova;
// Page sizes supported by the hardware and small enough for @size
*__slate_slot_7 = (*domain).pgsize_bitmap & GENMASK__7!(__fls(size), 0i32);
// Constrain the page sizes further based on the maximum alignment
if likely__2!(addr_merge) != 0 {
let __v1: u64 = *__slate_slot_7;
let __v0: u64 = __v1 & GENMASK__46!(if __slate_intrinsic_f217c4720f04f6db(addr_merge) { (addr_merge.trailing_zeros() as i32) as u64 } else { variable__ffs(addr_merge) }, 0i32);
*__slate_slot_7 = __v0;
}
// Make sure we have at least one suitable page size
'__join_9: {
loop {
if unlikely!(*__slate_slot_7 == 0) != 0 {
break '__join_9;
} else {
if true {
break;
}
}
}
// Pick the biggest page size remaining
*__slate_slot_9 = __fls(*__slate_slot_7) as u32;
*__slate_slot_5 = BIT!(*__slate_slot_9);
if count.is_null() {
return *__slate_slot_5;
} else {
// Find the next biggest support page size, if it exists
*__slate_slot_7 = (*domain).pgsize_bitmap & !GENMASK__44!(*__slate_slot_9, 0i32);
if *__slate_slot_7 != 0 {
*__slate_slot_8 = __ffs!(*__slate_slot_7) as u32;
*__slate_slot_4 = BIT!(*__slate_slot_8);
// There's no point trying a bigger page size unless the virtual
// and physical addresses are similarly offset within the larger page.
if (iova ^ paddr) & (*__slate_slot_4).wrapping_sub(1) == 0 {
// Calculate the offset to the next page size alignment boundary
*__slate_slot_6 = (*__slate_slot_4).wrapping_sub(addr_merge & (*__slate_slot_4).wrapping_sub(1));
// If size is big enough to accommodate the larger page, reduce
// the number of smaller pages.
if !check_add_overflow!(*__slate_slot_6, *__slate_slot_4, core::ptr::addr_of_mut!(*__slate_slot_3)) && *__slate_slot_3 <= size {
size = *__slate_slot_6;
}
}
}
*count = size >> *__slate_slot_9;
return *__slate_slot_5;
}
}
core::arch::asm!("768: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 768b - ., 3; .popsection\n# {0}", const 768 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_iommu_pgsize_10, sym __slate_asm_string_iommu_pgsize_11, const 2576 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

fn __iommu_map_domain_pgtbl(domain: *mut iommu_domain, mut iova: u64, mut paddr: u64, mut size: u64, prot: i32, gfp: u32, mapped: &mut u64) -> i32 {
let ops: *const iommu_domain_ops = unsafe { (*domain).ops };
let mut min_pagesz: u32 = 0;
let mut ret: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*ops).map_pages }) == None) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("769: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 769b - ., 3; .popsection\n# {0}", const 769 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_map_domain_pgtbl_1, sym __slate_asm_string___iommu_map_domain_pgtbl_2, const 2624 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("770: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 770b - ., 4; .popsection\n# {0}", const 770 as i32, options(nostack, att_syntax));
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
// find out the minimum page size supported
min_pagesz = (1i32 << __ffs!(unsafe { (*domain).pgsize_bitmap })) as u32;
// both the virtual address and the physical one, as well as
// the size of the mapping, must be aligned (at least) to the
// size of the smallest page supported by the hardware
if !IS_ALIGNED!(iova | paddr | size, min_pagesz) {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013iommu: unaligned: iova 0x%lx pa %pa size 0x%zx min_pagesz 0x%x\n".as_ptr() as *const u8, iova, core::ptr::addr_of_mut!(paddr), size, min_pagesz) };
return 22i32.wrapping_neg();
}
let mut __v7: i32;
if false {
unsafe { _printk(c"\x017iommu: map: iova 0x%lx pa %pa size 0x%zx\n".as_ptr() as *const u8, iova, core::ptr::addr_of_mut!(paddr), size) };
}
__v7 = 0;
while size != 0 {
let mut pgsize: u64 = 0;
let mut count: u64 = 0;
let mut op_mapped: u64 = 0;
pgsize = iommu_pgsize(domain, iova, paddr, size, core::ptr::addr_of_mut!(count));
let mut __v8: i32;
if false {
unsafe { _printk(c"\x017iommu: mapping: iova 0x%lx pa %pa pgsize 0x%zx count %zu\n".as_ptr() as *const u8, iova, core::ptr::addr_of_mut!(paddr), pgsize, count) };
}
__v8 = 0;
ret = unsafe { unsafe { (*ops).map_pages }.unwrap()(domain, iova, paddr, pgsize, count, prot, gfp, core::ptr::addr_of_mut!(op_mapped)) };
// Some pages may have been mapped, even if an error occurred,
// so we should account for those so they can be unmapped.
let __v10: u64 = unsafe { *mapped };
unsafe {
*mapped = __v10.wrapping_add(op_mapped);
}
if ret != 0 {
return ret;
}
size = size.wrapping_sub(op_mapped);
iova = iova.wrapping_add(op_mapped);
paddr = paddr.wrapping_add(op_mapped);
}
0
}

#[unsafe(no_mangle)]
extern "C" fn iommu_sync_map(domain: *mut iommu_domain, iova: u64, size: u64) -> i32 {
let ops: *const iommu_domain_ops = unsafe { (*domain).ops };
if (unsafe { (*ops).iotlb_sync_map }) == None {
return 0;
}
unsafe { unsafe { (*ops).iotlb_sync_map }.unwrap()(domain, iova, size) }
}

#[unsafe(no_mangle)]
extern "C" fn iommu_map_nosync(domain: *mut iommu_domain, iova: u64, paddr: u64, size: u64, prot: i32, gfp: u32) -> i32 {
let pt: *mut pt_iommu = iommupt_from_domain(domain);
let mut mapped: u64 = 0;
let mut ret: i32 = 0;
might_sleep_if!(gfpflags_allow_blocking(gfp));
// Discourage passing strange GFP flags or illegal domains
let mut __v2: i64;
let __ret_warn_on: i32 = ((unsafe { (*domain).r#type }) & 1u32 == 0 || (unsafe { (*domain).pgsize_bitmap }) == 0 || gfp & ((BIT!(18i32) as u32) | (BIT!(0i32) as u32) | (BIT!(2i32) as u32) | (BIT!(1i32) as u32)) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("771: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 771b - ., 3; .popsection\n# {0}", const 771 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_map_nosync_3, sym __slate_asm_string_iommu_map_nosync_4, const 2689 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("772: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 772b - ., 4; .popsection\n# {0}", const 772 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return 22i32.wrapping_neg();
}
if !pt.is_null() {
ret = unsafe { unsafe { (*(*pt).ops).map_range }.unwrap()(pt, iova, paddr, size, prot as u32, gfp, core::ptr::addr_of_mut!(mapped)) };
} else {
ret = __iommu_map_domain_pgtbl(domain, iova, paddr, size, prot, gfp, unsafe { &mut (*core::ptr::addr_of_mut!(mapped)) });
}
trace_map(iova, paddr, mapped);
iommu_debug_map(domain, paddr, mapped);
if ret != 0 {
iommu_unmap(domain, iova, mapped);
return ret;
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_map")]
extern "C" fn iommu_map(domain: *mut iommu_domain, iova: u64, paddr: u64, size: u64, prot: i32, gfp: u32) -> i32 {
let mut ret: i32 = iommu_map_nosync(domain, iova, paddr, size, prot, gfp);
if ret != 0 {
return ret;
}
ret = iommu_sync_map(domain, iova, size);
if ret != 0 {
iommu_unmap(domain, iova, size);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_map_773: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, i32, u32) -> i32>, *mut ()>(Some(iommu_map)) };

fn __iommu_unmap_domain_pgtbl(domain: *mut iommu_domain, mut iova: u64, size: u64, iotlb_gather: *mut iommu_iotlb_gather) -> u64 {
let ops: *const iommu_domain_ops = unsafe { (*domain).ops };
let mut unmapped_page: u64 = 0;
let mut unmapped: u64 = 0;
let mut min_pagesz: u32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*ops).unmap_pages }) == None) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("774: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 774b - ., 3; .popsection\n# {0}", const 774 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_unmap_domain_pgtbl_1, sym __slate_asm_string___iommu_unmap_domain_pgtbl_2, const 2733 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("775: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 775b - ., 4; .popsection\n# {0}", const 775 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 0;
}
// find out the minimum page size supported
min_pagesz = (1i32 << __ffs!(unsafe { (*domain).pgsize_bitmap })) as u32;
// The virtual address, as well as the size of the mapping, must be
// aligned (at least) to the size of the smallest page supported
// by the hardware
if !IS_ALIGNED!(iova | size, min_pagesz) {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013iommu: unaligned: iova 0x%lx size 0x%zx min_pagesz 0x%x\n".as_ptr() as *const u8, iova, size, min_pagesz) };
return 0;
}
let mut __v7: i32;
if false {
unsafe { _printk(c"\x017iommu: unmap this: iova 0x%lx size 0x%zx\n".as_ptr() as *const u8, iova, size) };
}
__v7 = 0;
// Keep iterating until we either unmap 'size' bytes (or more)
// or we hit an area that isn't mapped.
while unmapped < size {
let mut pgsize: u64 = 0;
let mut count: u64 = 0;
pgsize = iommu_pgsize(domain, iova, iova, size.wrapping_sub(unmapped), core::ptr::addr_of_mut!(count));
unmapped_page = unsafe { unsafe { (*ops).unmap_pages }.unwrap()(domain, iova, pgsize, count, iotlb_gather) };
if unmapped_page == 0 {
break;
}
let mut __v9: i32;
if false {
unsafe { _printk(c"\x017iommu: unmapped: iova 0x%lx size 0x%zx\n".as_ptr() as *const u8, iova, unmapped_page) };
}
__v9 = 0;
// If the driver itself isn't using the gather, make sure
// it looks non-empty so iotlb_sync will still be called.
if (unsafe { (*iotlb_gather).start }) >= unsafe { (*iotlb_gather).end } {
iommu_iotlb_gather_add_range(iotlb_gather, iova, size);
}
iova = iova.wrapping_add(unmapped_page);
unmapped = unmapped.wrapping_add(unmapped_page);
}
unmapped
}

fn __iommu_unmap(domain: *mut iommu_domain, iova: u64, size: u64, iotlb_gather: *mut iommu_iotlb_gather) -> u64 {
let pt: *mut pt_iommu = iommupt_from_domain(domain);
let mut unmapped: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*domain).r#type }) & 1u32 == 0 || (unsafe { (*domain).pgsize_bitmap }) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("776: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 776b - ., 3; .popsection\n# {0}", const 776 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_unmap_1, sym __slate_asm_string___iommu_unmap_2, const 2788 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("777: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 777b - ., 4; .popsection\n# {0}", const 777 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 0;
}
iommu_debug_unmap_begin(domain, iova, size);
if !pt.is_null() {
unmapped = unsafe { unsafe { (*(*pt).ops).unmap_range }.unwrap()(pt, iova, size, iotlb_gather) };
} else {
unmapped = __iommu_unmap_domain_pgtbl(domain, iova, size, iotlb_gather);
}
trace_unmap(iova, size, unmapped);
iommu_debug_unmap_end(domain, iova, size, unmapped);
unmapped
}

/// iommu_unmap() - Remove mappings from a range of IOVA
/// @domain: Domain to manipulate
/// @iova: IO virtual address to start
/// @size: Length of the range starting from @iova
///
/// iommu_unmap() will remove a translation created by iommu_map(). It cannot
/// subdivide a mapping created by iommu_map(), so it should be called with IOVA
/// ranges that match what was passed to iommu_map(). The range can aggregate
/// contiguous iommu_map() calls so long as no individual range is split.
///
/// Returns: Number of bytes of IOVA unmapped. iova + res will be the point
/// unmapping stopped.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_unmap")]
extern "C" fn iommu_unmap(domain: *mut iommu_domain, iova: u64, size: u64) -> u64 {
let mut iotlb_gather: iommu_iotlb_gather = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iotlb_gather) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut ret: u64 = 0;
iommu_iotlb_gather_init(core::ptr::addr_of_mut!(iotlb_gather));
ret = __iommu_unmap(domain, iova, size, core::ptr::addr_of_mut!(iotlb_gather));
iommu_iotlb_sync(domain, core::ptr::addr_of_mut!(iotlb_gather));
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_unmap_778: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64) -> u64>, *mut ()>(Some(iommu_unmap)) };

/// iommu_unmap_fast() - Remove mappings from a range of IOVA without IOTLB sync
/// @domain: Domain to manipulate
/// @iova: IO virtual address to start
/// @size: Length of the range starting from @iova
/// @iotlb_gather: range information for a pending IOTLB flush
///
/// iommu_unmap_fast() will remove a translation created by iommu_map().
/// It can't subdivide a mapping created by iommu_map(), so it should be
/// called with IOVA ranges that match what was passed to iommu_map(). The
/// range can aggregate contiguous iommu_map() calls so long as no individual
/// range is split.
///
/// Basically iommu_unmap_fast() is the same as iommu_unmap() but for callers
/// which manage the IOTLB flushing externally to perform a batched sync.
///
/// Returns: Number of bytes of IOVA unmapped. iova + res will be the point
/// unmapping stopped.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_unmap_fast")]
extern "C" fn iommu_unmap_fast(domain: *mut iommu_domain, iova: u64, size: u64, iotlb_gather: *mut iommu_iotlb_gather) -> u64 {
__iommu_unmap(domain, iova, size, iotlb_gather)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_unmap_fast_779: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, *mut iommu_iotlb_gather) -> u64>, *mut ()>(Some(iommu_unmap_fast)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_map_sg")]
extern "C" fn iommu_map_sg(domain: *mut iommu_domain, iova: u64, mut sg: *mut scatterlist, nents: u32, prot: i32, gfp: u32) -> i64 {
let mut s_phys: u64 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: u32 = unsafe { core::mem::zeroed() };
let mut start: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(start) as *mut u64;
let mut mapped: u64 = unsafe { core::mem::zeroed() };
let mut len: u64 = unsafe { core::mem::zeroed() };
unsafe {
len = 0;
mapped = 0;
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
i = 0;
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
'__join_0: {
while i <= nents {
s_phys = sg_phys(sg);
if len != 0 && s_phys != (*__slate_slot_9).wrapping_add(len) {
*__slate_slot_7 = iommu_map_nosync(domain, iova.wrapping_add(mapped), *__slate_slot_9, len, prot, gfp);
if *__slate_slot_7 != 0 {
break '__join_0;
} else {
mapped = mapped.wrapping_add(len);
len = 0;
}
}
if !sg_dma_is_bus_address(sg) {
if len != 0 {
len = len.wrapping_add((*sg).length as u64);
} else {
len = (*sg).length as u64;
*__slate_slot_9 = s_phys;
}
}
let __v0: u32 = i.wrapping_add(1);
i = __v0;
if __v0 < nents {
sg = sg_next(sg);
}
}
*__slate_slot_7 = iommu_sync_map(domain, iova, mapped);
if *__slate_slot_7 == 0 {
return mapped as i64;
}
}
iommu_unmap(domain, iova, mapped); // undo mappings already done
return *__slate_slot_7 as i64;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_map_sg_780: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, u64, *mut scatterlist, u32, i32, u32) -> i64>, *mut ()>(Some(iommu_map_sg)) };

/// report_iommu_fault() - report about an IOMMU fault to the IOMMU framework
/// @domain: the iommu domain where the fault has happened
/// @dev: the device where the fault has happened
/// @iova: the faulting address
/// @flags: mmu fault flags (e.g. IOMMU_FAULT_READ/IOMMU_FAULT_WRITE/...)
///
/// This function should be called by the low-level IOMMU implementations
/// whenever IOMMU faults happen, to allow high-level users, that are
/// interested in such events, to know about them.
///
/// This event may be useful for several possible use cases:
/// - mere logging of the event
/// - dynamic TLB/PTE loading
/// - if restarting of the faulting device is required
///
/// Returns 0 on success and an appropriate error code otherwise (if dynamic
/// PTE/TLB loading will one day be supported, implementations will be able
/// to tell whether it succeeded or not according to this return value).
///
/// Specifically, -ENOSYS is returned if a fault handler isn't installed
/// (though fault handlers can also return -ENOSYS, in case they want to
/// elicit the default behavior of the IOMMU drivers).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.report_iommu_fault")]
extern "C" fn report_iommu_fault(domain: *mut iommu_domain, dev: *mut device, iova: u64, flags: i32) -> i32 {
let mut ret: i32 = 38i32.wrapping_neg();
// if upper layers showed interest and installed a fault handler,
// invoke it.
if ((unsafe { (*domain).cookie_type }) as u32) == 3 && (unsafe { (*domain).__slate_anon_9.__slate_anon_3.handler }) != None {
ret = unsafe { unsafe { (*domain).__slate_anon_9.__slate_anon_3.handler }.unwrap()(domain, dev, iova, flags, (*domain).__slate_anon_9.__slate_anon_3.handler_token) };
}
trace_io_page_fault(dev, iova, flags);
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_report_iommu_fault_781: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u64, i32) -> i32>, *mut ()>(Some(report_iommu_fault)) };

#[unsafe(link_section = ".init.text")]
extern "C" fn iommu_init() -> i32 {
unsafe {
iommu_group_kset = kset_create_and_add(c"iommu_groups".as_ptr() as *const u8, core::ptr::null::<kset_uevent_ops>(), kernel_kobj);
}
loop {
if unlikely!(unsafe { iommu_group_kset }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("782: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 782b - ., 3; .popsection\n# {0}", const 782 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_iommu_init_0, sym __slate_asm_string_iommu_init_1, const 2956 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
iommu_debugfs_setup();
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_init_784: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(iommu_init)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_set_pgtable_quirks")]
extern "C" fn iommu_set_pgtable_quirks(domain: *mut iommu_domain, quirk: u64) -> i32 {
if (unsafe { (*domain).r#type }) != 1u32 {
return 22i32.wrapping_neg();
}
if (unsafe { (*(*domain).ops).set_pgtable_quirks }) == None {
return 22i32.wrapping_neg();
}
unsafe { unsafe { (*(*domain).ops).set_pgtable_quirks }.unwrap()(domain, quirk) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_set_pgtable_quirks_785: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, u64) -> i32>, *mut ()>(Some(iommu_set_pgtable_quirks)) };

/// iommu_get_resv_regions - get reserved regions
/// @dev: device for which to get reserved regions
/// @list: reserved region list for device
///
/// This returns a list of reserved IOVA regions specific to this device.
/// A domain user should not map IOVA in these ranges.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_get_resv_regions")]
extern "C" fn iommu_get_resv_regions(dev: *mut device, list: *mut list_head) {
let ops: *const iommu_ops = dev_iommu_ops(dev);
if (unsafe { (*ops).get_resv_regions }) != None {
unsafe { unsafe { (*ops).get_resv_regions }.unwrap()(dev, list) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_get_resv_regions_786: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, *mut list_head)>, *mut ()>(Some(iommu_get_resv_regions)) };

/// iommu_put_resv_regions - release reserved regions
/// @dev: device for which to free reserved regions
/// @list: reserved region list for device
///
/// This releases a reserved region list acquired by iommu_get_resv_regions().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_put_resv_regions")]
extern "C" fn iommu_put_resv_regions(dev: *mut device, list: *mut list_head) {
let mut entry: *mut iommu_resv_region = unsafe { core::mem::zeroed() };
let mut next: *mut iommu_resv_region = unsafe { core::mem::zeroed() };
entry = list_first_entry__7!(list);
next = list_next_entry__6!(entry, list);
while list_entry_is_head!(entry, list, list) == 0 {
if (unsafe { (*entry).free }) != None {
unsafe { unsafe { (*entry).free }.unwrap()(dev, entry) };
} else {
unsafe { kfree(entry as *const ()) };
}
entry = next;
next = list_next_entry__6!(next, list);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_put_resv_regions_787: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, *mut list_head)>, *mut ()>(Some(iommu_put_resv_regions)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_alloc_resv_region")]
extern "C" fn iommu_alloc_resv_region(start: u64, length: u64, prot: i32, r#type: u32, gfp: u32) -> *mut iommu_resv_region {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut region: *mut iommu_resv_region = unsafe { core::mem::zeroed() };
let mut __v0: *mut iommu_resv_region;
let __obj_size: u64 = size_mul(core::mem::size_of::<iommu_resv_region>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, gfp, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, gfp, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut iommu_resv_region;
region = __v0;
if region.is_null() {
return core::ptr::null_mut::<iommu_resv_region>();
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*region).list) });
unsafe {
(*region).start = start;
}
unsafe {
(*region).length = length;
}
unsafe {
(*region).prot = prot;
}
unsafe {
(*region).r#type = r#type;
}
region
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_alloc_resv_region_788: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, u64, i32, u32, u32) -> *mut iommu_resv_region>, *mut ()>(Some(iommu_alloc_resv_region)) };

#[unsafe(no_mangle)]
extern "C" fn iommu_set_default_passthrough(cmd_line: bool) {
if cmd_line {
let __v0: u32 = unsafe { iommu_cmd_line };
unsafe {
iommu_cmd_line = ((__v0 as u64) | BIT!(0i32)) as u32;
}
}
unsafe {
iommu_def_domain_type = 4u32;
}
}

#[unsafe(no_mangle)]
extern "C" fn iommu_set_default_translated(cmd_line: bool) {
if cmd_line {
let __v0: u32 = unsafe { iommu_cmd_line };
unsafe {
iommu_cmd_line = ((__v0 as u64) | BIT!(0i32)) as u32;
}
}
unsafe {
iommu_def_domain_type = 3u32;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_default_passthrough")]
extern "C" fn iommu_default_passthrough() -> bool {
(unsafe { iommu_def_domain_type }) == 4u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_default_passthrough_789: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> bool>, *mut ()>(Some(iommu_default_passthrough)) };

fn iommu_from_fwnode(fwnode: *const fwnode_handle) -> *const iommu_device {
let mut iommu: *const iommu_device = unsafe { core::mem::zeroed() };
let mut ret: *const iommu_device = core::ptr::null::<iommu_device>();
spin_lock(unsafe { core::ptr::addr_of_mut!(iommu_device_lock) });
iommu = list_first_entry__49!(unsafe { core::ptr::addr_of_mut!(iommu_device_list) });
while list_entry_is_head__5!(iommu, unsafe { core::ptr::addr_of_mut!(iommu_device_list) }, list) == 0 {
if (unsafe { (*iommu).fwnode }) == (fwnode as *mut fwnode_handle) {
ret = iommu;
break;
}
iommu = list_next_entry__51!(iommu, list);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(iommu_device_lock) });
ret
}

#[unsafe(no_mangle)]
extern "C" fn iommu_ops_from_fwnode(fwnode: *const fwnode_handle) -> *const iommu_ops {
let iommu: *const iommu_device = iommu_from_fwnode(fwnode);
if !iommu.is_null() { unsafe { (*iommu).ops } } else { core::ptr::null::<iommu_ops>() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_fwspec_init_791: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, *mut fwnode_handle) -> i32>, *mut ()>(Some(iommu_fwspec_init)) };

#[unsafe(no_mangle)]
extern "C" fn iommu_fwspec_free(dev: *mut device) {
let fwspec: *mut iommu_fwspec = dev_iommu_fwspec_get(dev);
if !fwspec.is_null() {
fwnode_handle_put(unsafe { (*fwspec).iommu_fwnode });
unsafe { kfree(fwspec as *const ()) };
dev_iommu_fwspec_set(dev, core::ptr::null_mut::<iommu_fwspec>());
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_fwspec_add_ids")]
extern "C" fn iommu_fwspec_add_ids(dev: *mut device, ids: *const u32, num_ids: i32) -> i32 {
let mut fwspec: *mut iommu_fwspec = dev_iommu_fwspec_get(dev);
let mut new_num: i32 = 0;
if fwspec.is_null() {
return 22i32.wrapping_neg();
}
new_num = unsafe { (*fwspec).num_ids }.wrapping_add(num_ids as u32) as i32;
if new_num > 1 {
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = unsafe { krealloc_node_align_noprof(fwspec as *const (), size_add(16u64, size_mul(new_num as u64, (core::mem::size_of::<u32>() as u64).wrapping_add(core::mem::size_of::<__SlateRecord_iommu_c_3122_8>() as u64))), 1u64, 3264u32, 1i32.wrapping_neg()) };
alloc_tag_restore!();
} else {
_res = unsafe { krealloc_node_align_noprof(fwspec as *const (), size_add(16u64, size_mul(new_num as u64, (core::mem::size_of::<u32>() as u64).wrapping_add(core::mem::size_of::<__SlateRecord_iommu_c_3122_12>() as u64))), 1u64, 3264u32, 1i32.wrapping_neg()) };
}
__v1 = _res;
__v0 = __v1;
fwspec = __v0 as *mut iommu_fwspec;
if fwspec.is_null() {
return 12i32.wrapping_neg();
}
dev_iommu_fwspec_set(dev, fwspec);
}
for i in 0..num_ids {
unsafe {
*(core::ptr::addr_of_mut!((*fwspec).ids) as *mut u32).wrapping_offset(unsafe { (*fwspec).num_ids }.wrapping_add(i as u32) as isize) = *ids.wrapping_offset(i as isize);
}
}
unsafe {
(*fwspec).num_ids = new_num as u32;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_fwspec_add_ids_792: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, *const u32, i32) -> i32>, *mut ()>(Some(iommu_fwspec_add_ids)) };

/// iommu_setup_default_domain - Set the default_domain for the group
/// @group: Group to change
/// @target_type: Domain type to set as the default_domain
///
/// Allocate a default domain and set it as the current domain on the group. If
/// the group already has a default domain it will be changed to the target_type.
/// When target_type is 0 the default domain is selected based on driver and
/// system preferences.
fn iommu_setup_default_domain(group: *mut iommu_group, target_type: i32) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut req_type: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(req_type) as *mut i32;
let mut direct_failed: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut bool = core::ptr::addr_of_mut!(direct_failed) as *mut bool;
let mut dom: core::mem::MaybeUninit<*mut iommu_domain> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut iommu_domain = core::ptr::addr_of_mut!(dom) as *mut *mut iommu_domain;
let mut gdev: core::mem::MaybeUninit<*mut group_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut group_device = core::ptr::addr_of_mut!(gdev) as *mut *mut group_device;
let mut old_dom: *mut iommu_domain = unsafe { core::mem::zeroed() };
unsafe {
old_dom = (*group).default_domain;
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
lockdep_assert_held!(core::ptr::addr_of_mut!((*group).mutex));
*__slate_slot_4 = iommu_get_default_domain_type(group, target_type);
if *__slate_slot_4 < 0 {
return 22i32.wrapping_neg();
} else {
*__slate_slot_6 = iommu_group_alloc_default_domain(group, *__slate_slot_4);
if IS_ERR(*__slate_slot_6 as *const ()) {
return PTR_ERR(*__slate_slot_6 as *const ()) as i32;
} else {
if (*group).default_domain == *__slate_slot_6 {
return 0;
} else {
if iommu_is_dma_domain(*__slate_slot_6) {
*__slate_slot_3 = iommu_get_dma_cookie(*__slate_slot_6);
if *__slate_slot_3 != 0 {
iommu_domain_free(*__slate_slot_6);
return *__slate_slot_3;
}
}
// IOMMU_RESV_DIRECT and IOMMU_RESV_DIRECT_RELAXABLE regions must be
// mapped before their device is attached, in order to guarantee
// continuity with any FW activity
*__slate_slot_5 = false;
*__slate_slot_7 = list_first_entry__3!(core::ptr::addr_of_mut!((*group).devices));
while list_entry_is_head!(*__slate_slot_7, core::ptr::addr_of_mut!((*group).devices), list) == 0 {
if iommu_create_device_direct_mappings(*__slate_slot_6, (*(*__slate_slot_7)).dev) != 0 {
*__slate_slot_5 = true;
loop {
if !__print_once {
__print_once = true;
while false {
}
_dev_warn((*(*(*(*(*__slate_slot_7)).dev).iommu).iommu_dev).dev as *const device, c"IOMMU driver was not able to establish FW requested direct mapping.".as_ptr() as *const u8);
}
if true {
break;
}
}
}
*__slate_slot_7 = list_next_entry__4!(*__slate_slot_7, list);
}
'__join_7: {
'__join_2: {
// We must set default_domain early for __iommu_device_set_domain
(*group).default_domain = *__slate_slot_6;
if unsafe { (*group).domain }.is_null() {
// Drivers are not allowed to fail the first domain attach.
// The only way to recover from this is to fail attaching the
// iommu driver and call ops->release_device. Put the domain
// in group->default_domain so it is freed after.
*__slate_slot_3 = __iommu_group_set_domain_internal(group, *__slate_slot_6, 1u32);
__ret_warn_on = (*__slate_slot_3 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("793: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 793b - ., 3; .popsection\n# {0}", const 793 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_setup_default_domain_11, sym __slate_asm_string_iommu_setup_default_domain_12, const 3205 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("794: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 794b - ., 4; .popsection\n# {0}", const 794 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if __ret_warn_on != 0 {
break '__join_7;
}
} else {
*__slate_slot_3 = __iommu_group_set_domain(group, *__slate_slot_6);
if *__slate_slot_3 != 0 {
break '__join_2;
}
}
// Drivers are supposed to allow mappings to be installed in a domain
// before device attachment, but some don't. Hack around this defect by
// trying again after attaching. If this happens it means the device
// will not continuously have the IOMMU_RESV_DIRECT map.
if *__slate_slot_5 {
*__slate_slot_7 = list_first_entry__3!(core::ptr::addr_of_mut!((*group).devices));
loop {
if list_entry_is_head!(*__slate_slot_7, core::ptr::addr_of_mut!((*group).devices), list) == 0 {
*__slate_slot_3 = iommu_create_device_direct_mappings(*__slate_slot_6, (*(*__slate_slot_7)).dev);
if *__slate_slot_3 != 0 {
break;
} else {
*__slate_slot_7 = list_next_entry__4!(*__slate_slot_7, list);
}
} else {
break '__join_7;
}
}
if !old_dom.is_null() {
__iommu_group_set_domain_internal(group, old_dom, 1u32);
}
} else {
break '__join_7;
}
}
if !old_dom.is_null() {
iommu_domain_free(*__slate_slot_6);
(*group).default_domain = old_dom;
}
return *__slate_slot_3;
}
if !old_dom.is_null() {
iommu_domain_free(old_dom);
}
return *__slate_slot_3;
}
}
}
}
}

#[unsafe(link_section = ".data..read_mostly")]
static mut __print_once: bool = unsafe { core::mem::zeroed() };

/// Changing the default domain through sysfs requires the users to unbind the
/// drivers from the devices in the iommu group, except for a DMA -> DMA-FQ
/// transition. Return failure if this isn't met.
///
/// We need to consider the race between this and the device release path.
/// group->mutex is used here to guarantee that the device release path
/// will not be entered at the same time.
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_store_type")]
extern "C" fn iommu_group_store_type(group: *mut iommu_group, buf: *const u8, count: u64) -> i64 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut req_type: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(req_type) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut gdev: core::mem::MaybeUninit<*mut group_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut group_device = core::ptr::addr_of_mut!(gdev) as *mut *mut group_device;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !capable(21i32) {
__v4 = true;
} else {
__v4 = !capable(17i32);
}
if __v4 {
return 13i32.wrapping_neg() as i64;
} else {
__ret_warn_on = group.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("795: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 795b - ., 3; .popsection\n# {0}", const 795 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_group_store_type_10, sym __slate_asm_string_iommu_group_store_type_11, const 3262 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("796: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 796b - ., 4; .popsection\n# {0}", const 796 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v3: i64 = (__ret_warn_on != 0) as i64;
if __v3 != 0 || unsafe { (*group).default_domain }.is_null() {
return 22i32.wrapping_neg() as i64;
} else {
if sysfs_streq(buf, c"identity".as_ptr() as *const u8) {
*__slate_slot_5 = 4i32;
} else {
if sysfs_streq(buf, c"DMA".as_ptr() as *const u8) {
*__slate_slot_5 = 3i32;
} else {
if sysfs_streq(buf, c"DMA-FQ".as_ptr() as *const u8) {
*__slate_slot_5 = 11i32;
} else {
if sysfs_streq(buf, c"auto".as_ptr() as *const u8) {
*__slate_slot_5 = 0;
} else {
return 22i32.wrapping_neg() as i64;
}
}
}
}
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
// We can bring up a flush queue without tearing down the domain.
if (*__slate_slot_5 as u32) == 11u32 && (*(*group).default_domain).r#type == 3u32 {
*__slate_slot_6 = iommu_dma_init_fq((*group).default_domain);
if *__slate_slot_6 == 0 {
(*(*group).default_domain).r#type = 11u32;
*__slate_slot_6 = count as i32;
}
} else {
// Otherwise, ensure that device exists and no driver is bound.
if list_empty(core::ptr::addr_of_mut!((*group).devices) as *const list_head) != 0 || (*group).owner_cnt != 0 {
*__slate_slot_6 = 1i32.wrapping_neg();
} else {
*__slate_slot_6 = iommu_setup_default_domain(group, *__slate_slot_5);
if *__slate_slot_6 == 0 {
// Make sure dma_ops is appropriatley set
*__slate_slot_7 = list_first_entry__3!(core::ptr::addr_of_mut!((*group).devices));
while list_entry_is_head!(*__slate_slot_7, core::ptr::addr_of_mut!((*group).devices), list) == 0 {
iommu_setup_dma_ops((*(*__slate_slot_7)).dev, (*group).default_domain);
*__slate_slot_7 = list_next_entry__4!(*__slate_slot_7, list);
}
}
}
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
let __v0: i32 = *__slate_slot_6;
return (if __v0 != 0 { __v0 as u64 } else { count }) as i64;
}
}
}
}

/// iommu_device_use_default_domain() - Device driver wants to handle device
///                                     DMA through the kernel DMA API.
/// @dev: The device.
///
/// The device driver about to bind @dev wants to do DMA through the kernel
/// DMA API. Return 0 if it is allowed, otherwise an error.
#[unsafe(no_mangle)]
extern "C" fn iommu_device_use_default_domain(dev: *mut device) -> i32 {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
// Caller is the driver core during the pre-probe path
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
unsafe {
group = (*dev).iommu_group;
ret = 0;
if group.is_null() {
return 0;
} else {
'__join_0: {
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
// We may race against bus_iommu_probe() finalising groups here
if unsafe { (*group).default_domain }.is_null() {
ret = 517i32.wrapping_neg();
} else {
if (*group).owner_cnt != 0 {
if (*group).domain != (*group).default_domain || !unsafe { (*group).owner }.is_null() {
__v3 = true;
} else {
__v3 = !xa_empty(core::ptr::addr_of_mut!((*group).pasid_array) as *const xarray);
}
if __v3 {
ret = 16i32.wrapping_neg();
break '__join_0;
}
}
let __v1: u32 = (*group).owner_cnt;
(*group).owner_cnt = __v1.wrapping_add(1);
}
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
return ret;
}
}
}

/// iommu_device_unuse_default_domain() - Device driver stops handling device
///                                       DMA through the kernel DMA API.
/// @dev: The device.
///
/// The device driver doesn't want to do DMA through kernel DMA API anymore.
/// It must be called after iommu_device_use_default_domain().
#[unsafe(no_mangle)]
extern "C" fn iommu_device_unuse_default_domain(dev: *mut device) {
// Caller is the driver core during the post-probe path
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
if group.is_null() {
return;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = (unsafe { (*group).owner_cnt }) == 0 || !xa_empty((unsafe { core::ptr::addr_of_mut!((*group).pasid_array) }) as *const xarray);
__ret_warn_on = __v1 as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("797: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 797b - ., 3; .popsection\n# {0}", const 797 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_device_unuse_default_domain_2, sym __slate_asm_string_iommu_device_unuse_default_domain_3, const 3363 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("798: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 798b - ., 4; .popsection\n# {0}", const 798 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 == 0 {
let __v7: u32 = unsafe { (*group).owner_cnt };
unsafe {
(*group).owner_cnt = __v7.wrapping_sub(1);
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
}

fn __iommu_group_alloc_blocking_domain(group: *mut iommu_group) -> i32 {
let dev: *mut device = iommu_group_first_dev(group);
let ops: *const iommu_ops = dev_iommu_ops(dev);
if !unsafe { (*group).blocking_domain }.is_null() {
return 0;
}
if !unsafe { (*ops).blocked_domain }.is_null() {
unsafe {
(*group).blocking_domain = (*ops).blocked_domain;
}
return 0;
}
// For drivers that do not yet understand IOMMU_DOMAIN_BLOCKED create an
// empty PAGING domain instead.
let domain: *mut iommu_domain = iommu_paging_domain_alloc(dev);
if IS_ERR(domain as *const ()) {
return PTR_ERR(domain as *const ()) as i32;
}
unsafe {
(*group).blocking_domain = domain;
}
0
}

fn __iommu_take_dma_ownership(group: *mut iommu_group, owner: *mut ()) -> i32 {
let mut ret: i32 = 0;
let __v0: bool = !unsafe { (*group).domain }.is_null() && (unsafe { (*group).domain }) != unsafe { (*group).default_domain } || !xa_empty((unsafe { core::ptr::addr_of_mut!((*group).pasid_array) }) as *const xarray);
if __v0 {
return 16i32.wrapping_neg();
}
ret = __iommu_group_alloc_blocking_domain(group);
if ret != 0 {
return ret;
}
ret = __iommu_group_set_domain(group, unsafe { (*group).blocking_domain });
if ret != 0 {
return ret;
}
unsafe {
(*group).owner = owner;
}
let __v2: u32 = unsafe { (*group).owner_cnt };
unsafe {
(*group).owner_cnt = __v2.wrapping_add(1);
}
0
}

/// iommu_group_claim_dma_owner() - Set DMA ownership of a group
/// @group: The group.
/// @owner: Caller specified pointer. Used for exclusive ownership.
///
/// This is to support backward compatibility for vfio which manages the dma
/// ownership in iommu_group level. New invocations on this interface should be
/// prohibited. Only a single owner may exist for a group.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_claim_dma_owner")]
extern "C" fn iommu_group_claim_dma_owner(group: *mut iommu_group, owner: *mut ()) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
unsafe {
ret = 0;
__ret_warn_on = owner.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("799: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 799b - ., 3; .popsection\n# {0}", const 799 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_group_claim_dma_owner_4, sym __slate_asm_string_iommu_group_claim_dma_owner_5, const 3427 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("800: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 800b - ., 4; .popsection\n# {0}", const 800 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return 22i32.wrapping_neg();
} else {
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
if (*group).owner_cnt != 0 {
ret = 1i32.wrapping_neg();
} else {
ret = __iommu_take_dma_ownership(group, owner);
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
return ret;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_claim_dma_owner_801: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group, *mut ()) -> i32>, *mut ()>(Some(iommu_group_claim_dma_owner)) };

/// iommu_device_claim_dma_owner() - Set DMA ownership of a device
/// @dev: The device.
/// @owner: Caller specified pointer. Used for exclusive ownership.
///
/// Claim the DMA ownership of a device. Multiple devices in the same group may
/// concurrently claim ownership if they present the same owner value. Returns 0
/// on success and error code on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_device_claim_dma_owner")]
extern "C" fn iommu_device_claim_dma_owner(dev: *mut device, owner: *mut ()) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
// Caller must be a probed driver on dev
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
unsafe {
group = (*dev).iommu_group;
ret = 0;
__ret_warn_on = owner.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("802: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 802b - ., 3; .popsection\n# {0}", const 802 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_device_claim_dma_owner_8, sym __slate_asm_string_iommu_device_claim_dma_owner_9, const 3459 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("803: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 803b - ., 4; .popsection\n# {0}", const 803 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v5: i64 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
return 22i32.wrapping_neg();
} else {
if group.is_null() {
return 19i32.wrapping_neg();
} else {
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
if (*group).owner_cnt != 0 {
if (*group).owner != owner {
ret = 1i32.wrapping_neg();
} else {
let __v1: u32 = (*group).owner_cnt;
(*group).owner_cnt = __v1.wrapping_add(1);
}
} else {
ret = __iommu_take_dma_ownership(group, owner);
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
return ret;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_device_claim_dma_owner_804: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, *mut ()) -> i32>, *mut ()>(Some(iommu_device_claim_dma_owner)) };

fn __iommu_release_dma_ownership(group: *mut iommu_group) {
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = (unsafe { (*group).owner_cnt }) == 0 || unsafe { (*group).owner }.is_null() || !xa_empty((unsafe { core::ptr::addr_of_mut!((*group).pasid_array) }) as *const xarray);
__ret_warn_on = __v1 as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("805: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 805b - ., 3; .popsection\n# {0}", const 805 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_release_dma_ownership_2, sym __slate_asm_string___iommu_release_dma_ownership_3, const 3485 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("806: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 806b - ., 4; .popsection\n# {0}", const 806 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return;
}
unsafe {
(*group).owner_cnt = 0;
}
unsafe {
(*group).owner = core::ptr::null_mut::<()>();
}
__iommu_group_set_domain_nofail(group, unsafe { (*group).default_domain });
}

/// iommu_group_release_dma_owner() - Release DMA ownership of a group
/// @group: The group
///
/// Release the DMA ownership claimed by iommu_group_claim_dma_owner().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_release_dma_owner")]
extern "C" fn iommu_group_release_dma_owner(group: *mut iommu_group) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
__iommu_release_dma_ownership(group);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_release_dma_owner_807: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group)>, *mut ()>(Some(iommu_group_release_dma_owner)) };

/// iommu_device_release_dma_owner() - Release DMA ownership of a device
/// @dev: The device.
///
/// Release the DMA ownership claimed by iommu_device_claim_dma_owner().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_device_release_dma_owner")]
extern "C" fn iommu_device_release_dma_owner(dev: *mut device) {
// Caller must be a probed driver on dev
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
if (unsafe { (*group).owner_cnt }) > 1 {
let __v1: u32 = unsafe { (*group).owner_cnt };
unsafe {
(*group).owner_cnt = __v1.wrapping_sub(1);
}
} else {
__iommu_release_dma_ownership(group);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_device_release_dma_owner_808: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device)>, *mut ()>(Some(iommu_device_release_dma_owner)) };

/// iommu_group_dma_owner_claimed() - Query group dma ownership status
/// @group: The group.
///
/// This provides status query on a given group. It is racy and only for
/// non-binding status reporting.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_group_dma_owner_claimed")]
extern "C" fn iommu_group_dma_owner_claimed(group: *mut iommu_group) -> bool {
let mut user: u32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
user = unsafe { (*group).owner_cnt };
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
user != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_group_dma_owner_claimed_809: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group) -> bool>, *mut ()>(Some(iommu_group_dma_owner_claimed)) };

fn iommu_remove_dev_pasid(dev: *mut device, pasid: u32, domain: *mut iommu_domain) {
let ops: *const iommu_ops = dev_iommu_ops(dev);
let blocked_domain: *mut iommu_domain = unsafe { (*ops).blocked_domain };
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { unsafe { (*(*blocked_domain).ops).set_dev_pasid }.unwrap()(blocked_domain, dev, pasid, domain) }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("810: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 810b - ., 3; .popsection\n# {0}", const 810 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_remove_dev_pasid_1, sym __slate_asm_string_iommu_remove_dev_pasid_2, const 3553 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("811: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 811b - ., 4; .popsection\n# {0}", const 811 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

fn __iommu_set_group_pasid(domain: *mut iommu_domain, group: *mut iommu_group, pasid: u32, old: *mut iommu_domain) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
// If no old domain, undo the succeeded devices/pasid.
// Otherwise, rollback the succeeded devices/pasid to
// the old domain. And it is a driver bug to fail
// attaching with a previously good domain.
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut last_gdev: core::mem::MaybeUninit<*mut group_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut group_device = core::ptr::addr_of_mut!(last_gdev) as *mut *mut group_device;
let mut device: core::mem::MaybeUninit<*mut group_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut group_device = core::ptr::addr_of_mut!(device) as *mut *mut group_device;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_6 = list_first_entry__3!(core::ptr::addr_of_mut!((*group).devices));
'__join_14: {
while list_entry_is_head!(*__slate_slot_6, core::ptr::addr_of_mut!((*group).devices), list) == 0 {
if (*(*(*(*__slate_slot_6)).dev).iommu).max_pasids > 0 {
*__slate_slot_4 = unsafe { (*(*domain).ops).set_dev_pasid }.unwrap()(domain, (*(*__slate_slot_6)).dev, pasid, old);
if *__slate_slot_4 != 0 {
break '__join_14;
}
}
*__slate_slot_6 = list_next_entry__4!(*__slate_slot_6, list);
}
return 0;
}
*__slate_slot_5 = *__slate_slot_6;
*__slate_slot_6 = list_first_entry__3!(core::ptr::addr_of_mut!((*group).devices));
while list_entry_is_head!(*__slate_slot_6, core::ptr::addr_of_mut!((*group).devices), list) == 0 {
if *__slate_slot_6 == *__slate_slot_5 {
break;
} else {
if (*(*(*(*__slate_slot_6)).dev).iommu).max_pasids > 0 {
if old.is_null() {
__v3 = true;
} else {
__ret_warn_on = (unsafe { (*(*old).ops).set_dev_pasid }.unwrap()(old, (*(*__slate_slot_6)).dev, pasid, domain) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("812: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 812b - ., 3; .popsection\n# {0}", const 812 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___iommu_set_group_pasid_8, sym __slate_asm_string___iommu_set_group_pasid_9, const 3588 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("813: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 813b - ., 4; .popsection\n# {0}", const 813 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
__v3 = __ret_warn_on != 0;
}
if __v3 {
iommu_remove_dev_pasid((*(*__slate_slot_6)).dev, pasid, domain);
}
}
*__slate_slot_6 = list_next_entry__4!(*__slate_slot_6, list);
}
}
return *__slate_slot_4;
}
}

fn __iommu_remove_group_pasid(group: *mut iommu_group, pasid: u32, domain: *mut iommu_domain) {
let mut device: *mut group_device = unsafe { core::mem::zeroed() };
device = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(device, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
// A group-level detach cannot fail, even if there is a blocked
// device. In fact, blocked devices must be already detached for
// a pending device recovery.
if !unsafe { (*device).blocked } && (unsafe { (*(*(*device).dev).iommu).max_pasids }) > 0 {
iommu_remove_dev_pasid(unsafe { (*device).dev }, pasid, domain);
}
device = list_next_entry__4!(device, list);
}
}

/// iommu_attach_device_pasid() - Attach a domain to pasid of device
/// @domain: the iommu domain.
/// @dev: the attached device.
/// @pasid: the pasid of the device.
/// @handle: the attach handle.
///
/// Caller should always provide a new handle to avoid race with the paths
/// that have lockless reference to handle if it intends to pass a valid handle.
///
/// Return: 0 on success, or an error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_attach_device_pasid")]
extern "C" fn iommu_attach_device_pasid(domain: *mut iommu_domain, dev: *mut device, pasid: u32, handle: *mut iommu_attach_handle) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
let mut ops: core::mem::MaybeUninit<*const iommu_ops> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *const iommu_ops = core::ptr::addr_of_mut!(ops) as *mut *const iommu_ops;
let mut device: core::mem::MaybeUninit<*mut group_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut group_device = core::ptr::addr_of_mut!(device) as *mut *mut group_device;
// Caller must be a probed driver on dev
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
unsafe {
group = (*dev).iommu_group;
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if group.is_null() {
return 19i32.wrapping_neg();
} else {
*__slate_slot_5 = dev_iommu_ops(dev);
if (*(*domain).ops).set_dev_pasid == None || unsafe { (*(*__slate_slot_5)).blocked_domain }.is_null() || (*(*(*(*__slate_slot_5)).blocked_domain).ops).set_dev_pasid == None {
return 95i32.wrapping_neg();
} else {
if !domain_iommu_ops_compatible(*__slate_slot_5, domain) || pasid == 0 {
return 22i32.wrapping_neg();
} else {
'__join_0: {
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
// This is a concurrent attach during device recovery. Reject it until
// pci_dev_reset_iommu_done() attaches the device to group->domain.
if (*group).recovery_cnt != 0 {
*__slate_slot_3 = 16i32.wrapping_neg();
} else {
*__slate_slot_6 = list_first_entry__3!(core::ptr::addr_of_mut!((*group).devices));
'__join_11: {
while list_entry_is_head!(*__slate_slot_6, core::ptr::addr_of_mut!((*group).devices), list) == 0 {
// Skip PASID validation for devices without PASID support
// (max_pasids = 0). These devices cannot issue transactions
// with PASID, so they don't affect group's PASID usage.
if (*(*(*(*__slate_slot_6)).dev).iommu).max_pasids > 0 && pasid >= (*(*(*(*__slate_slot_6)).dev).iommu).max_pasids {
break '__join_11;
} else {
*__slate_slot_6 = list_next_entry__4!(*__slate_slot_6, list);
}
}
*__slate_slot_4 = iommu_make_pasid_array_entry(domain, handle);
// Entry present is a failure case. Use xa_insert() instead of
// xa_reserve().
*__slate_slot_3 = xa_insert(core::ptr::addr_of_mut!((*group).pasid_array), pasid as u64, xa_mk_internal(257u64), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if *__slate_slot_3 != 0 {
break '__join_0;
} else {
*__slate_slot_3 = __iommu_set_group_pasid(domain, group, pasid, core::ptr::null_mut::<iommu_domain>());
if *__slate_slot_3 != 0 {
xa_release(core::ptr::addr_of_mut!((*group).pasid_array), pasid as u64);
break '__join_0;
} else {
let __ret_warn_on: i32 = !(!xa_is_err(xa_store(core::ptr::addr_of_mut!((*group).pasid_array), pasid as u64, *__slate_slot_4, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("814: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 814b - ., 3; .popsection\n# {0}", const 814 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_attach_device_pasid_8, sym __slate_asm_string_iommu_attach_device_pasid_9, const 3696 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("815: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 815b - ., 4; .popsection\n# {0}", const 815 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// The xa_insert() above reserved the memory, and the group->mutex is
// held, this cannot fail. The new domain cannot be visible until the
// operation succeeds as we cannot tolerate PRIs becoming concurrently
// queued and then failing attach.
break '__join_0;
}
}
}
*__slate_slot_3 = 22i32.wrapping_neg();
}
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
return *__slate_slot_3;
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_attach_device_pasid_816: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u32, *mut iommu_attach_handle) -> i32>, *mut ()>(Some(iommu_attach_device_pasid)) };

/// iommu_replace_device_pasid - Replace the domain that a specific pasid
///                              of the device is attached to
/// @domain: the new iommu domain
/// @dev: the attached device.
/// @pasid: the pasid of the device.
/// @handle: the attach handle.
///
/// This API allows the pasid to switch domains. The @pasid should have been
/// attached. Otherwise, this fails. The pasid will keep the old configuration
/// if replacement failed.
///
/// Caller should always provide a new handle to avoid race with the paths
/// that have lockless reference to handle if it intends to pass a valid handle.
///
/// Return 0 on success, or an error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_replace_device_pasid")]
extern "C" fn iommu_replace_device_pasid(domain: *mut iommu_domain, dev: *mut device, pasid: u32, handle: *mut iommu_attach_handle) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut curr: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut () = core::ptr::addr_of_mut!(curr) as *mut *mut ();
let mut curr_domain: core::mem::MaybeUninit<*mut iommu_domain> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut iommu_domain = core::ptr::addr_of_mut!(curr_domain) as *mut *mut iommu_domain;
let mut entry: core::mem::MaybeUninit<*mut iommu_attach_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut iommu_attach_handle = core::ptr::addr_of_mut!(entry) as *mut *mut iommu_attach_handle;
// Caller must be a probed driver on dev
let mut group: *mut iommu_group = unsafe { core::mem::zeroed() };
unsafe {
group = (*dev).iommu_group;
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if group.is_null() {
return 19i32.wrapping_neg();
} else {
if (*(*domain).ops).set_dev_pasid == None {
return 95i32.wrapping_neg();
} else {
if !domain_iommu_ops_compatible(dev_iommu_ops(dev), domain) || pasid == 0 || handle.is_null() {
return 22i32.wrapping_neg();
} else {
'__join_0: {
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
// This is a concurrent attach during device recovery. Reject it until
// pci_dev_reset_iommu_done() attaches the device to group->domain.
if (*group).recovery_cnt != 0 {
*__slate_slot_6 = 16i32.wrapping_neg();
} else {
*__slate_slot_9 = iommu_make_pasid_array_entry(domain, handle) as *mut iommu_attach_handle;
*__slate_slot_7 = xa_cmpxchg(core::ptr::addr_of_mut!((*group).pasid_array), pasid as u64, core::ptr::null_mut::<()>(), xa_mk_internal(257u64), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if xa_is_err(*__slate_slot_7 as *const ()) {
*__slate_slot_6 = xa_err(*__slate_slot_7);
} else {
// No domain (with or without handle) attached, hence not
// a replace case.
if (*__slate_slot_7).is_null() {
xa_release(core::ptr::addr_of_mut!((*group).pasid_array), pasid as u64);
*__slate_slot_6 = 22i32.wrapping_neg();
} else {
// Reusing handle is problematic as there are paths that refers
// the handle without lock. To avoid race, reject the callers that
// attempt it.
if *__slate_slot_7 == (*__slate_slot_9 as *mut ()) {
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("817: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 817b - ., 3; .popsection\n# {0}", const 817 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_replace_device_pasid_13, sym __slate_asm_string_iommu_replace_device_pasid_14, const 3777 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("818: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 818b - ., 4; .popsection\n# {0}", const 818 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
*__slate_slot_6 = 22i32.wrapping_neg();
} else {
*__slate_slot_8 = pasid_array_entry_to_domain(*__slate_slot_7);
*__slate_slot_6 = 0;
if *__slate_slot_8 != domain {
*__slate_slot_6 = __iommu_set_group_pasid(domain, group, pasid, *__slate_slot_8);
if *__slate_slot_6 != 0 {
break '__join_0;
}
}
let __ret_warn_on: i32 = !(!xa_is_err(xa_store(core::ptr::addr_of_mut!((*group).pasid_array), pasid as u64, *__slate_slot_9 as *mut (), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("819: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 819b - ., 3; .popsection\n# {0}", const 819 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_replace_device_pasid_11, sym __slate_asm_string_iommu_replace_device_pasid_12, const 3797 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("820: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 820b - ., 4; .popsection\n# {0}", const 820 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// The above xa_cmpxchg() reserved the memory, and the
// group->mutex is held, this cannot fail.
}
}
}
}
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
return *__slate_slot_6;
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_replace_device_pasid_821: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u32, *mut iommu_attach_handle) -> i32>, *mut ()>(Some(iommu_replace_device_pasid)) };

/// iommu_detach_device_pasid() - Detach the domain from pasid of device
/// @domain: the iommu domain.
/// @dev: the attached device.
/// @pasid: the pasid of the device.
///
/// The @domain must have been attached to @pasid of the @dev with
/// iommu_attach_device_pasid().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_detach_device_pasid")]
extern "C" fn iommu_detach_device_pasid(domain: *mut iommu_domain, dev: *mut device, pasid: u32) {
// Caller must be a probed driver on dev
let group: *mut iommu_group = unsafe { (*dev).iommu_group };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
__iommu_remove_group_pasid(group, pasid, domain);
unsafe { xa_erase(core::ptr::addr_of_mut!((*group).pasid_array), pasid as u64) };
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_detach_device_pasid_822: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u32)>, *mut ()>(Some(iommu_detach_device_pasid)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_alloc_global_pasid")]
extern "C" fn iommu_alloc_global_pasid(dev: *mut device) -> u32 {
let mut ret: i32 = 0;
// max_pasids == 0 means that the device does not support PASID
if (unsafe { (*(*dev).iommu).max_pasids }) == 0 {
return 1u32.wrapping_neg();
}
// max_pasids is set up by vendor driver based on number of PASID bits
// supported but the IDA allocation is inclusive.
ret = unsafe { ida_alloc_range(core::ptr::addr_of_mut!(iommu_global_pasid_ida), 1u32, unsafe { (*(*dev).iommu).max_pasids }.wrapping_sub(1), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
if ret < 0 { 1u32.wrapping_neg() } else { ret as u32 }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_alloc_global_pasid_823: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device) -> u32>, *mut ()>(Some(iommu_alloc_global_pasid)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_free_global_pasid")]
extern "C" fn iommu_free_global_pasid(pasid: u32) {
let mut __v0: i64;
let __ret_warn_on: i32 = (pasid == 1u32.wrapping_neg()) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("824: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 824b - ., 3; .popsection\n# {0}", const 824 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_free_global_pasid_1, sym __slate_asm_string_iommu_free_global_pasid_2, const 3847 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("825: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 825b - ., 4; .popsection\n# {0}", const 825 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return;
}
unsafe { ida_free(core::ptr::addr_of_mut!(iommu_global_pasid_ida), pasid) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_free_global_pasid_826: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32)>, *mut ()>(Some(iommu_free_global_pasid)) };

/// iommu_attach_handle_get - Return the attach handle
/// @group: the iommu group that domain was attached to
/// @pasid: the pasid within the group
/// @type: matched domain type, 0 for any match
///
/// Return handle or ERR_PTR(-ENOENT) on none, ERR_PTR(-EBUSY) on mismatch.
///
/// Return the attach handle to the caller. The life cycle of an iommu attach
/// handle is from the time when the domain is attached to the time when the
/// domain is detached. Callers are required to synchronize the call of
/// iommu_attach_handle_get() with domain attachment and detachment. The attach
/// handle can only be used during its life cycle.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_attach_handle_get")]
extern "C" fn iommu_attach_handle_get(group: *mut iommu_group, pasid: u32, r#type: u32) -> *mut iommu_attach_handle {
let mut handle: *mut iommu_attach_handle = unsafe { core::mem::zeroed() };
xa_lock!(unsafe { core::ptr::addr_of_mut!((*group).pasid_array) });
let entry: *mut () = unsafe { xa_load(core::ptr::addr_of_mut!((*group).pasid_array), pasid as u64) };
let __v0: bool = entry.is_null() || xa_pointer_tag(entry) != 1;
if __v0 {
handle = ERR_PTR(2i32.wrapping_neg() as i64) as *mut iommu_attach_handle;
} else {
handle = xa_untag_pointer(entry) as *mut iommu_attach_handle;
if r#type != 0 && (unsafe { (*(*handle).domain).r#type }) != r#type {
handle = ERR_PTR(16i32.wrapping_neg() as i64) as *mut iommu_attach_handle;
}
}
xa_unlock!(unsafe { core::ptr::addr_of_mut!((*group).pasid_array) });
handle
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_attach_handle_get_827: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group, u32, u32) -> *mut iommu_attach_handle>, *mut ()>(Some(iommu_attach_handle_get)) };

/// iommu_attach_group_handle - Attach an IOMMU domain to an IOMMU group
/// @domain: IOMMU domain to attach
/// @group: IOMMU group that will be attached
/// @handle: attach handle
///
/// Returns 0 on success and error code on failure.
///
/// This is a variant of iommu_attach_group(). It allows the caller to provide
/// an attach handle and use it when the domain is attached. This is currently
/// used by IOMMUFD to deliver the I/O page faults.
///
/// Caller should always provide a new handle to avoid race with the paths
/// that have lockless reference to handle.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_attach_group_handle")]
extern "C" fn iommu_attach_group_handle(domain: *mut iommu_domain, group: *mut iommu_group, handle: *mut iommu_attach_handle) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if handle.is_null() {
return 22i32.wrapping_neg();
} else {
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
*__slate_slot_4 = iommu_make_pasid_array_entry(domain, handle);
*__slate_slot_3 = xa_insert(core::ptr::addr_of_mut!((*group).pasid_array), 0u64, xa_mk_internal(257u64), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if *__slate_slot_3 == 0 {
*__slate_slot_3 = __iommu_attach_group(domain, group);
if *__slate_slot_3 != 0 {
xa_release(core::ptr::addr_of_mut!((*group).pasid_array), 0u64);
} else {
let __ret_warn_on: i32 = !(!xa_is_err(xa_store(core::ptr::addr_of_mut!((*group).pasid_array), 0u64, *__slate_slot_4, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("828: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 828b - ., 3; .popsection\n# {0}", const 828 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_attach_group_handle_5, sym __slate_asm_string_iommu_attach_group_handle_6, const 3934 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("829: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 829b - ., 4; .popsection\n# {0}", const 829 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// The xa_insert() above reserved the memory, and the group->mutex is
// held, this cannot fail. The new domain cannot be visible until the
// operation succeeds as we cannot tolerate PRIs becoming concurrently
// queued and then failing attach.
}
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
return *__slate_slot_3;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_attach_group_handle_830: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_group, *mut iommu_attach_handle) -> i32>, *mut ()>(Some(iommu_attach_group_handle)) };

/// iommu_detach_group_handle - Detach an IOMMU domain from an IOMMU group
/// @domain: IOMMU domain to attach
/// @group: IOMMU group that will be attached
///
/// Detach the specified IOMMU domain from the specified IOMMU group.
/// It must be used in conjunction with iommu_attach_group_handle().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_detach_group_handle")]
extern "C" fn iommu_detach_group_handle(domain: *mut iommu_domain, group: *mut iommu_group) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*group).mutex)) };
__iommu_group_set_core_domain(group);
unsafe { xa_erase(core::ptr::addr_of_mut!((*group).pasid_array), 0u64) };
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*group).mutex)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_detach_group_handle_831: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_group)>, *mut ()>(Some(iommu_detach_group_handle)) };

/// iommu_replace_group_handle - replace the domain that a group is attached to
/// @group: IOMMU group that will be attached to the new domain
/// @new_domain: new IOMMU domain to replace with
/// @handle: attach handle
///
/// This API allows the group to switch domains without being forced to go to
/// the blocking domain in-between. It allows the caller to provide an attach
/// handle for the new domain and use it when the domain is attached.
///
/// If the currently attached domain is a core domain (e.g. a default_domain),
/// it will act just like the iommu_attach_group_handle().
///
/// Caller should always provide a new handle to avoid race with the paths
/// that have lockless reference to handle.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.iommu_replace_group_handle")]
extern "C" fn iommu_replace_group_handle(group: *mut iommu_group, new_domain: *mut iommu_domain, handle: *mut iommu_attach_handle) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
let mut curr: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(curr) as *mut *mut ();
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if new_domain.is_null() || handle.is_null() {
return 22i32.wrapping_neg();
} else {
mutex_lock(core::ptr::addr_of_mut!((*group).mutex));
*__slate_slot_4 = iommu_make_pasid_array_entry(new_domain, handle);
*__slate_slot_3 = xa_reserve(core::ptr::addr_of_mut!((*group).pasid_array), 0u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if *__slate_slot_3 == 0 {
*__slate_slot_3 = __iommu_group_set_domain(group, new_domain);
if *__slate_slot_3 != 0 {
xa_release(core::ptr::addr_of_mut!((*group).pasid_array), 0u64);
} else {
*__slate_slot_5 = xa_store(core::ptr::addr_of_mut!((*group).pasid_array), 0u64, *__slate_slot_4, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
let __ret_warn_on: i32 = !(!xa_is_err(*__slate_slot_5 as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("832: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 832b - ., 3; .popsection\n# {0}", const 832 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_iommu_replace_group_handle_6, sym __slate_asm_string_iommu_replace_group_handle_7, const 3997 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("833: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 833b - ., 4; .popsection\n# {0}", const 833 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
return 0;
}
}
mutex_unlock(core::ptr::addr_of_mut!((*group).mutex));
return *__slate_slot_3;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_iommu_replace_group_handle_834: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut iommu_group, *mut iommu_domain, *mut iommu_attach_handle) -> i32>, *mut ()>(Some(iommu_replace_group_handle)) };

/// pci_dev_reset_iommu_prepare() - Block IOMMU to prepare for a PCI device reset
/// @pdev: PCI device that is going to enter a reset routine
///
/// The PCIe r6.0, sec 10.3.1 IMPLEMENTATION NOTE recommends to disable and block
/// ATS before initiating a reset. This means that a PCIe device during the reset
/// routine wants to block any IOMMU activity: translation and ATS invalidation.
///
/// This function attaches the device's RID/PASID(s) the group->blocking_domain,
/// incrementing the group->recovery_cnt, to allow the IOMMU driver pausing any
/// IOMMU activity while leaving the group->domain pointer intact. Later when the
/// reset is finished, pci_dev_reset_iommu_done() can restore everything.
///
/// Caller must use pci_dev_reset_iommu_prepare() with pci_dev_reset_iommu_done()
/// before/after the core-level reset routine, to decrement the recovery_cnt.
///
/// Return: 0 on success or negative error code if the preparation failed.
///
/// These two functions are designed to be used by PCI reset functions that would
/// not invoke any racy iommu_release_device(), since PCI sysfs node gets removed
/// before it notifies with a BUS_NOTIFY_REMOVED_DEVICE. When using them in other
/// case, callers must ensure there will be no racy iommu_release_device() call,
/// which otherwise would UAF the dev->iommu_group pointer.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.pci_dev_reset_iommu_prepare")]
extern "C" fn pci_dev_reset_iommu_prepare(pdev: *mut pci_dev) -> i32 {
let group: *mut iommu_group = unsafe { (*pdev).dev.iommu_group };
let mut pasid: u64 = 0;
let mut entry: *mut () = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let __v0: bool = !unsafe { pci_ats_supported(pdev) } || !dev_has_iommu(unsafe { core::ptr::addr_of_mut!((*pdev).dev) });
if __v0 {
return 0;
}
let mut __UNIQUE_ID_guard_835: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_835) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_unlock_836: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*group).mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_836) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _));
let gdev: *mut group_device = __dev_to_gdev(unsafe { core::ptr::addr_of_mut!((*pdev).dev) });
let mut __v3: i64;
let __ret_warn_on: i32 = gdev.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("837: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 837b - ., 3; .popsection\n# {0}", const 837 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_dev_reset_iommu_prepare_4, sym __slate_asm_string_pci_dev_reset_iommu_prepare_5, const 4048 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("838: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 838b - ., 4; .popsection\n# {0}", const 838 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
if __v3 != 0 {
return 19i32.wrapping_neg();
}
let __v9: u32 = unsafe { (*gdev).reset_depth };
unsafe {
(*gdev).reset_depth = __v9.wrapping_add(1);
}
if __v9 != 0 {
return 0;
}
ret = __iommu_group_alloc_blocking_domain(group);
if ret != 0 {
let __v12: u32 = unsafe { (*gdev).reset_depth };
unsafe {
(*gdev).reset_depth = __v12.wrapping_sub(1);
}
return ret;
}
// Stage RID domain at blocking_domain while retaining group->domain
if (unsafe { (*group).domain }) != unsafe { (*group).blocking_domain } {
ret = __iommu_attach_device(unsafe { (*group).blocking_domain }, unsafe { core::ptr::addr_of_mut!((*pdev).dev) }, unsafe { (*group).domain });
if ret != 0 {
let __v15: u32 = unsafe { (*gdev).reset_depth };
unsafe {
(*gdev).reset_depth = __v15.wrapping_sub(1);
}
return ret;
}
}
// Update gdev->blocked upon the domain change, as it is used to return
// the correct domain in iommu_driver_get_domain_for_dev() that might be
// called in a set_dev_pasid callback function.
unsafe {
(*gdev).blocked = true;
}
// Stage PASID domains at blocking_domain while retaining pasid_array.
//
// The pasid_array is mostly fenced by group->mutex, except one reader
// in iommu_attach_handle_get(), so it's safe to read without xa_lock.
if (unsafe { (*(*pdev).dev.iommu).max_pasids }) > 0 {
pasid = 1;
entry = unsafe { xa_find(core::ptr::addr_of_mut!((*group).pasid_array), core::ptr::addr_of_mut!(pasid), 18446744073709551615u64, 8u32) };
while !entry.is_null() {
let pasid_dom: *mut iommu_domain = pasid_array_entry_to_domain(entry);
iommu_remove_dev_pasid(unsafe { core::ptr::addr_of_mut!((*pdev).dev) }, pasid as u32, pasid_dom);
entry = unsafe { xa_find_after(core::ptr::addr_of_mut!((*group).pasid_array), core::ptr::addr_of_mut!(pasid), 18446744073709551615u64, 8u32) };
}
}
let __v20: u32 = unsafe { (*group).recovery_cnt };
unsafe {
(*group).recovery_cnt = __v20.wrapping_add(1);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_dev_reset_iommu_prepare_839: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_dev_reset_iommu_prepare)) };

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.__group_device_cmp_dma_alias")]
extern "C" fn __group_device_cmp_dma_alias(dev: *mut pci_dev, alias: u16, data: *mut ()) -> i32 {
((alias as i32) == ((unsafe { *(data as *mut u16) }) as i32)) as i32
}

#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.group_device_cmp_dma_alias")]
extern "C" fn group_device_cmp_dma_alias(dev: *mut pci_dev, mut alias: u16, data: *mut ()) -> i32 {
unsafe { pci_for_each_dma_alias(data as *mut pci_dev, Some(__group_device_cmp_dma_alias), core::ptr::addr_of_mut!(alias) as *mut ()) }
}

fn group_device_dma_alias_is_blocked(group: *mut iommu_group, gdev: *mut group_device) -> bool {
let mut sibling: *mut group_device = unsafe { core::mem::zeroed() };
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
if !dev_is_pci!(unsafe { (*gdev).dev }) {
return false;
}
sibling = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*group).devices) });
while list_entry_is_head!(sibling, unsafe { core::ptr::addr_of_mut!((*group).devices) }, list) == 0 {
if !(sibling == gdev || !unsafe { (*sibling).blocked } || !dev_is_pci!(unsafe { (*sibling).dev })) {
if (unsafe { pci_for_each_dma_alias(to_pci_dev!(unsafe { (*gdev).dev }), Some(group_device_cmp_dma_alias), to_pci_dev!(unsafe { (*sibling).dev }) as *mut ()) }) != 0 {
return true;
}
}
sibling = list_next_entry__4!(sibling, list);
}
false
}

/// pci_dev_reset_iommu_done() - Restore IOMMU after a PCI device reset is done
/// @pdev: PCI device that has finished a reset routine
///
/// After a PCIe device finishes a reset routine, it wants to restore its IOMMU
/// activity, including new translation and cache invalidation, by re-attaching
/// all RID/PASID of the device back to the domains retained in the core-level
/// structure.
///
/// Caller must pair it with a successful pci_dev_reset_iommu_prepare().
///
/// Note that, although unlikely, there is a risk that re-attaching domains might
/// fail due to some unexpected happening like OOM.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.iommu_iommu.pci_dev_reset_iommu_done")]
extern "C" fn pci_dev_reset_iommu_done(pdev: *mut pci_dev) {
let group: *mut iommu_group = unsafe { (*pdev).dev.iommu_group };
let mut pasid: u64 = 0;
let mut entry: *mut () = unsafe { core::mem::zeroed() };
let __v0: bool = !unsafe { pci_ats_supported(pdev) } || !dev_has_iommu(unsafe { core::ptr::addr_of_mut!((*pdev).dev) });
if __v0 {
return;
}
let mut __UNIQUE_ID_guard_840: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*group).mutex) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_840) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_unlock_841: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*group).mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_841) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _));
let gdev: *mut group_device = __dev_to_gdev(unsafe { core::ptr::addr_of_mut!((*pdev).dev) });
let mut __v3: i64;
{
let __ret_warn_on: i32 = gdev.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("842: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 842b - ., 3; .popsection\n# {0}", const 842 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_dev_reset_iommu_done_4, sym __slate_asm_string_pci_dev_reset_iommu_done_5, const 4159 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("843: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 843b - ., 4; .popsection\n# {0}", const 843 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
}
if __v3 != 0 {
return;
}
// Unbalanced done() calls would underflow the counter
let mut __v8: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*gdev).reset_depth }) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("844: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 844b - ., 3; .popsection\n# {0}", const 844 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_dev_reset_iommu_done_9, sym __slate_asm_string_pci_dev_reset_iommu_done_10, const 4163 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("845: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 845b - ., 4; .popsection\n# {0}", const 845 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v8 = (__ret_warn_on != 0) as i64;
}
if __v8 != 0 {
return;
}
let __v14: u32 = unsafe { (*gdev).reset_depth };
let __v15: u32 = __v14.wrapping_sub(1);
unsafe {
(*gdev).reset_depth = __v15;
}
if __v15 != 0 {
return;
}
let mut __v16: i64;
{
let __ret_warn_on: i32 = unsafe { (*group).blocking_domain }.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("846: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 846b - ., 3; .popsection\n# {0}", const 846 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_dev_reset_iommu_done_17, sym __slate_asm_string_pci_dev_reset_iommu_done_18, const 4168 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("847: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 847b - ., 4; .popsection\n# {0}", const 847 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v16 = (__ret_warn_on != 0) as i64;
}
if __v16 != 0 {
return;
}
if group_device_dma_alias_is_blocked(group, gdev) {
// FIXME: DMA aliased devices share the same RID, which would be
// convoluted to handle, as "gdev->blocked" is not sufficient:
//  - "blocked" state is effectively shared across these devices
//  - if the core skipped the blocking on the second device, the
//    IOMMU driver's attachment state would diverge from the HW
//    state
// For now, just warn and see whether real ATS use cases hit it.
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"DMA-aliased sibling may be prematurely unblocked\n".as_ptr() as *const u8) };
}
// Re-attach RID domain back to group->domain
//
// Leave the device parked in the blocking_domain if group->domain isn't
// initialized yet
if !unsafe { (*group).domain }.is_null() && (unsafe { (*group).domain }) != unsafe { (*group).blocking_domain } {
let mut __v22: i64;
let __ret_warn_on: i32 = (__iommu_attach_device(unsafe { (*group).domain }, unsafe { core::ptr::addr_of_mut!((*pdev).dev) }, unsafe { (*group).blocking_domain }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("848: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 848b - ., 3; .popsection\n# {0}", const 848 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_dev_reset_iommu_done_23, sym __slate_asm_string_pci_dev_reset_iommu_done_24, const 4193 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("849: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 849b - ., 4; .popsection\n# {0}", const 849 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v22 = (__ret_warn_on != 0) as i64;
}
// Update gdev->blocked upon the domain change, as it is used to return
// the correct domain in iommu_driver_get_domain_for_dev() that might be
// called in a set_dev_pasid callback function.
unsafe {
(*gdev).blocked = false;
}
// Re-attach PASID domains back to the domains retained in pasid_array.
//
// The pasid_array is mostly fenced by group->mutex, except one reader
// in iommu_attach_handle_get(), so it's safe to read without xa_lock.
if (unsafe { (*(*pdev).dev.iommu).max_pasids }) > 0 {
pasid = 1;
entry = unsafe { xa_find(core::ptr::addr_of_mut!((*group).pasid_array), core::ptr::addr_of_mut!(pasid), 18446744073709551615u64, 8u32) };
while !entry.is_null() {
let pasid_dom: *mut iommu_domain = pasid_array_entry_to_domain(entry);
let mut __v28: i64;
let __ret_warn_on: i32 = ((unsafe { unsafe { (*(*pasid_dom).ops).set_dev_pasid }.unwrap()(pasid_dom, core::ptr::addr_of_mut!((*pdev).dev), pasid as u32, (*group).blocking_domain) }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("850: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 850b - ., 3; .popsection\n# {0}", const 850 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_dev_reset_iommu_done_29, sym __slate_asm_string_pci_dev_reset_iommu_done_30, const 4216 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("851: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 851b - ., 4; .popsection\n# {0}", const 851 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v28 = (__ret_warn_on != 0) as i64;
entry = unsafe { xa_find_after(core::ptr::addr_of_mut!((*group).pasid_array), core::ptr::addr_of_mut!(pasid), 18446744073709551615u64, 8u32) };
}
}
let mut __v34: i64;
let __ret_warn_on: i32 = ((unsafe { (*group).recovery_cnt }) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("852: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 852b - ., 3; .popsection\n# {0}", const 852 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_dev_reset_iommu_done_35, sym __slate_asm_string_pci_dev_reset_iommu_done_36, const 4220 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("853: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 853b - ., 4; .popsection\n# {0}", const 853 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v34 = (__ret_warn_on != 0) as i64;
if __v34 == 0 {
let __v40: u32 = unsafe { (*group).recovery_cnt };
unsafe {
(*group).recovery_cnt = __v40.wrapping_sub(1);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_dev_reset_iommu_done_854: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_dev_reset_iommu_done)) };

static __slate_asm_string___iommu_group_free_device_1: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_group_free_device_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_group_set_domain_internal_11: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_group_set_domain_internal_12: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_group_set_domain_internal_16: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_group_set_domain_internal_17: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_group_set_domain_nofail_1: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_group_set_domain_nofail_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_map_domain_pgtbl_1: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_map_domain_pgtbl_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_probe_device_12: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_probe_device_13: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_probe_device_14: [u8; 63] = *b"%s %s: late IOMMU probe at driver bind, something fishy here!\n\0";

static __slate_asm_string___iommu_probe_device_15: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_release_dma_ownership_2: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_release_dma_ownership_3: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_set_group_pasid_8: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_set_group_pasid_9: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_unmap_1: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_unmap_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___iommu_unmap_domain_pgtbl_1: [u8; 1] = *b"\0";

static __slate_asm_string___iommu_unmap_domain_pgtbl_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_iommu_attach_device_pasid_8: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_attach_device_pasid_9: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_attach_group_handle_5: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_attach_group_handle_6: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_create_device_direct_mappings_19: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_create_device_direct_mappings_20: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_deferred_attach_3: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_deferred_attach_4: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_detach_device_10: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_detach_device_11: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_detach_device_8: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_detach_device_9: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_device_claim_dma_owner_8: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_device_claim_dma_owner_9: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_device_register_1: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_device_register_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_device_unuse_default_domain_2: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_device_unuse_default_domain_3: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_driver_get_domain_for_dev_2: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_driver_get_domain_for_dev_3: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_free_global_pasid_1: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_free_global_pasid_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_get_default_domain_type_12: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_get_default_domain_type_13: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_get_default_domain_type_5: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_get_default_domain_type_6: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_group_claim_dma_owner_4: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_group_claim_dma_owner_5: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_group_release_3: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_group_release_4: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_group_release_8: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_group_release_9: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_group_store_type_10: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_group_store_type_11: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_init_0: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_init_1: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_init_device_7: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_init_device_8: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_map_nosync_3: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_map_nosync_4: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_pgsize_10: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_pgsize_11: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_remove_dev_pasid_1: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_remove_dev_pasid_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_replace_device_pasid_11: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_replace_device_pasid_12: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_replace_device_pasid_13: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_replace_device_pasid_14: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_replace_group_handle_6: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_replace_group_handle_7: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_set_fault_handler_1: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_set_fault_handler_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_iommu_setup_default_domain_11: [u8; 1] = *b"\0";

static __slate_asm_string_iommu_setup_default_domain_12: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_pci_dev_reset_iommu_done_10: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_pci_dev_reset_iommu_done_17: [u8; 1] = *b"\0";

static __slate_asm_string_pci_dev_reset_iommu_done_18: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_pci_dev_reset_iommu_done_23: [u8; 1] = *b"\0";

static __slate_asm_string_pci_dev_reset_iommu_done_24: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_pci_dev_reset_iommu_done_29: [u8; 1] = *b"\0";

static __slate_asm_string_pci_dev_reset_iommu_done_30: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_pci_dev_reset_iommu_done_35: [u8; 1] = *b"\0";

static __slate_asm_string_pci_dev_reset_iommu_done_36: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_pci_dev_reset_iommu_done_4: [u8; 1] = *b"\0";

static __slate_asm_string_pci_dev_reset_iommu_done_5: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_pci_dev_reset_iommu_done_9: [u8; 1] = *b"\0";

static __slate_asm_string_pci_dev_reset_iommu_prepare_4: [u8; 1] = *b"\0";

static __slate_asm_string_pci_dev_reset_iommu_prepare_5: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";

static __slate_asm_string_pci_device_group_2: [u8; 1] = *b"\0";

static __slate_asm_string_pci_device_group_3: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/iommu/iommu.c\0";
