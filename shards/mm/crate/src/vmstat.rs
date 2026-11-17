
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
//! linux/mm/vmstat.c
//!
//! Manages VM statistics
//! Copyright (C) 1991, 1992, 1993, 1994  Linus Torvalds
//!
//! zoned VM statistics
//! Copyright (C) 2006 Silicon Graphics, Inc.,
//! 	Christoph Lameter <cl@gentwo.org>
//! Copyright (C) 2008-2014 Christoph Lameter
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
//! Internal kernel timer frequency
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
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! # chars in a path name including nul
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
//! Helper macro for scoped_guard().
//!
//! Note that the "!__is_cond_ptr(_name)" part of the condition ensures that
//! compiler would be sure that for the unconditional locks the body of the
//! loop (caller-provided code glued to the else clause) could not be skipped.
//! It is needed because the other part - "__guard_ptr(_name)(&scope)" - is too
//! hard to deduce (even if could be proven true for unconditional locks).
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
//! Huge pages are a constant size, but don't exceed the maximum allocation
//! granularity.
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
//! The first word is the work queue pointer and the flags rolled into
//! one
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
//! GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
//! __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
//! entry starting with bit 0. Bit is set if the combination is not
//! allowed.
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
//! Extract the high bits for type
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
//! Keep the same order as in fs/proc/proc_sysctl.c
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
//! Generate SD flag indexes
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! Enabling zone reclaim means the page allocator will attempt to fulfill
//! the allocation request on the current node by triggering reclaim and
//! trying to shrink the current node.
//! Fallback allocations on the next candidates in the zonelist are considered
//! when reclaim fails to free up enough memory in the current node/zone.
//!
//! These bit locations are exposed in the vm.zone_reclaim_mode sysctl.
//! New bits are OK, but existing bits should not be changed.
//! swapcache pages are stored in the swapper_space radix tree.  We want to
//! get good packing density in that tree, so the index should be dense in
//! the low-order bits.
//!
//! We arrange the `type' and `offset' fields so that `type' is at the six
//! high-order bits of the swp_entry_t and `offset' is right-aligned in the
//! remaining bits.  Although `type' itself needs only five bits, we allow for
//! shmem/tmpfs to shift it all up a further one bit: see swp_to_radix_entry().
//!
//! swp_entry_t's are *never* stored anywhere in their arch-dependent format.
//! Temporary until swp_entry_t eliminated.
//! See the comment in asm-generic/io.h about ioremap_np().
//! Compare PTEs respecting the dirty bit.
//! Convert GFP flags to their corresponding migrate type
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
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// this_cpu_xchg() is implemented using CMPXCHG without a LOCK prefix.
// XCHG is expensive due to the implied LOCK prefix. The processor
// cannot prefetch cachelines if XCHG is used.
// this_cpu_read() makes the compiler load the per-CPU variable every time
// it is accessed while this_cpu_read_stable() allows the value to be cached.
// this_cpu_read_stable() is more efficient and can be used if its value
// is guaranteed to be valid across CPUs.  The current users include
// current_task and cpu_current_top_of_stack, both of which are
// actually per-thread variables implemented as per-CPU variables and
// thus stable for the duration of the respective task.
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
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// No such file or directory
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Invalid system call number
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
// To support Context Analysis, we need to allow the compiler to see the
// acquisition and release of the context lock. However, the "cleanup" helpers
// wrap the lock in a struct passed through separate helper functions, which
// hides the lock alias from the compiler (no inter-procedural analysis).
//
// To make it work, we introduce an explicit alias to the context lock instance
// that is "cleaned" up with a separate cleanup helper. This helper is a dummy
// function that does nothing at runtime, but has the "_unlock" attribute to
// tell the compiler what happens at the end of the scope.
//
// To generalize the pattern, the WITH_LOCK_GUARD_1_ATTRS() macro should be used
// to redefine the constructor, which then also creates the alias variable with
// the right "cleanup" attribute, *after* DECLARE_LOCK_GUARD_1_ATTRS() has been
// used.
//
// Example usage:
//
//   DECLARE_LOCK_GUARD_1_ATTRS(mutex, __acquires(_T), __releases(*(struct mutex **)_T))
//   #define class_mutex_constructor(_T) WITH_LOCK_GUARD_1_ATTRS(mutex, _T)
//
// Note: To support the for-loop based scoped helpers, the auxiliary variable
// must be a pointer to the "class" type because it is defined in the same
// statement as the guard variable. However, we initialize it with the lock
// pointer (despite the type mismatch, the compiler's alias analysis still works
// as expected). The "_unlock" attribute receives a pointer to the auxiliary
// variable (a double pointer to the class type), and must be cast and
// dereferenced appropriately.
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
// min_t - return minimum of two values, using the specified type
// @type: data type to use
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
// N 32-bit words worth of info
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
// Don't call further
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
// Shift up (to get rid of type), then down to get value
// Shift the offset up "too far" by TYPE bits, then down again
// The offset is inverted by a binary not operation to make the high
// physical bits set.
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
// Implementation specific max size
// Generate SD flag bits
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
// Enable zone reclaim
// Writeout pages during reclaim
// Unmap pages during reclaim
// "Poisoned" here is meant in the very general sense of "future accesses are
// invalid", instead of referring very specifically to hardware memory errors.
// This marker is meant to represent any of various different causes of this.
//
// Note that, when encountered by the faulting logic, PTEs with this marker will
// result in VM_FAULT_HWPOISON and thus regardless trigger hardware memory error
// logic.
// Indicates that, on fault, this PTE will case a SIGSEGV signal to be
// sent. This means guard markers behave in effect as if the region were mapped
// PROT_NONE, rather than if they were a memory hole or equivalent.
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
macro_rules! ARRAY_SIZE__16 {
() => {
((core::mem::size_of::<[ctl_table; 3]>() as u64) / (core::mem::size_of::<ctl_table>() as u64)).wrapping_add(__must_be_array__60!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__53 {
() => {
(((const { core::mem::size_of::<[*const u8; 161]>() }) as u64) / ((const { core::mem::size_of::<*const u8>() }) as u64)).wrapping_add(__must_be_array__36!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__56 {
() => {
((core::mem::size_of::<[i64; 4]>() as u64) / (core::mem::size_of::<i64>() as u64)).wrapping_add(__must_be_array__17!() as u64)
};
}

/// ASSERT_EXCLUSIVE_BITS - assert no concurrent writes to subset of bits in @var
///
/// Bit-granular variant of ASSERT_EXCLUSIVE_WRITER().
///
/// Assert that there are no concurrent writes to a subset of bits in @var;
/// concurrent readers are permitted. This assertion captures more detailed
/// bit-level properties, compared to the other (word granularity) assertions.
/// Only the bits set in @mask are checked for concurrent modifications, while
/// ignoring the remaining bits, i.e. concurrent writes (or reads) to ~mask bits
/// are ignored.
///
/// Use this for variables, where some bits must not be modified concurrently,
/// yet other bits are expected to be modified concurrently.
///
/// For example, variables where, after initialization, some bits are read-only,
/// but other bits may still be modified concurrently. A reader may wish to
/// assert that this is true as follows:
///
/// .. code-block:: c
///
/// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
/// 	foo = (READ_ONCE(flags) & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
///
/// Note: The access that immediately follows ASSERT_EXCLUSIVE_BITS() is assumed
/// to access the masked bits only, and KCSAN optimistically assumes it is
/// therefore safe, even in the presence of data races, and marking it with
/// READ_ONCE() is optional from KCSAN's point-of-view. We caution, however, that
/// it may still be advisable to do so, since we cannot reason about all compiler
/// optimizations when it comes to bit manipulations (on the reader and writer
/// side). If you are sure nothing can go wrong, we can write the above simply
/// as:
///
/// .. code-block:: c
///
/// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
/// 	foo = (flags & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
///
/// Another example, where this may be used, is when certain bits of @var may
/// only be modified when holding the appropriate lock, but other bits may still
/// be modified concurrently. Writers, where other bits may change concurrently,
/// could use the assertion as follows:
///
/// .. code-block:: c
///
/// 	spin_lock(&foo_lock);
/// 	ASSERT_EXCLUSIVE_BITS(flags, FOO_MASK);
/// 	old_flags = flags;
/// 	new_flags = (old_flags & ~FOO_MASK) | (new_foo << FOO_SHIFT);
/// 	if (cmpxchg(&flags, old_flags, new_flags) != old_flags) { ... }
/// 	spin_unlock(&foo_lock);
///
/// @var: variable to assert on
/// @mask: only check for modifications to bits set in @mask
macro_rules! ASSERT_EXCLUSIVE_BITS {
($var:expr, $mask:expr) => {
loop {
kcsan_set_access_mask($mask);
__kcsan_check_access((unsafe { core::ptr::addr_of!($var) }) as *const (), core::mem::size_of::<u64>() as u64, 8i32);
kcsan_set_access_mask(0u64);
kcsan_atomic_next(1i32);
if true {
break;
}
}
};
}

macro_rules! BITMAP_LAST_WORD_MASK {
($nbits:expr) => {
18446744073709551615u64 >> ($nbits.wrapping_neg() & 63u32)
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

/// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
/// expression but avoids the generation of any code, even if that expression
/// has side-effects.
macro_rules! BUILD_BUG_ON_INVALID {
() => {
};
}

macro_rules! GENMASK_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_find_h_67>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__2 {
() => {
core::mem::size_of::<__SlateRecord_find_h_207>() as i32
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
macro_rules! GENMASK_TYPE__2 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__2!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
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
macro_rules! GENMASK_TYPE {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

macro_rules! GENMASK {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
};
}

macro_rules! GENMASK__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__2!(u64, $h, $l)
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

macro_rules! MIN_T__2 {
($type:ty, $a:expr, $b:expr) => {
if (($a) as $type) < (($b) as $type) { ($a) as $type } else { ($b) as $type }
};
}

macro_rules! MIN_T {
($type:ty, $a:expr) => {
($a) as $type
};
}

macro_rules! NODE_DATA {
($nid:expr) => {
unsafe { *(core::ptr::addr_of_mut!(node_data) as *mut *mut pglist_data).wrapping_offset(($nid) as isize) }
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PFN_PHYS {
($x:expr) => {
(($x) as u64) << 12i32
};
}

/// Page flags policies wrt compound pages
///
/// PF_POISONED_CHECK
///     check if this struct page poisoned/uninitialized
///
/// PF_ANY:
///     the page flag is relevant for small, head and tail pages.
///
/// PF_HEAD:
///     for compound page all operations related to the page flag applied to
///     head page.
///
/// PF_NO_TAIL:
///     modifications of the page flag must be done on small or head pages,
///     checks can be done on tail pages too.
///
/// PF_NO_COMPOUND:
///     the page flag is not relevant for compound pages.
///
/// PF_SECOND:
///     the page flag is stored in the first tail page.
macro_rules! PF_POISONED_CHECK {
($page:expr) => {
{
let mut __v: *const page;
VM_BUG_ON_PGFLAGS!();
__v = $page;
__v
}
};
}

macro_rules! PHYS_PFN {
($x:expr) => {
($x) >> 12i32
};
}

macro_rules! RCU_LOCKDEP_WARN {
($c:expr) => {
loop {
let __v: bool = false && $c;
if !__v {
break;
}
}
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! SECTION_NR_TO_ROOT {
($sec:expr) => {
($sec) / ((_AC!() << 12i32) / (core::mem::size_of::<mem_section>() as u64))
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VM_BUG_ON_PGFLAGS {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_ONCE {
() => {
BUILD_BUG_ON_INVALID!();
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

macro_rules! __WRITE_ONCE__6 {
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

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __must_be_array__17 {
() => {
core::mem::size_of::<__SlateRecord_vmstat_c_1813>() as i32
};
}

macro_rules! __must_be_array__36 {
() => {
((const { core::mem::size_of::<__SlateRecord_vmstat_c_1907>() }) as u64) as i32
};
}

macro_rules! __must_be_array__60 {
() => {
core::mem::size_of::<__SlateRecord_vmstat_c_2316>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__19 {
() => {
((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
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
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
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
macro_rules! __sign_use__4 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u64>() as u64) < 4) as i32)) }
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

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
};
}

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__4!($uy)
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
macro_rules! __verify_pcpu_ptr__6 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<u64>() as *const ();
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
macro_rules! __verify_pcpu_ptr__24 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<per_cpu_nodestat>() as *const ();
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
macro_rules! __verify_pcpu_ptr__25 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
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
macro_rules! __verify_pcpu_ptr__7 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i8>() as *const ();
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
macro_rules! __verify_pcpu_ptr__13 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<per_cpu_pages>() as *const ();
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
macro_rules! __verify_pcpu_ptr__16 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<i32>() as *const ();
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
macro_rules! __verify_pcpu_ptr__11 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<i8>() as *const ();
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
macro_rules! __verify_pcpu_ptr__14 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<per_cpu_zonestat>() as *const ();
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
macro_rules! __verify_pcpu_ptr__15 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<u8>() as *const ();
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
macro_rules! __verify_pcpu_ptr__31 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<vm_event_state>() as *const ();
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
macro_rules! __verify_pcpu_ptr__17 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<per_cpu_pages>() as *const ();
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

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__5 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
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
macro_rules! __verify_pcpu_ptr__39 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
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

/// abs - return absolute value of an argument
/// @x: the value.
///
/// If it is unsigned type, @x is converted to signed type first.
/// char is treated as if it was signed (regardless of whether it really is)
/// but the macro's return type is preserved as char.
///
/// NOTE, for signed type if @x is the minimum, the returned result is undefined
/// as there is not enough bits to represent it as a positive number.
///
/// Return: an absolute value of @x.
macro_rules! abs__2 {
($x:expr) => {
{
let mut __v: i64;
let __v2: i64 = $x;
__v = if __v2 < 0 { __v2.wrapping_neg() } else { __v2 };
__v
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

macro_rules! cond_resched__35 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/vmstat.c".as_ptr() as *const u8, 1628i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__55 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/vmstat.c".as_ptr() as *const u8, 829i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__83 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/vmstat.c".as_ptr() as *const u8, 2166i32, 0u32);
__v = _cond_resched();
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

/// cpumask_bits - get the bits in a cpumask
/// @maskp: the struct cpumask *
///
/// You should only assume nr_cpu_ids bits of this mask are valid.  This is
/// a macro so it's const-correct.
macro_rules! cpumask_bits {
($maskp:expr) => {
unsafe { (*$maskp).bits.as_ptr() as *const u64 }
};
}

/// data_race - mark an expression as containing intentional data races
///
/// This data_race() macro is useful for situations in which data races
/// should be forgiven.  One example is diagnostic code that accesses
/// shared variables but is not a part of the core synchronization design.
/// For example, if accesses to a given variable are protected by a lock,
/// except for diagnostic code, then the accesses under the lock should
/// be plain C-language accesses and those in the diagnostic code should
/// use data_race().  This way, KCSAN will complain if buggy lockless
/// accesses to that variable are introduced, even if the buggy accesses
/// are protected by READ_ONCE() or WRITE_ONCE().
///
/// This macro *does not* affect normal code generation, but is a hint
/// to tooling that data races here are to be ignored.  If the access must
/// be atomic *and* KCSAN should ignore the access, use both data_race()
/// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
macro_rules! data_race__2 {
($expr:expr) => {
{
let mut __v: u64;
__kcsan_disable_current();
let __v2: u64 = $expr;
__kcsan_enable_current();
__v = __v2;
__v
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
unsafe { debugfs_create_file_full(($name) as *const u8, ($mode) as u16, $parent, $data, core::ptr::null::<()>(), $fops) };
};
}

macro_rules! first_node {
($src:expr) => {
__first_node((unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

macro_rules! flush_workqueue {
($wq:expr) => {
let _wq: *mut workqueue_struct = $wq;
if (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_percpu_wq) }) && _wq == unsafe { system_percpu_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_highpri_wq) }) && _wq == unsafe { system_highpri_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_long_wq) }) && _wq == unsafe { system_long_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_dfl_long_wq) }) && _wq == unsafe { system_dfl_long_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_dfl_wq) }) && _wq == unsafe { system_dfl_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_freezable_wq) }) && _wq == unsafe { system_freezable_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_power_efficient_wq) }) && _wq == unsafe { system_power_efficient_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_freezable_power_efficient_wq) }) && _wq == unsafe { system_freezable_power_efficient_wq } {
unsafe { __warn_flushing_systemwide_wq() };
}
unsafe { __flush_workqueue(_wq) };
};
}

macro_rules! get_pageblock_migratetype {
($page:expr) => {
unsafe { get_pfnblock_migratetype(($page) as *const page, ($page.offset_from(vmemmap_base as *mut page) as i64) as u64) }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
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

macro_rules! lockdep_assert_preemption_disabled {
() => {
while false {
}
};
}

macro_rules! next_node {
($n:expr, $src:expr) => {
__next_node($n, (unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

/// No static inline type checking - see Subtlety (1) above.
macro_rules! node_isset {
($node:expr, $nodemask:expr) => {
test_bit__2!($node, unsafe { core::ptr::addr_of_mut!($nodemask.bits) as *mut u64 })
};
}

macro_rules! per_cpu {
($var:expr, $cpu:expr) => {
unsafe { *per_cpu_ptr__10!(core::ptr::addr_of_mut!($var), $cpu) }
};
}

macro_rules! per_cpu_offset {
($x:expr) => {
unsafe { __per_cpu_offset[($x) as usize] }
};
}

macro_rules! per_cpu_ptr__2 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut per_cpu_pages;
__verify_pcpu_ptr__17!();
__v = (((($ptr) as u64) as *mut per_cpu_pages) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut per_cpu_pages;
__v
}
};
}

macro_rules! per_cpu_ptr__5 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut per_cpu_nodestat;
__verify_pcpu_ptr__24!();
__v = (((($ptr) as u64) as *mut per_cpu_nodestat) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut per_cpu_nodestat;
__v
}
};
}

macro_rules! per_cpu_ptr__8 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut delayed_work;
__verify_pcpu_ptr__39!();
__v = (((($ptr) as u64) as *mut delayed_work) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut delayed_work;
__v
}
};
}

macro_rules! per_cpu_ptr {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut per_cpu_zonestat;
__verify_pcpu_ptr__14!();
__v = (((($ptr) as u64) as *mut per_cpu_zonestat) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut per_cpu_zonestat;
__v
}
};
}

macro_rules! per_cpu_ptr__13 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut vm_event_state;
__verify_pcpu_ptr__31!();
__v = (((($ptr) as u64) as *mut vm_event_state) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut vm_event_state;
__v
}
};
}

macro_rules! per_cpu_ptr__10 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut i32;
__verify_pcpu_ptr__16!();
__v = (((($ptr) as u64) as *mut i32) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut i32;
__v
}
};
}

macro_rules! pfn_to_online_page {
($pfn:expr) => {
{
let mut __v: *mut page;
let mut __v2: *mut page = core::ptr::null_mut::<page>();
if pfn_valid($pfn) != 0 {
__v2 = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(($pfn) as isize);
}
__v = __v2;
__v
}
};
}

macro_rules! pgtable_l5_enabled {
() => {
cpu_feature_enabled!(528i32)
};
}

macro_rules! preempt_count_add {
($val:expr) => {
__preempt_count_add($val);
};
}

macro_rules! preempt_count_dec_and_test {
() => {
__preempt_count_dec_and_test()
};
}

macro_rules! preempt_count_inc {
() => {
preempt_count_add!(1i32);
};
}

macro_rules! proc_create_seq {
($name:expr, $mode:expr, $ops:expr) => {
unsafe { proc_create_seq_private(($name) as *const u8, ($mode) as u16, core::ptr::null_mut::<proc_dir_entry>(), $ops, 0u32, core::ptr::null_mut::<()>()) };
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

macro_rules! rcu_lock_acquire {
() => {
while false {
}
};
}

macro_rules! rcu_lock_release {
() => {
while false {
}
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits {
($nbits:expr) => {
(unsafe { __slate_intrinsic_1c5364fcd9da51fe($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits__2 {
($nbits:expr) => {
(unsafe { __slate_intrinsic_f217c4720f04f6db($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
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

macro_rules! static_branch_disable {
($x:expr) => {
unsafe { static_key_disable(core::ptr::addr_of_mut!((*$x).key)) };
};
}

macro_rules! static_branch_enable {
($x:expr) => {
unsafe { static_key_enable(core::ptr::addr_of_mut!((*$x).key)) };
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

macro_rules! test_bit__2 {
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

macro_rules! test_bit__6 {
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

macro_rules! type_max {
() => {
__type_max!()
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
    static mut kmsg_fops: file_operations;
    static mut __per_cpu_offset: [u64; 64];
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut vmemmap_base: u64;
    static mut system_state: u32;
    static mut node_states: [__SlateRecord_nodemask_types_h_17; 6];
    static mut node_data: [*mut pglist_data; 0];
    static mut __cpu_possible_mask: cpumask;
    static mut __cpu_online_mask: cpumask;
    static mut __num_online_cpus: __SlateRecord_types_h_190;
    static mut boot_cpu_data: cpuinfo_x86;
    static mut cpuinfo_op: seq_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut cpu_number: i32;
    static mut __SCK__cond_resched: static_call_key;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut system_highpri_wq: *mut workqueue_struct;
    static mut system_long_wq: *mut workqueue_struct;
    static mut system_dfl_wq: *mut workqueue_struct;
    static mut system_freezable_wq: *mut workqueue_struct;
    static mut system_power_efficient_wq: *mut workqueue_struct;
    static mut system_freezable_power_efficient_wq: *mut workqueue_struct;
    static mut system_dfl_long_wq: *mut workqueue_struct;
    static mut migratetype_names: [*const u8; 4];
    static mut mem_section: *mut *mut mem_section;
    static mut node_to_cpumask_map: [[cpumask; 1]; 64];
    static mut numa_node: i32;
    static mut sighand_cachep: *mut kmem_cache;
    static mut sysctl_vals: [i32; 0];
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut vm_numa_stat_key: static_key_true;
    static mut kstat: kernel_stat;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn static_key_enable(arg0: *mut static_key);
    fn static_key_disable(arg0: *mut static_key);
    fn __xchg_wrong_size();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn _find_first_bit(arg0: *const u64, arg1: u64) -> u64;
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memchr_inv(arg0: *const (), arg1: i32, arg2: u64) -> *mut ();
    fn __cond_resched() -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_48();
    fn __compiletime_assert_51();
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn __SCT__cond_resched() -> i32;
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn get_pfnblock_migratetype(arg0: *const page, arg1: u64) -> u32;
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn proc_dointvec_jiffies(arg0: *const ctl_table, arg1: i32, arg2: *mut (), arg3: *mut u64, arg4: *mut i64) -> i32;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn round_jiffies_relative(arg0: u64) -> u64;
    fn alloc_workqueue_noprof(arg0: *const u8, arg1: u32, arg2: i32, ...) -> *mut workqueue_struct;
    fn queue_delayed_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut delayed_work, arg3: u64) -> bool;
    fn __flush_workqueue(arg0: *mut workqueue_struct);
    fn schedule_on_each_cpu(arg0: Option<unsafe extern "C" fn(*mut work_struct)>) -> i32;
    fn disable_delayed_work_sync(arg0: *mut delayed_work) -> bool;
    fn enable_delayed_work(arg0: *mut delayed_work) -> bool;
    fn work_busy(arg0: *mut work_struct) -> u32;
    fn __warn_flushing_systemwide_wq();
    fn __compiletime_assert_193();
    fn kswapd_test_hopeless(arg0: *mut pglist_data) -> bool;
    fn first_online_pgdat() -> *mut pglist_data;
    fn next_online_pgdat(arg0: *mut pglist_data) -> *mut pglist_data;
    fn next_zone(arg0: *mut zone) -> *mut zone;
    fn __compiletime_assert_201();
    fn proc_dointvec_minmax(arg0: *const ctl_table, arg1: i32, arg2: *mut (), arg3: *mut u64, arg4: *mut i64) -> i32;
    fn __register_sysctl_init(arg0: *const u8, arg1: *const ctl_table, arg2: *const u8, arg3: u64);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn __compiletime_assert_384();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_390();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn cpus_read_lock();
    fn cpus_read_unlock();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn __cpuhp_setup_state(arg0: i32, arg1: *const u8, arg2: bool, arg3: Option<unsafe extern "C" fn(u32) -> i32>, arg4: Option<unsafe extern "C" fn(u32) -> i32>, arg5: bool) -> i32;
    fn proc_create_seq_private(arg0: *const u8, arg1: u16, arg2: *mut proc_dir_entry, arg3: *const seq_operations, arg4: u32, arg5: *mut ()) -> *mut proc_dir_entry;
    fn seq_open(arg0: *mut file, arg1: *const seq_operations) -> i32;
    fn seq_read(arg0: *mut file, arg1: *mut u8, arg2: u64, arg3: *mut i64) -> i64;
    fn seq_lseek(arg0: *mut file, arg1: i64, arg2: i32) -> i64;
    fn seq_release(arg0: *mut inode, arg1: *mut file) -> i32;
    fn seq_write(arg0: *mut seq_file, arg1: *const (), arg2: u64) -> i32;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn seq_putc(arg0: *mut seq_file, arg1: u8);
    fn __seq_puts(arg0: *mut seq_file, arg1: *const u8);
    #[link_name = "strlen"]
    fn __builtin_strlen(arg0: *const u8) -> u64;
    fn seq_put_decimal_ull(arg0: *mut seq_file, arg1: *const u8, arg2: u64);
    fn debugfs_create_file_full(arg0: *const u8, arg1: u16, arg2: *mut dentry, arg3: *mut (), arg4: *const (), arg5: *const file_operations) -> *mut dentry;
    fn debugfs_create_file_short(arg0: *const u8, arg1: u16, arg2: *mut dentry, arg3: *mut (), arg4: *const (), arg5: *const debugfs_short_fops) -> *mut dentry;
    fn debugfs_create_dir(arg0: *const u8, arg1: *mut dentry) -> *mut dentry;
    fn global_dirty_limits(arg0: *mut u64, arg1: *mut u64);
    fn housekeeping_test_cpu(arg0: i32, arg1: u32) -> bool;
    fn decay_pcp_high(arg0: *mut zone, arg1: *mut per_cpu_pages) -> bool;
    fn drain_zone_pages(arg0: *mut zone, arg1: *mut per_cpu_pages);
    fn __compiletime_assert_664();
    fn __compiletime_assert_667();
    fn __compiletime_assert_670();
    fn __compiletime_assert_673();
    fn __compiletime_assert_688();
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

fn __kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

fn kcsan_atomic_next(n: i32) {
}

fn kcsan_set_access_mask(mask: u64) {
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


fn __kcsan_enable_current() {
}

fn __kcsan_disable_current() {
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
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
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


fn div_u64_rem(dividend: u64, divisor: u32, remainder: &mut u32) -> u64 {
unsafe {
*(remainder as *mut u32) = (dividend % (divisor as u64)) as u32;
}
dividend / (divisor as u64)
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
#[inline(always)]
fn __this_cpu_preempt_check(op: *const u8) {
}

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

#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x0_0 {
    __slate_bits_0: __slate_bits::__SlateBits98U0,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x1_0 {
    __slate_bits_0: __slate_bits::__SlateBits99U0,
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
struct cpuid_table {
    leaves: cpuid_leaves,
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
struct __SlateRecord_find_h_67 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_207 {
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
    __slate_bits_0: __slate_bits::__SlateBits149U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_119 {
    __slate_bits_0: __slate_bits::__SlateBits150U0,
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
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
    __slate_bits_0: __slate_bits::__SlateBits124U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_true {
    key: static_key,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits130U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_set(v: *mut __SlateRecord_types_h_197, i: i64) {
__WRITE_ONCE__6!((*v).counter, i);
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
fn arch_atomic64_add(mut i: i64, mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock addq {1:r}, ({0:r})\n# {2}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u64, in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic64_set(v: *mut __SlateRecord_types_h_197, i: i64) {
arch_atomic64_set(v, i);
}

#[inline(always)]
fn raw_atomic64_add(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_set(v: *mut __SlateRecord_types_h_197, i: i64) {
raw_atomic64_set(v, i);
}

#[inline(always)]
fn raw_atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_add(i, v);
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn atomic_long_set(v: *mut __SlateRecord_types_h_197, i: i64) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_set(v, i);
}

#[inline(always)]
fn atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add(i, v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits__2!(size) {
let mut val: u64 = 0;
if unlikely!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset) }
}

#[inline(always)]
fn find_first_bit(addr: *const u64, size: u64) -> u64 {
if small_const_nbits__2!(size) {
let val: u64 = (unsafe { *addr }) & GENMASK__2!(size.wrapping_sub(1), 0i32);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_first_bit(addr, size) }
}

#[inline(always)]
fn bitmap_empty(src: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits) == 0;
}
find_first_bit(src, nbits as u64) == (nbits as u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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


fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

// list_entry - get the struct for this entry
// @ptr:	the &struct list_head pointer.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
}


#[inline(always)]
fn __node_set(node: i32, dstp: *mut __SlateRecord_nodemask_types_h_17) {
set_bit(node as i64, unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 });
}

#[inline(always)]
fn __node_clear(node: i32, dstp: *mut __SlateRecord_nodemask_types_h_17) {
clear_bit(node as i64, unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 });
}

#[inline(always)]
fn __first_node(srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__46: i32 = 64i32;
let __UNIQUE_ID_y__47: u64 = find_first_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64);
loop {
if __types_ok__3!(__UNIQUE_ID_x__46, __UNIQUE_ID_y__47) == 0 {
unsafe { __compiletime_assert_48() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__46 as u64) < __UNIQUE_ID_y__47 { __UNIQUE_ID_x__46 as u64 } else { __UNIQUE_ID_y__47 };
__v0 as u32
}

#[inline(always)]
fn __next_node(n: i32, srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__49: i32 = 64i32;
let __UNIQUE_ID_y__50: u64 = find_next_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64, n.wrapping_add(1) as u64);
loop {
if __types_ok__3!(__UNIQUE_ID_x__49, __UNIQUE_ID_y__50) == 0 {
unsafe { __compiletime_assert_51() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__49 as u64) < __UNIQUE_ID_y__50 { __UNIQUE_ID_x__49 as u64 } else { __UNIQUE_ID_y__50 };
__v0 as u32
}

#[inline(always)]
fn node_state(node: i32, state: u32) -> i32 {
node_isset!(node, node_states[(state as u32) as usize])
}

#[inline(always)]
fn node_set_state(node: i32, state: u32) {
__node_set(node, unsafe { core::ptr::addr_of_mut!(node_states) as *mut __SlateRecord_nodemask_types_h_17 }.wrapping_offset((state as u32) as isize) as *mut __SlateRecord_nodemask_types_h_17);
}

#[inline(always)]
fn node_clear_state(node: i32, state: u32) {
__node_clear(node, unsafe { core::ptr::addr_of_mut!(node_states) as *mut __SlateRecord_nodemask_types_h_17 }.wrapping_offset((state as u32) as isize) as *mut __SlateRecord_nodemask_types_h_17);
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
test_bit__6!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
}

#[inline(always)]
fn cpumask_empty(srcp: *const cpumask) -> bool {
bitmap_empty(cpumask_bits!(srcp), 64u32)
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
fn num_online_cpus() -> u32 {
raw_atomic_read((unsafe { core::ptr::addr_of_mut!(__num_online_cpus) }) as *const __SlateRecord_types_h_190) as u32
}

#[inline(always)]
fn cpu_online(cpu: u32) -> bool {
cpumask_test_cpu(cpu as i32, (unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
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
struct rseq {
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
    __slate_bits_0: __slate_bits::__SlateBits97U0,
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
    __slate_bits_0: __slate_bits::__SlateBits131U0,
    pkru: u32,
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
}


// We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
#[inline(always)]
fn __preempt_count_add(val: i32) {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4(val) }) && (val == 1 || val == 1i32.wrapping_neg()) { val } else { 0 };
if false {
let pao_tmp__: u64 = val as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:{0}(%rip)", sym __preempt_count, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:{0}(%rip)", sym __preempt_count, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = val as u64;
if false {
let pto_tmp__: u64 = val as u64;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:{0}(%rip)", sym __preempt_count, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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

#[inline(always)]
fn __preempt_count_dec_and_test() -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("decq %gs:{0}(%rip)\nsete {1}", sym __preempt_count, lateout(reg_byte) __t0, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

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
// preempt_disable_nested - Disable preemption inside a normally preempt disabled section
//
// Use for code which requires preemption protection inside a critical
// section which has preemption disabled implicitly on non-PREEMPT_RT
// enabled kernels, by e.g.:
//  - holding a spinlock/rwlock
//  - soft interrupt context
//  - regular interrupt handlers
//
// On PREEMPT_RT enabled kernels spinlock/rwlock held sections, soft
// interrupt context and regular interrupt handlers are preemptible and
// only prevent migration. preempt_disable_nested() ensures that preemption
// is disabled for cases which require CPU local serialization even on
// PREEMPT_RT. For non-PREEMPT_RT kernels this is a NOP.
//
// The use cases are code sequences which are not serialized by a
// particular lock instance, e.g.:
//  - seqcount write side critical sections where the seqcount is not
//    associated to a particular lock and therefore the automatic
//    protection mechanism does not work. This prevents a live lock
//    against a preempting high priority reader.
//  - RMW per CPU variable updates like vmstat.
//
// Macro to avoid header recursion hell vs. lockdep
#[inline(always)]
fn preempt_enable_nested() {
if IS_ENABLED!() != 0 {
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely!(preempt_count_dec_and_test!()) != 0 {
loop {
unsafe {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
}
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
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_56: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
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
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


fn mem_alloc_profiling_enabled() -> bool {
false
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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct gendisk {
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
struct vm_struct {
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
struct mem_cgroup {
}


#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct page_frag {
    page: *mut page,
    offset: u32,
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
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
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct load_weight {
    weight: u64,
    inv_weight: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits301U0,
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
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
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
    __slate_bits_0: __slate_bits::__SlateBits67U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits67U1,
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
    __slate_bits_2: __slate_bits::__SlateBits67U2,
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
// I am kcompactd
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
struct memory_tier {
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
struct irq_domain {
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
#[inline(always)]
fn rcu_read_lock() {
unsafe { __rcu_read_lock() };
__acquire_shared!();
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
}

fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!();
unsafe { __rcu_read_unlock() };
}

fn rcu_read_lock_sched() {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
__acquire_shared!();
__acquire_shared!();
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
}

fn rcu_read_unlock_sched() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!();
__release_shared!();
__release_shared!();
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely!(preempt_count_dec_and_test!()) != 0 {
loop {
unsafe {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
}
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

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_133: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


static class_rcu_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rcupdate_h_1216 {
    lock: *mut (),
}


#[inline(always)]
fn class_rcu_destructor(_T: *mut __SlateRecord_rcupdate_h_1216) {
rcu_read_unlock();
}

#[inline(always)]
fn class_rcu_lock_ptr(_T: &mut __SlateRecord_rcupdate_h_1216) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_rcupdate_h_1216)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


#[inline(always)]
fn class_rcu_constructor() -> __SlateRecord_rcupdate_h_1216 {
let _t: __SlateRecord_rcupdate_h_1216 = __SlateRecord_rcupdate_h_1216 { lock: 1i32 as *mut () };
rcu_read_lock();
return _t;
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
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
}


fn __init_work(work: *mut work_struct, onstack: i32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


// work_pending - Find out whether a work item is currently pending
// @work: The work item in question
// delayed_work_pending - Find out whether a delayable work item is currently
// pending
// @w: The work item in question
fn queue_delayed_work(wq: *mut workqueue_struct, dwork: *mut delayed_work, delay: u64) -> bool {
unsafe { queue_delayed_work_on(64i32, wq, dwork, delay) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


fn schedule_delayed_work(dwork: *mut delayed_work, delay: u64) -> bool {
queue_delayed_work(unsafe { system_percpu_wq }, dwork, delay)
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
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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
struct pidfs_attr {
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


// page_private can be used on tail pages.  However, PagePrivate is only
// checked by the VM on the head page.  So page_private on the tail pages
// should be used for data that's ancillary to the head page (eg attaching
// buffer heads to tail pages after attaching buffer heads to the head page)
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
    __slate_bits_0: __slate_bits::__SlateBits439U0,
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
// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
fn PagePoisoned(page: &page) -> i32 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_193() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(page as *const page)).flags.f) as *const u64) };
(__v0 == (1i64.wrapping_neg() as u64)) as i32
}

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
#[repr(C)]
#[derive(Clone, Copy)]
struct zswap_lruvec_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct free_area {
    free_list: [list_head; 4],
    nr_free: u64,
}


#[inline(always)]
fn vmstat_item_print_in_thp(item: u32) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
(item as u32) == 27 || (item as u32) == 25 || (item as u32) == 23 || (item as u32) == 24 || (item as u32) == 26
}

#[inline(always)]
fn vmstat_item_in_bytes(idx: i32) -> bool {
// Global and per-node slab counters track slab pages.
// It's expected that changes are multiples of PAGE_SIZE.
// Internally values are stored in pages.
//
// Per-memcg and per-lruvec counters track memory, consumed
// by individual slab objects. These counters are actually
// byte-precise.
idx == 5 || idx == 6
}

// We do arithmetic on the LRU lists in various places in the code,
// so it is important to keep the active lists LRU_ACTIVE higher in
// the array than the corresponding inactive lists, and to keep
// the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//
// This has to be kept in sync with the statistics in zone_stat_item
// above and the descriptions in vmstat_text in mm/vmstat.c
#[repr(C)]
#[derive(Clone, Copy)]
struct lru_cost {
    count: u64,
    last_rotated: u64,
    last_io: u64,
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
struct lruvec {
    lists: [list_head; 5],
    lru_lock: spinlock,
    cost: [lru_cost; 2],
    cost_lock: spinlock,
    nonresident_age: __SlateRecord_types_h_197,
    refaults: [u64; 2],
    flags: u64,
    zswap_lruvec_state: zswap_lruvec_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct per_cpu_pages {
    lock: spinlock,
    count: i32,
    high: i32,
    high_min: i32,
    high_max: i32,
    batch: i32,
    flags: u8,
    alloc_factor: u8,
    expire: u8,
    free_count: i16,
    lists: [list_head; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct per_cpu_zonestat {
    vm_stat_diff: [i8; 10],
    stat_threshold: i8,
    vm_numa_event: [u64; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct per_cpu_nodestat {
    stat_threshold: i8,
    vm_node_stat_diff: [i8; 66],
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
struct zone {
    _watermark: [u64; 4],
    watermark_boost: u64,
    nr_reserved_highatomic: u64,
    nr_free_highatomic: u64,
    lowmem_reserve: [i64; 4],
    node: i32,
    zone_pgdat: *mut pglist_data,
    per_cpu_pageset: *mut per_cpu_pages,
    per_cpu_zonestats: *mut per_cpu_zonestat,
    pageset_high_min: i32,
    pageset_high_max: i32,
    pageset_batch: i32,
    zone_start_pfn: u64,
    managed_pages: __SlateRecord_types_h_197,
    spanned_pages: u64,
    present_pages: u64,
    name: *const u8,
    initialized: i32,
    _pad1_: cacheline_padding,
    free_area: [free_area; 11],
    flags: u64,
    lock: spinlock,
    trylock_free_pages: llist_head,
    _pad2_: cacheline_padding,
    percpu_drift_mark: u64,
    compact_cached_free_pfn: u64,
    compact_cached_migrate_pfn: [u64; 2],
    compact_init_migrate_pfn: u64,
    compact_init_free_pfn: u64,
    compact_considered: u32,
    compact_defer_shift: u32,
    compact_order_failed: i32,
    compact_blockskip_flush: bool,
    contiguous: bool,
    _pad3_: cacheline_padding,
    vm_stat: [__SlateRecord_types_h_197; 10],
    vm_numa_event: [__SlateRecord_types_h_197; 6],
    vmemmap_tails: [*mut page; 16],
}


fn wmark_pages(z: *const zone, w: u32) -> u64 {
unsafe { *((*z)._watermark.as_ptr() as *const u64).wrapping_offset((w as u32) as isize) }.wrapping_add(unsafe { (*z).watermark_boost })
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


fn min_wmark_pages(z: *const zone) -> u64 {
wmark_pages(z, 0u32)
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


fn low_wmark_pages(z: *const zone) -> u64 {
wmark_pages(z, 1u32)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
}


fn high_wmark_pages(z: *const zone) -> u64 {
wmark_pages(z, 2u32)
}

fn promo_wmark_pages(z: *const zone) -> u64 {
wmark_pages(z, 3u32)
}

fn zone_managed_pages(zone: *const zone) -> u64 {
atomic_long_read(unsafe { core::ptr::addr_of!((*zone).managed_pages) }) as u64
}

fn zone_cma_pages(zone: *mut zone) -> u64 {
0
}

fn zone_end_pfn(zone: *const zone) -> u64 {
unsafe { (*zone).zone_start_pfn }.wrapping_add(unsafe { (*zone).spanned_pages })
}

fn memdesc_zonenum(flags: *const __SlateRecord_mm_types_h_41) -> u32 {
ASSERT_EXCLUSIVE_BITS!((*flags).f, 3u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64));
((unsafe { (*flags).f }) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64) & 3u64) as u32
}

fn page_zonenum(page: *const page) -> u32 {
memdesc_zonenum(unsafe { core::ptr::addr_of!((*page).flags) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


// Maximum number of zones on a zonelist
#[repr(C)]
#[derive(Clone, Copy)]
struct zoneref {
    zone: *mut zone,
    zone_idx: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct zonelist {
    _zonerefs: [zoneref; 257],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pglist_data {
    node_zones: [zone; 4],
    node_zonelists: [zonelist; 2],
    nr_zones: i32,
    node_start_pfn: u64,
    node_present_pages: u64,
    node_spanned_pages: u64,
    node_id: i32,
    kswapd_wait: wait_queue_head,
    pfmemalloc_wait: wait_queue_head,
    reclaim_wait: [wait_queue_head; 4],
    nr_writeback_throttled: __SlateRecord_types_h_190,
    nr_reclaim_start: u64,
    kswapd: *mut task_struct,
    kswapd_order: i32,
    kswapd_highest_zoneidx: u32,
    kswapd_failures: __SlateRecord_types_h_190,
    kcompactd_max_order: i32,
    kcompactd_highest_zoneidx: u32,
    kcompactd_wait: wait_queue_head,
    kcompactd: *mut task_struct,
    proactive_compact_trigger: bool,
    totalreserve_pages: u64,
    min_unmapped_pages: u64,
    min_slab_pages: u64,
    _pad1_: cacheline_padding,
    __lruvec: lruvec,
    flags: u64,
    _pad2_: cacheline_padding,
    per_cpu_nodestats: *mut per_cpu_nodestat,
    vm_stat: [__SlateRecord_types_h_197; 66],
    memtier: *mut memory_tier,
}


// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
fn populated_zone(zone: *const zone) -> bool {
(unsafe { (*zone).present_pages }) != 0
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


fn zone_to_nid(zone: *const zone) -> i32 {
unsafe { (*zone).node }
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


// for_each_online_pgdat - helper macro to iterate over all online nodes
// @pgdat: pointer to a pg_data_t variable
// do nothing
fn pfn_to_section_nr(pfn: u64) -> u64 {
pfn >> 15i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mem_section_usage {
    rcu: callback_head,
    subsection_map: [u64; 1],
    pageblock_flags: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_section {
    section_mem_map: u64,
    usage: *mut mem_section_usage,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


fn __nr_to_section(nr: u64) -> *mut mem_section {
let root: u64 = SECTION_NR_TO_ROOT!(nr);
if unlikely!(root >= (1u64 << if pgtable_l5_enabled!() != 0 { 52i32 } else { 46i32 }.wrapping_sub(27)).wrapping_add(4096u64 / (core::mem::size_of::<mem_section>() as u64)).wrapping_sub(1) / (4096u64 / (core::mem::size_of::<mem_section>() as u64))) != 0 {
return core::ptr::null_mut::<mem_section>();
}
if unsafe { mem_section }.is_null() || unsafe { *unsafe { mem_section }.wrapping_offset(root as isize) }.is_null() {
return core::ptr::null_mut::<mem_section>();
}
unsafe { *unsafe { mem_section }.wrapping_offset(root as isize) }.wrapping_offset((nr & ((_AC!() << 12i32) / (core::mem::size_of::<mem_section>() as u64)).wrapping_sub(1)) as isize)
}

fn valid_section(section: *const mem_section) -> i32 {
(!section.is_null() && (unsafe { (*section).section_mem_map }) & BIT!(1i32) != 0) as i32
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


fn early_section(section: *const mem_section) -> i32 {
(!section.is_null() && (unsafe { (*section).section_mem_map }) & BIT!(3i32) != 0) as i32
}

fn __pfn_to_section(pfn: u64) -> *mut mem_section {
__nr_to_section(pfn_to_section_nr(pfn))
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


fn subsection_map_index(pfn: u64) -> i32 {
((pfn & 32767u64) / 512u64) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


fn pfn_section_valid(ms: *mut mem_section, pfn: u64) -> i32 {
let idx: i32 = subsection_map_index(pfn);
let mut __v0: *mut mem_section_usage;
loop {
if !(__native_word__19!() || ((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_201() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ms).usage) as *const *mut mem_section_usage) };
let usage: *mut mem_section_usage = __v0;
let __v2: i32 = if !usage.is_null() {
test_bit__2!(idx, unsafe { (*usage).subsection_map.as_mut_ptr() as *mut u64 })
} else {
0
};
__v2
}

fn pfn_valid(pfn: u64) -> i32 {
let mut ret: i32 = 0;
// Ensure the upper PAGE_SHIFT bits are clear in the
// pfn. Else it might lead to false positives when
// some of the upper bits are set, but the lower bits
// match a valid pfn.
if PHYS_PFN!(PFN_PHYS!(pfn)) != pfn {
return 0;
}
if pfn_to_section_nr(pfn) >= 1u64 << if pgtable_l5_enabled!() != 0 { 52i32 } else { 46i32 }.wrapping_sub(27) {
return 0;
}
let ms: *mut mem_section = __pfn_to_section(pfn);
rcu_read_lock_sched();
if valid_section(ms as *const mem_section) == 0 {
rcu_read_unlock_sched();
return 0;
}
// Traditionally early sections always returned pfn_valid() for
// the entire section-sized span.
let __v0: bool = early_section(ms as *const mem_section) != 0 || pfn_section_valid(ms, pfn) != 0;
ret = __v0 as i32;
rcu_read_unlock_sched();
ret
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


fn cpumask_of_node(node: i32) -> *const cpumask {
(unsafe { core::ptr::addr_of_mut!(node_to_cpumask_map[node as usize]) as *mut cpumask }) as *const cpumask
}

fn numa_node_id() -> i32 {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym numa_node, options(pure, readonly, nostack, att_syntax));
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
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym numa_node, options(pure, readonly, nostack, att_syntax));
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
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym numa_node, options(pure, readonly, nostack, att_syntax));
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
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym numa_node, options(pure, readonly, nostack, att_syntax));
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
__v0
}

fn cpu_to_node(cpu: i32) -> i32 {
per_cpu!(numa_node, cpu)
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
struct key_restriction {
    check: Option<unsafe extern "C" fn(*mut key, *const key_type, *const key_payload, *mut key) -> i32>,
    key: *mut key,
    keytype: *mut key_type,
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


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

#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
struct wake_irq {
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
    __slate_bits_0: __slate_bits::__SlateBits304U0,
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
struct dev_pm_qos {
}


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct cgroup_taskset {
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
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
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
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
#[repr(C)]
#[derive(Clone, Copy)]
struct swap_cluster_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits620U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
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
struct __SlateRecord_vmstat_c_1813 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_vmstat_c_1907 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_vmstat_c_2316 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(C, align(64))]
struct __SlateAlign64<T>(T);

unsafe impl<T> Sync for __SlateAlign64<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits588U0 {
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
    pub struct __SlateBits620U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits301U0 {
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
    pub struct __SlateBits67U0 {
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
    pub struct __SlateBits67U1 {
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
    pub struct __SlateBits693U0 {
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
    pub struct __SlateBits304U0 {
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
    pub struct __SlateBits67U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits124U0 {
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
    pub struct __SlateBits131U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits757U0 {
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
    pub struct __SlateBits754U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits757U1 {
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
    pub struct __SlateBits757U2 {
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
    pub struct __SlateBits757U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits632U0 {
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
    pub struct __SlateBits130U0 {
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
    pub struct __SlateBits439U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits149U0 {
        #[bits(24)]
        pub intel_native_model_id: u32,
        #[bits(8)]
        pub intel_type: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits150U0 {
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
    pub struct __SlateBits98U0 {
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
    pub struct __SlateBits99U0 {
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
    pub struct __SlateBits97U0 {
        #[bits(1)]
        pub initialized: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

fn _kmalloc_array_noprof(n: u64, size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
let mut bytes: u64 = 0;
if unlikely!(check_mul_overflow!(n, size, core::ptr::addr_of_mut!(bytes))) != 0 {
return core::ptr::null_mut::<()>();
}
_kmalloc_noprof(bytes, flags, token)
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
    __slate_bits_0: __slate_bits::__SlateBits693U0,
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
    __slate_bits_0: __slate_bits::__SlateBits632U0,
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


// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
// Test whether any specified VMA flag is set, e.g.:
//
// if (vma_flags_test_any(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
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
fn memdesc_nid(mdf: *const __SlateRecord_mm_types_h_41) -> i32 {
ASSERT_EXCLUSIVE_BITS!((*mdf).f, 63u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64));
((unsafe { (*mdf).f }) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64) & 63u64) as i32
}

fn page_to_nid(page: *const page) -> i32 {
memdesc_nid(unsafe { core::ptr::addr_of!((*PF_POISONED_CHECK!(page)).flags) })
}

fn page_zone(page: *const page) -> *mut zone {
unsafe { (*NODE_DATA!(page_to_nid(page))).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((page_zonenum(page) as u32) as isize)
}

fn page_pgdat(page: *const page) -> *mut pglist_data {
NODE_DATA!(page_to_nid(page))
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_event_state {
    event: [u64; 75],
}


fn count_vm_events(mut item: u32, delta: i64) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = ((delta as u64) & 255) as u8;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = ((delta as u64) & 65535) as u16;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = ((delta as u64) & 4294967295) as u32;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = delta as u64;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..cacheline_aligned")]
static mut vm_zone_stat: __SlateAlign64<[__SlateRecord_types_h_197; 10]> = __SlateAlign64(unsafe { core::mem::zeroed() });

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_vm_event_states: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_all_vm_events: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vm_zone_stat: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vm_node_stat: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___mod_zone_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___mod_node_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___inc_zone_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___inc_node_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___dec_zone_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___dec_node_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mod_zone_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_inc_zone_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dec_zone_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mod_node_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_inc_node_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dec_node_page_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section\t\".initcall7.init\", \"a\"\t\t\n__initcall__kmod_vmstat__690_2249_vmstat_late_init7:\t\t\t\n.long\t{16} - .\t\n.previous\t\t\t\t\t\n\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_vmstat__692_2451_extfrag_debug_init6:\t\t\t\n.long\t{17} - .\t\n.previous\t\t\t\t\t\n", sym vm_event_states, sym all_vm_events, sym vm_zone_stat, sym vm_node_stat, sym __mod_zone_page_state, sym __mod_node_page_state, sym __inc_zone_page_state, sym __inc_node_page_state, sym __dec_zone_page_state, sym __dec_node_page_state, sym mod_zone_page_state, sym inc_zone_page_state, sym dec_zone_page_state, sym mod_node_page_state, sym inc_node_page_state, sym dec_node_page_state, sym vmstat_late_init, sym extfrag_debug_init, options(att_syntax));

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..cacheline_aligned")]
static mut vm_node_stat: __SlateAlign64<[__SlateRecord_types_h_197; 66]> = __SlateAlign64(unsafe { core::mem::zeroed() });

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..cacheline_aligned")]
static mut vm_numa_event: __SlateAlign64<[__SlateRecord_types_h_197; 6]> = __SlateAlign64(unsafe { core::mem::zeroed() });

fn zone_numa_event_add(x: i64, zone: *mut zone, item: u32) {
atomic_long_add(x, unsafe { (*zone).vm_numa_event.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize));
atomic_long_add(x, unsafe { core::ptr::addr_of_mut!(vm_numa_event.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize));
}

fn zone_numa_event_state(zone: *mut zone, item: u32) -> u64 {
atomic_long_read(unsafe { (*zone).vm_numa_event.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197) as u64
}

fn global_numa_event_state(item: u32) -> u64 {
atomic_long_read(unsafe { core::ptr::addr_of_mut!(vm_numa_event.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197) as u64
}

fn zone_page_state_add(x: i64, zone: *mut zone, item: u32) {
atomic_long_add(x, unsafe { (*zone).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize));
atomic_long_add(x, unsafe { core::ptr::addr_of_mut!(vm_zone_stat.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize));
}

fn node_page_state_add(x: i64, pgdat: *mut pglist_data, item: u32) {
atomic_long_add(x, unsafe { (*pgdat).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize));
atomic_long_add(x, unsafe { core::ptr::addr_of_mut!(vm_node_stat.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize));
}

fn global_zone_page_state(item: u32) -> u64 {
let mut x: i64 = atomic_long_read(unsafe { core::ptr::addr_of_mut!(vm_zone_stat.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197);
if x < 0 {
x = 0;
}
x as u64
}

fn global_node_page_state_pages(item: u32) -> u64 {
let mut x: i64 = atomic_long_read(unsafe { core::ptr::addr_of_mut!(vm_node_stat.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197);
if x < 0 {
x = 0;
}
x as u64
}

fn zone_page_state(zone: *mut zone, item: u32) -> u64 {
let mut x: i64 = atomic_long_read(unsafe { (*zone).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197);
if x < 0 {
x = 0;
}
x as u64
}

#[unsafe(no_mangle)]
extern "C" fn refresh_zone_stat_thresholds() {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut cpu: i32 = 0;
let mut threshold: i32 = 0;
// Zero current pgdat thresholds
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() {
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
unsafe {
(*per_cpu_ptr__5!(unsafe { (*pgdat).per_cpu_nodestats }, cpu)).stat_threshold = 0;
}
cpu = cpu.wrapping_add(1);
}
pgdat = unsafe { next_online_pgdat(pgdat) };
}
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
let pgdat: *mut pglist_data = unsafe { (*zone).zone_pgdat };
let mut max_drift: u64 = 0;
let mut tolerate_drift: u64 = 0;
threshold = calculate_normal_threshold(zone);
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let mut pgdat_threshold: i32 = 0;
unsafe {
(*per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, cpu)).stat_threshold = threshold as i8;
}
// Base nodestat threshold on the largest populated zone.
pgdat_threshold = (unsafe { (*per_cpu_ptr__5!(unsafe { (*pgdat).per_cpu_nodestats }, cpu)).stat_threshold }) as i32;
let mut __v13: i32;
let __UNIQUE_ID_x__671: i32 = threshold;
let __UNIQUE_ID_y__672: i32 = pgdat_threshold;
loop {
if __types_ok!(__UNIQUE_ID_x__671, __UNIQUE_ID_y__672) == 0 {
unsafe { __compiletime_assert_673() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v13 = if __UNIQUE_ID_x__671 > __UNIQUE_ID_y__672 { __UNIQUE_ID_x__671 } else { __UNIQUE_ID_y__672 };
unsafe {
(*per_cpu_ptr__5!(unsafe { (*pgdat).per_cpu_nodestats }, cpu)).stat_threshold = __v13 as i8;
}
cpu = cpu.wrapping_add(1);
}
// Only set percpu_drift_mark if there is a danger that
// NR_FREE_PAGES reports the low watermark is ok when in fact
// the min watermark could be breached by an allocation
tolerate_drift = low_wmark_pages(zone as *const zone).wrapping_sub(min_wmark_pages(zone as *const zone));
max_drift = num_online_cpus().wrapping_mul(threshold as u32) as u64;
if max_drift > tolerate_drift {
unsafe {
(*zone).percpu_drift_mark = high_wmark_pages(zone as *const zone).wrapping_add(max_drift);
}
}
}
zone = unsafe { next_zone(zone) };
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.calculate_pressure_threshold")]
extern "C" fn calculate_pressure_threshold(zone: *mut zone) -> i32 {
let mut threshold: i32 = 0;
let mut watermark_distance: i32 = 0;
// As vmstats are not up to date, there is drift between the estimated
// and real values. For high thresholds and a high number of CPUs, it
// is possible for the min watermark to be breached while the estimated
// value looks fine. The pressure threshold is a reduced value such
// that even the maximum amount of drift will not accidentally breach
// the min watermark
watermark_distance = low_wmark_pages(zone as *const zone).wrapping_sub(min_wmark_pages(zone as *const zone)) as i32;
let mut __v0: i32;
let __UNIQUE_ID_x__662: i32 = 1;
let __UNIQUE_ID_y__663: i32 = ((watermark_distance as u32) / num_online_cpus()) as i32;
loop {
if __types_ok!(__UNIQUE_ID_x__662, __UNIQUE_ID_y__663) == 0 {
unsafe { __compiletime_assert_664() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__662 > __UNIQUE_ID_y__663 { __UNIQUE_ID_x__662 } else { __UNIQUE_ID_y__663 };
threshold = __v0;
// Maximum threshold is 125
let mut __v2: i32;
let __UNIQUE_ID_x__665: i32 = 125;
let __UNIQUE_ID_y__666: i32 = threshold;
loop {
if __types_ok!(__UNIQUE_ID_x__665, __UNIQUE_ID_y__666) == 0 {
unsafe { __compiletime_assert_667() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__665 < __UNIQUE_ID_y__666 { __UNIQUE_ID_x__665 } else { __UNIQUE_ID_y__666 };
threshold = __v2;
threshold
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.calculate_normal_threshold")]
extern "C" fn calculate_normal_threshold(zone: *mut zone) -> i32 {
let mut threshold: i32 = 0;
let mut mem: i32 = 0; // memory in 128 MB units
// The threshold scales with the number of processors and the amount
// of memory per zone. More memory means that we can defer updates for
// longer, more processors could lead to more contention.
// fls() is used to have a cheap way of logarithmic scaling.
//
// Some sample thresholds:
//
// Threshold	Processors	(fls)	Zonesize	fls(mem)+1
// 8		1		1	0.9-1 GB	4
// 16		2		2	0.9-1 GB	4
// 20 		2		2	1-2 GB		5
// 24		2		2	2-4 GB		6
// 28		2		2	4-8 GB		7
// 32		2		2	8-16 GB		8
// 4		2		2	<128M		1
// 30		4		3	2-4 GB		5
// 48		4		3	8-16 GB		8
// 32		8		4	1-2 GB		4
// 32		8		4	0.9-1GB		4
// 10		16		5	<128M		1
// 40		16		5	900M		4
// 70		64		7	2-4 GB		5
// 84		64		7	4-8 GB		6
// 108		512		9	4-8 GB		6
// 125		1024		10	8-16 GB		8
// 125		1024		10	16-32 GB	9
mem = (zone_managed_pages(zone as *const zone) >> 15i32) as i32;
threshold = 2i32.wrapping_mul(fls(num_online_cpus())).wrapping_mul(1i32.wrapping_add(fls(mem as u32)));
// Maximum threshold is 125
let mut __v0: i32;
let __UNIQUE_ID_x__668: i32 = 125;
let __UNIQUE_ID_y__669: i32 = threshold;
loop {
if __types_ok!(__UNIQUE_ID_x__668, __UNIQUE_ID_y__669) == 0 {
unsafe { __compiletime_assert_670() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__668 < __UNIQUE_ID_y__669 { __UNIQUE_ID_x__668 } else { __UNIQUE_ID_y__669 };
threshold = __v0;
threshold
}

#[unsafe(no_mangle)]
static vmstat_text: __SlateAlign16<[*const u8; 161]> = __SlateAlign16([(b"nr_free_pages\0".as_ptr() as *mut u8) as *const u8, (b"nr_free_pages_blocks\0".as_ptr() as *mut u8) as *const u8, (b"nr_zone_inactive_anon\0".as_ptr() as *mut u8) as *const u8, (b"nr_zone_active_anon\0".as_ptr() as *mut u8) as *const u8, (b"nr_zone_inactive_file\0".as_ptr() as *mut u8) as *const u8, (b"nr_zone_active_file\0".as_ptr() as *mut u8) as *const u8, (b"nr_zone_unevictable\0".as_ptr() as *mut u8) as *const u8, (b"nr_zone_write_pending\0".as_ptr() as *mut u8) as *const u8, (b"nr_mlock\0".as_ptr() as *mut u8) as *const u8, (b"nr_free_cma\0".as_ptr() as *mut u8) as *const u8, (b"numa_hit\0".as_ptr() as *mut u8) as *const u8, (b"numa_miss\0".as_ptr() as *mut u8) as *const u8, (b"numa_foreign\0".as_ptr() as *mut u8) as *const u8, (b"numa_interleave\0".as_ptr() as *mut u8) as *const u8, (b"numa_local\0".as_ptr() as *mut u8) as *const u8, (b"numa_other\0".as_ptr() as *mut u8) as *const u8, (b"nr_inactive_anon\0".as_ptr() as *mut u8) as *const u8, (b"nr_active_anon\0".as_ptr() as *mut u8) as *const u8, (b"nr_inactive_file\0".as_ptr() as *mut u8) as *const u8, (b"nr_active_file\0".as_ptr() as *mut u8) as *const u8, (b"nr_unevictable\0".as_ptr() as *mut u8) as *const u8, (b"nr_slab_reclaimable\0".as_ptr() as *mut u8) as *const u8, (b"nr_slab_unreclaimable\0".as_ptr() as *mut u8) as *const u8, (b"nr_isolated_anon\0".as_ptr() as *mut u8) as *const u8, (b"nr_isolated_file\0".as_ptr() as *mut u8) as *const u8, (b"workingset_nodes\0".as_ptr() as *mut u8) as *const u8, (b"workingset_refault_anon\0".as_ptr() as *mut u8) as *const u8, (b"workingset_refault_file\0".as_ptr() as *mut u8) as *const u8, (b"workingset_activate_anon\0".as_ptr() as *mut u8) as *const u8, (b"workingset_activate_file\0".as_ptr() as *mut u8) as *const u8, (b"workingset_restore_anon\0".as_ptr() as *mut u8) as *const u8, (b"workingset_restore_file\0".as_ptr() as *mut u8) as *const u8, (b"workingset_nodereclaim\0".as_ptr() as *mut u8) as *const u8, (b"nr_anon_pages\0".as_ptr() as *mut u8) as *const u8, (b"nr_mapped\0".as_ptr() as *mut u8) as *const u8, (b"nr_file_pages\0".as_ptr() as *mut u8) as *const u8, (b"nr_dirty\0".as_ptr() as *mut u8) as *const u8, (b"nr_writeback\0".as_ptr() as *mut u8) as *const u8, (b"nr_shmem\0".as_ptr() as *mut u8) as *const u8, (b"nr_shmem_hugepages\0".as_ptr() as *mut u8) as *const u8, (b"nr_shmem_pmdmapped\0".as_ptr() as *mut u8) as *const u8, (b"nr_file_hugepages\0".as_ptr() as *mut u8) as *const u8, (b"nr_file_pmdmapped\0".as_ptr() as *mut u8) as *const u8, (b"nr_anon_transparent_hugepages\0".as_ptr() as *mut u8) as *const u8, (b"nr_vmscan_write\0".as_ptr() as *mut u8) as *const u8, (b"nr_vmscan_immediate_reclaim\0".as_ptr() as *mut u8) as *const u8, (b"nr_dirtied\0".as_ptr() as *mut u8) as *const u8, (b"nr_written\0".as_ptr() as *mut u8) as *const u8, (b"nr_throttled_written\0".as_ptr() as *mut u8) as *const u8, (b"nr_kernel_misc_reclaimable\0".as_ptr() as *mut u8) as *const u8, (b"nr_foll_pin_acquired\0".as_ptr() as *mut u8) as *const u8, (b"nr_foll_pin_released\0".as_ptr() as *mut u8) as *const u8, (b"nr_vmalloc\0".as_ptr() as *mut u8) as *const u8, (b"nr_kernel_stack\0".as_ptr() as *mut u8) as *const u8, (b"nr_page_table_pages\0".as_ptr() as *mut u8) as *const u8, (b"nr_sec_page_table_pages\0".as_ptr() as *mut u8) as *const u8, (b"nr_iommu_pages\0".as_ptr() as *mut u8) as *const u8, (b"nr_swapcached\0".as_ptr() as *mut u8) as *const u8, (b"pgdemote_kswapd\0".as_ptr() as *mut u8) as *const u8, (b"pgdemote_direct\0".as_ptr() as *mut u8) as *const u8, (b"pgdemote_khugepaged\0".as_ptr() as *mut u8) as *const u8, (b"pgdemote_proactive\0".as_ptr() as *mut u8) as *const u8, (b"pgsteal_kswapd\0".as_ptr() as *mut u8) as *const u8, (b"pgsteal_direct\0".as_ptr() as *mut u8) as *const u8, (b"pgsteal_khugepaged\0".as_ptr() as *mut u8) as *const u8, (b"pgsteal_proactive\0".as_ptr() as *mut u8) as *const u8, (b"pgsteal_anon\0".as_ptr() as *mut u8) as *const u8, (b"pgsteal_file\0".as_ptr() as *mut u8) as *const u8, (b"pgscan_kswapd\0".as_ptr() as *mut u8) as *const u8, (b"pgscan_direct\0".as_ptr() as *mut u8) as *const u8, (b"pgscan_khugepaged\0".as_ptr() as *mut u8) as *const u8, (b"pgscan_proactive\0".as_ptr() as *mut u8) as *const u8, (b"pgscan_anon\0".as_ptr() as *mut u8) as *const u8, (b"pgscan_file\0".as_ptr() as *mut u8) as *const u8, (b"pgrotate_anon\0".as_ptr() as *mut u8) as *const u8, (b"pgrotate_file\0".as_ptr() as *mut u8) as *const u8, (b"pgrefill\0".as_ptr() as *mut u8) as *const u8, (b"nr_hugetlb\0".as_ptr() as *mut u8) as *const u8, (b"nr_balloon_pages\0".as_ptr() as *mut u8) as *const u8, (b"nr_kernel_file_pages\0".as_ptr() as *mut u8) as *const u8, (b"nr_gpu_active\0".as_ptr() as *mut u8) as *const u8, (b"nr_gpu_reclaim\0".as_ptr() as *mut u8) as *const u8, (b"nr_dirty_threshold\0".as_ptr() as *mut u8) as *const u8, (b"nr_dirty_background_threshold\0".as_ptr() as *mut u8) as *const u8, (b"nr_memmap_pages\0".as_ptr() as *mut u8) as *const u8, (b"nr_memmap_boot_pages\0".as_ptr() as *mut u8) as *const u8, (b"pgpgin\0".as_ptr() as *mut u8) as *const u8, (b"pgpgout\0".as_ptr() as *mut u8) as *const u8, (b"pswpin\0".as_ptr() as *mut u8) as *const u8, (b"pswpout\0".as_ptr() as *mut u8) as *const u8, (b"pgalloc_dma\0".as_ptr() as *mut u8) as *const u8, (b"pgalloc_dma32\0".as_ptr() as *mut u8) as *const u8, (b"pgalloc_normal\0".as_ptr() as *mut u8) as *const u8, (b"pgalloc_movable\0".as_ptr() as *mut u8) as *const u8, (b"allocstall_dma\0".as_ptr() as *mut u8) as *const u8, (b"allocstall_dma32\0".as_ptr() as *mut u8) as *const u8, (b"allocstall_normal\0".as_ptr() as *mut u8) as *const u8, (b"allocstall_movable\0".as_ptr() as *mut u8) as *const u8, (b"pgskip_dma\0".as_ptr() as *mut u8) as *const u8, (b"pgskip_dma32\0".as_ptr() as *mut u8) as *const u8, (b"pgskip_normal\0".as_ptr() as *mut u8) as *const u8, (b"pgskip_movable\0".as_ptr() as *mut u8) as *const u8, (b"pgfree\0".as_ptr() as *mut u8) as *const u8, (b"pgactivate\0".as_ptr() as *mut u8) as *const u8, (b"pgdeactivate\0".as_ptr() as *mut u8) as *const u8, (b"pglazyfree\0".as_ptr() as *mut u8) as *const u8, (b"pgfault\0".as_ptr() as *mut u8) as *const u8, (b"pgmajfault\0".as_ptr() as *mut u8) as *const u8, (b"pglazyfreed\0".as_ptr() as *mut u8) as *const u8, (b"pgreuse\0".as_ptr() as *mut u8) as *const u8, (b"pgscan_direct_throttle\0".as_ptr() as *mut u8) as *const u8, (b"zone_reclaim_success\0".as_ptr() as *mut u8) as *const u8, (b"zone_reclaim_failed\0".as_ptr() as *mut u8) as *const u8, (b"pginodesteal\0".as_ptr() as *mut u8) as *const u8, (b"slabs_scanned\0".as_ptr() as *mut u8) as *const u8, (b"kswapd_inodesteal\0".as_ptr() as *mut u8) as *const u8, (b"kswapd_low_wmark_hit_quickly\0".as_ptr() as *mut u8) as *const u8, (b"kswapd_high_wmark_hit_quickly\0".as_ptr() as *mut u8) as *const u8, (b"pageoutrun\0".as_ptr() as *mut u8) as *const u8, (b"pgrotated\0".as_ptr() as *mut u8) as *const u8, (b"drop_pagecache\0".as_ptr() as *mut u8) as *const u8, (b"drop_slab\0".as_ptr() as *mut u8) as *const u8, (b"oom_kill\0".as_ptr() as *mut u8) as *const u8, (b"pgmigrate_success\0".as_ptr() as *mut u8) as *const u8, (b"pgmigrate_fail\0".as_ptr() as *mut u8) as *const u8, (b"thp_migration_success\0".as_ptr() as *mut u8) as *const u8, (b"thp_migration_fail\0".as_ptr() as *mut u8) as *const u8, (b"thp_migration_split\0".as_ptr() as *mut u8) as *const u8, (b"compact_migrate_scanned\0".as_ptr() as *mut u8) as *const u8, (b"compact_free_scanned\0".as_ptr() as *mut u8) as *const u8, (b"compact_isolated\0".as_ptr() as *mut u8) as *const u8, (b"compact_stall\0".as_ptr() as *mut u8) as *const u8, (b"compact_fail\0".as_ptr() as *mut u8) as *const u8, (b"compact_success\0".as_ptr() as *mut u8) as *const u8, (b"compact_daemon_wake\0".as_ptr() as *mut u8) as *const u8, (b"compact_daemon_migrate_scanned\0".as_ptr() as *mut u8) as *const u8, (b"compact_daemon_free_scanned\0".as_ptr() as *mut u8) as *const u8, (b"htlb_buddy_alloc_success\0".as_ptr() as *mut u8) as *const u8, (b"htlb_buddy_alloc_fail\0".as_ptr() as *mut u8) as *const u8, (b"unevictable_pgs_culled\0".as_ptr() as *mut u8) as *const u8, (b"unevictable_pgs_scanned\0".as_ptr() as *mut u8) as *const u8, (b"unevictable_pgs_rescued\0".as_ptr() as *mut u8) as *const u8, (b"unevictable_pgs_mlocked\0".as_ptr() as *mut u8) as *const u8, (b"unevictable_pgs_munlocked\0".as_ptr() as *mut u8) as *const u8, (b"unevictable_pgs_cleared\0".as_ptr() as *mut u8) as *const u8, (b"unevictable_pgs_stranded\0".as_ptr() as *mut u8) as *const u8, (b"swap_ra\0".as_ptr() as *mut u8) as *const u8, (b"swap_ra_hit\0".as_ptr() as *mut u8) as *const u8, (b"swpin_zero\0".as_ptr() as *mut u8) as *const u8, (b"swpout_zero\0".as_ptr() as *mut u8) as *const u8, (b"direct_map_level2_splits\0".as_ptr() as *mut u8) as *const u8, (b"direct_map_level3_splits\0".as_ptr() as *mut u8) as *const u8, (b"direct_map_level2_collapses\0".as_ptr() as *mut u8) as *const u8, (b"direct_map_level3_collapses\0".as_ptr() as *mut u8) as *const u8, (b"kstack_1k\0".as_ptr() as *mut u8) as *const u8, (b"kstack_2k\0".as_ptr() as *mut u8) as *const u8, (b"kstack_4k\0".as_ptr() as *mut u8) as *const u8, (b"kstack_8k\0".as_ptr() as *mut u8) as *const u8, (b"kstack_16k\0".as_ptr() as *mut u8) as *const u8, (b"nrswpin\0".as_ptr() as *mut u8) as *const u8, (b"nrswpout\0".as_ptr() as *mut u8) as *const u8]);

fn zone_stat_name(item: u32) -> *const u8 {
unsafe { *(core::ptr::addr_of!(vmstat_text.0) as *const *const u8).wrapping_offset((item as u32) as isize) }
}

fn numa_stat_name(item: u32) -> *const u8 {
unsafe { *(core::ptr::addr_of!(vmstat_text.0) as *const *const u8).wrapping_offset(10u32.wrapping_add(item as u32) as isize) }
}

fn node_stat_name(item: u32) -> *const u8 {
unsafe { *(core::ptr::addr_of!(vmstat_text.0) as *const *const u8).wrapping_offset(16u32.wrapping_add(item as u32) as isize) }
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


// module_init() - driver initialization entry point
// @x: function to be run at kernel boot time or module insertion
//
// module_init() will either be called during do_initcalls() (if
// builtin) or at module insertion time (if a module).  There can only
// be one per module.
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
    __slate_bits_0: __slate_bits::__SlateBits757U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits757U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits757U2,
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
    __slate_bits_3: __slate_bits::__SlateBits757U3,
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
    __slate_bits_0: __slate_bits::__SlateBits754U0,
}


fn cpuhp_setup_state_nocalls(state: i32, name: *const u8, startup: Option<unsafe extern "C" fn(u32) -> i32>, teardown: Option<unsafe extern "C" fn(u32) -> i32>) -> i32 {
unsafe { __cpuhp_setup_state(state, name, false, startup, teardown, false) }
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
struct debugfs_short_fops {
    read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
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
    __slate_bits_0: __slate_bits::__SlateBits588U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
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


fn cpu_is_isolated(cpu: i32) -> bool {
!unsafe { housekeeping_test_cpu(cpu, 1u32) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma {
    root: *mut anon_vma,
    rwsem: rw_semaphore,
    refcount: __SlateRecord_types_h_190,
    num_children: u64,
    num_active_vmas: u64,
    parent: *mut anon_vma,
    rb_root: rb_root_cached,
}


// Compare PTEs respecting the soft-dirty bit.
// Compare PTEs respecting the writable bit.
// Merge PTE write bits: if any PTE in the batch is writable, modify the
// PTE at @ptentp to be writable.
// Merge PTE young and dirty bits: if any PTE in the batch is young or dirty,
// modify the PTE at @ptentp to be young or dirty, respectively.
#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
    zone: *mut zone,
    migratetype: i32,
    order: i32,
    page: *mut page,
}


#[unsafe(no_mangle)]
static mut mm_percpu_wq: *mut workqueue_struct = unsafe { core::mem::zeroed() };

static mut sysctl_vm_numa_stat: i32 = 1;

/// zero numa counters within a zone
fn zero_zone_numa_counters(zone: *mut zone) {
let mut cpu: i32 = 0;
for item in 0i32..6i32 {
atomic_long_set(unsafe { (*zone).vm_numa_event.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset(item as isize), 0i64);
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
unsafe {
*((*per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, cpu)).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset(item as isize) = 0;
}
cpu = cpu.wrapping_add(1);
}
}
}

/// zero numa counters of all the populated zones
fn zero_zones_numa_counters() {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
zero_zone_numa_counters(zone);
}
zone = unsafe { next_zone(zone) };
}
}

/// zero global numa counters
fn zero_global_numa_counters() {
for item in 0i32..6i32 {
atomic_long_set(unsafe { core::ptr::addr_of_mut!(vm_numa_event.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset(item as isize), 0i64);
}
}

fn invalid_numa_statistics() {
zero_zones_numa_counters();
zero_global_numa_counters();
}

static mut vm_numa_stat_lock: mutex = {
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

#[unsafe(link_section = ".text.slate_distinct.vmstat.sysctl_vm_numa_stat_handler")]
extern "C" fn sysctl_vm_numa_stat_handler(table: *const ctl_table, write: i32, buffer: *mut (), length: *mut u64, ppos: *mut i64) -> i32 {
let mut oldval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(oldval) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
mutex_lock(core::ptr::addr_of_mut!(vm_numa_stat_lock));
if write != 0 {
*__slate_slot_2 = sysctl_vm_numa_stat;
}
*__slate_slot_3 = proc_dointvec_minmax(table, write, buffer, length, ppos);
if !(*__slate_slot_3 != 0 || write == 0) {
if *__slate_slot_2 != sysctl_vm_numa_stat {
if sysctl_vm_numa_stat == 1 {
static_branch_enable!(core::ptr::addr_of_mut!(vm_numa_stat_key));
while false {
}
_printk(c"\x016enable numa statistics\n".as_ptr() as *const u8);
} else {
static_branch_disable!(core::ptr::addr_of_mut!(vm_numa_stat_key));
invalid_numa_statistics();
while false {
}
_printk(c"\x016disable numa statistics, and clear numa counters\n".as_ptr() as *const u8);
}
}
}
mutex_unlock(core::ptr::addr_of_mut!(vm_numa_stat_lock));
return *__slate_slot_3;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..percpu")]
static mut vm_event_states: vm_event_state = vm_event_state { event: {
let mut __t0: [u64; 75] = unsafe { core::mem::zeroed() };
__t0[0] = 0u64;
__t0
} };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vm_event_states_658: *mut () = (unsafe { core::ptr::addr_of_mut!(vm_event_states) }) as *mut ();

fn sum_vm_events(ret: *mut u64) {
let mut cpu: i32 = 0;
unsafe { memset(ret as *mut (), 0i32, 75u64.wrapping_mul(core::mem::size_of::<u64>() as u64)) };
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let this: *mut vm_event_state = per_cpu_ptr__13!(unsafe { core::ptr::addr_of_mut!(vm_event_states) }, cpu);
for i in 0i32..75i32 {
let __v3: *mut u64 = ret.wrapping_offset(i as isize);
let __v4: u64 = unsafe { *__v3 };
let __v5: u64 = __v4.wrapping_add(unsafe { *((*this).event.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) });
unsafe {
*__v3 = __v5;
}
}
cpu = cpu.wrapping_add(1);
}
}

/// Accumulate the vm event counters across all CPUs.
/// The result is unavoidably approximate - it can change
/// during and after execution of this function.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.all_vm_events")]
extern "C" fn all_vm_events(ret: *mut u64) {
unsafe { cpus_read_lock() };
sum_vm_events(ret);
unsafe { cpus_read_unlock() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_all_vm_events_659: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u64)>, *mut ()>(Some(all_vm_events)) };

/// Fold the foreign cpu events into our own.
///
/// This is adding to the events on one processor
/// but keeps the global counts constant.
#[unsafe(no_mangle)]
extern "C" fn vm_events_fold_cpu(cpu: i32) {
let fold_state: *mut vm_event_state = per_cpu_ptr__13!(unsafe { core::ptr::addr_of_mut!(vm_event_states) }, cpu);
for i in 0i32..75i32 {
count_vm_events(i as u32, (unsafe { *((*fold_state).event.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) }) as i64);
unsafe {
*((*fold_state).event.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) = 0;
}
}
}

/// Manage combined zone based / global counters
///
/// vm_stat contains the global counters
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vm_zone_stat_660: *mut () = (unsafe { core::ptr::addr_of_mut!(vm_zone_stat.0) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vm_node_stat_661: *mut () = (unsafe { core::ptr::addr_of_mut!(vm_node_stat.0) }) as *mut ();

fn fold_vm_zone_numa_events(zone: *mut zone) {
let mut zone_numa_events: __SlateAlign16<[u64; 6]> = __SlateAlign16({
let mut __t0: [u64; 6] = unsafe { core::mem::zeroed() };
__t0[0] = 0u64;
__t0
});
let mut cpu: i32 = 0;
let mut item: u32 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let pzstats: *mut per_cpu_zonestat = per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, cpu);
item = 0;
while (item as u32) < 6 {
let mut __v3: u64;
let mut __ai_ptr: *mut u64 = unsafe { (*pzstats).vm_numa_event.as_mut_ptr() as *mut u64 }.wrapping_offset((item as u32) as isize);
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u64>() as u64);
let mut __v5: u64;
let mut __ret: u64 = 0;
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
unsafe {
let __t1: u64;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u64 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
}
2 => {
unsafe {
let __t2: u64;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u64 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t2 as u64) };
}
}
4 => {
unsafe {
let __t3: u64;
core::arch::asm!("xchgl {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
}
8 => {
unsafe {
let __t4: u64;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t4, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t4 as u64) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v5 = __ret;
__v3 = __v5;
let __v7: u64 = __v3;
let __v8: *mut u64 = (zone_numa_events.0.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize);
let __v9: u64 = unsafe { *__v8 };
unsafe {
*__v8 = __v9.wrapping_add(__v7);
}
item = (item as u32).wrapping_add(1) as u32;
}
cpu = cpu.wrapping_add(1);
}
item = 0;
while (item as u32) < 6 {
zone_numa_event_add(zone_numa_events.0[(item as u32) as usize] as i64, zone, item);
item = (item as u32).wrapping_add(1) as u32;
}
}

#[unsafe(no_mangle)]
extern "C" fn fold_vm_numa_events() {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
fold_vm_zone_numa_events(zone);
}
zone = unsafe { next_zone(zone) };
}
}

// Refresh the thresholds for each zone.
#[unsafe(no_mangle)]
extern "C" fn set_pgdat_percpu_threshold(pgdat: *mut pglist_data, calculate_pressure: Option<unsafe extern "C" fn(*mut zone) -> i32>) {
let mut cpu: i32 = 0;
let mut threshold: i32 = 0;
let mut i: i32 = 0;
while i < unsafe { (*pgdat).nr_zones } {
let zone: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset(i as isize);
if (unsafe { (*zone).percpu_drift_mark }) != 0 {
threshold = unsafe { calculate_pressure.unwrap()(zone) };
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
unsafe {
(*per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, cpu)).stat_threshold = threshold as i8;
}
cpu = cpu.wrapping_add(1);
}
}
i = i.wrapping_add(1);
}
}

/// For use when we know that interrupts are disabled,
/// or when we know that preemption is disabled and that
/// particular counter cannot be updated from interrupt context.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.__mod_zone_page_state")]
extern "C" fn __mod_zone_page_state(zone: *mut zone, item: u32, delta: i64) {
let mut pcp: *mut per_cpu_zonestat = unsafe { (*zone).per_cpu_zonestats };
let mut p: *mut i8 = unsafe { (*pcp).vm_stat_diff.as_mut_ptr() as *mut i8 }.wrapping_offset((item as u32) as isize);
let mut x: i64 = 0;
let mut t: i64 = 0;
// Accurate vmstat updates require a RMW. On !PREEMPT_RT kernels,
// atomicity is provided by IRQs being disabled -- either explicitly
// or via local_lock_irq. On PREEMPT_RT, local_lock_irq only disables
// CPU migrations and preemption potentially corrupts a counter so
// disable preemption.
loop {
if IS_ENABLED!() != 0 {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
} else {
lockdep_assert_preemption_disabled!();
}
if true {
break;
}
}
let mut __v3: i8;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v4: i8;
{
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v6: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t0, in(reg) (p as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v6 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v6;
}
2 => {
let mut __v8: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t1, in(reg) (p as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v8 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v8;
}
4 => {
let mut __v9: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t2, in(reg) (p as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v9 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v9;
}
8 => {
let mut __v10: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t3, in(reg) (p as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v10 = pfo_val__ as i8;
pscr_ret__ = __v10;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v4 = pscr_ret__;
}
__v3 = __v4;
x = delta.wrapping_add((__v3 as i32) as i64);
let mut __v11: i8;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v12: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v14: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t4, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v14 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v14;
}
2 => {
let mut __v16: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t5, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v16 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v16;
}
4 => {
let mut __v17: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t6, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v17 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v17;
}
8 => {
let mut __v18: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t7, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v18 = pfo_val__ as i8;
pscr_ret__ = __v18;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v12 = pscr_ret__;
__v11 = __v12;
t = __v11 as i64;
if unlikely!(abs__2!(x) > t) != 0 {
zone_page_state_add(x, zone, item);
x = 0;
}
__this_cpu_preempt_check(c"write".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
loop {
let mut pto_val__: u8 = ((x as u64) & 255) as u8;
if false {
let pto_tmp__: i8 = x as i8;
}
unsafe {
core::arch::asm!("movb {1:l}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
let mut pto_val__: u16 = ((x as u64) & 65535) as u16;
if false {
let pto_tmp__: i8 = x as i8;
}
unsafe {
core::arch::asm!("movw {1:x}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
let mut pto_val__: u32 = ((x as u64) & 4294967295) as u32;
if false {
let pto_tmp__: i8 = x as i8;
}
unsafe {
core::arch::asm!("movl {1:e}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
let mut pto_val__: u64 = x as u64;
if false {
let pto_tmp__: i8 = x as i8;
}
unsafe {
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
preempt_enable_nested();
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___mod_zone_page_state_674: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut zone, u32, i64)>, *mut ()>(Some(__mod_zone_page_state)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.__mod_node_page_state")]
extern "C" fn __mod_node_page_state(pgdat: *mut pglist_data, item: u32, mut delta: i64) {
let mut pcp: *mut per_cpu_nodestat = unsafe { (*pgdat).per_cpu_nodestats };
let mut p: *mut i8 = unsafe { (*pcp).vm_node_stat_diff.as_mut_ptr() as *mut i8 }.wrapping_offset((item as u32) as isize);
let mut x: i64 = 0;
let mut t: i64 = 0;
if vmstat_item_in_bytes((item as u32) as i32) {
// Only cgroups use subpage accounting right now; at
// the global level, these items still change in
// multiples of whole pages. Store them as pages
// internally to keep the per-cpu counters compact.
VM_WARN_ON_ONCE!();
delta >>= 12i32;
}
// See __mod_zone_page_state()
loop {
if IS_ENABLED!() != 0 {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
} else {
lockdep_assert_preemption_disabled!();
}
if true {
break;
}
}
let mut __v5: i8;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v6: i8;
{
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v8: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t0, in(reg) (p as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v8 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v8;
}
2 => {
let mut __v10: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t1, in(reg) (p as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v10 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v10;
}
4 => {
let mut __v11: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t2, in(reg) (p as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v11 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v11;
}
8 => {
let mut __v12: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t3, in(reg) (p as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v12 = pfo_val__ as i8;
pscr_ret__ = __v12;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v6 = pscr_ret__;
}
__v5 = __v6;
x = delta.wrapping_add((__v5 as i32) as i64);
let mut __v13: i8;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v14: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v16: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t4, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v16 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v16;
}
2 => {
let mut __v18: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t5, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v18 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v18;
}
4 => {
let mut __v19: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t6, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v19 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v19;
}
8 => {
let mut __v20: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t7, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v20 = pfo_val__ as i8;
pscr_ret__ = __v20;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v14 = pscr_ret__;
__v13 = __v14;
t = __v13 as i64;
if unlikely!(abs__2!(x) > t) != 0 {
node_page_state_add(x, pgdat, item);
x = 0;
}
__this_cpu_preempt_check(c"write".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
loop {
let mut pto_val__: u8 = ((x as u64) & 255) as u8;
if false {
let pto_tmp__: i8 = x as i8;
}
unsafe {
core::arch::asm!("movb {1:l}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
let mut pto_val__: u16 = ((x as u64) & 65535) as u16;
if false {
let pto_tmp__: i8 = x as i8;
}
unsafe {
core::arch::asm!("movw {1:x}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
let mut pto_val__: u32 = ((x as u64) & 4294967295) as u32;
if false {
let pto_tmp__: i8 = x as i8;
}
unsafe {
core::arch::asm!("movl {1:e}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
let mut pto_val__: u64 = x as u64;
if false {
let pto_tmp__: i8 = x as i8;
}
unsafe {
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
preempt_enable_nested();
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___mod_node_page_state_675: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pglist_data, u32, i64)>, *mut ()>(Some(__mod_node_page_state)) };

/// Optimized increment and decrement functions.
///
/// These are only for a single page and therefore can take a struct page *
/// argument instead of struct zone *. This allows the inclusion of the code
/// generated for page_zone(page) into the optimized functions.
///
/// No overflow check is necessary and therefore the differential can be
/// incremented or decremented in place which may allow the compilers to
/// generate better code.
/// The increment or decrement is known and therefore one boundary check can
/// be omitted.
///
/// NOTE: These functions are very performance sensitive. Change only
/// with care.
///
/// Some processors have inc/dec instructions that are atomic vs an interrupt.
/// However, the code must first determine the differential location in a zone
/// based on the processor number and then inc/dec the counter. There is no
/// guarantee without disabling preemption that the processor will not change
/// in between and therefore the atomicity vs. interrupt cannot be exploited
/// in a useful way here.
#[unsafe(no_mangle)]
extern "C" fn __inc_zone_state(zone: *mut zone, item: u32) {
let mut pcp: *mut per_cpu_zonestat = unsafe { (*zone).per_cpu_zonestats };
let mut p: *mut i8 = unsafe { (*pcp).vm_stat_diff.as_mut_ptr() as *mut i8 }.wrapping_offset((item as u32) as isize);
let mut v: i8 = 0;
let mut t: i8 = 0;
// See __mod_zone_page_state()
loop {
if IS_ENABLED!() != 0 {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
} else {
lockdep_assert_preemption_disabled!();
}
if true {
break;
}
}
let mut __v3: i8;
__this_cpu_preempt_check(c"add_return".as_ptr() as *const u8);
let mut __v4: i8;
let mut pscr2_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v6: i8;
let mut paro_tmp__: u8 = 1u8;
unsafe {
let __t0: u32;
core::arch::asm!("xaddb {0:l}, %gs:({1:r})", inlateout(reg) (paro_tmp__ as u8) as u32 => __t0, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v6 = ((paro_tmp__ as i32).wrapping_add(1) as u64) as i8;
pscr2_ret__ = __v6;
}
2 => {
let mut __v8: i8;
let mut paro_tmp__: u16 = 1u16;
unsafe {
let __t1: u16;
core::arch::asm!("xaddw {0:x}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u16 => __t1, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v8 = ((paro_tmp__ as i32).wrapping_add(1) as u64) as i8;
pscr2_ret__ = __v8;
}
4 => {
let mut __v9: i8;
let mut paro_tmp__: u32 = 1u32;
unsafe {
let __t2: u32;
core::arch::asm!("xaddl {0:e}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u32 => __t2, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v9 = (paro_tmp__.wrapping_add(1) as u64) as i8;
pscr2_ret__ = __v9;
}
8 => {
let mut __v10: i8;
let mut paro_tmp__: u64 = 1;
unsafe {
let __t3: u64;
core::arch::asm!("xaddq {0:r}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u64 => __t3, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v10 = paro_tmp__.wrapping_add(1) as i8;
pscr2_ret__ = __v10;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v4 = pscr2_ret__;
__v3 = __v4;
v = __v3;
let mut __v11: i8;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v12: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v14: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t4, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v14 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v14;
}
2 => {
let mut __v16: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t5, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v16 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v16;
}
4 => {
let mut __v17: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t6, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v17 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v17;
}
8 => {
let mut __v18: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t7, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v18 = pfo_val__ as i8;
pscr_ret__ = __v18;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v12 = pscr_ret__;
__v11 = __v12;
t = __v11;
if unlikely!((v as i32) > (t as i32)) != 0 {
let overstep: i8 = ((t as i32) >> 1i32) as i8;
zone_page_state_add((v as i32).wrapping_add(overstep as i32) as i64, zone, item);
__this_cpu_preempt_check(c"write".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
loop {
let mut pto_val__: u8 = (((overstep as i32).wrapping_neg() as u64) & 255) as u8;
if false {
let pto_tmp__: i8 = (overstep as i32).wrapping_neg() as i8;
}
unsafe {
core::arch::asm!("movb {1:l}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
let mut pto_val__: u16 = (((overstep as i32).wrapping_neg() as u64) & 65535) as u16;
if false {
let pto_tmp__: i8 = (overstep as i32).wrapping_neg() as i8;
}
unsafe {
core::arch::asm!("movw {1:x}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
let mut pto_val__: u32 = (((overstep as i32).wrapping_neg() as u64) & 4294967295) as u32;
if false {
let pto_tmp__: i8 = (overstep as i32).wrapping_neg() as i8;
}
unsafe {
core::arch::asm!("movl {1:e}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
let mut pto_val__: u64 = (overstep as i32).wrapping_neg() as u64;
if false {
let pto_tmp__: i8 = (overstep as i32).wrapping_neg() as i8;
}
unsafe {
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
}
preempt_enable_nested();
}

#[unsafe(no_mangle)]
extern "C" fn __inc_node_state(pgdat: *mut pglist_data, item: u32) {
let mut pcp: *mut per_cpu_nodestat = unsafe { (*pgdat).per_cpu_nodestats };
let mut p: *mut i8 = unsafe { (*pcp).vm_node_stat_diff.as_mut_ptr() as *mut i8 }.wrapping_offset((item as u32) as isize);
let mut v: i8 = 0;
let mut t: i8 = 0;
VM_WARN_ON_ONCE!();
// See __mod_zone_page_state()
loop {
if IS_ENABLED!() != 0 {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
} else {
lockdep_assert_preemption_disabled!();
}
if true {
break;
}
}
let mut __v3: i8;
__this_cpu_preempt_check(c"add_return".as_ptr() as *const u8);
let mut __v4: i8;
let mut pscr2_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v6: i8;
let mut paro_tmp__: u8 = 1u8;
unsafe {
let __t0: u32;
core::arch::asm!("xaddb {0:l}, %gs:({1:r})", inlateout(reg) (paro_tmp__ as u8) as u32 => __t0, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v6 = ((paro_tmp__ as i32).wrapping_add(1) as u64) as i8;
pscr2_ret__ = __v6;
}
2 => {
let mut __v8: i8;
let mut paro_tmp__: u16 = 1u16;
unsafe {
let __t1: u16;
core::arch::asm!("xaddw {0:x}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u16 => __t1, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v8 = ((paro_tmp__ as i32).wrapping_add(1) as u64) as i8;
pscr2_ret__ = __v8;
}
4 => {
let mut __v9: i8;
let mut paro_tmp__: u32 = 1u32;
unsafe {
let __t2: u32;
core::arch::asm!("xaddl {0:e}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u32 => __t2, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v9 = (paro_tmp__.wrapping_add(1) as u64) as i8;
pscr2_ret__ = __v9;
}
8 => {
let mut __v10: i8;
let mut paro_tmp__: u64 = 1;
unsafe {
let __t3: u64;
core::arch::asm!("xaddq {0:r}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u64 => __t3, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v10 = paro_tmp__.wrapping_add(1) as i8;
pscr2_ret__ = __v10;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v4 = pscr2_ret__;
__v3 = __v4;
v = __v3;
let mut __v11: i8;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v12: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v14: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t4, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v14 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v14;
}
2 => {
let mut __v16: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t5, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v16 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v16;
}
4 => {
let mut __v17: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t6, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v17 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v17;
}
8 => {
let mut __v18: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t7, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v18 = pfo_val__ as i8;
pscr_ret__ = __v18;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v12 = pscr_ret__;
__v11 = __v12;
t = __v11;
if unlikely!((v as i32) > (t as i32)) != 0 {
let overstep: i8 = ((t as i32) >> 1i32) as i8;
node_page_state_add((v as i32).wrapping_add(overstep as i32) as i64, pgdat, item);
__this_cpu_preempt_check(c"write".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
loop {
let mut pto_val__: u8 = (((overstep as i32).wrapping_neg() as u64) & 255) as u8;
if false {
let pto_tmp__: i8 = (overstep as i32).wrapping_neg() as i8;
}
unsafe {
core::arch::asm!("movb {1:l}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
let mut pto_val__: u16 = (((overstep as i32).wrapping_neg() as u64) & 65535) as u16;
if false {
let pto_tmp__: i8 = (overstep as i32).wrapping_neg() as i8;
}
unsafe {
core::arch::asm!("movw {1:x}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
let mut pto_val__: u32 = (((overstep as i32).wrapping_neg() as u64) & 4294967295) as u32;
if false {
let pto_tmp__: i8 = (overstep as i32).wrapping_neg() as i8;
}
unsafe {
core::arch::asm!("movl {1:e}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
let mut pto_val__: u64 = (overstep as i32).wrapping_neg() as u64;
if false {
let pto_tmp__: i8 = (overstep as i32).wrapping_neg() as i8;
}
unsafe {
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
}
preempt_enable_nested();
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.__inc_zone_page_state")]
extern "C" fn __inc_zone_page_state(page: *mut page, item: u32) {
__inc_zone_state(page_zone(page as *const page), item);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___inc_zone_page_state_676: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(__inc_zone_page_state)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.__inc_node_page_state")]
extern "C" fn __inc_node_page_state(page: *mut page, item: u32) {
__inc_node_state(page_pgdat(page as *const page), item);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___inc_node_page_state_677: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(__inc_node_page_state)) };

#[unsafe(no_mangle)]
extern "C" fn __dec_zone_state(zone: *mut zone, item: u32) {
let mut pcp: *mut per_cpu_zonestat = unsafe { (*zone).per_cpu_zonestats };
let mut p: *mut i8 = unsafe { (*pcp).vm_stat_diff.as_mut_ptr() as *mut i8 }.wrapping_offset((item as u32) as isize);
let mut v: i8 = 0;
let mut t: i8 = 0;
// See __mod_zone_page_state()
loop {
if IS_ENABLED!() != 0 {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
} else {
lockdep_assert_preemption_disabled!();
}
if true {
break;
}
}
let mut __v3: i8;
__this_cpu_preempt_check(c"add_return".as_ptr() as *const u8);
let mut __v4: i8;
let mut pscr2_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v6: i8;
let mut paro_tmp__: u8 = ((1i32.wrapping_neg() as u64) & 255) as u8;
unsafe {
let __t0: u32;
core::arch::asm!("xaddb {0:l}, %gs:({1:r})", inlateout(reg) (paro_tmp__ as u8) as u32 => __t0, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v6 = ((paro_tmp__ as i32).wrapping_add(1i32.wrapping_neg()) as u64) as i8;
pscr2_ret__ = __v6;
}
2 => {
let mut __v8: i8;
let mut paro_tmp__: u16 = ((1i32.wrapping_neg() as u64) & 65535) as u16;
unsafe {
let __t1: u16;
core::arch::asm!("xaddw {0:x}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u16 => __t1, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v8 = ((paro_tmp__ as i32).wrapping_add(1i32.wrapping_neg()) as u64) as i8;
pscr2_ret__ = __v8;
}
4 => {
let mut __v9: i8;
let mut paro_tmp__: u32 = ((1i32.wrapping_neg() as u64) & 4294967295) as u32;
unsafe {
let __t2: u32;
core::arch::asm!("xaddl {0:e}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u32 => __t2, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v9 = (paro_tmp__.wrapping_add(1i32.wrapping_neg() as u32) as u64) as i8;
pscr2_ret__ = __v9;
}
8 => {
let mut __v10: i8;
let mut paro_tmp__: u64 = 1i32.wrapping_neg() as u64;
unsafe {
let __t3: u64;
core::arch::asm!("xaddq {0:r}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u64 => __t3, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v10 = paro_tmp__.wrapping_add(1i32.wrapping_neg() as u64) as i8;
pscr2_ret__ = __v10;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v4 = pscr2_ret__;
__v3 = __v4;
v = __v3;
let mut __v11: i8;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v12: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v14: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t4, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v14 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v14;
}
2 => {
let mut __v16: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t5, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v16 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v16;
}
4 => {
let mut __v17: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t6, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v17 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v17;
}
8 => {
let mut __v18: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t7, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v18 = pfo_val__ as i8;
pscr_ret__ = __v18;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v12 = pscr_ret__;
__v11 = __v12;
t = __v11;
if unlikely!((v as i32) < (t as i32).wrapping_neg()) != 0 {
let overstep: i8 = ((t as i32) >> 1i32) as i8;
zone_page_state_add((v as i32).wrapping_sub(overstep as i32) as i64, zone, item);
__this_cpu_preempt_check(c"write".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
loop {
let mut pto_val__: u8 = ((overstep as u64) & 255) as u8;
if false {
let pto_tmp__: i8 = overstep;
}
unsafe {
core::arch::asm!("movb {1:l}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
let mut pto_val__: u16 = ((overstep as u64) & 65535) as u16;
if false {
let pto_tmp__: i8 = overstep;
}
unsafe {
core::arch::asm!("movw {1:x}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
let mut pto_val__: u32 = ((overstep as u64) & 4294967295) as u32;
if false {
let pto_tmp__: i8 = overstep;
}
unsafe {
core::arch::asm!("movl {1:e}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
let mut pto_val__: u64 = overstep as u64;
if false {
let pto_tmp__: i8 = overstep;
}
unsafe {
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
}
preempt_enable_nested();
}

#[unsafe(no_mangle)]
extern "C" fn __dec_node_state(pgdat: *mut pglist_data, item: u32) {
let mut pcp: *mut per_cpu_nodestat = unsafe { (*pgdat).per_cpu_nodestats };
let mut p: *mut i8 = unsafe { (*pcp).vm_node_stat_diff.as_mut_ptr() as *mut i8 }.wrapping_offset((item as u32) as isize);
let mut v: i8 = 0;
let mut t: i8 = 0;
VM_WARN_ON_ONCE!();
// See __mod_zone_page_state()
loop {
if IS_ENABLED!() != 0 {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
} else {
lockdep_assert_preemption_disabled!();
}
if true {
break;
}
}
let mut __v3: i8;
__this_cpu_preempt_check(c"add_return".as_ptr() as *const u8);
let mut __v4: i8;
let mut pscr2_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v6: i8;
let mut paro_tmp__: u8 = ((1i32.wrapping_neg() as u64) & 255) as u8;
unsafe {
let __t0: u32;
core::arch::asm!("xaddb {0:l}, %gs:({1:r})", inlateout(reg) (paro_tmp__ as u8) as u32 => __t0, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v6 = ((paro_tmp__ as i32).wrapping_add(1i32.wrapping_neg()) as u64) as i8;
pscr2_ret__ = __v6;
}
2 => {
let mut __v8: i8;
let mut paro_tmp__: u16 = ((1i32.wrapping_neg() as u64) & 65535) as u16;
unsafe {
let __t1: u16;
core::arch::asm!("xaddw {0:x}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u16 => __t1, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v8 = ((paro_tmp__ as i32).wrapping_add(1i32.wrapping_neg()) as u64) as i8;
pscr2_ret__ = __v8;
}
4 => {
let mut __v9: i8;
let mut paro_tmp__: u32 = ((1i32.wrapping_neg() as u64) & 4294967295) as u32;
unsafe {
let __t2: u32;
core::arch::asm!("xaddl {0:e}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u32 => __t2, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v9 = (paro_tmp__.wrapping_add(1i32.wrapping_neg() as u32) as u64) as i8;
pscr2_ret__ = __v9;
}
8 => {
let mut __v10: i8;
let mut paro_tmp__: u64 = 1i32.wrapping_neg() as u64;
unsafe {
let __t3: u64;
core::arch::asm!("xaddq {0:r}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u64 => __t3, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v10 = paro_tmp__.wrapping_add(1i32.wrapping_neg() as u64) as i8;
pscr2_ret__ = __v10;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v4 = pscr2_ret__;
__v3 = __v4;
v = __v3;
let mut __v11: i8;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v12: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v14: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t4, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v14 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v14;
}
2 => {
let mut __v16: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t5, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v16 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v16;
}
4 => {
let mut __v17: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t6, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v17 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v17;
}
8 => {
let mut __v18: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t7, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v18 = pfo_val__ as i8;
pscr_ret__ = __v18;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v12 = pscr_ret__;
__v11 = __v12;
t = __v11;
if unlikely!((v as i32) < (t as i32).wrapping_neg()) != 0 {
let overstep: i8 = ((t as i32) >> 1i32) as i8;
node_page_state_add((v as i32).wrapping_sub(overstep as i32) as i64, pgdat, item);
__this_cpu_preempt_check(c"write".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
loop {
let mut pto_val__: u8 = ((overstep as u64) & 255) as u8;
if false {
let pto_tmp__: i8 = overstep;
}
unsafe {
core::arch::asm!("movb {1:l}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
let mut pto_val__: u16 = ((overstep as u64) & 65535) as u16;
if false {
let pto_tmp__: i8 = overstep;
}
unsafe {
core::arch::asm!("movw {1:x}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
let mut pto_val__: u32 = ((overstep as u64) & 4294967295) as u32;
if false {
let pto_tmp__: i8 = overstep;
}
unsafe {
core::arch::asm!("movl {1:e}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
let mut pto_val__: u64 = overstep as u64;
if false {
let pto_tmp__: i8 = overstep;
}
unsafe {
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) (p as u64) as *mut i8, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
}
preempt_enable_nested();
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.__dec_zone_page_state")]
extern "C" fn __dec_zone_page_state(page: *mut page, item: u32) {
__dec_zone_state(page_zone(page as *const page), item);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___dec_zone_page_state_678: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(__dec_zone_page_state)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.__dec_node_page_state")]
extern "C" fn __dec_node_page_state(page: *mut page, item: u32) {
__dec_node_state(page_pgdat(page as *const page), item);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___dec_node_page_state_679: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(__dec_node_page_state)) };

/// If we have cmpxchg_local support then we do not need to incur the overhead
/// that comes with local_irq_save/restore if we use this_cpu_try_cmpxchg().
///
/// mod_state() modifies the zone counter state through atomic per cpu
/// operations.
///
/// Overstep mode specifies how overstep should handled:
///     0       No overstepping
///     1       Overstepping half of threshold
///     -1      Overstepping minus half of threshold
fn mod_zone_state(zone: *mut zone, item: u32, delta: i64, overstep_mode: i32) {
let mut pcp: *mut per_cpu_zonestat = unsafe { (*zone).per_cpu_zonestats };
let mut p: *mut i8 = unsafe { (*pcp).vm_stat_diff.as_mut_ptr() as *mut i8 }.wrapping_offset((item as u32) as isize);
let mut n: i64 = 0;
let mut t: i64 = 0;
let mut z: i64 = 0;
let mut o: i8 = 0;
let mut __v18: i8;
{
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v20: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t0, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v20 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v20;
}
2 => {
let mut __v22: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t1, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v22 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v22;
}
4 => {
let mut __v23: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t2, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v23 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v23;
}
8 => {
let mut __v24: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t3, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v24 = pfo_val__ as i8;
pscr_ret__ = __v24;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v18 = pscr_ret__;
}
o = __v18;
loop {
z = 0; // overflow to zone counters
// The fetching of the stat_threshold is racy. We may apply
// a counter threshold to the wrong the cpu if we get
// rescheduled while executing here. However, the next
// counter update will apply the threshold again and
// therefore bring the counter under the threshold again.
//
// Most of the time the thresholds are the same anyways
// for all cpus in a zone.
let mut __v25: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v27: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t4, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v27 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v27;
}
2 => {
let mut __v29: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t5, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v29 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v29;
}
4 => {
let mut __v30: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t6, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v30 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v30;
}
8 => {
let mut __v31: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t7, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v31 = pfo_val__ as i8;
pscr_ret__ = __v31;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v25 = pscr_ret__;
t = __v25 as i64;
n = delta.wrapping_add(o as i64);
if abs__2!(n) > t {
let os: i32 = (overstep_mode as i64).wrapping_mul(t >> 1i32) as i32;
// Overflow must be added to zone counters
z = n.wrapping_add(os as i64);
n = os.wrapping_neg() as i64;
}
let mut __v35: bool;
let mut __v0: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v37: i64;
let mut __v2: bool = false;
let __v3: *mut u8 = core::ptr::addr_of_mut!(o) as *mut u8;
let mut __v4: u8 = unsafe { *__v3 };
let mut __v5: u8 = ((n as u64) & 255) as u8;
unsafe {
let __t8: u8;
let __t9: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t8, in(reg) (p as u64) as *mut i8, in(reg) (__v5 as u8) as u32, inlateout("ax") (__v4 as u8) as u32 => __t9, options(nostack, att_syntax));
__v2 = (__t8 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u8, u8>(__t9 as u8) };
}
if unlikely!(!__v2) != 0 {
unsafe {
*__v3 = __v4;
}
}
__v37 = __v2 as i64;
__v0 = __v37 != 0;
}
2 => {
let mut __v39: i64;
let mut __v6: bool = false;
let __v7: *mut u16 = core::ptr::addr_of_mut!(o) as *mut u16;
let mut __v8: u16 = unsafe { *__v7 };
let mut __v9: u16 = ((n as u64) & 65535) as u16;
unsafe {
let __t10: u8;
let __t11: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t10, in(reg) (p as u64) as *mut i8, in(reg) __v9 as u16, inlateout("ax") __v8 as u16 => __t11, options(nostack, att_syntax));
__v6 = (__t10 as u8) != 0;
__v8 = unsafe { core::mem::transmute::<u16, u16>(__t11 as u16) };
}
if unlikely!(!__v6) != 0 {
unsafe {
*__v7 = __v8;
}
}
__v39 = __v6 as i64;
__v0 = __v39 != 0;
}
4 => {
let mut __v40: i64;
let mut __v10: bool = false;
let __v11: *mut u32 = core::ptr::addr_of_mut!(o) as *mut u32;
let mut __v12: u32 = unsafe { *__v11 };
let mut __v13: u32 = ((n as u64) & 4294967295) as u32;
unsafe {
let __t12: u8;
let __t13: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t12, in(reg) (p as u64) as *mut i8, in(reg) __v13 as u32, inlateout("ax") __v12 as u32 => __t13, options(nostack, att_syntax));
__v10 = (__t12 as u8) != 0;
__v12 = unsafe { core::mem::transmute::<u32, u32>(__t13 as u32) };
}
if unlikely!(!__v10) != 0 {
unsafe {
*__v11 = __v12;
}
}
__v40 = __v10 as i64;
__v0 = __v40 != 0;
}
8 => {
let mut __v41: i64;
let mut __v14: bool = false;
let __v15: *mut u64 = core::ptr::addr_of_mut!(o) as *mut u64;
let mut __v16: u64 = unsafe { *__v15 };
let mut __v17: u64 = n as u64;
unsafe {
let __t14: u8;
let __t15: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t14, in(reg) (p as u64) as *mut i8, in(reg) __v17 as u64, inlateout("ax") __v16 as u64 => __t15, options(nostack, att_syntax));
__v14 = (__t14 as u8) != 0;
__v16 = unsafe { core::mem::transmute::<u64, u64>(__t15 as u64) };
}
if unlikely!(!__v14) != 0 {
unsafe {
*__v15 = __v16;
}
}
__v41 = __v14 as i64;
__v0 = __v41 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v35 = __v0;
if __v35 {
break;
}
}
if z != 0 {
zone_page_state_add(z, zone, item);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.mod_zone_page_state")]
extern "C" fn mod_zone_page_state(zone: *mut zone, item: u32, delta: i64) {
mod_zone_state(zone, item, delta, 0i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mod_zone_page_state_680: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut zone, u32, i64)>, *mut ()>(Some(mod_zone_page_state)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.inc_zone_page_state")]
extern "C" fn inc_zone_page_state(page: *mut page, item: u32) {
mod_zone_state(page_zone(page as *const page), item, 1i64, 1i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_inc_zone_page_state_681: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(inc_zone_page_state)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.dec_zone_page_state")]
extern "C" fn dec_zone_page_state(page: *mut page, item: u32) {
mod_zone_state(page_zone(page as *const page), item, 1i32.wrapping_neg() as i64, 1i32.wrapping_neg());
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dec_zone_page_state_682: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(dec_zone_page_state)) };

fn mod_node_state(pgdat: *mut pglist_data, item: u32, mut delta: i32, overstep_mode: i32) {
let mut pcp: *mut per_cpu_nodestat = unsafe { (*pgdat).per_cpu_nodestats };
let mut p: *mut i8 = unsafe { (*pcp).vm_node_stat_diff.as_mut_ptr() as *mut i8 }.wrapping_offset((item as u32) as isize);
let mut n: i64 = 0;
let mut t: i64 = 0;
let mut z: i64 = 0;
let mut o: i8 = 0;
if vmstat_item_in_bytes((item as u32) as i32) {
// Only cgroups use subpage accounting right now; at
// the global level, these items still change in
// multiples of whole pages. Store them as pages
// internally to keep the per-cpu counters compact.
VM_WARN_ON_ONCE!();
delta >>= 12i32;
}
let mut __v20: i8;
{
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v22: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t0, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v22 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v22;
}
2 => {
let mut __v24: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t1, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v24 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v24;
}
4 => {
let mut __v25: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t2, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v25 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v25;
}
8 => {
let mut __v26: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t3, in(reg) (p as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v26 = pfo_val__ as i8;
pscr_ret__ = __v26;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v20 = pscr_ret__;
}
o = __v20;
loop {
z = 0; // overflow to node counters
// The fetching of the stat_threshold is racy. We may apply
// a counter threshold to the wrong the cpu if we get
// rescheduled while executing here. However, the next
// counter update will apply the threshold again and
// therefore bring the counter under the threshold again.
//
// Most of the time the thresholds are the same anyways
// for all cpus in a node.
let mut __v27: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v29: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t4, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v29 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v29;
}
2 => {
let mut __v31: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t5, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v31 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v31;
}
4 => {
let mut __v32: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t6, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v32 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v32;
}
8 => {
let mut __v33: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t7, in(reg) (core::ptr::addr_of_mut!((*pcp).stat_threshold) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v33 = pfo_val__ as i8;
pscr_ret__ = __v33;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v27 = pscr_ret__;
t = __v27 as i64;
n = (delta as i64).wrapping_add(o as i64);
if abs__2!(n) > t {
let os: i32 = (overstep_mode as i64).wrapping_mul(t >> 1i32) as i32;
// Overflow must be added to node counters
z = n.wrapping_add(os as i64);
n = os.wrapping_neg() as i64;
}
let mut __v37: bool;
let mut __v0: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v39: i64;
let mut __v2: bool = false;
let __v3: *mut u8 = core::ptr::addr_of_mut!(o) as *mut u8;
let mut __v4: u8 = unsafe { *__v3 };
let mut __v5: u8 = ((n as u64) & 255) as u8;
unsafe {
let __t8: u8;
let __t9: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t8, in(reg) (p as u64) as *mut i8, in(reg) (__v5 as u8) as u32, inlateout("ax") (__v4 as u8) as u32 => __t9, options(nostack, att_syntax));
__v2 = (__t8 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u8, u8>(__t9 as u8) };
}
if unlikely!(!__v2) != 0 {
unsafe {
*__v3 = __v4;
}
}
__v39 = __v2 as i64;
__v0 = __v39 != 0;
}
2 => {
let mut __v41: i64;
let mut __v6: bool = false;
let __v7: *mut u16 = core::ptr::addr_of_mut!(o) as *mut u16;
let mut __v8: u16 = unsafe { *__v7 };
let mut __v9: u16 = ((n as u64) & 65535) as u16;
unsafe {
let __t10: u8;
let __t11: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t10, in(reg) (p as u64) as *mut i8, in(reg) __v9 as u16, inlateout("ax") __v8 as u16 => __t11, options(nostack, att_syntax));
__v6 = (__t10 as u8) != 0;
__v8 = unsafe { core::mem::transmute::<u16, u16>(__t11 as u16) };
}
if unlikely!(!__v6) != 0 {
unsafe {
*__v7 = __v8;
}
}
__v41 = __v6 as i64;
__v0 = __v41 != 0;
}
4 => {
let mut __v42: i64;
let mut __v10: bool = false;
let __v11: *mut u32 = core::ptr::addr_of_mut!(o) as *mut u32;
let mut __v12: u32 = unsafe { *__v11 };
let mut __v13: u32 = ((n as u64) & 4294967295) as u32;
unsafe {
let __t12: u8;
let __t13: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t12, in(reg) (p as u64) as *mut i8, in(reg) __v13 as u32, inlateout("ax") __v12 as u32 => __t13, options(nostack, att_syntax));
__v10 = (__t12 as u8) != 0;
__v12 = unsafe { core::mem::transmute::<u32, u32>(__t13 as u32) };
}
if unlikely!(!__v10) != 0 {
unsafe {
*__v11 = __v12;
}
}
__v42 = __v10 as i64;
__v0 = __v42 != 0;
}
8 => {
let mut __v43: i64;
let mut __v14: bool = false;
let __v15: *mut u64 = core::ptr::addr_of_mut!(o) as *mut u64;
let mut __v16: u64 = unsafe { *__v15 };
let mut __v17: u64 = n as u64;
unsafe {
let __t14: u8;
let __t15: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t14, in(reg) (p as u64) as *mut i8, in(reg) __v17 as u64, inlateout("ax") __v16 as u64 => __t15, options(nostack, att_syntax));
__v14 = (__t14 as u8) != 0;
__v16 = unsafe { core::mem::transmute::<u64, u64>(__t15 as u64) };
}
if unlikely!(!__v14) != 0 {
unsafe {
*__v15 = __v16;
}
}
__v43 = __v14 as i64;
__v0 = __v43 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v37 = __v0;
if __v37 {
break;
}
}
if z != 0 {
node_page_state_add(z, pgdat, item);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.mod_node_page_state")]
extern "C" fn mod_node_page_state(pgdat: *mut pglist_data, item: u32, delta: i64) {
mod_node_state(pgdat, item, delta as i32, 0i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mod_node_page_state_683: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pglist_data, u32, i64)>, *mut ()>(Some(mod_node_page_state)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.inc_node_page_state")]
extern "C" fn inc_node_page_state(page: *mut page, item: u32) {
mod_node_state(page_pgdat(page as *const page), item, 1i32, 1i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_inc_node_page_state_684: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(inc_node_page_state)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.vmstat.dec_node_page_state")]
extern "C" fn dec_node_page_state(page: *mut page, item: u32) {
mod_node_state(page_pgdat(page as *const page), item, 1i32.wrapping_neg(), 1i32.wrapping_neg());
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dec_node_page_state_685: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(dec_node_page_state)) };

/// Fold a differential into the global counters.
/// Returns whether counters were updated.
fn fold_diff(zone_diff: *mut i32, node_diff: *mut i32) -> i32 {
let mut i: i32 = 0;
let mut changed: bool = false;
while i < 10 {
if (unsafe { *zone_diff.wrapping_offset(i as isize) }) != 0 {
atomic_long_add((unsafe { *zone_diff.wrapping_offset(i as isize) }) as i64, unsafe { core::ptr::addr_of_mut!(vm_zone_stat.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset(i as isize));
changed = true;
}
i = i.wrapping_add(1);
}
i = 0;
while i < 66 {
if (unsafe { *node_diff.wrapping_offset(i as isize) }) != 0 {
atomic_long_add((unsafe { *node_diff.wrapping_offset(i as isize) }) as i64, unsafe { core::ptr::addr_of_mut!(vm_node_stat.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset(i as isize));
changed = true;
}
i = i.wrapping_add(1);
}
changed as i32
}

/// Update the zone counters for the current cpu.
///
/// Note that refresh_cpu_vm_stats strives to only access
/// node local memory. The per cpu pagesets on remote zones are placed
/// in the memory local to the processor using that pageset. So the
/// loop over all zones will access a series of cachelines local to
/// the processor.
///
/// The call to zone_page_state_add updates the cachelines with the
/// statistics in the remote zone struct as well as the global cachelines
/// with the global counters. These could cause remote node cache line
/// bouncing and will have to be only done when necessary.
///
/// The function returns whether global counters were updated.
fn refresh_cpu_vm_stats(do_pagesets: bool) -> bool {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut global_zone_diff: __SlateAlign16<[i32; 10]> = __SlateAlign16({
let mut __t0: [i32; 10] = unsafe { core::mem::zeroed() };
__t0[0] = 0i32;
__t0
});
let mut global_node_diff: __SlateAlign16<[i32; 66]> = __SlateAlign16({
let mut __t1: [i32; 66] = unsafe { core::mem::zeroed() };
__t1[0] = 0i32;
__t1
});
let mut changed: bool = false;
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
let mut pzstats: *mut per_cpu_zonestat = unsafe { (*zone).per_cpu_zonestats };
let mut pcp: *mut per_cpu_pages = unsafe { (*zone).per_cpu_pageset };
i = 0;
while i < 10 {
let mut v: i32 = 0;
let mut __v144: i8;
let mut pscr2_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v146: i8;
let mut pxo_old__: i8 = 0;
let mut __v147: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v149: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t2, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t2 as u8) };
}
__v149 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v149;
}
2 => {
let mut __v151: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t3: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t3, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t3 as u16) };
}
__v151 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v151;
}
4 => {
let mut __v152: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t4, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t4 as u32) };
}
__v152 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v152;
}
8 => {
let mut __v153: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t5: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t5, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t5 as u64) };
}
__v153 = pfo_val__ as i8;
pscr_ret__ = __v153;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v147 = pscr_ret__;
pxo_old__ = __v147;
loop {
let mut __v155: bool;
let mut __v0: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v157: i64;
let mut __v2: bool = false;
let __v3: *mut u8 = core::ptr::addr_of_mut!(pxo_old__) as *mut u8;
let mut __v4: u8 = unsafe { *__v3 };
let mut __v5: u8 = 0u8;
unsafe {
let __t6: u8;
let __t7: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) (__v5 as u8) as u32, inlateout("ax") (__v4 as u8) as u32 => __t7, options(nostack, att_syntax));
__v2 = (__t6 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u8, u8>(__t7 as u8) };
}
if unlikely!(!__v2) != 0 {
unsafe {
*__v3 = __v4;
}
}
__v157 = __v2 as i64;
__v0 = __v157 != 0;
}
2 => {
let mut __v159: i64;
let mut __v6: bool = false;
let __v7: *mut u16 = core::ptr::addr_of_mut!(pxo_old__) as *mut u16;
let mut __v8: u16 = unsafe { *__v7 };
let mut __v9: u16 = 0u16;
unsafe {
let __t8: u8;
let __t9: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t8, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v9 as u16, inlateout("ax") __v8 as u16 => __t9, options(nostack, att_syntax));
__v6 = (__t8 as u8) != 0;
__v8 = unsafe { core::mem::transmute::<u16, u16>(__t9 as u16) };
}
if unlikely!(!__v6) != 0 {
unsafe {
*__v7 = __v8;
}
}
__v159 = __v6 as i64;
__v0 = __v159 != 0;
}
4 => {
let mut __v160: i64;
let mut __v10: bool = false;
let __v11: *mut u32 = core::ptr::addr_of_mut!(pxo_old__) as *mut u32;
let mut __v12: u32 = unsafe { *__v11 };
let mut __v13: u32 = 0u32;
unsafe {
let __t10: u8;
let __t11: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t10, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v13 as u32, inlateout("ax") __v12 as u32 => __t11, options(nostack, att_syntax));
__v10 = (__t10 as u8) != 0;
__v12 = unsafe { core::mem::transmute::<u32, u32>(__t11 as u32) };
}
if unlikely!(!__v10) != 0 {
unsafe {
*__v11 = __v12;
}
}
__v160 = __v10 as i64;
__v0 = __v160 != 0;
}
8 => {
let mut __v161: i64;
let mut __v14: bool = false;
let __v15: *mut u64 = core::ptr::addr_of_mut!(pxo_old__) as *mut u64;
let mut __v16: u64 = unsafe { *__v15 };
let mut __v17: u64 = 0;
unsafe {
let __t12: u8;
let __t13: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t12, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v17 as u64, inlateout("ax") __v16 as u64 => __t13, options(nostack, att_syntax));
__v14 = (__t12 as u8) != 0;
__v16 = unsafe { core::mem::transmute::<u64, u64>(__t13 as u64) };
}
if unlikely!(!__v14) != 0 {
unsafe {
*__v15 = __v16;
}
}
__v161 = __v14 as i64;
__v0 = __v161 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v155 = __v0;
if __v155 {
break;
}
}
__v146 = pxo_old__;
pscr2_ret__ = __v146;
}
2 => {
let mut __v163: i8;
let mut pxo_old__: i8 = 0;
let mut __v164: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v166: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t14: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t14, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t14 as u8) };
}
__v166 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v166;
}
2 => {
let mut __v168: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t15: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t15, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t15 as u16) };
}
__v168 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v168;
}
4 => {
let mut __v169: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t16: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t16, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t16 as u32) };
}
__v169 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v169;
}
8 => {
let mut __v170: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t17: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t17, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t17 as u64) };
}
__v170 = pfo_val__ as i8;
pscr_ret__ = __v170;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v164 = pscr_ret__;
pxo_old__ = __v164;
loop {
let mut __v172: bool;
let mut __v18: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v174: i64;
let mut __v20: bool = false;
let __v21: *mut u8 = core::ptr::addr_of_mut!(pxo_old__) as *mut u8;
let mut __v22: u8 = unsafe { *__v21 };
let mut __v23: u8 = 0u8;
unsafe {
let __t18: u8;
let __t19: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t18, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) (__v23 as u8) as u32, inlateout("ax") (__v22 as u8) as u32 => __t19, options(nostack, att_syntax));
__v20 = (__t18 as u8) != 0;
__v22 = unsafe { core::mem::transmute::<u8, u8>(__t19 as u8) };
}
if unlikely!(!__v20) != 0 {
unsafe {
*__v21 = __v22;
}
}
__v174 = __v20 as i64;
__v18 = __v174 != 0;
}
2 => {
let mut __v176: i64;
let mut __v24: bool = false;
let __v25: *mut u16 = core::ptr::addr_of_mut!(pxo_old__) as *mut u16;
let mut __v26: u16 = unsafe { *__v25 };
let mut __v27: u16 = 0u16;
unsafe {
let __t20: u8;
let __t21: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t20, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v27 as u16, inlateout("ax") __v26 as u16 => __t21, options(nostack, att_syntax));
__v24 = (__t20 as u8) != 0;
__v26 = unsafe { core::mem::transmute::<u16, u16>(__t21 as u16) };
}
if unlikely!(!__v24) != 0 {
unsafe {
*__v25 = __v26;
}
}
__v176 = __v24 as i64;
__v18 = __v176 != 0;
}
4 => {
let mut __v177: i64;
let mut __v28: bool = false;
let __v29: *mut u32 = core::ptr::addr_of_mut!(pxo_old__) as *mut u32;
let mut __v30: u32 = unsafe { *__v29 };
let mut __v31: u32 = 0u32;
unsafe {
let __t22: u8;
let __t23: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t22, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v31 as u32, inlateout("ax") __v30 as u32 => __t23, options(nostack, att_syntax));
__v28 = (__t22 as u8) != 0;
__v30 = unsafe { core::mem::transmute::<u32, u32>(__t23 as u32) };
}
if unlikely!(!__v28) != 0 {
unsafe {
*__v29 = __v30;
}
}
__v177 = __v28 as i64;
__v18 = __v177 != 0;
}
8 => {
let mut __v178: i64;
let mut __v32: bool = false;
let __v33: *mut u64 = core::ptr::addr_of_mut!(pxo_old__) as *mut u64;
let mut __v34: u64 = unsafe { *__v33 };
let mut __v35: u64 = 0;
unsafe {
let __t24: u8;
let __t25: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t24, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v35 as u64, inlateout("ax") __v34 as u64 => __t25, options(nostack, att_syntax));
__v32 = (__t24 as u8) != 0;
__v34 = unsafe { core::mem::transmute::<u64, u64>(__t25 as u64) };
}
if unlikely!(!__v32) != 0 {
unsafe {
*__v33 = __v34;
}
}
__v178 = __v32 as i64;
__v18 = __v178 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v172 = __v18;
if __v172 {
break;
}
}
__v163 = pxo_old__;
pscr2_ret__ = __v163;
}
4 => {
let mut __v179: i8;
let mut pxo_old__: i8 = 0;
let mut __v180: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v182: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t26: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t26, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t26 as u8) };
}
__v182 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v182;
}
2 => {
let mut __v184: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t27: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t27, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t27 as u16) };
}
__v184 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v184;
}
4 => {
let mut __v185: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t28: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t28, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t28 as u32) };
}
__v185 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v185;
}
8 => {
let mut __v186: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t29: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t29, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t29 as u64) };
}
__v186 = pfo_val__ as i8;
pscr_ret__ = __v186;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v180 = pscr_ret__;
pxo_old__ = __v180;
loop {
let mut __v188: bool;
let mut __v36: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v190: i64;
let mut __v38: bool = false;
let __v39: *mut u8 = core::ptr::addr_of_mut!(pxo_old__) as *mut u8;
let mut __v40: u8 = unsafe { *__v39 };
let mut __v41: u8 = 0u8;
unsafe {
let __t30: u8;
let __t31: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t30, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) (__v41 as u8) as u32, inlateout("ax") (__v40 as u8) as u32 => __t31, options(nostack, att_syntax));
__v38 = (__t30 as u8) != 0;
__v40 = unsafe { core::mem::transmute::<u8, u8>(__t31 as u8) };
}
if unlikely!(!__v38) != 0 {
unsafe {
*__v39 = __v40;
}
}
__v190 = __v38 as i64;
__v36 = __v190 != 0;
}
2 => {
let mut __v192: i64;
let mut __v42: bool = false;
let __v43: *mut u16 = core::ptr::addr_of_mut!(pxo_old__) as *mut u16;
let mut __v44: u16 = unsafe { *__v43 };
let mut __v45: u16 = 0u16;
unsafe {
let __t32: u8;
let __t33: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t32, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v45 as u16, inlateout("ax") __v44 as u16 => __t33, options(nostack, att_syntax));
__v42 = (__t32 as u8) != 0;
__v44 = unsafe { core::mem::transmute::<u16, u16>(__t33 as u16) };
}
if unlikely!(!__v42) != 0 {
unsafe {
*__v43 = __v44;
}
}
__v192 = __v42 as i64;
__v36 = __v192 != 0;
}
4 => {
let mut __v193: i64;
let mut __v46: bool = false;
let __v47: *mut u32 = core::ptr::addr_of_mut!(pxo_old__) as *mut u32;
let mut __v48: u32 = unsafe { *__v47 };
let mut __v49: u32 = 0u32;
unsafe {
let __t34: u8;
let __t35: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t34, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v49 as u32, inlateout("ax") __v48 as u32 => __t35, options(nostack, att_syntax));
__v46 = (__t34 as u8) != 0;
__v48 = unsafe { core::mem::transmute::<u32, u32>(__t35 as u32) };
}
if unlikely!(!__v46) != 0 {
unsafe {
*__v47 = __v48;
}
}
__v193 = __v46 as i64;
__v36 = __v193 != 0;
}
8 => {
let mut __v194: i64;
let mut __v50: bool = false;
let __v51: *mut u64 = core::ptr::addr_of_mut!(pxo_old__) as *mut u64;
let mut __v52: u64 = unsafe { *__v51 };
let mut __v53: u64 = 0;
unsafe {
let __t36: u8;
let __t37: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t36, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v53 as u64, inlateout("ax") __v52 as u64 => __t37, options(nostack, att_syntax));
__v50 = (__t36 as u8) != 0;
__v52 = unsafe { core::mem::transmute::<u64, u64>(__t37 as u64) };
}
if unlikely!(!__v50) != 0 {
unsafe {
*__v51 = __v52;
}
}
__v194 = __v50 as i64;
__v36 = __v194 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v188 = __v36;
if __v188 {
break;
}
}
__v179 = pxo_old__;
pscr2_ret__ = __v179;
}
8 => {
let mut __v195: i8;
let mut pxo_old__: i8 = 0;
let mut __v196: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v198: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t38: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t38, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t38 as u8) };
}
__v198 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v198;
}
2 => {
let mut __v200: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t39: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t39, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t39 as u16) };
}
__v200 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v200;
}
4 => {
let mut __v201: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t40: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t40, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t40 as u32) };
}
__v201 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v201;
}
8 => {
let mut __v202: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t41: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t41, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t41 as u64) };
}
__v202 = pfo_val__ as i8;
pscr_ret__ = __v202;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v196 = pscr_ret__;
pxo_old__ = __v196;
loop {
let mut __v204: bool;
let mut __v54: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v206: i64;
let mut __v56: bool = false;
let __v57: *mut u8 = core::ptr::addr_of_mut!(pxo_old__) as *mut u8;
let mut __v58: u8 = unsafe { *__v57 };
let mut __v59: u8 = 0u8;
unsafe {
let __t42: u8;
let __t43: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t42, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) (__v59 as u8) as u32, inlateout("ax") (__v58 as u8) as u32 => __t43, options(nostack, att_syntax));
__v56 = (__t42 as u8) != 0;
__v58 = unsafe { core::mem::transmute::<u8, u8>(__t43 as u8) };
}
if unlikely!(!__v56) != 0 {
unsafe {
*__v57 = __v58;
}
}
__v206 = __v56 as i64;
__v54 = __v206 != 0;
}
2 => {
let mut __v208: i64;
let mut __v60: bool = false;
let __v61: *mut u16 = core::ptr::addr_of_mut!(pxo_old__) as *mut u16;
let mut __v62: u16 = unsafe { *__v61 };
let mut __v63: u16 = 0u16;
unsafe {
let __t44: u8;
let __t45: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t44, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v63 as u16, inlateout("ax") __v62 as u16 => __t45, options(nostack, att_syntax));
__v60 = (__t44 as u8) != 0;
__v62 = unsafe { core::mem::transmute::<u16, u16>(__t45 as u16) };
}
if unlikely!(!__v60) != 0 {
unsafe {
*__v61 = __v62;
}
}
__v208 = __v60 as i64;
__v54 = __v208 != 0;
}
4 => {
let mut __v209: i64;
let mut __v64: bool = false;
let __v65: *mut u32 = core::ptr::addr_of_mut!(pxo_old__) as *mut u32;
let mut __v66: u32 = unsafe { *__v65 };
let mut __v67: u32 = 0u32;
unsafe {
let __t46: u8;
let __t47: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t46, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v67 as u32, inlateout("ax") __v66 as u32 => __t47, options(nostack, att_syntax));
__v64 = (__t46 as u8) != 0;
__v66 = unsafe { core::mem::transmute::<u32, u32>(__t47 as u32) };
}
if unlikely!(!__v64) != 0 {
unsafe {
*__v65 = __v66;
}
}
__v209 = __v64 as i64;
__v54 = __v209 != 0;
}
8 => {
let mut __v210: i64;
let mut __v68: bool = false;
let __v69: *mut u64 = core::ptr::addr_of_mut!(pxo_old__) as *mut u64;
let mut __v70: u64 = unsafe { *__v69 };
let mut __v71: u64 = 0;
unsafe {
let __t48: u8;
let __t49: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t48, in(reg) (((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v71 as u64, inlateout("ax") __v70 as u64 => __t49, options(nostack, att_syntax));
__v68 = (__t48 as u8) != 0;
__v70 = unsafe { core::mem::transmute::<u64, u64>(__t49 as u64) };
}
if unlikely!(!__v68) != 0 {
unsafe {
*__v69 = __v70;
}
}
__v210 = __v68 as i64;
__v54 = __v210 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v204 = __v54;
if __v204 {
break;
}
}
__v195 = pxo_old__;
pscr2_ret__ = __v195;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v144 = pscr2_ret__;
v = __v144 as i32;
if v != 0 {
atomic_long_add(v as i64, unsafe { (*zone).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset(i as isize));
let __v211: *mut i32 = (global_zone_diff.0.as_mut_ptr() as *mut i32).wrapping_offset(i as isize);
let __v212: i32 = unsafe { *__v211 };
unsafe {
*__v211 = __v212.wrapping_add(v);
}
// 3 seconds idle till flush
__this_cpu_preempt_check(c"write".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__15!();
match (const { core::mem::size_of::<u8>() }) as u64 {
1 => {
loop {
let mut pto_val__: u8 = 3u8;
if false {
let pto_tmp__: u8 = 3;
}
unsafe {
core::arch::asm!("movb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
let mut pto_val__: u16 = 3u16;
if false {
let pto_tmp__: u8 = 3;
}
unsafe {
core::arch::asm!("movw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
let mut pto_val__: u32 = 3u32;
if false {
let pto_tmp__: u8 = 3;
}
unsafe {
core::arch::asm!("movl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
let mut pto_val__: u64 = 3;
if false {
let pto_tmp__: u8 = 3;
}
unsafe {
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
}
i = i.wrapping_add(1);
}
if do_pagesets {
cond_resched__55!();
let mut __v225: *mut per_cpu_pages;
__verify_pcpu_ptr__13!();
let mut __v227: *mut per_cpu_pages;
{
let mut tcp_ptr__: u64 = 0;
let mut __v228: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t50: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t50, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t50 as u64) };
}
__v228 = pfo_val__;
tcp_ptr__ = __v228;
tcp_ptr__ = tcp_ptr__.wrapping_add(pcp as u64);
__v227 = tcp_ptr__ as *mut per_cpu_pages;
}
__v225 = __v227;
if unsafe { decay_pcp_high(zone, __v225) } {
changed = true;
}
// Deal with draining the remote pageset of this
// processor
//
// Check if there are pages remaining in this pageset
// if not then there is nothing to expire.
let mut __v231: u8;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v232: u8;
{
let mut pscr_ret__: u8 = 0;
__verify_pcpu_ptr__15!();
match (const { core::mem::size_of::<u8>() }) as u64 {
1 => {
let mut __v234: u8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t51: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t51, in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t51 as u8) };
}
__v234 = (pfo_val__ as u64) as u8;
pscr_ret__ = __v234;
}
2 => {
let mut __v236: u8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t52: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t52, in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t52 as u16) };
}
__v236 = (pfo_val__ as u64) as u8;
pscr_ret__ = __v236;
}
4 => {
let mut __v237: u8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t53: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t53, in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t53 as u32) };
}
__v237 = (pfo_val__ as u64) as u8;
pscr_ret__ = __v237;
}
8 => {
let mut __v238: u8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t54: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t54, in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t54 as u64) };
}
__v238 = pfo_val__ as u8;
pscr_ret__ = __v238;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v232 = pscr_ret__;
}
__v231 = __v232;
let __v239: bool = if __v231 == 0 {
true
} else {
let mut __v240: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v241: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v243: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t55: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t55, in(reg) (core::ptr::addr_of_mut!((*pcp).count) as u64) as *mut i32, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t55 as u8) };
}
__v243 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v243;
}
2 => {
let mut __v245: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t56: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t56, in(reg) (core::ptr::addr_of_mut!((*pcp).count) as u64) as *mut i32, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t56 as u16) };
}
__v245 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v245;
}
4 => {
let mut __v246: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t57: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t57, in(reg) (core::ptr::addr_of_mut!((*pcp).count) as u64) as *mut i32, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t57 as u32) };
}
__v246 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v246;
}
8 => {
let mut __v247: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t58: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t58, in(reg) (core::ptr::addr_of_mut!((*pcp).count) as u64) as *mut i32, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t58 as u64) };
}
__v247 = pfo_val__ as i32;
pscr_ret__ = __v247;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v241 = pscr_ret__;
__v240 = __v241;
__v240 == 0
};
if !__v239 {
// We never drain zones local to this processor.
if zone_to_nid(zone as *const zone) == numa_node_id() {
__this_cpu_preempt_check(c"write".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__15!();
match (const { core::mem::size_of::<u8>() }) as u64 {
1 => {
loop {
let mut pto_val__: u8 = 0u8;
if false {
let pto_tmp__: u8 = 0;
}
unsafe {
core::arch::asm!("movb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
let mut pto_val__: u16 = 0u16;
if false {
let pto_tmp__: u8 = 0;
}
unsafe {
core::arch::asm!("movw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
let mut pto_val__: u32 = 0u32;
if false {
let pto_tmp__: u8 = 0;
}
unsafe {
core::arch::asm!("movl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
let mut pto_val__: u64 = 0;
if false {
let pto_tmp__: u8 = 0;
}
unsafe {
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
let mut __v256: u8;
__this_cpu_preempt_check(c"add_return".as_ptr() as *const u8);
let mut __v257: u8;
let mut pscr2_ret__: u8 = 0;
__verify_pcpu_ptr__15!();
match (const { core::mem::size_of::<u8>() }) as u64 {
1 => {
let mut __v259: u8;
let mut paro_tmp__: u8 = ((1i32.wrapping_neg() as u64) & 255) as u8;
unsafe {
let __t59: u32;
core::arch::asm!("xaddb {0:l}, %gs:({1:r})", inlateout(reg) (paro_tmp__ as u8) as u32 => __t59, in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u8, u8>(__t59 as u8) };
}
__v259 = ((paro_tmp__ as i32).wrapping_add(1i32.wrapping_neg()) as u64) as u8;
pscr2_ret__ = __v259;
}
2 => {
let mut __v261: u8;
let mut paro_tmp__: u16 = ((1i32.wrapping_neg() as u64) & 65535) as u16;
unsafe {
let __t60: u16;
core::arch::asm!("xaddw {0:x}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u16 => __t60, in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u16, u16>(__t60 as u16) };
}
__v261 = ((paro_tmp__ as i32).wrapping_add(1i32.wrapping_neg()) as u64) as u8;
pscr2_ret__ = __v261;
}
4 => {
let mut __v262: u8;
let mut paro_tmp__: u32 = ((1i32.wrapping_neg() as u64) & 4294967295) as u32;
unsafe {
let __t61: u32;
core::arch::asm!("xaddl {0:e}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u32 => __t61, in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u32, u32>(__t61 as u32) };
}
__v262 = (paro_tmp__.wrapping_add(1i32.wrapping_neg() as u32) as u64) as u8;
pscr2_ret__ = __v262;
}
8 => {
let mut __v263: u8;
let mut paro_tmp__: u64 = 1i32.wrapping_neg() as u64;
unsafe {
let __t62: u64;
core::arch::asm!("xaddq {0:r}, %gs:({1:r})", inlateout(reg) paro_tmp__ as u64 => __t62, in(reg) (core::ptr::addr_of_mut!((*pcp).expire) as u64) as *mut u8, options(nostack, att_syntax));
paro_tmp__ = unsafe { core::mem::transmute::<u64, u64>(__t62 as u64) };
}
__v263 = paro_tmp__.wrapping_add(1i32.wrapping_neg() as u64) as u8;
pscr2_ret__ = __v263;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v257 = pscr2_ret__;
__v256 = __v257;
if __v256 != 0 {
changed = true;
} else {
let mut __v264: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v265: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v267: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t63: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t63, in(reg) (core::ptr::addr_of_mut!((*pcp).count) as u64) as *mut i32, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t63 as u8) };
}
__v267 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v267;
}
2 => {
let mut __v269: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t64: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t64, in(reg) (core::ptr::addr_of_mut!((*pcp).count) as u64) as *mut i32, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t64 as u16) };
}
__v269 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v269;
}
4 => {
let mut __v270: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t65: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t65, in(reg) (core::ptr::addr_of_mut!((*pcp).count) as u64) as *mut i32, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t65 as u32) };
}
__v270 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v270;
}
8 => {
let mut __v271: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t66: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t66, in(reg) (core::ptr::addr_of_mut!((*pcp).count) as u64) as *mut i32, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t66 as u64) };
}
__v271 = pfo_val__ as i32;
pscr_ret__ = __v271;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v265 = pscr_ret__;
__v264 = __v265;
if __v264 != 0 {
let mut __v272: *mut per_cpu_pages;
__verify_pcpu_ptr__13!();
let mut __v274: *mut per_cpu_pages;
let mut tcp_ptr__: u64 = 0;
let mut __v275: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t67: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t67, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t67 as u64) };
}
__v275 = pfo_val__;
tcp_ptr__ = __v275;
tcp_ptr__ = tcp_ptr__.wrapping_add(pcp as u64);
__v274 = tcp_ptr__ as *mut per_cpu_pages;
__v272 = __v274;
unsafe { drain_zone_pages(zone, __v272) };
changed = true;
}
}
}
}
}
}
zone = unsafe { next_zone(zone) };
}
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() {
let mut p: *mut per_cpu_nodestat = unsafe { (*pgdat).per_cpu_nodestats };
i = 0;
while i < 66 {
let mut v: i32 = 0;
let mut __v278: i8;
let mut pscr2_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v280: i8;
let mut pxo_old__: i8 = 0;
let mut __v281: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v283: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t68: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t68, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t68 as u8) };
}
__v283 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v283;
}
2 => {
let mut __v285: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t69: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t69, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t69 as u16) };
}
__v285 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v285;
}
4 => {
let mut __v286: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t70: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t70, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t70 as u32) };
}
__v286 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v286;
}
8 => {
let mut __v287: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t71: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t71, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t71 as u64) };
}
__v287 = pfo_val__ as i8;
pscr_ret__ = __v287;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v281 = pscr_ret__;
pxo_old__ = __v281;
loop {
let mut __v289: bool;
let mut __v72: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v291: i64;
let mut __v74: bool = false;
let __v75: *mut u8 = core::ptr::addr_of_mut!(pxo_old__) as *mut u8;
let mut __v76: u8 = unsafe { *__v75 };
let mut __v77: u8 = 0u8;
unsafe {
let __t72: u8;
let __t73: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t72, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) (__v77 as u8) as u32, inlateout("ax") (__v76 as u8) as u32 => __t73, options(nostack, att_syntax));
__v74 = (__t72 as u8) != 0;
__v76 = unsafe { core::mem::transmute::<u8, u8>(__t73 as u8) };
}
if unlikely!(!__v74) != 0 {
unsafe {
*__v75 = __v76;
}
}
__v291 = __v74 as i64;
__v72 = __v291 != 0;
}
2 => {
let mut __v293: i64;
let mut __v78: bool = false;
let __v79: *mut u16 = core::ptr::addr_of_mut!(pxo_old__) as *mut u16;
let mut __v80: u16 = unsafe { *__v79 };
let mut __v81: u16 = 0u16;
unsafe {
let __t74: u8;
let __t75: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t74, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v81 as u16, inlateout("ax") __v80 as u16 => __t75, options(nostack, att_syntax));
__v78 = (__t74 as u8) != 0;
__v80 = unsafe { core::mem::transmute::<u16, u16>(__t75 as u16) };
}
if unlikely!(!__v78) != 0 {
unsafe {
*__v79 = __v80;
}
}
__v293 = __v78 as i64;
__v72 = __v293 != 0;
}
4 => {
let mut __v294: i64;
let mut __v82: bool = false;
let __v83: *mut u32 = core::ptr::addr_of_mut!(pxo_old__) as *mut u32;
let mut __v84: u32 = unsafe { *__v83 };
let mut __v85: u32 = 0u32;
unsafe {
let __t76: u8;
let __t77: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t76, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v85 as u32, inlateout("ax") __v84 as u32 => __t77, options(nostack, att_syntax));
__v82 = (__t76 as u8) != 0;
__v84 = unsafe { core::mem::transmute::<u32, u32>(__t77 as u32) };
}
if unlikely!(!__v82) != 0 {
unsafe {
*__v83 = __v84;
}
}
__v294 = __v82 as i64;
__v72 = __v294 != 0;
}
8 => {
let mut __v295: i64;
let mut __v86: bool = false;
let __v87: *mut u64 = core::ptr::addr_of_mut!(pxo_old__) as *mut u64;
let mut __v88: u64 = unsafe { *__v87 };
let mut __v89: u64 = 0;
unsafe {
let __t78: u8;
let __t79: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t78, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v89 as u64, inlateout("ax") __v88 as u64 => __t79, options(nostack, att_syntax));
__v86 = (__t78 as u8) != 0;
__v88 = unsafe { core::mem::transmute::<u64, u64>(__t79 as u64) };
}
if unlikely!(!__v86) != 0 {
unsafe {
*__v87 = __v88;
}
}
__v295 = __v86 as i64;
__v72 = __v295 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v289 = __v72;
if __v289 {
break;
}
}
__v280 = pxo_old__;
pscr2_ret__ = __v280;
}
2 => {
let mut __v297: i8;
let mut pxo_old__: i8 = 0;
let mut __v298: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v300: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t80: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t80, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t80 as u8) };
}
__v300 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v300;
}
2 => {
let mut __v302: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t81: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t81, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t81 as u16) };
}
__v302 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v302;
}
4 => {
let mut __v303: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t82: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t82, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t82 as u32) };
}
__v303 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v303;
}
8 => {
let mut __v304: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t83: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t83, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t83 as u64) };
}
__v304 = pfo_val__ as i8;
pscr_ret__ = __v304;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v298 = pscr_ret__;
pxo_old__ = __v298;
loop {
let mut __v306: bool;
let mut __v90: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v308: i64;
let mut __v92: bool = false;
let __v93: *mut u8 = core::ptr::addr_of_mut!(pxo_old__) as *mut u8;
let mut __v94: u8 = unsafe { *__v93 };
let mut __v95: u8 = 0u8;
unsafe {
let __t84: u8;
let __t85: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t84, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) (__v95 as u8) as u32, inlateout("ax") (__v94 as u8) as u32 => __t85, options(nostack, att_syntax));
__v92 = (__t84 as u8) != 0;
__v94 = unsafe { core::mem::transmute::<u8, u8>(__t85 as u8) };
}
if unlikely!(!__v92) != 0 {
unsafe {
*__v93 = __v94;
}
}
__v308 = __v92 as i64;
__v90 = __v308 != 0;
}
2 => {
let mut __v310: i64;
let mut __v96: bool = false;
let __v97: *mut u16 = core::ptr::addr_of_mut!(pxo_old__) as *mut u16;
let mut __v98: u16 = unsafe { *__v97 };
let mut __v99: u16 = 0u16;
unsafe {
let __t86: u8;
let __t87: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t86, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v99 as u16, inlateout("ax") __v98 as u16 => __t87, options(nostack, att_syntax));
__v96 = (__t86 as u8) != 0;
__v98 = unsafe { core::mem::transmute::<u16, u16>(__t87 as u16) };
}
if unlikely!(!__v96) != 0 {
unsafe {
*__v97 = __v98;
}
}
__v310 = __v96 as i64;
__v90 = __v310 != 0;
}
4 => {
let mut __v311: i64;
let mut __v100: bool = false;
let __v101: *mut u32 = core::ptr::addr_of_mut!(pxo_old__) as *mut u32;
let mut __v102: u32 = unsafe { *__v101 };
let mut __v103: u32 = 0u32;
unsafe {
let __t88: u8;
let __t89: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t88, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v103 as u32, inlateout("ax") __v102 as u32 => __t89, options(nostack, att_syntax));
__v100 = (__t88 as u8) != 0;
__v102 = unsafe { core::mem::transmute::<u32, u32>(__t89 as u32) };
}
if unlikely!(!__v100) != 0 {
unsafe {
*__v101 = __v102;
}
}
__v311 = __v100 as i64;
__v90 = __v311 != 0;
}
8 => {
let mut __v312: i64;
let mut __v104: bool = false;
let __v105: *mut u64 = core::ptr::addr_of_mut!(pxo_old__) as *mut u64;
let mut __v106: u64 = unsafe { *__v105 };
let mut __v107: u64 = 0;
unsafe {
let __t90: u8;
let __t91: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t90, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v107 as u64, inlateout("ax") __v106 as u64 => __t91, options(nostack, att_syntax));
__v104 = (__t90 as u8) != 0;
__v106 = unsafe { core::mem::transmute::<u64, u64>(__t91 as u64) };
}
if unlikely!(!__v104) != 0 {
unsafe {
*__v105 = __v106;
}
}
__v312 = __v104 as i64;
__v90 = __v312 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v306 = __v90;
if __v306 {
break;
}
}
__v297 = pxo_old__;
pscr2_ret__ = __v297;
}
4 => {
let mut __v313: i8;
let mut pxo_old__: i8 = 0;
let mut __v314: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v316: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t92: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t92, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t92 as u8) };
}
__v316 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v316;
}
2 => {
let mut __v318: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t93: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t93, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t93 as u16) };
}
__v318 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v318;
}
4 => {
let mut __v319: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t94: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t94, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t94 as u32) };
}
__v319 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v319;
}
8 => {
let mut __v320: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t95: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t95, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t95 as u64) };
}
__v320 = pfo_val__ as i8;
pscr_ret__ = __v320;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v314 = pscr_ret__;
pxo_old__ = __v314;
loop {
let mut __v322: bool;
let mut __v108: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v324: i64;
let mut __v110: bool = false;
let __v111: *mut u8 = core::ptr::addr_of_mut!(pxo_old__) as *mut u8;
let mut __v112: u8 = unsafe { *__v111 };
let mut __v113: u8 = 0u8;
unsafe {
let __t96: u8;
let __t97: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t96, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) (__v113 as u8) as u32, inlateout("ax") (__v112 as u8) as u32 => __t97, options(nostack, att_syntax));
__v110 = (__t96 as u8) != 0;
__v112 = unsafe { core::mem::transmute::<u8, u8>(__t97 as u8) };
}
if unlikely!(!__v110) != 0 {
unsafe {
*__v111 = __v112;
}
}
__v324 = __v110 as i64;
__v108 = __v324 != 0;
}
2 => {
let mut __v326: i64;
let mut __v114: bool = false;
let __v115: *mut u16 = core::ptr::addr_of_mut!(pxo_old__) as *mut u16;
let mut __v116: u16 = unsafe { *__v115 };
let mut __v117: u16 = 0u16;
unsafe {
let __t98: u8;
let __t99: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t98, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v117 as u16, inlateout("ax") __v116 as u16 => __t99, options(nostack, att_syntax));
__v114 = (__t98 as u8) != 0;
__v116 = unsafe { core::mem::transmute::<u16, u16>(__t99 as u16) };
}
if unlikely!(!__v114) != 0 {
unsafe {
*__v115 = __v116;
}
}
__v326 = __v114 as i64;
__v108 = __v326 != 0;
}
4 => {
let mut __v327: i64;
let mut __v118: bool = false;
let __v119: *mut u32 = core::ptr::addr_of_mut!(pxo_old__) as *mut u32;
let mut __v120: u32 = unsafe { *__v119 };
let mut __v121: u32 = 0u32;
unsafe {
let __t100: u8;
let __t101: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t100, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v121 as u32, inlateout("ax") __v120 as u32 => __t101, options(nostack, att_syntax));
__v118 = (__t100 as u8) != 0;
__v120 = unsafe { core::mem::transmute::<u32, u32>(__t101 as u32) };
}
if unlikely!(!__v118) != 0 {
unsafe {
*__v119 = __v120;
}
}
__v327 = __v118 as i64;
__v108 = __v327 != 0;
}
8 => {
let mut __v328: i64;
let mut __v122: bool = false;
let __v123: *mut u64 = core::ptr::addr_of_mut!(pxo_old__) as *mut u64;
let mut __v124: u64 = unsafe { *__v123 };
let mut __v125: u64 = 0;
unsafe {
let __t102: u8;
let __t103: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t102, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v125 as u64, inlateout("ax") __v124 as u64 => __t103, options(nostack, att_syntax));
__v122 = (__t102 as u8) != 0;
__v124 = unsafe { core::mem::transmute::<u64, u64>(__t103 as u64) };
}
if unlikely!(!__v122) != 0 {
unsafe {
*__v123 = __v124;
}
}
__v328 = __v122 as i64;
__v108 = __v328 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v322 = __v108;
if __v322 {
break;
}
}
__v313 = pxo_old__;
pscr2_ret__ = __v313;
}
8 => {
let mut __v329: i8;
let mut pxo_old__: i8 = 0;
let mut __v330: i8;
let mut pscr_ret__: i8 = 0;
__verify_pcpu_ptr__7!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v332: i8;
let mut pfo_val__: u8 = 0;
unsafe {
let __t104: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t104, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t104 as u8) };
}
__v332 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v332;
}
2 => {
let mut __v334: i8;
let mut pfo_val__: u16 = 0;
unsafe {
let __t105: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t105, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t105 as u16) };
}
__v334 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v334;
}
4 => {
let mut __v335: i8;
let mut pfo_val__: u32 = 0;
unsafe {
let __t106: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t106, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t106 as u32) };
}
__v335 = (pfo_val__ as u64) as i8;
pscr_ret__ = __v335;
}
8 => {
let mut __v336: i8;
let mut pfo_val__: u64 = 0;
unsafe {
let __t107: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t107, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t107 as u64) };
}
__v336 = pfo_val__ as i8;
pscr_ret__ = __v336;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v330 = pscr_ret__;
pxo_old__ = __v330;
loop {
let mut __v338: bool;
let mut __v126: bool = false;
__verify_pcpu_ptr__11!();
match (const { core::mem::size_of::<i8>() }) as u64 {
1 => {
let mut __v340: i64;
let mut __v128: bool = false;
let __v129: *mut u8 = core::ptr::addr_of_mut!(pxo_old__) as *mut u8;
let mut __v130: u8 = unsafe { *__v129 };
let mut __v131: u8 = 0u8;
unsafe {
let __t108: u8;
let __t109: u32;
core::arch::asm!("cmpxchgb {2:l}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t108, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) (__v131 as u8) as u32, inlateout("ax") (__v130 as u8) as u32 => __t109, options(nostack, att_syntax));
__v128 = (__t108 as u8) != 0;
__v130 = unsafe { core::mem::transmute::<u8, u8>(__t109 as u8) };
}
if unlikely!(!__v128) != 0 {
unsafe {
*__v129 = __v130;
}
}
__v340 = __v128 as i64;
__v126 = __v340 != 0;
}
2 => {
let mut __v342: i64;
let mut __v132: bool = false;
let __v133: *mut u16 = core::ptr::addr_of_mut!(pxo_old__) as *mut u16;
let mut __v134: u16 = unsafe { *__v133 };
let mut __v135: u16 = 0u16;
unsafe {
let __t110: u8;
let __t111: u16;
core::arch::asm!("cmpxchgw {2:x}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t110, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v135 as u16, inlateout("ax") __v134 as u16 => __t111, options(nostack, att_syntax));
__v132 = (__t110 as u8) != 0;
__v134 = unsafe { core::mem::transmute::<u16, u16>(__t111 as u16) };
}
if unlikely!(!__v132) != 0 {
unsafe {
*__v133 = __v134;
}
}
__v342 = __v132 as i64;
__v126 = __v342 != 0;
}
4 => {
let mut __v343: i64;
let mut __v136: bool = false;
let __v137: *mut u32 = core::ptr::addr_of_mut!(pxo_old__) as *mut u32;
let mut __v138: u32 = unsafe { *__v137 };
let mut __v139: u32 = 0u32;
unsafe {
let __t112: u8;
let __t113: u32;
core::arch::asm!("cmpxchgl {2:e}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t112, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v139 as u32, inlateout("ax") __v138 as u32 => __t113, options(nostack, att_syntax));
__v136 = (__t112 as u8) != 0;
__v138 = unsafe { core::mem::transmute::<u32, u32>(__t113 as u32) };
}
if unlikely!(!__v136) != 0 {
unsafe {
*__v137 = __v138;
}
}
__v343 = __v136 as i64;
__v126 = __v343 != 0;
}
8 => {
let mut __v344: i64;
let mut __v140: bool = false;
let __v141: *mut u64 = core::ptr::addr_of_mut!(pxo_old__) as *mut u64;
let mut __v142: u64 = unsafe { *__v141 };
let mut __v143: u64 = 0;
unsafe {
let __t114: u8;
let __t115: u64;
core::arch::asm!("cmpxchgq {2:r}, %gs:({1:r})\nsetz {0}", lateout(reg_byte) __t114, in(reg) (((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) as u64) as *mut i8, in(reg) __v143 as u64, inlateout("ax") __v142 as u64 => __t115, options(nostack, att_syntax));
__v140 = (__t114 as u8) != 0;
__v142 = unsafe { core::mem::transmute::<u64, u64>(__t115 as u64) };
}
if unlikely!(!__v140) != 0 {
unsafe {
*__v141 = __v142;
}
}
__v344 = __v140 as i64;
__v126 = __v344 != 0;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v338 = __v126;
if __v338 {
break;
}
}
__v329 = pxo_old__;
pscr2_ret__ = __v329;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v278 = pscr2_ret__;
v = __v278 as i32;
if v != 0 {
atomic_long_add(v as i64, unsafe { (*pgdat).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset(i as isize));
let __v345: *mut i32 = (global_node_diff.0.as_mut_ptr() as *mut i32).wrapping_offset(i as isize);
let __v346: i32 = unsafe { *__v345 };
unsafe {
*__v345 = __v346.wrapping_add(v);
}
}
i = i.wrapping_add(1);
}
pgdat = unsafe { next_online_pgdat(pgdat) };
}
if fold_diff(global_zone_diff.0.as_mut_ptr() as *mut i32, global_node_diff.0.as_mut_ptr() as *mut i32) != 0 {
changed = true;
}
changed
}

/// Fold the data for an offline cpu into the global array.
/// There cannot be any access by the offline cpu and therefore
/// synchronization is simplified.
#[unsafe(no_mangle)]
extern "C" fn cpu_vm_stats_fold(cpu: i32) {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut global_zone_diff: __SlateAlign16<[i32; 10]> = __SlateAlign16({
let mut __t0: [i32; 10] = unsafe { core::mem::zeroed() };
__t0[0] = 0i32;
__t0
});
let mut global_node_diff: __SlateAlign16<[i32; 66]> = __SlateAlign16({
let mut __t1: [i32; 66] = unsafe { core::mem::zeroed() };
__t1[0] = 0i32;
__t1
});
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
let pzstats: *mut per_cpu_zonestat = per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, cpu);
i = 0;
while i < 10 {
if (unsafe { *((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) }) != 0 {
let v: i32 = (unsafe { *((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) }) as i32;
unsafe {
*((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) = 0;
}
atomic_long_add(v as i64, unsafe { (*zone).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset(i as isize));
let __v3: *mut i32 = (global_zone_diff.0.as_mut_ptr() as *mut i32).wrapping_offset(i as isize);
let __v4: i32 = unsafe { *__v3 };
unsafe {
*__v3 = __v4.wrapping_add(v);
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < 6 {
if (unsafe { *((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) }) != 0 {
let v: u64 = unsafe { *((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) };
unsafe {
*((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) = 0;
}
zone_numa_event_add(v as i64, zone, i as u32);
}
i = i.wrapping_add(1);
}
}
zone = unsafe { next_zone(zone) };
}
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() {
let p: *mut per_cpu_nodestat = per_cpu_ptr__5!(unsafe { (*pgdat).per_cpu_nodestats }, cpu);
i = 0;
while i < 66 {
if (unsafe { *((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) }) != 0 {
let v: i32 = (unsafe { *((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) }) as i32;
unsafe {
*((*p).vm_node_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) = 0;
}
atomic_long_add(v as i64, unsafe { (*pgdat).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset(i as isize));
let __v16: *mut i32 = (global_node_diff.0.as_mut_ptr() as *mut i32).wrapping_offset(i as isize);
let __v17: i32 = unsafe { *__v16 };
unsafe {
*__v16 = __v17.wrapping_add(v);
}
}
i = i.wrapping_add(1);
}
pgdat = unsafe { next_online_pgdat(pgdat) };
}
fold_diff(global_zone_diff.0.as_mut_ptr() as *mut i32, global_node_diff.0.as_mut_ptr() as *mut i32);
}

/// this is only called if !populated_zone(zone), which implies no other users of
/// pset->vm_stat_diff[] exist.
#[unsafe(no_mangle)]
extern "C" fn drain_zonestat(zone: *mut zone, pzstats: *mut per_cpu_zonestat) {
let mut v: u64 = 0;
let mut i: i32 = 0;
while i < 10 {
if (unsafe { *((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) }) != 0 {
v = (unsafe { *((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) }) as u64;
unsafe {
*((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset(i as isize) = 0;
}
zone_page_state_add(v as i64, zone, i as u32);
}
i = i.wrapping_add(1);
}
i = 0;
while i < 6 {
if (unsafe { *((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) }) != 0 {
v = unsafe { *((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) };
unsafe {
*((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) = 0;
}
zone_numa_event_add(v as i64, zone, i as u32);
}
i = i.wrapping_add(1);
}
}

/// Determine the per node value of a stat item. This function
/// is called frequently in a NUMA machine, so try to be as
/// frugal as possible.
#[unsafe(no_mangle)]
extern "C" fn sum_zone_node_page_state(node: i32, item: u32) -> u64 {
let zones: *mut zone = unsafe { (*NODE_DATA!(node)).node_zones.as_mut_ptr() as *mut zone };
let mut count: u64 = 0;
for i in 0i32..4i32 {
let __v0: u64 = zone_page_state(zones.wrapping_offset(i as isize), item);
count = count.wrapping_add(__v0);
}
count
}

/// Determine the per node value of a numa stat item.
#[unsafe(no_mangle)]
extern "C" fn sum_zone_numa_event_state(node: i32, item: u32) -> u64 {
let zones: *mut zone = unsafe { (*NODE_DATA!(node)).node_zones.as_mut_ptr() as *mut zone };
let mut count: u64 = 0;
for i in 0i32..4i32 {
let __v0: u64 = zone_numa_event_state(zones.wrapping_offset(i as isize), item);
count = count.wrapping_add(__v0);
}
count
}

/// Determine the per node value of a stat item.
#[unsafe(no_mangle)]
extern "C" fn node_page_state_pages(pgdat: *mut pglist_data, item: u32) -> u64 {
let mut x: i64 = atomic_long_read(unsafe { (*pgdat).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197);
if x < 0 {
x = 0;
}
x as u64
}

#[unsafe(no_mangle)]
extern "C" fn node_page_state(pgdat: *mut pglist_data, item: u32) -> u64 {
VM_WARN_ON_ONCE!();
node_page_state_pages(pgdat, item)
}

/// Non-clamping variant of node_page_state() intended for callers that
/// snapshot a monotonically-incremented counter and subtract two samples.
/// See global_node_page_state_monotonic() for the rationale.
#[unsafe(no_mangle)]
extern "C" fn node_page_state_monotonic(pgdat: *mut pglist_data, item: u32) -> u64 {
atomic_long_read(unsafe { (*pgdat).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197) as u64
}

/// Count number of pages "struct page" and "struct page_ext" consume.
/// nr_memmap_boot_pages: # of pages allocated by boot allocator
/// nr_memmap_pages: # of pages that were allocated by buddy allocator
static mut nr_memmap_boot_pages: __SlateRecord_types_h_197 = __SlateRecord_types_h_197 { counter: 0i64 };

static mut nr_memmap_pages: __SlateRecord_types_h_197 = __SlateRecord_types_h_197 { counter: 0i64 };

#[unsafe(no_mangle)]
extern "C" fn memmap_boot_pages_add(delta: i64) {
atomic_long_add(delta, unsafe { core::ptr::addr_of_mut!(nr_memmap_boot_pages) });
}

#[unsafe(no_mangle)]
extern "C" fn memmap_pages_add(delta: i64) {
atomic_long_add(delta, unsafe { core::ptr::addr_of_mut!(nr_memmap_pages) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct contig_page_info {
    free_pages: u64,
    free_blocks_total: u64,
    free_blocks_suitable: u64,
}


/// Calculate the number of free pages in a zone, how many contiguous
/// pages are free and how many are large enough to satisfy an allocation of
/// the target size. Note that this function makes no attempt to estimate
/// how many suitable free blocks there *might* be if MOVABLE pages were
/// migrated. Calculating that is possible, but expensive and can be
/// figured out from userspace
fn fill_contig_page_info(zone: *mut zone, suitable_order: u32, info: &mut contig_page_info) {
let mut order: u32 = 0;
unsafe {
(*(info as *mut contig_page_info)).free_pages = 0;
}
unsafe {
(*(info as *mut contig_page_info)).free_blocks_total = 0;
}
unsafe {
(*(info as *mut contig_page_info)).free_blocks_suitable = 0;
}
while order < 11u32 {
let mut blocks: u64 = 0;
// Count number of free blocks.
//
// Access to nr_free is lockless as nr_free is used only for
// diagnostic purposes. Use data_race to avoid KCSAN warning.
blocks = data_race__2!(unsafe { (*((*zone).free_area.as_mut_ptr() as *mut free_area).wrapping_offset(order as isize)).nr_free });
let __v3: u64 = unsafe { (*info).free_blocks_total };
unsafe {
(*info).free_blocks_total = __v3.wrapping_add(blocks);
}
// Count free base pages
let __v6: u64 = unsafe { (*info).free_pages };
unsafe {
(*info).free_pages = __v6.wrapping_add(blocks << order);
}
// Count the suitable free blocks
if order >= suitable_order {
let __v9: u64 = unsafe { (*info).free_blocks_suitable };
unsafe {
(*info).free_blocks_suitable = __v9.wrapping_add(blocks << order.wrapping_sub(suitable_order));
}
}
order = order.wrapping_add(1);
}
}

/// A fragmentation index only makes sense if an allocation of a requested
/// size would fail. If that is true, the fragmentation index indicates
/// whether external fragmentation or a lack of memory was the problem.
/// The value can be used to determine if page reclaim or compaction
/// should be used
fn __fragmentation_index(order: u32, info: &contig_page_info) -> i32 {
let requested: u64 = 1u64 << order;
let mut __v0: i64;
let __ret_warn_on: i32 = (order > 10) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("686: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 686b - ., 3; .popsection\n# {0}", const 686 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___fragmentation_index_1, sym __slate_asm_string___fragmentation_index_2, const 1118 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("687: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 687b - ., 4; .popsection\n# {0}", const 687 as i32, options(nostack, att_syntax));
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
if (unsafe { (*(info as *const contig_page_info)).free_blocks_total }) == 0 {
return 0;
}
// Fragmentation index only makes sense when a request would fail
if (unsafe { (*(info as *const contig_page_info)).free_blocks_suitable }) != 0 {
return 1000i32.wrapping_neg();
}
// Index is between 0 and 1 so return within 3 decimal places
//
// 0 => allocation would fail due to lack of memory
// 1 => allocation would fail due to fragmentation
1000u64.wrapping_sub(div_u64(1000u64.wrapping_add(div_u64(unsafe { (*(info as *const contig_page_info)).free_pages }.wrapping_mul(1000), requested as u32)), (unsafe { (*(info as *const contig_page_info)).free_blocks_total }) as u32)) as i32
}

/// Calculates external fragmentation within a zone wrt the given order.
/// It is defined as the percentage of pages found in blocks of size
/// less than 1 << order. It returns values in range [0, 100].
#[unsafe(no_mangle)]
extern "C" fn extfrag_for_order(zone: *mut zone, order: u32) -> u32 {
let mut info: contig_page_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
fill_contig_page_info(zone, order, unsafe { &mut (*core::ptr::addr_of_mut!(info)) });
if info.free_pages == 0 {
return 0;
}
div_u64(info.free_pages.wrapping_sub(info.free_blocks_suitable << order).wrapping_mul(100), info.free_pages as u32) as u32
}

/// Same as __fragmentation index but allocs contig_page_info on stack
#[unsafe(no_mangle)]
extern "C" fn fragmentation_index(zone: *mut zone, order: u32) -> i32 {
let mut info: contig_page_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
fill_contig_page_info(zone, order, unsafe { &mut (*core::ptr::addr_of_mut!(info)) });
__fragmentation_index(order, unsafe { &(*core::ptr::addr_of_mut!(info)) })
}

// enum zone_stat_item counters
// enum numa_stat_item counters
// enum node_stat_item counters
// system-wide enum vm_stat_item counters
// enum vm_event_item counters
#[unsafe(link_section = ".text.slate_distinct.vmstat.frag_start")]
extern "C" fn frag_start(m: *mut seq_file, pos: *mut i64) -> *mut () {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut node: i64 = unsafe { *pos };
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() && node != 0 {
node = node.wrapping_sub(1);
pgdat = unsafe { next_online_pgdat(pgdat) };
}
pgdat as *mut ()
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.frag_next")]
extern "C" fn frag_next(m: *mut seq_file, arg: *mut (), __v0: *mut i64) -> *mut () {
let pgdat: *mut pglist_data = arg as *mut pglist_data;
let __v1: i64 = unsafe { *__v0 };
unsafe {
*__v0 = __v1.wrapping_add(1);
}
(unsafe { next_online_pgdat(pgdat) }) as *mut ()
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.frag_stop")]
extern "C" fn frag_stop(m: *mut seq_file, arg: *mut ()) {
}

/// Walk zones in a node and print using a callback.
/// If @assert_populated is true, only use callback for zones that are populated.
fn walk_zones_in_node(m: *mut seq_file, pgdat: *mut pglist_data, assert_populated: bool, nolock: bool, print: Option<unsafe extern "C" fn(*mut seq_file, *mut pglist_data, *mut zone)>) {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let node_zones: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone };
let mut flags: u64 = 0;
zone = node_zones;
while ((unsafe { zone.offset_from(node_zones as *mut zone) }) as i64) < 4 {
let __v0: bool = assert_populated && !populated_zone(zone as *const zone);
if !__v0 {
if !nolock {
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
}
unsafe { print.unwrap()(m, pgdat, zone) };
if !nolock {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
}
}
let __v7: *mut zone = zone;
zone = __v7.wrapping_offset(1 as isize);
}
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.frag_show_print")]
extern "C" fn frag_show_print(m: *mut seq_file, pgdat: *mut pglist_data, zone: *mut zone) {
let mut order: i32 = 0;
unsafe { seq_printf(m, c"Node %d, zone %8s ".as_ptr() as *const u8, (*pgdat).node_id, (*zone).name) };
while order < 11i32 {
unsafe { seq_printf(m, c"%6lu ".as_ptr() as *const u8, data_race__2!(unsafe { (*((*zone).free_area.as_mut_ptr() as *mut free_area).wrapping_offset(order as isize)).nr_free })) };
order = order.wrapping_add(1);
}
// Access to nr_free is lockless as nr_free is used only for
// printing purposes. Use data_race to avoid KCSAN warning.
unsafe { seq_putc(m, 10u8) };
}

/// This walks the free areas for each zone.
#[unsafe(link_section = ".text.slate_distinct.vmstat.frag_show")]
extern "C" fn frag_show(m: *mut seq_file, arg: *mut ()) -> i32 {
let pgdat: *mut pglist_data = arg as *mut pglist_data;
walk_zones_in_node(m, pgdat, true, true, Some(frag_show_print));
0
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.pagetypeinfo_showfree_print")]
extern "C" fn pagetypeinfo_showfree_print(m: *mut seq_file, pgdat: *mut pglist_data, zone: *mut zone) {
let mut order: i32 = 0;
for mtype in 0i32..4i32 {
unsafe { seq_printf(m, c"Node %4d, zone %8s, type %12s ".as_ptr() as *const u8, (*pgdat).node_id, (*zone).name, migratetype_names[mtype as usize]) };
order = 0;
while order < 11i32 {
let mut freecount: u64 = 0;
let mut curr: *mut list_head = unsafe { core::mem::zeroed() };
let mut overflow: bool = false;
let area: *mut free_area = unsafe { (*zone).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(order as isize);
curr = unsafe { (*((*area).free_list.as_mut_ptr() as *mut list_head).wrapping_offset(mtype as isize)).next };
while list_is_head(curr as *const list_head, unsafe { (*area).free_list.as_mut_ptr() as *mut list_head }.wrapping_offset(mtype as isize) as *const list_head) == 0 {
// Cap the free_list iteration because it might
// be really large and we are under a spinlock
// so a long time spent here could trigger a
// hard lockup detector. Anyway this is a
// debugging tool so knowing there is a handful
// of pages of this order should be more than
// sufficient.
let __v1: u64 = freecount.wrapping_add(1);
freecount = __v1;
if __v1 >= 100000 {
overflow = true;
break;
}
curr = unsafe { (*curr).next };
}
unsafe { seq_printf(m, c"%s%6lu ".as_ptr() as *const u8, if overflow { c">".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, freecount) };
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*zone).lock) });
cond_resched__35!();
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*zone).lock) });
order = order.wrapping_add(1);
}
unsafe { seq_putc(m, 10u8) };
}
}

/// Print out the free pages at each order for each migratetype
fn pagetypeinfo_showfree(m: *mut seq_file, arg: *mut ()) {
let mut order: i32 = 0;
let pgdat: *mut pglist_data = arg as *mut pglist_data;
// Print header
unsafe { seq_printf(m, c"%-43s ".as_ptr() as *const u8, c"Free pages count per migrate type at order".as_ptr() as *mut u8) };
while order < 11i32 {
unsafe { seq_printf(m, c"%6d ".as_ptr() as *const u8, order) };
order = order.wrapping_add(1);
}
unsafe { seq_putc(m, 10u8) };
walk_zones_in_node(m, pgdat, true, false, Some(pagetypeinfo_showfree_print));
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.pagetypeinfo_showblockcount_print")]
extern "C" fn pagetypeinfo_showblockcount_print(m: *mut seq_file, pgdat: *mut pglist_data, zone: *mut zone) {
let mut mtype: i32 = 0;
let mut pfn: u64 = 0;
let start_pfn: u64 = unsafe { (*zone).zone_start_pfn };
let end_pfn: u64 = zone_end_pfn(zone as *const zone);
let mut count: __SlateAlign16<[u64; 4]> = __SlateAlign16({
let mut __t0: [u64; 4] = unsafe { core::mem::zeroed() };
__t0[0] = 0u64;
__t0
});
pfn = start_pfn;
while pfn < end_pfn {
let page: *mut page = pfn_to_online_page!(pfn);
if !page.is_null() {
if page_zone(page as *const page) == zone {
mtype = (get_pageblock_migratetype!(page) as u32) as i32;
if mtype < 4 {
let __v3: *mut u64 = (count.0.as_mut_ptr() as *mut u64).wrapping_offset(mtype as isize);
let __v4: u64 = unsafe { *__v3 };
unsafe {
*__v3 = __v4.wrapping_add(1);
}
}
}
}
let __v6: u64 = pfn;
pfn = __v6.wrapping_add(1u64 << MIN_T!(u32, 9i32));
}
// Print counts
unsafe { seq_printf(m, c"Node %d, zone %8s ".as_ptr() as *const u8, (*pgdat).node_id, (*zone).name) };
mtype = 0;
while mtype < 4 {
unsafe { seq_printf(m, c"%12lu ".as_ptr() as *const u8, count.0[mtype as usize]) };
mtype = mtype.wrapping_add(1);
}
unsafe { seq_putc(m, 10u8) };
}

/// Print out the number of pageblocks for each migratetype
fn pagetypeinfo_showblockcount(m: *mut seq_file, arg: *mut ()) {
let pgdat: *mut pglist_data = arg as *mut pglist_data;
unsafe { seq_printf(m, c"\n%-23s".as_ptr() as *const u8, c"Number of blocks type ".as_ptr() as *mut u8) };
for mtype in 0i32..4i32 {
unsafe { seq_printf(m, c"%12s ".as_ptr() as *const u8, migratetype_names[mtype as usize]) };
}
unsafe { seq_putc(m, 10u8) };
walk_zones_in_node(m, pgdat, true, false, Some(pagetypeinfo_showblockcount_print));
}

/// Print out the number of pageblocks for each migratetype that contain pages
/// of other types. This gives an indication of how well fallbacks are being
/// contained by rmqueue_fallback(). It requires information from PAGE_OWNER
/// to determine what is going on
fn pagetypeinfo_showmixedcount(m: *mut seq_file, pgdat: *mut pglist_data) {
}

/// This prints out statistics in relation to grouping pages by mobility.
/// It is expensive to collect so do not constantly read the file.
#[unsafe(link_section = ".text.slate_distinct.vmstat.pagetypeinfo_show")]
extern "C" fn pagetypeinfo_show(m: *mut seq_file, arg: *mut ()) -> i32 {
let pgdat: *mut pglist_data = arg as *mut pglist_data;
// check memoryless node
if node_state(unsafe { (*pgdat).node_id }, 3u32) == 0 {
return 0;
}
unsafe { seq_printf(m, c"Page block order: %d\n".as_ptr() as *const u8, MIN_T__2!(u32, 9i32, 10i32)) };
unsafe { seq_printf(m, c"Pages per block:  %lu\n".as_ptr() as *const u8, 1u64 << MIN_T__2!(u32, 9i32, 10i32)) };
unsafe { seq_putc(m, 10u8) };
pagetypeinfo_showfree(m, pgdat as *mut ());
pagetypeinfo_showblockcount(m, pgdat as *mut ());
pagetypeinfo_showmixedcount(m, pgdat);
0
}

static fragmentation_op: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(frag_start), stop: Some(frag_stop), next: Some(frag_next), show: Some(frag_show) });

static pagetypeinfo_op: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(frag_start), stop: Some(frag_stop), next: Some(frag_next), show: Some(pagetypeinfo_show) });

fn is_zone_first_populated(pgdat: *mut pglist_data, zone: *mut zone) -> bool {
for zid in 0i32..4i32 {
let compare: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset(zid as isize);
if populated_zone(compare as *const zone) {
return zone == compare;
}
}
false
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.zoneinfo_show_print")]
extern "C" fn zoneinfo_show_print(m: *mut seq_file, pgdat: *mut pglist_data, zone: *mut zone) {
let mut i: i32 = 0;
unsafe { seq_printf(m, c"Node %d, zone %8s".as_ptr() as *const u8, (*pgdat).node_id, (*zone).name) };
if is_zone_first_populated(pgdat, zone) {
unsafe { seq_printf(m, c"\n  per-node stats".as_ptr() as *const u8) };
i = 0;
while i < 66 {
let mut pages: u64 = node_page_state_pages(pgdat, i as u32);
if vmstat_item_print_in_thp(i as u32) {
pages /= 512u64;
}
unsafe { seq_printf(m, c"\n      %-12s %lu".as_ptr() as *const u8, node_stat_name(i as u32), pages) };
i = i.wrapping_add(1);
}
}
unsafe { seq_printf(m, c"\n  pages free     %lu\n        boost    %lu\n        min      %lu\n        low      %lu\n        high     %lu\n        promo    %lu\n        spanned  %lu\n        present  %lu\n        managed  %lu\n        cma      %lu".as_ptr() as *const u8, zone_page_state(zone, 0u32), (*zone).watermark_boost, min_wmark_pages(zone as *const zone), low_wmark_pages(zone as *const zone), high_wmark_pages(zone as *const zone), promo_wmark_pages(zone as *const zone), (*zone).spanned_pages, (*zone).present_pages, zone_managed_pages(zone as *const zone), zone_cma_pages(zone)) };
unsafe { seq_printf(m, c"\n        protection: (%ld".as_ptr() as *const u8, *((*zone).lowmem_reserve.as_mut_ptr() as *mut i64).wrapping_offset(0 as isize)) };
i = 1;
while (i as u64) < ARRAY_SIZE__56!() {
unsafe { seq_printf(m, c", %ld".as_ptr() as *const u8, *((*zone).lowmem_reserve.as_mut_ptr() as *mut i64).wrapping_offset(i as isize)) };
i = i.wrapping_add(1);
}
unsafe { seq_putc(m, 41u8) };
// If unpopulated, no other information is useful
if !populated_zone(zone as *const zone) {
unsafe { seq_putc(m, 10u8) };
return;
}
i = 0;
while i < 10 {
unsafe { seq_printf(m, c"\n      %-12s %lu".as_ptr() as *const u8, zone_stat_name(i as u32), zone_page_state(zone, i as u32)) };
i = i.wrapping_add(1);
}
fold_vm_zone_numa_events(zone);
i = 0;
while i < 6 {
unsafe { seq_printf(m, c"\n      %-12s %lu".as_ptr() as *const u8, numa_stat_name(i as u32), zone_numa_event_state(zone, i as u32)) };
i = i.wrapping_add(1);
}
unsafe { seq_printf(m, c"\n  pagesets".as_ptr() as *const u8) };
i = 0;
loop {
i = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, i as u64) as i32;
if (i as u32) >= 64 {
break;
}
let pcp: *mut per_cpu_pages = per_cpu_ptr__2!(unsafe { (*zone).per_cpu_pageset }, i);
unsafe { seq_printf(m, c"\n    cpu: %i\n              count:    %i\n              high:     %i\n              batch:    %i\n              high_min: %i\n              high_max: %i".as_ptr() as *const u8, i, (*pcp).count, (*pcp).high, (*pcp).batch, (*pcp).high_min, (*pcp).high_max) };
let pzstats: *mut per_cpu_zonestat = per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, i);
unsafe { seq_printf(m, c"\n  vm stats threshold: %d".as_ptr() as *const u8, (*pzstats).stat_threshold as i32) };
i = i.wrapping_add(1);
}
unsafe { seq_printf(m, c"\n  node_unreclaimable:  %u\n  start_pfn:           %lu\n  reserved_highatomic: %lu\n  free_highatomic:     %lu".as_ptr() as *const u8, kswapd_test_hopeless(pgdat) as i32, (*zone).zone_start_pfn, (*zone).nr_reserved_highatomic, (*zone).nr_free_highatomic) };
unsafe { seq_putc(m, 10u8) };
}

/// Output information about zones in @pgdat.  All zones are printed regardless
/// of whether they are populated or not: lowmem_reserve_ratio operates on the
/// set of all zones and userspace would not be aware of such zones if they are
/// suppressed here (zoneinfo displays the effect of lowmem_reserve_ratio).
#[unsafe(link_section = ".text.slate_distinct.vmstat.zoneinfo_show")]
extern "C" fn zoneinfo_show(m: *mut seq_file, arg: *mut ()) -> i32 {
let pgdat: *mut pglist_data = arg as *mut pglist_data;
walk_zones_in_node(m, pgdat, false, false, Some(zoneinfo_show_print));
0
}

/// iterate over all zones. The same as in
/// fragmentation.
static zoneinfo_op: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(frag_start), stop: Some(frag_stop), next: Some(frag_next), show: Some(zoneinfo_show) });

#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_start")]
extern "C" fn vmstat_start(m: *mut seq_file, pos: *mut i64) -> *mut () {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut v: *mut u64 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
if (unsafe { *pos }) >= (86i32.wrapping_add(if IS_ENABLED__2!() != 0 { 75 } else { 0 }) as i64) {
return core::ptr::null_mut::<()>();
}
loop {
if ARRAY_SIZE__53!() != (86i32.wrapping_add(if IS_ENABLED__2!() != 0 { 75 } else { 0 }) as u64) {
unsafe { __compiletime_assert_688() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
fold_vm_numa_events();
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_array_noprof(86i32.wrapping_add(if true { 75 } else { 0 }) as u64, core::mem::size_of::<u64>() as u64, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_array_noprof(86i32.wrapping_add(if true { 75 } else { 0 }) as u64, core::mem::size_of::<u64>() as u64, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
v = __v1 as *mut u64;
unsafe {
(*m).private = v as *mut ();
}
if v.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64);
}
while i < 10 {
unsafe {
*v.wrapping_offset(i as isize) = global_zone_page_state(i as u32);
}
i = i.wrapping_add(1);
}
let __v9: *mut u64 = v;
v = __v9.wrapping_offset(10 as isize);
i = 0;
while i < 6 {
unsafe {
*v.wrapping_offset(i as isize) = global_numa_event_state(i as u32);
}
i = i.wrapping_add(1);
}
let __v14: *mut u64 = v;
v = __v14.wrapping_offset(6 as isize);
i = 0;
while i < 66 {
unsafe {
*v.wrapping_offset(i as isize) = global_node_page_state_pages(i as u32);
}
if vmstat_item_print_in_thp(i as u32) {
let __v16: *mut u64 = v.wrapping_offset(i as isize);
unsafe {
*__v16 = *__v16 / 512u64;
}
}
i = i.wrapping_add(1);
}
let __v22: *mut u64 = v;
v = __v22.wrapping_offset(66 as isize);
unsafe { global_dirty_limits(v.wrapping_offset(1 as isize), v.wrapping_offset(0 as isize)) };
unsafe {
*v.wrapping_offset(2 as isize) = atomic_long_read(core::ptr::addr_of_mut!(nr_memmap_pages) as *const __SlateRecord_types_h_197) as u64;
}
unsafe {
*v.wrapping_offset(3 as isize) = atomic_long_read(core::ptr::addr_of_mut!(nr_memmap_boot_pages) as *const __SlateRecord_types_h_197) as u64;
}
let __v24: *mut u64 = v;
v = __v24.wrapping_offset(4 as isize);
all_vm_events(v);
let __v26: *mut u64 = v.wrapping_offset(0 as isize);
unsafe {
*__v26 = *__v26 / 2;
}
// sectors -> kbytes
let __v29: *mut u64 = v.wrapping_offset(1 as isize);
unsafe {
*__v29 = *__v29 / 2;
}
((unsafe { (*m).private }) as *mut u64).wrapping_offset((unsafe { *pos }) as isize) as *mut ()
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_next")]
extern "C" fn vmstat_next(m: *mut seq_file, arg: *mut (), pos: *mut i64) -> *mut () {
let __v1: i64 = unsafe { *pos };
unsafe {
*pos = __v1.wrapping_add(1);
}
if (unsafe { *pos }) >= (86i32.wrapping_add(if IS_ENABLED__2!() != 0 { 75 } else { 0 }) as i64) {
return core::ptr::null_mut::<()>();
}
((unsafe { (*m).private }) as *mut u64).wrapping_offset((unsafe { *pos }) as isize) as *mut ()
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_show")]
extern "C" fn vmstat_show(m: *mut seq_file, arg: *mut ()) -> i32 {
let l: *mut u64 = arg as *mut u64;
let off: u64 = ((unsafe { l.offset_from((*m).private as *mut u64) }) as i64) as u64;
seq_puts(m, unsafe { vmstat_text.0[off as usize] });
unsafe { seq_put_decimal_ull(m, c" ".as_ptr() as *const u8, *l) };
unsafe { seq_putc(m, 10u8) };
if off == (86i32.wrapping_add(if IS_ENABLED__2!() != 0 { 75 } else { 0 }).wrapping_sub(1) as u64) {
// We've come to the end - add any deprecated counters to avoid
// breaking userspace which might depend on them being present.
seq_puts(m, c"nr_unstable 0\n".as_ptr() as *const u8);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_stop")]
extern "C" fn vmstat_stop(m: *mut seq_file, arg: *mut ()) {
unsafe { kfree((*m).private as *const ()) };
unsafe {
(*m).private = core::ptr::null_mut::<()>();
}
}

static vmstat_op: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(vmstat_start), stop: Some(vmstat_stop), next: Some(vmstat_next), show: Some(vmstat_show) });

#[unsafe(link_section = ".data..percpu")]
static mut vmstat_work: delayed_work = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".data..read_mostly")]
static mut sysctl_stat_interval: i32 = 1000;

static mut vmstat_late_init_done: i32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.vmstat.refresh_vm_stats")]
extern "C" fn refresh_vm_stats(work: *mut work_struct) {
refresh_cpu_vm_stats(true);
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_refresh")]
extern "C" fn vmstat_refresh(table: *const ctl_table, write: i32, buffer: *mut (), lenp: *mut u64, ppos: *mut i64) -> i32 {
let mut val: i64 = 0;
let mut err: i32 = 0;
let mut i: i32 = 0;
// The regular update, every sysctl_stat_interval, may come later
// than expected: leaving a significant amount in per_cpu buckets.
// This is particularly misleading when checking a quantity of HUGE
// pages, immediately after running a test.  /proc/sys/vm/stat_refresh,
// which can equally be echo'ed to or cat'ted from (by root),
// can be used to update the stats just before reading them.
//
// Oh, and since global_zone_page_state() etc. are so careful to hide
// transiently negative values, report an error here if any of
// the stats is negative, so we know to go looking for imbalance.
err = unsafe { schedule_on_each_cpu(Some(refresh_vm_stats)) };
if err != 0 {
return err;
}
while i < 10 {
'__slate_continue_0: {
// Skip checking stats known to go negative occasionally.
match i {
7 | 9 => {
break '__slate_continue_0;
}
_ => {
}
}
val = atomic_long_read(unsafe { core::ptr::addr_of_mut!(vm_zone_stat.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset(i as isize) as *const __SlateRecord_types_h_197);
if val < 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014%s: %s %ld\n".as_ptr() as *const u8, c"vmstat_refresh".as_ptr() as *mut u8, zone_stat_name(i as u32), val) };
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < 66 {
'__slate_continue_6: {
// Skip checking stats known to go negative occasionally.
match i {
21 => {
break '__slate_continue_6;
}
_ => {
}
}
val = atomic_long_read(unsafe { core::ptr::addr_of_mut!(vm_node_stat.0) as *mut __SlateRecord_types_h_197 }.wrapping_offset(i as isize) as *const __SlateRecord_types_h_197);
if val < 0 {
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x014%s: %s %ld\n".as_ptr() as *const u8, c"vmstat_refresh".as_ptr() as *mut u8, node_stat_name(i as u32), val) };
}
}
i = i.wrapping_add(1);
}
if write != 0 {
unsafe {
*ppos = (*ppos as u64).wrapping_add(*lenp) as i64;
}
} else {
unsafe {
*lenp = 0;
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_update")]
extern "C" fn vmstat_update(w: *mut work_struct) {
if refresh_cpu_vm_stats(true) {
// Counters were updated so we expect more updates
// to occur in the future. Keep on running the
// update worker thread.
let mut __v0: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
__v0 = __v1;
let mut __v8: *mut delayed_work;
__verify_pcpu_ptr__25!();
let mut __v10: *mut delayed_work;
let mut tcp_ptr__: u64 = 0;
let mut __v11: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t4: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t4, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t4 as u64) };
}
__v11 = pfo_val__;
tcp_ptr__ = __v11;
let __v13: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(vmstat_work) }) as u64);
tcp_ptr__ = __v13;
__v10 = tcp_ptr__ as *mut delayed_work;
__v8 = __v10;
unsafe { queue_delayed_work_on(__v0, mm_percpu_wq, __v8, round_jiffies_relative(sysctl_stat_interval as u64)) };
}
}

/// Check if the diffs for a certain cpu indicate that
/// an update is needed.
fn need_update(cpu: i32) -> bool {
let mut last_pgdat: *mut pglist_data = core::ptr::null_mut::<pglist_data>();
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
let pzstats: *mut per_cpu_zonestat = per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, cpu);
// The fast way of checking if there are any vmstat diffs.
if !unsafe { memchr_inv(((*pzstats).vm_stat_diff.as_mut_ptr() as *mut i8) as *const (), 0i32, core::mem::size_of::<[i8; 10]>() as u64) }.is_null() {
return true;
}
if last_pgdat != unsafe { (*zone).zone_pgdat } {
last_pgdat = unsafe { (*zone).zone_pgdat };
let n: *mut per_cpu_nodestat = per_cpu_ptr__5!(unsafe { (*(*zone).zone_pgdat).per_cpu_nodestats }, cpu);
if !unsafe { memchr_inv(((*n).vm_node_stat_diff.as_mut_ptr() as *mut i8) as *const (), 0i32, core::mem::size_of::<[i8; 66]>() as u64) }.is_null() {
return true;
}
}
}
zone = unsafe { next_zone(zone) };
}
false
}

/// Switch off vmstat processing and then fold all the remaining differentials
/// until the diffs stay at zero. The function is used by NOHZ and can only be
/// invoked when tick processing is not active.
#[unsafe(no_mangle)]
extern "C" fn quiet_vmstat() {
if ((unsafe { system_state }) as u32) != 3 {
return;
}
let __v0: bool = if true && false {
let mut __v1: *mut delayed_work;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
if true {
break;
}
}
let mut __v3: *mut delayed_work;
let mut tcp_ptr__: u64 = 0;
let mut __v4: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v4 = pfo_val__;
tcp_ptr__ = __v4;
let __v6: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(vmstat_work) }) as u64);
tcp_ptr__ = __v6;
__v3 = tcp_ptr__ as *mut delayed_work;
__v1 = __v3;
(((unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*__v1).work)).data) }) as *mut u64) as u64) != 0
} else {
false
};
let __v7: i32 = if __v0 && false {
let mut __v8: *mut delayed_work;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
if true {
break;
}
}
let mut __v10: *mut delayed_work;
let mut tcp_ptr__: u64 = 0;
let mut __v11: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t1: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t1, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
__v11 = pfo_val__;
tcp_ptr__ = __v11;
let __v13: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(vmstat_work) }) as u64);
tcp_ptr__ = __v13;
__v10 = tcp_ptr__ as *mut delayed_work;
__v8 = __v10;
const_test_bit(0u64, ((unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*__v8).work)).data) }) as *mut u64) as *const u64) as i32
} else {
let mut __v14: *mut delayed_work;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
if true {
break;
}
}
let mut __v16: *mut delayed_work;
let mut tcp_ptr__: u64 = 0;
let mut __v17: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t2: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t2, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t2 as u64) };
}
__v17 = pfo_val__;
tcp_ptr__ = __v17;
let __v19: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(vmstat_work) }) as u64);
tcp_ptr__ = __v19;
__v16 = tcp_ptr__ as *mut delayed_work;
__v14 = __v16;
_test_bit(0u64, ((unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*__v14).work)).data) }) as *mut u64) as *const u64) as i32
};
if __v7 == 0 {
return;
}
let mut __v20: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v21: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v23: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t3: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t3 as u8) };
}
__v23 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v23;
}
2 => {
let mut __v25: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t4: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t4, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t4 as u16) };
}
__v25 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v25;
}
4 => {
let mut __v26: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t5: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t5, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t5 as u32) };
}
__v26 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v26;
}
8 => {
let mut __v27: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t6: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t6, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t6 as u64) };
}
__v27 = pfo_val__ as i32;
pscr_ret__ = __v27;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v21 = pscr_ret__;
__v20 = __v21;
if !need_update(__v20) {
return;
}
// Just refresh counters and do not care about the pending delayed
// vmstat_update. It doesn't fire that often to matter and canceling
// it would be too expensive from this path.
// vmstat_shepherd will take care about that for us.
refresh_cpu_vm_stats(false);
}

static mut shepherd: delayed_work = {
let mut __t0: delayed_work = unsafe { core::mem::zeroed() };
__t0.work = work_struct { data: __SlateRecord_types_h_197 { counter: 4503599625273344i64 }, entry: list_head { next: unsafe { core::ptr::addr_of_mut!(shepherd.work.entry) }, prev: unsafe { core::ptr::addr_of_mut!(shepherd.work.entry) } }, func: Some(vmstat_shepherd) };
__t0.timer = {
let mut __t1: timer_list = unsafe { core::mem::zeroed() };
__t1.entry = {
let mut __t2: hlist_node = unsafe { core::mem::zeroed() };
__t2.next = ((unsafe { ((768i32 as *mut ()) as *const u8).wrapping_offset(16045481047390945280u64 as isize) }) as *mut ()) as *mut hlist_node;
__t2
};
__t1.function = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) };
__t1.flags = 2621440u32;
__t1
};
__t0
};

#[unsafe(no_mangle)]
extern "C" fn vmstat_flush_workqueue() {
flush_workqueue!(unsafe { mm_percpu_wq });
}

/// Shepherd worker thread that checks the
/// differentials of processors that have their worker
/// threads for vm statistics updates disabled because of
/// inactivity.
#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_shepherd")]
extern "C" fn vmstat_shepherd(w: *mut work_struct) {
let mut __v2: bool = unsafe { core::mem::zeroed() };
// In kernel users of vmstat counters either require the precise value and
// they are using zone_page_state_snapshot interface or they can live with
// an imprecision as the regular flushing can happen at arbitrary time and
// cumulative error can grow (see calculate_normal_threshold).
//
// From that POV the regular flushing can be postponed for CPUs that have
// been isolated from the kernel interference without critical
// infrastructure ever noticing. Skip regular flushing from vmstat_shepherd
// for all isolated CPUs to avoid interference with the isolated workload.
let mut scope: core::mem::MaybeUninit<__SlateRecord_rcupdate_h_1216> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_rcupdate_h_1216 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rcupdate_h_1216;
let mut dw: *mut delayed_work = unsafe { core::mem::zeroed() };
let mut cpu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(cpu) as *mut i32;
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_rcu_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
cpus_read_lock();
// Check processors whose vmstat worker threads have been disabled
*__slate_slot_5 = 0;
loop {
*__slate_slot_5 = find_next_bit(cpumask_bits!(core::ptr::addr_of_mut!(__cpu_online_mask) as *const cpumask), 64u64, *__slate_slot_5 as u64) as i32;
if (*__slate_slot_5 as u32) < 64 {
dw = per_cpu_ptr__8!(core::ptr::addr_of_mut!(vmstat_work), *__slate_slot_5);
core::ptr::write(__slate_slot_3, class_rcu_constructor());
__slate_cleanup_armed_3.set(true);
if !class_rcu_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3))).is_null() || !class_rcu_is_conditional {
if true {
if !cpu_is_isolated(*__slate_slot_5) {
if work_busy(core::ptr::addr_of_mut!((*dw).work)) == 0 {
__v2 = need_update(*__slate_slot_5);
} else {
__v2 = false;
}
if __v2 {
queue_delayed_work_on(*__slate_slot_5, mm_percpu_wq, dw, 0u64);
}
}
}
}
__slate_cleanup_armed_3.set(false);
class_rcu_destructor(__slate_slot_3 as *mut _);
cond_resched__83!();
let __v1: i32 = *__slate_slot_5;
*__slate_slot_5 = __v1.wrapping_add(1);
} else {
break;
}
}
cpus_read_unlock();
schedule_delayed_work(core::ptr::addr_of_mut!(shepherd), round_jiffies_relative(sysctl_stat_interval as u64));
}
}

#[unsafe(link_section = ".init.text")]
fn start_shepherd_timer() {
let mut __slate_compound_4: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut cpu: i32 = 0;
'__slate_break_17: loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
'__slate_break_13: loop {
'__slate_break_10: loop {
'__slate_break_9: loop {
let mut __v0: *mut delayed_work;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
if true {
break;
}
}
__v0 = ((((unsafe { core::ptr::addr_of_mut!(vmstat_work) }) as u64) as *mut delayed_work) as u64).wrapping_add(unsafe { __per_cpu_offset[cpu as usize] }) as *mut delayed_work;
__init_work(unsafe { core::ptr::addr_of_mut!((*__v0).work) }, 0i32);
let mut __v2: *mut delayed_work;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
if true {
break;
}
}
__v2 = ((((unsafe { core::ptr::addr_of_mut!(vmstat_work) }) as u64) as *mut delayed_work) as u64).wrapping_add(unsafe { __per_cpu_offset[cpu as usize] }) as *mut delayed_work;
unsafe {
(*core::ptr::addr_of_mut!((*__v2).work)).data = *{
__slate_compound_4 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_4)
};
}
let mut __v5: *mut delayed_work;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
if true {
break;
}
}
__v5 = ((((unsafe { core::ptr::addr_of_mut!(vmstat_work) }) as u64) as *mut delayed_work) as u64).wrapping_add(unsafe { __per_cpu_offset[cpu as usize] }) as *mut delayed_work;
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*__v5).work)).entry) });
let mut __v7: *mut delayed_work;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
if true {
break;
}
}
__v7 = ((((unsafe { core::ptr::addr_of_mut!(vmstat_work) }) as u64) as *mut delayed_work) as u64).wrapping_add(unsafe { __per_cpu_offset[cpu as usize] }) as *mut delayed_work;
unsafe {
(*core::ptr::addr_of_mut!((*__v7).work)).func = Some(vmstat_update);
}
if true {
break;
}
}
if true {
break;
}
}
let mut __v11: *mut delayed_work;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<delayed_work>() as *const ();
if true {
break;
}
}
__v11 = ((((unsafe { core::ptr::addr_of_mut!(vmstat_work) }) as u64) as *mut delayed_work) as u64).wrapping_add(unsafe { __per_cpu_offset[cpu as usize] }) as *mut delayed_work;
unsafe { timer_init_key(core::ptr::addr_of_mut!((*__v11).timer), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2621440u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
if true {
break;
}
}
// For secondary CPUs during CPU hotplug scenarios,
// vmstat_cpu_online() will enable the work.
// mm/vmstat:online enables and disables vmstat_work
// symmetrically during CPU hotplug events.
if !cpu_online(cpu as u32) {
unsafe { disable_delayed_work_sync(per_cpu_ptr__8!(core::ptr::addr_of_mut!(vmstat_work), cpu)) };
}
cpu = cpu.wrapping_add(1);
}
schedule_delayed_work(unsafe { core::ptr::addr_of_mut!(shepherd) }, unsafe { round_jiffies_relative(sysctl_stat_interval as u64) });
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn init_cpu_node_state() {
let mut node: i32 = first_node!(node_states[1]) as i32;
while node < 64i32 {
if !cpumask_empty(cpumask_of_node(node)) {
node_set_state(node, 4u32);
}
node = next_node!(node, node_states[1]) as i32;
}
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_cpu_online")]
extern "C" fn vmstat_cpu_online(cpu: u32) -> i32 {
if (unsafe { vmstat_late_init_done }) != 0 {
refresh_zone_stat_thresholds();
}
if node_state(cpu_to_node(cpu as i32), 4u32) == 0 {
node_set_state(cpu_to_node(cpu as i32), 4u32);
}
unsafe { enable_delayed_work(per_cpu_ptr__8!(core::ptr::addr_of_mut!(vmstat_work), cpu)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_cpu_down_prep")]
extern "C" fn vmstat_cpu_down_prep(cpu: u32) -> i32 {
unsafe { disable_delayed_work_sync(per_cpu_ptr__8!(core::ptr::addr_of_mut!(vmstat_work), cpu)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.vmstat_cpu_dead")]
extern "C" fn vmstat_cpu_dead(cpu: u32) -> i32 {
let node: i32 = cpu_to_node(cpu as i32);
refresh_zone_stat_thresholds();
let node_cpus: *const cpumask = cpumask_of_node(node);
if !cpumask_empty(node_cpus) {
return 0;
}
node_clear_state(node, 4u32);
0
}

#[unsafe(link_section = ".init.text")]
extern "C" fn vmstat_late_init() -> i32 {
refresh_zone_stat_thresholds();
unsafe {
vmstat_late_init_done = 1;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vmstat_late_init_691: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(vmstat_late_init)) };

static vmstat_table: __SlateAlign16<[ctl_table; 3]> = __SlateAlign16([{
let mut __t0: ctl_table = unsafe { core::mem::zeroed() };
__t0.procname = (b"stat_interval\0".as_ptr() as *mut u8) as *const u8;
__t0.data = (unsafe { core::ptr::addr_of_mut!(sysctl_stat_interval) }) as *mut ();
__t0.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t0.mode = 420u16;
__t0.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_dointvec_jiffies as *const ()) };
__t0
}, {
let mut __t1: ctl_table = unsafe { core::mem::zeroed() };
__t1.procname = (b"stat_refresh\0".as_ptr() as *mut u8) as *const u8;
__t1.data = core::ptr::null_mut::<()>();
__t1.maxlen = 0i32;
__t1.mode = 384u16;
__t1.proc_handler = Some(vmstat_refresh);
__t1
}, {
let mut __t2: ctl_table = unsafe { core::mem::zeroed() };
__t2.procname = (b"numa_stat\0".as_ptr() as *mut u8) as *const u8;
__t2.data = (unsafe { core::ptr::addr_of_mut!(sysctl_vm_numa_stat) }) as *mut ();
__t2.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t2.mode = 420u16;
__t2.proc_handler = Some(sysctl_vm_numa_stat_handler);
__t2.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(0i32 as isize) as *mut ();
__t2.extra2 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(1i32 as isize) as *mut ();
__t2
}]);

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn init_mm_internals() {
let mut ret: i32 = 0;
let mut __v0: *mut workqueue_struct;
let mut __v1: *mut workqueue_struct;
let mut _res: *mut workqueue_struct = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = unsafe { alloc_workqueue_noprof(c"mm_percpu_wq".as_ptr() as *const u8, 264u32, 0i32) };
alloc_tag_restore!();
} else {
_res = unsafe { alloc_workqueue_noprof(c"mm_percpu_wq".as_ptr() as *const u8, 264u32, 0i32) };
}
__v1 = _res;
__v0 = __v1;
unsafe {
mm_percpu_wq = __v0;
}
ret = cpuhp_setup_state_nocalls(13i32, c"mm/vmstat:dead".as_ptr() as *const u8, None, Some(vmstat_cpu_dead));
if ret < 0 {
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x013vmstat: failed to register 'dead' hotplug state\n".as_ptr() as *const u8) };
}
ret = cpuhp_setup_state_nocalls(191i32, c"mm/vmstat:online".as_ptr() as *const u8, Some(vmstat_cpu_online), Some(vmstat_cpu_down_prep));
if ret < 0 {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013vmstat: failed to register 'online' hotplug state\n".as_ptr() as *const u8) };
}
unsafe { cpus_read_lock() };
init_cpu_node_state();
unsafe { cpus_read_unlock() };
start_shepherd_timer();
proc_create_seq!(c"buddyinfo".as_ptr() as *mut u8, 292i32, unsafe { core::ptr::addr_of!(fragmentation_op.0) });
proc_create_seq!(c"pagetypeinfo".as_ptr() as *mut u8, 256i32, unsafe { core::ptr::addr_of!(pagetypeinfo_op.0) });
proc_create_seq!(c"vmstat".as_ptr() as *mut u8, 292i32, unsafe { core::ptr::addr_of!(vmstat_op.0) });
proc_create_seq!(c"zoneinfo".as_ptr() as *mut u8, 292i32, unsafe { core::ptr::addr_of!(zoneinfo_op.0) });
unsafe { __register_sysctl_init(c"vm".as_ptr() as *const u8, core::ptr::addr_of!(vmstat_table.0) as *const ctl_table, c"vmstat_table".as_ptr() as *const u8, ARRAY_SIZE__16!()) };
}

/// Return an index indicating how much of the available free memory is
/// unusable for an allocation of the requested size.
fn unusable_free_index(order: u32, info: &contig_page_info) -> i32 {
// No free memory is interpreted as all free memory is unusable
if (unsafe { (*(info as *const contig_page_info)).free_pages }) == 0 {
return 1000;
}
// Index should be a value between 0 and 1. Return a value to 3
// decimal places.
//
// 0 => no fragmentation
// 1 => high fragmentation
div_u64(unsafe { (*(info as *const contig_page_info)).free_pages }.wrapping_sub((unsafe { (*(info as *const contig_page_info)).free_blocks_suitable }) << order).wrapping_mul(1000), (unsafe { (*(info as *const contig_page_info)).free_pages }) as u32) as i32
}

#[unsafe(link_section = ".text.slate_distinct.vmstat.unusable_show_print")]
extern "C" fn unusable_show_print(m: *mut seq_file, pgdat: *mut pglist_data, zone: *mut zone) {
let mut order: u32 = 0;
let mut index: i32 = 0;
let mut info: contig_page_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
unsafe { seq_printf(m, c"Node %d, zone %8s ".as_ptr() as *const u8, (*pgdat).node_id, (*zone).name) };
while order < 11u32 {
fill_contig_page_info(zone, order, unsafe { &mut (*core::ptr::addr_of_mut!(info)) });
index = unusable_free_index(order, unsafe { &(*core::ptr::addr_of_mut!(info)) });
unsafe { seq_printf(m, c"%d.%03d ".as_ptr() as *const u8, index / 1000, index % 1000) };
order = order.wrapping_add(1);
}
unsafe { seq_putc(m, 10u8) };
}

/// Display unusable free space index
///
/// The unusable free space index measures how much of the available free
/// memory cannot be used to satisfy an allocation of a given size and is a
/// value between 0 and 1. The higher the value, the more of free memory is
/// unusable and by implication, the worse the external fragmentation is. This
/// can be expressed as a percentage by multiplying by 100.
#[unsafe(link_section = ".text.slate_distinct.vmstat.unusable_show")]
extern "C" fn unusable_show(m: *mut seq_file, arg: *mut ()) -> i32 {
let pgdat: *mut pglist_data = arg as *mut pglist_data;
// check memoryless node
if node_state(unsafe { (*pgdat).node_id }, 3u32) == 0 {
return 0;
}
walk_zones_in_node(m, pgdat, true, false, Some(unusable_show_print));
0
}

static unusable_sops: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(frag_start), stop: Some(frag_stop), next: Some(frag_next), show: Some(unusable_show) });

#[unsafe(link_section = ".text.slate_distinct.vmstat.unusable_open")]
extern "C" fn unusable_open(inode: *mut inode, file: *mut file) -> i32 {
let ret: i32 = unsafe { seq_open(file, core::ptr::addr_of!(unusable_sops.0)) };
if ret == 0 && !unsafe { (*inode).i_private }.is_null() {
let seq_f: *mut seq_file = (unsafe { (*file).private_data }) as *mut seq_file;
unsafe {
(*seq_f).private = (*inode).i_private;
}
}
ret
}

static unusable_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.open = Some(unusable_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(seq_release as *const ()) };
__t0
});

#[unsafe(link_section = ".text.slate_distinct.vmstat.extfrag_show_print")]
extern "C" fn extfrag_show_print(m: *mut seq_file, pgdat: *mut pglist_data, zone: *mut zone) {
let mut order: u32 = 0;
let mut index: i32 = 0;
// Alloc on stack as interrupts are disabled for zone walk
let mut info: contig_page_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
unsafe { seq_printf(m, c"Node %d, zone %8s ".as_ptr() as *const u8, (*pgdat).node_id, (*zone).name) };
while order < 11u32 {
fill_contig_page_info(zone, order, unsafe { &mut (*core::ptr::addr_of_mut!(info)) });
index = __fragmentation_index(order, unsafe { &(*core::ptr::addr_of_mut!(info)) });
unsafe { seq_printf(m, c"%2d.%03d ".as_ptr() as *const u8, index / 1000, index % 1000) };
order = order.wrapping_add(1);
}
unsafe { seq_putc(m, 10u8) };
}

/// Display fragmentation index for orders that allocations would fail for
#[unsafe(link_section = ".text.slate_distinct.vmstat.extfrag_show")]
extern "C" fn extfrag_show(m: *mut seq_file, arg: *mut ()) -> i32 {
let pgdat: *mut pglist_data = arg as *mut pglist_data;
walk_zones_in_node(m, pgdat, true, false, Some(extfrag_show_print));
0
}

static extfrag_sops: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(frag_start), stop: Some(frag_stop), next: Some(frag_next), show: Some(extfrag_show) });

#[unsafe(link_section = ".text.slate_distinct.vmstat.extfrag_open")]
extern "C" fn extfrag_open(inode: *mut inode, file: *mut file) -> i32 {
let ret: i32 = unsafe { seq_open(file, core::ptr::addr_of!(extfrag_sops.0)) };
if ret == 0 && !unsafe { (*inode).i_private }.is_null() {
let seq_f: *mut seq_file = (unsafe { (*file).private_data }) as *mut seq_file;
unsafe {
(*seq_f).private = (*inode).i_private;
}
}
ret
}

static extfrag_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.open = Some(extfrag_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(seq_release as *const ()) };
__t0
});

#[unsafe(link_section = ".init.text")]
extern "C" fn extfrag_debug_init() -> i32 {
let extfrag_debug_root: *mut dentry = unsafe { debugfs_create_dir(c"extfrag".as_ptr() as *const u8, core::ptr::null_mut::<dentry>()) };
debugfs_create_file!(c"unusable_index".as_ptr() as *mut u8, 292i32, extfrag_debug_root, core::ptr::null_mut::<()>(), unsafe { core::ptr::addr_of!(unusable_fops.0) });
debugfs_create_file!(c"extfrag_index".as_ptr() as *mut u8, 292i32, extfrag_debug_root, core::ptr::null_mut::<()>(), unsafe { core::ptr::addr_of!(extfrag_fops.0) });
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_extfrag_debug_init_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(extfrag_debug_init)) };

static __slate_asm_string___fragmentation_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___fragmentation_index_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/mm/vmstat.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
