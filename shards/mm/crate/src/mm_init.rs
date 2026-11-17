
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
//! mm_init.c - Memory initialisation verification and debugging
//!
//! Copyright 2008 IBM Corporation, 2008
//! Author Mel Gorman <mel@csn.ul.ie>
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
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
//! container_of - cast a member of a structure out to the containing structure
//! @ptr:	the pointer to the member.
//! @type:	the type of the container struct this is embedded in.
//! @member:	the name of the member within the struct.
//!
//! WARNING: any const qualifier of @ptr is lost.
//! Do not use container_of() in new code.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! error conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! Operation not permitted
//! Value too large for defined data type
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
//! restart by calling sys_restart_syscall
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
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! __NR_PAGEFLAGS
//! see the comment on MAX_NR_TIERS
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
//! percpu_counter batch for local add or sub
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! Helper macro to avoid gfp flags if they are the default one
//! Keep unconverted code working
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
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
//! DEBUG: Poison objects
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
//! Must be a compile-time constant, so implement it as a macro
//! page can be read
//! stack-like segment
//! See the comment in asm-generic/io.h about ioremap_np().
//! 8237 DMA controllers
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! macro tricks to expand the set_buffer_foo(), clear_buffer_foo()
//! and buffer_foo() functions.
//! To avoid reset buffer flags that are already set, because that causes
//! a costly cache line transition, check the flag first.
//! ***DANGEROUS*** don't add guard page
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
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
//! Internal flags that share the struct mempolicy flags word with
//! "mode flags".  These flags are allocated from bit 0 up, as they
//! are never OR'ed into the mode in mempolicy API arguments.
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Number of chars for args
//! The per-cpu notes area is a list of notes terminated by a "NULL"
//! note header.  For kdump, the code in vmcore.c runs in the context
//! of the second kernel to combine them into one note.
//! The artificial cap on the number of segments passed to kexec_load.
//! For IS_ENABLED(CONFIG_CRASH_DUMP)
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
//! Compare PTEs respecting the dirty bit.
//! slab's alloc_flags definitions
//! Monochrome Text Display
//! number of edd_info structs starting at EDDBUF
//! This is the maximum number of entries in struct boot_params::e820_table
//! (the zeropage), which is part of the x86 boot protocol ABI:
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
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-weak-function-attribute
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-weak-variable-attribute
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
// check_mul_overflow() - Calculate multiplication with overflow checking
// @a: first factor
// @b: second factor
// @d: pointer to store product
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted multiplication, regardless of whether
// wrap-around occurred.
// flex_array_size() - Calculate size of a flexible array member
//                     within an enclosing structure.
// @p: Pointer to the structure.
// @member: Name of the flexible array member.
// @count: Number of elements in the array.
//
// Calculates size of a flexible array of @count number of @member
// elements, at the end of structure @p.
//
// Return: number of bytes needed or SIZE_MAX on overflow.
// struct_size() - Calculate size of structure with trailing flexible array.
// @p: Pointer to the structure.
// @member: Name of the array member.
// @count: Number of elements in the array.
//
// Calculates size of memory needed for structure of @p followed by an
// array of @count number of @member elements.
//
// Return: number of bytes needed or SIZE_MAX on overflow.
// struct_size_t() - Calculate size of structure with trailing flexible array
// @type: structure type name.
// @member: Name of the array member.
// @count: Number of elements in the array.
//
// Calculates size of memory needed for structure @type followed by an
// array of @count number of @member elements. Prefer using struct_size()
// when possible instead, to keep calculations associated with a specific
// instance variable of type @type.
//
// Return: number of bytes needed or SIZE_MAX on overflow.
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
// warning conditions
// debug-level messages
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// N 32-bit words worth of info
// N 32-bit bug flags
// "pse" Page Size Extensions
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "sse"
// "pdpe1gb" GB pages
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// No such file or directory
// No such device or address
// Argument list too long
// Exec format error
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Operation not supported on transport endpoint
// Memory page has hardware error
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// __MAX_NR_ZONES
// 0
// Same as rb_first(), but O(1)
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
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
// Generate SD flag bits
// Resource extended types
// No address assigned yet
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
// Indicate a kmalloc slab
// Use GFP_DMA memory
// DEBUG: Store the last owner for bug hunting
// define SLAB_TYPESAFE_BY_RCU - **WARNING** READ THIS!
//
// This delays freeing the SLAB page by a grace period, it does _NOT_
// delay object freeing. This means that if you do kmem_cache_free()
// that memory location is free to be reused at any time. Thus it may
// be possible to see another object there in the same RCU grace period.
//
// This feature only ensures the memory location backing the object
// stays valid, the trick to using this is relying on an independent
// object validation pass. Something like:
//
// ::
//
//  begin:
//   rcu_read_lock();
//   obj = lockless_lookup(key);
//   if (obj) {
//     if (!try_get_ref(obj)) // might fail for free objects
//       rcu_read_unlock();
//       goto begin;
//
//     if (obj->key != key) { // not the object we expected
//       put_ref(obj);
//       rcu_read_unlock();
//       goto begin;
//     }
//   }
//  rcu_read_unlock();
//
// This is useful if we need to approach a kernel structure obliquely,
// from its address obtained without the usual locking. We can lock
// the structure to stabilize it and check it's still at the given address,
// only if we can be sure that the memory has not been meanwhile reused
// for some other kind of object (which our subsystem's lock might corrupt).
//
// rcu_read_lock before reading the address, then rcu_read_unlock after
// taking the spinlock within the structure expected at that address.
//
// Note that object identity check has to be done *after* acquiring a
// reference, therefore user has to ensure proper ordering for loads.
// Similarly, when initializing objects allocated with SLAB_TYPESAFE_BY_RCU,
// the newly allocated object has to be fully initialized *before* its
// refcount gets initialized and proper ordering for stores is required.
// refcount_{add|inc}_not_zero_acquire() and refcount_set_release() are
// designed with the proper fences required for reference counting objects
// allocated with SLAB_TYPESAFE_BY_RCU.
//
// Note that it is not possible to acquire a lock within a structure
// allocated with SLAB_TYPESAFE_BY_RCU without first acquiring a reference
// as described above.  The reason is that SLAB_TYPESAFE_BY_RCU pages
// are not zeroed before being given to the slab, which means that any
// locks must be initialized after each and every kmem_struct_alloc().
// Alternatively, make the ctor passed to kmem_cache_create() initialize
// the locks at page-allocation time, as is done in __i915_request_ctor(),
// sighand_ctor(), and anon_vma_ctor().  Such a ctor permits readers
// to safely acquire those ctor-initialized locks under rcu_read_lock()
// protection.
//
// Note that SLAB_TYPESAFE_BY_RCU was originally named SLAB_DESTROY_BY_RCU.
// no-op
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// page can be written
// page can be executed
// page may be used for atomic ops
// perform synchronous page faults for the mapping
// pages are locked
// Optimisation macro.  It is equivalent to:
//      (x & bit1) ? bit2 : 0
// but this version is faster.
// ("bit1" and "bit2" must be single bits)
// 8 bit slave DMA, channels 0..3
// 16 bit master DMA, ch 4(=slave input)..7
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// DMA page registers
// test_set_buffer_foo() and test_clear_buffer_foo()
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
// identify shared policies
// Enabling zone reclaim means the page allocator will attempt to fulfill
// the allocation request on the current node by triggering reclaim and
// trying to shrink the current node.
// Fallback allocations on the next candidates in the zonelist are considered
// when reclaim fails to free up enough memory in the current node/zone.
//
// These bit locations are exposed in the vm.zone_reclaim_mode sysctl.
// New bits are OK, but existing bits should not be changed.
// Enable zone reclaim
// Writeout pages during reclaim
// Unmap pages during reclaim
// PT_TRACE_* event enable flags
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
// Convert GFP flags to their corresponding migrate type
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

macro_rules! GENMASK_INPUT_CHECK__6 {
() => {
core::mem::size_of::<__SlateRecord_find_h_407>() as i32
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
macro_rules! GENMASK_TYPE__6 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__6!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
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

macro_rules! GENMASK__6 {
($h:expr, $l:expr) => {
GENMASK_TYPE__6!(u64, $h, $l)
};
}

macro_rules! GENMASK__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__2!(u64, $h, $l)
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
macro_rules! IS_ENABLED {
() => {
__is_defined!()
};
}

/// in mm/page_alloc.c
macro_rules! K {
($x:expr) => {
($x) << 2i32
};
}

macro_rules! MIN_T__2 {
($type:ty, $a:expr, $b:expr) => {
if (($a) as $type) < (($b) as $type) { ($a) as $type } else { ($b) as $type }
};
}

macro_rules! NODE_DATA {
($nid:expr) => {
unsafe { *(core::ptr::addr_of_mut!(node_data) as *mut *mut pglist_data).wrapping_offset(($nid) as isize) }
};
}

macro_rules! PFN_DOWN {
($x:expr) => {
($x) >> 12i32
};
}

macro_rules! PFN_UP {
($x:expr) => {
$x.wrapping_add(_AC!() << 12i32).wrapping_sub(1) >> 12i32
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

macro_rules! VM_BUG_ON_PAGE {
() => {
VM_BUG_ON!();
};
}

macro_rules! VM_BUG_ON {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! ZERO_PAGE {
($vaddr:expr) => {
_zero_page(($vaddr) as u64)
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _AC__2 {
() => {
18446744071562067968u64
};
}

macro_rules! _AC__5 {
() => {
4294967296u64
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

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i32, $val) };
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

macro_rules! __cmp_once__6 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = $y;
__v = if __v2 > __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once {
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

macro_rules! __cmp_once__5 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
__v = if __v2 > __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __const_hweight16 {
($w:expr) => {
__const_hweight8!($w).wrapping_add(__const_hweight8!(($w) >> 8i32))
};
}

macro_rules! __const_hweight32 {
($w:expr) => {
__const_hweight16!($w).wrapping_add(__const_hweight16!(($w) >> 16i32))
};
}

macro_rules! __const_hweight64 {
($w:expr) => {
__const_hweight32!($w).wrapping_add(__const_hweight32!(($w) >> 32i32))
};
}

/// Compile time versions of __arch_hweightN()
macro_rules! __const_hweight8 {
($w:expr) => {
((($w) & 1u64 != 0) as i32).wrapping_add((($w) & 2u64 != 0) as i32).wrapping_add((($w) & 4u64 != 0) as i32).wrapping_add((($w) & 8u64 != 0) as i32).wrapping_add((($w) & 16u64 != 0) as i32).wrapping_add((($w) & 32u64 != 0) as i32).wrapping_add((($w) & 64u64 != 0) as i32).wrapping_add((($w) & 128u64 != 0) as i32) as u32
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
macro_rules! __round_mask {
($y:expr) => {
$y.wrapping_sub(1)
};
}

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask__2 {
($y:expr) => {
$y.wrapping_sub(1) as u64
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
2i32.wrapping_add(__is_nonneg!($ux) as i32)
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

macro_rules! __types_ok3 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use!($ux) & __sign_use!($uy) & __sign_use!($uz)
};
}

macro_rules! __types_ok__4 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy)
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__4!($uy)
};
}

macro_rules! __va {
($x:expr) => {
(($x) as u64).wrapping_add(unsafe { page_offset_base }) as *mut ()
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

macro_rules! _static_cpu_has {
($bit:expr) => {
__static_cpu_has__const1(($bit) as u16) as i32
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! adj_init_size {
($start:expr, $end:expr, $size:expr, $pos:expr, $adj:expr) => {
loop {
if unsafe { core::ptr::addr_of_mut!($start) as *mut u8 }.wrapping_offset(0 as isize) <= unsafe { core::ptr::addr_of_mut!($pos) as *mut u8 }.wrapping_offset(0 as isize) && unsafe { core::ptr::addr_of_mut!($pos) as *mut u8 }.wrapping_offset(0 as isize) < unsafe { core::ptr::addr_of_mut!($end) as *mut u8 }.wrapping_offset(0 as isize) && ($size) > $adj {
let __v: u64 = $size;
let __v2: u64 = __v.wrapping_sub($adj);
$size = __v2;
}
if true {
break;
}
}
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

macro_rules! cond_resched__70 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/mm_init.c".as_ptr() as *const u8, 2206i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__80 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/mm_init.c".as_ptr() as *const u8, 909i32, 0u32);
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
macro_rules! data_race {
($expr:expr) => {
{
let mut __v: u32;
__kcsan_disable_current();
let __v2: u32 = $expr;
__kcsan_enable_current();
__v = __v2;
__v
}
};
}

macro_rules! div64_ul {
($x:expr, $y:expr) => {
div64_u64($x, $y)
};
}

macro_rules! first_node {
($src:expr) => {
__first_node((unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

macro_rules! hweight32 {
($w:expr) => {
{
let __v: u32 = if unsafe { __slate_intrinsic_f217c4720f04f6db($w) } {
__const_hweight32!($w)
} else {
__arch_hweight32(($w) as u32)
};
__v
}
};
}

macro_rules! hweight64 {
($w:expr) => {
{
let __v: u64 = if unsafe { __slate_intrinsic_f217c4720f04f6db($w) } {
__const_hweight64!($w) as u64
} else {
__arch_hweight64($w)
};
__v
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

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
};
}

/// kcsan_check_write - check regular write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 1i32);
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__6!($type, $x, $y)
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__5!($type, $x, $y)
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

/// On some architectures it is expensive to call memset() for small sizes.
/// If an architecture decides to implement their own version of
/// mm_zero_struct_page they should wrap the defines below in a #ifndef and
/// define their own version of this macro in <asm/pgtable.h>
///
/// This function must be updated when the size of struct page grows above 96
/// or reduces below 56. The idea that compiler optimizes out switch()
/// statement, and only leaves move/store instructions. Also the compiler can
/// combine write statements if they are both assignments and can be reordered,
/// this can result in several of the writes here being dropped.
macro_rules! mm_zero_struct_page {
($pp:expr) => {
__mm_zero_struct_page($pp);
};
}

macro_rules! next_node {
($n:expr, $src:expr) => {
__next_node($n, (unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

macro_rules! node_present_pages {
($nid:expr) => {
unsafe { (*NODE_DATA!($nid)).node_present_pages }
};
}

macro_rules! nodes_weight {
($nodemask:expr) => {
__nodes_weight((unsafe { core::ptr::addr_of_mut!($nodemask) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

/// Definition of global_zone_page_state not available yet
macro_rules! nr_free_pages {
() => {
global_zone_page_state(0u32)
};
}

macro_rules! num_present_cpus {
() => {
cpumask_weight((unsafe { core::ptr::addr_of_mut!(__cpu_present_mask) }) as *const cpumask)
};
}

macro_rules! page_ref_tracepoint_active {
() => {
0i32
};
}

macro_rules! pageblock_aligned {
($pfn:expr) => {
IS_ALIGNED!($pfn, 1u64 << MIN_T__2!(u32, 9i32, 10i32))
};
}

macro_rules! pageblock_end_pfn__2 {
($pfn:expr) => {
ALIGN!($pfn.wrapping_add(1), 1u64 << if 9u32 < 10 { 9u32 } else { 10u32 })
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

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up__2 {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask__2!($y)).wrapping_add(1)
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

/// roundup_pow_of_two - round the given value up to nearest power of two
/// @n: parameter
///
/// round the given value up to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! roundup_pow_of_two__2 {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { if ($n) == 1 { 1u64 } else { 1u64 << ilog2!($n.wrapping_sub(1)).wrapping_add(1) } } else { __roundup_pow_of_two($n) }
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

macro_rules! static_branch_maybe {
($x:expr) => {
static_branch_unlikely__2!($x)
};
}

macro_rules! static_branch_maybe__3 {
($x:expr) => {
static_branch_unlikely!($x)
};
}

macro_rules! static_branch_unlikely {
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

macro_rules! test_bit__3 {
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

/// virt_to_page(kaddr) returns a valid pointer if and only if
/// virt_addr_valid(kaddr) returns true.
macro_rules! virt_to_page {
($kaddr:expr) => {
((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__pa!($kaddr) >> 12i32) as isize)
};
}

/// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
macro_rules! zone_idx {
($zone:expr) => {
(unsafe { $zone.offset_from(((*(*$zone).zone_pgdat).node_zones.as_mut_ptr() as *mut zone) as *mut zone) }) as i64
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut nr_node_ids: u32;
    static mut node_states: [__SlateRecord_nodemask_types_h_17; 6];
    static mut node_data: [*mut pglist_data; 0];
    static mut __cpu_present_mask: cpumask;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut boot_cpu_data: cpuinfo_x86;
    static mut cpuinfo_op: seq_operations;
    static mut __SCK__cond_resched: static_call_key;
    static mut movable_zone: i32;
    static mut mem_section: *mut *mut mem_section;
    static mut __highest_present_section_nr: u64;
    static mut kernel_kobj: *mut kobject;
    static mut sighand_cachep: *mut kmem_cache;
    static mut _totalram_pages: __SlateRecord_types_h_197;
    static mut vm_zone_stat: [__SlateRecord_types_h_197; 10];
    static mut sysctl_overcommit_memory: i32;
    static mut memblock: memblock;
    static mut _text: [u8; 0];
    static mut _stext: [u8; 0];
    static mut _etext: [u8; 0];
    static mut _data: [u8; 0];
    static mut _sdata: [u8; 0];
    static mut _edata: [u8; 0];
    static mut __bss_start: [u8; 0];
    static mut __bss_stop: [u8; 0];
    static mut __init_begin: [u8; 0];
    static mut __init_end: [u8; 0];
    static mut _sinittext: [u8; 0];
    static mut _einittext: [u8; 0];
    static mut __start_rodata: [u8; 0];
    static mut __end_rodata: [u8; 0];
    static mut kstat: kernel_stat;
    static mut totalcma_pages: u64;
    static mut elfcorehdr_addr: u64;
    static mut highest_memmap_pfn: u64;
    static mut zone_names: [*mut u8; 4];
    static mut kmem_cache: *mut kmem_cache;
    fn panic(arg0: *const u8, ...);
    fn pgtable_cache_init();
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn kstrtobool(arg0: *const u8, arg1: *mut bool) -> i32;
    fn simple_strtoull(arg0: *const u8, arg1: *mut *mut u8, arg2: u32) -> u64;
    fn __cond_resched() -> i32;
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_34();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn _find_first_bit(arg0: *const u64, arg1: u64) -> u64;
    fn _find_last_bit(arg0: *const u64, arg1: u64) -> u64;
    fn static_key_enable(arg0: *mut static_key);
    fn static_key_disable(arg0: *mut static_key);
    fn ____wrong_branch_error() -> bool;
    fn get_option(arg0: *mut *mut u8, arg1: *mut i32) -> i32;
    fn memparse(arg0: *const u8, arg1: *mut *mut u8) -> u64;
    fn parse_option_str(arg0: *const u8, arg1: *const u8) -> bool;
    fn __bitmap_weight(arg0: *const u64, arg1: u32) -> u32;
    fn __compiletime_assert_48();
    fn __compiletime_assert_51();
    fn alloc_offline_node_data(arg0: i32);
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn __SCT__cond_resched() -> i32;
    fn rcu_is_watching() -> bool;
    fn __init_waitqueue_head(arg0: *mut wait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn get_pfnblock_migratetype(arg0: *const page, arg1: u64) -> u32;
    fn __compiletime_assert_192();
    fn __compiletime_assert_193();
    fn build_all_zonelists(arg0: *mut pglist_data);
    fn lruvec_init(arg0: *mut lruvec);
    fn first_online_pgdat() -> *mut pglist_data;
    fn next_zone(arg0: *mut zone) -> *mut zone;
    fn __next_zones_zonelist(arg0: *mut zoneref, arg1: u32, arg2: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref;
    fn __compiletime_assert_202();
    fn alloc_pages_exact_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn kobject_create_and_add(arg0: *const u8, arg1: *mut kobject) -> *mut kobject;
    fn pti_init();
    fn mm_cache_init();
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn __compiletime_assert_468();
    fn __compiletime_assert_469();
    fn __compiletime_assert_470();
    fn arch_zone_limits_init(arg0: *mut u64);
    fn memblock_discard();
    fn __next_mem_range(arg0: *mut u64, arg1: i32, arg2: u32, arg3: *mut memblock_type, arg4: *mut memblock_type, arg5: *mut u64, arg6: *mut u64, arg7: *mut i32);
    fn memblock_search_pfn_nid(arg0: u64, arg1: *mut u64, arg2: *mut u64) -> i32;
    fn __next_mem_pfn_range(arg0: *mut i32, arg1: i32, arg2: *mut u64, arg3: *mut u64, arg4: *mut i32);
    fn memblock_alloc_exact_nid_raw(arg0: u64, arg1: u64, arg2: u64, arg3: u64, arg4: i32) -> *mut ();
    fn memblock_alloc_try_nid_raw(arg0: u64, arg1: u64, arg2: u64, arg3: u64, arg4: i32) -> *mut ();
    fn memblock_alloc_try_nid(arg0: u64, arg1: u64, arg2: u64, arg3: u64, arg4: i32) -> *mut ();
    fn memblock_start_of_DRAM() -> u64;
    fn memblock_end_of_DRAM() -> u64;
    fn init_pageblock_migratetype(arg0: *mut page, arg1: u32, arg2: bool);
    fn buffer_init();
    fn vmalloc_huge_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn find_vm_area(arg0: *const ()) -> *mut vm_struct;
    fn stack_depot_early_init() -> i32;
    fn execmem_init();
    fn hugetlb_bootmem_struct_page_init();
    fn hugetlb_bootmem_alloc();
    fn memblock_has_mirror() -> bool;
    fn memblock_free_all();
    fn __pageblock_pfn_to_page(arg0: u64, arg1: u64, arg2: *mut zone) -> *mut page;
    fn __free_pages_core(arg0: *mut page, arg1: u32, arg2: u32);
    fn zone_pcp_init(arg0: *mut zone);
    fn page_alloc_init_cpuhp();
    fn page_alloc_sysctl_init();
    fn sparse_init();
    fn reciprocal_value(arg0: u32) -> reciprocal_value;
    fn kmem_cache_init();
    fn vmalloc_init();
    fn init_espfix_bsp();
    fn __compiletime_assert_673();
    fn __compiletime_assert_676();
    fn __compiletime_assert_679();
    fn __builtin_unreachable();
    fn __compiletime_assert_702();
    fn __compiletime_assert_705();
    fn __compiletime_assert_708();
    fn __compiletime_assert_711();
    fn __compiletime_assert_714();
    fn __compiletime_assert_717();
    fn __compiletime_assert_720();
    fn __compiletime_assert_723();
    fn __compiletime_assert_726();
    fn __compiletime_assert_730();
    fn __compiletime_assert_731();
    fn __compiletime_assert_735();
    fn __compiletime_assert_736();
    fn __compiletime_assert_739();
    fn __compiletime_assert_743();
    fn __compiletime_assert_744();
    fn __compiletime_assert_748();
    fn __compiletime_assert_749();
    fn __compiletime_assert_753();
    fn __compiletime_assert_754();
    fn __compiletime_assert_758();
    fn __compiletime_assert_759();
    fn __compiletime_assert_762();
    fn __compiletime_assert_765();
    fn __compiletime_assert_768();
    fn __compiletime_assert_771();
    fn __compiletime_assert_776();
    fn __compiletime_assert_779();
    fn __compiletime_assert_782();
    fn __compiletime_assert_785();
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

fn __kcsan_enable_current() {
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


fn __kcsan_disable_current() {
}

// kcsan_check_read - check regular read access for races
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
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_key {
    func: *mut (),
    __slate_anon_1: __SlateRecord_static_call_types_h_65,
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
// NOTE: __setup functions return values:
// @fn returns 1 (or non-zero) if the option argument is "handled"
// and returns 0 if the option argument is "not handled".
// NOTE: @fn is as per module_param, not __setup!
// I.e., @fn returns 0 for no error or non-zero for error
// (possibly @fn returns a -errno value, but it does not matter).
// Emits warning if @fn returns non-zero.
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
}

// const_*() definitions provide good compile-time optimizations when
// the passed arguments can be resolved at compile time.
#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct __SlateRecord_find_h_67 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_207 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_407 {
}


#[inline(always)]
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
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
fn __arch_hweight32(mut w: u32) -> u32 {
let mut res: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight32\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntl %edi, %eax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u32, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
res
}

#[inline(always)]
fn __arch_hweight64(mut w: u64) -> u64 {
let mut res: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight64\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntq %rdi, %rax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u64, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
res
}

#[inline(always)]
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
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
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
#[inline(always)]
fn hweight_long(w: u64) -> u64 {
let __v0: u64 = if (core::mem::size_of::<u64>() as u64) == 4 {
hweight32!(w) as u64
} else {
hweight64!(w)
};
__v0
}

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
fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_5() };
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
unsafe { __compiletime_assert_6() };
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
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
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
unsafe { __compiletime_assert_34() };
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
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__3!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_set(v: *mut __SlateRecord_types_h_197, i: i64) {
__WRITE_ONCE__6!((*v).counter, i);
}

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
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
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
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
    __slate_bits_0: __slate_bits::__SlateBits171U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_119 {
    __slate_bits_0: __slate_bits::__SlateBits172U0,
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
fn raw_atomic_long_set(v: *mut __SlateRecord_types_h_197, i: i64) {
raw_atomic64_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[inline(always)]
fn atomic_long_set(v: *mut __SlateRecord_types_h_197, i: i64) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
fn find_last_bit(addr: *const u64, size: u64) -> u64 {
if small_const_nbits__2!(size) {
let val: u64 = (unsafe { *addr }) & GENMASK__6!(size.wrapping_sub(1), 0i32);
return if val != 0 { __fls(val) } else { size };
}
unsafe { _find_last_bit(addr, size) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
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
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym init_on_alloc, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym init_on_free, const 0 as u8, label {
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
struct futex_private_hash {
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


#[inline(always)]
fn bitmap_weight(src: *const u64, nbits: u32) -> u32 {
if small_const_nbits!(nbits) {
return hweight_long((unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits)) as u32;
}
unsafe { __bitmap_weight(src, nbits) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


// The inline keyword gives the compiler room to decide to inline, or
// not inline a function as it sees best.  However, as these functions
// are called in both __init and non-__init functions, if they are not
// inlined we will end up with a section mismatch error (of the type of
// freeable items not being freed).  So we must use __always_inline here
// to fix the problem.  If other functions in the future also end up in
// this situation they will also need to be annotated as __always_inline
#[inline(always)]
fn __node_set(node: i32, dstp: *mut __SlateRecord_nodemask_types_h_17) {
set_bit(node as i64, unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 });
}

// No static inline type checking - see Subtlety (1) above.
#[inline(always)]
fn __nodes_weight(srcp: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> i32 {
bitmap_weight(unsafe { (*srcp).bits.as_ptr() as *const u64 }, nbits) as i32
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
fn node_set_state(node: i32, state: u32) {
__node_set(node, unsafe { core::ptr::addr_of_mut!(node_states) as *mut __SlateRecord_nodemask_types_h_17 }.wrapping_offset((state as u32) as isize) as *mut __SlateRecord_nodemask_types_h_17);
}

#[inline(always)]
fn num_node_state(state: u32) -> i32 {
nodes_weight!(node_states[(state as u32) as usize])
}

// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_weight(srcp: *const cpumask) -> u32 {
bitmap_weight(cpumask_bits!(srcp), 64u32)
}

// cpumask_of - the cpumask containing just a given cpu
// @cpu: the cpu (<= nr_cpu_ids)
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
struct seccomp_filter {
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

fn div64_u64(dividend: u64, divisor: u64) -> u64 {
dividend / divisor
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


#[repr(C)]
#[derive(Clone, Copy)]
struct restart_block {
    arch_data: u64,
    r#fn: Option<unsafe extern "C" fn(*mut restart_block) -> i64>,
    __slate_anon_2: __SlateRecord_restart_block_h_29,
}


#[inline(always)]
fn __phys_addr_nodebug(mut x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__2!());
// use the carry flag to determine if x was < __START_KERNEL_map
x = y.wrapping_add(if x > y { unsafe { phys_base } } else { _AC__2!().wrapping_sub(unsafe { page_offset_base }) });
x
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
struct range {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
}


#[inline(always)]
fn get_order(mut size: u64) -> i32 {
if unsafe { __slate_intrinsic_f217c4720f04f6db(size) } {
if size == 0 {
return 52i32;
}
if size < 4096u64 {
return 0;
}
return ilog2!(size.wrapping_sub(1)).wrapping_sub(12).wrapping_add(1);
}
size = size.wrapping_sub(1);
size >>= 12i32;
fls64(size)
}

#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x0_0 {
    __slate_bits_0: __slate_bits::__SlateBits143U0,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x1_0 {
    __slate_bits_0: __slate_bits::__SlateBits144U0,
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
// PGDIR_SHIFT determines what a top-level page table entry can map
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
    __slate_bits_0: __slate_bits::__SlateBits165U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits166U0,
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
    __slate_bits_0: __slate_bits::__SlateBits142U0,
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
    __slate_bits_0: __slate_bits::__SlateBits167U0,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_ranges {
    cs_ranges: [futex_unlock_cs_range; 2],
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
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
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
struct sysv_sem {
    undo_list: *mut sem_undo_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
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
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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
struct gendisk {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits305U0,
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
    __slate_bits_0: __slate_bits::__SlateBits69U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits69U1,
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
    __slate_bits_2: __slate_bits::__SlateBits69U2,
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
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
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
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_105: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct memory_tier {
}


// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
}

fn debug_objects_mem_init() {
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
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
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
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits432U0,
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
struct wake_irq {
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
// page_folio - Converts from page to folio.
// @p: The page.
//
// Every page is part of a folio.  This function cannot be called on a
// NULL pointer.
//
// Context: No reference, nor lock is required on @page.  If the caller
// does not hold a reference, this call may race with a folio split, so
// it should re-check the folio still contains this page after gaining
// a reference on the folio.
// Return: The folio which contains this page.
// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
#[inline(always)]
fn PageCompound(page: *const page) -> i32 {
let __v0: bool = if test_bit__3!(6i32, unsafe { core::ptr::addr_of!((*page).flags.f) }) != 0 {
true
} else {
let mut __v2: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_192() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*page).__slate_anon_1.__slate_anon_2.compound_info) as *const u64) };
__v2 & 1 != 0
};
__v0 as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


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

fn page_init_poison(page: *mut page, size: u64) {
}

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
#[inline(always)]
fn __SetPageReserved(page: *mut page) {
let __v0: bool = if true && false {
let mut __v1: *mut page;
let mut __v2: *mut page;
__v2 = page;
__v1 = __v2;
((unsafe { core::ptr::addr_of_mut!((*__v1).flags.f) }) as u64) != 0
} else {
false
};
if __v0 && false {
let mut __v3: *mut page;
let mut __v4: *mut page;
__v4 = page;
__v3 = __v4;
generic___set_bit(13u64, (unsafe { core::ptr::addr_of_mut!((*__v3).flags.f) }) as *mut u64);
} else {
let mut __v5: *mut page;
let mut __v6: *mut page;
__v6 = page;
__v5 = __v6;
___set_bit(13u64, (unsafe { core::ptr::addr_of_mut!((*__v5).flags.f) }) as *mut u64);
}
}

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
#[inline(always)]
fn PageOffline(page: *const page) -> i32 {
(data_race!((unsafe { (*page).__slate_anon_2.page_type }) >> 24i32) == 241) as i32
}

#[inline(always)]
fn __SetPageOffline(page: *mut page) {
if PageOffline(page as *const page) != 0 {
return;
}
VM_BUG_ON_PAGE!();
unsafe {
(*page).__slate_anon_2.page_type = 4043309056u32;
}
}

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


// We do arithmetic on the LRU lists in various places in the code,
// so it is important to keep the active lists LRU_ACTIVE higher in
// the array than the corresponding inactive lists, and to keep
// the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//
// This has to be kept in sync with the statistics in zone_stat_item
// above and the descriptions in vmstat_text in mm/vmstat.c
fn lru_gen_init_pgdat(pgdat: *mut pglist_data) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct lru_cost {
    count: u64,
    last_rotated: u64,
    last_io: u64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


fn zone_end_pfn(zone: *const zone) -> u64 {
unsafe { (*zone).zone_start_pfn }.wrapping_add(unsafe { (*zone).spanned_pages })
}

fn zone_is_empty(zone: *const zone) -> bool {
(unsafe { (*zone).spanned_pages }) == 0
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


// Define the bit shifts to access each section.  For non-existent
// sections we define the shift as 0; that plus a 0 mask ensures
// the compiler will optimise away reference to them.
fn zone_intersects(zone: *const zone, start_pfn: u64, nr_pages: u64) -> bool {
if zone_is_empty(zone) {
return false;
}
if start_pfn >= zone_end_pfn(zone) || start_pfn.wrapping_add(nr_pages) <= unsafe { (*zone).zone_start_pfn } {
return false;
}
true
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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


fn zone_seqlock_init(zone: *mut zone) {
}

fn movable_node_is_enabled() -> bool {
false
}

fn pgdat_kswapd_lock_init(pgdat: *mut pglist_data) {
}

fn pgdat_resize_init(pgdat: *mut pglist_data) {
}

fn populated_zone(zone: *const zone) -> bool {
(unsafe { (*zone).present_pages }) != 0
}

fn zone_to_nid(zone: *const zone) -> i32 {
unsafe { (*zone).node }
}

fn zone_set_nid(zone: *mut zone, nid: i32) {
unsafe {
(*zone).node = nid;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


// for_each_zone - helper macro to iterate over all memory zones
// @zone: pointer to struct zone variable
//
// The user only needs to declare the zone variable, for_each_zone
// fills it in.
// do nothing
fn zonelist_zone(zoneref: *mut zoneref) -> *mut zone {
unsafe { (*zoneref).zone }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


fn zonelist_zone_idx(zoneref: *const zoneref) -> i32 {
unsafe { (*zoneref).zone_idx }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[inline(always)]
fn next_zones_zonelist(z: *mut zoneref, highest_zoneidx: u32, nodes: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref {
let __v0: bool = nodes.is_null() && (zonelist_zone_idx(z as *const zoneref) as u32) <= (highest_zoneidx as u32);
if __v0 {
return z;
}
unsafe { __next_zones_zonelist(z, highest_zoneidx, nodes) }
}

fn first_zones_zonelist(zonelist: *mut zonelist, highest_zoneidx: u32, nodes: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref {
next_zones_zonelist(unsafe { (*zonelist)._zonerefs.as_mut_ptr() as *mut zoneref }, highest_zoneidx, nodes)
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


// for_each_zone_zonelist_nodemask - helper macro to iterate over valid zones in a zonelist at or below a given zone index and within a nodemask
// @zone: The current zone in the iterator
// @z: The current pointer within zonelist->_zonerefs being iterated
// @zlist: The zonelist being iterated
// @highidx: The zone index of the highest zone to return
// @nodemask: Nodemask allowed by the allocator
//
// This iterator iterates though all zones at or below a given zone index and
// within a given nodemask
// for_each_zone_zonelist - helper macro to iterate over valid zones in a zonelist at or below a given zone index
// @zone: The current zone in the iterator
// @z: The current pointer within zonelist->zones being iterated
// @zlist: The zonelist being iterated
// @highidx: The zone index of the highest zone to return
//
// This iterator iterates though all zones at or below a given zone index.
fn pfn_to_section_nr(pfn: u64) -> u64 {
pfn >> 15i32
}

fn section_nr_to_pfn(sec: u64) -> u64 {
sec << 15i32
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
struct mem_section_usage {
    rcu: callback_head,
    subsection_map: [u64; 1],
    pageblock_flags: [u64; 0],
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
struct mem_section {
    section_mem_map: u64,
    usage: *mut mem_section_usage,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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

fn early_section(section: *const mem_section) -> i32 {
(!section.is_null() && (unsafe { (*section).section_mem_map }) & BIT!(3i32) != 0) as i32
}

fn __pfn_to_section(pfn: u64) -> *mut mem_section {
__nr_to_section(pfn_to_section_nr(pfn))
}

fn subsection_map_index(pfn: u64) -> i32 {
((pfn & 32767u64) / 512u64) as i32
}

fn pfn_section_first_valid(ms: *mut mem_section, pfn: &mut u64) -> bool {
let mut __v0: *mut mem_section_usage;
loop {
if !(__native_word__19!() || ((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_202() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ms).usage) as *const *mut mem_section_usage) };
let usage: *mut mem_section_usage = __v0;
let idx: i32 = subsection_map_index(unsafe { *(pfn as *mut u64) });
let mut bit: u64 = 0;
if usage.is_null() {
return false;
}
if test_bit__2!(idx, unsafe { (*usage).subsection_map.as_mut_ptr() as *mut u64 }) != 0 {
return true;
}
// Find the next subsection that exists
bit = find_next_bit((unsafe { (*usage).subsection_map.as_mut_ptr() as *mut u64 }) as *const u64, 64u64, idx as u64);
if bit == 64u64 {
return false;
}
unsafe {
*(pfn as *mut u64) = (*(pfn as *mut u64) & 18446744073709518848u64).wrapping_add(bit.wrapping_mul(512u64));
}
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
}


fn first_valid_pfn(mut pfn: u64, end_pfn: u64) -> u64 {
let mut nr: u64 = pfn_to_section_nr(pfn);
rcu_read_lock_sched();
while nr <= unsafe { __highest_present_section_nr } && pfn < end_pfn {
let ms: *mut mem_section = __pfn_to_section(pfn);
let __v0: bool = valid_section(ms as *const mem_section) != 0 && (early_section(ms as *const mem_section) != 0 || pfn_section_first_valid(ms, unsafe { &mut (*core::ptr::addr_of_mut!(pfn)) }));
if __v0 {
rcu_read_unlock_sched();
return pfn;
}
// Nothing left in this section? Skip to next section
nr = nr.wrapping_add(1);
pfn = section_nr_to_pfn(nr);
}
rcu_read_unlock_sched();
end_pfn
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


fn next_valid_pfn(mut pfn: u64, end_pfn: u64) -> u64 {
pfn = pfn.wrapping_add(1);
if pfn >= end_pfn {
return end_pfn;
}
// Either every PFN within the section (or subsection for VMEMMAP) is
// valid, or none of them are. So there's no point repeating the check
// for every PFN; only call first_valid_pfn() again when crossing a
// (sub)section boundary (i.e. !(pfn & ~PAGE_{SUB,}SECTION_MASK)).
if pfn & 511u64 != 0 {
return pfn;
}
first_valid_pfn(pfn, end_pfn)
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
    __slate_bits_0: __slate_bits::__SlateBits507U0,
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
    __slate_bits_0: __slate_bits::__SlateBits509U0,
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


#[unsafe(no_mangle)]
static mut mm_kobj: *mut kobject = unsafe { core::mem::zeroed() };

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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits534U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits534U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits534U2,
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
    __slate_bits_3: __slate_bits::__SlateBits534U3,
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
struct request_queue {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct posix_acl {
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
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
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


fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
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
struct task_cputime_atomic {
    utime: __SlateRecord_types_h_197,
    stime: __SlateRecord_types_h_197,
    sum_exec_runtime: __SlateRecord_types_h_197,
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
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
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
    __slate_bits_0: __slate_bits::__SlateBits308U0,
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
union __SlateRecord_vmalloc_h_53 {
    next: *mut vm_struct,
    llnode: llist_node,
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
struct cgroup_taskset {
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
struct swap_cluster_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qstr {
    __slate_anon_0: __SlateRecord_dcache_h_50,
    name: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mempolicy_h_54 {
    cpuset_mems_allowed: __SlateRecord_nodemask_types_h_17,
    user_nodemask: __SlateRecord_nodemask_types_h_17,
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
struct node_barn {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct slub_percpu_sheaves {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(C, align(4096))]
struct __SlateAlign4096<T>(T);

unsafe impl<T> Sync for __SlateAlign4096<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits647U0 {
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
    pub struct __SlateBits678U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits305U0 {
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
    pub struct __SlateBits69U0 {
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
    pub struct __SlateBits69U1 {
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
    pub struct __SlateBits507U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits820U0 {
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
    pub struct __SlateBits308U0 {
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
    pub struct __SlateBits69U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits165U0 {
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
    pub struct __SlateBits167U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits534U0 {
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
    pub struct __SlateBits531U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits534U1 {
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
    pub struct __SlateBits534U2 {
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
    pub struct __SlateBits534U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits509U0 {
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
    pub struct __SlateBits166U0 {
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
    pub struct __SlateBits432U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits171U0 {
        #[bits(24)]
        pub intel_native_model_id: u32,
        #[bits(8)]
        pub intel_type: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits172U0 {
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
    pub struct __SlateBits143U0 {
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
    pub struct __SlateBits144U0 {
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
    pub struct __SlateBits142U0 {
        #[bits(1)]
        pub initialized: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits678U0,
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


// Contains huge pages
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


// This is a #define so the string doesn't get put in every .o file
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
    __slate_bits_0: __slate_bits::__SlateBits531U0,
}


fn hotplug_memory_notifier(r#fn: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>, pri: i32) -> i32 {
0
}

fn clear_page_tag_ref(page: *mut page) {
}

fn alloc_tag_sec_init() {
}

fn page_ext_init() {
}

fn page_ext_init_flatmem_late() {
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_high_memory: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_zero_page_pfn: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_empty_zero_page: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___zero_page: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_mm_init__692_237_mm_compute_batch_init6:\t\t\t\n.long\t{4} - .\t\n.previous\t\t\t\t\t\n\n.section\t\".initcall2.init\", \"a\"\t\t\n__initcall__kmod_mm_init__694_249_mm_sysfs_init2:\t\t\t\n.long\t{5} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_init_on_alloc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_init_on_free: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous", sym high_memory, sym zero_page_pfn, sym empty_zero_page, sym __zero_page, sym mm_compute_batch_init, sym mm_sysfs_init, sym init_on_alloc, sym init_on_free, options(att_syntax));

fn page_ext_init_flatmem() {
}

fn __page_ref_set(page: *mut page, v: i32) {
}

fn set_page_count(page: *mut page, v: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }, v);
if page_ref_tracepoint_active!() != 0 {
__page_ref_set(page, v);
}
}

fn init_page_count(page: *mut page) {
set_page_count(page, 1i32);
}

// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
#[unsafe(no_mangle)]
static empty_zero_page: __SlateAlign4096<[u8; 4096]> = __SlateAlign4096(unsafe { core::mem::zeroed() });

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..ro_after_init")]
static mut __zero_page: *mut page = unsafe { core::mem::zeroed() };

fn _zero_page(addr: u64) -> *mut page {
unsafe { __zero_page }
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


#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn mm_core_init() {
arch_mm_preinit();
init_zero_page_pfn();
// Initializations relying on SMP setup
while false {
}
unsafe { build_all_zonelists(core::ptr::null_mut::<pglist_data>()) };
unsafe { page_alloc_init_cpuhp() };
alloc_tag_sec_init();
// page_ext requires contiguous pages,
// bigger than MAX_PAGE_ORDER unless SPARSEMEM.
page_ext_init_flatmem();
mem_debugging_and_hardening_init();
kfence_alloc_pool_and_metadata();
report_meminit();
kmsan_init_shadow();
unsafe { stack_depot_early_init() };
// KHO memory setup must happen while memblock is still active, but
// as close as possible to buddy initialization
kho_memory_init();
unsafe { memblock_free_all() };
mem_init();
unsafe { kmem_cache_init() };
// page_owner must be initialized after buddy is ready, and also after
// slab is ready so that stack_depot_init() works properly
page_ext_init_flatmem_late();
kmemleak_init();
ptlock_cache_init();
unsafe { pgtable_cache_init() };
debug_objects_mem_init();
unsafe { vmalloc_init() };
// If no deferred init page_ext now, as vmap is fully initialized
if !unsafe { deferred_struct_pages } {
page_ext_init();
}
// Should be run before the first non-init thread is created
unsafe { init_espfix_bsp() };
// Should be run after espfix64 is set up.
unsafe { pti_init() };
kmsan_init_runtime();
unsafe { mm_cache_init() };
unsafe { execmem_init() };
}

fn totalram_pages() -> u64 {
atomic_long_read((unsafe { core::ptr::addr_of_mut!(_totalram_pages) }) as *const __SlateRecord_types_h_197) as u64
}

#[unsafe(no_mangle)]
static mut high_memory: *mut () = unsafe { core::mem::zeroed() };

fn __mm_zero_struct_page(page: *mut page) {
let _pp: *mut u64 = page as *mut u64;
// Check that struct page is either 56, 64, 72, 80, 88 or 96 bytes
loop {
if ((const { core::mem::size_of::<page>() }) as u64) & 7 != 0 {
unsafe { __compiletime_assert_468() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<page>() }) as u64) < 56 {
unsafe { __compiletime_assert_469() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<page>() }) as u64) > 96 {
unsafe { __compiletime_assert_470() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __t0: i64 = match (const { core::mem::size_of::<page>() }) as u64 { 96 => 0, 88 => 1, 80 => 2, 72 => 3, 64 => 4, 56 => 5, _ => -1 };
loop {
match __t0 {
0 => {
unsafe {
*_pp.wrapping_offset(11 as isize) = 0;
}
__t0 = 1;
continue;
}
1 => {
unsafe {
*_pp.wrapping_offset(10 as isize) = 0;
}
__t0 = 2;
continue;
}
2 => {
unsafe {
*_pp.wrapping_offset(9 as isize) = 0;
}
__t0 = 3;
continue;
}
3 => {
unsafe {
*_pp.wrapping_offset(8 as isize) = 0;
}
__t0 = 4;
continue;
}
4 => {
unsafe {
*_pp.wrapping_offset(7 as isize) = 0;
}
__t0 = 5;
continue;
}
5 => {
unsafe {
*_pp.wrapping_offset(6 as isize) = 0;
}
unsafe {
*_pp.wrapping_offset(5 as isize) = 0;
}
unsafe {
*_pp.wrapping_offset(4 as isize) = 0;
}
unsafe {
*_pp.wrapping_offset(3 as isize) = 0;
}
unsafe {
*_pp.wrapping_offset(2 as isize) = 0;
}
unsafe {
*_pp.wrapping_offset(1 as isize) = 0;
}
unsafe {
*_pp.wrapping_offset(0 as isize) = 0;
}
break;
}
_ => {
break;
}
}
}
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
fn page_cpupid_reset_last(page: *mut page) {
}

fn page_kasan_tag_reset(page: *mut page) {
}

fn set_page_zone(page: *mut page, zone: u32) {
let __v2: u64 = (unsafe { (*page).flags.f }) & !(3u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64));
unsafe {
(*page).flags.f = __v2;
}
unsafe {
(*page).flags.f = (*page).flags.f | (((zone as u32) as u64) & 3u64) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64);
}
}

fn set_page_node(page: *mut page, node: u64) {
let __v2: u64 = (unsafe { (*page).flags.f }) & !(63u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64));
unsafe {
(*page).flags.f = __v2;
}
unsafe {
(*page).flags.f = (*page).flags.f | (node & 63u64) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64);
}
}

fn set_page_links(page: *mut page, zone: u32, node: u64, pfn: u64) {
set_page_zone(page, zone);
set_page_node(page, node);
}

fn global_zone_page_state(item: u32) -> u64 {
let mut x: i64 = atomic_long_read(unsafe { core::ptr::addr_of_mut!(vm_zone_stat) as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197);
if x < 0 {
x = 0;
}
x as u64
}

fn ptlock_cache_init() {
}

fn get_num_physpages() -> u64 {
let mut nid: i32 = first_node!(node_states[1]) as i32;
let mut phys_pages: u64 = 0;
while nid < 64i32 {
let __v0: u64 = phys_pages;
phys_pages = __v0.wrapping_add(node_present_pages!(nid));
nid = next_node!(nid, node_states[1]) as i32;
}
phys_pages
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn get_pfn_range_for_nid(nid: u32, start_pfn: *mut u64, end_pfn: *mut u64) {
let mut this_start_pfn: u64 = 0;
let mut this_end_pfn: u64 = 0;
let mut i: i32 = 0;
unsafe {
*start_pfn = 1u64.wrapping_neg();
}
unsafe {
*end_pfn = 0;
}
i = 1i32.wrapping_neg();
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), nid as i32, core::ptr::addr_of_mut!(this_start_pfn), core::ptr::addr_of_mut!(this_end_pfn), core::ptr::null_mut::<i32>()) };
while i >= 0 {
let mut __v0: u64;
let __UNIQUE_ID_x__766: u64 = unsafe { *start_pfn };
let __UNIQUE_ID_y__767: u64 = this_start_pfn;
loop {
if __types_ok__2!(__UNIQUE_ID_x__766, __UNIQUE_ID_y__767) == 0 {
unsafe { __compiletime_assert_768() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__766 < __UNIQUE_ID_y__767 { __UNIQUE_ID_x__766 } else { __UNIQUE_ID_y__767 };
unsafe {
*start_pfn = __v0;
}
let mut __v2: u64;
let __UNIQUE_ID_x__769: u64 = unsafe { *end_pfn };
let __UNIQUE_ID_y__770: u64 = this_end_pfn;
loop {
if __types_ok__2!(__UNIQUE_ID_x__769, __UNIQUE_ID_y__770) == 0 {
unsafe { __compiletime_assert_771() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__769 > __UNIQUE_ID_y__770 { __UNIQUE_ID_x__769 } else { __UNIQUE_ID_y__770 };
unsafe {
*end_pfn = __v2;
}
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), nid as i32, core::ptr::addr_of_mut!(this_start_pfn), core::ptr::addr_of_mut!(this_end_pfn), core::ptr::null_mut::<i32>()) };
}
if (unsafe { *start_pfn }) == 1u64.wrapping_neg() {
unsafe {
*start_pfn = 0;
}
}
}

fn want_init_on_alloc(flags: u32) -> bool {
if static_branch_maybe!(unsafe { core::ptr::addr_of_mut!(init_on_alloc) }) != 0 {
return true;
}
flags & (BIT!(8i32) as u32) != 0
}

fn want_init_on_free() -> bool {
static_branch_maybe__3!(unsafe { core::ptr::addr_of_mut!(init_on_free) }) != 0
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
fn phys_to_virt(address: u64) -> *mut () {
__va!(address)
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
struct memblock_region {
    base: u64,
    size: u64,
    flags: u32,
    nid: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct memblock_type {
    cnt: u64,
    max: u64,
    total_size: u64,
    regions: *mut memblock_region,
    name: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct memblock {
    bottom_up: bool,
    current_limit: u64,
    memory: memblock_type,
    reserved: memblock_type,
}


// __for_each_mem_range - iterate through memblock areas from type_a and not
// included in type_b. Or just type_a if type_b is NULL.
// @i: u64 used as loop variable
// @type_a: ptr to memblock_type to iterate
// @type_b: ptr to memblock_type which excludes from the iteration
// @nid: node selector, %NUMA_NO_NODE for all nodes
// @flags: pick from blocks based on memory attributes
// @p_start: ptr to phys_addr_t for start address of the range, can be %NULL
// @p_end: ptr to phys_addr_t for end address of the range, can be %NULL
// @p_nid: ptr to int for nid of the range, can be %NULL
fn memblock_is_hotpluggable(m: *mut memblock_region) -> bool {
((unsafe { (*m).flags }) as u32) & 1 != 0
}

fn memblock_is_mirror(m: *mut memblock_region) -> bool {
((unsafe { (*m).flags }) as u32) & 2 != 0
}

// for_each_mem_pfn_range - early memory pfn range iterator
// @i: an integer used as loop variable
// @nid: node selector, %MAX_NUMNODES for all nodes
// @p_start: ptr to ulong for start pfn of the range, can be %NULL
// @p_end: ptr to ulong for end pfn of the range, can be %NULL
// @p_nid: ptr to int for nid of the range, can be %NULL
//
// Walks over configured memory ranges.
// for_each_free_mem_range - iterate through free memblock areas
// @i: u64 used as loop variable
// @nid: node selector, %NUMA_NO_NODE for all nodes
// @flags: pick from blocks based on memory attributes
// @p_start: ptr to phys_addr_t for start address of the range, can be %NULL
// @p_end: ptr to phys_addr_t for end address of the range, can be %NULL
// @p_nid: ptr to int for nid of the range, can be %NULL
//
// Walks over free (memory && !reserved) areas of memblock.  Available as
// soon as memblock is initialized.
fn memblock_get_region_node(r: *const memblock_region) -> i32 {
unsafe { (*r).nid }
}

// MEMBLOCK_ALLOC_NOLEAKTRACE avoids kmemleak tracing. It implies
// MEMBLOCK_ALLOC_ACCESSIBLE
// We are using top down, so it is safe to use 0 here
#[inline(always)]
fn memblock_alloc(size: u64, align: u64) -> *mut () {
unsafe { memblock_alloc_try_nid(size, align, 0u64, 0u64, 1i32.wrapping_neg()) }
}

fn memblock_alloc_raw(size: u64, align: u64) -> *mut () {
unsafe { memblock_alloc_try_nid_raw(size, align, 0u64, 0u64, 1i32.wrapping_neg()) }
}

fn memblock_region_memory_base_pfn(reg: *const memblock_region) -> u64 {
PFN_UP!(unsafe { (*reg).base })
}

// for_each_mem_region - iterate over memory regions
// @region: loop variable
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn alloc_large_system_hash(tablename: *const u8, bucketsize: u64, mut numentries: u64, mut scale: i32, flags: i32, _hash_shift: *mut u32, _hash_mask: *mut u32, low_limit: u64, high_limit: u64) -> *mut () {
let mut max: u64 = high_limit;
let mut log2qty: u64 = 0;
let mut size: u64 = 0;
let mut table: *mut () = unsafe { core::mem::zeroed() };
let mut gfp_flags: u32 = 0;
let mut virt: bool = false;
let mut huge: bool = false;
// allow the kernel cmdline to have a say
if numentries == 0 {
// round applicable memory size up to nearest megabyte
numentries = unsafe { nr_kernel_pages };
// It isn't necessary when PAGE_SIZE >= 1MB
if _AC!() << 12i32 < 1048576 {
numentries = round_up!(numentries, 1048576 / (_AC!() << 12i32));
}
if high_limit == 0 {
let mut adapt: u64 = 16777216u64;
while adapt < numentries {
scale = scale.wrapping_add(1);
adapt <<= 2i32;
}
}
// limit to 1 bucket per 2^scale bytes of low memory
if scale > 12 {
numentries >>= scale.wrapping_sub(12);
} else {
numentries <<= 12i32.wrapping_sub(scale);
}
if unlikely!(numentries.wrapping_mul(bucketsize) < _AC!() << 12i32) != 0 {
numentries = (_AC!() << 12i32) / bucketsize;
}
}
numentries = roundup_pow_of_two__2!(numentries);
// limit allocation size to 1/16 total memory by default
if max == 0 {
max = (unsafe { nr_all_pages }) << 12i32 >> 4i32;
max = div64_ul!(max, bucketsize);
}
let mut __v9: u64;
let __UNIQUE_ID_x__780: u64 = max;
let __UNIQUE_ID_y__781: u64 = 2147483648;
loop {
if __types_ok__2!(__UNIQUE_ID_x__780, __UNIQUE_ID_y__781) == 0 {
unsafe { __compiletime_assert_782() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = if __UNIQUE_ID_x__780 < __UNIQUE_ID_y__781 { __UNIQUE_ID_x__780 } else { __UNIQUE_ID_y__781 };
max = __v9;
if numentries < low_limit {
numentries = low_limit;
}
if numentries > max {
numentries = max;
}
log2qty = ilog2!(numentries) as u64;
gfp_flags = if flags & 2 != 0 { (BIT!(5i32) as u32) | (BIT!(11i32) as u32) | (BIT!(8i32) as u32) } else { (BIT!(5i32) as u32) | (BIT!(11i32) as u32) };
loop {
virt = false;
size = bucketsize << log2qty;
if flags & 1 != 0 {
if flags & 2 != 0 {
table = memblock_alloc(size, 64u64);
} else {
table = memblock_alloc_raw(size, 64u64);
}
} else {
if get_order(size) > 10 || unsafe { hashdist } {
table = vmalloc_huge(size, gfp_flags);
virt = true;
if !table.is_null() {
huge = is_vm_area_hugepages(table as *const ());
}
} else {
// If bucketsize is not a power-of-two, we may free
// some pages at the end of hash table which
// alloc_pages_exact() automatically does
table = alloc_hooks!(unsafe { alloc_pages_exact_noprof(size, gfp_flags) });
kmemleak_alloc(table as *const (), size, 1i32, gfp_flags);
}
}
let __v17: bool = if table.is_null() && size > _AC!() << 12i32 {
let __v19: u64 = log2qty.wrapping_sub(1);
log2qty = __v19;
__v19 != 0
} else {
false
};
if !__v17 {
break;
}
}
if table.is_null() {
unsafe { panic(c"Failed to allocate %s hash table\n".as_ptr() as *const u8, tablename) };
unsafe { core::hint::unreachable_unchecked() };
}
let mut __v20: i32;
while false {
}
__v20 = unsafe { _printk(c"\x016%s hash table entries: %ld (order: %d, %lu bytes, %s)\n".as_ptr() as *const u8, tablename, 1u64 << log2qty, get_order(size), size, if virt { if huge { c"vmalloc hugepage".as_ptr() as *mut u8 } else { c"vmalloc".as_ptr() as *mut u8 } } else { c"linear".as_ptr() as *mut u8 }) };
if !_hash_shift.is_null() {
unsafe {
*_hash_shift = log2qty as u32;
}
}
if !_hash_mask.is_null() {
unsafe {
*_hash_mask = (1i32 << log2qty).wrapping_sub(1) as u32;
}
}
table
}

// Allocating during early boot?
// Zero allocated hash table
fn kmsan_init_shadow() {
}

fn kmsan_init_runtime() {
}

fn kmsan_memblock_free_pages(page: *mut page, order: u32) -> bool {
true
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


fn vmalloc_huge(size: u64, gfp_mask: u32) -> *mut () {
alloc_hooks!(unsafe { vmalloc_huge_node_noprof(size, gfp_mask, 1i32.wrapping_neg()) })
}

fn is_vm_area_hugepages(addr: *const ()) -> bool {
// This may not 100% tell if the area is mapped with > PAGE_SIZE
// page table entries, if for some reason the architecture indicates
// larger sizes are available but decides not to use them, nothing
// prevents that. This only indicates the size of the physical page
// allocated in the vmalloc layer.
let area: *mut vm_struct = unsafe { find_vm_area(addr) };
!area.is_null() && (unsafe { (*area).page_order }) > 0
}

fn kmemleak_init() {
}

fn kmemleak_alloc(ptr: *const (), size: u64, min_count: i32, gfp: u32) {
}

fn kfence_alloc_pool_and_metadata() {
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


fn str_on_off(v: bool) -> *const u8 {
(if v { c"on".as_ptr() as *mut u8 } else { c"off".as_ptr() as *mut u8 }) as *const u8
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
    __slate_bits_0: __slate_bits::__SlateBits820U0,
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
    __slate_bits_0: __slate_bits::__SlateBits647U0,
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

fn kho_memory_init() {
}

fn kho_memory_init_early() {
}

fn kho_scratch_migratetype(pfn: u64, mt: u32) -> u32 {
mt
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mempolicy {
    refcnt: __SlateRecord_types_h_190,
    mode: u16,
    flags: u16,
    nodes: __SlateRecord_nodemask_types_h_17,
    home_node: i32,
    w: __SlateRecord_mempolicy_h_54,
    rcu: callback_head,
}


#[unsafe(link_section = ".init.text")]
fn hugetlb_cma_reserve() {
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
extern "C" fn set_zone_contiguous(zone: *mut zone) {
let mut block_start_pfn: u64 = unsafe { (*zone).zone_start_pfn };
let mut block_end_pfn: u64 = pageblock_end_pfn__2!(block_start_pfn);
while block_start_pfn < zone_end_pfn(zone as *const zone) {
let mut __v0: u64;
let __UNIQUE_ID_x__777: u64 = block_end_pfn;
let __UNIQUE_ID_y__778: u64 = zone_end_pfn(zone as *const zone);
loop {
if __types_ok__2!(__UNIQUE_ID_x__777, __UNIQUE_ID_y__778) == 0 {
unsafe { __compiletime_assert_779() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__777 < __UNIQUE_ID_y__778 { __UNIQUE_ID_x__777 } else { __UNIQUE_ID_y__778 };
block_end_pfn = __v0;
if unsafe { __pageblock_pfn_to_page(block_start_pfn, block_end_pfn, zone) }.is_null() {
return;
}
cond_resched__70!();
block_start_pfn = block_end_pfn;
let __v4: u64 = block_end_pfn;
block_end_pfn = __v4.wrapping_add(1u64 << MIN_T__2!(u32, 9i32, 10i32));
}
// We confirm that there is no hole
unsafe {
(*zone).contiguous = true;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.data")]
static mut mminit_loglevel: i32 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn mminit_verify_pageflags_layout() {
let mut shift: i32 = 64;
let mut width: i32 = 0;
let mut or_mask: u64 = 0;
let mut add_mask: u64 = 0;
let mut __v0: i32;
let __UNIQUE_ID_x__671: i32 = 0;
let __UNIQUE_ID_y__672: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__671, __UNIQUE_ID_y__672) == 0 {
unsafe { __compiletime_assert_673() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__671 < __UNIQUE_ID_y__672 { __UNIQUE_ID_x__671 } else { __UNIQUE_ID_y__672 };
width = shift.wrapping_sub(8i32.wrapping_add(__v0));
loop {
if 2 < unsafe { mminit_loglevel } {
let mut __v2: i32;
while false {
}
let mut __v4: i32;
let __UNIQUE_ID_x__677: i32 = 0;
let __UNIQUE_ID_y__678: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__677, __UNIQUE_ID_y__678) == 0 {
unsafe { __compiletime_assert_679() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__677 < __UNIQUE_ID_y__678 { __UNIQUE_ID_x__677 } else { __UNIQUE_ID_y__678 };
__v2 = unsafe { _printk(c"\x017mminit::pageflags_layout_widths Section %d Node %d Zone %d Lastcpupid %d Kasantag %d Gen %d Tier %d Flags %d\n".as_ptr() as *const u8, 0i32, 6i32, 2i32, 0i32, 0i32, 0i32, __v4, 22i32) };
}
if true {
break;
}
}
loop {
if 2 < unsafe { mminit_loglevel } {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x017mminit::pageflags_layout_shifts Section %d Node %d Zone %d Lastcpupid %d Kasantag %d\n".as_ptr() as *const u8, if pgtable_l5_enabled!() != 0 { 52i32 } else { 46i32 }.wrapping_sub(27), 6i32, 2i32, 0i32, 0i32) };
}
if true {
break;
}
}
loop {
if 2 < unsafe { mminit_loglevel } {
let mut __v10: i32;
while false {
}
__v10 = unsafe { _printk(c"\x017mminit::pageflags_layout_pgshifts Section %lu Node %lu Zone %lu Lastcpupid %lu Kasantag %lu\n".as_ptr() as *const u8, (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_mul((false as i32) as u64), (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64), (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64), (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_mul((false as i32) as u64), (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_mul((false as i32) as u64)) };
}
if true {
break;
}
}
loop {
if 2 < unsafe { mminit_loglevel } {
let mut __v13: i32;
while false {
}
__v13 = unsafe { _printk(c"\x017mminit::pageflags_layout_nodezoneid Node/Zone ID: %lu -> %lu\n".as_ptr() as *const u8, (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_add(8u64), (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2)) };
}
if true {
break;
}
}
loop {
if 2 < unsafe { mminit_loglevel } {
let mut __v16: i32;
while false {
}
__v16 = unsafe { _printk(c"\x017mminit::pageflags_layout_usage location: %d -> %d layout %d -> %d unused %d -> %d page-flags\n".as_ptr() as *const u8, shift, width, width, 22i32, 22i32, 0i32) };
}
if true {
break;
}
}
shift = shift.wrapping_sub(6);
loop {
if unlikely!((shift as u64) != (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64)) != 0 {
loop {
unsafe {
core::arch::asm!("681: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 681b - ., 3; .popsection\n# {0}", const 681 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mminit_verify_pageflags_layout_21, sym __slate_asm_string_mminit_verify_pageflags_layout_22, const 165 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
shift = shift.wrapping_sub(2);
loop {
if unlikely!((shift as u64) != (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64)) != 0 {
loop {
unsafe {
core::arch::asm!("682: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 682b - ., 3; .popsection\n# {0}", const 682 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mminit_verify_pageflags_layout_28, sym __slate_asm_string_mminit_verify_pageflags_layout_29, const 169 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
// Check for bitmask overlaps
or_mask = 3u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64) | 63u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64) | 0u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_mul((false as i32) as u64);
add_mask = (3u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64)).wrapping_add(63u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64)).wrapping_add(0u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_mul((false as i32) as u64));
loop {
if unlikely!(or_mask != add_mask) != 0 {
loop {
unsafe {
core::arch::asm!("683: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 683b - ., 3; .popsection\n# {0}", const 683 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mminit_verify_pageflags_layout_33, sym __slate_asm_string_mminit_verify_pageflags_layout_34, const 179 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.data")]
static mut mirrored_kernelcore: bool = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct reciprocal_value {
    m: u32,
    sh1: u8,
    sh2: u8,
}


// no flags
// a kmalloc_nolock() allocation
// prevent obj_exts array allocation
// spinning not allowed
// slab_folio - The folio allocated for a slab
// @s: The slab.
//
// Slabs are allocated as folios that contain the individual objects and are
// using some fields in the first struct page of the folio - those fields are
// now accessed by struct slab. It is occasionally necessary to convert back to
// a folio in order to communicate with the rest of the mm.  Please use this
// helper function instead of casting yourself, as the implementation may change
// in the future.
#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache_order_objects {
    x: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache_per_node_ptrs {
    barn: *mut node_barn,
    node: *mut kmem_cache_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
    cpu_sheaves: *mut slub_percpu_sheaves,
    flags: u32,
    min_partial: u64,
    size: u32,
    object_size: u32,
    reciprocal_size: reciprocal_value,
    offset: u32,
    sheaf_capacity: u32,
    oo: kmem_cache_order_objects,
    min: kmem_cache_order_objects,
    allocflags: u32,
    refcount: i32,
    ctor: Option<unsafe extern "C" fn(*mut ())>,
    inuse: u32,
    align: u32,
    red_left_pad: u32,
    name: *const u8,
    list: list_head,
    kobj: kobject,
    remote_node_defrag_ratio: u32,
    per_node: [kmem_cache_per_node_ptrs; 64],
}


fn shuffle_free_memory(pgdat: *mut pglist_data) {
}

/// high_memory defines the upper bound on direct map memory, then end
/// of ZONE_NORMAL.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_high_memory_667: *mut () = (unsafe { core::ptr::addr_of_mut!(high_memory) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..ro_after_init")]
static mut zero_page_pfn: u64 = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_zero_page_pfn_668: *mut () = (unsafe { core::ptr::addr_of_mut!(zero_page_pfn) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_empty_zero_page_669: *mut () = (unsafe { core::ptr::addr_of!(empty_zero_page.0) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___zero_page_670: *mut () = (unsafe { core::ptr::addr_of_mut!(__zero_page) }) as *mut ();

/// The zonelists are simply reported, validation is manual.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn mminit_verify_zonelist() {
let mut nid: i32 = 0;
if (unsafe { mminit_loglevel }) < 1 {
return;
}
nid = first_node!(node_states[1]) as i32;
while nid < 64i32 {
let pgdat: *mut pglist_data = NODE_DATA!(nid);
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut z: *mut zoneref = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut listid: i32 = 0;
let mut zoneid: i32 = 0;
while i < 8i32 {
// Identify the zone and nodelist
zoneid = i % 4;
listid = i / 4;
let zonelist: *mut zonelist = unsafe { (*pgdat).node_zonelists.as_mut_ptr() as *mut zonelist }.wrapping_offset(listid as isize);
zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset(zoneid as isize);
if populated_zone(zone as *const zone) {
// Print information about the zonelist
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x017mminit::zonelist %s %d:%s = ".as_ptr() as *const u8, if listid > 0 { c"thisnode".as_ptr() as *mut u8 } else { c"general".as_ptr() as *mut u8 }, nid, (*zone).name) };
// Iterate the zonelist
z = first_zones_zonelist(zonelist, zoneid as u32, core::ptr::null::<__SlateRecord_nodemask_types_h_17>());
zone = zonelist_zone(z);
while !zone.is_null() {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x01c%d:%s ".as_ptr() as *const u8, zone_to_nid(zone as *const zone), (*zone).name) };
let __v7: *mut zoneref = z;
let __v8: *mut zoneref = __v7.wrapping_offset(1 as isize);
z = __v8;
z = next_zones_zonelist(__v8, zoneid as u32, core::ptr::null::<__SlateRecord_nodemask_types_h_17>());
zone = zonelist_zone(z);
}
let mut __v10: i32;
while false {
}
__v10 = unsafe { _printk(c"\x01c\n".as_ptr() as *const u8) };
}
i = i.wrapping_add(1);
}
nid = next_node!(nid, node_states[1]) as i32;
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn set_mminit_loglevel(mut str: *mut u8) -> i32 {
unsafe { get_option(core::ptr::addr_of_mut!(str), core::ptr::addr_of_mut!(mminit_loglevel)) };
0
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_set_mminit_loglevel: [u8; 16] = [109, 109, 105, 110, 105, 116, 95, 108, 111, 103, 108, 101, 118, 101, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_set_mminit_loglevel: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_set_mminit_loglevel) as *const u8 };
__t0.setup_func = Some(set_mminit_loglevel);
__t0.early = 1i32;
__t0
};

#[unsafe(no_mangle)]
static mut vm_committed_as_batch: i32 = 32;

#[unsafe(no_mangle)]
extern "C" fn mm_compute_batch(overcommit_policy: i32) {
let mut memsized_batch: u64 = 0;
let nr: i32 = num_present_cpus!() as i32;
let batch: i32 = max_t__2!(i32, nr.wrapping_mul(2), 32i32);
let ram_pages: u64 = totalram_pages();
// For policy OVERCOMMIT_NEVER, set batch size to 0.4% of
// (total memory/#cpus), and lift it to 25% for other policies
// to ease the possible lock contention for percpu_counter
// vm_committed_as, while the max limit is INT_MAX
if overcommit_policy == 2 {
memsized_batch = min_t!(u64, ram_pages / (nr as u64) / 256, 2147483647i32);
} else {
memsized_batch = min_t!(u64, ram_pages / (nr as u64) / 4, 2147483647i32);
}
unsafe {
vm_committed_as_batch = max_t!(i32, memsized_batch, batch);
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn mm_compute_batch_notifier(self_: *mut notifier_block, action: u64, arg: *mut ()) -> i32 {
match action {
0 | 2 => {
mm_compute_batch(unsafe { sysctl_overcommit_memory });
}
_ => {
}
}
1
}

#[unsafe(link_section = ".init.text")]
extern "C" fn mm_compute_batch_init() -> i32 {
mm_compute_batch(unsafe { sysctl_overcommit_memory });
hotplug_memory_notifier(Some(mm_compute_batch_notifier), 10i32);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mm_compute_batch_init_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(mm_compute_batch_init)) };

#[unsafe(link_section = ".init.text")]
extern "C" fn mm_sysfs_init() -> i32 {
unsafe {
mm_kobj = kobject_create_and_add(c"mm".as_ptr() as *const u8, kernel_kobj);
}
if unsafe { mm_kobj }.is_null() {
return 12i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mm_sysfs_init_695: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(mm_sysfs_init)) };

#[unsafe(link_section = ".init.data")]
static mut arch_zone_lowest_possible_pfn: __SlateAlign16<[u64; 4]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(link_section = ".init.data")]
static mut arch_zone_highest_possible_pfn: __SlateAlign16<[u64; 4]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(link_section = ".init.data")]
static mut zone_movable_pfn: __SlateAlign16<[u64; 64]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(link_section = ".init.data")]
static mut required_kernelcore: u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut required_kernelcore_percent: u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut required_movablecore: u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut required_movablecore_percent: u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut nr_kernel_pages: u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut nr_all_pages: u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut deferred_struct_pages: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".data..percpu")]
static mut boot_nodestats: per_cpu_nodestat = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn cmdline_parse_core(mut p: *mut u8, core: *mut u64, percent: *mut u64) -> i32 {
let mut coremem: u64 = 0;
let mut endptr: *mut u8 = unsafe { core::mem::zeroed() };
if p.is_null() {
return 22i32.wrapping_neg();
}
// Value may be a percentage of total memory, otherwise bytes
coremem = unsafe { simple_strtoull(p as *const u8, core::ptr::addr_of_mut!(endptr), 0u32) };
if ((unsafe { *endptr }) as i32) == 37 {
// Paranoid check for percent values greater than 100
let mut __v0: i64;
let __ret_warn_on: i32 = (coremem > 100) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("696: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 696b - ., 3; .popsection\n# {0}", const 696 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cmdline_parse_core_1, sym __slate_asm_string_cmdline_parse_core_2, const 280 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("697: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 697b - ., 4; .popsection\n# {0}", const 697 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe {
*percent = coremem;
}
} else {
coremem = unsafe { memparse(p as *const u8, core::ptr::addr_of_mut!(p)) };
// Paranoid check that UL is enough for the coremem value
let mut __v5: i64;
let __ret_warn_on: i32 = (coremem >> 12i32 > !0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("698: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 698b - ., 3; .popsection\n# {0}", const 698 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cmdline_parse_core_6, sym __slate_asm_string_cmdline_parse_core_7, const 286 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("699: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 699b - ., 4; .popsection\n# {0}", const 699 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
unsafe {
*core = coremem >> 12i32;
}
unsafe {
*percent = 0;
}
}
0
}

/// kernelcore=size sets the amount of memory for use for allocations that
/// cannot be reclaimed or migrated.
#[unsafe(link_section = ".init.text")]
extern "C" fn cmdline_parse_kernelcore(p: *mut u8) -> i32 {
// parse kernelcore=mirror
if unsafe { parse_option_str(p as *const u8, c"mirror".as_ptr() as *const u8) } {
unsafe {
mirrored_kernelcore = true;
}
return 0;
}
cmdline_parse_core(p, unsafe { core::ptr::addr_of_mut!(required_kernelcore) }, unsafe { core::ptr::addr_of_mut!(required_kernelcore_percent) })
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_cmdline_parse_kernelcore: [u8; 11] = [107, 101, 114, 110, 101, 108, 99, 111, 114, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_cmdline_parse_kernelcore: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_cmdline_parse_kernelcore) as *const u8 };
__t0.setup_func = Some(cmdline_parse_kernelcore);
__t0.early = 1i32;
__t0
};

/// movablecore=size sets the amount of memory for use for allocations that
/// can be reclaimed or migrated.
#[unsafe(link_section = ".init.text")]
extern "C" fn cmdline_parse_movablecore(p: *mut u8) -> i32 {
cmdline_parse_core(p, unsafe { core::ptr::addr_of_mut!(required_movablecore) }, unsafe { core::ptr::addr_of_mut!(required_movablecore_percent) })
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_cmdline_parse_movablecore: [u8; 12] = [109, 111, 118, 97, 98, 108, 101, 99, 111, 114, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_cmdline_parse_movablecore: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_cmdline_parse_movablecore) as *const u8 };
__t0.setup_func = Some(cmdline_parse_movablecore);
__t0.early = 1i32;
__t0
};

/// early_calculate_totalpages()
/// Sum pages in active regions for movable zone.
/// Populate N_MEMORY for calculating usable_nodes.
#[unsafe(link_section = ".init.text")]
fn early_calculate_totalpages() -> u64 {
let mut totalpages: u64 = 0;
let mut start_pfn: u64 = 0;
let mut end_pfn: u64 = 0;
let mut i: i32 = 1i32.wrapping_neg();
let mut nid: i32 = 0;
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn), core::ptr::addr_of_mut!(nid)) };
while i >= 0 {
let pages: u64 = end_pfn.wrapping_sub(start_pfn);
totalpages = totalpages.wrapping_add(pages);
node_set_state(nid, 3u32);
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn), core::ptr::addr_of_mut!(nid)) };
}
totalpages
}

/// This finds a zone that can be used for ZONE_MOVABLE pages. The
/// assumption is made that zones within a node are ordered in monotonic
/// increasing memory addresses so that the "highest" populated zone is used
#[unsafe(link_section = ".init.text")]
fn find_usable_zone_for_movable() {
let mut zone_index: i32 = 3i32;
while zone_index >= 0 {
if zone_index != 3 {
if (unsafe { arch_zone_highest_possible_pfn.0[zone_index as usize] }) > unsafe { arch_zone_lowest_possible_pfn.0[zone_index as usize] } {
break;
}
}
zone_index = zone_index.wrapping_sub(1);
}
VM_BUG_ON!();
unsafe {
movable_zone = zone_index;
}
}

// Find the PFN the Movable zone begins in each node. Kernel memory
// is spread evenly between nodes as long as the nodes have enough
// memory. When they don't, some nodes will have more kernelcore than
// others
#[unsafe(link_section = ".init.text")]
fn find_zone_movable_pfns_for_nodes() {
let mut end_pfn: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u64 = core::ptr::addr_of_mut!(end_pfn) as *mut u64;
let mut start_pfn: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(start_pfn) as *mut u64;
let mut __UNIQUE_ID_y__725: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__724: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__722: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__721: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__719: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__718: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__716: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__715: u64 = unsafe { core::mem::zeroed() };
let mut kernel_pages: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u64 = core::ptr::addr_of_mut!(kernel_pages) as *mut u64;
let mut __UNIQUE_ID_y__713: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__712: u64 = unsafe { core::mem::zeroed() };
let mut size_pages: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut u64 = core::ptr::addr_of_mut!(size_pages) as *mut u64;
let mut end_pfn: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut u64 = core::ptr::addr_of_mut!(end_pfn) as *mut u64;
let mut start_pfn: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u64 = core::ptr::addr_of_mut!(start_pfn) as *mut u64;
let mut __UNIQUE_ID_y__710: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__709: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__707: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__706: u64 = unsafe { core::mem::zeroed() };
let mut corepages: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut u64 = core::ptr::addr_of_mut!(corepages) as *mut u64;
let mut __UNIQUE_ID_y__704: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__703: u64 = unsafe { core::mem::zeroed() };
let mut __v47: u64 = unsafe { core::mem::zeroed() };
let mut mem_below_4gb_not_mirrored: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__701: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__700: u64 = unsafe { core::mem::zeroed() };
let mut __v56: u64 = unsafe { core::mem::zeroed() };
let mut r: core::mem::MaybeUninit<*mut memblock_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_57: *mut *mut memblock_region = core::ptr::addr_of_mut!(r) as *mut *mut memblock_region;
let mut usable_nodes: i32 = unsafe { core::mem::zeroed() };
let mut totalpages: u64 = unsafe { core::mem::zeroed() };
// save the state before borrow the nodemask
let mut saved_node_state: core::mem::MaybeUninit<__SlateRecord_nodemask_types_h_17> = core::mem::MaybeUninit::uninit();
let __slate_slot_60: *mut __SlateRecord_nodemask_types_h_17 = core::ptr::addr_of_mut!(saved_node_state) as *mut __SlateRecord_nodemask_types_h_17;
let mut kernelcore_remaining: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_61: *mut u64 = core::ptr::addr_of_mut!(kernelcore_remaining) as *mut u64;
let mut kernelcore_node: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_62: *mut u64 = core::ptr::addr_of_mut!(kernelcore_node) as *mut u64;
let mut usable_startpfn: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_63: *mut u64 = core::ptr::addr_of_mut!(usable_startpfn) as *mut u64;
let mut nid: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_64: *mut i32 = core::ptr::addr_of_mut!(nid) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_65: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
'__join_0: {
let __slate_auto_bytes_3 = __slate_slot_65 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_64 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_63 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_62 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_61 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_60, node_states[3]);
totalpages = early_calculate_totalpages();
usable_nodes = nodes_weight!(node_states[3]);
let __slate_auto_bytes_8 = __slate_slot_57 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
// Need to find movable_zone earlier when movable_node is specified.
find_usable_zone_for_movable();
// If movable_node is specified, ignore kernelcore and movablecore
// options.
if movable_node_is_enabled() {
*__slate_slot_57 = memblock.memory.regions;
while *__slate_slot_57 < unsafe { memblock.memory.regions }.wrapping_offset(memblock.memory.cnt as isize) {
if memblock_is_hotpluggable(*__slate_slot_57) {
*__slate_slot_64 = memblock_get_region_node(*__slate_slot_57 as *const memblock_region);
*__slate_slot_63 = memblock_region_memory_base_pfn(*__slate_slot_57 as *const memblock_region);
if zone_movable_pfn.0[*__slate_slot_64 as usize] != 0 {
__UNIQUE_ID_x__700 = *__slate_slot_63;
__UNIQUE_ID_y__701 = zone_movable_pfn.0[*__slate_slot_64 as usize];
while false {
}
__v56 = if __UNIQUE_ID_x__700 < __UNIQUE_ID_y__701 { __UNIQUE_ID_x__700 } else { __UNIQUE_ID_y__701 };
} else {
__v56 = *__slate_slot_63;
}
zone_movable_pfn.0[*__slate_slot_64 as usize] = __v56;
}
let __v52: *mut memblock_region = *__slate_slot_57;
*__slate_slot_57 = __v52.wrapping_offset(1 as isize);
}
} else {
// If kernelcore=mirror is specified, ignore movablecore option
if mirrored_kernelcore {
mem_below_4gb_not_mirrored = false;
if !memblock_has_mirror() {
while false {
}
_printk(c"\x014The system has no mirror memory, ignore kernelcore=mirror.\n".as_ptr() as *const u8);
break '__join_0;
} else {
if is_kdump_kernel() {
while false {
}
_printk(c"\x014The system is under kdump, ignore kernelcore=mirror.\n".as_ptr() as *const u8);
break '__join_0;
} else {
*__slate_slot_57 = memblock.memory.regions;
while *__slate_slot_57 < unsafe { memblock.memory.regions }.wrapping_offset(memblock.memory.cnt as isize) {
if !memblock_is_mirror(*__slate_slot_57) {
*__slate_slot_64 = memblock_get_region_node(*__slate_slot_57 as *const memblock_region);
*__slate_slot_63 = memblock_region_memory_base_pfn(*__slate_slot_57 as *const memblock_region);
if *__slate_slot_63 < PHYS_PFN!(_AC__5!()) {
mem_below_4gb_not_mirrored = true;
} else {
if zone_movable_pfn.0[*__slate_slot_64 as usize] != 0 {
__UNIQUE_ID_x__703 = *__slate_slot_63;
__UNIQUE_ID_y__704 = zone_movable_pfn.0[*__slate_slot_64 as usize];
while false {
}
__v47 = if __UNIQUE_ID_x__703 < __UNIQUE_ID_y__704 { __UNIQUE_ID_x__703 } else { __UNIQUE_ID_y__704 };
} else {
__v47 = *__slate_slot_63;
}
zone_movable_pfn.0[*__slate_slot_64 as usize] = __v47;
}
}
let __v43: *mut memblock_region = *__slate_slot_57;
*__slate_slot_57 = __v43.wrapping_offset(1 as isize);
}
if mem_below_4gb_not_mirrored {
while false {
}
_printk(c"\x014This configuration results in unmirrored kernel memory.\n".as_ptr() as *const u8);
}
}
}
} else {
// If kernelcore=nn% or movablecore=nn% was specified, calculate the
// amount of necessary memory.
if required_kernelcore_percent != 0 {
required_kernelcore = totalpages.wrapping_mul(100).wrapping_mul(required_kernelcore_percent) / 10000;
}
if required_movablecore_percent != 0 {
required_movablecore = totalpages.wrapping_mul(100).wrapping_mul(required_movablecore_percent) / 10000;
}
// If movablecore= was specified, calculate what size of
// kernelcore that corresponds so that memory usable for
// any allocation type is evenly spread. If both kernelcore
// and movablecore are specified, then the value of kernelcore
// will be used for required_kernelcore if it's greater than
// what movablecore would have allowed.
if required_movablecore != 0 {
let __slate_auto_bytes = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Round-up so that ZONE_MOVABLE is at least as large as what
// was requested by the user
required_movablecore = round_up__2!(unsafe { required_movablecore }, 1024i32);
__UNIQUE_ID_x__706 = totalpages;
__UNIQUE_ID_y__707 = required_movablecore;
while false {
}
required_movablecore = if __UNIQUE_ID_x__706 < __UNIQUE_ID_y__707 { __UNIQUE_ID_x__706 } else { __UNIQUE_ID_y__707 };
*__slate_slot_40 = totalpages.wrapping_sub(required_movablecore);
__UNIQUE_ID_x__709 = required_kernelcore;
__UNIQUE_ID_y__710 = *__slate_slot_40;
while false {
}
required_kernelcore = if __UNIQUE_ID_x__709 > __UNIQUE_ID_y__710 { __UNIQUE_ID_x__709 } else { __UNIQUE_ID_y__710 };
}
// If kernelcore was not specified or kernelcore size is larger
// than totalpages, there is no ZONE_MOVABLE.
if required_kernelcore == 0 || required_kernelcore >= totalpages {
break '__join_0;
} else {
// usable_startpfn is the lowest possible pfn ZONE_MOVABLE can be at
*__slate_slot_63 = arch_zone_lowest_possible_pfn.0[movable_zone as usize];
loop {
*__slate_slot_62 = required_kernelcore / (usable_nodes as u64); // Spread kernelcore memory as evenly as possible throughout nodes
*__slate_slot_64 = first_node!(node_states[3]) as i32;
while *__slate_slot_64 < 64i32 {
{
let __slate_auto_bytes_2 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Recalculate kernelcore_node if the division per node
// now exceeds what is necessary to satisfy the requested
// amount of memory for the kernel
if required_kernelcore < *__slate_slot_62 {
*__slate_slot_62 = required_kernelcore / (usable_nodes as u64);
}
}
// As the map is walked, we track how much memory is usable
// by the kernel using kernelcore_remaining. When it is
// 0, the rest of the node is usable by ZONE_MOVABLE
*__slate_slot_61 = *__slate_slot_62;
// Go through each range of PFNs within this node
*__slate_slot_65 = 1i32.wrapping_neg();
__next_mem_pfn_range(core::ptr::addr_of_mut!(*__slate_slot_65), *__slate_slot_64, core::ptr::addr_of_mut!(*__slate_slot_33), core::ptr::addr_of_mut!(*__slate_slot_32), core::ptr::null_mut::<i32>());
'__loop_10: while *__slate_slot_65 >= 0 {
let __slate_auto_bytes_1 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
__UNIQUE_ID_x__712 = *__slate_slot_33;
__UNIQUE_ID_y__713 = zone_movable_pfn.0[*__slate_slot_64 as usize];
while false {
}
'__join_11: {
*__slate_slot_33 = if __UNIQUE_ID_x__712 > __UNIQUE_ID_y__713 { __UNIQUE_ID_x__712 } else { __UNIQUE_ID_y__713 };
if !(*__slate_slot_33 >= *__slate_slot_32) {
// Account for what is only usable for kernelcore
if *__slate_slot_33 < *__slate_slot_63 {
let __slate_auto_bytes = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
__UNIQUE_ID_x__715 = *__slate_slot_32;
__UNIQUE_ID_y__716 = *__slate_slot_63;
while false {
}
let __v26: u64 = if __UNIQUE_ID_x__715 < __UNIQUE_ID_y__716 { __UNIQUE_ID_x__715 } else { __UNIQUE_ID_y__716 };
*__slate_slot_27 = __v26.wrapping_sub(*__slate_slot_33);
__UNIQUE_ID_x__718 = *__slate_slot_27;
__UNIQUE_ID_y__719 = *__slate_slot_61;
while false {
}
let __v19: u64 = *__slate_slot_61;
*__slate_slot_61 = __v19.wrapping_sub(if __UNIQUE_ID_x__718 < __UNIQUE_ID_y__719 { __UNIQUE_ID_x__718 } else { __UNIQUE_ID_y__719 });
__UNIQUE_ID_x__721 = *__slate_slot_27;
__UNIQUE_ID_y__722 = required_kernelcore;
while false {
}
let __v13: u64 = required_kernelcore;
required_kernelcore = __v13.wrapping_sub(if __UNIQUE_ID_x__721 < __UNIQUE_ID_y__722 { __UNIQUE_ID_x__721 } else { __UNIQUE_ID_y__722 });
// Continue if range is now fully accounted
if *__slate_slot_32 <= *__slate_slot_63 {
// Push zone_movable_pfn to the end so
// that if we have to rebalance
// kernelcore across nodes, we will
// not double account here
zone_movable_pfn.0[*__slate_slot_64 as usize] = *__slate_slot_32;
break '__join_11;
} else {
*__slate_slot_33 = *__slate_slot_63;
}
}
// The usable PFN range for ZONE_MOVABLE is from
// start_pfn->end_pfn. Calculate size_pages as the
// number of pages used as kernelcore
*__slate_slot_31 = (*__slate_slot_32).wrapping_sub(*__slate_slot_33);
if *__slate_slot_31 > *__slate_slot_61 {
*__slate_slot_31 = *__slate_slot_61;
}
zone_movable_pfn.0[*__slate_slot_64 as usize] = (*__slate_slot_33).wrapping_add(*__slate_slot_31);
__UNIQUE_ID_x__724 = required_kernelcore;
__UNIQUE_ID_y__725 = *__slate_slot_31;
while false {
}
let __v7: u64 = required_kernelcore;
// Some kernelcore has been met, update counts and
// break if the kernelcore for this node has been
// satisfied
required_kernelcore = __v7.wrapping_sub(if __UNIQUE_ID_x__724 < __UNIQUE_ID_y__725 { __UNIQUE_ID_x__724 } else { __UNIQUE_ID_y__725 });
let __v5: u64 = *__slate_slot_61;
*__slate_slot_61 = __v5.wrapping_sub(*__slate_slot_31);
if *__slate_slot_61 == 0 {
break '__loop_10;
}
}
}
__next_mem_pfn_range(core::ptr::addr_of_mut!(*__slate_slot_65), *__slate_slot_64, core::ptr::addr_of_mut!(*__slate_slot_33), core::ptr::addr_of_mut!(*__slate_slot_32), core::ptr::null_mut::<i32>());
}
*__slate_slot_64 = next_node!(*__slate_slot_64, node_states[3]) as i32;
}
// If there is still required_kernelcore, we do another pass with one
// less node in the count. This will push zone_movable_pfn[nid] further
// along on the nodes that still have memory until kernelcore is
// satisfied
usable_nodes = usable_nodes.wrapping_sub(1);
if !(usable_nodes != 0 && required_kernelcore > (usable_nodes as u64)) {
break;
}
}
}
}
}
*__slate_slot_64 = first_node!(node_states[3]) as i32;
while *__slate_slot_64 < 64i32 {
let __slate_auto_bytes_9 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
zone_movable_pfn.0[*__slate_slot_64 as usize] = round_up__2!(unsafe { zone_movable_pfn.0[*__slate_slot_64 as usize] }, 1024i32);
get_pfn_range_for_nid(*__slate_slot_64 as u32, core::ptr::addr_of_mut!(*__slate_slot_1), core::ptr::addr_of_mut!(*__slate_slot_0));
if zone_movable_pfn.0[*__slate_slot_64 as usize] >= *__slate_slot_0 {
zone_movable_pfn.0[*__slate_slot_64 as usize] = 0;
}
*__slate_slot_64 = next_node!(*__slate_slot_64, node_states[3]) as i32;
}
// Align start of ZONE_MOVABLE on all nids to MAX_ORDER_NR_PAGES
}
node_states[3] = *__slate_slot_60;
// restore the node_state
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn __init_single_page(page: *mut page, pfn: u64, zone: u64, nid: i32) {
mm_zero_struct_page!(page);
set_page_links(page, zone as u32, nid as u64, pfn);
init_page_count(page);
atomic_set(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_2._mapcount) }, 1i32.wrapping_neg());
page_cpupid_reset_last(page);
page_kasan_tag_reset(page);
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.lru) });
}

/// During memory init memblocks map pfns to nids. The search is expensive and
/// this caches recent lookups. The implementation of __early_pfn_to_nid
/// treats start/end as pfns.
#[repr(C)]
#[derive(Clone, Copy)]
struct mminit_pfnnid_cache {
    last_start: u64,
    last_end: u64,
    last_nid: i32,
}


#[unsafe(link_section = ".init.data")]
static mut early_pfnnid_cache: mminit_pfnnid_cache = unsafe { core::mem::zeroed() };

/// Required by SPARSEMEM. Given a PFN, return what node the PFN is on.
#[unsafe(link_section = ".init.text")]
fn __early_pfn_to_nid(pfn: u64, state: *mut mminit_pfnnid_cache) -> i32 {
let mut start_pfn: u64 = 0;
let mut end_pfn: u64 = 0;
let mut nid: i32 = 0;
if (unsafe { (*state).last_start }) <= pfn && pfn < unsafe { (*state).last_end } {
return unsafe { (*state).last_nid };
}
nid = unsafe { memblock_search_pfn_nid(pfn, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn)) };
if nid != 1i32.wrapping_neg() {
unsafe {
(*state).last_start = start_pfn;
}
unsafe {
(*state).last_end = end_pfn;
}
unsafe {
(*state).last_nid = nid;
}
}
nid
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn early_pfn_to_nid(pfn: u64) -> i32 {
let mut nid: i32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!(early_pfn_lock) });
nid = __early_pfn_to_nid(pfn, unsafe { core::ptr::addr_of_mut!(early_pfnnid_cache) });
if nid < 0 {
nid = first_node!(node_states[1]) as i32;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(early_pfn_lock) });
nid
}

static mut early_pfn_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

#[unsafe(no_mangle)]
static mut hashdist: bool = 1i32 != 0i32;

#[unsafe(link_section = ".init.text")]
extern "C" fn set_hashdist(str: *mut u8) -> i32 {
((unsafe { kstrtobool(str as *const u8, core::ptr::addr_of_mut!(hashdist)) }) == 0) as i32
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_set_hashdist: [u8; 10] = [104, 97, 115, 104, 100, 105, 115, 116, 61, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_set_hashdist: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_set_hashdist) as *const u8 };
__t0.setup_func = Some(set_hashdist);
__t0.early = 0i32;
__t0
};

fn fixup_hashdist() {
if num_node_state(3u32) == 1 {
unsafe {
hashdist = false;
}
}
}

fn pgdat_set_deferred_range(pgdat: *mut pglist_data) {
}

fn early_page_initialised(pfn: u64, nid: i32) -> bool {
true
}

fn defer_init(nid: i32, pfn: u64, end_pfn: u64) -> bool {
false
}

fn __init_deferred_page(pfn: u64, nid: i32) {
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn init_deferred_page(pfn: u64, nid: i32) {
__init_deferred_page(pfn, nid);
}

/// Only struct pages that correspond to ranges defined by memblock.memory
/// are zeroed and initialized by going through __init_single_page() during
/// memmap_init_zone_range().
///
/// But, there could be struct pages that correspond to holes in
/// memblock.memory. This can happen because of the following reasons:
/// - physical memory bank size is not necessarily the exact multiple of the
///   arbitrary section size
/// - early reserved memory may not be listed in memblock.memory
/// - non-memory regions covered by the contiguous flatmem mapping
/// - memory layouts defined with memmap= kernel parameter may not align
///   nicely with memmap sections
///
/// Explicitly initialize those struct pages so that:
/// - PG_Reserved is set
/// - zone and node links point to zone and node that span the page if the
///   hole is in the middle of a zone
/// - zone and node links point to adjacent zone/node if the hole falls on
///   the zone boundary; the pages in such holes will be prepended to the
///   zone/node above the hole except for the trailing pages in the last
///   section that will be appended to the zone/node below.
#[unsafe(link_section = ".init.text")]
fn init_unavailable_range(spfn: u64, epfn: u64, zone: i32, node: i32) {
let mut pfn: u64 = first_valid_pfn(spfn, epfn);
let mut pgcnt: u64 = 0;
while pfn < epfn {
__init_single_page(((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pfn as isize), pfn, zone as u64, node);
__SetPageReserved(((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pfn as isize));
pgcnt = pgcnt.wrapping_add(1);
pfn = next_valid_pfn(pfn, epfn);
}
if pgcnt != 0 {
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x016On node %d, zone %s: %lld pages in unavailable ranges\n".as_ptr() as *const u8, node, zone_names[zone as usize], pgcnt) };
}
}

/// Initially all pages are reserved - free ones are freed
/// up by memblock_free_all() once the early boot process is
/// done. Non-atomic initialization, single-pass.
///
/// All aligned pageblocks are initialized to the specified migratetype
/// (usually MIGRATE_MOVABLE). Besides setting the migratetype, no related
/// zone stats (e.g., nr_isolate_pageblock) are touched.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn memmap_init_range(size: u64, nid: i32, zone: u64, start_pfn: u64, zone_end_pfn: u64, context: u32, altmap: *mut vmem_altmap, migratetype: i32, isolate_pageblock: bool) {
let end_pfn: u64 = start_pfn.wrapping_add(size);
if (unsafe { highest_memmap_pfn }) < end_pfn.wrapping_sub(1) {
unsafe {
highest_memmap_pfn = end_pfn.wrapping_sub(1);
}
}
for pfn in start_pfn..end_pfn {
// There can be holes in boot-time mem_map[]s handed to this
// function.  They do not exist on hotplugged memory.
if (context as u32) == 0 {
if defer_init(nid, pfn, zone_end_pfn) {
unsafe {
deferred_struct_pages = true;
}
break;
}
}
let page: *mut page = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pfn as isize);
__init_single_page(page, pfn, zone, nid);
if (context as u32) == 1 {
__SetPageOffline(page);
}
// Usually, we want to mark the pageblock MIGRATE_MOVABLE,
// such that unmovable allocations won't be scattered all
// over the place during system boot.
if pageblock_aligned!(pfn) {
let mt: u32 = kho_scratch_migratetype(pfn, migratetype as u32);
unsafe { init_pageblock_migratetype(page, mt, isolate_pageblock) };
cond_resched__80!();
}
}
}

#[unsafe(link_section = ".init.text")]
fn memmap_init_zone_range(zone: *mut zone, mut start_pfn: u64, mut end_pfn: u64, hole_pfn: &mut u64) {
let zone_start_pfn: u64 = unsafe { (*zone).zone_start_pfn };
let zone_end_pfn: u64 = zone_start_pfn.wrapping_add(unsafe { (*zone).spanned_pages });
let nid: i32 = zone_to_nid(zone as *const zone);
let zone_id: i32 = zone_idx!(zone) as i32;
let mut __v0: u64;
let __UNIQUE_ID_v__727: u64 = start_pfn;
let __UNIQUE_ID_l__728: u64 = zone_start_pfn;
let __UNIQUE_ID_h__729: u64 = zone_end_pfn;
loop {
if !(!statically_true!(__UNIQUE_ID_l__728 > __UNIQUE_ID_h__729)) {
unsafe { __compiletime_assert_730() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
while false {
}
__v0 = if __UNIQUE_ID_v__727 >= __UNIQUE_ID_h__729 { __UNIQUE_ID_h__729 } else { if __UNIQUE_ID_v__727 <= __UNIQUE_ID_l__728 { __UNIQUE_ID_l__728 } else { __UNIQUE_ID_v__727 } };
start_pfn = __v0;
let mut __v3: u64;
let __UNIQUE_ID_v__732: u64 = end_pfn;
let __UNIQUE_ID_l__733: u64 = zone_start_pfn;
let __UNIQUE_ID_h__734: u64 = zone_end_pfn;
loop {
if !(!statically_true!(__UNIQUE_ID_l__733 > __UNIQUE_ID_h__734)) {
unsafe { __compiletime_assert_735() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
while false {
}
__v3 = if __UNIQUE_ID_v__732 >= __UNIQUE_ID_h__734 { __UNIQUE_ID_h__734 } else { if __UNIQUE_ID_v__732 <= __UNIQUE_ID_l__733 { __UNIQUE_ID_l__733 } else { __UNIQUE_ID_v__732 } };
end_pfn = __v3;
if start_pfn >= end_pfn {
return;
}
memmap_init_range(end_pfn.wrapping_sub(start_pfn), nid, zone_id as u64, start_pfn, zone_end_pfn, 0u32, core::ptr::null_mut::<vmem_altmap>(), 1i32, false);
if (unsafe { *(hole_pfn as *mut u64) }) < start_pfn {
init_unavailable_range(unsafe { *(hole_pfn as *mut u64) }, start_pfn, zone_id, nid);
}
unsafe {
*(hole_pfn as *mut u64) = end_pfn;
}
}

#[unsafe(link_section = ".init.text")]
fn memmap_init() {
let mut start_pfn: u64 = 0;
let mut end_pfn: u64 = 0;
let mut hole_pfn: u64 = 0;
let mut i: i32 = 1i32.wrapping_neg();
let mut zone_id: i32 = 0;
let mut nid: i32 = 0;
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn), core::ptr::addr_of_mut!(nid)) };
while i >= 0 {
let node: *mut pglist_data = NODE_DATA!(nid);
for j in 0i32..4i32 {
let zone: *mut zone = unsafe { (*node).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset(j as isize);
if populated_zone(zone as *const zone) {
memmap_init_zone_range(zone, start_pfn, end_pfn, unsafe { &mut (*core::ptr::addr_of_mut!(hole_pfn)) });
zone_id = j;
}
}
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn), core::ptr::addr_of_mut!(nid)) };
}
// Initialize the memory map for hole in the range [memory_end,
// section_end] for SPARSEMEM and in the range [memory_end, memmap_end]
// for FLATMEM.
// Append the pages in this hole to the highest zone in the last
// node.
end_pfn = round_up!(end_pfn, 32768u64);
if hole_pfn < end_pfn {
init_unavailable_range(hole_pfn, end_pfn, zone_id, nid);
}
}

// The zone ranges provided by the architecture do not include ZONE_MOVABLE
// because it is sized independent of architecture. Unlike the other zones,
// the starting point for ZONE_MOVABLE is not fixed. It may be different
// in each node depending on the size of each node and how evenly kernelcore
// is distributed. This helper function adjusts the zone ranges
// provided by the architecture for a given node by using the end of the
// highest usable zone for ZONE_MOVABLE. This preserves the assumption that
// zones within a node are in order of monotonic increases memory addresses
#[unsafe(link_section = ".init.text")]
fn adjust_zone_range_for_zone_movable(nid: i32, zone_type: u64, node_end_pfn: u64, zone_start_pfn: &mut u64, zone_end_pfn: &mut u64) {
// Only adjust if ZONE_MOVABLE is on this node
if (unsafe { zone_movable_pfn.0[nid as usize] }) != 0 {
// Size ZONE_MOVABLE
if zone_type == 3 {
unsafe {
*(zone_start_pfn as *mut u64) = zone_movable_pfn.0[nid as usize];
}
let mut __v0: u64;
let __UNIQUE_ID_x__737: u64 = node_end_pfn;
let __UNIQUE_ID_y__738: u64 = unsafe { arch_zone_highest_possible_pfn.0[movable_zone as usize] };
loop {
if __types_ok__2!(__UNIQUE_ID_x__737, __UNIQUE_ID_y__738) == 0 {
unsafe { __compiletime_assert_739() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__737 < __UNIQUE_ID_y__738 { __UNIQUE_ID_x__737 } else { __UNIQUE_ID_y__738 };
unsafe {
*(zone_end_pfn as *mut u64) = __v0;
}
// Adjust for ZONE_MOVABLE starting within this range
} else {
if (unsafe { *(zone_start_pfn as *mut u64) }) < unsafe { zone_movable_pfn.0[nid as usize] } && (unsafe { *(zone_end_pfn as *mut u64) }) > unsafe { zone_movable_pfn.0[nid as usize] } {
unsafe {
*(zone_end_pfn as *mut u64) = zone_movable_pfn.0[nid as usize];
}
// Check if this whole range is within ZONE_MOVABLE
} else {
if (unsafe { *(zone_start_pfn as *mut u64) }) >= unsafe { zone_movable_pfn.0[nid as usize] } {
unsafe {
*(zone_start_pfn as *mut u64) = *(zone_end_pfn as *mut u64);
}
}
}
}
}
}

// Return the number of holes in a range on a node. If nid is MAX_NUMNODES,
// then all holes in the requested range will be accounted for.
#[unsafe(link_section = ".init.text")]
fn __absent_pages_in_range(nid: i32, range_start_pfn: u64, range_end_pfn: u64) -> u64 {
let mut nr_absent: u64 = range_end_pfn.wrapping_sub(range_start_pfn);
let mut start_pfn: u64 = 0;
let mut end_pfn: u64 = 0;
let mut i: i32 = 1i32.wrapping_neg();
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), nid, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn), core::ptr::null_mut::<i32>()) };
while i >= 0 {
let mut __v0: u64;
let __UNIQUE_ID_v__740: u64 = start_pfn;
let __UNIQUE_ID_l__741: u64 = range_start_pfn;
let __UNIQUE_ID_h__742: u64 = range_end_pfn;
loop {
if !(!statically_true!(__UNIQUE_ID_l__741 > __UNIQUE_ID_h__742)) {
unsafe { __compiletime_assert_743() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3!(__UNIQUE_ID_v__740, __UNIQUE_ID_l__741, __UNIQUE_ID_h__742) == 0 {
unsafe { __compiletime_assert_744() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_v__740 >= __UNIQUE_ID_h__742 { __UNIQUE_ID_h__742 } else { if __UNIQUE_ID_v__740 <= __UNIQUE_ID_l__741 { __UNIQUE_ID_l__741 } else { __UNIQUE_ID_v__740 } };
start_pfn = __v0;
let mut __v3: u64;
let __UNIQUE_ID_v__745: u64 = end_pfn;
let __UNIQUE_ID_l__746: u64 = range_start_pfn;
let __UNIQUE_ID_h__747: u64 = range_end_pfn;
loop {
if !(!statically_true!(__UNIQUE_ID_l__746 > __UNIQUE_ID_h__747)) {
unsafe { __compiletime_assert_748() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3!(__UNIQUE_ID_v__745, __UNIQUE_ID_l__746, __UNIQUE_ID_h__747) == 0 {
unsafe { __compiletime_assert_749() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_v__745 >= __UNIQUE_ID_h__747 { __UNIQUE_ID_h__747 } else { if __UNIQUE_ID_v__745 <= __UNIQUE_ID_l__746 { __UNIQUE_ID_l__746 } else { __UNIQUE_ID_v__745 } };
end_pfn = __v3;
nr_absent = nr_absent.wrapping_sub(end_pfn.wrapping_sub(start_pfn));
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), nid, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn), core::ptr::null_mut::<i32>()) };
}
nr_absent
}

/// absent_pages_in_range - Return number of page frames in holes within a range
/// @start_pfn: The start PFN to start searching for holes
/// @end_pfn: The end PFN to stop searching for holes
///
/// Return: the number of pages frames in memory holes within a range.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn absent_pages_in_range(start_pfn: u64, end_pfn: u64) -> u64 {
__absent_pages_in_range(64i32, start_pfn, end_pfn)
}

/// Return the number of page frames in holes in a zone on a node
#[unsafe(link_section = ".init.text")]
fn zone_absent_pages_in_node(nid: i32, zone_type: u64, zone_start_pfn: u64, zone_end_pfn: u64) -> u64 {
// zone is empty, we don't have any absent pages
if zone_start_pfn == zone_end_pfn {
return 0;
}
__absent_pages_in_range(nid, zone_start_pfn, zone_end_pfn)
}

// Return the number of pages a zone spans in a node, including holes
// present_pages = zone_spanned_pages_in_node() - zone_absent_pages_in_node()
#[unsafe(link_section = ".init.text")]
fn zone_spanned_pages_in_node(nid: i32, zone_type: u64, node_start_pfn: u64, node_end_pfn: u64, zone_start_pfn: &mut u64, zone_end_pfn: &mut u64) -> u64 {
let zone_low: u64 = unsafe { arch_zone_lowest_possible_pfn.0[zone_type as usize] };
let zone_high: u64 = unsafe { arch_zone_highest_possible_pfn.0[zone_type as usize] };
// Get the start and end of the zone
let mut __v0: u64;
let __UNIQUE_ID_v__750: u64 = node_start_pfn;
let __UNIQUE_ID_l__751: u64 = zone_low;
let __UNIQUE_ID_h__752: u64 = zone_high;
loop {
if !(!statically_true!(__UNIQUE_ID_l__751 > __UNIQUE_ID_h__752)) {
unsafe { __compiletime_assert_753() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3!(__UNIQUE_ID_v__750, __UNIQUE_ID_l__751, __UNIQUE_ID_h__752) == 0 {
unsafe { __compiletime_assert_754() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_v__750 >= __UNIQUE_ID_h__752 { __UNIQUE_ID_h__752 } else { if __UNIQUE_ID_v__750 <= __UNIQUE_ID_l__751 { __UNIQUE_ID_l__751 } else { __UNIQUE_ID_v__750 } };
unsafe {
*(zone_start_pfn as *mut u64) = __v0;
}
let mut __v3: u64;
let __UNIQUE_ID_v__755: u64 = node_end_pfn;
let __UNIQUE_ID_l__756: u64 = zone_low;
let __UNIQUE_ID_h__757: u64 = zone_high;
loop {
if !(!statically_true!(__UNIQUE_ID_l__756 > __UNIQUE_ID_h__757)) {
unsafe { __compiletime_assert_758() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3!(__UNIQUE_ID_v__755, __UNIQUE_ID_l__756, __UNIQUE_ID_h__757) == 0 {
unsafe { __compiletime_assert_759() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_v__755 >= __UNIQUE_ID_h__757 { __UNIQUE_ID_h__757 } else { if __UNIQUE_ID_v__755 <= __UNIQUE_ID_l__756 { __UNIQUE_ID_l__756 } else { __UNIQUE_ID_v__755 } };
unsafe {
*(zone_end_pfn as *mut u64) = __v3;
}
adjust_zone_range_for_zone_movable(nid, zone_type, node_end_pfn, zone_start_pfn, zone_end_pfn);
// Check that this node has pages within the zone's required range
if (unsafe { *(zone_end_pfn as *mut u64) }) < node_start_pfn || (unsafe { *(zone_start_pfn as *mut u64) }) > node_end_pfn {
return 0;
}
// Move the zone boundaries inside the node if necessary
let mut __v6: u64;
let __UNIQUE_ID_x__760: u64 = unsafe { *(zone_end_pfn as *mut u64) };
let __UNIQUE_ID_y__761: u64 = node_end_pfn;
loop {
if __types_ok__2!(__UNIQUE_ID_x__760, __UNIQUE_ID_y__761) == 0 {
unsafe { __compiletime_assert_762() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = if __UNIQUE_ID_x__760 < __UNIQUE_ID_y__761 { __UNIQUE_ID_x__760 } else { __UNIQUE_ID_y__761 };
unsafe {
*(zone_end_pfn as *mut u64) = __v6;
}
let mut __v8: u64;
let __UNIQUE_ID_x__763: u64 = unsafe { *(zone_start_pfn as *mut u64) };
let __UNIQUE_ID_y__764: u64 = node_start_pfn;
loop {
if __types_ok__2!(__UNIQUE_ID_x__763, __UNIQUE_ID_y__764) == 0 {
unsafe { __compiletime_assert_765() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = if __UNIQUE_ID_x__763 > __UNIQUE_ID_y__764 { __UNIQUE_ID_x__763 } else { __UNIQUE_ID_y__764 };
unsafe {
*(zone_start_pfn as *mut u64) = __v8;
}
// Return the spanned pages
unsafe { *(zone_end_pfn as *mut u64) }.wrapping_sub(unsafe { *(zone_start_pfn as *mut u64) })
}

#[unsafe(link_section = ".init.text")]
fn reset_memoryless_node_totalpages(pgdat: *mut pglist_data) {
let mut z: *mut zone = unsafe { core::mem::zeroed() };
z = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone };
while z < unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset(4 as isize) {
unsafe {
(*z).zone_start_pfn = 0;
}
unsafe {
(*z).spanned_pages = 0;
}
unsafe {
(*z).present_pages = 0;
}
let __v1: *mut zone = z;
z = __v1.wrapping_offset(1 as isize);
}
unsafe {
(*pgdat).node_spanned_pages = 0;
}
unsafe {
(*pgdat).node_present_pages = 0;
}
let mut __v3: i32;
if false {
unsafe { _printk(c"\x017On node %d totalpages: 0\n".as_ptr() as *const u8, (*pgdat).node_id) };
}
__v3 = 0;
}

#[unsafe(link_section = ".init.text")]
fn calc_nr_kernel_pages() {
let mut start_pfn: u64 = 0;
let mut end_pfn: u64 = 0;
let mut start_addr: u64 = 0;
let mut end_addr: u64 = 0;
let mut u: u64 = 0;
unsafe { __next_mem_range(core::ptr::addr_of_mut!(u), 1i32.wrapping_neg(), 0u32, core::ptr::addr_of_mut!(memblock.memory), core::ptr::addr_of_mut!(memblock.reserved), core::ptr::addr_of_mut!(start_addr), core::ptr::addr_of_mut!(end_addr), core::ptr::null_mut::<i32>()) };
while u != !0 {
start_pfn = PFN_UP!(start_addr);
end_pfn = PFN_DOWN!(end_addr);
if start_pfn < end_pfn {
let __v0: u64 = unsafe { nr_all_pages };
unsafe {
nr_all_pages = __v0.wrapping_add(end_pfn.wrapping_sub(start_pfn));
}
let __v2: u64 = unsafe { nr_kernel_pages };
unsafe {
nr_kernel_pages = __v2.wrapping_add(end_pfn.wrapping_sub(start_pfn));
}
}
unsafe { __next_mem_range(core::ptr::addr_of_mut!(u), 1i32.wrapping_neg(), 0u32, core::ptr::addr_of_mut!(memblock.memory), core::ptr::addr_of_mut!(memblock.reserved), core::ptr::addr_of_mut!(start_addr), core::ptr::addr_of_mut!(end_addr), core::ptr::null_mut::<i32>()) };
}
}

#[unsafe(link_section = ".init.text")]
fn calculate_node_totalpages(pgdat: *mut pglist_data, node_start_pfn: u64, node_end_pfn: u64) {
let mut realtotalpages: u64 = 0;
let mut i: u32 = 0;
while (i as u32) < 4 {
let zone: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((i as u32) as isize);
let mut zone_start_pfn: u64 = 0;
let mut zone_end_pfn: u64 = 0;
let spanned: u64 = zone_spanned_pages_in_node(unsafe { (*pgdat).node_id }, (i as u32) as u64, node_start_pfn, node_end_pfn, unsafe { &mut (*core::ptr::addr_of_mut!(zone_start_pfn)) }, unsafe { &mut (*core::ptr::addr_of_mut!(zone_end_pfn)) });
let absent: u64 = zone_absent_pages_in_node(unsafe { (*pgdat).node_id }, (i as u32) as u64, zone_start_pfn, zone_end_pfn);
let real_size: u64 = spanned.wrapping_sub(absent);
if spanned != 0 {
unsafe {
(*zone).zone_start_pfn = zone_start_pfn;
}
} else {
unsafe {
(*zone).zone_start_pfn = 0;
}
}
unsafe {
(*zone).spanned_pages = spanned;
}
unsafe {
(*zone).present_pages = real_size;
}
realtotalpages = realtotalpages.wrapping_add(real_size);
i = (i as u32).wrapping_add(1) as u32;
}
unsafe {
(*pgdat).node_spanned_pages = node_end_pfn.wrapping_sub(node_start_pfn);
}
unsafe {
(*pgdat).node_present_pages = realtotalpages;
}
let mut __v5: i32;
if false {
unsafe { _printk(c"\x017On node %d totalpages: %lu\n".as_ptr() as *const u8, (*pgdat).node_id, realtotalpages) };
}
__v5 = 0;
}

fn pgdat_init_kcompactd(pgdat: *mut pglist_data) {
loop {
unsafe { __init_waitqueue_head(core::ptr::addr_of_mut!((*pgdat).kcompactd_wait), c"&pgdat->kcompactd_wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1322)) };
if true {
break;
}
}
}

static mut __key__1322: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn pgdat_init_internals(pgdat: *mut pglist_data) {
pgdat_resize_init(pgdat);
pgdat_kswapd_lock_init(pgdat);
pgdat_init_kcompactd(pgdat);
loop {
unsafe { __init_waitqueue_head(core::ptr::addr_of_mut!((*pgdat).kswapd_wait), c"&pgdat->kswapd_wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1336)) };
if true {
break;
}
}
loop {
unsafe { __init_waitqueue_head(core::ptr::addr_of_mut!((*pgdat).pfmemalloc_wait), c"&pgdat->pfmemalloc_wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1337)) };
if true {
break;
}
}
for i in 0i32..4i32 {
loop {
unsafe { __init_waitqueue_head(((*pgdat).reclaim_wait.as_mut_ptr() as *mut wait_queue_head).wrapping_offset(i as isize), c"&pgdat->reclaim_wait[i]".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1340)) };
if true {
break;
}
}
}
unsafe { lruvec_init(core::ptr::addr_of_mut!((*pgdat).__lruvec)) };
}

static mut __key__1336: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1337: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1340: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn zone_init_internals(zone: *mut zone, idx: u32, nid: i32, remaining_pages: u64) {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
atomic_long_set(unsafe { core::ptr::addr_of_mut!((*zone).managed_pages) }, remaining_pages as i64);
zone_set_nid(zone, nid);
unsafe {
(*zone).name = zone_names[(idx as u32) as usize] as *const u8;
}
unsafe {
(*zone).zone_pgdat = NODE_DATA!(nid);
}
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*zone).lock) });
unsafe {
*core::ptr::addr_of_mut!((*zone).lock) = *{
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
zone_seqlock_init(zone);
unsafe { zone_pcp_init(zone) };
}

#[unsafe(link_section = ".init.text")]
fn zone_init_free_lists(zone: *mut zone) {
let mut list: *mut list_head = unsafe { core::mem::zeroed() };
let mut order: u32 = 0;
while order < 11u32 {
let mut __type: u32 = 0;
loop {
let __v0: bool = if __type < 4 {
list = unsafe { (*((*zone).free_area.as_mut_ptr() as *mut free_area).wrapping_offset(order as isize)).free_list.as_mut_ptr() as *mut list_head }.wrapping_offset(__type as isize);
true
} else {
false
};
if !__v0 {
break;
}
INIT_LIST_HEAD(list);
__type = __type.wrapping_add(1);
}
order = order.wrapping_add(1);
}
order = 0;
while order < 11u32 {
unsafe {
(*((*zone).free_area.as_mut_ptr() as *mut free_area).wrapping_offset(order as isize)).nr_free = 0;
}
order = order.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn init_currently_empty_zone(zone: *mut zone, zone_start_pfn: u64, size: u64) {
let pgdat: *mut pglist_data = unsafe { (*zone).zone_pgdat };
let zone_idx: i32 = zone_idx!(zone).wrapping_add(1) as i32;
if zone_idx > unsafe { (*pgdat).nr_zones } {
unsafe {
(*pgdat).nr_zones = zone_idx;
}
}
unsafe {
(*zone).zone_start_pfn = zone_start_pfn;
}
loop {
if 2 < unsafe { mminit_loglevel } {
if false {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014mminit::memmap_init Initialising map node %d zone %lu pfns %lu -> %lu\n".as_ptr() as *const u8, (*pgdat).node_id, zone_idx!(zone) as u64, zone_start_pfn, zone_start_pfn.wrapping_add(size)) };
} else {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x017mminit::memmap_init Initialising map node %d zone %lu pfns %lu -> %lu\n".as_ptr() as *const u8, (*pgdat).node_id, zone_idx!(zone) as u64, zone_start_pfn, zone_start_pfn.wrapping_add(size)) };
}
}
if true {
break;
}
}
zone_init_free_lists(zone);
unsafe {
(*zone).initialized = 1;
}
}

fn setup_usemap(zone: *mut zone) {
}

/// When CONFIG_HUGETLB_PAGE_SIZE_VARIABLE is not set, set_pageblock_order()
/// is unused as pageblock_order is set at compile-time. See
/// include/linux/pageblock-flags.h for the values of pageblock_order based on
/// the kernel config
#[unsafe(link_section = ".init.text")]
fn set_pageblock_order() {
}

/// Set up the zone data structures
/// - init pgdat internals
/// - init all zones belonging to this node
///
/// NOTE: this function is only called during memory hotplug
#[unsafe(link_section = ".init.text")]
fn free_area_init_core(pgdat: *mut pglist_data) {
let mut j: u32 = 0;
let nid: i32 = unsafe { (*pgdat).node_id };
pgdat_init_internals(pgdat);
unsafe {
(*pgdat).per_cpu_nodestats = core::ptr::addr_of_mut!(boot_nodestats);
}
while (j as u32) < 4 {
let zone: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((j as u32) as isize);
let size: u64 = unsafe { (*zone).spanned_pages };
// Initialize zone->managed_pages as 0 , it will be reset
// when memblock allocator frees pages into buddy system.
zone_init_internals(zone, j, nid, unsafe { (*zone).present_pages });
if size != 0 {
setup_usemap(zone);
init_currently_empty_zone(zone, unsafe { (*zone).zone_start_pfn }, size);
}
j = (j as u32).wrapping_add(1) as u32;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn memmap_alloc(size: u64, align: u64, min_addr: u64, nid: i32, exact_nid: bool) -> *mut () {
let mut ptr: *mut () = unsafe { core::mem::zeroed() };
// Kmemleak will explicitly scan mem_map by traversing all valid
// `struct *page`,so memblock does not need to be added to the scan list.
if exact_nid {
ptr = unsafe { memblock_alloc_exact_nid_raw(size, align, min_addr, 1u64, nid) };
} else {
ptr = unsafe { memblock_alloc_try_nid_raw(size, align, min_addr, 1u64, nid) };
}
if !ptr.is_null() && size > 0 {
page_init_poison(ptr as *mut page, size);
}
ptr
}

fn alloc_node_mem_map(pgdat: *mut pglist_data) {
}

// get_pfn_range_for_nid - Return the start and end page frames for a node
// @nid: The nid to return the range for. If MAX_NUMNODES, the min and max PFN are returned.
// @start_pfn: Passed by reference. On return, it will have the node start_pfn.
// @end_pfn: Passed by reference. On return, it will have the node end_pfn.
//
// It returns the start and end page frame of a node based on information
// provided by memblock_set_node(). If called for a node
// with no available memory, the start and end PFNs will be 0.
#[unsafe(link_section = ".init.text")]
fn free_area_init_node(nid: i32) {
let pgdat: *mut pglist_data = NODE_DATA!(nid);
let mut start_pfn: u64 = 0;
let mut end_pfn: u64 = 0;
// pg_data_t should be reset to zero when it's allocated
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*pgdat).nr_zones }) != 0 || ((unsafe { (*pgdat).kswapd_highest_zoneidx }) as u32) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("772: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 772b - ., 3; .popsection\n# {0}", const 772 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_free_area_init_node_1, sym __slate_asm_string_free_area_init_node_2, const 1663 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("773: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 773b - ., 4; .popsection\n# {0}", const 773 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
get_pfn_range_for_nid(nid as u32, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn));
unsafe {
(*pgdat).node_id = nid;
}
unsafe {
(*pgdat).node_start_pfn = start_pfn;
}
if start_pfn != end_pfn {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x016Initmem setup node %d [mem %#018Lx-%#018Lx]\n".as_ptr() as *const u8, nid, start_pfn << 12i32, if end_pfn != 0 { (end_pfn << 12i32).wrapping_sub(1) } else { 0u64 }) };
calculate_node_totalpages(pgdat, start_pfn, end_pfn);
} else {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x016Initmem setup node %d as memoryless\n".as_ptr() as *const u8, nid) };
reset_memoryless_node_totalpages(pgdat);
}
alloc_node_mem_map(pgdat);
pgdat_set_deferred_range(pgdat);
free_area_init_core(pgdat);
lru_gen_init_pgdat(pgdat);
}

/// Any regular or high memory on that node?
#[unsafe(link_section = ".init.text")]
fn check_for_memory(pgdat: *mut pglist_data) {
let mut zone_type: u32 = 0;
while (zone_type as u32) <= 2u32 {
let zone: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((zone_type as u32) as isize);
if populated_zone(zone as *const zone) {
if IS_ENABLED!() != 0 {
node_set_state(unsafe { (*pgdat).node_id }, 2u32);
}
if (zone_type as u32) <= 2 {
node_set_state(unsafe { (*pgdat).node_id }, 2u32);
}
break;
}
zone_type = (zone_type as u32).wrapping_add(1) as u32;
}
}

/// Figure out the number of possible node ids.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn setup_nr_node_ids() {
let highest: u32 = find_last_bit((unsafe { core::ptr::addr_of_mut!(node_states[0].bits) as *mut u64 }) as *const u64, 64u64) as u32;
unsafe {
nr_node_ids = highest.wrapping_add(1);
}
}

/// Some architectures, e.g. ARC may have ZONE_HIGHMEM below ZONE_NORMAL. For
/// such cases we allow max_zone_pfn sorted in the descending order
fn arch_has_descending_max_zone_pfns() -> bool {
IS_ENABLED!() != 0 && IS_ENABLED!() == 0
}

#[unsafe(link_section = ".init.text")]
fn set_high_memory() {
let highmem: u64 = unsafe { memblock_end_of_DRAM() };
// Some architectures (e.g. ARM) set high_memory very early and
// use it in arch setup code.
// If an architecture already set high_memory don't overwrite it
if !unsafe { high_memory }.is_null() {
return;
}
unsafe {
high_memory = (phys_to_virt(highmem.wrapping_sub(1)) as *const u8).wrapping_offset(1 as isize) as *mut ();
}
}

// free_area_init - Initialise all pg_data_t and zone data
//
// This will call free_area_init_node() for each active node in the system.
// Using the page ranges provided by memblock_set_node(), the size of each
// zone in each node and their holes is calculated. If the maximum PFN
// between two adjacent zones match, it is assumed that the zone is empty.
// For example, if arch_max_dma_pfn == arch_max_dma32_pfn, it is assumed
// that arch_max_dma32_pfn has no pages. It is also assumed that a zone
// starts where the previous one ended. For example, ZONE_DMA32 starts
// at arch_max_dma_pfn.
#[unsafe(link_section = ".init.text")]
fn free_area_init() {
let mut max_zone_pfn: __SlateAlign16<[u64; 4]> = __SlateAlign16({
let mut __t0: [u64; 4] = unsafe { core::mem::zeroed() };
__t0[0] = 0u64;
__t0
});
let mut start_pfn: u64 = 0;
let mut end_pfn: u64 = 0;
let mut i: i32 = 0;
let mut nid: i32 = 0;
let mut zone: i32 = 0;
let mut descending: bool = false;
unsafe { arch_zone_limits_init(max_zone_pfn.0.as_mut_ptr() as *mut u64) };
start_pfn = PHYS_PFN!(unsafe { memblock_start_of_DRAM() });
descending = arch_has_descending_max_zone_pfns();
while i < 4 {
if descending {
zone = 4i32.wrapping_sub(i).wrapping_sub(1);
} else {
zone = i;
}
if zone != 3 {
let mut __v1: u64;
let __UNIQUE_ID_x__774: u64 = max_zone_pfn.0[zone as usize];
let __UNIQUE_ID_y__775: u64 = start_pfn;
while false {
}
__v1 = if __UNIQUE_ID_x__774 > __UNIQUE_ID_y__775 { __UNIQUE_ID_x__774 } else { __UNIQUE_ID_y__775 };
end_pfn = __v1;
unsafe {
arch_zone_lowest_possible_pfn.0[zone as usize] = start_pfn;
}
unsafe {
arch_zone_highest_possible_pfn.0[zone as usize] = end_pfn;
}
start_pfn = end_pfn;
}
i = i.wrapping_add(1);
}
// Find the PFNs that ZONE_MOVABLE begins at in each node
find_zone_movable_pfns_for_nodes();
// Print out the zone ranges
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x016Zone ranges:\n".as_ptr() as *const u8) };
i = 0;
while i < 4 {
if i != 3 {
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x016  %-8s ".as_ptr() as *const u8, zone_names[i as usize]) };
if (unsafe { arch_zone_lowest_possible_pfn.0[i as usize] }) == unsafe { arch_zone_highest_possible_pfn.0[i as usize] } {
let mut __v10: i32;
while false {
}
__v10 = unsafe { _printk(c"\x01cempty\n".as_ptr() as *const u8) };
} else {
let mut __v12: i32;
while false {
}
__v12 = unsafe { _printk(c"\x01c[mem %#018Lx-%#018Lx]\n".as_ptr() as *const u8, arch_zone_lowest_possible_pfn.0[i as usize] << 12i32, (arch_zone_highest_possible_pfn.0[i as usize] << 12i32).wrapping_sub(1)) };
}
}
i = i.wrapping_add(1);
}
// Print out the PFNs ZONE_MOVABLE begins at in each node
let mut __v16: i32;
while false {
}
__v16 = unsafe { _printk(c"\x016Movable zone start for each node\n".as_ptr() as *const u8) };
i = 0;
while i < 64i32 {
if (unsafe { zone_movable_pfn.0[i as usize] }) != 0 {
let mut __v18: i32;
while false {
}
__v18 = unsafe { _printk(c"\x016  Node %d: %#018Lx\n".as_ptr() as *const u8, i, zone_movable_pfn.0[i as usize] << 12i32) };
}
i = i.wrapping_add(1);
}
// Print out the early node map.
let mut __v23: i32;
while false {
}
__v23 = unsafe { _printk(c"\x016Early memory node ranges\n".as_ptr() as *const u8) };
i = 1i32.wrapping_neg();
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn), core::ptr::addr_of_mut!(nid)) };
while i >= 0 {
let mut __v25: i32;
while false {
}
__v25 = unsafe { _printk(c"\x016  node %3d: [mem %#018Lx-%#018Lx]\n".as_ptr() as *const u8, nid, start_pfn << 12i32, (end_pfn << 12i32).wrapping_sub(1)) };
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::addr_of_mut!(start_pfn), core::ptr::addr_of_mut!(end_pfn), core::ptr::addr_of_mut!(nid)) };
}
// Initialise every node
mminit_verify_pageflags_layout();
setup_nr_node_ids();
set_pageblock_order();
nid = first_node!(node_states[0]) as i32;
while nid < 64i32 {
// If an architecture has not allocated node data for
// this node, presume the node is memoryless or offline.
if NODE_DATA!(nid).is_null() {
unsafe { alloc_offline_node_data(nid) };
}
let pgdat: *mut pglist_data = NODE_DATA!(nid);
free_area_init_node(nid);
// No sysfs hierarchy will be created via register_node()
// for memory-less node because here it's not marked as N_MEMORY
// and won't be set online later. The benefit is userspace
// program won't be confused by sysfs files/directories of
// memory-less node. The pgdat will get fully initialized by
// hotadd_init_pgdat() when memory is hotplugged into this node.
if (unsafe { (*pgdat).node_present_pages }) != 0 {
node_set_state(nid, 3u32);
check_for_memory(pgdat);
}
nid = next_node!(nid, node_states[0]) as i32;
}
calc_nr_kernel_pages();
// disable hash distribution for systems with a single node
fixup_hashdist();
set_high_memory();
}

/// node_map_pfn_alignment - determine the maximum internode alignment
///
/// This function should be called after node map is populated and sorted.
/// It calculates the maximum power of two alignment which can distinguish
/// all the nodes.
///
/// For example, if all nodes are 1GiB and aligned to 1GiB, the return value
/// would indicate 1GiB alignment with (1 << (30 - PAGE_SHIFT)).  If the
/// nodes are shifted by 256MiB, 256MiB.  Note that if only the last node is
/// shifted, 1GiB is enough and this function will indicate so.
///
/// This is used to test whether pfn -> nid mapping of the chosen memory
/// model has fine enough granularity to avoid incorrect mapping for the
/// populated node map.
///
/// Return: the determined alignment in pfn's.  0 if there is no alignment
/// requirement (single node).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn node_map_pfn_alignment() -> u64 {
let mut accl_mask: u64 = 0;
let mut last_end: u64 = 0;
let mut start: u64 = 0;
let mut end: u64 = 0;
let mut mask: u64 = 0;
let mut last_nid: i32 = 1i32.wrapping_neg();
let mut i: i32 = 1i32.wrapping_neg();
let mut nid: i32 = 0;
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::addr_of_mut!(start), core::ptr::addr_of_mut!(end), core::ptr::addr_of_mut!(nid)) };
while i >= 0 {
if start == 0 || last_nid < 0 || last_nid == nid {
last_nid = nid;
last_end = end;
} else {
// Start with a mask granular enough to pin-point to the
// start pfn and tick off bits one-by-one until it becomes
// too coarse to separate the current node from the last.
mask = !(1i32 << __ffs!(start)).wrapping_sub(1) as u64;
while mask != 0 && last_end <= start & mask << 1i32 {
mask <<= 1i32;
}
// accumulate all internode masks
accl_mask |= mask;
}
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::addr_of_mut!(start), core::ptr::addr_of_mut!(end), core::ptr::addr_of_mut!(nid)) };
}
// convert mask to number of pages
(!accl_mask).wrapping_add(1)
}

/// Check if a PFN range intersects multiple zones on one or more
/// NUMA nodes. Specify the @nid argument if it is known that this
/// PFN range is on one node, NUMA_NO_NODE otherwise.
#[unsafe(no_mangle)]
extern "C" fn pfn_range_intersects_zones(nid: i32, start_pfn: u64, nr_pages: u64) -> bool {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut izone: *mut zone = core::ptr::null_mut::<zone>();
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
let __v0: bool = nid != 1i32.wrapping_neg() && zone_to_nid(zone as *const zone) != nid;
if !__v0 {
if zone_intersects(zone as *const zone, start_pfn, nr_pages) {
if !izone.is_null() {
return true;
}
izone = zone;
}
}
zone = unsafe { next_zone(zone) };
}
false
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn page_alloc_init_late() {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut nid: i32 = 0;
unsafe { hugetlb_bootmem_struct_page_init() };
// Accounting of total+free memory is stable at this point.
mem_init_print_info();
unsafe { buffer_init() };
// Discard memblock private memory
unsafe { memblock_discard() };
nid = first_node!(node_states[3]) as i32;
while nid < 64i32 {
shuffle_free_memory(NODE_DATA!(nid));
nid = next_node!(nid, node_states[3]) as i32;
}
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
set_zone_contiguous(zone);
}
zone = unsafe { next_zone(zone) };
}
// Initialize page ext after all struct pages are initialized.
if unsafe { deferred_struct_pages } {
page_ext_init();
}
unsafe { page_alloc_sysctl_init() };
}

// Adaptive scale is meant to reduce sizes of hash tables on large memory
// machines. As memory size is increased the scale is also increased but at
// slower pace.  Starting from ADAPT_SCALE_BASE (64G), every time memory
// quadruples the scale is increased by one, which means the size of hash table
// only doubles, instead of quadrupling as well.
// Because 32-bit systems cannot have large physical memory, where this scaling
// makes sense, it is disabled on such platforms.
// allocate a large system hash table from bootmem
// - it is assumed that the hash table must contain an exact power-of-2
//   quantity of entries
// - limit is the number of hash buckets, not the total allocation size
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn memblock_free_pages(pfn: u64, order: u32) {
let page: *mut page = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pfn as isize);
if IS_ENABLED!() != 0 {
let nid: i32 = early_pfn_to_nid(pfn);
if !early_page_initialised(pfn, nid) {
return;
}
}
if !kmsan_memblock_free_pages(page, order) {
// KMSAN will take care of these pages.
return;
}
// pages were reserved and not allocated
clear_page_tag_ref(page);
unsafe { __free_pages_core(page, order, 0u32) };
}

#[unsafe(no_mangle)]
static mut init_on_alloc: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_on_alloc_783: *mut () = (unsafe { core::ptr::addr_of_mut!(init_on_alloc) }) as *mut ();

#[unsafe(no_mangle)]
static mut init_on_free: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_on_free_784: *mut () = (unsafe { core::ptr::addr_of_mut!(init_on_free) }) as *mut ();

#[unsafe(link_section = ".data..read_mostly")]
static mut _init_on_alloc_enabled_early: bool = 0i32 != 0i32;

#[unsafe(link_section = ".init.text")]
extern "C" fn early_init_on_alloc(buf: *mut u8) -> i32 {
unsafe { kstrtobool(buf as *const u8, core::ptr::addr_of_mut!(_init_on_alloc_enabled_early)) }
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_early_init_on_alloc: [u8; 14] = [105, 110, 105, 116, 95, 111, 110, 95, 97, 108, 108, 111, 99, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_early_init_on_alloc: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_early_init_on_alloc) as *const u8 };
__t0.setup_func = Some(early_init_on_alloc);
__t0.early = 1i32;
__t0
};

#[unsafe(link_section = ".data..read_mostly")]
static mut _init_on_free_enabled_early: bool = 0i32 != 0i32;

#[unsafe(link_section = ".init.text")]
extern "C" fn early_init_on_free(buf: *mut u8) -> i32 {
unsafe { kstrtobool(buf as *const u8, core::ptr::addr_of_mut!(_init_on_free_enabled_early)) }
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_early_init_on_free: [u8; 13] = [105, 110, 105, 116, 95, 111, 110, 95, 102, 114, 101, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_early_init_on_free: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_early_init_on_free) as *const u8 };
__t0.setup_func = Some(early_init_on_free);
__t0.early = 1i32;
__t0
};

#[unsafe(no_mangle)]
static mut check_pages_enabled: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

#[unsafe(link_section = ".init.data")]
static mut check_pages_enabled_early: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
extern "C" fn early_check_pages(buf: *mut u8) -> i32 {
unsafe { kstrtobool(buf as *const u8, core::ptr::addr_of_mut!(check_pages_enabled_early)) }
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_early_check_pages: [u8; 12] = [99, 104, 101, 99, 107, 95, 112, 97, 103, 101, 115, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_early_check_pages: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_early_check_pages) as *const u8 };
__t0.setup_func = Some(early_check_pages);
__t0.early = 1i32;
__t0
};

/// Enable static keys related to various memory debugging and hardening options.
/// Some override others, and depend on early params that are evaluated in the
/// order of appearance. So we need to first gather the full picture of what was
/// enabled, and then make decisions.
#[unsafe(link_section = ".init.text")]
fn mem_debugging_and_hardening_init() {
let page_poisoning_requested: bool = false;
let mut want_check_pages: bool = unsafe { check_pages_enabled_early };
if ((unsafe { _init_on_alloc_enabled_early }) || unsafe { _init_on_free_enabled_early }) && page_poisoning_requested {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016mem auto-init: CONFIG_PAGE_POISONING is on, will take precedence over init_on_alloc and init_on_free\n".as_ptr() as *const u8) };
unsafe {
_init_on_alloc_enabled_early = false;
}
unsafe {
_init_on_free_enabled_early = false;
}
}
if unsafe { _init_on_alloc_enabled_early } {
want_check_pages = true;
static_branch_enable!(unsafe { core::ptr::addr_of_mut!(init_on_alloc) });
} else {
static_branch_disable!(unsafe { core::ptr::addr_of_mut!(init_on_alloc) });
}
if unsafe { _init_on_free_enabled_early } {
want_check_pages = true;
static_branch_enable!(unsafe { core::ptr::addr_of_mut!(init_on_free) });
} else {
static_branch_disable!(unsafe { core::ptr::addr_of_mut!(init_on_free) });
}
if IS_ENABLED!() != 0 && ((unsafe { _init_on_alloc_enabled_early }) || unsafe { _init_on_free_enabled_early }) {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016mem auto-init: please make sure init_on_alloc and init_on_free are disabled when running KMSAN\n".as_ptr() as *const u8) };
}
// Any page debugging or hardening option also enables sanity checking
// of struct pages being allocated or freed. With CONFIG_DEBUG_VM it's
// enabled already.
if IS_ENABLED!() == 0 && want_check_pages {
static_branch_enable!(unsafe { core::ptr::addr_of_mut!(check_pages_enabled) });
}
}

/// Report memory auto-initialization states for this boot.
#[unsafe(link_section = ".init.text")]
fn report_meminit() {
let stack: *const u8 = c"all(zero)".as_ptr() as *const u8;
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016mem auto-init: stack:%s, heap alloc:%s, heap free:%s\n".as_ptr() as *const u8, stack, str_on_off(want_init_on_alloc(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))), str_on_off(want_init_on_free())) };
if want_init_on_free() {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016mem auto-init: clearing system memory may take some time...\n".as_ptr() as *const u8) };
}
}

#[unsafe(link_section = ".init.text")]
fn mem_init_print_info() {
let physpages: u64 = get_num_physpages();
let mut codesize: u64 = ((unsafe { (core::ptr::addr_of_mut!(_etext) as *mut u8).offset_from((core::ptr::addr_of_mut!(_stext) as *mut u8) as *mut u8) }) as i64) as u64;
let mut datasize: u64 = ((unsafe { (core::ptr::addr_of_mut!(_edata) as *mut u8).offset_from((core::ptr::addr_of_mut!(_sdata) as *mut u8) as *mut u8) }) as i64) as u64;
let rosize: u64 = ((unsafe { (core::ptr::addr_of_mut!(__end_rodata) as *mut u8).offset_from((core::ptr::addr_of_mut!(__start_rodata) as *mut u8) as *mut u8) }) as i64) as u64;
let bss_size: u64 = ((unsafe { (core::ptr::addr_of_mut!(__bss_stop) as *mut u8).offset_from((core::ptr::addr_of_mut!(__bss_start) as *mut u8) as *mut u8) }) as i64) as u64;
let mut init_code_size: u64 = 0;
let mut init_data_size: u64 = ((unsafe { (core::ptr::addr_of_mut!(__init_end) as *mut u8).offset_from((core::ptr::addr_of_mut!(__init_begin) as *mut u8) as *mut u8) }) as i64) as u64;
init_code_size = ((unsafe { (core::ptr::addr_of_mut!(_einittext) as *mut u8).offset_from((core::ptr::addr_of_mut!(_sinittext) as *mut u8) as *mut u8) }) as i64) as u64;
// Detect special cases and adjust section sizes accordingly:
// 1) .init.* may be embedded into .data sections
// 2) .init.text.* may be out of [__init_begin, __init_end],
//    please refer to arch/tile/kernel/vmlinux.lds.S.
// 3) .rodata.* may be embedded into .text or .data sections.
adj_init_size!(__init_begin, __init_end, init_data_size, _sinittext, init_code_size);
adj_init_size!(_stext, _etext, codesize, _sinittext, init_code_size);
adj_init_size!(_sdata, _edata, datasize, __init_begin, init_data_size);
adj_init_size!(_stext, _etext, codesize, __start_rodata, rosize);
adj_init_size!(_sdata, _edata, datasize, __start_rodata, rosize);
let mut __v15: i32;
while false {
}
__v15 = unsafe { _printk(c"\x016Memory: %luK/%luK available (%luK kernel code, %luK rwdata, %luK rodata, %luK init, %luK bss, %luK reserved, %luK cma-reserved)\n".as_ptr() as *const u8, K!(nr_free_pages!()), K!(physpages), codesize / 1024, datasize / 1024, rosize / 1024, init_data_size.wrapping_add(init_code_size) / 1024, bss_size / 1024, K!(physpages.wrapping_sub(totalram_pages()).wrapping_sub(totalcma_pages)), K!(unsafe { totalcma_pages })) };
}

/// architectures that __HAVE_COLOR_ZERO_PAGE must define this function
#[unsafe(no_mangle)]
#[linkage = "weak"]
#[unsafe(link_section = ".init.text")]
extern "C" fn arch_setup_zero_pages() {
unsafe {
__zero_page = virt_to_page!(core::ptr::addr_of!(empty_zero_page.0) as *const u8);
}
}

#[unsafe(link_section = ".init.text")]
fn init_zero_page_pfn() {
arch_setup_zero_pages();
unsafe {
zero_page_pfn = (ZERO_PAGE!(0i32).offset_from(vmemmap_base as *mut page) as i64) as u64;
}
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
#[unsafe(link_section = ".init.text")]
extern "C" fn arch_mm_preinit() {
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
#[unsafe(link_section = ".init.text")]
extern "C" fn mem_init() {
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn mm_core_init_early() {
kho_memory_init_early();
free_area_init();
hugetlb_cma_reserve();
unsafe { hugetlb_bootmem_alloc() };
unsafe { sparse_init() };
memmap_init();
}

// Set up kernel memory allocators
static __slate_asm_string_cmdline_parse_core_1: [u8; 1] = *b"\0";

static __slate_asm_string_cmdline_parse_core_2: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/mm_init.c\0";

static __slate_asm_string_cmdline_parse_core_6: [u8; 1] = *b"\0";

static __slate_asm_string_cmdline_parse_core_7: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/mm_init.c\0";

static __slate_asm_string_free_area_init_node_1: [u8; 1] = *b"\0";

static __slate_asm_string_free_area_init_node_2: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/mm_init.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_mminit_verify_pageflags_layout_21: [u8; 1] = *b"\0";

static __slate_asm_string_mminit_verify_pageflags_layout_22: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/mm_init.c\0";

static __slate_asm_string_mminit_verify_pageflags_layout_28: [u8; 1] = *b"\0";

static __slate_asm_string_mminit_verify_pageflags_layout_29: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/mm_init.c\0";

static __slate_asm_string_mminit_verify_pageflags_layout_33: [u8; 1] = *b"\0";

static __slate_asm_string_mminit_verify_pageflags_layout_34: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/mm_init.c\0";
