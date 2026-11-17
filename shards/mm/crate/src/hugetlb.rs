
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
//! Generic hugetlb support.
//! (C) Nadia Yvette Chambers, April 2004
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
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! # chars in a path name including nul
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! array_index_mask_nospec() - generate a mask that is ~0UL when the
//! 	bounds check succeeds and 0 otherwise
//! @idx: array element index
//! @sz: number of elements in array
//!
//! Returns:
//!     0 - (@idx < @sz)
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! This counts to 15. Any more, it will return 16th argument.
//! Operation not permitted
//! Value too large for defined data type
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
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! Defines x86 CPU feature bits
//! 64bit FS base
//! Needs two entries
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! writeable
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
//! used to decode ioctl numbers..
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! The size of each LDT entry.
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
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
//! signal pending
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
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
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! Convenience macros for the sake of set_current_state:
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
//! Uprobes on this MM assume 32-bit code
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! vmemmap optimization (like HVO) is only possible for page orders that fill
//! two or more pages with struct pages.
//! Helper macro to avoid gfp flags if they are the default one
//! MXCSR will be initialized
//! Resource type
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
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! array_index_nospec - sanitize an array index after a bounds check
//!
//! For a code sequence like:
//!
//!     if (index < size) {
//!         index = array_index_nospec(index, size);
//!         val = array[index];
//!     }
//!
//! ...if the CPU speculates past the bounds check then
//! array_index_nospec() will clamp the index within the range of [0,
//! size).
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
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
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
//! If more than 16 keys are ever supported, a thorough audit
//! will be necessary to ensure that the types that store key
//! numbers and masks have sufficient capacity.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Set global bits for all 4 regs
//! Define bits that are always set to 1 in DR7, only bit 10 is
//! architecturally reserved to '1'.
//!
//! This is also the init/reset value for DR7.
//! Replace with "lkgs %di" once binutils support LKGS instruction
//! Macro to enforce the same ordering and stack sizes
//! See the comment in asm-generic/io.h about ioremap_np().
//! 8237 DMA controllers
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
//! For HugeTLB page, there are more metadata to save in the struct page. But
//! the head struct page cannot meet our needs, so we have to abuse other tail
//! struct page to store the metadata.
//! Monochrome Text Display
//! number of edd_info structs starting at EDDBUF
//! This is the maximum number of entries in struct boot_params::e820_table
//! (the zeropage), which is part of the x86 boot protocol ABI:
//! Like buddy_order(), but for callers who cannot afford to hold the zone lock.
//! PageBuddy() should be checked first by the caller to minimize race window,
//! and invalid values must be handled gracefully.
//!
//! READ_ONCE is used so that if the caller assigns the result into a local
//! variable and e.g. tests it for valid range before using, the compiler cannot
//! decide to remove the variable and inline the page_private(page) multiple
//! times, potentially observing different values in the tests and the actual
//! use of the result.
//! Reserve one vmemmap page, all vmemmap addresses are mapped to it. See
//! Documentation/mm/vmemmap_dedup.rst.
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
// Make the optimizer believe the variable can be manipulated arbitrarily.
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
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
// Prevent speculative execution past this barrier.
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// Operation not supported on transport endpoint
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "hypervisor" Running on a hypervisor
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
// Like MOV_GS except MSR_KERNEL_GS_BASE = GS.base
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
// Needs two entries
// Number of entries in the GDT table:
// Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
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
// N 32-bit words worth of info
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
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
// Resource extended types
// No address assigned yet
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
// Generate SD flag bits
// Implementation specific max size
// Internal flags that share the struct mempolicy flags word with
// "mode flags".  These flags are allocated from bit 0 up, as they
// are never OR'ed into the mode in mempolicy API arguments.
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
// These special macros can be used to get or set a debugging register
// 8 bit slave DMA, channels 0..3
// 16 bit master DMA, ch 4(=slave input)..7
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// DMA page registers
// for_each_mem_pfn_range - early memory pfn range iterator
// @i: an integer used as loop variable
// @nid: node selector, %MAX_NUMNODES for all nodes
// @p_start: ptr to ulong for start pfn of the range, can be %NULL
// @p_end: ptr to ulong for end pfn of the range, can be %NULL
// @p_nid: ptr to int for nid of the range, can be %NULL
//
// Walks over configured memory ranges.
// We are using top down, so it is safe to use 0 here
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
// cgroup_subsys_enabled - fast test on whether a subsys is enabled
// @ss: subsystem in question
// task_css_check - obtain css for (task, subsys) w/ extra access conds
// @task: the target task
// @subsys_id: the target subsystem ID
// @__c: extra condition expression to be passed to rcu_dereference_check()
//
// Return the cgroup_subsys_state for the (@task, @subsys_id) pair.  The
// synchronization rules are the same as task_css_set_check().
// __put_unaligned_t - write an unaligned value to memory.
// @type:	the type of the value to store.
// @val:	the value to store.
// @ptr:	the pointer to store to.
//
// Use memcpy to affect an unaligned type sized store avoiding undefined
// behavior from approaches like type punning that require -fno-strict-aliasing
// in order to be correct. The void* cast silences ubsan warnings.
// Watch out for PAE architecture, which has an unsigned long, and might not
// have enough bits to store all physical address and flags. So far we have
// enough room for all our flags.
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
// Convert GFP flags to their corresponding migrate type
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
macro_rules! ALIGN_DOWN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x.wrapping_sub($a.wrapping_sub(1)), $a)
};
}

/// @a is a power of 2 value
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

/// @a is a power of 2 value
macro_rules! ALIGN__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL__2!($x, $a)
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

macro_rules! GENMASK_INPUT_CHECK__4 {
() => {
core::mem::size_of::<__SlateRecord_page_flags_h_276>() as i32
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

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__4 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__4!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
};
}

macro_rules! GENMASK__3 {
($h:expr, $l:expr) => {
GENMASK_TYPE__4!(u64, $h, $l)
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

macro_rules! NODEMASK_FREE {
() => {
while false {
}
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

macro_rules! PF_ANY {
($page:expr) => {
PF_POISONED_CHECK__3!($page)
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
macro_rules! PF_POISONED_CHECK__3 {
($page:expr) => {
{
let mut __v: *mut page;
VM_BUG_ON_PGFLAGS!();
__v = $page;
__v
}
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
macro_rules! PF_POISONED_CHECK__2 {
($page:expr) => {
let mut __v: *const page;
VM_BUG_ON_PGFLAGS!();
__v = $page;
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

macro_rules! REQUIRED_MASK_BIT_SET {
($x:expr) => {
(if ($x) >> 5i32 == 0 { 117481833 } else { if ($x) >> 5i32 == 1 { 536870912 } else { if ($x) >> 5i32 == 3 { 36732928 } else { 0 } } }) & 1u32 << (($x) & 31)
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VM_BUG_ON_FOLIO {
() => {
VM_BUG_ON!();
};
}

macro_rules! VM_BUG_ON_PAGE {
() => {
VM_BUG_ON!();
};
}

macro_rules! VM_BUG_ON_PGFLAGS {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_BUG_ON_VMA {
() => {
VM_BUG_ON!();
};
}

macro_rules! VM_BUG_ON {
() => {
BUILD_BUG_ON_INVALID!();
};
}

/// Encode hstate index for a hwpoisoned large page
macro_rules! VM_FAULT_SET_HINDEX {
($x:expr) => {
(($x) << 16i32) as u32
};
}

macro_rules! VM_WARN_ON_FOLIO {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_ONCE_VMA {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_ONCE {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON {
() => {
BUILD_BUG_ON_INVALID!();
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

macro_rules! _AC__3 {
() => {
16045481047390945280u64
};
}

macro_rules! _AT {
($T:ty, $X:expr) => {
($X) as $T
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

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u32).wrapping_sub(1))
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

macro_rules! __WRITE_ONCE__5 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut __SlateRecord_pgtable_64_types_h_21, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__8 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, ($val) as u64) };
if true {
break;
}
}
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

macro_rules! __WRITE_ONCE__11 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut __SlateRecord_pgtable_types_h_368, $val) };
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

macro_rules! __WRITE_ONCE__4 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, $val) };
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

macro_rules! __clear_bit {
($nr:expr, $addr:expr) => {
let __v: bool = true && false && (($addr) as u64) != 0;
if __v && false {
generic___clear_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___clear_bit(($nr) as u64, ($addr) as *mut u64);
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

/// __jhash_final - final mixing of 3 32-bit values (a,b,c) into c
macro_rules! __jhash_final {
($a:expr, $b:expr, $c:expr) => {
let __v: u32 = $c;
let __v2: u32 = __v ^ $b;
$c = __v2;
let __v3: u32 = rol32($b, 14u32);
let __v4: u32 = $c;
let __v5: u32 = __v4.wrapping_sub(__v3);
$c = __v5;
let __v6: u32 = $a;
let __v7: u32 = __v6 ^ $c;
$a = __v7;
let __v8: u32 = rol32($c, 11u32);
let __v9: u32 = $a;
let __v10: u32 = __v9.wrapping_sub(__v8);
$a = __v10;
let __v11: u32 = $b;
let __v12: u32 = __v11 ^ $a;
$b = __v12;
let __v13: u32 = rol32($a, 25u32);
let __v14: u32 = $b;
let __v15: u32 = __v14.wrapping_sub(__v13);
$b = __v15;
let __v16: u32 = $c;
let __v17: u32 = __v16 ^ $b;
$c = __v17;
let __v18: u32 = rol32($b, 16u32);
let __v19: u32 = $c;
let __v20: u32 = __v19.wrapping_sub(__v18);
$c = __v20;
let __v21: u32 = $a;
let __v22: u32 = __v21 ^ $c;
$a = __v22;
let __v23: u32 = rol32($c, 4u32);
let __v24: u32 = $a;
let __v25: u32 = __v24.wrapping_sub(__v23);
$a = __v25;
let __v26: u32 = $b;
let __v27: u32 = __v26 ^ $a;
$b = __v27;
let __v28: u32 = rol32($a, 14u32);
let __v29: u32 = $b;
let __v30: u32 = __v29.wrapping_sub(__v28);
$b = __v30;
let __v31: u32 = $c;
let __v32: u32 = __v31 ^ $b;
$c = __v32;
let __v33: u32 = rol32($b, 24u32);
let __v34: u32 = $c;
let __v35: u32 = __v34.wrapping_sub(__v33);
$c = __v35;
};
}

/// __jhash_mix - mix 3 32-bit values reversibly.
macro_rules! __jhash_mix {
($a:expr, $b:expr, $c:expr) => {
let __v: u32 = $a;
let __v2: u32 = __v.wrapping_sub($c);
$a = __v2;
let __v3: u32 = rol32($c, 4u32);
let __v4: u32 = $a;
let __v5: u32 = __v4 ^ __v3;
$a = __v5;
let __v6: u32 = $c;
let __v7: u32 = __v6.wrapping_add($b);
$c = __v7;
let __v8: u32 = $b;
let __v9: u32 = __v8.wrapping_sub($a);
$b = __v9;
let __v10: u32 = rol32($a, 6u32);
let __v11: u32 = $b;
let __v12: u32 = __v11 ^ __v10;
$b = __v12;
let __v13: u32 = $a;
let __v14: u32 = __v13.wrapping_add($c);
$a = __v14;
let __v15: u32 = $c;
let __v16: u32 = __v15.wrapping_sub($b);
$c = __v16;
let __v17: u32 = rol32($b, 8u32);
let __v18: u32 = $c;
let __v19: u32 = __v18 ^ __v17;
$c = __v19;
let __v20: u32 = $b;
let __v21: u32 = __v20.wrapping_add($a);
$b = __v21;
let __v22: u32 = $a;
let __v23: u32 = __v22.wrapping_sub($c);
$a = __v23;
let __v24: u32 = rol32($c, 16u32);
let __v25: u32 = $a;
let __v26: u32 = __v25 ^ __v24;
$a = __v26;
let __v27: u32 = $c;
let __v28: u32 = __v27.wrapping_add($b);
$c = __v28;
let __v29: u32 = $b;
let __v30: u32 = __v29.wrapping_sub($a);
$b = __v30;
let __v31: u32 = rol32($a, 19u32);
let __v32: u32 = $b;
let __v33: u32 = __v32 ^ __v31;
$b = __v33;
let __v34: u32 = $a;
let __v35: u32 = __v34.wrapping_add($c);
$a = __v35;
let __v36: u32 = $c;
let __v37: u32 = __v36.wrapping_sub($b);
$c = __v37;
let __v38: u32 = rol32($b, 4u32);
let __v39: u32 = $c;
let __v40: u32 = __v39 ^ __v38;
$c = __v40;
let __v41: u32 = $b;
let __v42: u32 = __v41.wrapping_add($a);
$b = __v42;
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__14 {
() => {
((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__22 {
() => {
((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__24 {
() => {
((const { core::mem::size_of::<__SlateRecord_types_h_190>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_types_h_190>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_types_h_190>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_types_h_190>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__9 {
() => {
((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! __pte {
($x:expr) => {
native_make_pte($x)
};
}

macro_rules! __pud {
($x:expr) => {
native_make_pud($x)
};
}

macro_rules! __ratelimit__15 {
($state:expr) => {
unsafe { ___ratelimit($state, c"hugetlb_no_page".as_ptr() as *const u8) }
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

/// The following macros are non-atomic versions of their non-underscored
/// counterparts.
macro_rules! __set_bit {
($nr:expr, $addr:expr) => {
let __v: bool = true && false && (($addr) as u64) != 0;
if __v && false {
generic___set_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___set_bit(($nr) as u64, ($addr) as *mut u64);
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
macro_rules! __sign_use__5 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
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
macro_rules! __sign_use__6 {
($ux:expr) => {
if is_signed_type__4!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i64>() as u64) < 4) as i32)) }
};
}

/// Atomic operations are already serializing on x86
macro_rules! __smp_mb__before_atomic {
() => {
while false {
}
};
}

macro_rules! __swp_entry_to_pte {
($x:expr) => {
__pte!($x.val)
};
}

/// Shift up (to get rid of type), then down to get value
macro_rules! __swp_offset {
($x:expr) => {
!$x.val << 5i32 >> 14i32
};
}

/// Extract the high bits for type
macro_rules! __swp_type {
($x:expr) => {
$x.val >> 59i32
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

macro_rules! __types_ok__5 {
($ux:expr, $uy:expr) => {
__sign_use__5!($ux) & __sign_use__5!($uy)
};
}

macro_rules! __types_ok__7 {
($ux:expr, $uy:expr) => {
__sign_use__4!($ux) & __sign_use__2!($uy)
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

macro_rules! __types_ok__6 {
($ux:expr, $uy:expr) => {
__sign_use__6!($ux) & __sign_use__6!($uy)
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

macro_rules! __vma_lockdep_release_read {
() => {
lock_release!();
};
}

macro_rules! _static_cpu_has {
($bit:expr) => {
__static_cpu_has__const1(($bit) as u16) as i32
};
}

macro_rules! _static_cpu_has__4 {
($bit:expr) => {
__static_cpu_has__const2(($bit) as u16) as i32
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! alloc_hooks__3 {
($_do_alloc:expr) => {
{
let mut __v: *mut page;
let mut __v2: *mut page;
let mut __v3: *mut page = unsafe { core::mem::zeroed() };
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

macro_rules! arch_atomic_fetch_sub {
($i:expr, $v:expr) => {
arch_atomic_fetch_add($i.wrapping_neg(), $v)
};
}

macro_rules! bitmap_size {
($nbits:expr) => {
ALIGN__2!($nbits, 64i32) / 8
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

macro_rules! compound_head {
($page:expr) => {
_compound_head($page) as *const page
};
}

macro_rules! cond_resched__10 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 3522i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__16 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 1540i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__29 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 3861i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__31 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 2285i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__36 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 1590i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__46 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 2677i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__50 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 3327i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__56 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 3438i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__65 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 3508i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__81 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 1988i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__92 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 1658i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__101 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 1615i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__107 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/hugetlb.c".as_ptr() as *const u8, 6645i32, 0u32);
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
macro_rules! container_of__9 {
($ptr:expr) => {
{
let mut __v: *mut huge_bootmem_page;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__23!() as isize)) }) as *mut ()) as *mut huge_bootmem_page;
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
macro_rules! container_of__31 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__32!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__2 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__15 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__20!() as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__75!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__55 {
($ptr:expr) => {
{
let mut __v: *mut file_region;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__66!() as isize)) }) as *mut ()) as *mut file_region;
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
macro_rules! container_of__4 {
($ptr:expr) => {
{
let mut __v: *mut folio;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut folio;
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

/// This is the default CPU features testing macro to use in code.
///
/// It is for detection of features which need kernel infrastructure to be
/// used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
/// if you want true runtime testing of CPU features, like in hypervisor code
/// where you are supporting a possible guest feature where host support for it
/// is not relevant.
macro_rules! cpu_feature_enabled__5 {
($bit:expr) => {
_static_cpu_has__4!($bit)
};
}

macro_rules! cpu_has {
($c:expr, $bit:expr) => {
{
let __v: i32 = if true && REQUIRED_MASK_BIT_SET!($bit) != 0 {
1
} else {
test_cpu_cap!($c, $bit) as i32
};
__v
}
};
}

macro_rules! current_thread_info {
() => {
get_current() as *mut thread_info
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

macro_rules! first_node {
($src:expr) => {
__first_node((unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

/// folio_page - Return a page from a folio.
/// @folio: The folio.
/// @n: The page number to return.
///
/// @n is relative to the start of the folio.  This function does not
/// check that the page number lies within @folio; the caller is presumed
/// to have a reference to the page.
macro_rules! folio_page {
($folio:expr, $n:expr) => {
unsafe { core::ptr::addr_of_mut!((*$folio).__slate_anon_0.page) }.wrapping_offset(($n) as isize)
};
}

macro_rules! hugepages_supported {
() => {
boot_cpu_has!(3i32)
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

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2__2 {
($n:expr) => {
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub($n.leading_zeros() as i32) } } else { if (core::mem::size_of::<u64>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64($n) } }
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
macro_rules! ilog2__9 {
($n:expr) => {
{
let __v: i32 = if false {
let __v2: i32 = if ($n) < 2 {
0
} else {
63i32.wrapping_sub((($n) as u64).leading_zeros() as i32)
};
__v2
} else {
let __v3: i32 = if (core::mem::size_of::<u32>() as u64) <= 4 {
__ilog2_u32($n)
} else {
__ilog2_u64(($n) as u64)
};
__v3
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
macro_rules! ilog2__3 {
($n:expr) => {
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
(1i32.wrapping_neg() as u32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__4 {
() => {
(1i32.wrapping_neg() as i64) < 1
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

/// kcsan_check_read_write - check regular read-write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 3i32);
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

macro_rules! kcsan_rmb {
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

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! likely__4 {
($x:expr) => {
(($x) != 0) as i64
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
macro_rules! list_entry__12 {
($ptr:expr) => {
container_of__55!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr) => {
container_of__4!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__10 {
($ptr:expr) => {
container_of__9!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__2 {
($ptr:expr, $type:ty) => {
container_of__2!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__27 {
($ptr:expr, $type:ty) => {
container_of__31!($ptr, $type)
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
macro_rules! list_entry_is_head__2 {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).__slate_anon_0.__slate_anon_0.__slate_anon_1.$member) }) as *const list_head, ($head) as *const list_head)
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
list_entry!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__16 {
($ptr:expr, $type:ty) => {
list_entry__27!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__8 {
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
macro_rules! list_first_entry__13 {
($ptr:expr) => {
list_entry__10!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).__slate_anon_0.__slate_anon_0.__slate_anon_1.$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__4 {
($pos:expr, $member:ident) => {
list_entry__12!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__9 {
($pos:expr, $member:ident) => {
list_entry__10!(unsafe { (*$pos).$member.next })
};
}

/// list_prev_entry - get the prev element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_prev_entry__4 {
($pos:expr, $member:ident) => {
list_entry__12!(unsafe { (*$pos).$member.prev })
};
}

macro_rules! lock_acquire {
() => {
while false {
}
};
}

macro_rules! lock_acquire_exclusive {
() => {
lock_acquire!();
};
}

macro_rules! lock_map_acquire {
() => {
lock_acquire_exclusive!();
};
}

macro_rules! lock_map_release {
() => {
lock_release!();
};
}

macro_rules! lock_release {
() => {
while false {
}
};
}

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

macro_rules! lockdep_assert_held_write {
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

macro_rules! mm_pmd_folded {
() => {
__is_defined!()
};
}

macro_rules! next_node {
($n:expr, $src:expr) => {
__next_node($n, (unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

/// Find the next present node in src, starting after node n, wrapping around to
/// the first node in src if needed.  Returns MAX_NUMNODES if src is empty.
macro_rules! next_node_in__3 {
($n:expr, $src:expr) => {
__next_node_in($n, (unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

macro_rules! node_clear {
($node:expr, $dst:expr) => {
__node_clear($node, (unsafe { core::ptr::addr_of_mut!($dst) }) as *mut __SlateRecord_nodemask_types_h_17);
};
}

/// No static inline type checking - see Subtlety (1) above.
macro_rules! node_isset__2 {
($node:expr, $nodemask:expr) => {
test_bit__2!($node, unsafe { $nodemask.bits.as_mut_ptr() as *mut u64 })
};
}

/// No static inline type checking - see Subtlety (1) above.
macro_rules! node_isset {
($node:expr, $nodemask:expr) => {
test_bit__2!($node, unsafe { core::ptr::addr_of_mut!($nodemask.bits) as *mut u64 })
};
}

macro_rules! node_online__2 {
($node:expr) => {
node_state(($node) as i32, 1u32)
};
}

/// The inline keyword gives the compiler room to decide to inline, or
/// not inline a function as it sees best.  However, as these functions
/// are called in both __init and non-__init functions, if they are not
/// inlined we will end up with a section mismatch error (of the type of
/// freeable items not being freed).  So we must use __always_inline here
/// to fix the problem.  If other functions in the future also end up in
/// this situation they will also need to be annotated as __always_inline
macro_rules! node_set {
($node:expr, $dst:expr) => {
__node_set($node, (unsafe { core::ptr::addr_of_mut!($dst) }) as *mut __SlateRecord_nodemask_types_h_17);
};
}

macro_rules! nodes_and__5 {
($dst:expr, $src1:expr, $src2:expr) => {
__nodes_and(core::ptr::addr_of_mut!($dst), (unsafe { core::ptr::addr_of_mut!($src1) }) as *const __SlateRecord_nodemask_types_h_17, (unsafe { core::ptr::addr_of_mut!($src2) }) as *const __SlateRecord_nodemask_types_h_17, 64u32);
};
}

macro_rules! nodes_clear {
($dst:expr) => {
__nodes_clear(unsafe { core::ptr::addr_of_mut!($dst) }, 64u32);
};
}

macro_rules! nodes_clear__2 {
($dst:expr) => {
__nodes_clear(core::ptr::addr_of_mut!($dst), 64u32);
};
}

macro_rules! nodes_empty__2 {
($src:expr) => {
__nodes_empty((unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

macro_rules! nodes_weight {
($nodemask:expr) => {
__nodes_weight((unsafe { core::ptr::addr_of_mut!($nodemask) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

macro_rules! offsetof__32 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, link) as u64
};
}

macro_rules! offsetof__10 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, vfs_inode) as u64
};
}

macro_rules! offsetof {
() => {
8u64
};
}

macro_rules! offsetof__20 {
() => {
24u64
};
}

macro_rules! offsetof__66 {
() => {
core::mem::offset_of!(file_region, link) as u64
};
}

macro_rules! offsetof__23 {
() => {
core::mem::offset_of!(huge_bootmem_page, list) as u64
};
}

macro_rules! offsetof__75 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, refs) as u64
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
macro_rules! order_base_2__2 {
($n:expr) => {
if true { if ($n) == 0 || ($n) == 1 { 0i32 } else { ilog2__3!($n.wrapping_sub(1)).wrapping_add(1) } } else { __order_base_2(($n) as u64) }
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
if true { if ($n) == 0 || ($n) == 1 { 0i32 } else { ilog2__2!($n.wrapping_sub(1)).wrapping_add(1) } } else { __order_base_2($n) }
};
}

macro_rules! p4d_val {
($x:expr) => {
native_p4d_val($x)
};
}

/// page_folio - Converts from page to folio.
/// @p: The page.
///
/// Every page is part of a folio.  This function cannot be called on a
/// NULL pointer.
///
/// Context: No reference, nor lock is required on @page.  If the caller
/// does not hold a reference, this call may race with a folio split, so
/// it should re-check the folio still contains this page after gaining
/// a reference on the folio.
/// Return: The folio which contains this page.
macro_rules! page_folio {
($p:expr) => {
_compound_head(($p) as *const page) as *mut folio
};
}

/// page_folio - Converts from page to folio.
/// @p: The page.
///
/// Every page is part of a folio.  This function cannot be called on a
/// NULL pointer.
///
/// Context: No reference, nor lock is required on @page.  If the caller
/// does not hold a reference, this call may race with a folio split, so
/// it should re-check the folio still contains this page after gaining
/// a reference on the folio.
/// Return: The folio which contains this page.
macro_rules! page_folio__2 {
($p:expr) => {
_compound_head($p) as *const folio
};
}

macro_rules! page_ptdesc {
($p:expr) => {
($p) as *mut ptdesc
};
}

macro_rules! page_ref_tracepoint_active {
() => {
0i32
};
}

macro_rules! page_to_virt__2 {
($x:expr) => {
__va!(PFN_PHYS!(((unsafe { $x.offset_from(vmemmap_base as *mut page) }) as i64) as u64))
};
}

macro_rules! pageblock_aligned {
($pfn:expr) => {
IS_ALIGNED!($pfn, 1u64 << MIN_T__2!(u32, 9i32, 10i32))
};
}

macro_rules! persistent_huge_pages {
($h:expr) => {
unsafe { (*$h).nr_huge_pages }.wrapping_sub(unsafe { (*$h).surplus_huge_pages })
};
}

/// Must be a compile-time constant, so implement it as a macro
macro_rules! pgd_index {
($a:expr) => {
($a) >> unsafe { pgdir_shift } & 511u64
};
}

macro_rules! pgd_offset {
($mm:expr, $address:expr) => {
pgd_offset_pgd(unsafe { (*$mm).__slate_anon_0.pgd }, $address)
};
}

macro_rules! pgd_val {
($x:expr) => {
native_pgd_val($x)
};
}

macro_rules! pgprot_val {
($x:expr) => {
$x.pgprot
};
}

macro_rules! pgtable_l5_enabled__2 {
() => {
cpu_feature_enabled__5!(528i32)
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

macro_rules! pte_clear {
($mm:expr, $addr:expr, $ptep:expr) => {
native_pte_clear($mm, $addr, $ptep);
};
}

macro_rules! pte_next_pfn {
($pte:expr) => {
pte_advance_pfn($pte, 1u64)
};
}

macro_rules! pte_val {
($x:expr) => {
native_pte_val($x)
};
}

macro_rules! pud_clear {
($pud:expr) => {
native_pud_clear($pud);
};
}

macro_rules! pud_val {
($x:expr) => {
native_pud_val($x)
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

/// Always evaluate the 'subclass' argument to avoid that the compiler
/// warns about set-but-not-used variables when building with
/// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
macro_rules! raw_spin_lock_nested {
($lock:expr, $subclass:expr) => {
$subclass;
unsafe { _raw_spin_lock($lock) };
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

/// read_seqcount_begin() - begin a seqcount_t read critical section
/// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
///
/// Return: count to be passed to read_seqcount_retry()
macro_rules! read_seqcount_begin {
($s:expr) => {
{
let mut __v: u32;
let mut __v2: u32;
let mut __v3: u32 = 0;
loop {
let __v4: u32 = seqprop_sequence!($s);
__v3 = __v4;
if __v4 & 1 == 0 {
break;
}
cpu_relax();
}
kcsan_atomic_next(1000i32);
__v2 = __v3;
__v = __v2;
__v
}
};
}

/// read_seqcount_retry() - end a seqcount_t read critical section
/// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
/// @start: count, from read_seqcount_begin()
///
/// read_seqcount_retry closes the read critical section of given
/// seqcount_t.  If the critical section was invalid, it must be ignored
/// (and typically retried).
///
/// Return: true if a read section retry is required, else false
macro_rules! read_seqcount_retry {
($s:expr, $start:expr) => {
do_read_seqcount_retry(seqprop_const_ptr!($s), $start)
};
}

/// roundup_pow_of_two - round the given value up to nearest power of two
/// @n: parameter
///
/// round the given value up to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! roundup_pow_of_two__3 {
($n:expr) => {
{
let __v: u64 = if false {
let __v2: u64 = if ($n) == 1 {
1
} else {
1u64 << ilog2__9!($n.wrapping_sub(1)).wrapping_add(1)
};
__v2
} else {
__roundup_pow_of_two(($n) as u64)
};
__v
}
};
}

macro_rules! seqprop_const_ptr {
($s:expr) => {
__seqprop_spinlock_const_ptr(($s) as *const seqcount_spinlock)
};
}

macro_rules! seqprop_preemptible {
($s:expr) => {
__seqprop_preemptible(($s) as *const seqcount)
};
}

macro_rules! seqprop_ptr {
($s:expr) => {
__seqprop_ptr($s)
};
}

macro_rules! seqprop_sequence {
($s:expr) => {
__seqprop_spinlock_sequence(($s) as *const seqcount_spinlock)
};
}

macro_rules! set_pte {
($ptep:expr, $pte:expr) => {
native_set_pte($ptep, $pte);
};
}

macro_rules! set_pte_at {
($mm:expr, $addr:expr, $ptep:expr, $pte:expr) => {
set_ptes($mm, $addr, $ptep, $pte, 1u32);
};
}

macro_rules! set_pud {
($pudp:expr, $pud:expr) => {
native_set_pud($pudp, $pud);
};
}

/// sizeof_field() - Report the size of a struct field in bytes
///
/// @TYPE: The structure containing the field of interest
/// @MEMBER: The field to return the size of
macro_rules! sizeof_field {
() => {
(const { core::mem::size_of::<u64>() }) as u64
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

macro_rules! spin_lock_nested {
($lock:expr, $subclass:expr) => {
loop {
raw_spin_lock_nested!(spinlock_check($lock), $subclass);
__release!();
__acquire!();
if true {
break;
}
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

macro_rules! task_thread_info {
($task:expr) => {
unsafe { core::ptr::addr_of_mut!((*$task).thread_info) }
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

macro_rules! test_bit__5 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
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
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
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
let __v: bool = true && false && (($addr) as u64) != 0;
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
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_cpu_cap {
($c:expr, $bit:expr) => {
arch_test_bit(($bit) as u64, ((unsafe { (*$c).__slate_anon_9.x86_capability.as_mut_ptr() as *mut u32 }) as *mut u64) as *const u64)
};
}

macro_rules! test_thread_flag {
($flag:expr) => {
test_ti_thread_flag(current_thread_info!(), $flag)
};
}

macro_rules! tlb_remove_huge_tlb_entry {
($h:expr, $tlb:expr, $ptep:expr, $address:expr) => {
loop {
let _sz: u64 = huge_page_size(($h) as *const hstate);
if _sz >= _AC!() << 39i32 {
tlb_flush_p4d_range($tlb, $address, _sz);
} else {
if _sz >= _AC!() << 30i32 {
tlb_flush_pud_range($tlb, $address, _sz);
} else {
if _sz >= _AC!() << 21i32 {
tlb_flush_pmd_range($tlb, $address, _sz);
} else {
tlb_flush_pte_range($tlb, $address, _sz);
}
}
}
__tlb_remove_tlb_entry($tlb, $ptep, $address);
if true {
break;
}
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

macro_rules! unlikely__4 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__5 {
($x:expr) => {
!$x.is_null() as i64
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
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut pgdir_shift: u32;
    static mut ptrs_per_p4d: u32;
    static mut __supported_pte_mask: u64;
    static mut __SCK__might_resched: static_call_key;
    static mut nr_node_ids: u32;
    static mut node_states: [__SlateRecord_nodemask_types_h_17; 6];
    static mut node_data: [*mut pglist_data; 0];
    static mut __num_online_cpus: __SlateRecord_types_h_190;
    static mut __num_possible_cpus: u32;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut boot_cpu_data: cpuinfo_x86;
    static mut cpuinfo_op: seq_operations;
    static mut __SCK__cond_resched: static_call_key;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut numa_node: i32;
    static mut zero_page_pfn: u64;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut vm_event_states: vm_event_state;
    static mut cpusets_pre_enable_key: static_key_false;
    static mut cpusets_enabled_key: static_key_false;
    static mut kstat: kernel_stat;
    static mut movable_gigantic_pages: i32;
    static mut delayacct_key: static_key_false;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __compiletime_assert_0();
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __bad_size_call_parameter();
    fn ____wrong_branch_error() -> bool;
    fn __xchg_wrong_size();
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn __compiletime_assert_34();
    fn __compiletime_assert_35();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn _find_first_bit(arg0: *const u64, arg1: u64) -> u64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn memparse(arg0: *const u8, arg1: *mut *mut u8) -> u64;
    fn __bitmap_and(arg0: *mut u64, arg1: *const u64, arg2: *const u64, arg3: u32) -> bool;
    fn __bitmap_weight(arg0: *const u64, arg1: u32) -> u32;
    fn _kstrtoul(arg0: *const u8, arg1: u32, arg2: *mut u64) -> i32;
    fn kstrtoull(arg0: *const u8, arg1: u32, arg2: *mut u64) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn sscanf(arg0: *const u8, arg1: *const u8, ...) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_50();
    fn __compiletime_assert_53();
    fn __compiletime_assert_57();
    fn __compiletime_assert_61();
    fn __compiletime_assert_63();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn __compiletime_assert_92();
    fn __compiletime_assert_93();
    fn __compiletime_assert_94();
    fn __compiletime_assert_95();
    fn __compiletime_assert_96();
    fn __compiletime_assert_97();
    fn __compiletime_assert_98();
    fn __compiletime_assert_99();
    fn __compiletime_assert_100();
    fn __compiletime_assert_101();
    fn __compiletime_assert_102();
    fn __compiletime_assert_103();
    fn __compiletime_assert_104();
    fn __compiletime_assert_105();
    fn __compiletime_assert_106();
    fn __compiletime_assert_107();
    fn __compiletime_assert_108();
    fn __compiletime_assert_109();
    fn __compiletime_assert_110();
    fn get_pfnblock_migratetype(arg0: *const page, arg1: u64) -> u32;
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn __SCT__cond_resched() -> i32;
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn __init_rwsem(arg0: *mut rw_semaphore, arg1: *const u8, arg2: *mut lock_class_key);
    fn down_read(arg0: *mut rw_semaphore);
    fn down_write(arg0: *mut rw_semaphore);
    fn down_write_trylock(arg0: *mut rw_semaphore) -> i32;
    fn up_read(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn flush_work(arg0: *mut work_struct) -> bool;
    fn tlb_gather_mmu(arg0: *mut mmu_gather, arg1: *mut mm_struct);
    fn tlb_gather_mmu_vma(arg0: *mut mmu_gather, arg1: *mut vm_area_struct);
    fn tlb_finish_mmu(arg0: *mut mmu_gather);
    fn __compiletime_assert_187();
    fn __compiletime_assert_188();
    fn __compiletime_assert_189();
    fn __compiletime_assert_190();
    fn __compiletime_assert_192();
    fn __compiletime_assert_193();
    fn first_online_pgdat() -> *mut pglist_data;
    fn next_zone(arg0: *mut zone) -> *mut zone;
    fn __next_zones_zonelist(arg0: *mut zoneref, arg1: u32, arg2: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref;
    fn rcuwait_wake_up(arg0: *mut rcuwait) -> i32;
    fn __compiletime_assert_211();
    fn __compiletime_assert_224();
    fn __compiletime_assert_226();
    fn ptep_set_access_flags(arg0: *mut vm_area_struct, arg1: u64, arg2: *mut __SlateRecord_pgtable_64_types_h_21, arg3: __SlateRecord_pgtable_64_types_h_21, arg4: i32) -> i32;
    fn __compiletime_assert_230();
    fn __compiletime_assert_234();
    fn ptep_clear_flush(arg0: *mut vm_area_struct, arg1: u64, arg2: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21;
    fn kfree(arg0: *const ());
    fn __compiletime_assert_257();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn __mmu_notifier_invalidate_range_start(arg0: *mut mmu_notifier_range) -> i32;
    fn __mmu_notifier_invalidate_range_end(arg0: *mut mmu_notifier_range);
    fn flush_tlb_mm_range(arg0: *mut mm_struct, arg1: u64, arg2: u64, arg3: u32, arg4: bool);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_376();
    fn vma_mmu_pagesize(arg0: *mut vm_area_struct) -> u64;
    fn prep_compound_page(arg0: *mut page, arg1: u32);
    fn __compiletime_assert_419();
    fn __compiletime_assert_427();
    fn __compiletime_assert_428();
    fn sysfs_emit_at(arg0: *mut u8, arg1: i32, arg2: *const u8, ...) -> i32;
    fn __folio_put(arg0: *mut folio);
    fn mod_node_page_state(arg0: *mut pglist_data, arg1: u32, arg2: i64);
    fn folio_mark_dirty(arg0: *mut folio) -> bool;
    fn __p4d_alloc(arg0: *mut mm_struct, arg1: *mut __SlateRecord_pgtable_types_h_295, arg2: u64) -> i32;
    fn __pud_alloc(arg0: *mut mm_struct, arg1: *mut __SlateRecord_pgtable_types_h_342, arg2: u64) -> i32;
    fn __pmd_alloc(arg0: *mut mm_struct, arg1: *mut __SlateRecord_pgtable_types_h_368, arg2: u64) -> i32;
    fn __compiletime_assert_479();
    fn __compiletime_assert_480();
    fn adjust_managed_page_count(arg0: *mut page, arg1: i64);
    fn early_pfn_to_nid(arg0: u64) -> i32;
    fn si_mem_available() -> i64;
    fn mapping_rmap_tree_iter_first(arg0: *mut address_space, arg1: u64, arg2: u64) -> *mut vm_area_struct;
    fn mapping_rmap_tree_iter_next(arg0: *mut vm_area_struct, arg1: u64, arg2: u64) -> *mut vm_area_struct;
    fn folio_zero_user(arg0: *mut folio, arg1: u64);
    fn copy_user_large_folio(arg0: *mut folio, arg1: *mut folio, arg2: u64, arg3: *mut vm_area_struct) -> i32;
    fn string_get_size(arg0: u64, arg1: u64, arg2: u32, arg3: *mut u8, arg4: i32) -> i32;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn folio_mapping(arg0: *const folio) -> *mut address_space;
    fn __filemap_get_folio_mpol(arg0: *mut address_space, arg1: u64, arg2: u32, arg3: u32, arg4: *mut mempolicy) -> *mut folio;
    fn __folio_lock(arg0: *mut folio);
    fn folio_unlock(arg0: *mut folio);
    fn folio_wait_bit(arg0: *mut folio, arg1: i32);
    fn __filemap_add_folio(arg0: *mut address_space, arg1: *mut folio, arg2: u64, arg3: u32, arg4: *mut *mut ()) -> i32;
    fn __mpol_put(arg0: *mut mempolicy);
    fn get_task_policy(arg0: *mut task_struct) -> *mut mempolicy;
    fn huge_node(arg0: *mut vm_area_struct, arg1: u64, arg2: u32, arg3: *mut *mut mempolicy, arg4: *mut *mut __SlateRecord_nodemask_types_h_17) -> i32;
    fn init_nodemask_of_mempolicy(arg0: *mut __SlateRecord_nodemask_types_h_17) -> bool;
    fn apply_policy_zone(arg0: *mut mempolicy, arg1: u32) -> bool;
    fn cpuset_nodemask_valid_mems_allowed(arg0: *const __SlateRecord_nodemask_types_h_17) -> i32;
    fn cpuset_current_node_allowed(arg0: i32, arg1: u32) -> bool;
    fn memblock_reserved_mark_noinit(arg0: u64, arg1: u64) -> i32;
    fn memblock_free(arg0: *mut (), arg1: u64);
    fn __next_mem_pfn_range(arg0: *mut i32, arg1: i32, arg2: *mut u64, arg3: *mut u64, arg4: *mut i32);
    fn memblock_alloc_hugetlb(arg0: u64, arg1: i32, arg2: bool) -> *mut ();
    fn folio_move_anon_rmap(arg0: *mut folio, arg1: *mut vm_area_struct);
    fn hugetlb_add_new_anon_rmap(arg0: *mut folio, arg1: *mut vm_area_struct, arg2: u64);
    fn migration_entry_wait_huge(arg0: *mut vm_area_struct, arg1: u64, arg2: *mut __SlateRecord_pgtable_64_types_h_21);
    fn __compiletime_assert_661();
    fn putback_movable_pages(arg0: *mut list_head);
    fn __delayacct_wpcopy_start();
    fn __delayacct_wpcopy_end();
    fn padata_do_multithreaded(arg0: *mut padata_mt_job);
    fn tlb_remove_table_sync_one();
    fn __tlb_remove_page_size(arg0: *mut mmu_gather, arg1: *mut page, arg2: i32) -> bool;
    fn tlb_flush_mmu(arg0: *mut mmu_gather);
    fn __compiletime_assert_679();
    fn __compiletime_assert_682();
    fn __vmf_anon_prepare(arg0: *mut vm_fault) -> u32;
    fn __alloc_frozen_pages_noprof(arg0: u32, arg1: u32, arg2: i32, arg3: *mut __SlateRecord_nodemask_types_h_17, arg4: u32) -> *mut page;
    fn free_frozen_pages(arg0: *mut page, arg1: u32);
    fn hugetlb_vmemmap_restore_folio(arg0: *const hstate, arg1: *mut folio) -> i32;
    fn hugetlb_vmemmap_restore_folios(arg0: *const hstate, arg1: *mut list_head, arg2: *mut list_head) -> i64;
    fn hugetlb_vmemmap_optimize_folio(arg0: *const hstate, arg1: *mut folio);
    fn hugetlb_vmemmap_optimize_folios(arg0: *mut hstate, arg1: *mut list_head);
    fn hugetlb_vmemmap_optimize_bootmem_folios(arg0: *mut hstate, arg1: *mut list_head);
    fn hugetlb_cgroup_charge_cgroup(arg0: i32, arg1: u64, arg2: *mut *mut hugetlb_cgroup) -> i32;
    fn hugetlb_cgroup_charge_cgroup_rsvd(arg0: i32, arg1: u64, arg2: *mut *mut hugetlb_cgroup) -> i32;
    fn hugetlb_cgroup_commit_charge(arg0: i32, arg1: u64, arg2: *mut hugetlb_cgroup, arg3: *mut folio);
    fn hugetlb_cgroup_commit_charge_rsvd(arg0: i32, arg1: u64, arg2: *mut hugetlb_cgroup, arg3: *mut folio);
    fn hugetlb_cgroup_uncharge_folio(arg0: i32, arg1: u64, arg2: *mut folio);
    fn hugetlb_cgroup_uncharge_folio_rsvd(arg0: i32, arg1: u64, arg2: *mut folio);
    fn hugetlb_cgroup_uncharge_cgroup(arg0: i32, arg1: u64, arg2: *mut hugetlb_cgroup);
    fn hugetlb_cgroup_uncharge_cgroup_rsvd(arg0: i32, arg1: u64, arg2: *mut hugetlb_cgroup);
    fn hugetlb_cgroup_uncharge_counter(arg0: *mut resv_map, arg1: u64, arg2: u64);
    fn hugetlb_cgroup_uncharge_file_region(arg0: *mut resv_map, arg1: *mut file_region, arg2: u64, arg3: bool);
    fn hugetlb_cgroup_file_init();
    fn hugetlb_cgroup_migrate(arg0: *mut folio, arg1: *mut folio);
    fn hugetlb_sysfs_init();
    fn hugetlb_sysctl_init();
    fn pfn_range_intersects_zones(arg0: i32, arg1: u64, arg2: u64) -> bool;
    fn __init_single_page(arg0: *mut page, arg1: u64, arg2: u64, arg3: i32);
    fn init_pageblock_migratetype(arg0: *mut page, arg1: u32, arg2: bool);
    fn __compiletime_assert_700();
    fn __compiletime_assert_703();
    fn __compiletime_assert_706();
    fn __compiletime_assert_709();
    fn __compiletime_assert_711();
    fn __compiletime_assert_720();
    fn __compiletime_assert_723();
    fn __compiletime_assert_724();
    fn __compiletime_assert_732();
    fn __compiletime_assert_733();
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

fn __kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

fn kcsan_nestable_atomic_begin() {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


fn kcsan_nestable_atomic_end() {
}

fn kcsan_atomic_next(n: i32) {
}

fn kcsan_set_access_mask(mask: u64) {
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
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


fn __kcsan_enable_current() {
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


fn __kcsan_disable_current() {
}

// kcsan_check_read - check regular read access for races
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_move(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add(list, head);
}

fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_207 {
}


fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct io_bitmap {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[inline(always)]
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpusets_enabled_key, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpusets_pre_enable_key, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpusets_enabled_key, const 0 as u8, label {
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
union __SlateRecord_processor_h_111 {
    hw_cpu_type: u32,
    __slate_anon_1: __SlateRecord_processor_h_113,
    __slate_anon_2: __SlateRecord_processor_h_119,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits156U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_119 {
    __slate_bits_0: __slate_bits::__SlateBits157U0,
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
fn arch_static_branch__const6(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym delayacct_key, const 0 as u8, label {
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
struct perf_event {
}


#[inline(always)]
fn arch_static_branch__const7(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym delayacct_key, const 0 as u8, label {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits107U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits108U0,
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


// PGDIR_SHIFT determines what a top-level page table entry can map
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


// No need to mask any bits for !PAE
fn native_pgd_val(pgd: __SlateRecord_pgtable_types_h_295) -> u64 {
pgd.pgd & !0
}

fn pgd_flags(pgd: __SlateRecord_pgtable_types_h_295) -> u64 {
native_pgd_val(pgd) & !(((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_342 {
    p4d: u64,
}


fn native_p4d_val(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
p4d.p4d
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


fn native_make_pud(val: u64) -> __SlateRecord_pgtable_types_h_368 {
let mut __slate_compound_0: __SlateRecord_pgtable_types_h_368 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_pgtable_types_h_368 { pud: val };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


fn native_pud_val(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
pud.pud
}

fn p4d_pfn_mask(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
// No 512 GiB huge pages yet
((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64
}

fn p4d_flags_mask(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
!p4d_pfn_mask(p4d)
}

fn p4d_flags(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
native_p4d_val(p4d) & p4d_flags_mask(p4d)
}

fn pud_pfn_mask(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
if native_pud_val(pud) & _AT!(u64, 1i32) << 7i32 != 0 {
return ((!(_AC!() << 30i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64;
} else {
return ((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64;
}
}

fn pud_flags_mask(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
!pud_pfn_mask(pud)
}

fn pud_flags(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
native_pud_val(pud) & pud_flags_mask(pud)
}

fn native_make_pte(val: u64) -> __SlateRecord_pgtable_64_types_h_21 {
let mut __slate_compound_0: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_pgtable_64_types_h_21 { pte: val };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

fn native_pte_val(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
pte.pte
}

fn pte_flags(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
native_pte_val(pte) & !(((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)
}

#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__3!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
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
fn arch_atomic_fetch_add(mut __ret: i32, mut v: *mut __SlateRecord_types_h_190) -> i32 {
let mut __v0: i32;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
unsafe {
let __t0: u32;
core::arch::asm!("lock xaddb {0:l}, ({1:r})", inlateout(reg) __ret as u32 => __t0, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
unsafe {
let __t1: u32;
core::arch::asm!("lock xaddw {0:x}, ({1:r})", inlateout(reg) __ret as u32 => __t1, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
unsafe {
let __t2: u32;
core::arch::asm!("lock xaddl {0:e}, ({1:r})", inlateout(reg) __ret as u32 => __t2, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
unsafe {
let __t3: u32;
core::arch::asm!("lock xaddq {0:r}, ({1:r})", inlateout(reg) __ret as u32 => __t3, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
_ => {
unsafe { __xadd_wrong_size() };
}
}
__v0 = __ret;
__v0
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
fn arch_atomic_cmpxchg(v: *mut __SlateRecord_types_h_190, mut __old: i32, mut __new: i32) -> i32 {
let mut __v0: i32;
let mut __ret: i32 = 0;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t3: u32;
core::arch::asm!("lock cmpxchgq {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v0 = __ret;
__v0
}

#[inline(always)]
fn arch_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, mut __new: i32) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i32 = unsafe { *old };
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u8;
let __t1: u32;
core::arch::asm!("lock cmpxchgb {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
success = (__t0 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t2: u8;
let __t3: u32;
core::arch::asm!("lock cmpxchgw {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
success = (__t2 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t4: u8;
let __t5: u32;
core::arch::asm!("lock cmpxchgl {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t5, options(nostack, att_syntax));
success = (__t4 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t5 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t6: u8;
let __t7: u32;
core::arch::asm!("lock cmpxchgq {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t7, options(nostack, att_syntax));
success = (__t6 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t7 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely!(!success) != 0 {
unsafe {
*old = __old;
}
}
__v0 = success as i64;
__v0 != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
fn arch_atomic64_sub(mut i: i64, mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock subq {1:r}, ({0:r})\n# {2}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u64, in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
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
fn arch_atomic64_sub_and_test(mut i: i64, mut v: *mut __SlateRecord_types_h_197) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock subq {2:r}, ({0:r})\nsete {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), lateout(reg_byte) __t0, in(reg) i as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[inline(always)]
fn raw_atomic_set_release(v: *mut __SlateRecord_types_h_190, i: i32) {
if __native_word__24!() {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__9!() {
unsafe { __compiletime_assert_34() };
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
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_35() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!(*unsafe { core::ptr::addr_of_mut!((*v).counter) }, i);
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
} else {
loop {
kcsan_mb!();
__smp_mb__before_atomic!();
if true {
break;
}
}
raw_atomic_set(v, i);
}
}

#[inline(always)]
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

#[inline(always)]
fn raw_atomic_cmpxchg(v: *mut __SlateRecord_types_h_190, old: i32, new: i32) -> i32 {
arch_atomic_cmpxchg(v, old, new)
}

#[inline(always)]
fn raw_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, new: i32) -> bool {
arch_atomic_try_cmpxchg(v, old, new)
}

#[inline(always)]
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[inline(always)]
fn raw_atomic_fetch_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> i32 {
let mut c: i32 = raw_atomic_read(v as *const __SlateRecord_types_h_190);
while unlikely!(c == u) == 0 {
if !(!raw_atomic_try_cmpxchg(v, unsafe { &mut (*core::ptr::addr_of_mut!(c)) }, c.wrapping_add(a))) {
break;
}
}
c
}

#[inline(always)]
fn raw_atomic_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> bool {
raw_atomic_fetch_add_unless(v, a, u) != u
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
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
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
fn raw_atomic64_add(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic64_sub(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_sub(i, v);
}

#[inline(always)]
fn raw_atomic64_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
arch_atomic64_sub_and_test(i, v)
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic_long_sub(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_sub(i, v);
}

#[inline(always)]
fn raw_atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_sub_and_test(i, v)
}

#[inline(always)]
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
}

#[inline(always)]
fn instrument_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_read_write!(v, size);
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
core::arch::asm!("40: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 40b - ., 3; .popsection\n# {0}", const 40 as i32, options(nostack, att_syntax));
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
core::arch::asm!("41: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 41b - ., 4; .popsection\n# {0}", const 41 as i32, options(nostack, att_syntax));
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
struct workqueue_struct {
}


#[inline(always)]
fn instrument_atomic_read(v: *const (), size: u64) {
kasan_check_read(v, size as u32);
kcsan_check_atomic_read!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
instrument_atomic_check_alignment(v, size);
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
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
}

#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[inline(always)]
fn atomic_set_release(v: *mut __SlateRecord_types_h_190, i: i32) {
kcsan_release!();
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set_release(v, i);
}

#[inline(always)]
fn atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_add_relaxed(i, v)
}

#[inline(always)]
fn atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_release!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_sub_release(i, v)
}

#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[inline(always)]
fn atomic_cmpxchg(v: *mut __SlateRecord_types_h_190, old: i32, new: i32) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_cmpxchg(v, old, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_and_test(v)
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


#[inline(always)]
fn atomic_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_add_unless(v, a, u)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[inline(always)]
fn atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add(i, v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


#[inline(always)]
fn atomic_long_sub(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub(i, v);
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


#[inline(always)]
fn atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_and_test(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


#[inline(always)]
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
}

#[inline(always)]
fn generic___clear_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let __v2: u64 = (unsafe { *p }) & !mask;
unsafe {
*p = __v2;
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
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
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
fn arch___clear_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
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
struct rq_flags {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
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
fn ___clear_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___clear_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


fn test_and_set_bit_lock(nr: i64, addr: *mut u64) -> bool {
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit_lock(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


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

fn rol32(word: u32, shift: u32) -> u32 {
word << (shift & 31) | word >> (shift.wrapping_neg() & 31)
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
fn bitmap_zero(dst: *mut u64, nbits: u32) {
let len: u32 = bitmap_size!(nbits);
if small_const_nbits!(nbits) {
unsafe {
*dst = 0;
}
} else {
unsafe { memset(dst as *mut (), 0i32, len as u64) };
}
}

#[inline(always)]
fn bitmap_and(dst: *mut u64, src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
let __v0: u64 = (unsafe { *src1 }) & unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits);
unsafe {
*dst = __v0;
}
return __v0 != 0;
}
unsafe { __bitmap_and(dst, src1, src2, nbits) }
}

#[inline(always)]
fn bitmap_empty(src: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits) == 0;
}
find_first_bit(src, nbits as u64) == (nbits as u64)
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
struct uprobe {
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


fn kstrtoul(s: *const u8, base: u32, res: *mut u64) -> i32 {
// We want to shortcut function call, but
// __builtin_types_compatible_p(unsigned long, unsigned long long) = 0.
if ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u64>() }) as u64) && ((const { core::mem::align_of::<u64>() }) as u64) == ((const { core::mem::align_of::<u64>() }) as u64) {
return unsafe { kstrtoull(s, base, res as *mut u64) };
} else {
return unsafe { _kstrtoul(s, base, res) };
}
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

#[inline(always)]
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
}

fn __roundup_pow_of_two(n: u64) -> u64 {
1u64 << fls_long(n.wrapping_sub(1))
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
struct uprobe_xol_ops {
}


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
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


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_44: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

#[inline(always)]
fn __node_set(node: i32, dstp: *mut __SlateRecord_nodemask_types_h_17) {
set_bit(node as i64, unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 });
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


#[inline(always)]
fn __node_clear(node: i32, dstp: *mut __SlateRecord_nodemask_types_h_17) {
clear_bit(node as i64, unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[inline(always)]
fn __nodes_clear(dstp: *mut __SlateRecord_nodemask_types_h_17, nbits: u32) {
bitmap_zero(unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 }, nbits);
}

#[inline(always)]
fn __nodes_and(dstp: *mut __SlateRecord_nodemask_types_h_17, src1p: *const __SlateRecord_nodemask_types_h_17, src2p: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> bool {
bitmap_and(unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 }, unsafe { (*src1p).bits.as_ptr() as *const u64 }, unsafe { (*src2p).bits.as_ptr() as *const u64 }, nbits)
}

#[inline(always)]
fn __nodes_empty(srcp: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> bool {
bitmap_empty(unsafe { (*srcp).bits.as_ptr() as *const u64 }, nbits)
}

#[inline(always)]
fn __nodes_weight(srcp: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> i32 {
bitmap_weight(unsafe { (*srcp).bits.as_ptr() as *const u64 }, nbits) as i32
}

#[inline(always)]
fn __first_node(srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__48: i32 = 64i32;
let __UNIQUE_ID_y__49: u64 = find_first_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64);
loop {
if __types_ok__3!(__UNIQUE_ID_x__48, __UNIQUE_ID_y__49) == 0 {
unsafe { __compiletime_assert_50() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__48 as u64) < __UNIQUE_ID_y__49 { __UNIQUE_ID_x__48 as u64 } else { __UNIQUE_ID_y__49 };
__v0 as u32
}

#[inline(always)]
fn __next_node(n: i32, srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__51: i32 = 64i32;
let __UNIQUE_ID_y__52: u64 = find_next_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64, n.wrapping_add(1) as u64);
loop {
if __types_ok__3!(__UNIQUE_ID_x__51, __UNIQUE_ID_y__52) == 0 {
unsafe { __compiletime_assert_53() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__51 as u64) < __UNIQUE_ID_y__52 { __UNIQUE_ID_x__51 as u64 } else { __UNIQUE_ID_y__52 };
__v0 as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[inline(always)]
fn __next_node_in(node: i32, srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut ret: u32 = __next_node(node, srcp);
if ret == 64u32 {
ret = __first_node(srcp);
}
ret
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
struct encoded_page {
}


#[inline(always)]
fn init_nodemask_of_node(mask: *mut __SlateRecord_nodemask_types_h_17, node: i32) {
nodes_clear!(*mask);
node_set!(node, *mask);
}

#[inline(always)]
fn node_state(node: i32, state: u32) -> i32 {
node_isset!(node, node_states[(state as u32) as usize])
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


#[inline(always)]
fn num_node_state(state: u32) -> i32 {
nodes_weight!(node_states[(state as u32) as usize])
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
union __SlateRecord_mm_types_h_579 {
    pt_rcu_head: callback_head,
    pt_list: list_head,
    __slate_anon_2: __SlateRecord_mm_types_h_582,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_582 {
    _pt_pad_1: u64,
    pmd_huge_pte: *mut page,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_589 {
    pt_index: u64,
    pt_mm: *mut mm_struct,
    pt_frag_refcount: __SlateRecord_types_h_190,
    pt_share_count: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_598 {
    _pt_pad_2: u64,
    ptl: spinlock,
}


// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
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
fn num_possible_cpus() -> u32 {
unsafe { __num_possible_cpus }
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
unsafe { __compiletime_assert_57() };
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


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x0_0 {
    __slate_bits_0: __slate_bits::__SlateBits146U0,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x1_0 {
    __slate_bits_0: __slate_bits::__SlateBits147U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct leaf_parse_info {
    nr_entries: u32,
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
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuid_table {
    leaves: cpuid_leaves,
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
fn __phys_addr_nodebug(mut x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__2!());
// use the carry flag to determine if x was < __START_KERNEL_map
x = y.wrapping_add(if x > y { unsafe { phys_base } } else { _AC__2!().wrapping_sub(unsafe { page_offset_base }) });
x
}

#[inline(always)]
fn task_size_max() -> u64 {
let mut ret: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\tmovq ${2},{0:r}\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte (16*32+16)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tmovq ${3},{0:r}\n775:\n.popsection\n\n# {1}", lateout(reg) __t0, const 0 as i32, const 140737488351232 as u64, const 72057594037923840 as u64, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_page_flags_h_276 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[inline(always)]
fn native_pause() {
unsafe {
core::arch::asm!("pause", options(nostack, att_syntax));
}
}

#[inline(always)]
fn cpu_relax() {
native_pause();
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
    __slate_bits_0: __slate_bits::__SlateBits144U0,
}


// sizeof(unsigned long) coming from an extra "long" at the end of the
// iobitmap. The limit is inclusive, i.e. the last valid byte.
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
    __slate_bits_0: __slate_bits::__SlateBits130U0,
    pkru: u32,
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


fn llist_empty(head: *const llist_head) -> bool {
let mut __v0: *mut llist_node;
loop {
if !(__native_word__22!() || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_61() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).first) as *const *mut llist_node) };
__v0.is_null()
}

fn llist_add_batch(new_first: *mut llist_node, new_last: *mut llist_node, head: *mut llist_head) -> bool {
let mut first: *mut llist_node = unsafe { core::mem::zeroed() };
let mut __v10: *mut llist_node;
loop {
if !(__native_word__22!() || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_63() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head).first) as *const *mut llist_node) };
first = __v10;
loop {
unsafe {
(*new_last).next = first;
}
let mut __v13: i64;
let __v0: *mut *mut llist_node = unsafe { core::ptr::addr_of_mut!((*head).first) };
let __v1: *mut *mut llist_node = core::ptr::addr_of_mut!(first);
kcsan_mb!();
instrument_atomic_read_write(__v0 as *const (), core::mem::size_of::<*mut llist_node>() as u64);
instrument_read_write(__v1 as *const (), core::mem::size_of::<*mut llist_node>() as u64);
let mut __v15: i64;
let mut __v2: bool = false;
let mut __v4: *mut llist_node = unsafe { *__v1 };
let mut __v5: *mut llist_node = new_first;
match (const { core::mem::size_of::<*mut llist_node>() }) as u64 {
1 => {
let mut __v6: *mut u8 = __v0 as *mut u8;
unsafe {
let __t0: u8;
let __t1: u64;
core::arch::asm!("lock cmpxchgb {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __v6, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t1, options(nostack, att_syntax));
__v2 = (__t0 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t1 as u64) };
}
}
2 => {
let mut __v7: *mut u16 = __v0 as *mut u16;
unsafe {
let __t2: u8;
let __t3: u64;
core::arch::asm!("lock cmpxchgw {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __v7, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t3, options(nostack, att_syntax));
__v2 = (__t2 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t3 as u64) };
}
}
4 => {
let mut __v8: *mut u32 = __v0 as *mut u32;
unsafe {
let __t4: u8;
let __t5: u64;
core::arch::asm!("lock cmpxchgl {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __v8, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t5, options(nostack, att_syntax));
__v2 = (__t4 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t5 as u64) };
}
}
8 => {
let mut __v9: *mut u64 = __v0 as *mut u64;
unsafe {
let __t6: u8;
let __t7: u64;
core::arch::asm!("lock cmpxchgq {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __v9, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t7, options(nostack, att_syntax));
__v2 = (__t6 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t7 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely!(!__v2) != 0 {
unsafe {
*__v1 = __v4;
}
}
__v15 = __v2 as i64;
__v13 = __v15;
if __v13 != 0 {
break;
}
}
first.is_null()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct memory_tier {
}


fn llist_add(new: *mut llist_node, head: *mut llist_head) -> bool {
llist_add_batch(new, new, head)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


fn llist_del_all(head: *mut llist_head) -> *mut llist_node {
let mut __v0: *mut llist_node;
let mut __ai_ptr: *mut *mut llist_node = unsafe { core::ptr::addr_of_mut!((*head).first) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<*mut llist_node>() as u64);
let mut __v2: *mut llist_node;
let mut __ret: *mut llist_node = core::ptr::null_mut::<llist_node>();
match (const { core::mem::size_of::<*mut llist_node>() }) as u64 {
1 => {
unsafe {
let __t0: u64;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u64 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t0 as u64) };
}
}
2 => {
unsafe {
let __t1: u64;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u64 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t1 as u64) };
}
}
4 => {
unsafe {
let __t2: u64;
core::arch::asm!("xchgl {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t2 as u64) };
}
}
8 => {
unsafe {
let __t3: u64;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t3 as u64) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v2 = __ret;
__v0 = __v2;
__v0
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
struct kmem_cache {
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


#[inline(always)]
fn __static_cpu_has__const1(bit: u16) -> bool {
unsafe {
let mut __slate_asm_target: usize = 0usize;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\tjmp 6f\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 3*32+21)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tjmp {4}\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte {0}\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\t\n775:\n.popsection\n.pushsection .altinstr_aux,\"ax\"\n6:\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n testb ${1}, {2}+112(%rip)\n jnz {3}\n jmp {4}\n.popsection\n", const 519 as u16, const 128 as i32, sym boot_cpu_data, label {
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

#[inline(always)]
fn __static_cpu_has__const2(bit: u16) -> bool {
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


// uses I/O bitmap
// 32-bit address space on 64 bits
fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit__2!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
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
struct mnt_idmap {
}


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


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
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
struct seqcount_raw_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_rwlock {
    seqcount: seqcount,
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
struct seqcount_mutex {
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


// SEQCOUNT_LOCKNAME()	- Instantiate seqcount_LOCKNAME_t and helpers
// seqprop_LOCKNAME_*()	- Property accessors for seqcount_LOCKNAME_t
//
// @lockname:		"LOCKNAME" part of seqcount_LOCKNAME_t
// @locktype:		LOCKNAME canonical C data type
// @preemptible:	preemptibility of above locktype
// @lockbase:		prefix for associated lock/unlock
// 									 * Re-read the sequence counter since the (possibly			 * preempted) writer made progress.
// PREEMPT_RT relies on the above LOCK+UNLOCK
fn __seqprop_ptr(s: *mut seqcount) -> *mut seqcount {
s
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


fn __seqprop_const_ptr(s: &seqcount) -> *const seqcount {
s as *const seqcount
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


fn __seqprop_sequence(s: &seqcount) -> u32 {
let mut __v0: u32;
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_92() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount)).sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_93() };
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


fn __seqprop_preemptible(s: *const seqcount) -> bool {
false
}

#[inline(always)]
fn __seqprop_raw_spinlock_ptr(s: &mut seqcount_raw_spinlock) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_raw_spinlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_raw_spinlock_const_ptr(s: &seqcount_raw_spinlock) -> *const seqcount {
unsafe { core::ptr::addr_of!((*(s as *const seqcount_raw_spinlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_raw_spinlock_sequence(s: &seqcount_raw_spinlock) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_94() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_raw_spinlock)).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_95() };
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
}
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if false && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_96() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_raw_spinlock)).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_97() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
}

#[inline(always)]
fn __seqprop_raw_spinlock_preemptible(s: &seqcount_raw_spinlock) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
false
}

#[inline(always)]
fn __seqprop_spinlock_ptr(s: &mut seqcount_spinlock) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_spinlock)).seqcount) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
}


#[inline(always)]
fn __seqprop_spinlock_const_ptr(s: *const seqcount_spinlock) -> *const seqcount {
unsafe { core::ptr::addr_of!((*s).seqcount) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


#[inline(always)]
fn __seqprop_spinlock_sequence(s: *const seqcount_spinlock) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_98() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_99() };
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
}
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if false && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_100() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_101() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
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


#[inline(always)]
fn __seqprop_spinlock_preemptible(s: &seqcount_spinlock) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
false
}

#[inline(always)]
fn __seqprop_rwlock_ptr(s: &mut seqcount_rwlock) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_rwlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_rwlock_const_ptr(s: &seqcount_rwlock) -> *const seqcount {
unsafe { core::ptr::addr_of!((*(s as *const seqcount_rwlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_rwlock_sequence(s: &seqcount_rwlock) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_102() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_rwlock)).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_103() };
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
}
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if false && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_104() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_rwlock)).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_105() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
}

#[inline(always)]
fn __seqprop_rwlock_preemptible(s: &seqcount_rwlock) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
false
}

#[inline(always)]
fn __seqprop_mutex_ptr(s: &mut seqcount_mutex) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_mutex)).seqcount) }
}

#[inline(always)]
fn __seqprop_mutex_const_ptr(s: &seqcount_mutex) -> *const seqcount {
unsafe { core::ptr::addr_of!((*(s as *const seqcount_mutex)).seqcount) }
}

#[inline(always)]
fn __seqprop_mutex_sequence(s: &seqcount_mutex) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_106() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_mutex)).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_107() };
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
}
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if true && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_108() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_mutex)).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_109() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
}

#[inline(always)]
fn __seqprop_mutex_preemptible(s: &seqcount_mutex) -> bool {
if IS_ENABLED!() == 0 {
return true;
}
false
}

// __read_seqcount_begin() - begin a seqcount_t read section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount_begin() - begin a seqcount_t read section w/o lockdep
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
fn do___read_seqcount_retry(s: *const seqcount, start: u32) -> i32 {
kcsan_atomic_next(0i32);
let mut __v0: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_110() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).sequence) as *const u32) };
((__v0 != start) as i64) as i32
}

fn do_read_seqcount_retry(s: *const seqcount, start: u32) -> i32 {
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
do___read_seqcount_retry(s, start)
}

// raw_write_seqcount_begin() - start a seqcount_t write section w/o lockdep
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: check write_seqcount_begin()
fn do_raw_write_seqcount_begin(__v0: *mut seqcount) {
kcsan_nestable_atomic_begin();
let __v1: u32 = unsafe { (*__v0).sequence };
unsafe {
(*__v0).sequence = __v1.wrapping_add(1);
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


// raw_write_seqcount_end() - end a seqcount_t write section w/o lockdep
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: check write_seqcount_end()
fn do_raw_write_seqcount_end(__v2: *mut seqcount) {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let __v3: u32 = unsafe { (*__v2).sequence };
unsafe {
(*__v2).sequence = __v3.wrapping_add(1);
}
kcsan_nestable_atomic_end();
}

// write_seqcount_begin() - start a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: sequence counter write side sections must be serialized and
// non-preemptible. Preemption will be automatically disabled if and
// only if the seqcount write serialization lock is associated, and
// preemptible.  If readers can be invoked from hardirq or softirq
// context, interrupts or bottom halves must be respectively disabled.
// write_seqcount_end() - end a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: Preemption will be automatically re-enabled if and only if
// the seqcount write serialization lock is associated, and preemptible.
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
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


fn refcount_set(r: *mut refcount_struct, n: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*r).refs) }, n);
}

fn __refcount_add(i: i32, r: *mut refcount_struct, oldp: *mut i32) {
let old: i32 = atomic_fetch_add_relaxed(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if unlikely!(old == 0) != 0 {
unsafe { refcount_warn_saturate(r, 2u32) };
} else {
if unlikely!(old < 0 || old.wrapping_add(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 1u32) };
}
}
}

fn __refcount_inc(r: *mut refcount_struct, oldp: *mut i32) {
__refcount_add(1i32, r, oldp);
}

fn refcount_inc(r: *mut refcount_struct) {
__refcount_inc(r, core::ptr::null_mut::<i32>());
}

fn __refcount_sub_and_test(i: i32, r: *mut refcount_struct, oldp: *mut i32) -> bool {
let old: i32 = atomic_fetch_sub_release(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if old > 0 && old == i {
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
return true;
}
if unlikely!(old <= 0 || old.wrapping_sub(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 3u32) };
}
false
}

fn __refcount_dec_and_test(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_sub_and_test(1i32, r, oldp)
}

fn refcount_dec_and_test(r: *mut refcount_struct) -> bool {
__refcount_dec_and_test(r, core::ptr::null_mut::<i32>())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


fn kref_init(kref: *mut kref) {
refcount_set(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }, 1i32);
}

fn kref_get(kref: *mut kref) {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*kref).refcount) });
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


fn kref_put(kref: *mut kref, release: Option<unsafe extern "C" fn(*mut kref)>) -> i32 {
if refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }) {
unsafe { release.unwrap()(kref) };
return 1;
}
0
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
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
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
struct futex_unlock_cs_range {
    start_ip: u64,
    len: u32,
    pop_size32: u32,
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
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
struct timer_list {
    entry: hlist_node,
    expires: u64,
    function: Option<unsafe extern "C" fn(*mut timer_list)>,
    flags: u32,
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
struct posix_acl {
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
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
struct wake_irq {
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
struct dev_pm_qos {
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
    __slate_bits_0: __slate_bits::__SlateBits324U0,
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
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
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
fn test_tsk_thread_flag(tsk: *mut task_struct, flag: i32) -> i32 {
test_ti_thread_flag(task_thread_info!(tsk), flag)
}

#[inline(always)]
fn _cond_resched() -> i32 {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__cond_resched as *const ()) };
unsafe { __v0.unwrap()() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__cond_resched_116: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__cond_resched) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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


fn rwsem_assert_held_write_nolockdep(sem: *const rw_semaphore) {
let mut __v0: i64;
let __ret_warn_on: i32 = ((atomic_long_read(unsafe { core::ptr::addr_of!((*sem).count) }) as u64) & 1u64 == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("156: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 156b - ., 3; .popsection\n# {0}", const 156 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_rwsem_assert_held_write_nolockdep_1, sym __slate_asm_string_rwsem_assert_held_write_nolockdep_2, const 87 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("157: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 157b - ., 4; .popsection\n# {0}", const 157 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

fn rwsem_assert_held_write(sem: *const rw_semaphore) {
if IS_ENABLED!() != 0 {
lockdep_assert_held_write!(sem);
} else {
rwsem_assert_held_write_nolockdep(sem);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
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
struct completion {
    done: u32,
    wait: swait_queue_head,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
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


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
fn jiffies_to_msecs(j: u64) -> u32 {
1u64.wrapping_mul(j) as u32
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
union __SlateRecord_mempolicy_h_54 {
    cpuset_mems_allowed: __SlateRecord_nodemask_types_h_17,
    user_nodemask: __SlateRecord_nodemask_types_h_17,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
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


// Tracks mm_cpumask
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
struct ptdesc {
    pt_flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_579,
    __page_mapping: u64,
    __slate_anon_3: __SlateRecord_mm_types_h_589,
    __slate_anon_4: __SlateRecord_mm_types_h_598,
    __page_type: u32,
    __page_refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


fn ptdesc_pmd_pts_inc(ptdesc: *mut ptdesc) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*ptdesc).__slate_anon_3.pt_share_count) });
}

fn ptdesc_pmd_pts_dec(ptdesc: *mut ptdesc) {
atomic_dec(unsafe { core::ptr::addr_of_mut!((*ptdesc).__slate_anon_3.pt_share_count) });
}

fn ptdesc_pmd_pts_count(ptdesc: *const ptdesc) -> i32 {
atomic_read(unsafe { core::ptr::addr_of!((*ptdesc).__slate_anon_3.pt_share_count) })
}

fn ptdesc_pmd_is_shared(ptdesc: *mut ptdesc) -> bool {
ptdesc_pmd_pts_count(ptdesc as *const ptdesc) != 0
}

// page_private can be used on tail pages.  However, PagePrivate is only
// checked by the VM on the head page.  So page_private on the tail pages
// should be used for data that's ancillary to the head page (eg attaching
// buffer heads to tail pages after attaching buffer heads to the head page)
fn folio_get_private(folio: *const folio) -> *mut () {
unsafe { (*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private }
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
    __slate_bits_0: __slate_bits::__SlateBits454U0,
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
#[inline(always)]
fn compound_info_has_mask() -> bool {
// Limit mask usage to HugeTLB vmemmap optimization (HVO) where it
// makes a difference.
//
// The approach with mask would work in the wider set of conditions,
// but it requires validating that struct pages are naturally aligned
// for all orders up to the MAX_FOLIO_ORDER, which can be tricky.
if IS_ENABLED__2!() == 0 {
return false;
}
is_power_of_2(core::mem::size_of::<page>() as u64)
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
    pub struct __SlateBits636U0 {
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
    pub struct __SlateBits512U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits324U0 {
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
    pub struct __SlateBits723U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits859U0 {
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
    pub struct __SlateBits327U0 {
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
    pub struct __SlateBits108U0 {
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
    pub struct __SlateBits130U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits752U0 {
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
    pub struct __SlateBits749U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits752U1 {
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
    pub struct __SlateBits752U2 {
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
    pub struct __SlateBits752U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits674U0 {
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
    pub struct __SlateBits107U0 {
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
    pub struct __SlateBits454U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits156U0 {
        #[bits(24)]
        pub intel_native_model_id: u32,
        #[bits(8)]
        pub intel_type: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits157U0 {
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
    pub struct __SlateBits146U0 {
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
    pub struct __SlateBits147U0 {
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
    pub struct __SlateBits144U0 {
        #[bits(1)]
        pub initialized: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits477U0 {
        #[bits(1)]
        pub fullmm: u32,
        #[bits(1)]
        pub need_flush_all: u32,
        #[bits(1)]
        pub freed_tables: u32,
        #[bits(1)]
        pub delayed_rmap: u32,
        #[bits(1)]
        pub cleared_ptes: u32,
        #[bits(1)]
        pub cleared_pmds: u32,
        #[bits(1)]
        pub cleared_puds: u32,
        #[bits(1)]
        pub cleared_p4ds: u32,
        #[bits(1)]
        pub vma_exec: u32,
        #[bits(1)]
        pub vma_huge: u32,
        #[bits(1)]
        pub vma_pfn: u32,
        #[bits(1)]
        pub unshared_tables: u32,
        #[bits(1)]
        pub fully_unshared_tables: u32,
        #[bits(3, access = na)]
        pub __slate_pad_13: u8,
    }
}

#[inline(always)]
fn _compound_head(page: *const page) -> u64 {
let mut info: u64 = 0;
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_187() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*page).__slate_anon_1.__slate_anon_2.compound_info) as *const u64) };
info = __v0;
let mut mask: u64 = 0;
if !compound_info_has_mask() {
// Bit 0 encodes PageTail()
if info & 1 != 0 {
return info.wrapping_sub(1);
}
return page as u64;
}
// If compound_info_has_mask() is true the rest of the info encodes
// the mask that converts the address of the tail page to the head page.
//
// No need to clear bit 0 in the mask as 'page' always has it clear.
//
// Let's do it in a branchless manner.
// Non-tail: -1UL, Tail: 0
mask = (info & 1).wrapping_sub(1);
// Non-tail: -1UL, Tail: info
mask |= info;
(page as u64) & mask
}

#[inline(always)]
fn set_compound_head(tail: *mut page, head: *const page, order: u32) {
let mut shift: u32 = 0;
let mut mask: u64 = 0;
if !compound_info_has_mask() {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_188() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*tail).__slate_anon_1.__slate_anon_2.compound_info, (head as u64) | 1);
if true {
break;
}
}
return;
}
// If the size of struct page is power-of-2, bits [shift:0] of the
// virtual address of compound head are zero.
//
// Calculate mask that can be applied to the virtual address of
// the tail page to get address of the head page.
shift = order.wrapping_add(order_base_2!(core::mem::size_of::<page>() as u64) as u32);
mask = GENMASK__3!(63i32, shift);
// Bit 0 encodes PageTail()
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_189() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*tail).__slate_anon_1.__slate_anon_2.compound_info, mask | 1);
if true {
break;
}
}
}

#[inline(always)]
fn clear_compound_head(page: *mut page) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_190() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__8!((*page).__slate_anon_1.__slate_anon_2.compound_info, 0i32);
if true {
break;
}
}
}

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

fn const_folio_flags(folio: *const folio, n: u32) -> *const u64 {
let page: *const page = unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) };
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
unsafe { core::ptr::addr_of!((*page.wrapping_offset(n as isize)).flags.f) }
}

fn folio_flags(folio: *mut folio, n: u32) -> *mut u64 {
let page: *mut page = unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) };
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
unsafe { core::ptr::addr_of_mut!((*page.wrapping_offset(n as isize)).flags.f) }
}

// Which page is the flag stored in
// Macros to create function definitions for page flags
#[inline(always)]
fn folio_test_locked(folio: *const folio) -> bool {
test_bit!(0i32, const_folio_flags(folio, 0u32)) != 0
}

#[inline(always)]
fn __folio_set_locked(folio: *mut folio) {
__set_bit!(0i32, folio_flags(folio, 0u32));
}

#[inline(always)]
fn __folio_clear_locked(folio: *mut folio) {
__clear_bit!(0i32, folio_flags(folio, 0u32));
}

#[inline(always)]
fn __folio_clear_reserved(folio: *mut folio) {
__clear_bit!(13i32, folio_flags(folio, 0u32));
}

fn folio_test_hwpoison(folio: *const folio) -> bool {
false
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
fn folio_test_anon(folio: *const folio) -> bool {
((unsafe { (*folio).__slate_anon_0.__slate_anon_0.mapping }) as u64) & 1 != 0
}

#[inline(always)]
fn PageAnonNotKsm(page: *const page) -> bool {
let flags: u64 = (unsafe { (*page_folio__2!(page)).__slate_anon_0.__slate_anon_0.mapping }) as u64;
flags & 3u64 == 1
}

#[inline(always)]
fn PageAnon(page: *const page) -> bool {
folio_test_anon(page_folio__2!(page))
}

#[inline(always)]
fn __folio_mark_uptodate(folio: *mut folio) {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
__set_bit!(3i32, folio_flags(folio, 0u32));
}

#[inline(always)]
fn folio_test_head(folio: *const folio) -> bool {
test_bit!(6i32, const_folio_flags(folio, 0u32)) != 0
}

#[inline(always)]
fn PageHead(page: *const page) -> i32 {
PF_POISONED_CHECK__2!(page);
test_bit__3!(6i32, unsafe { core::ptr::addr_of!((*page).flags.f) })
}

#[inline(always)]
fn __folio_set_head(folio: *mut folio) {
__set_bit!(6i32, folio_flags(folio, 0u32));
}

fn folio_test_large(folio: *const folio) -> bool {
folio_test_head(folio)
}

fn page_type_has_type(page_type: i32) -> bool {
page_type < -16777216i32
}

fn page_mapcount_is_type(mapcount: u32) -> bool {
page_type_has_type(mapcount.wrapping_sub(1) as i32)
}

#[inline(always)]
fn PageBuddy(page: *const page) -> i32 {
(data_race!((unsafe { (*page).__slate_anon_2.page_type }) >> 24i32) == 240) as i32
}

#[inline(always)]
fn folio_test_hugetlb(folio: *const folio) -> bool {
data_race!((unsafe { (*folio).__slate_anon_0.page.__slate_anon_2.page_type }) >> 24i32) == 244
}

#[inline(always)]
fn __folio_set_hugetlb(folio: *mut folio) {
if folio_test_hugetlb(folio as *const folio) {
return;
}
VM_BUG_ON_FOLIO!();
unsafe {
(*folio).__slate_anon_0.page.__slate_anon_2.page_type = 4093640704u32;
}
}

#[inline(always)]
fn __folio_clear_hugetlb(folio: *mut folio) {
if (unsafe { (*folio).__slate_anon_0.page.__slate_anon_2.page_type }) == !0 {
return;
}
VM_BUG_ON_FOLIO!();
unsafe {
(*folio).__slate_anon_0.page.__slate_anon_2.page_type = !0;
}
}

fn PageHuge(page: *const page) -> bool {
folio_test_hugetlb(page_folio__2!(page))
}

#[inline(always)]
fn PageAnonExclusive(mut page: *const page) -> i32 {
VM_BUG_ON_PGFLAGS!();
// HugeTLB stores this information on the head page; THP keeps it per
// page
if PageHuge(page) {
page = compound_head!(page);
}
let __v0: bool = if true && false {
let mut __v1: *const page;
__v1 = page;
((unsafe { core::ptr::addr_of!((*__v1).flags.f) }) as u64) != 0
} else {
false
};
let __v2: i32 = if __v0 && false {
let mut __v3: *const page;
__v3 = page;
const_test_bit(11u64, (unsafe { core::ptr::addr_of!((*__v3).flags.f) }) as *const u64) as i32
} else {
let mut __v4: *const page;
__v4 = page;
_test_bit(11u64, (unsafe { core::ptr::addr_of!((*__v4).flags.f) }) as *const u64) as i32
};
__v2
}

#[inline(always)]
fn SetPageAnonExclusive(page: *mut page) {
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
set_bit(11i64, (unsafe { core::ptr::addr_of_mut!((*PF_ANY!(page)).flags.f) }) as *mut u64);
}

#[inline(always)]
fn ClearPageAnonExclusive(page: *mut page) {
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
clear_bit(11i64, (unsafe { core::ptr::addr_of_mut!((*PF_ANY!(page)).flags.f) }) as *mut u64);
}

#[inline(always)]
fn __ClearPageAnonExclusive(page: *mut page) {
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
let __v0: bool = if true && false {
let mut __v1: *mut page;
__v1 = page;
((unsafe { core::ptr::addr_of_mut!((*__v1).flags.f) }) as u64) != 0
} else {
false
};
if __v0 && false {
let mut __v2: *mut page;
__v2 = page;
generic___clear_bit(11u64, (unsafe { core::ptr::addr_of_mut!((*__v2).flags.f) }) as *mut u64);
} else {
let mut __v3: *mut page;
__v3 = page;
___clear_bit(11u64, (unsafe { core::ptr::addr_of_mut!((*__v3).flags.f) }) as *mut u64);
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


fn memdesc_zonenum(flags: *const __SlateRecord_mm_types_h_41) -> u32 {
ASSERT_EXCLUSIVE_BITS!((*flags).f, 3u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64));
((unsafe { (*flags).f }) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64) & 3u64) as u32
}

core::arch::global_asm!(".weak arch_alloc_bootmem_huge_page\n.set arch_alloc_bootmem_huge_page, {}", sym __alloc_bootmem_huge_page);

core::arch::global_asm!(".section\t\".initcall4.init\", \"a\"\t\t\n__initcall__kmod_hugetlb__726_4187_hugetlb_init4:\t\t\t\n.long\t{0} - .\t\n.previous\t\t\t\t\t\n", sym hugetlb_init, options(att_syntax));

fn folio_zonenum(folio: *const folio) -> u32 {
memdesc_zonenum(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags) })
}

fn memdesc_is_zone_device(mdf: *const __SlateRecord_mm_types_h_41) -> bool {
false
}

fn folio_is_zone_device(folio: *const folio) -> bool {
memdesc_is_zone_device(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags) })
}

fn folio_is_zone_movable(folio: *const folio) -> bool {
(folio_zonenum(folio) as u32) == 3
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


fn zone_to_nid(zone: *const zone) -> i32 {
unsafe { (*zone).node }
}

// for_each_zone - helper macro to iterate over all memory zones
// @zone: pointer to struct zone variable
//
// The user only needs to declare the zone variable, for_each_zone
// fills it in.
fn zonelist_zone(zoneref: *mut zoneref) -> *mut zone {
unsafe { (*zoneref).zone }
}

fn zonelist_zone_idx(zoneref: *const zoneref) -> i32 {
unsafe { (*zoneref).zone_idx }
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

// for_each_zone_zonelist_nodemask - helper macro to iterate over valid zones in a zonelist at or below a given zone index and within a nodemask
// @zone: The current zone in the iterator
// @z: The current pointer within zonelist->_zonerefs being iterated
// @zlist: The zonelist being iterated
// @highidx: The zone index of the highest zone to return
// @nodemask: Nodemask allowed by the allocator
//
// This iterator iterates though all zones at or below a given zone index and
// within a given nodemask
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

fn numa_mem_id() -> i32 {
numa_node_id()
}

// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
fn gfp_zone(flags: u32) -> u32 {
let mut z: u32 = 0;
let bit: i32 = (flags & ((BIT!(0i32) as u32) | (BIT!(1i32) as u32) | (BIT!(2i32) as u32) | (BIT!(3i32) as u32))) as i32;
z = ((2i32 | 0i32 << BIT!(0i32).wrapping_mul(2) | 2i32 << BIT!(1i32).wrapping_mul(2) | 1i32 << BIT!(2i32).wrapping_mul(2) | 2i32 << BIT!(3i32).wrapping_mul(2) | 0i32 << (BIT!(3i32) | BIT!(0i32)).wrapping_mul(2) | 3i32 << (BIT!(3i32) | BIT!(1i32)).wrapping_mul(2) | 1i32 << (BIT!(3i32) | BIT!(2i32)).wrapping_mul(2)) >> bit.wrapping_mul(2) & 3i32) as u32;
VM_BUG_ON!();
z
}

fn gfp_zonelist(flags: u32) -> i32 {
if unlikely__3!(flags & (BIT!(21i32) as u32)) != 0 {
return 1;
}
0
}

fn node_zonelist(nid: i32, flags: u32) -> *mut zonelist {
unsafe { (*NODE_DATA!(nid)).node_zonelists.as_mut_ptr() as *mut zonelist }.wrapping_offset(gfp_zonelist(flags) as isize)
}

fn pgalloc_tag_split(folio: *mut folio, old_order: i32, new_order: i32) {
}

fn mmap_assert_write_locked(mm: *const mm_struct) {
rwsem_assert_held_write(unsafe { core::ptr::addr_of!((*mm).__slate_anon_0.mmap_lock) });
}

// VMA locks do not behave like most ordinary locks found in the kernel, so we
// cannot quite have full lockdep tracking in the way we would ideally prefer.
//
// Read locks act as shared locks which exclude an exclusive lock being
// taken. We therefore mark these accordingly on read lock acquire/release.
//
// Write locks are acquired exclusively per-VMA, but released in a shared
// fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
// that write lock is released.
//
// We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
// is the fact that, of course, we do lockdep-track the mmap lock rwsem which
// must be held when taking a VMA write lock.
//
// We do, however, want to indicate that during either acquisition of a VMA
// write lock or detachment of a VMA that we require the lock held be exclusive,
// so we utilise lockdep to do so.
fn __vma_are_readers_excluded(refcnt: i32) -> bool {
// See the comment describing the vm_area_struct->vm_refcnt field for
// details of possible refcnt values.
(refcnt as u32) & 1073741824u32 != 0 && (refcnt as u32) <= 1073741825u32
}

fn __vma_refcount_put_return(vma: *mut vm_area_struct) -> u32 {
let mut oldcnt: i32 = 0;
if __refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*vma).vm_refcnt) }, core::ptr::addr_of_mut!(oldcnt)) {
return 0;
}
oldcnt.wrapping_sub(1) as u32
}

fn vma_refcount_put(vma: *mut vm_area_struct) {
// Use a copy of vm_mm in case vma is freed after we drop vm_refcnt.
let mm: *mut mm_struct = unsafe { (*vma).vm_mm };
let mut newcnt: i32 = 0;
__vma_lockdep_release_read!();
newcnt = __vma_refcount_put_return(vma) as i32;
// __vma_start_exclude_readers() may be sleeping waiting for readers to
// drop their reference count, so wake it up if we were the last reader
// blocking it from being acquired.
//
// We may be raced by other readers temporarily incrementing the
// reference count, though the race window is very small, this might
// cause spurious wakeups.
let __v1: bool = newcnt != 0 && __vma_are_readers_excluded(newcnt);
if __v1 {
unsafe { rcuwait_wake_up(core::ptr::addr_of_mut!((*mm).__slate_anon_0.vma_writer_wait)) };
}
}

fn vma_end_read(vma: *mut vm_area_struct) {
vma_refcount_put(vma);
}

fn __vma_raw_mm_seqnum(vma: &vm_area_struct) -> u32 {
let mm: *const mm_struct = (unsafe { (*(vma as *const vm_area_struct)).vm_mm }) as *const mm_struct;
// We must hold an exclusive write lock for this access to be valid.
mmap_assert_write_locked((unsafe { (*(vma as *const vm_area_struct)).vm_mm }) as *const mm_struct);
unsafe { (*mm).__slate_anon_0.mm_lock_seq.sequence }
}

fn __is_vma_write_locked(vma: &vm_area_struct) -> bool {
// current task is holding mmap_write_lock, both vma->vm_lock_seq and
// mm->mm_lock_seq can't be concurrently modified.
(unsafe { (*(vma as *const vm_area_struct)).vm_lock_seq }) == __vma_raw_mm_seqnum(vma)
}

fn vma_assert_write_locked(vma: *mut vm_area_struct) {
VM_WARN_ON_ONCE_VMA!();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits512U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


fn __ref_is_percpu(r#ref: *mut percpu_ref, percpu_countp: &mut *mut u64) -> bool {
let mut percpu_ptr: u64 = 0;
// The value of @ref->percpu_count_ptr is tested for
// !__PERCPU_REF_ATOMIC, which may be set asynchronously, and then
// used as a pointer.  If the compiler generates a separate fetch
// when using it as a pointer, __PERCPU_REF_ATOMIC may be set in
// between contaminating the pointer value, meaning that
// READ_ONCE() is required when fetching it.
//
// The dependency ordering from the READ_ONCE() pairs
// with smp_store_release() in __percpu_ref_switch_to_percpu().
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_211() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*r#ref).percpu_count_ptr) as *const u64) };
percpu_ptr = __v0;
// Theoretically, the following could test just ATOMIC; however,
// then we'd have to mask off DEAD separately as DEAD may be
// visible without ATOMIC if we race with percpu_ref_kill().  DEAD
// implies ATOMIC anyway.  Test them together.
if unlikely__4!(percpu_ptr & 3) != 0 {
return false;
}
unsafe {
*(percpu_countp as *mut *mut u64) = percpu_ptr as *mut u64;
}
true
}

fn percpu_ref_get_many(r#ref: *mut percpu_ref, nr: u64) {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (nr & 255) as u8;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (nr & 65535) as u16;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (nr & 4294967295) as u32;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = nr;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
atomic_long_add(nr as i64, unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) });
}
rcu_read_unlock();
}

fn percpu_ref_get(r#ref: *mut percpu_ref) {
percpu_ref_get_many(r#ref, 1u64);
}

fn percpu_ref_put_many(r#ref: *mut percpu_ref, nr: u64) {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (nr.wrapping_neg() & 255) as u8;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (nr.wrapping_neg() & 65535) as u16;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (nr.wrapping_neg() & 4294967295) as u32;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = nr.wrapping_neg();
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
if unlikely!(atomic_long_sub_and_test(nr as i64, unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) })) != 0 {
unsafe { unsafe { (*(*r#ref).data).release }.unwrap()(r#ref) };
}
}
rcu_read_unlock();
}

fn percpu_ref_put(r#ref: *mut percpu_ref) {
percpu_ref_put_many(r#ref, 1u64);
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


fn __page_ref_set(page: *mut page, v: i32) {
}

fn __page_ref_mod(page: *mut page, v: i32) {
}

fn __page_ref_mod_and_test(page: *mut page, v: i32, ret: i32) {
}

fn __page_ref_mod_unless(page: *mut page, v: i32, u: i32) {
}

fn __page_ref_freeze(page: *mut page, v: i32, ret: i32) {
}

fn __page_ref_unfreeze(page: *mut page, v: i32) {
}

fn page_ref_count(page: *const page) -> i32 {
atomic_read(unsafe { core::ptr::addr_of!((*page)._refcount) })
}

fn folio_ref_count(folio: *const folio) -> i32 {
page_ref_count(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) })
}

fn page_count(page: *const page) -> i32 {
folio_ref_count(page_folio__2!(page))
}

fn set_page_count(page: *mut page, v: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }, v);
if page_ref_tracepoint_active!() != 0 {
__page_ref_set(page, v);
}
}

fn page_ref_inc(page: *mut page) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*page)._refcount) });
if page_ref_tracepoint_active!() != 0 {
__page_ref_mod(page, 1i32);
}
}

fn folio_ref_inc(folio: *mut folio) {
page_ref_inc(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) });
}

fn page_ref_dec_and_test(page: *mut page) -> i32 {
let ret: i32 = atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }) as i32;
if page_ref_tracepoint_active!() != 0 {
__page_ref_mod_and_test(page, 1i32.wrapping_neg(), ret);
}
ret
}

fn page_ref_add_unless_zero(page: *mut page, nr: i32) -> bool {
let ret: bool = atomic_add_unless(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }, nr, 0i32);
if page_ref_tracepoint_active!() != 0 {
__page_ref_mod_unless(page, nr, ret as i32);
}
ret
}

fn folio_ref_add_unless_zero(folio: *mut folio, nr: i32) -> bool {
page_ref_add_unless_zero(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, nr)
}

fn folio_try_get(folio: *mut folio) -> bool {
folio_ref_add_unless_zero(folio, 1i32)
}

fn page_ref_freeze(page: *mut page, count: i32) -> i32 {
let ret: i32 = likely!(atomic_cmpxchg(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }, count, 0i32) == count) as i32;
if page_ref_tracepoint_active!() != 0 {
__page_ref_freeze(page, count, ret);
}
ret
}

fn folio_ref_freeze(folio: *mut folio, count: i32) -> i32 {
page_ref_freeze(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, count)
}

fn page_ref_unfreeze(page: *mut page, count: i32) {
VM_BUG_ON_PAGE!();
VM_BUG_ON!();
atomic_set_release(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }, count);
if page_ref_tracepoint_active!() != 0 {
__page_ref_unfreeze(page, count);
}
}

fn folio_ref_unfreeze(folio: *mut folio, count: i32) {
page_ref_unfreeze(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, count);
}

fn cc_get_mask() -> u64 {
0
}

#[inline(always)]
fn pte_uffd(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
0
}

#[inline(always)]
fn pte_mkuffd(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte;
}

#[inline(always)]
fn pte_clear_uffd(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte;
}

#[inline(always)]
fn pte_swp_mkuffd(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte;
}

#[inline(always)]
fn pte_swp_uffd(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
0
}

#[inline(always)]
fn pte_swp_clear_uffd(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte;
}

fn page_table_check_pte_clear(mm: *mut mm_struct, addr: u64, pte: __SlateRecord_pgtable_64_types_h_21) {
}

fn page_table_check_ptes_set(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21, pte: __SlateRecord_pgtable_64_types_h_21, nr: u32) {
}

fn pte_dirty(pte: __SlateRecord_pgtable_64_types_h_21) -> bool {
pte_flags(pte) & (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 58i32) != 0
}

fn pte_shstk(pte: __SlateRecord_pgtable_64_types_h_21) -> bool {
cpu_feature_enabled!(519i32) != 0 && pte_flags(pte) & (_AT!(u64, 1i32) << 1i32 | _AT!(u64, 1i32) << 6i32) == _AT!(u64, 1i32) << 6i32
}

fn pte_write(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
// Shadow stack pages are logically writable, but do not have
// _PAGE_RW.  Check for them separately from _PAGE_RW itself.
let __v0: bool = pte_flags(pte) & _AT!(u64, 1i32) << 1i32 != 0 || pte_shstk(pte);
__v0 as i32
}

fn pte_huge(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte_flags(pte) & _AT!(u64, 1i32) << 7i32) as i32
}

fn pte_pfn(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
let mut pfn: u64 = pte_val!(pte);
let __v0: u64 = protnone_mask(pfn);
pfn ^= __v0;
(pfn & (((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)) >> 12i32
}

fn pte_set_flags(pte: __SlateRecord_pgtable_64_types_h_21, set: u64) -> __SlateRecord_pgtable_64_types_h_21 {
let v: u64 = native_pte_val(pte);
return native_make_pte(v | set);
}

fn pte_clear_flags(pte: __SlateRecord_pgtable_64_types_h_21, clear: u64) -> __SlateRecord_pgtable_64_types_h_21 {
let v: u64 = native_pte_val(pte);
return native_make_pte(v & !clear);
}

fn mksaveddirty_shift(mut v: u64) -> u64 {
let cond: u64 = !v >> 1i32 & 1;
v |= (v >> 6i32 & cond) << 58i32;
let __v3: u64 = v & !(cond << 6i32);
v = __v3;
v
}

fn clear_saveddirty_shift(mut v: u64) -> u64 {
let cond: u64 = v >> 1i32 & 1;
v |= (v >> 58i32 & cond) << 6i32;
let __v3: u64 = v & !(cond << 58i32);
v = __v3;
v
}

fn pte_mksaveddirty(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
let mut v: u64 = native_pte_val(pte);
v = mksaveddirty_shift(v);
return native_make_pte(v);
}

fn pte_clear_saveddirty(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
let mut v: u64 = native_pte_val(pte);
v = clear_saveddirty_shift(v);
return native_make_pte(v);
}

fn pte_wrprotect(mut pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
pte = pte_clear_flags(pte, _AT!(u64, 1i32) << 1i32);
// Blindly clearing _PAGE_RW might accidentally create
// a shadow stack PTE (Write=0,Dirty=1). Move the hardware
// dirty value to the software bit, if present.
return pte_mksaveddirty(pte);
}

fn pte_mkdirty(mut pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
pte = pte_set_flags(pte, _AT!(u64, 1i32) << 6i32 | _AT!(u64, 0i32));
return pte_mksaveddirty(pte);
}

fn pte_mkyoung(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_set_flags(pte, _AT!(u64, 1i32) << 5i32);
}

fn pte_mkwrite_novma(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_set_flags(pte, _AT!(u64, 1i32) << 1i32);
}

fn pte_mkhuge(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_set_flags(pte, _AT!(u64, 1i32) << 7i32);
}

fn massage_pgprot(pgprot: pgprot) -> u64 {
let mut protval: u64 = pgprot_val!(pgprot);
if protval & _AT!(u64, 1i32) << 0i32 != 0 {
protval &= unsafe { __supported_pte_mask };
}
protval
}

fn check_pgprot(pgprot: pgprot) -> u64 {
let massaged_val: u64 = massage_pgprot(pgprot);
// mmdebug.h can not be included here because of dependencies
massaged_val
}

fn pfn_pte(page_nr: u64, pgprot: pgprot) -> __SlateRecord_pgtable_64_types_h_21 {
let mut pfn: u64 = page_nr << 12i32;
// This bit combination is used to mark shadow stacks
let mut __v0: i64;
let __ret_warn_on: i32 = (pgprot_val!(pgprot) & (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 1i32) == _AT!(u64, 1i32) << 6i32) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("220: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 220b - ., 3; .popsection\n# {0}", const 220 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pfn_pte_1, sym __slate_asm_string_pfn_pte_2, const 735 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("221: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 221b - ., 4; .popsection\n# {0}", const 221 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
let __v5: u64 = protnone_mask(pgprot_val!(pgprot));
pfn ^= __v5;
let __v8: u64 = pfn;
pfn = __v8 & (((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64);
return __pte!(pfn | check_pgprot(pgprot));
}

fn pte_modify(pte: __SlateRecord_pgtable_64_types_h_21, newprot: pgprot) -> __SlateRecord_pgtable_64_types_h_21 {
let mut val: u64 = pte_val!(pte);
let oldval: u64 = val;
let mut pte_result: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(pte_result) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
// Chop off the NX bit (if present), and add the NX portion of
// the newprot (if present):
let __v0: u64 = ((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 9i32 | _AT!(u64, 1i32) << 5i32 | (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 58i32) | _AT!(u64, 0i32) | _AT!(u64, cc_get_mask()) | _AT!(u64, 0i32) | _AT!(u64, 1i32) << 7i32;
val &= __v0;
let __v3: u64 = check_pgprot(newprot) & !(((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 9i32 | _AT!(u64, 1i32) << 5i32 | (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 58i32) | _AT!(u64, 0i32) | _AT!(u64, cc_get_mask()) | _AT!(u64, 0i32) | _AT!(u64, 1i32) << 7i32);
val |= __v3;
val = flip_protnone_guard(oldval, val, ((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64);
pte_result = __pte!(val);
// To avoid creating Write=0,Dirty=1 PTEs, pte_modify() needs to avoid:
//  1. Marking Write=0 PTEs Dirty=1
//  2. Marking Dirty=1 PTEs Write=0
//
// The first case cannot happen because the _PAGE_CHG_MASK will filter
// out any Dirty bit passed in newprot. Handle the second case by
// going through the mksaveddirty exercise. Only do this if the old
// value was Write=1 to avoid doing this on Shadow Stack PTEs.
if oldval & _AT!(u64, 1i32) << 1i32 != 0 {
pte_result = pte_mksaveddirty(pte_result);
} else {
pte_result = pte_clear_saveddirty(pte_result);
}
return pte_result;
}

fn native_set_pte(ptep: *mut __SlateRecord_pgtable_64_types_h_21, pte: __SlateRecord_pgtable_64_types_h_21) {
loop {
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_224() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__5!(*ptep, pte);
if true {
break;
}
}
}

fn native_pte_clear(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) {
native_set_pte(ptep, native_make_pte(0u64));
}

fn native_ptep_get_and_clear(xp: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
let mut __v0: u64;
let mut __ai_ptr: *mut u64 = unsafe { core::ptr::addr_of_mut!((*xp).pte) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u64>() as u64);
let mut __v2: u64;
let mut __ret: u64 = 0;
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
unsafe {
let __t0: u64;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u64 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
}
2 => {
unsafe {
let __t1: u64;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u64 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
}
4 => {
unsafe {
let __t2: u64;
core::arch::asm!("xchgl {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t2 as u64) };
}
}
8 => {
unsafe {
let __t3: u64;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v2 = __ret;
__v0 = __v2;
return native_make_pte(__v0);
}

fn native_set_pud(pudp: *mut __SlateRecord_pgtable_types_h_368, pud: __SlateRecord_pgtable_types_h_368) {
loop {
loop {
if !(__native_word__14!() || ((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_226() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__11!(*pudp, pud);
if true {
break;
}
}
}

fn native_pud_clear(pud: *mut __SlateRecord_pgtable_types_h_368) {
native_set_pud(pud, native_make_pud(0u64));
}

// Shift the offset up "too far" by TYPE bits, then down again
// The offset is inverted by a binary not operation to make the high
// physical bits set.
fn __pte_needs_invert(val: u64) -> bool {
val != 0 && val & _AT!(u64, 1i32) << 0i32 == 0
}

fn protnone_mask(val: u64) -> u64 {
if __pte_needs_invert(val) { !0 } else { 0 }
}

fn flip_protnone_guard(oldval: u64, mut val: u64, mask: u64) -> u64 {
// When a PTE transitions from NONE to !NONE or vice-versa
// invert the PFN part to stop speculation.
// pte_pfn undoes this when needed.
if (__pte_needs_invert(oldval) as i32) != (__pte_needs_invert(val) as i32) {
val = val & !mask | !val & mask;
}
val
}

// mprotect needs to preserve PAT and encryption bits when updating
// vm_page_prot
fn pte_none(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte.pte & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
}

fn pte_same(a: __SlateRecord_pgtable_64_types_h_21, b: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(a.pte == b.pte) as i32
}

fn pte_advance_pfn(pte: __SlateRecord_pgtable_64_types_h_21, nr: u64) -> __SlateRecord_pgtable_64_types_h_21 {
if __pte_needs_invert(pte_val!(pte)) {
return __pte!(pte_val!(pte).wrapping_sub(nr << 12i32));
}
return __pte!(pte_val!(pte).wrapping_add(nr << 12i32));
}

fn pte_present(a: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte_flags(a) & (_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 8i32)) as i32
}

fn pte_protnone(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte_flags(pte) & (_AT!(u64, 1i32) << 8i32 | _AT!(u64, 1i32) << 0i32) == _AT!(u64, 1i32) << 8i32) as i32
}

// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
fn pud_none(pud: __SlateRecord_pgtable_types_h_368) -> i32 {
(native_pud_val(pud) & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
}

fn pud_present(pud: __SlateRecord_pgtable_types_h_368) -> i32 {
(pud_flags(pud) & _AT!(u64, 1i32) << 0i32) as i32
}

fn pud_pgtable(pud: __SlateRecord_pgtable_types_h_368) -> *mut __SlateRecord_pgtable_64_types_h_22 {
__va!(pud_val!(pud) & pud_pfn_mask(pud)) as *mut __SlateRecord_pgtable_64_types_h_22
}

fn p4d_none(p4d: __SlateRecord_pgtable_types_h_342) -> i32 {
(native_p4d_val(p4d) & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
}

fn p4d_present(p4d: __SlateRecord_pgtable_types_h_342) -> i32 {
(p4d_flags(p4d) & _AT!(u64, 1i32) << 0i32) as i32
}

fn p4d_pgtable(p4d: __SlateRecord_pgtable_types_h_342) -> *mut __SlateRecord_pgtable_types_h_368 {
__va!(p4d_val!(p4d) & p4d_pfn_mask(p4d)) as *mut __SlateRecord_pgtable_types_h_368
}

fn p4d_index(address: u64) -> u64 {
address >> 39i32 & (unsafe { ptrs_per_p4d }.wrapping_sub(1) as u64)
}

fn pgd_present(pgd: __SlateRecord_pgtable_types_h_295) -> i32 {
if pgtable_l5_enabled__2!() == 0 {
return 1;
}
(pgd_flags(pgd) & _AT!(u64, 1i32) << 0i32) as i32
}

fn pgd_page_vaddr(pgd: __SlateRecord_pgtable_types_h_295) -> u64 {
__va!(pgd_val!(pgd) & (((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)) as u64
}

fn p4d_offset(pgd: *mut __SlateRecord_pgtable_types_h_295, address: u64) -> *mut __SlateRecord_pgtable_types_h_342 {
if pgtable_l5_enabled__2!() == 0 {
return pgd as *mut __SlateRecord_pgtable_types_h_342;
}
(pgd_page_vaddr(unsafe { *pgd }) as *mut __SlateRecord_pgtable_types_h_342).wrapping_offset(p4d_index(address) as isize)
}

fn pgd_none(pgd: __SlateRecord_pgtable_types_h_295) -> i32 {
if pgtable_l5_enabled__2!() == 0 {
return 0;
}
// There is no need to do a workaround for the KNL stray
// A/D bit erratum here.  PGDs only point to page tables
// except on 32-bit non-PAE which is not supported on
// KNL.
(native_pgd_val(pgd) == 0) as i32
}

fn ptep_get_and_clear(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
let pte: __SlateRecord_pgtable_64_types_h_21 = native_ptep_get_and_clear(ptep);
page_table_check_pte_clear(mm, addr, pte);
return pte;
}

fn ptep_set_wrprotect(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) {
// Avoid accidentally creating shadow stack PTEs
// (Write=0,Dirty=1).  Use cmpxchg() to prevent races with
// the hardware setting Dirty=1.
let mut old_pte: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(old_pte) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut new_pte: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(new_pte) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut __v10: __SlateRecord_pgtable_64_types_h_21;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_230() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(ptep as *const __SlateRecord_pgtable_64_types_h_21) };
old_pte = __v10;
loop {
new_pte = pte_wrprotect(old_pte);
let mut __v13: i64;
let __v0: *mut i64 = (unsafe { core::ptr::addr_of_mut!((*ptep).pte) }) as *mut i64;
let __v1: *mut i64 = core::ptr::addr_of_mut!(old_pte) as *mut i64;
kcsan_mb!();
instrument_atomic_read_write(__v0 as *const (), core::mem::size_of::<i64>() as u64);
instrument_read_write(__v1 as *const (), core::mem::size_of::<i64>() as u64);
let mut __v15: i64;
let mut __v2: bool = false;
let mut __v4: i64 = unsafe { *__v1 };
let mut __v5: i64 = unsafe { *(core::ptr::addr_of_mut!(new_pte) as *mut i64) };
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
let mut __v6: *mut u8 = __v0 as *mut u8;
unsafe {
let __t0: u8;
let __t1: u64;
core::arch::asm!("lock cmpxchgb {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __v6, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t1, options(nostack, att_syntax));
__v2 = (__t0 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, i64>(__t1 as u64) };
}
}
2 => {
let mut __v7: *mut u16 = __v0 as *mut u16;
unsafe {
let __t2: u8;
let __t3: u64;
core::arch::asm!("lock cmpxchgw {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __v7, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t3, options(nostack, att_syntax));
__v2 = (__t2 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, i64>(__t3 as u64) };
}
}
4 => {
let mut __v8: *mut u32 = __v0 as *mut u32;
unsafe {
let __t4: u8;
let __t5: u64;
core::arch::asm!("lock cmpxchgl {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __v8, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t5, options(nostack, att_syntax));
__v2 = (__t4 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, i64>(__t5 as u64) };
}
}
8 => {
let mut __v9: *mut u64 = __v0 as *mut u64;
unsafe {
let __t6: u8;
let __t7: u64;
core::arch::asm!("lock cmpxchgq {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __v9, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t7, options(nostack, att_syntax));
__v2 = (__t6 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, i64>(__t7 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely!(!__v2) != 0 {
unsafe {
*__v1 = __v4;
}
}
__v15 = __v2 as i64;
__v13 = __v15;
if __v13 != 0 {
break;
}
}
}

// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
fn update_mmu_cache(vma: *mut vm_area_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) {
}

fn pte_swp_exclusive(pte: __SlateRecord_pgtable_64_types_h_21) -> bool {
pte_flags(pte) & _AT!(u64, 1i32) << 3i32 != 0
}

fn pte_swp_clear_exclusive(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_clear_flags(pte, _AT!(u64, 1i32) << 3i32);
}

fn pte_swp_soft_dirty(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte_flags(pte) & _AT!(u64, 0i32)) as i32
}

fn pte_swp_clear_soft_dirty(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_clear_flags(pte, _AT!(u64, 0i32));
}

fn pmd_index(address: u64) -> u64 {
address >> 21i32 & 511u64
}

fn pud_index(address: u64) -> u64 {
address >> 30i32 & 511u64
}

fn pmd_offset(pud: *mut __SlateRecord_pgtable_types_h_368, address: u64) -> *mut __SlateRecord_pgtable_64_types_h_22 {
pud_pgtable(unsafe { *pud }).wrapping_offset(pmd_index(address) as isize)
}

fn pud_offset(p4d: *mut __SlateRecord_pgtable_types_h_342, address: u64) -> *mut __SlateRecord_pgtable_types_h_368 {
p4d_pgtable(unsafe { *p4d }).wrapping_offset(pud_index(address) as isize)
}

fn pgd_offset_pgd(pgd: *mut __SlateRecord_pgtable_types_h_295, address: u64) -> *mut __SlateRecord_pgtable_types_h_295 {
pgd.wrapping_offset(pgd_index!(address) as isize)
}

// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
fn set_ptes(mm: *mut mm_struct, addr: u64, mut ptep: *mut __SlateRecord_pgtable_64_types_h_21, mut pte: __SlateRecord_pgtable_64_types_h_21, mut nr: u32) {
page_table_check_ptes_set(mm, addr, ptep, pte, nr);
loop {
set_pte!(ptep, pte);
let __v1: u32 = nr.wrapping_sub(1);
nr = __v1;
if __v1 == 0 {
break;
}
let __v3: *mut __SlateRecord_pgtable_64_types_h_21 = ptep;
ptep = __v3.wrapping_offset(1 as isize);
pte = pte_next_pfn!(pte);
}
}

fn ptep_get(ptep: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
let mut __v0: __SlateRecord_pgtable_64_types_h_21;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_234() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(ptep as *const __SlateRecord_pgtable_64_types_h_21) };
return __v0;
}

fn ptep_get_lockless(ptep: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return ptep_get(ptep);
}

fn is_zero_pfn(pfn: u64) -> i32 {
(pfn == unsafe { zero_page_pfn }) as i32
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


fn folio_is_device_coherent(folio: *const folio) -> bool {
folio_is_zone_device(folio) && ((unsafe { (*(*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.pgmap).r#type }) as u32) == 2
}

fn folio_is_fsdax(folio: *const folio) -> bool {
folio_is_zone_device(folio) && ((unsafe { (*(*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.pgmap).r#type }) as u32) == 3
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
unsafe { __compiletime_assert_257() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("258: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 258b - ., 3; .popsection\n# {0}", const 258 as i32, options(nostack, att_syntax));
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
// kmalloc_obj - Allocate a single instance of the given type
// @VAR_OR_TYPE: Variable or type to allocate.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to a @VAR_OR_TYPE on success, or NULL
// on failure.
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
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_notifier_range {
    mm: *mut mm_struct,
    start: u64,
    end: u64,
    flags: u32,
    event: u32,
    owner: *mut (),
}


fn mm_has_notifiers(mm: *mut mm_struct) -> i32 {
unlikely__5!(unsafe { (*mm).__slate_anon_0.notifier_subscriptions }) as i32
}

fn mmu_notifier_range_blockable(range: *const mmu_notifier_range) -> bool {
(unsafe { (*range).flags }) & 1u32 != 0
}

fn mmu_notifier_invalidate_range_start(range: *mut mmu_notifier_range) {
might_sleep!();
lock_map_acquire!();
if mm_has_notifiers(unsafe { (*range).mm }) != 0 {
unsafe {
(*range).flags = (*range).flags | 1u32;
}
unsafe { __mmu_notifier_invalidate_range_start(range) };
}
lock_map_release!();
}

fn mmu_notifier_invalidate_range_end(range: *mut mmu_notifier_range) {
if mmu_notifier_range_blockable(range as *const mmu_notifier_range) {
might_sleep!();
}
if mm_has_notifiers(unsafe { (*range).mm }) != 0 {
unsafe { __mmu_notifier_invalidate_range_end(range) };
}
}

fn mmu_notifier_range_init(range: &mut mmu_notifier_range, event: u32, flags: u32, mm: *mut mm_struct, start: u64, end: u64) {
unsafe {
(*(range as *mut mmu_notifier_range)).event = event;
}
unsafe {
(*(range as *mut mmu_notifier_range)).mm = mm;
}
unsafe {
(*(range as *mut mmu_notifier_range)).start = start;
}
unsafe {
(*(range as *mut mmu_notifier_range)).end = end;
}
unsafe {
(*(range as *mut mmu_notifier_range)).flags = flags;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits327U0,
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
fn task_sigpending(p: *mut task_struct) -> i32 {
unlikely__2!(test_tsk_thread_flag(p, 1i32)) as i32
}

fn signal_pending(p: *mut task_struct) -> i32 {
// TIF_NOTIFY_SIGNAL isn't really a signal, but it requires the same
// behavior in terms of ensuring that we break out of wait loops
// so that notify signal callbacks can be processed.
if unlikely__2!(test_tsk_thread_flag(p, 2i32)) != 0 {
return 1;
}
task_sigpending(p)
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_376() };
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

fn folio_page_idx(folio: *const folio, page: *const page) -> u64 {
((unsafe { page.offset_from(core::ptr::addr_of!((*folio).__slate_anon_0.page) as *const page) }) as i64) as u64
}

// Despite the naming, these are FLAGS not bits.
// Bits set in the VMA until the stack is in its final location
// VMA basic access permission flags
// This mask represents all the VMA flag bits used by mlock
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


fn mm_flags_test(flag: i32, mm: *const mm_struct) -> bool {
test_bit__5!(flag, unsafe { (*core::ptr::addr_of!((*mm).__slate_anon_0.flags)).__mm_flags.as_ptr() as *const u64 }) != 0
}

// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
#[inline(always)]
fn vma_flags_test(flags: *const __SlateRecord_mm_types_h_871, bit: i32) -> bool {
let bitmap: *const u64 = unsafe { (*flags).__vma_flags.as_ptr() as *const u64 };
test_bit__5!(bit, bitmap) != 0
}

// Test whether any specified VMA flag is set, e.g.:
//
// if (vma_flags_test_any(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
#[inline(always)]
fn vma_test(vma: *const vm_area_struct, bit: i32) -> bool {
vma_flags_test(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) }, bit)
}

fn vma_is_accessible(vma: *const vm_area_struct) -> bool {
(unsafe { (*vma).__slate_anon_3.vm_flags }) & (BIT!(0i32) | BIT!(1i32) | BIT!(2i32)) != 0
}

fn folio_large_order(folio: *const folio) -> u32 {
((unsafe { (*folio).__slate_anon_1.__slate_anon_0._flags_1 }) & 255) as u32
}

fn folio_large_nr_pages(folio: *const folio) -> u64 {
(1i64 << folio_large_order(folio)) as u64
}

fn compound_order(page: *const page) -> u32 {
let folio: *const folio = page as *const folio;
if test_bit__3!(6i32, unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags.f) }) == 0 {
return 0;
}
folio_large_order(folio)
}

fn folio_order(folio: *const folio) -> u32 {
if !folio_test_large(folio) {
return 0;
}
folio_large_order(folio)
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
unsafe { __compiletime_assert_419() };
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
fn i_mmap_lock_write(mapping: *mut address_space) {
unsafe { down_write(core::ptr::addr_of_mut!((*mapping).i_mmap_rwsem)) };
}

fn i_mmap_trylock_write(mapping: *mut address_space) -> i32 {
unsafe { down_write_trylock(core::ptr::addr_of_mut!((*mapping).i_mmap_rwsem)) }
}

fn i_mmap_unlock_write(mapping: *mut address_space) {
unsafe { up_write(core::ptr::addr_of_mut!((*mapping).i_mmap_rwsem)) };
}

fn i_mmap_lock_read(mapping: *mut address_space) {
unsafe { down_read(core::ptr::addr_of_mut!((*mapping).i_mmap_rwsem)) };
}

fn i_mmap_unlock_read(mapping: *mut address_space) {
unsafe { up_read(core::ptr::addr_of_mut!((*mapping).i_mmap_rwsem)) };
}

fn i_mmap_assert_write_locked(mapping: *mut address_space) {
lockdep_assert_held_write!(unsafe { core::ptr::addr_of_mut!((*mapping).i_mmap_rwsem) });
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
fn i_size_read(inode: *const inode) -> i64 {
// Pairs with smp_store_release() in i_size_write()
let mut __v0: i64;
let mut ___p1: i64 = 0;
let mut __v1: i64;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_427() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*inode).i_size) as *const i64) };
___p1 = __v1;
loop {
if !__native_word__8!() {
unsafe { __compiletime_assert_428() };
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
    __slate_bits_0: __slate_bits::__SlateBits723U0,
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
    __slate_bits_0: __slate_bits::__SlateBits674U0,
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


fn put_page_testzero(page: *mut page) -> i32 {
VM_BUG_ON_PAGE!();
page_ref_dec_and_test(page)
}

fn folio_put_testzero(folio: *mut folio) -> i32 {
put_page_testzero(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) })
}

fn folio_large_mapcount(folio: *const folio) -> i32 {
VM_WARN_ON_FOLIO!();
atomic_read(unsafe { core::ptr::addr_of!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._large_mapcount) }).wrapping_add(1)
}

fn folio_mapcount(folio: *const folio) -> i32 {
let mut mapcount: i32 = 0;
if likely!(!folio_test_large(folio)) != 0 {
mapcount = atomic_read(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0._mapcount) }).wrapping_add(1);
if page_mapcount_is_type(mapcount as u32) {
mapcount = 0;
}
return mapcount;
}
folio_large_mapcount(folio)
}

fn folio_mapped(folio: *const folio) -> bool {
folio_mapcount(folio) >= 1
}

fn page_size(page: *const page) -> u64 {
_AC!() << 12i32 << compound_order(page)
}

// 127: arbitrary random number, small enough to assemble well
fn folio_get(folio: *mut folio) {
VM_BUG_ON_FOLIO!();
folio_ref_inc(folio);
}

fn folio_put(folio: *mut folio) {
if folio_put_testzero(folio) != 0 {
unsafe { __folio_put(folio) };
}
}

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
fn vma_flags_is_cow_mapping(flags: *const __SlateRecord_mm_types_h_871) -> bool {
vma_flags_test(flags, 5i32) && !vma_flags_test(flags, 3i32)
}

fn vma_is_cow_mapping(vma: *const vm_area_struct) -> bool {
vma_flags_is_cow_mapping(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) })
}

fn memdesc_nid(mdf: *const __SlateRecord_mm_types_h_41) -> i32 {
ASSERT_EXCLUSIVE_BITS!((*mdf).f, 63u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64));
((unsafe { (*mdf).f }) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64) & 63u64) as i32
}

fn folio_nid(folio: *const folio) -> i32 {
memdesc_nid(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags) })
}

fn folio_pgdat(folio: *const folio) -> *mut pglist_data {
NODE_DATA!(folio_nid(folio))
}

fn folio_pfn(folio: *const folio) -> u64 {
((unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page).offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64
}

fn pfn_folio(pfn: u64) -> *mut folio {
page_folio!(((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pfn as isize))
}

fn folio_mk_pte(folio: *const folio, pgprot: pgprot) -> __SlateRecord_pgtable_64_types_h_21 {
return pfn_pte(folio_pfn(folio), pgprot);
}

fn folio_has_pincount(folio: *const folio) -> bool {
if IS_ENABLED__2!() != 0 {
return folio_test_large(folio);
}
folio_order(folio) > 1
}

fn folio_maybe_dma_pinned(folio: *mut folio) -> bool {
if folio_has_pincount(folio as *const folio) {
return atomic_read((unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._pincount) }) as *const __SlateRecord_types_h_190) > 0;
}
// folio_ref_count() is signed. If that refcount overflows, then
// folio_ref_count() returns a negative value, and callers will avoid
// further incrementing the refcount.
//
// Here, for that overflow case, use the sign bit to count a little
// bit higher via unsigned math, and thus still get an accurate result.
(folio_ref_count(folio as *const folio) as u32) >= 1024u32
}

fn folio_needs_cow_for_dma(vma: *mut vm_area_struct, folio: *mut folio) -> bool {
VM_BUG_ON!();
if !mm_flags_test(27i32, (unsafe { (*vma).vm_mm }) as *const mm_struct) {
return false;
}
folio_maybe_dma_pinned(folio)
}

fn is_zero_page(page: *const page) -> bool {
is_zero_pfn(((unsafe { page.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64) != 0
}

fn is_zero_folio(folio: *const folio) -> bool {
is_zero_page(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) })
}

fn folio_is_longterm_pinnable(folio: *mut folio) -> bool {
// The zero page can be "pinned" but gets special handling.
if is_zero_folio(folio as *const folio) {
return true;
}
// Coherent device memory must always allow eviction.
if folio_is_device_coherent(folio as *const folio) {
return false;
}
// Filesystems can only tolerate transient delays to truncate and
// hole-punch operations
if folio_is_fsdax(folio as *const folio) {
return false;
}
// Otherwise, non-movable zone folios can be pinned.
!folio_is_zone_movable(folio as *const folio)
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

fn folio_nr_pages(folio: *const folio) -> u64 {
if !folio_test_large(folio) {
return 1;
}
folio_large_nr_pages(folio)
}

fn folio_size(folio: *const folio) -> u64 {
_AC!() << 12i32 << folio_order(folio)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_event_state {
    event: [u64; 75],
}


fn __count_vm_event(mut item: u32) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
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
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: u64 = 1;
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
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
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
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: u64 = 1;
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
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
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
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: u64 = 1;
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
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
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
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: u64 = 1;
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

fn lruvec_stat_mod_folio(folio: *mut folio, idx: u32, val: i32) {
unsafe { mod_node_page_state(folio_pgdat(folio as *const folio), idx, val as i64) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct zap_details {
    single_folio: *mut folio,
    skip_cows: bool,
    reclaim_pt: bool,
    reaping: bool,
    zap_flags: u32,
}


// Whether to drop the pte markers, for example, the uffd-wp information for
// file-backed memory.  This should only be specified when we will completely
// drop the page in the mm, either by truncation or unmapping of the vma.  By
// default, the flag is not set.
// Set in unmap_vmas() to indicate a final unmap call.  Only used by hugetlb
// Whether this change is for write protecting
// do wp
// Resolve wp
// Whether this change is for uffd RWP
// do rwp
// resolve rwp
fn mm_inc_nr_pmds(mm: *mut mm_struct) {
if mm_pmd_folded!() != 0 {
return;
}
atomic_long_add(512u64.wrapping_mul(core::mem::size_of::<__SlateRecord_pgtable_64_types_h_22>() as u64) as i64, unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.pgtables_bytes) });
}

fn mm_dec_nr_pmds(mm: *mut mm_struct) {
if mm_pmd_folded!() != 0 {
return;
}
atomic_long_sub(512u64.wrapping_mul(core::mem::size_of::<__SlateRecord_pgtable_64_types_h_22>() as u64) as i64, unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.pgtables_bytes) });
}

fn p4d_alloc(mm: *mut mm_struct, pgd: *mut __SlateRecord_pgtable_types_h_295, address: u64) -> *mut __SlateRecord_pgtable_types_h_342 {
let __v0: bool = unlikely__2!(pgd_none(unsafe { *pgd })) != 0 && (unsafe { __p4d_alloc(mm, pgd, address) }) != 0;
let __v1: *mut __SlateRecord_pgtable_types_h_342 = if __v0 {
core::ptr::null_mut::<__SlateRecord_pgtable_types_h_342>()
} else {
p4d_offset(pgd, address)
};
__v1
}

fn pud_alloc(mm: *mut mm_struct, p4d: *mut __SlateRecord_pgtable_types_h_342, address: u64) -> *mut __SlateRecord_pgtable_types_h_368 {
let __v0: bool = unlikely__2!(p4d_none(unsafe { *p4d })) != 0 && (unsafe { __pud_alloc(mm, p4d, address) }) != 0;
let __v1: *mut __SlateRecord_pgtable_types_h_368 = if __v0 {
core::ptr::null_mut::<__SlateRecord_pgtable_types_h_368>()
} else {
pud_offset(p4d, address)
};
__v1
}

fn pmd_alloc(mm: *mut mm_struct, pud: *mut __SlateRecord_pgtable_types_h_368, address: u64) -> *mut __SlateRecord_pgtable_64_types_h_22 {
let __v0: bool = unlikely__2!(pud_none(unsafe { *pud })) != 0 && (unsafe { __pmd_alloc(mm, pud, address) }) != 0;
let __v1: *mut __SlateRecord_pgtable_64_types_h_22 = if __v0 {
core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_22>()
} else {
pmd_offset(pud, address)
};
__v1
}

fn virt_to_ptdesc(x: *const ()) -> *mut ptdesc {
page_ptdesc!(((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__phys_addr_nodebug(x as u64) >> 12i32) as isize))
}

fn ptlock_ptr(ptdesc: *mut ptdesc) -> *mut spinlock {
unsafe { core::ptr::addr_of_mut!((*ptdesc).__slate_anon_4.ptl) }
}

fn ptep_lockptr(mm: *mut mm_struct, pte: *mut __SlateRecord_pgtable_64_types_h_21) -> *mut spinlock {
loop {
if IS_ENABLED!() != 0 {
unsafe { __compiletime_assert_479() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 512u64.wrapping_mul((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) > _AC!() << 12i32 {
unsafe { __compiletime_assert_480() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
ptlock_ptr(virt_to_ptdesc(pte as *const ()))
}

fn pmd_pgtable_page(pmd: *mut __SlateRecord_pgtable_64_types_h_22) -> *mut page {
let mask: u64 = !512u64.wrapping_mul(core::mem::size_of::<__SlateRecord_pgtable_64_types_h_22>() as u64).wrapping_sub(1);
virt_to_page!(((pmd as u64) & mask) as *mut ())
}

fn pmd_ptdesc(pmd: *mut __SlateRecord_pgtable_64_types_h_22) -> *mut ptdesc {
page_ptdesc!(pmd_pgtable_page(pmd))
}

fn pmd_lockptr(mm: *mut mm_struct, pmd: *mut __SlateRecord_pgtable_64_types_h_22) -> *mut spinlock {
ptlock_ptr(pmd_ptdesc(pmd))
}

fn pud_lockptr(mm: *mut mm_struct, pud: *mut __SlateRecord_pgtable_types_h_368) -> *mut spinlock {
unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.page_table_lock) }
}

fn vma_start_pgoff(vma: *const vm_area_struct) -> u64 {
unsafe { (*vma).vm_pgoff }
}

fn range_is_subset(outer_start: u64, outer_end: u64, inner_start: u64, inner_end: u64) -> bool {
outer_start <= inner_start && inner_end <= outer_end
}

fn range_in_vma(vma: *const vm_area_struct, start: u64, end: u64) -> bool {
if vma.is_null() {
return false;
}
range_is_subset(unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }, start, end)
}

fn vmf_error(err: i32) -> u32 {
if err == 12i32.wrapping_neg() {
return 1;
} else {
if err == 133i32.wrapping_neg() {
return 16;
}
}
2
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
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


fn flush_cache_range(vma: *mut vm_area_struct, start: u64, end: u64) {
}

fn is_vma_hugetlb_flags(flags: *const __SlateRecord_mm_types_h_871) -> bool {
vma_flags_test(flags, 22i32)
}

fn is_vm_hugetlb_page(vma: *const vm_area_struct) -> bool {
is_vma_hugetlb_flags(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) })
}

fn folio_change_private(folio: *mut folio, data: *mut ()) -> *mut () {
let old: *mut () = folio_get_private(folio as *const folio);
unsafe {
(*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private = data;
}
old
}

fn __filemap_get_folio(mapping: *mut address_space, index: u64, fgf_flags: u32, gfp: u32) -> *mut folio {
unsafe { __filemap_get_folio_mpol(mapping, index, fgf_flags, gfp, core::ptr::null_mut::<mempolicy>()) }
}

fn filemap_get_folio(mapping: *mut address_space, index: u64) -> *mut folio {
__filemap_get_folio(mapping, index, 0u32, 0u32)
}

fn filemap_lock_folio(mapping: *mut address_space, index: u64) -> *mut folio {
__filemap_get_folio(mapping, index, 2u32, 0u32)
}

fn linear_page_delta(vma: *const vm_area_struct, address: u64) -> u64 {
address.wrapping_sub(unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }) >> 12i32
}

fn linear_page_index(vma: *const vm_area_struct, address: u64) -> u64 {
linear_page_delta(vma, address).wrapping_add(vma_start_pgoff(vma))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
    folio: *mut folio,
    bit_nr: i32,
    wait: wait_queue_entry,
}


fn folio_trylock(folio: *mut folio) -> bool {
likely!(!test_and_set_bit_lock(0i64, folio_flags(folio, 0u32) as *mut u64)) != 0
}

fn folio_lock(folio: *mut folio) {
might_sleep!();
if !folio_trylock(folio) {
unsafe { __folio_lock(folio) };
}
}

fn folio_wait_locked(folio: *mut folio) {
if folio_test_locked(folio as *const folio) {
unsafe { folio_wait_bit(folio, 0i32) };
}
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
    __slate_bits_0: __slate_bits::__SlateBits752U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits752U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits752U2,
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
    __slate_bits_3: __slate_bits::__SlateBits752U3,
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
    __slate_bits_0: __slate_bits::__SlateBits749U0,
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


fn mpol_needs_cond_ref(pol: *mut mempolicy) -> i32 {
(!pol.is_null() && ((unsafe { (*pol).flags }) as i32) & 1i32 != 0) as i32
}

fn mpol_cond_put(pol: *mut mempolicy) {
if mpol_needs_cond_ref(pol) != 0 {
unsafe { __mpol_put(pol) };
}
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
#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
    entries: *mut desc_struct,
    nr_entries: u32,
    slot: i32,
}


// Init a new mm.  Used on mm copies, like at fork()
// and on mm's that are brand-new, like at execve().
fn cpusets_enabled() -> bool {
static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(cpusets_enabled_key) }) != 0
}

fn __cpuset_zone_allowed(z: *mut zone, gfp_mask: u32) -> bool {
unsafe { cpuset_current_node_allowed(zone_to_nid(z as *const zone), gfp_mask) }
}

fn cpuset_zone_allowed(z: *mut zone, gfp_mask: u32) -> bool {
if cpusets_enabled() {
return __cpuset_zone_allowed(z, gfp_mask);
}
true
}

fn read_mems_allowed_begin() -> u32 {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(cpusets_pre_enable_key) }) == 0 {
return 0;
}
read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!((*get_current()).mems_allowed_seq) })
}

fn read_mems_allowed_retry(seq: u32) -> bool {
if static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(cpusets_enabled_key) }) == 0 {
return false;
}
read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!((*get_current()).mems_allowed_seq) }, seq) != 0
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
    __slate_bits_0: __slate_bits::__SlateBits859U0,
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


fn css_get(css: *mut cgroup_subsys_state) {
if (unsafe { (*css).flags }) & 1 == 0 {
percpu_ref_get(unsafe { core::ptr::addr_of_mut!((*css).refcnt) });
}
}

fn css_put(css: *mut cgroup_subsys_state) {
if (unsafe { (*css).flags }) & 1 == 0 {
percpu_ref_put(unsafe { core::ptr::addr_of_mut!((*css).refcnt) });
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct page_counter {
    usage: __SlateRecord_types_h_197,
    failcnt: u64,
    _pad1_: cacheline_padding,
    emin: u64,
    min_usage: __SlateRecord_types_h_197,
    children_min_usage: __SlateRecord_types_h_197,
    elow: u64,
    low_usage: __SlateRecord_types_h_197,
    children_low_usage: __SlateRecord_types_h_197,
    watermark: u64,
    local_watermark: u64,
    _pad2_: cacheline_padding,
    protection_support: bool,
    track_failcnt: bool,
    min: u64,
    low: u64,
    high: u64,
    max: u64,
    parent: *mut page_counter,
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
    __slate_bits_0: __slate_bits::__SlateBits636U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


fn mem_cgroup_charge_hugetlb(folio: *mut folio, gfp: u32) -> i32 {
0
}

fn mem_cgroup_uncharge(folio: *mut folio) {
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


fn hugetlb_try_dup_anon_rmap(folio: *mut folio, vma: *mut vm_area_struct) -> i32 {
VM_WARN_ON_FOLIO!();
VM_WARN_ON_FOLIO!();
if PageAnonExclusive((unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }) as *const page) != 0 {
if unlikely!(folio_needs_cow_for_dma(vma, folio)) != 0 {
return 16i32.wrapping_neg();
}
ClearPageAnonExclusive(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) });
}
atomic_inc(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._entire_mapcount) });
atomic_inc(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._large_mapcount) });
0
}

fn hugetlb_add_file_rmap(folio: *mut folio) {
VM_WARN_ON_FOLIO!();
VM_WARN_ON_FOLIO!();
atomic_inc(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._entire_mapcount) });
atomic_inc(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._large_mapcount) });
}

fn hugetlb_remove_rmap(folio: *mut folio) {
VM_WARN_ON_FOLIO!();
atomic_dec(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._entire_mapcount) });
atomic_dec(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._large_mapcount) });
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


fn pte_swp_clear_flags(mut pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
if pte_swp_exclusive(pte) {
pte = pte_swp_clear_exclusive(pte);
}
if pte_swp_soft_dirty(pte) != 0 {
pte = pte_swp_clear_soft_dirty(pte);
}
if pte_swp_uffd(pte) != 0 {
pte = pte_swp_clear_uffd(pte);
}
return pte;
}

fn swp_entry(r#type: u64, offset: u64) -> __SlateRecord_mm_types_h_288 {
let mut ret: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ret) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
ret.val = r#type << 58i32 | offset & 288230376151711743u64;
return ret;
}

fn swp_type(entry: __SlateRecord_mm_types_h_288) -> u32 {
(entry.val >> 58i32) as u32
}

fn swp_offset(entry: __SlateRecord_mm_types_h_288) -> u64 {
entry.val & 288230376151711743u64
}

fn swp_entry_to_pte(entry: __SlateRecord_mm_types_h_288) -> __SlateRecord_pgtable_64_types_h_21 {
let mut __slate_compound_0: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
let mut arch_entry: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(arch_entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
arch_entry = unsafe { *{
__slate_compound_0 = __SlateRecord_mm_types_h_288 { val: !(swp_offset(entry) as u64) << 14i32 >> 5i32 | (swp_type(entry) as u64) << 59i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
return __swp_entry_to_pte!(arch_entry);
}

fn make_readable_migration_entry(offset: u64) -> __SlateRecord_mm_types_h_288 {
return swp_entry(28u64, offset);
}

fn make_readable_exclusive_migration_entry(offset: u64) -> __SlateRecord_mm_types_h_288 {
return swp_entry(29u64, offset);
}

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
fn make_pte_marker_entry(marker: u64) -> __SlateRecord_mm_types_h_288 {
return swp_entry(31u64, marker);
}

fn make_pte_marker(marker: u64) -> __SlateRecord_pgtable_64_types_h_21 {
return swp_entry_to_pte(make_pte_marker_entry(marker));
}

fn softleaf_mk_none() -> __SlateRecord_mm_types_h_288 {
let mut __slate_compound_0: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_mm_types_h_288 { val: 0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

fn softleaf_from_pte(mut pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_mm_types_h_288 {
let mut __slate_compound_1: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
let mut arch_entry: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(arch_entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let __v0: bool = pte_present(pte) != 0 || pte_none(pte) != 0;
if __v0 {
return softleaf_mk_none();
}
pte = pte_swp_clear_flags(pte);
arch_entry = unsafe { *{
__slate_compound_1 = __SlateRecord_mm_types_h_288 { val: pte_val!(pte) };
core::ptr::addr_of_mut!(__slate_compound_1)
} };
// Temporary until swp_entry_t eliminated.
return swp_entry(__swp_type!(arch_entry), __swp_offset!(arch_entry));
}

fn softleaf_is_none(entry: __SlateRecord_mm_types_h_288) -> bool {
entry.val == 0
}

fn softleaf_type(entry: __SlateRecord_mm_types_h_288) -> u32 {
let mut type_num: u32 = 0;
if softleaf_is_none(entry) {
return 0;
}
type_num = (entry.val >> 58i32) as u32;
if type_num < 28u32 {
return 1;
}
match type_num {
28 => {
return 2;
}
29 => {
return 3;
}
30 => {
return 4;
}
31 => {
return 9;
}
_ => {
}
}
// Unknown entry type.
VM_WARN_ON_ONCE!();
0
}

fn softleaf_is_migration_write(entry: __SlateRecord_mm_types_h_288) -> bool {
(softleaf_type(entry) as u32) == 4
}

fn softleaf_is_migration_read(entry: __SlateRecord_mm_types_h_288) -> bool {
(softleaf_type(entry) as u32) == 2
}

fn softleaf_is_migration(entry: __SlateRecord_mm_types_h_288) -> bool {
match softleaf_type(entry) as u32 {
2..=4 => {
return true;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

fn softleaf_is_device_private(entry: __SlateRecord_mm_types_h_288) -> bool {
match softleaf_type(entry) as u32 {
5 | 6 => {
return true;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

fn softleaf_is_device_exclusive(entry: __SlateRecord_mm_types_h_288) -> bool {
(softleaf_type(entry) as u32) == 7
}

fn softleaf_is_hwpoison(entry: __SlateRecord_mm_types_h_288) -> bool {
(softleaf_type(entry) as u32) == 8
}

fn softleaf_is_marker(entry: __SlateRecord_mm_types_h_288) -> bool {
(softleaf_type(entry) as u32) == 9
}

fn softleaf_to_marker(entry: __SlateRecord_mm_types_h_288) -> u64 {
VM_WARN_ON_ONCE!();
swp_offset(entry) & BIT!(3i32).wrapping_sub(1)
}

fn softleaf_to_pfn(entry: __SlateRecord_mm_types_h_288) -> u64 {
VM_WARN_ON_ONCE!();
// Temporary until swp_entry_t eliminated.
swp_offset(entry) & BIT!(if pgtable_l5_enabled__2!() != 0 { 52i32 } else { 46i32 }.wrapping_sub(12)).wrapping_sub(1)
}

fn softleaf_migration_sync(entry: __SlateRecord_mm_types_h_288, folio: *mut folio) {
// Ensure we do not race with split, which might alter tail pages into new
// folios and thus result in observing an unlocked folio.
// This matches the write barrier in __split_folio_to_order().
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
// Any use of migration entries may only occur while the
// corresponding page is locked
VM_WARN_ON_ONCE!();
}

fn softleaf_to_folio(entry: __SlateRecord_mm_types_h_288) -> *mut folio {
let folio: *mut folio = pfn_folio(softleaf_to_pfn(entry));
VM_WARN_ON_ONCE!();
if softleaf_is_migration(entry) {
softleaf_migration_sync(entry, folio);
}
folio
}

fn softleaf_is_uffd_wp_marker(entry: __SlateRecord_mm_types_h_288) -> bool {
if !softleaf_is_marker(entry) {
return false;
}
softleaf_to_marker(entry) & BIT!(0i32) != 0
}

fn pte_is_marker(pte: __SlateRecord_pgtable_64_types_h_21) -> bool {
softleaf_is_marker(softleaf_from_pte(pte))
}

fn pte_is_uffd_wp_marker(pte: __SlateRecord_pgtable_64_types_h_21) -> bool {
let entry: __SlateRecord_mm_types_h_288 = softleaf_from_pte(pte);
softleaf_is_uffd_wp_marker(entry)
}

// An arbitrary initial parameter
fn jhash2(mut k: *const u32, mut length: u32, initval: u32) -> u32 {
let mut a: u32 = 0;
let mut b: u32 = 0;
let mut c: u32 = 0;
// Set up the internal state
let __v0: u32 = 3735928559u32.wrapping_add(length << 2i32).wrapping_add(initval);
c = __v0;
let __v1: u32 = __v0;
b = __v1;
a = __v1;
// Handle most of the key
while length > 3 {
a = a.wrapping_add(unsafe { *k.wrapping_offset(0 as isize) });
b = b.wrapping_add(unsafe { *k.wrapping_offset(1 as isize) });
c = c.wrapping_add(unsafe { *k.wrapping_offset(2 as isize) });
__jhash_mix!(a, b, c);
length = length.wrapping_sub(3);
let __v52: *const u32 = k;
k = __v52.wrapping_offset(3 as isize);
}
// Handle the last 3 u32's
// Nothing left to add
'__slate_break_55: {
match length {
3 => {
c = c.wrapping_add(unsafe { *k.wrapping_offset(2 as isize) });
let _v62: u32 = b;
let _v63: u32 = _v62.wrapping_add(unsafe { *k.wrapping_offset(1 as isize) });
b = _v63;
let _v64: u32 = a;
let _v65: u32 = _v64.wrapping_add(unsafe { *k.wrapping_offset(0 as isize) });
a = _v65;
__jhash_final!(a, b, c);
}
2 => {
b = b.wrapping_add(unsafe { *k.wrapping_offset(1 as isize) });
let _v66: u32 = a;
let _v67: u32 = _v66.wrapping_add(unsafe { *k.wrapping_offset(0 as isize) });
a = _v67;
__jhash_final!(a, b, c);
}
1 => {
a = a.wrapping_add(unsafe { *k.wrapping_offset(0 as isize) });
__jhash_final!(a, b, c);
}
0 => {
break '__slate_break_55; // Nothing left to add
}
_ => {
}
}
}
c
}

fn handle_userfault(vmf: *mut vm_fault, reason: u64) -> u32 {
2
}

fn userfaultfd_missing(vma: *mut vm_area_struct) -> bool {
false
}

fn userfaultfd_wp(vma: *mut vm_area_struct) -> bool {
false
}

fn userfaultfd_minor(vma: *mut vm_area_struct) -> bool {
false
}

fn userfaultfd_rwp(vma: *mut vm_area_struct) -> bool {
false
}

fn userfaultfd_protected(vma: *mut vm_area_struct) -> bool {
false
}

fn userfaultfd_wp_async(vma: *mut vm_area_struct) -> bool {
false
}

fn userfaultfd_rwp_async(vma: *mut vm_area_struct) -> bool {
false
}

fn vma_has_uffd_without_event_remap(vma: *mut vm_area_struct) -> bool {
false
}

fn pte_swp_uffd_any(pte: __SlateRecord_pgtable_64_types_h_21) -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hugepage_subpool {
    lock: spinlock,
    count: i64,
    max_hpages: i64,
    used_hpages: i64,
    hstate: *mut hstate,
    min_hpages: i64,
    rsv_hpages: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct resv_map {
    refs: kref,
    lock: spinlock,
    regions: list_head,
    adds_in_progress: i64,
    region_cache: list_head,
    region_cache_count: i64,
    rw_sema: rw_semaphore,
    reservation_counter: *mut page_counter,
    pages_per_hpage: u64,
    css: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_region {
    link: list_head,
    from: i64,
    to: i64,
    reservation_counter: *mut page_counter,
    css: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hugetlb_vma_lock {
    refs: kref,
    rw_sema: rw_semaphore,
    vma: *mut vm_area_struct,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut hugetlb_max_hstate: i32 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..ro_after_init")]
static mut hugetlb_fault_mutex_table: *mut mutex = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.data")]
static mut huge_boot_pages: __SlateAlign16<[list_head; 64]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.data")]
static mut hugetlb_bootmem_nodes: __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct hugetlbfs_sb_info {
    max_inodes: i64,
    free_inodes: i64,
    stat_lock: spinlock,
    hstate: *mut hstate,
    spool: *mut hugepage_subpool,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    mode: u16,
}


fn HUGETLBFS_SB(sb: *mut super_block) -> *mut hugetlbfs_sb_info {
(unsafe { (*sb).s_fs_info }) as *mut hugetlbfs_sb_info
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hugetlbfs_inode_info {
    vfs_inode: inode,
    resv_map: *mut resv_map,
    seals: u32,
}


fn HUGETLBFS_I(inode: *mut inode) -> *mut hugetlbfs_inode_info {
container_of__6!(inode, hugetlbfs_inode_info)
}

fn hstate_inode(i: *mut inode) -> *mut hstate {
unsafe { (*HUGETLBFS_SB((*i).i_sb)).hstate }
}

#[inline(always)]
fn folio_test_hugetlb_restore_reserve(folio: *mut folio) -> bool {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
test_bit__4!(0i32, private) != 0
}

#[inline(always)]
fn folio_set_hugetlb_restore_reserve(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
set_bit(0i64, private as *mut u64);
}

#[inline(always)]
fn folio_clear_hugetlb_restore_reserve(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
clear_bit(0i64, private as *mut u64);
}

#[inline(always)]
fn folio_test_hugetlb_migratable(folio: *mut folio) -> bool {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
test_bit__4!(1i32, private) != 0
}

#[inline(always)]
fn folio_set_hugetlb_migratable(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
set_bit(1i64, private as *mut u64);
}

#[inline(always)]
fn folio_clear_hugetlb_migratable(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
clear_bit(1i64, private as *mut u64);
}

#[inline(always)]
fn folio_test_hugetlb_temporary(folio: *mut folio) -> bool {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
test_bit__4!(2i32, private) != 0
}

#[inline(always)]
fn folio_set_hugetlb_temporary(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
set_bit(2i64, private as *mut u64);
}

#[inline(always)]
fn folio_clear_hugetlb_temporary(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
clear_bit(2i64, private as *mut u64);
}

#[inline(always)]
fn folio_test_hugetlb_freed(folio: *mut folio) -> bool {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
test_bit__4!(3i32, private) != 0
}

#[inline(always)]
fn folio_set_hugetlb_freed(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
set_bit(3i64, private as *mut u64);
}

#[inline(always)]
fn folio_clear_hugetlb_freed(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
clear_bit(3i64, private as *mut u64);
}

#[inline(always)]
fn folio_test_hugetlb_vmemmap_optimized(folio: *mut folio) -> bool {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
test_bit__4!(4i32, private) != 0
}

#[inline(always)]
fn folio_set_hugetlb_vmemmap_optimized(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
set_bit(4i64, private as *mut u64);
}

#[inline(always)]
fn folio_test_hugetlb_raw_hwp_unreliable(folio: *mut folio) -> bool {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
test_bit__4!(5i32, private) != 0
}

#[inline(always)]
fn folio_test_hugetlb_cma(folio: *mut folio) -> bool {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
test_bit__4!(6i32, private) != 0
}

#[inline(always)]
fn folio_set_hugetlb_cma(folio: *mut folio) {
let private: *mut () = (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private) }) as *mut ();
set_bit(6i64, private as *mut u64);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hstate {
    resize_lock: mutex,
    resize_key: lock_class_key,
    next_nid_to_alloc: i32,
    next_nid_to_free: i32,
    order: u32,
    demote_order: u32,
    mask: u64,
    max_huge_pages: u64,
    nr_huge_pages: u64,
    free_huge_pages: u64,
    resv_huge_pages: u64,
    surplus_huge_pages: u64,
    nr_overcommit_huge_pages: u64,
    hugepage_activelist: list_head,
    hugepage_freelists: [list_head; 64],
    max_huge_pages_node: [u32; 64],
    nr_huge_pages_node: [u32; 64],
    free_huge_pages_node: [u32; 64],
    surplus_huge_pages_node: [u32; 64],
    name: [u8; 32],
}


#[unsafe(no_mangle)]
extern "C" fn replace_free_hugepage_folios(mut start_pfn: u64, end_pfn: u64) -> i32 {
let mut nr: u64 = 0;
let mut page: *mut page = unsafe { core::mem::zeroed() };
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut list: list_head = unsafe { core::mem::zeroed() };
list = list_head { next: core::ptr::addr_of_mut!(list), prev: core::ptr::addr_of_mut!(list) };
let mut ret: i32 = 0;
// Avoid pfn iterations if no free non-gigantic huge pages
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
if !hstate_is_gigantic(h) {
nr = nr.wrapping_add(unsafe { (*h).free_huge_pages });
if nr != 0 {
break;
}
}
let __v3: *mut hstate = h;
h = __v3.wrapping_offset(1 as isize);
}
if nr == 0 {
return 0;
}
while start_pfn < end_pfn {
page = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(start_pfn as isize);
nr = 1;
let __v5: bool = PageHuge(page as *const page) || PageCompound(page as *const page) != 0;
if __v5 {
let folio: *mut folio = page_folio!(page);
nr = folio_nr_pages(folio as *const folio).wrapping_sub(folio_page_idx(folio as *const folio, page as *const page));
// Don't disrupt normal path by vainly holding
// hugetlb_lock
let __v6: bool = folio_test_hugetlb(folio as *const folio) && folio_ref_count(folio as *const folio) == 0;
if __v6 {
if order_is_gigantic(folio_order(folio as *const folio)) {
ret = 12i32.wrapping_neg();
break;
}
ret = alloc_and_dissolve_hugetlb_folio(folio, core::ptr::addr_of_mut!(list));
if ret != 0 {
break;
}
unsafe { putback_movable_pages(core::ptr::addr_of_mut!(list)) };
}
} else {
if PageBuddy(page as *const page) != 0 {
// Buddy order check without zone lock is unsafe and
// the order is maybe invalid, but race should be
// small, and the worst thing is skipping free hugetlb.
let mut order: u32 = 0;
let mut __v8: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_711() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.private) as *const u64) };
order = __v8 as u32;
if order <= 10 {
nr = 1u64 << order;
}
}
}
start_pfn = start_pfn.wrapping_add(nr);
}
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mempolicy_interpreted {
    nid: i32,
    nodemask: *mut __SlateRecord_nodemask_types_h_17,
    mode: u32,
}


#[unsafe(no_mangle)]
static mut hstates: __SlateAlign16<[hstate; 2]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(no_mangle)]
static mut default_hstate_idx: u32 = unsafe { core::mem::zeroed() };

fn subpool_inode(inode: *mut inode) -> *mut hugepage_subpool {
unsafe { (*HUGETLBFS_SB((*inode).i_sb)).spool }
}

fn hugetlb_folio_subpool(folio: *mut folio) -> *mut hugepage_subpool {
(unsafe { (*folio).__slate_anon_3.__slate_anon_0._hugetlb_subpool }) as *mut hugepage_subpool
}

fn hugetlb_set_folio_subpool(folio: *mut folio, subpool: *mut hugepage_subpool) {
unsafe {
(*folio).__slate_anon_3.__slate_anon_0._hugetlb_subpool = subpool as *mut ();
}
}

fn hstate_file(f: *mut file) -> *mut hstate {
hstate_inode(file_inode(f as *const file))
}

fn hstate_vma(vma: *mut vm_area_struct) -> *mut hstate {
hstate_file(unsafe { (*vma).vm_file })
}

fn huge_page_size(h: *const hstate) -> u64 {
_AC!() << 12i32 << unsafe { (*h).order }
}

fn huge_page_mask(h: *mut hstate) -> u64 {
unsafe { (*h).mask }
}

fn huge_page_order(h: *mut hstate) -> u32 {
unsafe { (*h).order }
}

fn huge_page_shift(h: *mut hstate) -> u32 {
unsafe { (*h).order }.wrapping_add(12)
}

fn order_is_gigantic(order: u32) -> bool {
order > 10
}

fn hstate_is_gigantic(h: *mut hstate) -> bool {
order_is_gigantic(huge_page_order(h))
}

fn pages_per_huge_page(h: *const hstate) -> u32 {
(1i32 << unsafe { (*h).order }) as u32
}

fn blocks_per_huge_page(h: *mut hstate) -> u32 {
(huge_page_size(h as *const hstate) / 512) as u32
}

fn filemap_lock_hugetlb_folio(h: *mut hstate, mapping: *mut address_space, idx: u64) -> *mut folio {
filemap_lock_folio(mapping, idx << huge_page_order(h))
}

fn huge_pte_write(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
pte_write(pte) as u64
}

fn huge_pte_dirty(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
pte_dirty(pte) as u64
}

fn huge_pte_mkwrite(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_mkwrite_novma(pte);
}

fn huge_pte_wrprotect(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_wrprotect(pte);
}

fn huge_pte_mkdirty(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_mkdirty(pte);
}

fn huge_pte_modify(pte: __SlateRecord_pgtable_64_types_h_21, newprot: pgprot) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_modify(pte, newprot);
}

fn huge_pte_mkuffd(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return huge_pte_wrprotect(pte_mkuffd(pte));
}

fn huge_pte_clear_uffd(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_clear_uffd(pte);
}

fn huge_pte_uffd(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
pte_uffd(pte)
}

fn huge_pte_clear(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21, sz: u64) {
pte_clear!(mm, addr, ptep);
}

fn set_huge_pte_at(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21, pte: __SlateRecord_pgtable_64_types_h_21, sz: u64) {
set_pte_at!(mm, addr, ptep, pte);
}

fn huge_ptep_get_and_clear(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21, sz: u64) -> __SlateRecord_pgtable_64_types_h_21 {
return ptep_get_and_clear(mm, addr, ptep);
}

fn huge_ptep_clear_flush(vma: *mut vm_area_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return unsafe { ptep_clear_flush(vma, addr, ptep) };
}

fn huge_pte_none(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
pte_none(pte)
}

fn huge_ptep_set_wrprotect(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) {
ptep_set_wrprotect(mm, addr, ptep);
}

fn huge_ptep_set_access_flags(vma: *mut vm_area_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21, pte: __SlateRecord_pgtable_64_types_h_21, dirty: i32) -> i32 {
unsafe { ptep_set_access_flags(vma, addr, ptep, pte, dirty) }
}

fn huge_ptep_get(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return ptep_get(ptep);
}

fn gigantic_page_runtime_supported() -> bool {
IS_ENABLED__2!() != 0
}

fn arch_clear_hugetlb_flags(folio: *mut folio) {
}

fn arch_make_huge_pte(entry: __SlateRecord_pgtable_64_types_h_21, shift: u32, flags: u64) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_mkhuge(entry);
}

fn folio_hstate(folio: *mut folio) -> *mut hstate {
VM_BUG_ON_FOLIO!();
size_to_hstate(folio_size(folio as *const folio))
}

fn hstate_index(h: *mut hstate) -> i32 {
((unsafe { h.offset_from((core::ptr::addr_of_mut!(hstates.0) as *mut hstate) as *mut hstate) }) as i64) as i32
}

#[unsafe(no_mangle)]
extern "C" fn dissolve_free_hugetlb_folios(start_pfn: u64, end_pfn: u64) -> i32 {
let mut pfn: u64 = 0;
let mut rc: i32 = 0;
let mut order: u32 = 0;
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
if hugepages_supported!() == 0 {
return rc;
}
order = huge_page_order(unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { default_hstate_idx }) as isize));
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
let mut __v1: u32;
let __UNIQUE_ID_x__704: u32 = order;
let __UNIQUE_ID_y__705: u32 = huge_page_order(h);
loop {
if __types_ok__5!(__UNIQUE_ID_x__704, __UNIQUE_ID_y__705) == 0 {
unsafe { __compiletime_assert_706() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__704 < __UNIQUE_ID_y__705 { __UNIQUE_ID_x__704 } else { __UNIQUE_ID_y__705 };
order = __v1;
let __v4: *mut hstate = h;
h = __v4.wrapping_offset(1 as isize);
}
pfn = start_pfn;
while pfn < end_pfn {
let folio: *mut folio = pfn_folio(pfn);
rc = dissolve_free_hugetlb_folio(folio);
if rc != 0 {
break;
}
pfn = pfn.wrapping_add((1i32 << order) as u64);
}
rc
}

fn folio_clear_hugetlb_hwpoison(folio: *mut folio) {
}

fn arch_hugetlb_migration_supported(h: *mut hstate) -> bool {
let __v0: bool = huge_page_shift(h) == 21 || huge_page_shift(h) == 30;
let __v1: bool = __v0 || huge_page_shift(h) == unsafe { pgdir_shift };
if __v1 {
return true;
} else {
return false;
}
}

fn hugepage_migration_supported(h: *mut hstate) -> bool {
arch_hugetlb_migration_supported(h)
}

fn hugepage_movable_supported(h: *mut hstate) -> bool {
if !hugepage_migration_supported(h) {
return false;
}
if hstate_is_gigantic(h) && (unsafe { movable_gigantic_pages }) == 0 {
return false;
}
true
}

fn htlb_alloc_mask(h: *mut hstate) -> u32 {
let mut gfp: u32 = (BIT!(18i32) as u32) | (BIT!(13i32) as u32);
let __v0: u32 = if hugepage_movable_supported(h) { ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(20i32) as u32) | (BIT!(1i32) as u32) | (BIT!(3i32) as u32) | 0 } else { ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(20i32) as u32) | (BIT!(1i32) as u32) };
gfp |= __v0;
gfp
}

fn huge_pte_lockptr(h: *mut hstate, mm: *mut mm_struct, pte: *mut __SlateRecord_pgtable_64_types_h_21) -> *mut spinlock {
let size: u64 = huge_page_size(h as *const hstate);
VM_WARN_ON!();
// hugetlb must use the exact same PT locks as core-mm page table
// walkers would. When modifying a PTE table, hugetlb must take the
// PTE PT lock, when modifying a PMD table, hugetlb must take the PMD
// PT lock etc.
//
// The expectation is that any hugetlb folio smaller than a PMD is
// always mapped into a single PTE table and that any hugetlb folio
// smaller than a PUD (but at least as big as a PMD) is always mapped
// into a single PMD table.
//
// If that does not hold for an architecture, then that architecture
// must disable split PT locks such that all *_lockptr() functions
// will give us the same result: the per-MM PT lock.
//
// Note that with e.g., CONFIG_PGTABLE_LEVELS=2 where
// PGDIR_SIZE==P4D_SIZE==PUD_SIZE==PMD_SIZE, we'd use pud_lockptr()
// and core-mm would use pmd_lockptr(). However, in such configurations
// split PMD locks are disabled -- they don't make sense on a single
// PGDIR page table -- and the end result is the same.
if size >= _AC!() << 30i32 {
return pud_lockptr(mm, pte as *mut __SlateRecord_pgtable_types_h_368);
} else {
if size >= _AC!() << 21i32 || IS_ENABLED!() != 0 {
return pmd_lockptr(mm, pte as *mut __SlateRecord_pgtable_64_types_h_22);
}
}
// pte_alloc_huge() only applies with !CONFIG_HIGHPTE
ptep_lockptr(mm, pte)
}

fn hugetlb_count_add(l: i64, mm: *mut mm_struct) {
atomic_long_add(l, unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.hugetlb_usage) });
}

fn hugetlb_count_sub(l: i64, mm: *mut mm_struct) {
atomic_long_sub(l, unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.hugetlb_usage) });
}

fn huge_ptep_modify_prot_start(vma: *mut vm_area_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
let psize: u64 = huge_page_size(hstate_vma(vma) as *const hstate);
return huge_ptep_get_and_clear(unsafe { (*vma).vm_mm }, addr, ptep, psize);
}

fn huge_ptep_modify_prot_commit(vma: *mut vm_area_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21, old_pte: __SlateRecord_pgtable_64_types_h_21, pte: __SlateRecord_pgtable_64_types_h_21) {
let psize: u64 = huge_page_size(hstate_vma(vma) as *const hstate);
set_huge_pte_at(unsafe { (*vma).vm_mm }, addr, ptep, pte, psize);
}

fn huge_pte_lock(h: *mut hstate, mm: *mut mm_struct, pte: *mut __SlateRecord_pgtable_64_types_h_21) -> *mut spinlock {
let ptl: *mut spinlock = huge_pte_lockptr(h, mm, pte);
spin_lock(ptl);
ptl
}

fn __vma_shareable_lock(vma: *mut vm_area_struct) -> bool {
(unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) != 0 && !unsafe { (*vma).vm_private_data }.is_null()
}

fn hugetlb_walk(vma: *mut vm_area_struct, addr: u64, sz: u64) -> *mut __SlateRecord_pgtable_64_types_h_21 {
huge_pte_offset(unsafe { (*vma).vm_mm }, addr, sz)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct task_delay_info {
    lock: raw_spinlock,
    blkio_start: u64,
    blkio_delay_max: u64,
    blkio_delay_min: u64,
    blkio_delay: u64,
    swapin_start: u64,
    swapin_delay_max: u64,
    swapin_delay_min: u64,
    swapin_delay: u64,
    blkio_count: u32,
    swapin_count: u32,
    freepages_start: u64,
    freepages_delay_max: u64,
    freepages_delay_min: u64,
    freepages_delay: u64,
    thrashing_start: u64,
    thrashing_delay_max: u64,
    thrashing_delay_min: u64,
    thrashing_delay: u64,
    compact_start: u64,
    compact_delay_max: u64,
    compact_delay_min: u64,
    compact_delay: u64,
    wpcopy_start: u64,
    wpcopy_delay_max: u64,
    wpcopy_delay_min: u64,
    wpcopy_delay: u64,
    irq_delay_max: u64,
    irq_delay_min: u64,
    irq_delay: u64,
    freepages_count: u32,
    thrashing_count: u32,
    compact_count: u32,
    wpcopy_count: u32,
    irq_count: u32,
    blkio_delay_max_ts: timespec64,
    swapin_delay_max_ts: timespec64,
    freepages_delay_max_ts: timespec64,
    thrashing_delay_max_ts: timespec64,
    compact_delay_max_ts: timespec64,
    wpcopy_delay_max_ts: timespec64,
    irq_delay_max_ts: timespec64,
}


fn delayacct_wpcopy_start() {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(delayacct_key) }) == 0 {
return;
}
if !unsafe { (*get_current()).delays }.is_null() {
unsafe { __delayacct_wpcopy_start() };
}
}

fn delayacct_wpcopy_end() {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(delayacct_key) }) == 0 {
return;
}
if !unsafe { (*get_current()).delays }.is_null() {
unsafe { __delayacct_wpcopy_end() };
}
}

fn copy_pte_marker(entry: __SlateRecord_mm_types_h_288, dst_vma: *mut vm_area_struct) -> u64 {
let srcm: u64 = softleaf_to_marker(entry);
// Always copy error entries.
let mut dstm: u64 = srcm & (BIT!(1i32) | BIT!(2i32));
// Only copy PTE markers if UFFD register matches.
let __v0: bool = srcm & BIT!(0i32) != 0 && userfaultfd_wp(dst_vma);
if __v0 {
let __v1: u64 = dstm;
dstm = __v1 | BIT!(0i32);
}
dstm
}

#[repr(C)]
#[derive(Clone, Copy)]
struct padata_mt_job {
    thread_fn: Option<unsafe extern "C" fn(u64, u64, *mut ())>,
    fn_arg: *mut (),
    start: u64,
    size: u64,
    align: u64,
    min_chunk: u64,
    max_threads: i32,
    numa_aware: bool,
}


fn paravirt_alloc_pmd(mm: *mut mm_struct, pfn: u64) {
}

fn pud_populate(mm: *mut mm_struct, pud: *mut __SlateRecord_pgtable_types_h_368, pmd: *mut __SlateRecord_pgtable_64_types_h_22) {
paravirt_alloc_pmd(mm, __pa!(pmd) >> 12i32);
set_pud!(pud, __pud!(_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | _AT!(u64, 1i32) << 2i32 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64) | __pa!(pmd)));
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_table_batch {
    rcu: callback_head,
    nr: u32,
    tables: [*mut (); 0],
}


// If we can't allocate a page to make a big batch of page pointers
// to work on, then just handle a few from the on-stack structure.
#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_gather_batch {
    next: *mut mmu_gather_batch,
    nr: u32,
    max: u32,
    encoded_pages: [*mut encoded_page; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_gather {
    mm: *mut mm_struct,
    batch: *mut mmu_table_batch,
    start: u64,
    end: u64,
    __slate_bits_0: __slate_bits::__SlateBits477U0,
    batch_count: u32,
    active: *mut mmu_gather_batch,
    local: mmu_gather_batch,
    __pages: [*mut page; 8],
}


fn __tlb_adjust_range(tlb: *mut mmu_gather, address: u64, range_size: u32) {
let mut __v0: u64;
let __UNIQUE_ID_x__677: u64 = unsafe { (*tlb).start };
let __UNIQUE_ID_y__678: u64 = address;
loop {
if __types_ok__2!(__UNIQUE_ID_x__677, __UNIQUE_ID_y__678) == 0 {
unsafe { __compiletime_assert_679() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__677 < __UNIQUE_ID_y__678 { __UNIQUE_ID_x__677 } else { __UNIQUE_ID_y__678 };
unsafe {
(*tlb).start = __v0;
}
let mut __v2: u64;
let __UNIQUE_ID_x__680: u64 = unsafe { (*tlb).end };
let __UNIQUE_ID_y__681: u64 = address.wrapping_add(range_size as u64);
loop {
if __types_ok__2!(__UNIQUE_ID_x__680, __UNIQUE_ID_y__681) == 0 {
unsafe { __compiletime_assert_682() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__680 > __UNIQUE_ID_y__681 { __UNIQUE_ID_x__680 } else { __UNIQUE_ID_y__681 };
unsafe {
(*tlb).end = __v2;
}
}

fn __tlb_reset_range(tlb: *mut mmu_gather) {
if ((unsafe { (*tlb).__slate_bits_0.__get_fullmm() }) as i32) != 0 {
let __v0: u64 = 18446744073709551615u64;
unsafe {
(*tlb).end = __v0;
}
unsafe {
(*tlb).start = __v0;
}
} else {
let __v1: u64 = if test_thread_flag!(27i32) != 0 {
(if (unsafe { (*get_current()).personality }) & 134217728 != 0 { 3221225472u32 } else { 4294959104u32 }) as u64
} else {
task_size_max()
};
unsafe {
(*tlb).start = __v1;
}
unsafe {
(*tlb).end = 0;
}
}
unsafe {
(*tlb).__slate_bits_0.__set_freed_tables(0u32);
}
unsafe {
(*tlb).__slate_bits_0.__set_cleared_ptes(0u32);
}
unsafe {
(*tlb).__slate_bits_0.__set_cleared_pmds(0u32);
}
unsafe {
(*tlb).__slate_bits_0.__set_cleared_puds(0u32);
}
unsafe {
(*tlb).__slate_bits_0.__set_cleared_p4ds(0u32);
}
unsafe {
(*tlb).__slate_bits_0.__set_unshared_tables(0u32);
}
// Do not reset mmu_gather::vma_* fields here, we do not
// call into tlb_start_vma() again to set them if there is an
// intermediate flush.
}

fn tlb_update_vma_flags(tlb: *mut mmu_gather, vma: *mut vm_area_struct) {
// flush_tlb_range() implementations that look at VM_HUGETLB (tile,
// mips-4k) flush only large pages.
//
// flush_tlb_range() implementations that flush I-TLB also flush D-TLB
// (tile, xtensa, arm), so it's ok to just add VM_EXEC to an existing
// range.
//
// We rely on tlb_end_vma() to issue a flush, such that when we reset
// these values the batch is empty.
unsafe {
let __t0 = is_vm_hugetlb_page(vma as *const vm_area_struct) as u32;
(*tlb).__slate_bits_0.__set_vma_huge(__t0);
}
unsafe {
let __t1 = ((*vma).__slate_anon_3.vm_flags & BIT!(2i32) != 0) as u32;
(*tlb).__slate_bits_0.__set_vma_exec(__t1);
}
// Track if there's at least one VM_PFNMAP/VM_MIXEDMAP vma
// in the tracked range, see tlb_free_vmas().
let __v1: u32 = unsafe { (*tlb).__slate_bits_0.__get_vma_pfn() };
let __v2: u32 = ((__v1 as i32) | (((unsafe { (*vma).__slate_anon_3.vm_flags }) & (BIT!(10i32) | BIT!(28i32)) != 0) as i32)) as u32;
unsafe {
(*tlb).__slate_bits_0.__set_vma_pfn(__v2);
}
}

fn tlb_flush_mmu_tlbonly(tlb: *mut mmu_gather) {
// Anything calling __tlb_adjust_range() also sets at least one of
// these bits.
if !(((unsafe { (*tlb).__slate_bits_0.__get_freed_tables() }) as i32) != 0 || ((unsafe { (*tlb).__slate_bits_0.__get_cleared_ptes() }) as i32) != 0 || ((unsafe { (*tlb).__slate_bits_0.__get_cleared_pmds() }) as i32) != 0 || ((unsafe { (*tlb).__slate_bits_0.__get_cleared_puds() }) as i32) != 0 || ((unsafe { (*tlb).__slate_bits_0.__get_cleared_p4ds() }) as i32) != 0 || ((unsafe { (*tlb).__slate_bits_0.__get_unshared_tables() }) as i32) != 0) {
return;
}
tlb_flush(tlb);
__tlb_reset_range(tlb);
}

fn tlb_remove_page_size(tlb: *mut mmu_gather, page: *mut page, page_size: i32) {
if unsafe { __tlb_remove_page_size(tlb, page, page_size) } {
unsafe { tlb_flush_mmu(tlb) };
}
}

fn tlb_change_page_size(tlb: *mut mmu_gather, page_size: u32) {
}

fn tlb_get_unmap_shift(tlb: *mut mmu_gather) -> u64 {
if ((unsafe { (*tlb).__slate_bits_0.__get_cleared_ptes() }) as i32) != 0 {
return 12;
}
if ((unsafe { (*tlb).__slate_bits_0.__get_cleared_pmds() }) as i32) != 0 {
return 21;
}
if ((unsafe { (*tlb).__slate_bits_0.__get_cleared_puds() }) as i32) != 0 {
return 30;
}
if ((unsafe { (*tlb).__slate_bits_0.__get_cleared_p4ds() }) as i32) != 0 {
return 39;
}
12
}

fn tlb_start_vma(tlb: *mut mmu_gather, vma: *mut vm_area_struct) {
if ((unsafe { (*tlb).__slate_bits_0.__get_fullmm() }) as i32) != 0 {
return;
}
tlb_update_vma_flags(tlb, vma);
flush_cache_range(vma, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end });
}

fn tlb_end_vma(tlb: *mut mmu_gather, vma: *mut vm_area_struct) {
if ((unsafe { (*tlb).__slate_bits_0.__get_fullmm() }) as i32) != 0 || IS_ENABLED__2!() != 0 {
return;
}
// Do a TLB flush and reset the range at VMA boundaries; this avoids
// the ranges growing with the unused space between consecutive VMAs,
// but also the mmu_gather::vma_* flags from tlb_start_vma() rely on
// this.
tlb_flush_mmu_tlbonly(tlb);
}

fn tlb_flush_pte_range(tlb: *mut mmu_gather, address: u64, size: u64) {
__tlb_adjust_range(tlb, address, size as u32);
unsafe {
(*tlb).__slate_bits_0.__set_cleared_ptes(1u32);
}
}

fn tlb_flush_pmd_range(tlb: *mut mmu_gather, address: u64, size: u64) {
__tlb_adjust_range(tlb, address, size as u32);
unsafe {
(*tlb).__slate_bits_0.__set_cleared_pmds(1u32);
}
}

fn tlb_flush_pud_range(tlb: *mut mmu_gather, address: u64, size: u64) {
__tlb_adjust_range(tlb, address, size as u32);
unsafe {
(*tlb).__slate_bits_0.__set_cleared_puds(1u32);
}
}

fn tlb_flush_p4d_range(tlb: *mut mmu_gather, address: u64, size: u64) {
__tlb_adjust_range(tlb, address, size as u32);
unsafe {
(*tlb).__slate_bits_0.__set_cleared_p4ds(1u32);
}
}

fn __tlb_remove_tlb_entry(tlb: *mut mmu_gather, ptep: *mut __SlateRecord_pgtable_64_types_h_21, address: u64) {
}

fn tlb_unshare_pmd_ptdesc(tlb: *mut mmu_gather, pt: *mut ptdesc, addr: u64) {
// The caller must make sure that concurrent unsharing + exclusive
// reuse is impossible until tlb_flush_unshared_tables() was called.
VM_WARN_ON_ONCE!();
ptdesc_pmd_pts_dec(pt);
// Clearing a PUD pointing at a PMD table with PMD leaves.
tlb_flush_pmd_range(tlb, addr & !(_AC!() << 30i32).wrapping_sub(1), _AC!() << 30i32);
// If the page table is now exclusively owned, we fully unshared
// a page table.
if !ptdesc_pmd_is_shared(pt) {
unsafe {
(*tlb).__slate_bits_0.__set_fully_unshared_tables(1u32);
}
}
unsafe {
(*tlb).__slate_bits_0.__set_unshared_tables(1u32);
}
}

fn tlb_flush_unshared_tables(tlb: *mut mmu_gather) {
// As soon as the caller drops locks to allow for reuse of
// previously-shared tables, these tables could get modified and
// even reused outside of hugetlb context, so we have to make sure that
// any page table walkers (incl. TLB, GUP-fast) are aware of that
// change.
//
// Even if we are not fully unsharing a PMD table, we must
// flush the TLB for the unsharer now.
if ((unsafe { (*tlb).__slate_bits_0.__get_unshared_tables() }) as i32) != 0 {
tlb_flush_mmu_tlbonly(tlb);
}
// Similarly, we must make sure that concurrent GUP-fast will not
// walk previously-shared page tables that are getting modified+reused
// elsewhere. So broadcast an IPI to wait for any concurrent GUP-fast.
//
// We only perform this when we are the last sharer of a page table,
// as the IPI will reach all CPUs: any GUP-fast.
//
// Note that on configs where tlb_remove_table_sync_one() is a NOP,
// the expectation is that the tlb_flush_mmu_tlbonly() would have issued
// required IPIs already for us.
if ((unsafe { (*tlb).__slate_bits_0.__get_fully_unshared_tables() }) as i32) != 0 {
unsafe { tlb_remove_table_sync_one() };
unsafe {
(*tlb).__slate_bits_0.__set_fully_unshared_tables(0u32);
}
}
}

fn tlb_flush(tlb: *mut mmu_gather) {
let mut start: u64 = 0;
let mut end: u64 = 1u64.wrapping_neg();
let stride_shift: u32 = tlb_get_unmap_shift(tlb) as u32;
if ((unsafe { (*tlb).__slate_bits_0.__get_fullmm() }) as i32) == 0 && ((unsafe { (*tlb).__slate_bits_0.__get_need_flush_all() }) as i32) == 0 {
start = unsafe { (*tlb).start };
end = unsafe { (*tlb).end };
}
unsafe { flush_tlb_mm_range((*tlb).mm, start, end, stride_shift, ((*tlb).__slate_bits_0.__get_freed_tables() as i32) != 0) };
}

// INVLPGB can be targeted by virtual address, PCID, ASID, or any combination
// of the three. For example:
// - FLAG_VA | FLAG_INCLUDE_GLOBAL: invalidate all TLB entries at the address
// - FLAG_PCID:			    invalidate all TLB entries matching the PCID
//
// The first is used to invalidate (kernel) mappings at a particular
// address across all processes.
//
// The latter invalidates all TLB entries matching a PCID.
// The implied mode when all bits are clear:
fn split_page_owner(page: *mut page, old_order: i32, new_order: i32) {
}

fn folio_set_owner_migrate_reason(folio: *mut folio, reason: u32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct huge_bootmem_page {
    list: list_head,
    hstate: *mut hstate,
    flags: u64,
}


// Compare PTEs respecting the dirty bit.
// Compare PTEs respecting the soft-dirty bit.
// Compare PTEs respecting the writable bit.
// Merge PTE write bits: if any PTE in the batch is writable, modify the
// PTE at @ptentp to be writable.
// Merge PTE young and dirty bits: if any PTE in the batch is young or dirty,
// modify the PTE at @ptentp to be young or dirty, respectively.
fn folio_set_order(folio: *mut folio, order: u32) {
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = order == 0 || !folio_test_large(folio as *const folio);
__ret_warn_on = __v1 as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("690: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 690b - ., 3; .popsection\n# {0}", const 690 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_folio_set_order_2, sym __slate_asm_string_folio_set_order_3, const 737 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
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
if __v0 != 0 {
return;
}
VM_WARN_ON_ONCE!();
unsafe {
(*folio).__slate_anon_1.__slate_anon_0._flags_1 = (*folio).__slate_anon_1.__slate_anon_0._flags_1 & !255 | (order as u64);
}
}

fn prep_compound_head(page: *mut page, order: u32) {
let folio: *mut folio = page as *mut folio;
folio_set_order(folio, order);
atomic_set(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._large_mapcount) }, 1i32.wrapping_neg());
if IS_ENABLED__2!() != 0 {
atomic_set(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._nr_pages_mapped) }, 0i32);
}
if IS_ENABLED!() != 0 {
unsafe {
(*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0.__slate_anon_5._mm_ids = 0;
}
unsafe {
*((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._mm_id_mapcount.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize) = 1i32.wrapping_neg();
}
unsafe {
*((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._mm_id_mapcount.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) = 1i32.wrapping_neg();
}
}
if IS_ENABLED__2!() != 0 || order > 1 {
atomic_set(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._pincount) }, 0i32);
atomic_set(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._entire_mapcount) }, 1i32.wrapping_neg());
}
if order > 1 {
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_2.__slate_anon_0._deferred_list) });
}
}

fn prep_compound_tail(tail: *mut page, head: *const page, order: u32) {
unsafe {
(*tail).__slate_anon_1.__slate_anon_0.mapping = ((1024i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut address_space;
}
set_compound_head(tail, head, order);
VM_WARN_ON_ONCE!();
}

fn init_compound_tail(tail: *mut page, head: *const page, order: u32, zone: *mut zone) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*tail).__slate_anon_2._mapcount) }, 1i32.wrapping_neg());
set_page_node(tail, zone_to_nid(zone as *const zone) as u64);
set_page_zone(tail, zone_idx!(zone) as u32);
prep_compound_tail(tail, head, order);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
    zone: *mut zone,
    migratetype: i32,
    order: i32,
    page: *mut page,
}


fn hugetlb_vmemmap_size(h: *const hstate) -> u32 {
(pages_per_huge_page(h) as u64).wrapping_mul(core::mem::size_of::<page>() as u64) as u32
}

fn hugetlb_vmemmap_optimizable_size(h: *const hstate) -> u32 {
let size: i32 = (hugetlb_vmemmap_size(h) as u64).wrapping_sub(_AC!() << 12i32) as i32;
if !is_power_of_2(core::mem::size_of::<page>() as u64) {
return 0;
}
(if size > 0 { size } else { 0i32 }) as u32
}

fn hugetlb_vmemmap_optimizable(h: *const hstate) -> bool {
hugetlb_vmemmap_optimizable_size(h) != 0
}

fn hugetlb_cma_free_frozen_folio(folio: *mut folio) {
}

fn hugetlb_cma_alloc_bootmem(h: *mut hstate, nid: i32, node_exact: bool) -> *mut () {
core::ptr::null_mut::<()>()
}

fn hugetlb_cma_total_size() -> u64 {
0
}

fn hugetlb_cma_validate_params() {
}

fn hugetlb_early_cma(h: *mut hstate) -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hugetlb_cgroup_per_node {
    usage: [u64; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hugetlb_cgroup {
    css: cgroup_subsys_state,
    hugepage: [page_counter; 2],
    rsvd_hugepage: [page_counter; 2],
    events: [[__SlateRecord_types_h_197; 1]; 2],
    events_local: [[__SlateRecord_types_h_197; 1]; 2],
    events_file: [cgroup_file; 2],
    events_local_file: [cgroup_file; 2],
    nodeinfo: [*mut hugetlb_cgroup_per_node; 0],
}


fn __hugetlb_cgroup_from_folio(folio: &folio, rsvd: bool) -> *mut hugetlb_cgroup {
VM_BUG_ON_FOLIO!();
if rsvd {
return (unsafe { (*(folio as *const folio)).__slate_anon_3.__slate_anon_0._hugetlb_cgroup_rsvd }) as *mut hugetlb_cgroup;
} else {
return (unsafe { (*(folio as *const folio)).__slate_anon_3.__slate_anon_0._hugetlb_cgroup }) as *mut hugetlb_cgroup;
}
}

fn hugetlb_cgroup_from_folio(folio: &folio) -> *mut hugetlb_cgroup {
__hugetlb_cgroup_from_folio(folio, false)
}

fn hugetlb_cgroup_from_folio_rsvd(folio: &folio) -> *mut hugetlb_cgroup {
__hugetlb_cgroup_from_folio(folio, true)
}

fn __set_hugetlb_cgroup(folio: *mut folio, h_cg: *mut hugetlb_cgroup, rsvd: bool) {
VM_BUG_ON_FOLIO!();
if rsvd {
unsafe {
(*folio).__slate_anon_3.__slate_anon_0._hugetlb_cgroup_rsvd = h_cg as *mut ();
}
} else {
unsafe {
(*folio).__slate_anon_3.__slate_anon_0._hugetlb_cgroup = h_cg as *mut ();
}
}
}

fn set_hugetlb_cgroup(folio: *mut folio, h_cg: *mut hugetlb_cgroup) {
__set_hugetlb_cgroup(folio, h_cg, false);
}

fn set_hugetlb_cgroup_rsvd(folio: *mut folio, h_cg: *mut hugetlb_cgroup) {
__set_hugetlb_cgroup(folio, h_cg, true);
}

fn hugetlb_cgroup_put_rsvd_cgroup(h_cg: *mut hugetlb_cgroup) {
css_put(unsafe { core::ptr::addr_of_mut!((*h_cg).css) });
}

fn resv_map_dup_hugetlb_cgroup_uncharge_info(resv_map: *mut resv_map) {
if !unsafe { (*resv_map).css }.is_null() {
css_get(unsafe { (*resv_map).css });
}
}

fn resv_map_put_hugetlb_cgroup_uncharge_info(resv_map: *mut resv_map) {
if !unsafe { (*resv_map).css }.is_null() {
css_put(unsafe { (*resv_map).css });
}
}

fn hstate_is_gigantic_no_runtime(h: *mut hstate) -> bool {
hstate_is_gigantic(h) && !gigantic_page_runtime_supported()
}

fn next_node_allowed(mut nid: i32, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17) -> i32 {
nid = next_node_in__3!(nid, *nodes_allowed) as i32;
VM_BUG_ON!();
nid
}

fn get_valid_node_allowed(mut nid: i32, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17) -> i32 {
if node_isset__2!(nid, *nodes_allowed) == 0 {
nid = next_node_allowed(nid, nodes_allowed);
}
nid
}

fn hstate_next_node_to_alloc(next_node: *mut i32, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17) -> i32 {
let mut nid: i32 = 0;
VM_BUG_ON!();
nid = get_valid_node_allowed(unsafe { *next_node }, nodes_allowed);
unsafe {
*next_node = next_node_allowed(nid, nodes_allowed);
}
nid
}

fn hstate_next_node_to_free(h: *mut hstate, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17) -> i32 {
let mut nid: i32 = 0;
VM_BUG_ON!();
nid = get_valid_node_allowed(unsafe { (*h).next_nid_to_free }, nodes_allowed);
unsafe {
(*h).next_nid_to_free = next_node_allowed(nid, nodes_allowed);
}
nid
}

fn init_cma_pageblock(page: *mut page) {
}

fn is_migrate_isolate_page(page: *mut page) -> bool {
false
}

// Due to ordering constraints across the init code for various
// architectures, hugetlb hstate cmdline parameters can't simply
// be early_param. early_param might call the setup function
// before valid hugetlb page sizes are determined, leading to
// incorrect rejection of valid hugepagesz= options.
//
// So, record the parameters early and consume them whenever the
// init code is ready for them, by calling hugetlb_parse_params().
// one (hugepagesz=,hugepages=) pair per hstate, one default_hugepagesz
#[repr(C)]
#[derive(Clone, Copy)]
struct hugetlb_cmdline {
    val: *mut u8,
    setup: Option<unsafe extern "C" fn(*mut u8) -> i32>,
}


/// for command line parsing
#[unsafe(link_section = ".init.data")]
static mut parsed_hstate: *mut hstate = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut default_hstate_max_huge_pages: u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut parsed_valid_hugepagesz: bool = 1i32 != 0i32;

#[unsafe(link_section = ".init.data")]
static mut parsed_default_hugepagesz: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut default_hugepages_in_node: __SlateAlign16<[u32; 64]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(link_section = ".init.data")]
static mut hugepage_allocation_threads: u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut hstate_cmdline_buf: __SlateAlign16<[u8; 2048]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(link_section = ".init.data")]
static mut hstate_cmdline_index: i32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut hugetlb_params: __SlateAlign16<[hugetlb_cmdline; 5]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(link_section = ".init.data")]
static mut hugetlb_param_index: i32 = unsafe { core::mem::zeroed() };

/// Protects updates to hugepage_freelists, hugepage_activelist, nr_huge_pages,
/// free_huge_pages, and surplus_huge_pages.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..cacheline_aligned")]
static mut hugetlb_lock: __SlateAlign64<spinlock> = __SlateAlign64(spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} });

/// Serializes faults on the same logical page.  This is used to
/// prevent spurious OOMs when the hugepage pool is fully utilized.
#[unsafe(link_section = ".data..ro_after_init")]
static mut num_fault_mutexes: i32 = unsafe { core::mem::zeroed() };

fn subpool_is_free(spool: *mut hugepage_subpool) -> bool {
if (unsafe { (*spool).count }) != 0 {
return false;
}
if (unsafe { (*spool).max_hpages }) != (1i32.wrapping_neg() as i64) {
return (unsafe { (*spool).used_hpages }) == 0;
}
if (unsafe { (*spool).min_hpages }) != (1i32.wrapping_neg() as i64) {
return (unsafe { (*spool).rsv_hpages }) == unsafe { (*spool).min_hpages };
}
true
}

fn unlock_or_release_subpool(spool: *mut hugepage_subpool, irq_flags: u64) {
let free_subpool: bool = subpool_is_free(spool);
// If no pages are used, and no other handles to the subpool
// remain, give up any reservations based on minimum size and
// free the subpool
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*spool).lock) }, irq_flags);
if free_subpool {
if (unsafe { (*spool).min_hpages }) != (1i32.wrapping_neg() as i64) {
hugetlb_acct_memory(unsafe { (*spool).hstate }, unsafe { (*spool).min_hpages }.wrapping_neg());
}
unsafe { kfree(spool as *const ()) };
}
}

#[unsafe(no_mangle)]
extern "C" fn hugepage_new_subpool(h: *mut hstate, max_hpages: i64, min_hpages: i64) -> *mut hugepage_subpool {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: spinlock = unsafe { core::mem::zeroed() };
let mut spool: *mut hugepage_subpool = unsafe { core::mem::zeroed() };
let mut __v0: *mut hugepage_subpool;
let __obj_size: u64 = size_mul(core::mem::size_of::<hugepage_subpool>() as u64, 1u64);
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
__v0 = __v1 as *mut hugepage_subpool;
spool = __v0;
if spool.is_null() {
return core::ptr::null_mut::<hugepage_subpool>();
}
'__slate_break_7: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*spool).lock) });
unsafe {
*core::ptr::addr_of_mut!((*spool).lock) = *{
__slate_compound_6 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_6)
};
}
if true {
break;
}
}
unsafe {
(*spool).count = 1;
}
unsafe {
(*spool).max_hpages = max_hpages;
}
unsafe {
(*spool).hstate = h;
}
unsafe {
(*spool).min_hpages = min_hpages;
}
let __v8: bool = min_hpages != (1i32.wrapping_neg() as i64) && hugetlb_acct_memory(h, min_hpages) != 0;
if __v8 {
unsafe { kfree(spool as *const ()) };
return core::ptr::null_mut::<hugepage_subpool>();
}
unsafe {
(*spool).rsv_hpages = min_hpages;
}
spool
}

#[unsafe(no_mangle)]
extern "C" fn hugepage_put_subpool(spool: *mut hugepage_subpool) {
let mut flags: u64 = 0;
if spool.is_null() {
return;
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*spool).lock) }, flags);
loop {
if unlikely!((unsafe { (*spool).count }) == 0) != 0 {
loop {
unsafe {
core::arch::asm!("697: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 697b - ., 3; .popsection\n# {0}", const 697 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_hugepage_put_subpool_5, sym __slate_asm_string_hugepage_put_subpool_6, const 191 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
let __v11: i64 = unsafe { (*spool).count };
unsafe {
(*spool).count = __v11.wrapping_sub(1);
}
unlock_or_release_subpool(spool, flags);
}

/// Subpool accounting for allocating and reserving pages.
/// Return -ENOMEM if there are not enough resources to satisfy the
/// request.  Otherwise, return the number of pages by which the
/// global pools must be adjusted (upward).  The returned value may
/// only be different than the passed value (delta) in the case where
/// a subpool minimum size must be maintained.
fn hugepage_subpool_get_pages(spool: *mut hugepage_subpool, delta: i64) -> i64 {
let mut ret: i64 = unsafe { core::mem::zeroed() };
unsafe {
ret = delta;
if spool.is_null() {
return ret;
} else {
'__join_0: {
spin_lock_irq(core::ptr::addr_of_mut!((*spool).lock));
if (*spool).max_hpages != (1i32.wrapping_neg() as i64) {
// maximum size accounting
if unsafe { (*spool).used_hpages }.wrapping_add(delta) <= (*spool).max_hpages {
let __v4: i64 = (*spool).used_hpages;
(*spool).used_hpages = __v4.wrapping_add(delta);
} else {
ret = 12i32.wrapping_neg() as i64;
break '__join_0;
}
}
// minimum size accounting
if (*spool).min_hpages != (1i32.wrapping_neg() as i64) && (*spool).rsv_hpages != 0 {
if delta > (*spool).rsv_hpages {
// Asking for more reserves than those already taken on
// behalf of subpool.  Return difference.
ret = delta.wrapping_sub((*spool).rsv_hpages);
(*spool).rsv_hpages = 0;
} else {
ret = 0; // reserves already accounted for
let __v1: i64 = (*spool).rsv_hpages;
(*spool).rsv_hpages = __v1.wrapping_sub(delta);
}
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*spool).lock));
return ret;
}
}
}

/// Subpool accounting for freeing and unreserving pages.
/// Return the number of global page reservations that must be dropped.
/// The return value may only be different than the passed value (delta)
/// in the case where a subpool minimum size must be maintained.
fn hugepage_subpool_put_pages(spool: *mut hugepage_subpool, delta: i64) -> i64 {
let mut ret: i64 = delta;
let mut flags: u64 = 0;
if spool.is_null() {
return delta;
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*spool).lock) }, flags);
if (unsafe { (*spool).max_hpages }) != (1i32.wrapping_neg() as i64) {
let __v6: i64 = unsafe { (*spool).used_hpages };
unsafe {
(*spool).used_hpages = __v6.wrapping_sub(delta);
}
}
// maximum size accounting
// minimum size accounting
if (unsafe { (*spool).min_hpages }) != (1i32.wrapping_neg() as i64) && (unsafe { (*spool).used_hpages }) < unsafe { (*spool).min_hpages } {
if unsafe { (*spool).rsv_hpages }.wrapping_add(delta) <= unsafe { (*spool).min_hpages } {
ret = 0;
} else {
ret = unsafe { (*spool).rsv_hpages }.wrapping_add(delta).wrapping_sub(unsafe { (*spool).min_hpages });
}
let __v9: i64 = unsafe { (*spool).rsv_hpages };
unsafe {
(*spool).rsv_hpages = __v9.wrapping_add(delta);
}
if (unsafe { (*spool).rsv_hpages }) > unsafe { (*spool).min_hpages } {
unsafe {
(*spool).rsv_hpages = (*spool).min_hpages;
}
}
}
// If hugetlbfs_put_super couldn't free spool due to an outstanding
// quota reference, free it now.
unlock_or_release_subpool(spool, flags);
ret
}

fn subpool_vma(vma: *mut vm_area_struct) -> *mut hugepage_subpool {
subpool_inode(file_inode((unsafe { (*vma).vm_file }) as *const file))
}

/// hugetlb vma_lock helper routines
#[unsafe(no_mangle)]
extern "C" fn hugetlb_vma_lock_read(vma: *mut vm_area_struct) {
if __vma_shareable_lock(vma) {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
unsafe { down_read(core::ptr::addr_of_mut!((*vma_lock).rw_sema)) };
} else {
if __vma_private_lock(vma) {
let resv_map: *mut resv_map = vma_resv_map(vma);
unsafe { down_read(core::ptr::addr_of_mut!((*resv_map).rw_sema)) };
}
}
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_vma_unlock_read(vma: *mut vm_area_struct) {
if __vma_shareable_lock(vma) {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
unsafe { up_read(core::ptr::addr_of_mut!((*vma_lock).rw_sema)) };
} else {
if __vma_private_lock(vma) {
let resv_map: *mut resv_map = vma_resv_map(vma);
unsafe { up_read(core::ptr::addr_of_mut!((*resv_map).rw_sema)) };
}
}
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_vma_lock_write(vma: *mut vm_area_struct) {
if __vma_shareable_lock(vma) {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
unsafe { down_write(core::ptr::addr_of_mut!((*vma_lock).rw_sema)) };
} else {
if __vma_private_lock(vma) {
let resv_map: *mut resv_map = vma_resv_map(vma);
unsafe { down_write(core::ptr::addr_of_mut!((*resv_map).rw_sema)) };
}
}
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_vma_unlock_write(vma: *mut vm_area_struct) {
if __vma_shareable_lock(vma) {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
unsafe { up_write(core::ptr::addr_of_mut!((*vma_lock).rw_sema)) };
} else {
if __vma_private_lock(vma) {
let resv_map: *mut resv_map = vma_resv_map(vma);
unsafe { up_write(core::ptr::addr_of_mut!((*resv_map).rw_sema)) };
}
}
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_vma_trylock_write(vma: *mut vm_area_struct) -> i32 {
if __vma_shareable_lock(vma) {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
return unsafe { down_write_trylock(core::ptr::addr_of_mut!((*vma_lock).rw_sema)) };
} else {
if __vma_private_lock(vma) {
let resv_map: *mut resv_map = vma_resv_map(vma);
return unsafe { down_write_trylock(core::ptr::addr_of_mut!((*resv_map).rw_sema)) };
}
}
1
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_vma_assert_locked(vma: *mut vm_area_struct) {
if __vma_shareable_lock(vma) {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*vma_lock).rw_sema) });
} else {
if __vma_private_lock(vma) {
let resv_map: *mut resv_map = vma_resv_map(vma);
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*resv_map).rw_sema) });
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hugetlb.hugetlb_vma_lock_release")]
extern "C" fn hugetlb_vma_lock_release(kref: *mut kref) {
let vma_lock: *mut hugetlb_vma_lock = container_of__16!(kref, hugetlb_vma_lock);
unsafe { kfree(vma_lock as *const ()) };
}

fn __hugetlb_vma_unlock_write_put(vma_lock: *mut hugetlb_vma_lock) {
let vma: *mut vm_area_struct = unsafe { (*vma_lock).vma };
// vma_lock structure may or not be released as a result of put,
// it certainly will no longer be attached to vma so clear pointer.
// Semaphore synchronizes access to vma_lock->vma field.
unsafe {
(*vma_lock).vma = core::ptr::null_mut::<vm_area_struct>();
}
unsafe {
(*vma).vm_private_data = core::ptr::null_mut::<()>();
}
unsafe { up_write(core::ptr::addr_of_mut!((*vma_lock).rw_sema)) };
kref_put(unsafe { core::ptr::addr_of_mut!((*vma_lock).refs) }, Some(hugetlb_vma_lock_release));
}

fn __hugetlb_vma_unlock_write_free(vma: *mut vm_area_struct) {
if __vma_shareable_lock(vma) {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
__hugetlb_vma_unlock_write_put(vma_lock);
} else {
if __vma_private_lock(vma) {
let resv_map: *mut resv_map = vma_resv_map(vma);
// no free for anon vmas, but still need to unlock
unsafe { up_write(core::ptr::addr_of_mut!((*resv_map).rw_sema)) };
}
}
}

fn hugetlb_vma_lock_free(vma: *mut vm_area_struct) {
// Only present in sharable vmas.
let __v0: bool = vma.is_null() || !__vma_shareable_lock(vma);
if __v0 {
return;
}
if !unsafe { (*vma).vm_private_data }.is_null() {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
unsafe { down_write(core::ptr::addr_of_mut!((*vma_lock).rw_sema)) };
__hugetlb_vma_unlock_write_put(vma_lock);
}
}

fn hugetlb_vma_lock_alloc(vma: *mut vm_area_struct) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut vma_lock: *mut hugetlb_vma_lock = unsafe { core::mem::zeroed() };
// Only establish in (flags) sharable vmas
if vma.is_null() || (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) == 0 {
return;
}
// Should never get here with non-NULL vm_private_data
if !unsafe { (*vma).vm_private_data }.is_null() {
return;
}
let mut __v0: *mut hugetlb_vma_lock;
let __obj_size: u64 = size_mul(core::mem::size_of::<hugetlb_vma_lock>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut hugetlb_vma_lock;
vma_lock = __v0;
if vma_lock.is_null() {
// If we can not allocate structure, then vma can not
// participate in pmd sharing.  This is only a possible
// performance enhancement and memory saving issue.
// However, the lock is also used to synchronize page
// faults with truncation.  If the lock is not present,
// unlikely races could leave pages in a file past i_size
// until the file is removed.  Warn in the unlikely case of
// allocation failure.
let mut __v6: i64;
let __ret_do_once: bool = true;
let mut __v7: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely!(__ret_cond) != 0 && unlikely!(!unsafe { __already_done__450 }) != 0 {
unsafe {
__already_done__450 = true;
}
__ret_once = true;
}
__v7 = __ret_once as i64;
if __v7 != 0 {
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x014HugeTLB: unable to allocate vma specific lock\n".as_ptr() as *const u8) };
}
__v6 = __ret_do_once as i64;
return;
}
kref_init(unsafe { core::ptr::addr_of_mut!((*vma_lock).refs) });
loop {
unsafe { __init_rwsem(core::ptr::addr_of_mut!((*vma_lock).rw_sema), c"&vma_lock->rw_sema".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__455)) };
if true {
break;
}
}
unsafe {
(*vma_lock).vma = vma;
}
unsafe {
(*vma).vm_private_data = vma_lock as *mut ();
}
}

#[unsafe(link_section = ".data..once")]
static mut __already_done__450: bool = unsafe { core::mem::zeroed() };

static mut __key__455: lock_class_key = unsafe { core::mem::zeroed() };

/// Helper that removes a struct file_region from the resv_map cache and returns
/// it for use.
fn get_file_region_entry_from_cache(resv: *mut resv_map, from: i64, to: i64) -> *mut file_region {
let mut nrg: *mut file_region = unsafe { core::mem::zeroed() };
VM_BUG_ON!();
let __v1: i64 = unsafe { (*resv).region_cache_count };
unsafe {
(*resv).region_cache_count = __v1.wrapping_sub(1);
}
nrg = list_first_entry__16!(unsafe { core::ptr::addr_of_mut!((*resv).region_cache) }, file_region);
list_del(unsafe { core::ptr::addr_of_mut!((*nrg).link) });
unsafe {
(*nrg).from = from;
}
unsafe {
(*nrg).to = to;
}
nrg
}

fn copy_hugetlb_cgroup_uncharge_info(nrg: *mut file_region, rg: *mut file_region) {
unsafe {
(*nrg).reservation_counter = (*rg).reservation_counter;
}
unsafe {
(*nrg).css = (*rg).css;
}
if !unsafe { (*rg).css }.is_null() {
css_get(unsafe { (*rg).css });
}
}

/// Helper that records hugetlb_cgroup uncharge info.
fn record_hugetlb_cgroup_uncharge_info(h_cg: *mut hugetlb_cgroup, h: *mut hstate, resv: *mut resv_map, nrg: *mut file_region) {
if !h_cg.is_null() {
unsafe {
(*nrg).reservation_counter = ((*h_cg).rsvd_hugepage.as_mut_ptr() as *mut page_counter).wrapping_offset(hstate_index(h) as isize);
}
unsafe {
(*nrg).css = core::ptr::addr_of_mut!((*h_cg).css);
}
// The caller will hold exactly one h_cg->css reference for the
// whole contiguous reservation region. But this area might be
// scattered when there are already some file_regions reside in
// it. As a result, many file_regions may share only one css
// reference. In order to ensure that one file_region must hold
// exactly one h_cg->css reference, we should do css_get for
// each file_region and leave the reference held by caller
// untouched.
css_get(unsafe { core::ptr::addr_of_mut!((*h_cg).css) });
if (unsafe { (*resv).pages_per_hpage }) == 0 {
unsafe {
(*resv).pages_per_hpage = pages_per_huge_page(h as *const hstate) as u64;
}
}
// pages_per_hpage should be the same for all entries in
// a resv_map.
VM_BUG_ON!();
} else {
unsafe {
(*nrg).reservation_counter = core::ptr::null_mut::<page_counter>();
}
unsafe {
(*nrg).css = core::ptr::null_mut::<cgroup_subsys_state>();
}
}
}

fn put_uncharge_info(rg: *mut file_region) {
if !unsafe { (*rg).css }.is_null() {
css_put(unsafe { (*rg).css });
}
}

fn has_same_uncharge_info(rg: *mut file_region, org: *mut file_region) -> bool {
(unsafe { (*rg).reservation_counter }) == unsafe { (*org).reservation_counter } && (unsafe { (*rg).css }) == unsafe { (*org).css }
}

fn coalesce_file_region(resv: *mut resv_map, mut rg: *mut file_region) {
let mut nrg: *mut file_region = unsafe { core::mem::zeroed() };
let mut prg: *mut file_region = unsafe { core::mem::zeroed() };
prg = list_prev_entry__4!(rg, link);
let __v1: bool = (unsafe { core::ptr::addr_of_mut!((*prg).link) }) != unsafe { core::ptr::addr_of_mut!((*resv).regions) } && (unsafe { (*prg).to }) == unsafe { (*rg).from } && has_same_uncharge_info(prg, rg);
if __v1 {
unsafe {
(*prg).to = (*rg).to;
}
list_del(unsafe { core::ptr::addr_of_mut!((*rg).link) });
put_uncharge_info(rg);
unsafe { kfree(rg as *const ()) };
rg = prg;
}
nrg = list_next_entry__4!(rg, link);
let __v3: bool = (unsafe { core::ptr::addr_of_mut!((*nrg).link) }) != unsafe { core::ptr::addr_of_mut!((*resv).regions) } && (unsafe { (*nrg).from }) == unsafe { (*rg).to } && has_same_uncharge_info(nrg, rg);
if __v3 {
unsafe {
(*nrg).from = (*rg).from;
}
list_del(unsafe { core::ptr::addr_of_mut!((*rg).link) });
put_uncharge_info(rg);
unsafe { kfree(rg as *const ()) };
}
}

fn hugetlb_resv_map_add(map: *mut resv_map, rg: *mut list_head, from: i64, to: i64, h: *mut hstate, cg: *mut hugetlb_cgroup, regions_needed: *mut i64) -> i64 {
let mut nrg: *mut file_region = unsafe { core::mem::zeroed() };
if regions_needed.is_null() {
nrg = get_file_region_entry_from_cache(map, from, to);
record_hugetlb_cgroup_uncharge_info(cg, h, map, nrg);
list_add(unsafe { core::ptr::addr_of_mut!((*nrg).link) }, rg);
coalesce_file_region(map, nrg);
} else {
let __v1: i64 = unsafe { *regions_needed };
unsafe {
*regions_needed = __v1.wrapping_add(1);
}
}
to.wrapping_sub(from)
}

/// Must be called with resv->lock held.
///
/// Calling this with regions_needed != NULL will count the number of pages
/// to be added but will not modify the linked list. And regions_needed will
/// indicate the number of file_regions needed in the cache to carry out to add
/// the regions for this range.
fn add_reservation_in_range(resv: *mut resv_map, f: i64, t: i64, h_cg: *mut hugetlb_cgroup, h: *mut hstate, regions_needed: *mut i64) -> i64 {
let mut add: i64 = 0;
let head: *mut list_head = unsafe { core::ptr::addr_of_mut!((*resv).regions) };
let mut last_accounted_offset: i64 = f;
let mut iter: *mut file_region = unsafe { core::mem::zeroed() };
let mut trg: *mut file_region = core::ptr::null_mut::<file_region>();
let mut rg: *mut list_head = core::ptr::null_mut::<list_head>();
if !regions_needed.is_null() {
unsafe {
*regions_needed = 0;
}
}
// In this loop, we essentially handle an entry for the range
// [last_accounted_offset, iter->from), at every iteration, with some
// bounds checking.
iter = list_first_entry__8!(head);
trg = list_next_entry__4!(iter, link);
while list_entry_is_head!(iter, head, link) == 0 {
// Skip irrelevant regions that start before our range.
if (unsafe { (*iter).from }) < f {
// If this region ends after the last accounted offset,
// then we need to update last_accounted_offset.
if (unsafe { (*iter).to }) > last_accounted_offset {
last_accounted_offset = unsafe { (*iter).to };
}
} else {
// When we find a region that starts beyond our range, we've
// finished.
if (unsafe { (*iter).from }) >= t {
rg = unsafe { (*iter).link.prev };
break;
}
// Add an entry for last_accounted_offset -> iter->from, and
// update last_accounted_offset.
if (unsafe { (*iter).from }) > last_accounted_offset {
let __v4: i64 = hugetlb_resv_map_add(resv, unsafe { (*iter).link.prev }, last_accounted_offset, unsafe { (*iter).from }, h, h_cg, regions_needed);
add = add.wrapping_add(__v4);
}
last_accounted_offset = unsafe { (*iter).to };
}
iter = trg;
trg = list_next_entry__4!(trg, link);
}
// Handle the case where our range extends beyond
// last_accounted_offset.
if rg.is_null() {
rg = unsafe { (*head).prev };
}
if last_accounted_offset < t {
let __v9: i64 = hugetlb_resv_map_add(resv, rg, last_accounted_offset, t, h, h_cg, regions_needed);
add = add.wrapping_add(__v9);
}
add
}

/// Must be called with resv->lock acquired. Will drop lock to allocate entries.
fn allocate_file_region_entries(resv: *mut resv_map, regions_needed: i32) -> i32 {
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut rg: *mut file_region = unsafe { core::mem::zeroed() };
let mut trg: *mut file_region = unsafe { core::mem::zeroed() };
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut to_allocate: i32 = unsafe { core::mem::zeroed() };
let mut allocated_regions: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut list_head = core::ptr::addr_of_mut!(allocated_regions) as *mut list_head;
unsafe {
core::ptr::write(__slate_slot_17, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_17), prev: core::ptr::addr_of_mut!(*__slate_slot_17) });
to_allocate = 0;
i = 0;
trg = core::ptr::null_mut::<file_region>();
rg = core::ptr::null_mut::<file_region>();
VM_BUG_ON!();
// Check for sufficient descriptors in the cache to accommodate
// the number of in progress add operations plus regions_needed.
//
// This is a while loop because when we drop the lock, some other call
// to region_add or region_del may have consumed some region_entries,
// so we keep looping here until we finally have enough entries for
// (adds_in_progress + regions_needed).
'__join_3: {
while (*resv).region_cache_count < unsafe { (*resv).adds_in_progress }.wrapping_add(regions_needed as i64) {
to_allocate = unsafe { (*resv).adds_in_progress }.wrapping_add(regions_needed as i64).wrapping_sub((*resv).region_cache_count) as i32;
// At this point, we should have enough entries in the cache
// for all the existing adds_in_progress. We should only be
// needing to allocate for regions_needed.
VM_BUG_ON!();
spin_unlock(core::ptr::addr_of_mut!((*resv).lock));
i = 0;
while i < to_allocate {
__obj_size = size_mul(core::mem::size_of::<file_region>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_7 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_8 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
alloc_tag_restore!();
} else {
*__slate_slot_8 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_23 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_23)
});
}
trg = *__slate_slot_8 as *mut file_region;
if trg.is_null() {
break '__join_3;
} else {
list_add(core::ptr::addr_of_mut!((*trg).link), core::ptr::addr_of_mut!(*__slate_slot_17));
i = i.wrapping_add(1);
}
}
spin_lock(core::ptr::addr_of_mut!((*resv).lock));
list_splice_init(core::ptr::addr_of_mut!(*__slate_slot_17), core::ptr::addr_of_mut!((*resv).region_cache));
let __v3: i64 = (*resv).region_cache_count;
(*resv).region_cache_count = __v3.wrapping_add(to_allocate as i64);
}
return 0;
}
rg = list_first_entry__8!(core::ptr::addr_of_mut!(*__slate_slot_17));
trg = list_next_entry__4!(rg, link);
while list_entry_is_head!(rg, core::ptr::addr_of_mut!(*__slate_slot_17), link) == 0 {
list_del(core::ptr::addr_of_mut!((*rg).link));
kfree(rg as *const ());
rg = trg;
trg = list_next_entry__4!(trg, link);
}
return 12i32.wrapping_neg();
}
}

/// Add the huge page range represented by [f, t) to the reserve
/// map.  Regions will be taken from the cache to fill in this range.
/// Sufficient regions should exist in the cache due to the previous
/// call to region_chg with the same range, but in some cases the cache will not
/// have sufficient entries due to races with other code doing region_add or
/// region_del.  The extra needed entries will be allocated.
///
/// regions_needed is the out value provided by a previous call to region_chg.
///
/// Return the number of new huge pages added to the map.  This number is greater
/// than or equal to zero.  If file_region entries needed to be allocated for
/// this operation and we were not able to allocate, it returns -ENOMEM.
/// region_add of regions of length 1 never allocate file_regions and cannot
/// fail; region_chg will always allocate at least 1 entry and a region_add for
/// 1 page will only require at most 1 entry.
fn region_add(resv: *mut resv_map, f: i64, t: i64, in_regions_needed: i64, h: *mut hstate, h_cg: *mut hugetlb_cgroup) -> i64 {
let mut actual_regions_needed: i64 = unsafe { core::mem::zeroed() };
let mut add: i64 = unsafe { core::mem::zeroed() };
unsafe {
add = 0;
actual_regions_needed = 0;
spin_lock(core::ptr::addr_of_mut!((*resv).lock));
loop {
add_reservation_in_range(resv, f, t, core::ptr::null_mut::<hugetlb_cgroup>(), core::ptr::null_mut::<hstate>(), core::ptr::addr_of_mut!(actual_regions_needed)); // Count how many regions are actually needed to execute this add.
// Check for sufficient descriptors in the cache to accommodate
// this add operation. Note that actual_regions_needed may be greater
// than in_regions_needed, as the resv_map may have been modified since
// the region_chg call. In this case, we need to make sure that we
// allocate extra entries, such that we have enough for all the
// existing adds_in_progress, plus the excess needed for this
// operation.
if actual_regions_needed > in_regions_needed && (*resv).region_cache_count < unsafe { (*resv).adds_in_progress }.wrapping_add(actual_regions_needed.wrapping_sub(in_regions_needed)) {
// region_add operation of range 1 should never need to
// allocate file_region entries.
VM_BUG_ON!();
if allocate_file_region_entries(resv, actual_regions_needed.wrapping_sub(in_regions_needed) as i32) != 0 {
return 12i32.wrapping_neg() as i64;
}
} else {
break;
}
}
add = add_reservation_in_range(resv, f, t, h_cg, h, core::ptr::null_mut::<i64>());
let __v1: i64 = (*resv).adds_in_progress;
(*resv).adds_in_progress = __v1.wrapping_sub(in_regions_needed);
spin_unlock(core::ptr::addr_of_mut!((*resv).lock));
return add;
}
}

/// Examine the existing reserve map and determine how many
/// huge pages in the specified range [f, t) are NOT currently
/// represented.  This routine is called before a subsequent
/// call to region_add that will actually modify the reserve
/// map to add the specified range [f, t).  region_chg does
/// not change the number of huge pages represented by the
/// map.  A number of new file_region structures is added to the cache as a
/// placeholder, for the subsequent region_add call to use. At least 1
/// file_region structure is added.
///
/// out_regions_needed is the number of regions added to the
/// resv->adds_in_progress.  This value needs to be provided to a follow up call
/// to region_add or region_abort for proper accounting.
///
/// Returns the number of huge pages that need to be added to the existing
/// reservation map for the range [f, t).  This number is greater or equal to
/// zero.  -ENOMEM is returned if a new file_region structure or cache entry
/// is needed and can not be allocated.
fn region_chg(resv: *mut resv_map, f: i64, t: i64, out_regions_needed: *mut i64) -> i64 {
let mut chg: i64 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!((*resv).lock) });
// Count how many hugepages in this range are NOT represented.
chg = add_reservation_in_range(resv, f, t, core::ptr::null_mut::<hugetlb_cgroup>(), core::ptr::null_mut::<hstate>(), out_regions_needed);
if (unsafe { *out_regions_needed }) == 0 {
unsafe {
*out_regions_needed = 1;
}
}
if allocate_file_region_entries(resv, (unsafe { *out_regions_needed }) as i32) != 0 {
return 12i32.wrapping_neg() as i64;
}
let __v1: i64 = unsafe { (*resv).adds_in_progress };
unsafe {
(*resv).adds_in_progress = __v1.wrapping_add(*out_regions_needed);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*resv).lock) });
chg
}

/// Abort the in progress add operation.  The adds_in_progress field
/// of the resv_map keeps track of the operations in progress between
/// calls to region_chg and region_add.  Operations are sometimes
/// aborted after the call to region_chg.  In such cases, region_abort
/// is called to decrement the adds_in_progress counter. regions_needed
/// is the value returned by the region_chg call, it is used to decrement
/// the adds_in_progress counter.
///
/// NOTE: The range arguments [f, t) are not needed or used in this
/// routine.  They are kept to make reading the calling code easier as
/// arguments will match the associated region_chg call.
fn region_abort(resv: *mut resv_map, f: i64, t: i64, regions_needed: i64) {
spin_lock(unsafe { core::ptr::addr_of_mut!((*resv).lock) });
VM_BUG_ON!();
let __v1: i64 = unsafe { (*resv).adds_in_progress };
unsafe {
(*resv).adds_in_progress = __v1.wrapping_sub(regions_needed);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*resv).lock) });
}

/// Delete the specified range [f, t) from the reserve map.  If the
/// t parameter is LONG_MAX, this indicates that ALL regions after f
/// should be deleted.  Locate the regions which intersect [f, t)
/// and either trim, delete or split the existing regions.
///
/// Returns the number of huge pages deleted from the reserve map.
/// In the normal case, the return value is zero or more.  In the
/// case where a region must be split, a new region descriptor must
/// be allocated.  If the allocation fails, -ENOMEM will be returned.
/// NOTE: If the parameter t == LONG_MAX, then we will never split
/// a region and possibly return -ENOMEM.  Callers specifying
/// t == LONG_MAX do not need to check for -ENOMEM error.
fn region_del(resv: *mut resv_map, f: i64, t: i64) -> i64 {
let mut __slate_compound_25: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_27: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut del: i64 = unsafe { core::mem::zeroed() };
let mut nrg: *mut file_region = unsafe { core::mem::zeroed() };
let mut trg: core::mem::MaybeUninit<*mut file_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut file_region = core::ptr::addr_of_mut!(trg) as *mut *mut file_region;
let mut rg: core::mem::MaybeUninit<*mut file_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut file_region = core::ptr::addr_of_mut!(rg) as *mut *mut file_region;
let mut head: *mut list_head = unsafe { core::mem::zeroed() };
unsafe {
head = core::ptr::addr_of_mut!((*resv).regions);
let __slate_auto_bytes_2 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
nrg = core::ptr::null_mut::<file_region>();
del = 0;
'__join_0: {
loop {
spin_lock(core::ptr::addr_of_mut!((*resv).lock));
*__slate_slot_22 = list_first_entry__8!(head);
let __v18: *mut file_region = list_next_entry__4!(*__slate_slot_22, link);
*__slate_slot_21 = __v18;
loop {
if list_entry_is_head!(*__slate_slot_22, head, link) == 0 {
// Skip regions before the range to be deleted.  file_region
// ranges are normally of the form [from, to).  However, there
// may be a "placeholder" entry in the map which is of the form
// (from, to) with from == to.  Check for placeholder entries
// at the beginning of the range to be deleted.
if !((*(*__slate_slot_22)).to <= f && ((*(*__slate_slot_22)).to != (*(*__slate_slot_22)).from || (*(*__slate_slot_22)).to != f)) {
if (*(*__slate_slot_22)).from >= t {
break '__join_0;
} else {
if f > (*(*__slate_slot_22)).from && t < (*(*__slate_slot_22)).to {
break;
} else {
if f <= (*(*__slate_slot_22)).from && t >= (*(*__slate_slot_22)).to {
// Remove entire region
del = del.wrapping_add(unsafe { (*(*__slate_slot_22)).to }.wrapping_sub((*(*__slate_slot_22)).from));
hugetlb_cgroup_uncharge_file_region(resv, *__slate_slot_22, unsafe { (*(*__slate_slot_22)).to }.wrapping_sub((*(*__slate_slot_22)).from) as u64, true);
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_22)).link));
kfree(*__slate_slot_22 as *const ());
} else {
if f <= (*(*__slate_slot_22)).from {
// Trim beginning of region
hugetlb_cgroup_uncharge_file_region(resv, *__slate_slot_22, t.wrapping_sub((*(*__slate_slot_22)).from) as u64, false);
del = del.wrapping_add(t.wrapping_sub((*(*__slate_slot_22)).from));
(*(*__slate_slot_22)).from = t;
} else {
// Trim end of region
hugetlb_cgroup_uncharge_file_region(resv, *__slate_slot_22, unsafe { (*(*__slate_slot_22)).to }.wrapping_sub(f) as u64, false);
del = del.wrapping_add(unsafe { (*(*__slate_slot_22)).to }.wrapping_sub(f));
(*(*__slate_slot_22)).to = f;
}
}
}
}
}
*__slate_slot_22 = *__slate_slot_21;
let __v0: *mut file_region = list_next_entry__4!(*__slate_slot_21, link);
*__slate_slot_21 = __v0;
} else {
break '__join_0;
}
}
// Must split region
// Check for an entry in the cache before dropping
// lock and attempting allocation.
if nrg.is_null() && (*resv).region_cache_count > (*resv).adds_in_progress {
nrg = list_first_entry__16!(core::ptr::addr_of_mut!((*resv).region_cache), file_region);
list_del(core::ptr::addr_of_mut!((*nrg).link));
let __v16: i64 = (*resv).region_cache_count;
(*resv).region_cache_count = __v16.wrapping_sub(1);
}
if nrg.is_null() {
spin_unlock(core::ptr::addr_of_mut!((*resv).lock));
__obj_size = size_mul(core::mem::size_of::<file_region>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_9 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_10 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_25 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_25)
});
alloc_tag_restore!();
} else {
*__slate_slot_10 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_27 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_27)
});
}
nrg = *__slate_slot_10 as *mut file_region;
if nrg.is_null() {
return 12i32.wrapping_neg() as i64;
}
} else {
break;
}
}
del = del.wrapping_add(t.wrapping_sub(f));
hugetlb_cgroup_uncharge_file_region(resv, *__slate_slot_22, t.wrapping_sub(f) as u64, false);
// New entry for end of split region
(*nrg).from = t;
(*nrg).to = (*(*__slate_slot_22)).to;
copy_hugetlb_cgroup_uncharge_info(nrg, *__slate_slot_22);
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*nrg).link));
// Original entry is trimmed
(*(*__slate_slot_22)).to = f;
list_add(core::ptr::addr_of_mut!((*nrg).link), core::ptr::addr_of_mut!((*(*__slate_slot_22)).link));
nrg = core::ptr::null_mut::<file_region>();
}
spin_unlock(core::ptr::addr_of_mut!((*resv).lock));
kfree(nrg as *const ());
return del;
}
}

/// A rare out of memory error was encountered which prevented removal of
/// the reserve map region for a page.  The huge page itself was free'ed
/// and removed from the page cache.  This routine will adjust the subpool
/// usage count, and the global reserve count if needed.  By incrementing
/// these counts, the reserve map entry which could not be deleted will
/// appear as a "reserved" entry instead of simply dangling with incorrect
/// counts.
#[unsafe(no_mangle)]
extern "C" fn hugetlb_fix_reserve_counts(inode: *mut inode) {
let spool: *mut hugepage_subpool = subpool_inode(inode);
let rsv_adjust: i64 = hugepage_subpool_get_pages(spool, 1i64);
let mut reserved: bool = false;
if rsv_adjust > 0 {
let h: *mut hstate = hstate_inode(inode);
if hugetlb_acct_memory(h, 1i64) == 0 {
reserved = true;
}
} else {
if rsv_adjust == 0 {
reserved = true;
}
}
if !reserved {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014hugetlb: Huge Page Reserved count may go negative.\n".as_ptr() as *const u8) };
}
}

// Count and return the number of huge pages in the reserve map
// that intersect with the range [f, t).
fn region_count(resv: *mut resv_map, f: i64, t: i64) -> i64 {
let head: *mut list_head = unsafe { core::ptr::addr_of_mut!((*resv).regions) };
let mut rg: *mut file_region = unsafe { core::mem::zeroed() };
let mut chg: i64 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!((*resv).lock) });
// Locate each segment we overlap with, and count that overlap.
rg = list_first_entry__8!(head);
while list_entry_is_head!(rg, head, link) == 0 {
let mut seg_from: i64 = 0;
let mut seg_to: i64 = 0;
if (unsafe { (*rg).to }) > f {
if (unsafe { (*rg).from }) >= t {
break;
}
let mut __v2: i64;
let __UNIQUE_ID_x__698: i64 = unsafe { (*rg).from };
let __UNIQUE_ID_y__699: i64 = f;
loop {
if __types_ok__6!(__UNIQUE_ID_x__698, __UNIQUE_ID_y__699) == 0 {
unsafe { __compiletime_assert_700() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__698 > __UNIQUE_ID_y__699 { __UNIQUE_ID_x__698 } else { __UNIQUE_ID_y__699 };
seg_from = __v2;
let mut __v4: i64;
let __UNIQUE_ID_x__701: i64 = unsafe { (*rg).to };
let __UNIQUE_ID_y__702: i64 = t;
loop {
if __types_ok__6!(__UNIQUE_ID_x__701, __UNIQUE_ID_y__702) == 0 {
unsafe { __compiletime_assert_703() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__701 < __UNIQUE_ID_y__702 { __UNIQUE_ID_x__701 } else { __UNIQUE_ID_y__702 };
seg_to = __v4;
chg = chg.wrapping_add(seg_to.wrapping_sub(seg_from));
}
rg = list_next_entry__4!(rg, link);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*resv).lock) });
chg
}

/// Convert the address within this vma to the page offset within
/// the mapping, huge page units here.
fn vma_hugecache_offset(h: *mut hstate, vma: *mut vm_area_struct, address: u64) -> u64 {
linear_page_index(vma as *const vm_area_struct, address) >> huge_page_order(h)
}

// Flags for MAP_PRIVATE reservations.  These are stored in the bottom
// bits of the reservation map pointer, which are always clear due to
// alignment.
/// These helpers are used to track how many pages are reserved for
/// faults in a MAP_PRIVATE mapping. Only the process that called mmap()
/// is guaranteed to have their future faults succeed.
///
/// With the exception of hugetlb_dup_vma_private() which is called at fork(),
/// the reserve counters are updated with the hugetlb_lock held. It is safe
/// to reset the VMA at fork() time as it is not in use yet and there is no
/// chance of the global counters getting corrupted as a result of the values.
///
/// The private mapping reservation is represented in a subtly different
/// manner to a shared mapping.  A shared mapping has a region map associated
/// with the underlying file, this region map represents the backing file
/// pages which have ever had a reservation assigned which this persists even
/// after the page is instantiated.  A private mapping has a region map
/// associated with the original mmap which is attached to all VMAs which
/// reference it, this region map represents those offsets which have consumed
/// reservation ie. where pages have been instantiated.
fn get_vma_private_data(vma: *mut vm_area_struct) -> u64 {
(unsafe { (*vma).vm_private_data }) as u64
}

fn set_vma_private_data(vma: *mut vm_area_struct, value: u64) {
unsafe {
(*vma).vm_private_data = value as *mut ();
}
}

fn resv_map_set_hugetlb_cgroup_uncharge_info(resv_map: *mut resv_map, h_cg: *mut hugetlb_cgroup, h: *mut hstate) {
if h_cg.is_null() || h.is_null() {
unsafe {
(*resv_map).reservation_counter = core::ptr::null_mut::<page_counter>();
}
unsafe {
(*resv_map).pages_per_hpage = 0;
}
unsafe {
(*resv_map).css = core::ptr::null_mut::<cgroup_subsys_state>();
}
} else {
unsafe {
(*resv_map).reservation_counter = ((*h_cg).rsvd_hugepage.as_mut_ptr() as *mut page_counter).wrapping_offset(hstate_index(h) as isize);
}
unsafe {
(*resv_map).pages_per_hpage = pages_per_huge_page(h as *const hstate) as u64;
}
unsafe {
(*resv_map).css = core::ptr::addr_of_mut!((*h_cg).css);
}
}
}

#[unsafe(no_mangle)]
extern "C" fn resv_map_alloc() -> *mut resv_map {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: spinlock = unsafe { core::mem::zeroed() };
let mut resv_map: *mut resv_map = unsafe { core::mem::zeroed() };
let mut __v0: *mut resv_map;
{
let __obj_size: u64 = size_mul(core::mem::size_of::<resv_map>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut resv_map;
}
resv_map = __v0;
let mut rg: *mut file_region = unsafe { core::mem::zeroed() };
let mut __v6: *mut file_region;
let __obj_size: u64 = size_mul(core::mem::size_of::<file_region>() as u64, 1u64);
let mut __v7: *mut ();
let mut __v8: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
} });
}
__v8 = _res;
__v7 = __v8;
__v6 = __v7 as *mut file_region;
rg = __v6;
if resv_map.is_null() || rg.is_null() {
unsafe { kfree(resv_map as *const ()) };
unsafe { kfree(rg as *const ()) };
return core::ptr::null_mut::<resv_map>();
}
kref_init(unsafe { core::ptr::addr_of_mut!((*resv_map).refs) });
'__slate_break_13: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*resv_map).lock) });
unsafe {
*core::ptr::addr_of_mut!((*resv_map).lock) = *{
__slate_compound_12 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_12)
};
}
if true {
break;
}
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*resv_map).regions) });
loop {
unsafe { __init_rwsem(core::ptr::addr_of_mut!((*resv_map).rw_sema), c"&resv_map->rw_sema".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1095)) };
if true {
break;
}
}
unsafe {
(*resv_map).adds_in_progress = 0;
}
// Initialize these to 0. On shared mappings, 0's here indicate these
// fields don't do cgroup accounting. On private mappings, these will be
// re-initialized to the proper values, to indicate that hugetlb cgroup
// reservations are to be un-charged from here.
resv_map_set_hugetlb_cgroup_uncharge_info(resv_map, core::ptr::null_mut::<hugetlb_cgroup>(), core::ptr::null_mut::<hstate>());
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*resv_map).region_cache) });
list_add(unsafe { core::ptr::addr_of_mut!((*rg).link) }, unsafe { core::ptr::addr_of_mut!((*resv_map).region_cache) });
unsafe {
(*resv_map).region_cache_count = 1;
}
resv_map
}

static mut __key__1095: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hugetlb.resv_map_release")]
extern "C" fn resv_map_release(r#ref: *mut kref) {
let resv_map: *mut resv_map = container_of__16!(r#ref, resv_map);
let head: *mut list_head = unsafe { core::ptr::addr_of_mut!((*resv_map).region_cache) };
let mut rg: *mut file_region = unsafe { core::mem::zeroed() };
let mut trg: *mut file_region = unsafe { core::mem::zeroed() };
// Clear out any active regions before we release the map.
region_del(resv_map, 0i64, 9223372036854775807i64);
// ... and any entries left in the cache
rg = list_first_entry__8!(head);
trg = list_next_entry__4!(rg, link);
while list_entry_is_head!(rg, head, link) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*rg).link) });
unsafe { kfree(rg as *const ()) };
rg = trg;
trg = list_next_entry__4!(trg, link);
}
VM_BUG_ON!();
unsafe { kfree(resv_map as *const ()) };
}

fn inode_resv_map(inode: *mut inode) -> *mut resv_map {
unsafe { (*HUGETLBFS_I(inode)).resv_map }
}

fn vma_resv_map(vma: *mut vm_area_struct) -> *mut resv_map {
VM_BUG_ON_VMA!();
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) != 0 {
let mapping: *mut address_space = unsafe { (*(*vma).vm_file).f_mapping };
let inode: *mut inode = unsafe { (*mapping).host };
return inode_resv_map(inode);
} else {
return (get_vma_private_data(vma) & 18446744073709551612u64) as *mut resv_map;
}
}

fn set_vma_resv_map(vma: *mut vm_area_struct, map: *mut resv_map) {
VM_WARN_ON_ONCE_VMA!();
VM_WARN_ON_ONCE_VMA!();
set_vma_private_data(vma, map as u64);
}

fn set_vma_resv_flags(vma: *mut vm_area_struct, flags: u64) {
VM_WARN_ON_ONCE_VMA!();
VM_WARN_ON_ONCE_VMA!();
set_vma_private_data(vma, get_vma_private_data(vma) | flags);
}

fn is_vma_resv_set(vma: *mut vm_area_struct, flag: u64) -> i32 {
VM_BUG_ON_VMA!();
(get_vma_private_data(vma) & flag != 0) as i32
}

#[unsafe(no_mangle)]
extern "C" fn __vma_private_lock(vma: *mut vm_area_struct) -> bool {
let __v0: bool = (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) == 0 && get_vma_private_data(vma) & 18446744073709551612u64 != 0;
__v0 && is_vma_resv_set(vma, 1u64) != 0
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_dup_vma_private(vma: *mut vm_area_struct) {
VM_BUG_ON_VMA!();
// Clear vm_private_data
// - For shared mappings this is a per-vma semaphore that may be
//   allocated in a subsequent call to hugetlb_vm_op_open.
//   Before clearing, make sure pointer is not associated with vma
//   as this will leak the structure.  This is the case when called
//   via clear_vma_resv_huge_pages() and hugetlb_vm_op_open has already
//   been called to allocate a new structure.
// - For MAP_PRIVATE mappings, this is the reserve map which does
//   not apply to children.  Faults generated by the children are
//   not guaranteed to succeed, even if read-only.
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) != 0 {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
if !vma_lock.is_null() && (unsafe { (*vma_lock).vma }) != vma {
unsafe {
(*vma).vm_private_data = core::ptr::null_mut::<()>();
}
}
} else {
unsafe {
(*vma).vm_private_data = core::ptr::null_mut::<()>();
}
}
}

/// Reset and decrement one ref on hugepage private reservation.
/// Called with mm->mmap_lock writer semaphore held.
/// This function should be only used by mremap and operate on
/// same sized vma. It should never come here with last ref on the
/// reservation.
#[unsafe(no_mangle)]
extern "C" fn clear_vma_resv_huge_pages(vma: *mut vm_area_struct) {
// Clear the old hugetlb private page reservation.
// It has already been transferred to new_vma.
//
// During a mremap() operation of a hugetlb vma we call move_vma()
// which copies vma into new_vma and unmaps vma. After the copy
// operation both new_vma and vma share a reference to the resv_map
// struct, and at that point vma is about to be unmapped. We don't
// want to return the reservation to the pool at unmap of vma because
// the reservation still lives on in new_vma, so simply decrement the
// ref here and remove the resv_map reference from this vma.
let reservations: *mut resv_map = vma_resv_map(vma);
let __v0: bool = !reservations.is_null() && is_vma_resv_set(vma, 1u64) != 0;
if __v0 {
resv_map_put_hugetlb_cgroup_uncharge_info(reservations);
kref_put(unsafe { core::ptr::addr_of_mut!((*reservations).refs) }, Some(resv_map_release));
}
hugetlb_dup_vma_private(vma);
}

fn enqueue_hugetlb_folio(h: *mut hstate, folio: *mut folio) {
let nid: i32 = folio_nid(folio as *const folio);
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
VM_BUG_ON_FOLIO!();
list_move(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) }, unsafe { (*h).hugepage_freelists.as_mut_ptr() as *mut list_head }.wrapping_offset(nid as isize));
let __v2: u64 = unsafe { (*h).free_huge_pages };
unsafe {
(*h).free_huge_pages = __v2.wrapping_add(1);
}
let __v4: *mut u32 = unsafe { (*h).free_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(nid as isize);
let __v5: u32 = unsafe { *__v4 };
unsafe {
*__v4 = __v5.wrapping_add(1);
}
folio_set_hugetlb_freed(folio);
}

fn dequeue_hugetlb_folio_node_exact(h: *mut hstate, nid: i32) -> *mut folio {
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let pin: bool = (unsafe { (*get_current()).flags }) & 268435456 != 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
folio = list_first_entry!(unsafe { (*h).hugepage_freelists.as_mut_ptr() as *mut list_head }.wrapping_offset(nid as isize));
while list_entry_is_head__2!(folio, unsafe { (*h).hugepage_freelists.as_mut_ptr() as *mut list_head }.wrapping_offset(nid as isize), lru) == 0 {
let __v2: bool = pin && !folio_is_longterm_pinnable(folio);
if !__v2 {
if !folio_test_hwpoison(folio as *const folio) {
if !is_migrate_isolate_page(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }) {
list_move(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) }, unsafe { core::ptr::addr_of_mut!((*h).hugepage_activelist) });
folio_ref_unfreeze(folio, 1i32);
folio_clear_hugetlb_freed(folio);
let __v5: u64 = unsafe { (*h).free_huge_pages };
unsafe {
(*h).free_huge_pages = __v5.wrapping_sub(1);
}
let __v7: *mut u32 = unsafe { (*h).free_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(nid as isize);
let __v8: u32 = unsafe { *__v7 };
unsafe {
*__v7 = __v8.wrapping_sub(1);
}
return folio;
}
}
}
folio = list_next_entry!(folio, lru);
}
core::ptr::null_mut::<folio>()
}

fn dequeue_hugetlb_folio_nodemask(h: *mut hstate, gfp_mask: u32, mut nid: i32, nmask: *mut __SlateRecord_nodemask_types_h_17) -> *mut folio {
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut node: i32 = unsafe { core::mem::zeroed() };
let mut z: core::mem::MaybeUninit<*mut zoneref> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut zoneref = core::ptr::addr_of_mut!(z) as *mut *mut zoneref;
let mut zone: core::mem::MaybeUninit<*mut zone> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut zone = core::ptr::addr_of_mut!(zone) as *mut *mut zone;
let mut zonelist: core::mem::MaybeUninit<*mut zonelist> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut zonelist = core::ptr::addr_of_mut!(zonelist) as *mut *mut zonelist;
let mut cpuset_mems_cookie: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(cpuset_mems_cookie) as *mut u32;
unsafe {
{
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
node = 1i32.wrapping_neg();
// 'nid' should not be NUMA_NO_NODE. Try to catch any misuse of it and rectifiy.
if nid == 1i32.wrapping_neg() {
nid = numa_node_id();
}
}
*__slate_slot_8 = node_zonelist(nid, gfp_mask);
loop {
*__slate_slot_9 = read_mems_allowed_begin();
*__slate_slot_6 = first_zones_zonelist(*__slate_slot_8, gfp_zone(gfp_mask), nmask as *const __SlateRecord_nodemask_types_h_17);
let __v4: *mut zone = zonelist_zone(*__slate_slot_6);
*__slate_slot_7 = __v4;
while !(*__slate_slot_7).is_null() {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if cpuset_zone_allowed(*__slate_slot_7, gfp_mask) {
// no need to ask again on the same node. Pool is node rather than
// zone aware
if zone_to_nid(*__slate_slot_7 as *const zone) != node {
node = zone_to_nid(*__slate_slot_7 as *const zone);
*__slate_slot_3 = dequeue_hugetlb_folio_node_exact(h, node);
if !(*__slate_slot_3).is_null() {
return *__slate_slot_3;
}
}
}
let __v2: *mut zoneref = *__slate_slot_6;
let __v1: *mut zoneref = __v2.wrapping_offset(1 as isize);
*__slate_slot_6 = __v1;
*__slate_slot_6 = next_zones_zonelist(__v1, gfp_zone(gfp_mask), nmask as *const __SlateRecord_nodemask_types_h_17);
let __v0: *mut zone = zonelist_zone(*__slate_slot_6);
*__slate_slot_7 = __v0;
}
if unlikely!(read_mems_allowed_retry(*__slate_slot_9)) == 0 {
break;
}
}
return core::ptr::null_mut::<folio>();
}
}

fn available_huge_pages(h: *mut hstate) -> u64 {
unsafe { (*h).free_huge_pages }.wrapping_sub(unsafe { (*h).resv_huge_pages })
}

fn dequeue_hugetlb_folio(h: *mut hstate, gfp_mask: u32, mpoli: *mut mempolicy_interpreted) -> *mut folio {
let mut nodemask: *mut __SlateRecord_nodemask_types_h_17 = unsafe { (*mpoli).nodemask };
let mut folio: *mut folio = core::ptr::null_mut::<folio>();
if ((unsafe { (*mpoli).mode }) as u32) == 5 {
folio = dequeue_hugetlb_folio_nodemask(h, gfp_mask, unsafe { (*mpoli).nid }, nodemask);
// Fallback to all nodes if page==NULL
nodemask = core::ptr::null_mut::<__SlateRecord_nodemask_types_h_17>();
}
if folio.is_null() {
folio = dequeue_hugetlb_folio_nodemask(h, gfp_mask, unsafe { (*mpoli).nid }, nodemask);
}
folio
}

fn alloc_gigantic_frozen_folio(order: i32, gfp_mask: u32, nid: i32, nodemask: *mut __SlateRecord_nodemask_types_h_17) -> *mut folio {
core::ptr::null_mut::<folio>()
}

/// Remove hugetlb folio from lists.
/// If vmemmap exists for the folio, clear the hugetlb flag so that the
/// folio appears as just a compound page.  Otherwise, wait until after
/// allocating vmemmap to clear the flag.
///
/// Must be called with hugetlb lock held.
#[unsafe(no_mangle)]
extern "C" fn remove_hugetlb_folio(h: *mut hstate, folio: *mut folio, adjust_surplus: bool) {
let nid: i32 = folio_nid(folio as *const folio);
VM_BUG_ON_FOLIO!();
VM_BUG_ON_FOLIO!();
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
if hstate_is_gigantic_no_runtime(h) {
return;
}
list_del(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) });
if folio_test_hugetlb_freed(folio) {
folio_clear_hugetlb_freed(folio);
let __v2: u64 = unsafe { (*h).free_huge_pages };
unsafe {
(*h).free_huge_pages = __v2.wrapping_sub(1);
}
let __v4: *mut u32 = unsafe { (*h).free_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(nid as isize);
let __v5: u32 = unsafe { *__v4 };
unsafe {
*__v4 = __v5.wrapping_sub(1);
}
}
if adjust_surplus {
let __v8: u64 = unsafe { (*h).surplus_huge_pages };
unsafe {
(*h).surplus_huge_pages = __v8.wrapping_sub(1);
}
let __v10: *mut u32 = unsafe { (*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(nid as isize);
let __v11: u32 = unsafe { *__v10 };
unsafe {
*__v10 = __v11.wrapping_sub(1);
}
}
// We can only clear the hugetlb flag after allocating vmemmap
// pages.  Otherwise, someone (memory error handling) may try to write
// to tail struct pages.
if !folio_test_hugetlb_vmemmap_optimized(folio) {
__folio_clear_hugetlb(folio);
}
let __v14: u64 = unsafe { (*h).nr_huge_pages };
unsafe {
(*h).nr_huge_pages = __v14.wrapping_sub(1);
}
let __v16: *mut u32 = unsafe { (*h).nr_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(nid as isize);
let __v17: u32 = unsafe { *__v16 };
unsafe {
*__v16 = __v17.wrapping_sub(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn add_hugetlb_folio(h: *mut hstate, folio: *mut folio, adjust_surplus: bool) {
let nid: i32 = folio_nid(folio as *const folio);
VM_BUG_ON_FOLIO!();
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) });
let __v2: u64 = unsafe { (*h).nr_huge_pages };
unsafe {
(*h).nr_huge_pages = __v2.wrapping_add(1);
}
let __v4: *mut u32 = unsafe { (*h).nr_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(nid as isize);
let __v5: u32 = unsafe { *__v4 };
unsafe {
*__v4 = __v5.wrapping_add(1);
}
if adjust_surplus {
let __v8: u64 = unsafe { (*h).surplus_huge_pages };
unsafe {
(*h).surplus_huge_pages = __v8.wrapping_add(1);
}
let __v10: *mut u32 = unsafe { (*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(nid as isize);
let __v11: u32 = unsafe { *__v10 };
unsafe {
*__v10 = __v11.wrapping_add(1);
}
}
__folio_set_hugetlb(folio);
folio_change_private(folio, core::ptr::null_mut::<()>());
// We have to set hugetlb_vmemmap_optimized again as above
// folio_change_private(folio, NULL) cleared it.
folio_set_hugetlb_vmemmap_optimized(folio);
arch_clear_hugetlb_flags(folio);
enqueue_hugetlb_folio(h, folio);
}

fn __update_and_free_hugetlb_folio(h: *mut hstate, folio: *mut folio) {
let clear_flag: bool = folio_test_hugetlb_vmemmap_optimized(folio);
if hstate_is_gigantic_no_runtime(h) {
return;
}
// If we don't know which subpages are hwpoisoned, we can't free
// the hugepage, so it's leaked intentionally.
if folio_test_hugetlb_raw_hwp_unreliable(folio) {
return;
}
// If folio is not vmemmap optimized (!clear_flag), then the folio
// is no longer identified as a hugetlb page.  hugetlb_vmemmap_restore_folio
// can only be passed hugetlb pages and will BUG otherwise.
let __v0: bool = clear_flag && (unsafe { hugetlb_vmemmap_restore_folio(h as *const hstate, folio) }) != 0;
if __v0 {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
// If we cannot allocate vmemmap pages, just refuse to free the
// page and put the page back on the hugetlb free list and treat
// as a surplus page.
add_hugetlb_folio(h, folio, true);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
return;
}
// If vmemmap pages were allocated above, then we need to clear the
// hugetlb flag under the hugetlb lock.
if folio_test_hugetlb(folio as *const folio) {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
__folio_clear_hugetlb(folio);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
}
// Move PageHWPoison flag from head page to the raw error pages,
// which makes any healthy subpages reusable.
if unlikely!(folio_test_hwpoison(folio as *const folio)) != 0 {
folio_clear_hugetlb_hwpoison(folio);
}
VM_BUG_ON_FOLIO!();
if folio_test_hugetlb_cma(folio) {
hugetlb_cma_free_frozen_folio(folio);
} else {
unsafe { free_frozen_pages(core::ptr::addr_of_mut!((*folio).__slate_anon_0.page), folio_order(folio as *const folio)) };
}
}

/// As update_and_free_hugetlb_folio() can be called under any context, so we cannot
/// use GFP_KERNEL to allocate vmemmap pages. However, we can defer the
/// actual freeing in a workqueue to prevent from using GFP_ATOMIC to allocate
/// the vmemmap pages.
///
/// free_hpage_workfn() locklessly retrieves the linked list of pages to be
/// freed and frees them one-by-one. As the page->mapping pointer is going
/// to be cleared in free_hpage_workfn() anyway, it is reused as the llist_node
/// structure of a lockless linked list of huge pages to be freed.
static mut hpage_freelist: llist_head = llist_head { first: core::ptr::null_mut::<llist_node>() };

#[unsafe(link_section = ".text.slate_distinct.hugetlb.free_hpage_workfn")]
extern "C" fn free_hpage_workfn(work: *mut work_struct) {
let mut node: *mut llist_node = unsafe { core::mem::zeroed() };
node = llist_del_all(unsafe { core::ptr::addr_of_mut!(hpage_freelist) });
while !node.is_null() {
let folio: *mut folio = container_of__15!(node as *mut *mut address_space, folio);
node = unsafe { (*node).next };
unsafe {
(*folio).__slate_anon_0.__slate_anon_0.mapping = core::ptr::null_mut::<address_space>();
}
// The VM_BUG_ON_FOLIO(!folio_test_hugetlb(folio), folio) in
// folio_hstate() is going to trigger because a previous call to
// remove_hugetlb_folio() will clear the hugetlb bit, so do
// not use folio_hstate() directly.
let h: *mut hstate = size_to_hstate(folio_size(folio as *const folio));
__update_and_free_hugetlb_folio(h, folio);
cond_resched__16!();
}
}

static mut free_hpage_work: work_struct = work_struct { data: __SlateRecord_types_h_197 { counter: 4503599625273344i64 }, entry: list_head { next: unsafe { core::ptr::addr_of_mut!(free_hpage_work.entry) }, prev: unsafe { core::ptr::addr_of_mut!(free_hpage_work.entry) } }, func: Some(free_hpage_workfn) };

fn flush_free_hpage_work(h: *mut hstate) {
if hugetlb_vmemmap_optimizable(h as *const hstate) {
unsafe { flush_work(core::ptr::addr_of_mut!(free_hpage_work)) };
}
}

fn update_and_free_hugetlb_folio(h: *mut hstate, folio: *mut folio, atomic: bool) {
if !folio_test_hugetlb_vmemmap_optimized(folio) || !atomic {
__update_and_free_hugetlb_folio(h, folio);
return;
}
// Defer freeing to avoid using GFP_ATOMIC to allocate vmemmap pages.
//
// Only call schedule_work() if hpage_freelist is previously
// empty. Otherwise, schedule_work() had been called but the workfn
// hasn't retrieved the list yet.
if llist_add((unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.mapping) }) as *mut llist_node, unsafe { core::ptr::addr_of_mut!(hpage_freelist) }) {
schedule_work(unsafe { core::ptr::addr_of_mut!(free_hpage_work) });
}
}

fn bulk_vmemmap_restore_error(h: *mut hstate, folio_list: *mut list_head, non_hvo_folios: *mut list_head) {
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let mut t_folio: *mut folio = unsafe { core::mem::zeroed() };
if list_empty(non_hvo_folios as *const list_head) == 0 {
// Free any restored hugetlb pages so that restore of the
// entire list can be retried.
// The idea is that in the common case of ENOMEM errors freeing
// hugetlb pages with vmemmap we will free up memory so that we
// can allocate vmemmap for more hugetlb pages.
folio = list_first_entry!(non_hvo_folios);
t_folio = list_next_entry!(folio, lru);
while list_entry_is_head__2!(folio, non_hvo_folios, lru) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) });
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
__folio_clear_hugetlb(folio);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
update_and_free_hugetlb_folio(h, folio, false);
cond_resched__36!();
folio = t_folio;
t_folio = list_next_entry!(t_folio, lru);
}
} else {
// In the case where there are no folios which can be
// immediately freed, we loop through the list trying to restore
// vmemmap individually in the hope that someone elsewhere may
// have done something to cause success (such as freeing some
// memory).  If unable to restore a hugetlb page, the hugetlb
// page is made a surplus page and removed from the list.
// If are able to restore vmemmap and free one hugetlb page, we
// quit processing the list to retry the bulk operation.
folio = list_first_entry!(folio_list);
t_folio = list_next_entry!(folio, lru);
while list_entry_is_head__2!(folio, folio_list, lru) == 0 {
if (unsafe { hugetlb_vmemmap_restore_folio(h as *const hstate, folio) }) != 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) });
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
add_hugetlb_folio(h, folio, true);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
} else {
list_del(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) });
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
__folio_clear_hugetlb(folio);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
update_and_free_hugetlb_folio(h, folio, false);
cond_resched__101!();
break;
}
folio = t_folio;
t_folio = list_next_entry!(t_folio, lru);
}
}
}

fn update_and_free_pages_bulk(h: *mut hstate, folio_list: *mut list_head) {
let mut non_hvo_folios: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut list_head = core::ptr::addr_of_mut!(non_hvo_folios) as *mut list_head;
let mut t_folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut folio = core::ptr::addr_of_mut!(t_folio) as *mut *mut folio;
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut ret: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i64 = core::ptr::addr_of_mut!(ret) as *mut i64;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_2, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_2), prev: core::ptr::addr_of_mut!(*__slate_slot_2) });
// First allocate required vmemmmap (if necessary) for all folios.
// Carefully handle errors and free up any available hugetlb pages
// in an effort to make forward progress.
loop {
*__slate_slot_5 = hugetlb_vmemmap_restore_folios(h as *const hstate, folio_list, core::ptr::addr_of_mut!(*__slate_slot_2));
if *__slate_slot_5 < 0 {
bulk_vmemmap_restore_error(h, folio_list, core::ptr::addr_of_mut!(*__slate_slot_2));
} else {
break;
}
}
// At this point, list should be empty, ret should be >= 0 and there
// should only be pages on the non_hvo_folios list.
// Do note that the non_hvo_folios list could be empty.
// Without HVO enabled, ret will be 0 and there is no need to call
// __folio_clear_hugetlb as this was done previously.
VM_WARN_ON!();
VM_WARN_ON!();
if list_empty(core::ptr::addr_of_mut!(*__slate_slot_2) as *const list_head) == 0 && *__slate_slot_5 != 0 {
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
*__slate_slot_4 = list_first_entry!(core::ptr::addr_of_mut!(*__slate_slot_2));
while list_entry_is_head__2!(*__slate_slot_4, core::ptr::addr_of_mut!(*__slate_slot_2), lru) == 0 {
__folio_clear_hugetlb(*__slate_slot_4);
*__slate_slot_4 = list_next_entry!(*__slate_slot_4, lru);
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
}
*__slate_slot_4 = list_first_entry!(core::ptr::addr_of_mut!(*__slate_slot_2));
let __v1: *mut folio = list_next_entry!(*__slate_slot_4, lru);
*__slate_slot_3 = __v1;
while list_entry_is_head__2!(*__slate_slot_4, core::ptr::addr_of_mut!(*__slate_slot_2), lru) == 0 {
update_and_free_hugetlb_folio(h, *__slate_slot_4, false);
cond_resched__92!();
*__slate_slot_4 = *__slate_slot_3;
let __v0: *mut folio = list_next_entry!(*__slate_slot_3, lru);
*__slate_slot_3 = __v0;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn size_to_hstate(size: u64) -> *mut hstate {
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
if huge_page_size(h as *const hstate) == size {
return h;
}
let __v1: *mut hstate = h;
h = __v1.wrapping_offset(1 as isize);
}
core::ptr::null_mut::<hstate>()
}

#[unsafe(no_mangle)]
extern "C" fn free_huge_folio(folio: *mut folio) {
// Can't pass hstate in here because it is called from the
// generic mm code.
let h: *mut hstate = folio_hstate(folio);
let nid: i32 = folio_nid(folio as *const folio);
let spool: *mut hugepage_subpool = hugetlb_folio_subpool(folio);
let mut restore_reserve: bool = false;
let mut flags: u64 = 0;
VM_BUG_ON_FOLIO!();
VM_BUG_ON_FOLIO!();
hugetlb_set_folio_subpool(folio, core::ptr::null_mut::<hugepage_subpool>());
if folio_test_anon(folio as *const folio) {
__ClearPageAnonExclusive(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) });
}
unsafe {
(*folio).__slate_anon_0.__slate_anon_0.mapping = core::ptr::null_mut::<address_space>();
}
restore_reserve = folio_test_hugetlb_restore_reserve(folio);
folio_clear_hugetlb_restore_reserve(folio);
// If HPageRestoreReserve was set on page, page allocation consumed a
// reservation.  If the page was associated with a subpool, there
// would have been a page reserved in the subpool before allocation
// via hugepage_subpool_get_pages().  Since we are 'restoring' the
// reservation, do not call hugepage_subpool_put_pages() as this will
// remove the reserved page from the subpool.
if !restore_reserve {
// A return code of zero implies that the subpool will be
// under its minimum size if the reservation is not restored
// after page is free.  Therefore, force restore_reserve
// operation.
if hugepage_subpool_put_pages(spool, 1i64) == 0 {
restore_reserve = true;
}
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) }, flags);
folio_clear_hugetlb_migratable(folio);
unsafe { hugetlb_cgroup_uncharge_folio(hstate_index(h), pages_per_huge_page(h as *const hstate) as u64, folio) };
unsafe { hugetlb_cgroup_uncharge_folio_rsvd(hstate_index(h), pages_per_huge_page(h as *const hstate) as u64, folio) };
lruvec_stat_mod_folio(folio, 61u32, pages_per_huge_page(h as *const hstate).wrapping_neg() as i32);
mem_cgroup_uncharge(folio);
if restore_reserve {
let __v6: u64 = unsafe { (*h).resv_huge_pages };
unsafe {
(*h).resv_huge_pages = __v6.wrapping_add(1);
}
}
if folio_test_hugetlb_temporary(folio) {
remove_hugetlb_folio(h, folio, false);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) }, flags);
update_and_free_hugetlb_folio(h, folio, true);
} else {
if (unsafe { *((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize) }) != 0 {
// remove the page from active list
remove_hugetlb_folio(h, folio, true);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) }, flags);
update_and_free_hugetlb_folio(h, folio, true);
} else {
arch_clear_hugetlb_flags(folio);
enqueue_hugetlb_folio(h, folio);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) }, flags);
}
}
}

/// Must be called with the hugetlb lock held
fn account_new_hugetlb_folio(h: *mut hstate, folio: *mut folio) {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
let __v2: u64 = unsafe { (*h).nr_huge_pages };
unsafe {
(*h).nr_huge_pages = __v2.wrapping_add(1);
}
let __v4: *mut u32 = unsafe { (*h).nr_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(folio_nid(folio as *const folio) as isize);
let __v5: u32 = unsafe { *__v4 };
unsafe {
*__v4 = __v5.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn init_new_hugetlb_folio(folio: *mut folio) {
__folio_set_hugetlb(folio);
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) });
hugetlb_set_folio_subpool(folio, core::ptr::null_mut::<hugepage_subpool>());
set_hugetlb_cgroup(folio, core::ptr::null_mut::<hugetlb_cgroup>());
set_hugetlb_cgroup_rsvd(folio, core::ptr::null_mut::<hugetlb_cgroup>());
}

/// Find and lock address space (mapping) in write mode.
///
/// Upon entry, the folio is locked which means that folio_mapping() is
/// stable.  Due to locking order, we can only trylock_write.  If we can
/// not get the lock, simply return NULL to caller.
#[unsafe(no_mangle)]
extern "C" fn hugetlb_folio_mapping_lock_write(folio: *mut folio) -> *mut address_space {
let mapping: *mut address_space = unsafe { folio_mapping(folio as *const folio) };
if mapping.is_null() {
return mapping;
}
if i_mmap_trylock_write(mapping) != 0 {
return mapping;
}
core::ptr::null_mut::<address_space>()
}

fn alloc_buddy_frozen_folio(order: i32, mut gfp_mask: u32, nid: i32, nmask: *mut __SlateRecord_nodemask_types_h_17, node_alloc_noretry: *mut __SlateRecord_nodemask_types_h_17) -> *mut folio {
let mut alloc_try_hard: bool = true;
// By default we always try hard to allocate the folio with
// __GFP_RETRY_MAYFAIL flag.  However, if we are allocating folios in
// a loop (to adjust global huge page counts) and previous allocation
// failed, do not continue to try hard on the same node.  Use the
// node_alloc_noretry bitmap to manage this state information.
let __v0: bool = !node_alloc_noretry.is_null() && node_isset__2!(nid, *node_alloc_noretry) != 0;
if __v0 {
alloc_try_hard = false;
}
if alloc_try_hard {
let __v2: u32 = gfp_mask;
gfp_mask = __v2 | (BIT!(14i32) as u32);
}
let folio: *mut folio = alloc_hooks__3!(unsafe { __alloc_frozen_pages_noprof(gfp_mask, order as u32, nid, nmask, 0u32) }) as *mut folio;
// If we did not specify __GFP_RETRY_MAYFAIL, but still got a
// folio this indicates an overall state change.  Clear bit so
// that we resume normal 'try hard' allocations.
if !node_alloc_noretry.is_null() && !folio.is_null() && !alloc_try_hard {
node_clear!(nid, *node_alloc_noretry);
}
// If we tried hard to get a folio but failed, set bit so that
// subsequent attempts will not try as hard until there is an
// overall state change.
if !node_alloc_noretry.is_null() && folio.is_null() && alloc_try_hard {
node_set!(nid, *node_alloc_noretry);
}
if folio.is_null() {
__count_vm_event(52u32);
return core::ptr::null_mut::<folio>();
}
__count_vm_event(51u32);
folio
}

fn only_alloc_fresh_hugetlb_folio(h: *mut hstate, gfp_mask: u32, mut nid: i32, nmask: *mut __SlateRecord_nodemask_types_h_17, node_alloc_noretry: *mut __SlateRecord_nodemask_types_h_17) -> *mut folio {
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let order: i32 = huge_page_order(h) as i32;
if nid == 1i32.wrapping_neg() {
nid = numa_mem_id();
}
if order_is_gigantic(order as u32) {
folio = alloc_gigantic_frozen_folio(order, gfp_mask, nid, nmask);
} else {
folio = alloc_buddy_frozen_folio(order, gfp_mask, nid, nmask, node_alloc_noretry);
}
if !folio.is_null() {
init_new_hugetlb_folio(folio);
}
folio
}

/// Common helper to allocate a fresh hugetlb folio. All specific allocators
/// should use this function to get new hugetlb folio
///
/// Note that returned folio is 'frozen':  ref count of head page and all tail
/// pages is zero, and the accounting must be done in the caller.
fn alloc_fresh_hugetlb_folio(h: *mut hstate, gfp_mask: u32, nid: i32, nmask: *mut __SlateRecord_nodemask_types_h_17) -> *mut folio {
let folio: *mut folio = only_alloc_fresh_hugetlb_folio(h, gfp_mask, nid, nmask, core::ptr::null_mut::<__SlateRecord_nodemask_types_h_17>());
if !folio.is_null() {
unsafe { hugetlb_vmemmap_optimize_folio(h as *const hstate, folio) };
}
folio
}

#[unsafe(no_mangle)]
extern "C" fn prep_and_add_allocated_folios(h: *mut hstate, folio_list: *mut list_head) {
let mut flags: u64 = 0;
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let mut tmp_f: *mut folio = unsafe { core::mem::zeroed() };
// Send list for bulk vmemmap optimization processing
unsafe { hugetlb_vmemmap_optimize_folios(h, folio_list) };
// Add all new pool pages to free lists in one lock cycle
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) }, flags);
folio = list_first_entry!(folio_list);
tmp_f = list_next_entry!(folio, lru);
while list_entry_is_head__2!(folio, folio_list, lru) == 0 {
account_new_hugetlb_folio(h, folio);
enqueue_hugetlb_folio(h, folio);
folio = tmp_f;
tmp_f = list_next_entry!(tmp_f, lru);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) }, flags);
}

/// Allocates a fresh hugetlb page in a node interleaved manner.  The page
/// will later be added to the appropriate hugetlb pool.
fn alloc_pool_huge_folio(h: *mut hstate, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17, node_alloc_noretry: *mut __SlateRecord_nodemask_types_h_17, next_node: *mut i32) -> *mut folio {
let gfp_mask: u32 = htlb_alloc_mask(h) | (BIT!(21i32) as u32);
let mut nr_nodes: i32 = nodes_weight!(*nodes_allowed);
let mut node: i32 = 0;
loop {
let __v0: bool = if nr_nodes > 0 {
let __v1: i32 = hstate_next_node_to_alloc(next_node, nodes_allowed);
node = __v1;
__v1 != 0 || true
} else {
false
};
if !__v0 {
break;
}
let folio: *mut folio = only_alloc_fresh_hugetlb_folio(h, gfp_mask, node, nodes_allowed, node_alloc_noretry);
if !folio.is_null() {
return folio;
}
nr_nodes = nr_nodes.wrapping_sub(1);
}
core::ptr::null_mut::<folio>()
}

/// Remove huge page from pool from next node to free.  Attempt to keep
/// persistent huge pages more or less balanced over allowed nodes.
/// This routine only 'removes' the hugetlb page.  The caller must make
/// an additional call to free the page to low level allocators.
/// Called with hugetlb_lock locked.
fn remove_pool_hugetlb_folio(h: *mut hstate, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17, acct_surplus: bool) -> *mut folio {
let mut nr_nodes: i32 = 0;
let mut node: i32 = 0;
let mut folio: *mut folio = core::ptr::null_mut::<folio>();
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
nr_nodes = nodes_weight!(*nodes_allowed);
loop {
let __v4: bool = if nr_nodes > 0 {
let __v5: i32 = hstate_next_node_to_free(h, nodes_allowed);
node = __v5;
__v5 != 0 || true
} else {
false
};
if !__v4 {
break;
}
// If we're returning unused surplus pages, only examine
// nodes with surplus pages.
let __v1: bool = (!acct_surplus || (unsafe { *((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(node as isize) }) != 0) && list_empty(unsafe { (*h).hugepage_freelists.as_mut_ptr() as *mut list_head }.wrapping_offset(node as isize) as *const list_head) == 0;
if __v1 {
folio = list_entry__2!(unsafe { (*((*h).hugepage_freelists.as_mut_ptr() as *mut list_head).wrapping_offset(node as isize)).next }, folio);
remove_hugetlb_folio(h, folio, acct_surplus);
break;
}
nr_nodes = nr_nodes.wrapping_sub(1);
}
folio
}

/// Dissolve a given free hugetlb folio into free buddy pages. This function
/// does nothing for in-use hugetlb folios and non-hugetlb folios.
/// This function returns values like below:
///
///  -ENOMEM: failed to allocate vmemmap pages to free the freed hugepages
///           when the system is under memory pressure and the feature of
///           freeing unused vmemmap pages associated with each hugetlb page
///           is enabled.
///  -EBUSY:  failed to dissolved free hugepages or the hugepage is in-use
///           (allocated or reserved.)
///       0:  successfully dissolved free hugepages or the page is not a
///           hugepage (considered as already dissolved)
#[unsafe(no_mangle)]
extern "C" fn dissolve_free_hugetlb_folio(folio: *mut folio) -> i32 {
let mut adjust_surplus: bool = unsafe { core::mem::zeroed() };
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut rc: i32 = unsafe { core::mem::zeroed() };
unsafe {
rc = 16i32.wrapping_neg();
'__join_0: {
'__join_16: {
loop {
if !folio_test_hugetlb(folio as *const folio) {
return 0;
} else {
// Not to disrupt normal path by vainly holding hugetlb_lock
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
if !folio_test_hugetlb(folio as *const folio) {
break '__join_16;
} else {
if folio_ref_count(folio as *const folio) == 0 {
h = folio_hstate(folio);
adjust_surplus = false;
// remove_hugetlb_folio()/update_and_free_hugetlb_folio() bail
// for gigantic hstates without runtime support, so dissolving one
// here would leave it on the free list and, on vmemmap restore
// failure, the add_hugetlb_folio() rollback corrupts that list.
if hstate_is_gigantic_no_runtime(h) {
break '__join_0;
} else {
if available_huge_pages(h) == 0 {
break '__join_0;
} else {
// We should make sure that the page is already on the free list
// when it is dissolved.
if unlikely!(!folio_test_hugetlb_freed(folio)) != 0 {
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
cond_resched__81!();
// Theoretically, we should return -EBUSY when we
// encounter this race. In fact, we have a chance
// to successfully dissolve the page if we do a
// retry. Because the race window is quite small.
// If we seize this opportunity, it is an optimization
// for increasing the success rate of dissolving page.
} else {
break;
}
}
}
} else {
break '__join_0;
}
}
}
}
if *((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(folio_nid(folio as *const folio) as isize) != 0 {
adjust_surplus = true;
}
remove_hugetlb_folio(h, folio, adjust_surplus);
if !adjust_surplus {
let __v4: u64 = (*h).max_huge_pages;
(*h).max_huge_pages = __v4.wrapping_sub(1);
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
// Normally update_and_free_hugtlb_folio will allocate required vmemmmap
// before freeing the page.  update_and_free_hugtlb_folio will fail to
// free the page if it can not allocate required vmemmap.  We
// need to adjust max_huge_pages if the page is not freed.
// Attempt to allocate vmemmmap here so that we can take
// appropriate action on failure.
//
// The folio_test_hugetlb check here is because
// remove_hugetlb_folio will clear hugetlb folio flag for
// non-vmemmap optimized hugetlb folios.
if folio_test_hugetlb(folio as *const folio) {
rc = hugetlb_vmemmap_restore_folio(h as *const hstate, folio);
if rc != 0 {
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
add_hugetlb_folio(h, folio, adjust_surplus);
if !adjust_surplus {
let __v1: u64 = (*h).max_huge_pages;
(*h).max_huge_pages = __v1.wrapping_add(1);
break '__join_0;
} else {
break '__join_0;
}
}
} else {
rc = 0;
}
update_and_free_hugetlb_folio(h, folio, false);
return rc;
}
rc = 0;
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
return rc;
}
}

// Dissolve free hugepages in a given pfn range. Used by memory hotplug to
// make specified memory blocks removable from the system.
// Note that this will dissolve a free gigantic hugepage completely, if any
// part of it lies within the given range.
// Also note that if dissolve_free_hugetlb_folio() returns with an error, all
// free hugetlb folios that were dissolved before that error are lost.
/// Allocates a fresh surplus page from the page allocator.
fn alloc_surplus_hugetlb_folio(h: *mut hstate, gfp_mask: u32, nid: i32, nmask: *mut __SlateRecord_nodemask_types_h_17) -> *mut folio {
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
unsafe {
folio = core::ptr::null_mut::<folio>();
if hstate_is_gigantic_no_runtime(h) {
return core::ptr::null_mut::<folio>();
} else {
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
if !((*h).surplus_huge_pages >= (*h).nr_overcommit_huge_pages) {
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
folio = alloc_fresh_hugetlb_folio(h, gfp_mask, nid, nmask);
if folio.is_null() {
return core::ptr::null_mut::<folio>();
} else {
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
// nr_huge_pages needs to be adjusted within the same lock cycle
// as surplus_pages, otherwise it might confuse
// persistent_huge_pages() momentarily.
account_new_hugetlb_folio(h, folio);
// We could have raced with the pool size change.
// Double check that and simply deallocate the new page
// if we would end up overcommiting the surpluses. Abuse
// temporary page to workaround the nasty free_huge_folio
// codeflow
if (*h).surplus_huge_pages >= (*h).nr_overcommit_huge_pages {
folio_set_hugetlb_temporary(folio);
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
free_huge_folio(folio);
return core::ptr::null_mut::<folio>();
} else {
let __v4: u64 = (*h).surplus_huge_pages;
(*h).surplus_huge_pages = __v4.wrapping_add(1);
let __v2: *mut u32 = ((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(folio_nid(folio as *const folio) as isize);
let __v1: u32 = *__v2;
*__v2 = __v1.wrapping_add(1);
}
}
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
return folio;
}
}
}

fn alloc_migrate_hugetlb_folio(h: *mut hstate, gfp_mask: u32, nid: i32, nmask: *mut __SlateRecord_nodemask_types_h_17) -> *mut folio {
if hstate_is_gigantic(h) {
return core::ptr::null_mut::<folio>();
}
let folio: *mut folio = alloc_fresh_hugetlb_folio(h, gfp_mask, nid, nmask);
if folio.is_null() {
return core::ptr::null_mut::<folio>();
}
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
account_new_hugetlb_folio(h, folio);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
// fresh huge pages are frozen
folio_ref_unfreeze(folio, 1i32);
// We do not account these pages as surplus because they are only
// temporary and will be released properly on the last reference
folio_set_hugetlb_temporary(folio);
folio
}

fn alloc_buddy_hugetlb_folio(h: *mut hstate, gfp_mask: u32, mpoli: *mut mempolicy_interpreted) -> *mut folio {
let mut folio: *mut folio = core::ptr::null_mut::<folio>();
let mut nodemask: *mut __SlateRecord_nodemask_types_h_17 = unsafe { (*mpoli).nodemask };
if ((unsafe { (*mpoli).mode }) as u32) == 5 {
let gfp: u32 = gfp_mask & !((BIT!(10i32) as u32) | (BIT!(15i32) as u32));
folio = alloc_surplus_hugetlb_folio(h, gfp, unsafe { (*mpoli).nid }, nodemask);
// Fallback to all nodes if page==NULL
nodemask = core::ptr::null_mut::<__SlateRecord_nodemask_types_h_17>();
}
if folio.is_null() {
folio = alloc_surplus_hugetlb_folio(h, gfp_mask, unsafe { (*mpoli).nid }, nodemask);
}
folio
}

#[unsafe(no_mangle)]
extern "C" fn alloc_hugetlb_folio_reserve(h: *mut hstate, preferred_nid: i32, nmask: *mut __SlateRecord_nodemask_types_h_17, gfp_mask: u32) -> *mut folio {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
if (unsafe { (*h).resv_huge_pages }) == 0 {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
return core::ptr::null_mut::<folio>();
}
let folio: *mut folio = dequeue_hugetlb_folio_nodemask(h, gfp_mask, preferred_nid, nmask);
if !folio.is_null() {
let __v1: u64 = unsafe { (*h).resv_huge_pages };
unsafe {
(*h).resv_huge_pages = __v1.wrapping_sub(1);
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
folio
}

/// folio migration callback function
#[unsafe(no_mangle)]
extern "C" fn alloc_hugetlb_folio_nodemask(h: *mut hstate, preferred_nid: i32, nmask: *mut __SlateRecord_nodemask_types_h_17, mut gfp_mask: u32, allow_alloc_fallback: bool) -> *mut folio {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
if available_huge_pages(h) != 0 {
let folio: *mut folio = dequeue_hugetlb_folio_nodemask(h, gfp_mask, preferred_nid, nmask);
if !folio.is_null() {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
return folio;
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
// We cannot fallback to other nodes, as we could break the per-node pool.
if !allow_alloc_fallback {
let __v0: u32 = gfp_mask;
gfp_mask = __v0 | (BIT!(21i32) as u32);
}
alloc_migrate_hugetlb_folio(h, gfp_mask, preferred_nid, nmask)
}

fn policy_mbind_nodemask(gfp: u32) -> *mut __SlateRecord_nodemask_types_h_17 {
let mpol: *mut mempolicy = unsafe { get_task_policy(get_current()) };
// Only enforce MPOL_BIND policy which overlaps with cpuset policy
// (from policy_nodemask) specifically for hugetlb case
let __v0: bool = ((unsafe { (*mpol).mode }) as i32) == 2 && ((unsafe { apply_policy_zone(mpol, gfp_zone(gfp)) }) && (unsafe { cpuset_nodemask_valid_mems_allowed(core::ptr::addr_of_mut!((*mpol).nodes) as *const __SlateRecord_nodemask_types_h_17) }) != 0);
if __v0 {
return unsafe { core::ptr::addr_of_mut!((*mpol).nodes) };
}
core::ptr::null_mut::<__SlateRecord_nodemask_types_h_17>()
}

/// Increase the hugetlb pool such that it can accommodate a reservation
/// of size 'delta'.
fn gather_surplus_pages(h: *mut hstate, delta: i64) -> i32 {
let mut alloc_nodemask: core::mem::MaybeUninit<__SlateRecord_nodemask_types_h_17> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut __SlateRecord_nodemask_types_h_17 = core::ptr::addr_of_mut!(alloc_nodemask) as *mut __SlateRecord_nodemask_types_h_17;
let mut mbind_nodemask: core::mem::MaybeUninit<*mut __SlateRecord_nodemask_types_h_17> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut __SlateRecord_nodemask_types_h_17 = core::ptr::addr_of_mut!(mbind_nodemask) as *mut *mut __SlateRecord_nodemask_types_h_17;
let mut alloc_ok: bool = unsafe { core::mem::zeroed() };
let mut allocated: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut i64 = core::ptr::addr_of_mut!(allocated) as *mut i64;
let mut needed: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i64 = core::ptr::addr_of_mut!(needed) as *mut i64;
let mut i: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut i64 = core::ptr::addr_of_mut!(i) as *mut i64;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut tmp: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut folio = core::ptr::addr_of_mut!(tmp) as *mut *mut folio;
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut surplus_list: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut list_head = core::ptr::addr_of_mut!(surplus_list) as *mut list_head;
unsafe {
core::ptr::write(__slate_slot_27, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_27), prev: core::ptr::addr_of_mut!(*__slate_slot_27) });
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
alloc_ok = true;
let __slate_auto_bytes_6 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_19 = policy_mbind_nodemask(htlb_alloc_mask(h));
if !(*__slate_slot_19).is_null() {
nodes_and__5!(*__slate_slot_18, *(*__slate_slot_19), (*get_current()).mems_allowed);
} else {
*__slate_slot_18 = (*get_current()).mems_allowed;
}
lockdep_assert_held!(core::ptr::addr_of_mut!(hugetlb_lock.0));
*__slate_slot_22 = unsafe { (*h).resv_huge_pages }.wrapping_add(delta as u64).wrapping_sub((*h).free_huge_pages) as i64;
if *__slate_slot_22 <= 0 {
let __v16: u64 = (*h).resv_huge_pages;
(*h).resv_huge_pages = __v16.wrapping_add(delta as u64);
return 0;
} else {
*__slate_slot_21 = 0;
*__slate_slot_24 = 12i32.wrapping_neg();
'__join_3: {
'__join_8: {
loop {
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
*__slate_slot_23 = 0;
'__join_10: {
loop {
if *__slate_slot_23 < *__slate_slot_22 {
*__slate_slot_26 = core::ptr::null_mut::<folio>();
// It is okay to use NUMA_NO_NODE because we use numa_mem_id()
// down the road to pick the current node if that is the case.
*__slate_slot_26 = alloc_surplus_hugetlb_folio(h, htlb_alloc_mask(h), 1i32.wrapping_neg(), core::ptr::addr_of_mut!(*__slate_slot_18));
if (*__slate_slot_26).is_null() {
break;
} else {
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_26)).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru), core::ptr::addr_of_mut!(*__slate_slot_27));
cond_resched__31!();
let __v14: i64 = *__slate_slot_23;
*__slate_slot_23 = __v14.wrapping_add(1);
}
} else {
break '__join_10;
}
}
alloc_ok = false;
}
let __v12: i64 = *__slate_slot_21;
*__slate_slot_21 = __v12.wrapping_add(*__slate_slot_23);
// After retaking hugetlb_lock, we need to recalculate 'needed'
// because either resv_huge_pages or free_huge_pages may have changed.
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
*__slate_slot_22 = unsafe { (*h).resv_huge_pages }.wrapping_add(delta as u64).wrapping_sub(unsafe { (*h).free_huge_pages }.wrapping_add(*__slate_slot_21 as u64)) as i64;
if *__slate_slot_22 > 0 {
if !alloc_ok {
break '__join_8;
}
} else {
break;
}
}
// The surplus_list now contains _at_least_ the number of extra pages
// needed to accommodate the reservation.  Add the appropriate number
// of pages to the hugetlb pool and free the extras back to the buddy
// allocator.  Commit the entire reservation here to prevent another
// process from stealing the pages as they are added to the pool but
// before they are reserved.
let __v10: i64 = *__slate_slot_22;
*__slate_slot_22 = __v10.wrapping_add(*__slate_slot_21);
let __v7: u64 = (*h).resv_huge_pages;
(*h).resv_huge_pages = __v7.wrapping_add(delta as u64);
*__slate_slot_24 = 0;
// Free the needed pages to the hugetlb pool
*__slate_slot_26 = list_first_entry!(core::ptr::addr_of_mut!(*__slate_slot_27));
let __v5: *mut folio = list_next_entry!(*__slate_slot_26, lru);
*__slate_slot_25 = __v5;
loop {
if list_entry_is_head__2!(*__slate_slot_26, core::ptr::addr_of_mut!(*__slate_slot_27), lru) == 0 {
let __v4: i64 = *__slate_slot_22;
let __v3: i64 = __v4.wrapping_sub(1);
*__slate_slot_22 = __v3;
if __v3 < 0 {
break '__join_3;
} else {
// Add the page to the hugetlb allocator
enqueue_hugetlb_folio(h, *__slate_slot_26);
*__slate_slot_26 = *__slate_slot_25;
let __v2: *mut folio = list_next_entry!(*__slate_slot_25, lru);
*__slate_slot_25 = __v2;
}
} else {
break '__join_3;
}
}
}
// We were not able to allocate enough pages to
// satisfy the entire reservation so we free what
// we've allocated so far.
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
// Free unnecessary surplus pages to the buddy allocator.
// Pages have no ref count, call free_huge_folio directly.
*__slate_slot_26 = list_first_entry!(core::ptr::addr_of_mut!(*__slate_slot_27));
let __v1: *mut folio = list_next_entry!(*__slate_slot_26, lru);
*__slate_slot_25 = __v1;
while list_entry_is_head__2!(*__slate_slot_26, core::ptr::addr_of_mut!(*__slate_slot_27), lru) == 0 {
free_huge_folio(*__slate_slot_26);
*__slate_slot_26 = *__slate_slot_25;
let __v0: *mut folio = list_next_entry!(*__slate_slot_25, lru);
*__slate_slot_25 = __v0;
}
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
return *__slate_slot_24;
}
}
}

// This routine has two main purposes:
// 1) Decrement the reservation count (resv_huge_pages) by the value passed
//    in unused_resv_pages.  This corresponds to the prior adjustments made
//    to the associated reservation map.
// 2) Free any unused surplus pages that may have been allocated to satisfy
//    the reservation.  As many as unused_resv_pages may be freed.
fn return_unused_surplus_pages(h: *mut hstate, unused_resv_pages: u64) {
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut __UNIQUE_ID_y__708: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__707: u64 = unsafe { core::mem::zeroed() };
let mut page_list: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut list_head = core::ptr::addr_of_mut!(page_list) as *mut list_head;
let mut nr_pages: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(nr_pages) as *mut u64;
unsafe {
'__join_0: {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_9, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_9), prev: core::ptr::addr_of_mut!(*__slate_slot_9) });
lockdep_assert_held!(core::ptr::addr_of_mut!(hugetlb_lock.0));
// Uncommit the reservation
let __v7: u64 = (*h).resv_huge_pages;
(*h).resv_huge_pages = __v7.wrapping_sub(unused_resv_pages);
if hstate_is_gigantic_no_runtime(h) {
break '__join_0;
} else {
__UNIQUE_ID_x__707 = unused_resv_pages;
__UNIQUE_ID_y__708 = (*h).surplus_huge_pages;
while false {
}
// Part (or even all) of the reservation could have been backed
// by pre-allocated pages. Only free surplus pages.
*__slate_slot_10 = if __UNIQUE_ID_x__707 < __UNIQUE_ID_y__708 { __UNIQUE_ID_x__707 } else { __UNIQUE_ID_y__708 };
// We want to release as many surplus pages as possible, spread
// evenly across all nodes with memory. Iterate across these nodes
// until we can no longer free unreserved surplus pages. This occurs
// when the nodes with surplus pages have no free pages.
// remove_pool_hugetlb_folio() will balance the freed pages across the
// on-line nodes with memory and will handle the hstate accounting.
loop {
let __v1: u64 = *__slate_slot_10;
*__slate_slot_10 = __v1.wrapping_sub(1);
if __v1 != 0 {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = remove_pool_hugetlb_folio(h, (core::ptr::addr_of_mut!(node_states) as *mut __SlateRecord_nodemask_types_h_17).wrapping_offset(3 as isize), true);
if (*__slate_slot_2).is_null() {
break '__join_0;
} else {
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_2)).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru), core::ptr::addr_of_mut!(*__slate_slot_9));
}
} else {
break;
}
}
}
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
update_and_free_pages_bulk(h, core::ptr::addr_of_mut!(*__slate_slot_9));
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
}
}

// vma_needs_reservation, vma_commit_reservation and vma_end_reservation
// are used by the huge page allocation routines to manage reservations.
//
// vma_needs_reservation is called to determine if the huge page at addr
// within the vma has an associated reservation.  If a reservation is
// needed, the value 1 is returned.  The caller is then responsible for
// managing the global reservation and subpool usage counts.  After
// the huge page has been allocated, vma_commit_reservation is called
// to add the page to the reservation map.  If the page allocation fails,
// the reservation must be ended instead of committed.  vma_end_reservation
// is called in such cases.
//
// In the normal case, vma_commit_reservation returns the same value
// as the preceding vma_needs_reservation call.  The only time this
// is not the case is if a reserve map was changed between calls.  It
// is the responsibility of the caller to notice the difference and
// take appropriate action.
//
// vma_add_reservation is used in error paths where a reservation must
// be restored when a newly allocated huge page must be freed.  It is
// to be called after calling vma_needs_reservation to determine if a
// reservation exists.
//
// vma_del_reservation is used in error paths where an entry in the reserve
// map was created during huge page allocation and must be removed.  It is to
// be called after calling vma_needs_reservation to determine if a reservation
// exists.
fn __vma_reservation_common(h: *mut hstate, vma: *mut vm_area_struct, addr: u64, mode: u32) -> i64 {
let mut idx: u64 = 0;
let mut ret: i64 = 0;
let mut dummy_out_regions_needed: i64 = 0;
let resv: *mut resv_map = vma_resv_map(vma);
if resv.is_null() {
return 1;
}
idx = vma_hugecache_offset(h, vma, addr);
match mode as u32 {
0 => {
ret = region_chg(resv, idx as i64, idx.wrapping_add(1) as i64, core::ptr::addr_of_mut!(dummy_out_regions_needed));
// We assume that vma_reservation_* routines always operate on
// 1 page, and that adding to resv map a 1 page entry can only
// ever require 1 region.
VM_BUG_ON!();
}
1 => {
ret = region_add(resv, idx as i64, idx.wrapping_add(1) as i64, 1i64, core::ptr::null_mut::<hstate>(), core::ptr::null_mut::<hugetlb_cgroup>());
// region_add calls of range 1 should never fail.
VM_BUG_ON!();
}
2 => {
region_abort(resv, idx as i64, idx.wrapping_add(1) as i64, 1i64);
ret = 0;
}
3 => {
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) != 0 {
ret = region_add(resv, idx as i64, idx.wrapping_add(1) as i64, 1i64, core::ptr::null_mut::<hstate>(), core::ptr::null_mut::<hugetlb_cgroup>());
// region_add calls of range 1 should never fail.
VM_BUG_ON!();
} else {
region_abort(resv, idx as i64, idx.wrapping_add(1) as i64, 1i64);
ret = region_del(resv, idx as i64, idx.wrapping_add(1) as i64);
}
}
4 => {
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) != 0 {
region_abort(resv, idx as i64, idx.wrapping_add(1) as i64, 1i64);
ret = region_del(resv, idx as i64, idx.wrapping_add(1) as i64);
} else {
ret = region_add(resv, idx as i64, idx.wrapping_add(1) as i64, 1i64, core::ptr::null_mut::<hstate>(), core::ptr::null_mut::<hugetlb_cgroup>());
// region_add calls of range 1 should never fail.
VM_BUG_ON!();
}
}
_ => {
loop {
unsafe {
core::arch::asm!("710: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 710b - ., 3; .popsection\n# {0}", const 710 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___vma_reservation_common_1, sym __slate_asm_string___vma_reservation_common_2, const 2480 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) != 0 || (mode as u32) == 4 {
return ret;
}
// We know private mapping must have HPAGE_RESV_OWNER set.
//
// In most cases, reserves always exist for private mappings.
// However, a file associated with mapping could have been
// hole punched or truncated after reserves were consumed.
// As subsequent fault on such a range will not use reserves.
// Subtle - The reserve map for private mappings has the
// opposite meaning than that of shared mappings.  If NO
// entry is in the reserve map, it means a reservation exists.
// If an entry exists in the reserve map, it means the
// reservation has already been consumed.  As a result, the
// return value of this routine is the opposite of the
// value returned from reserve map manipulation routines above.
if ret > 0 {
return 0;
}
if ret == 0 {
return 1;
}
ret
}

fn vma_needs_reservation(h: *mut hstate, vma: *mut vm_area_struct, addr: u64) -> i64 {
__vma_reservation_common(h, vma, addr, 0u32)
}

fn vma_commit_reservation(h: *mut hstate, vma: *mut vm_area_struct, addr: u64) -> i64 {
__vma_reservation_common(h, vma, addr, 1u32)
}

fn vma_end_reservation(h: *mut hstate, vma: *mut vm_area_struct, addr: u64) {
__vma_reservation_common(h, vma, addr, 2u32);
}

fn vma_add_reservation(h: *mut hstate, vma: *mut vm_area_struct, addr: u64) -> i64 {
__vma_reservation_common(h, vma, addr, 3u32)
}

fn vma_del_reservation(h: *mut hstate, vma: *mut vm_area_struct, addr: u64) -> i64 {
__vma_reservation_common(h, vma, addr, 4u32)
}

/// This routine is called to restore reservation information on error paths.
/// It should ONLY be called for folios allocated via alloc_hugetlb_folio(),
/// and the hugetlb mutex should remain held when calling this routine.
///
/// It handles two specific cases:
/// 1) A reservation was in place and the folio consumed the reservation.
///    hugetlb_restore_reserve is set in the folio.
/// 2) No reservation was in place for the page, so hugetlb_restore_reserve is
///    not set.  However, alloc_hugetlb_folio always updates the reserve map.
///
/// In case 1, free_huge_folio later in the error path will increment the
/// global reserve count.  But, free_huge_folio does not have enough context
/// to adjust the reservation map.  This case deals primarily with private
/// mappings.  Adjust the reserve map here to be consistent with global
/// reserve count adjustments to be made by free_huge_folio.  Make sure the
/// reserve map indicates there is a reservation present.
///
/// In case 2, simply undo reserve map modifications done by alloc_hugetlb_folio.
#[unsafe(no_mangle)]
extern "C" fn restore_reserve_on_error(h: *mut hstate, vma: *mut vm_area_struct, address: u64, folio: *mut folio) {
let mut rc: i64 = vma_needs_reservation(h, vma, address);
if folio_test_hugetlb_restore_reserve(folio) {
if unlikely!(rc < 0) != 0 {
folio_clear_hugetlb_restore_reserve(folio);
} else {
if rc != 0 {
vma_add_reservation(h, vma, address);
} else {
vma_end_reservation(h, vma, address);
}
}
// Rare out of memory condition in reserve map
// manipulation.  Clear hugetlb_restore_reserve so
// that global reserve count will not be incremented
// by free_huge_folio.  This will make it appear
// as though the reservation for this folio was
// consumed.  This may prevent the task from
// faulting in the folio at a later time.  This
// is better than inconsistent global huge page
// accounting of reserve counts.
} else {
if rc == 0 {
// This indicates there is an entry in the reserve map
// not added by alloc_hugetlb_folio.  We know it was added
// before the alloc_hugetlb_folio call, otherwise
// hugetlb_restore_reserve would be set on the folio.
// Remove the entry so that a subsequent allocation
// does not consume a reservation.
rc = vma_del_reservation(h, vma, address);
if rc < 0 {
folio_set_hugetlb_restore_reserve(folio);
}
// VERY rare out of memory condition.  Since
// we can not delete the entry, set
// hugetlb_restore_reserve so that the reserve
// count will be incremented when the folio
// is freed.  This reserve will be consumed
// on a subsequent allocation.
} else {
if rc < 0 {
// Rare out of memory condition from
// vma_needs_reservation call.  Memory allocation is
// only attempted if a new entry is needed.  Therefore,
// this implies there is not an entry in the
// reserve map.
//
// For shared mappings, no entry in the map indicates
// no reservation.  We are done.
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) == 0 {
folio_set_hugetlb_restore_reserve(folio);
}
// For private mappings, no entry indicates
// a reservation is present.  Since we can
// not add an entry, set hugetlb_restore_reserve
// on the folio so reserve count will be
// incremented when freed.  This reserve will
// be consumed on a subsequent allocation.
} else {
// No reservation present, do nothing
vma_end_reservation(h, vma, address);
}
}
}
}

/// alloc_and_dissolve_hugetlb_folio - Allocate a new folio and dissolve
/// the old one
/// @old_folio: Old folio to dissolve
/// @list: List to isolate the page in case we need to
/// Returns 0 on success, otherwise negated error.
fn alloc_and_dissolve_hugetlb_folio(old_folio: *mut folio, list: *mut list_head) -> i32 {
let mut isolated: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut bool = core::ptr::addr_of_mut!(isolated) as *mut bool;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut new_folio: *mut folio = unsafe { core::mem::zeroed() };
let mut nid: i32 = unsafe { core::mem::zeroed() };
let mut h: core::mem::MaybeUninit<*mut hstate> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut hstate = core::ptr::addr_of_mut!(h) as *mut *mut hstate;
let mut gfp_mask: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(gfp_mask) as *mut u32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
nid = folio_nid(old_folio as *const folio);
new_folio = core::ptr::null_mut::<folio>();
ret = 0;
'__join_2: {
'__join_3: {
'__join_4: {
loop {
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
// The old_folio might have been dissolved from under our feet, so make sure
// to carefully check the state under the lock.
if !folio_test_hugetlb(old_folio as *const folio) {
break '__join_3;
} else {
if folio_ref_count(old_folio as *const folio) != 0 {
break '__join_4;
} else {
if !folio_test_hugetlb_freed(old_folio) {
// Folio's refcount is 0 but it has not been enqueued in the
// freelist yet. Race window is small, so we can succeed here if
// we retry.
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
cond_resched__46!();
} else {
*__slate_slot_4 = folio_hstate(old_folio);
if new_folio.is_null() {
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
*__slate_slot_5 = htlb_alloc_mask(*__slate_slot_4) | (BIT!(21i32) as u32);
new_folio = alloc_fresh_hugetlb_folio(*__slate_slot_4, *__slate_slot_5, nid, core::ptr::null_mut::<__SlateRecord_nodemask_types_h_17>());
if new_folio.is_null() {
return 12i32.wrapping_neg();
}
} else {
break;
}
}
}
}
}
// Ok, old_folio is still a genuine free hugepage. Remove it from
// the freelist and decrease the counters. These will be
// incremented again when calling account_new_hugetlb_folio()
// and enqueue_hugetlb_folio() for new_folio. The counters will
// remain stable since this happens under the lock.
remove_hugetlb_folio(*__slate_slot_4, old_folio, false);
// Ref count on new_folio is already zero as it was dropped
// earlier.  It can be directly added to the pool free list.
account_new_hugetlb_folio(*__slate_slot_4, new_folio);
enqueue_hugetlb_folio(*__slate_slot_4, new_folio);
// Folio has been replaced, we can safely free the old one.
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
update_and_free_hugetlb_folio(*__slate_slot_4, old_folio, false);
return ret;
}
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
// Someone has grabbed the folio, try to isolate it here.
// Fail with -EBUSY if not possible.
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
*__slate_slot_0 = folio_isolate_hugetlb(old_folio, list);
ret = if *__slate_slot_0 { 0 } else { 16i32.wrapping_neg() };
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
break '__join_2;
}
// Freed from under us. Drop new_folio too.
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
if !new_folio.is_null() {
update_and_free_hugetlb_folio(*__slate_slot_4, new_folio, false);
}
return ret;
}
}

#[unsafe(no_mangle)]
extern "C" fn isolate_or_dissolve_huge_folio(folio: *mut folio, list: *mut list_head) -> i32 {
let mut ret: i32 = 16i32.wrapping_neg();
// Not to disrupt normal path by vainly holding hugetlb_lock
if !folio_test_hugetlb(folio as *const folio) {
return 0;
}
// Fence off gigantic pages as there is a cyclic dependency between
// alloc_contig_range and them. Return -ENOMEM as this has the effect
// of bailing out right away without further retrying.
if order_is_gigantic(folio_order(folio as *const folio)) {
return 12i32.wrapping_neg();
}
let __v0: bool = folio_ref_count(folio as *const folio) != 0 && folio_isolate_hugetlb(folio, list);
if __v0 {
ret = 0;
} else {
if folio_ref_count(folio as *const folio) == 0 {
ret = alloc_and_dissolve_hugetlb_folio(folio, list);
}
}
ret
}

// replace_free_hugepage_folios - Replace free hugepage folios in a given pfn
// range with new folios.
// @start_pfn: start pfn of the given pfn range
// @end_pfn: end pfn of the given pfn range
// Returns 0 on success, otherwise negated error.
#[unsafe(no_mangle)]
extern "C" fn wait_for_freed_hugetlb_folios() {
if llist_empty((unsafe { core::ptr::addr_of_mut!(hpage_freelist) }) as *const llist_head) {
return;
}
unsafe { flush_work(core::ptr::addr_of_mut!(free_hpage_work)) };
}

/// hugetlb_alloc_folio - Allocate a hugetlb folio.
/// @h: Hugetlb state control block.
/// @mpoli: Interpreted memory policy to use for allocation.
/// @alloc_flags: Flags controlling the allocation behavior.
///
/// Allocates a hugetlb folio and handles cgroup charging and global hstate
/// reservations.
///
/// Return: A pointer to the allocated folio, or an ERR_PTR on failure.
///         -ENOSPC if cgroup charging fails or no folio is available.
///         -ENOMEM if mem cgroup charging fails.
#[unsafe(no_mangle)]
extern "C" fn hugetlb_alloc_folio(h: *mut hstate, mpoli: *mut mempolicy_interpreted, alloc_flags: u8) -> *mut folio {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut idx: i32 = unsafe { core::mem::zeroed() };
let mut gfp: u32 = unsafe { core::mem::zeroed() };
let mut h_cg: *mut hugetlb_cgroup = unsafe { core::mem::zeroed() };
let mut h_cg_rsvd: *mut hugetlb_cgroup = unsafe { core::mem::zeroed() };
let mut nr_pages: u64 = unsafe { core::mem::zeroed() };
let mut use_global_reservation: bool = unsafe { core::mem::zeroed() };
let mut charge_hugetlb_cgroup_rsvd: bool = unsafe { core::mem::zeroed() };
unsafe {
charge_hugetlb_cgroup_rsvd = ((alloc_flags as i32) as u64) & BIT!(0i32) != 0;
use_global_reservation = ((alloc_flags as i32) as u64) & BIT!(1i32) != 0;
nr_pages = pages_per_huge_page(h as *const hstate) as u64;
h_cg_rsvd = core::ptr::null_mut::<hugetlb_cgroup>();
h_cg = core::ptr::null_mut::<hugetlb_cgroup>();
gfp = htlb_alloc_mask(h);
idx = hstate_index(h);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if charge_hugetlb_cgroup_rsvd {
__v4 = hugetlb_cgroup_charge_cgroup_rsvd(idx, nr_pages, core::ptr::addr_of_mut!(h_cg_rsvd)) != 0;
} else {
__v4 = false;
}
if __v4 {
return ERR_PTR(28i32.wrapping_neg() as i64) as *mut folio;
} else {
'__join_2: {
if hugetlb_cgroup_charge_cgroup(idx, nr_pages, core::ptr::addr_of_mut!(h_cg)) != 0 {
*__slate_slot_5 = 28i32.wrapping_neg();
} else {
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
*__slate_slot_6 = core::ptr::null_mut::<folio>();
if use_global_reservation {
__v3 = true;
} else {
__v3 = available_huge_pages(h) != 0;
}
if __v3 {
*__slate_slot_6 = dequeue_hugetlb_folio(h, gfp, mpoli);
}
if (*__slate_slot_6).is_null() {
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
*__slate_slot_6 = alloc_buddy_hugetlb_folio(h, gfp, mpoli);
if (*__slate_slot_6).is_null() {
*__slate_slot_5 = 28i32.wrapping_neg();
hugetlb_cgroup_uncharge_cgroup(idx, nr_pages, h_cg);
break '__join_2;
} else {
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_6)).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru), core::ptr::addr_of_mut!((*h).hugepage_activelist));
folio_ref_unfreeze(*__slate_slot_6, 1i32);
}
}
if use_global_reservation {
folio_set_hugetlb_restore_reserve(*__slate_slot_6);
let __v1: u64 = (*h).resv_huge_pages;
(*h).resv_huge_pages = __v1.wrapping_sub(1);
}
hugetlb_cgroup_commit_charge(idx, nr_pages, h_cg, *__slate_slot_6);
if charge_hugetlb_cgroup_rsvd {
hugetlb_cgroup_commit_charge_rsvd(idx, nr_pages, h_cg_rsvd, *__slate_slot_6);
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
*__slate_slot_5 = mem_cgroup_charge_hugetlb(*__slate_slot_6, gfp | (BIT!(14i32) as u32));
// Unconditionally increment NR_HUGETLB here because if
// mem_cgroup_charge_hugetlb failed, freeing the page will
// decrement NR_HUGETLB.
lruvec_stat_mod_folio(*__slate_slot_6, 61u32, nr_pages as i32);
if *__slate_slot_5 == 12i32.wrapping_neg() {
free_huge_folio(*__slate_slot_6);
// Skip uncharging hugetlb_cgroup since the charges
// were committed to the folio and freeing the folio
// would have cleared those up.
return ERR_PTR(*__slate_slot_5 as i64) as *mut folio;
} else {
return *__slate_slot_6;
}
}
}
if charge_hugetlb_cgroup_rsvd {
hugetlb_cgroup_uncharge_cgroup_rsvd(idx, nr_pages, h_cg_rsvd);
}
return ERR_PTR(*__slate_slot_5 as i64) as *mut folio;
}
}
}

/// NOTE! "cow_from_owner" represents a very hacky usage only used in CoW
/// faults of hugetlb private mappings on top of a non-page-cache folio (in
/// which case even if there's a private vma resv map it won't cover such
/// allocation).  New call sites should (probably) never set it to true!!
/// When it's set, the allocation will bypass all vma level reservations.
#[unsafe(no_mangle)]
extern "C" fn alloc_hugetlb_folio(vma: *mut vm_area_struct, addr: u64, cow_from_owner: bool) -> *mut folio {
let mut __slate_compound_19: mempolicy_interpreted = unsafe { core::mem::zeroed() };
let mut rsv_adjust: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i64 = core::ptr::addr_of_mut!(rsv_adjust) as *mut i64;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut nid: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(nid) as *mut i32;
let mut alloc_flags: u8 = unsafe { core::mem::zeroed() };
let mut nodemask: core::mem::MaybeUninit<*mut __SlateRecord_nodemask_types_h_17> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut __SlateRecord_nodemask_types_h_17 = core::ptr::addr_of_mut!(nodemask) as *mut *mut __SlateRecord_nodemask_types_h_17;
let mut mpol: core::mem::MaybeUninit<*mut mempolicy> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut mempolicy = core::ptr::addr_of_mut!(mpol) as *mut *mut mempolicy;
let mut gfp: u32 = unsafe { core::mem::zeroed() };
let mut mpoli: core::mem::MaybeUninit<mempolicy_interpreted> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut mempolicy_interpreted = core::ptr::addr_of_mut!(mpoli) as *mut mempolicy_interpreted;
let mut map_chg: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(map_chg) as *mut u32;
let mut gbl_reserve: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i64 = core::ptr::addr_of_mut!(gbl_reserve) as *mut i64;
let mut gbl_chg: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i64 = core::ptr::addr_of_mut!(gbl_chg) as *mut i64;
let mut retval: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i64 = core::ptr::addr_of_mut!(retval) as *mut i64;
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut spool: *mut hugepage_subpool = unsafe { core::mem::zeroed() };
unsafe {
{
spool = subpool_vma(vma);
h = hstate_vma(vma);
let __slate_auto_bytes_1 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 24usize);
gfp = htlb_alloc_mask(h);
let __slate_auto_bytes_7 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
alloc_flags = 0;
let __slate_auto_bytes_9 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Whether we need a separate per-vma reservation?
if cow_from_owner {
// Special case!  Since it's a CoW on top of a reserved
// page, the private resv map doesn't count.  So it cannot
// consume the per-vma resv map even if it's reserved.
*__slate_slot_12 = 2;
} else {
// Examine the region/reserve map to determine if the process
// has a reservation for the page to be allocated.  A return
// code of zero indicates a reservation exists (no change).
*__slate_slot_15 = vma_needs_reservation(h, vma, addr);
if *__slate_slot_15 < 0 {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut folio;
} else {
*__slate_slot_12 = (if *__slate_slot_15 != 0 { 1i32 } else { 0i32 }) as u32;
}
}
}
'__join_2: {
// Whether we need a separate global reservation?
//
// Processes that did not create the mapping will have no
// reserves as indicated by the region/reserve map. Check
// that the allocation will not exceed the subpool limit.
// Or if it can get one from the pool reservation directly.
if (*__slate_slot_12 as u32) != 0 {
*__slate_slot_14 = hugepage_subpool_get_pages(spool, 1i64);
if *__slate_slot_14 < 0 {
*__slate_slot_5 = 28i32.wrapping_neg();
break '__join_2;
}
} else {
// If we have the vma reservation ready, no need for extra
// global reservation.
*__slate_slot_14 = 0;
}
// If allocation doesn't reuse a reservation in the resv_map,
// charge for the reservation.
if (*__slate_slot_12 as u32) != 0 {
let __v4: u8 = alloc_flags;
alloc_flags = (((__v4 as i32) as u64) | BIT!(0i32)) as u8;
}
// gbl_chg == 0 indicates a reservation exists for this
// allocation, so try to use it.
if *__slate_slot_14 == 0 {
let __v2: u8 = alloc_flags;
alloc_flags = (((__v2 as i32) as u64) | BIT!(1i32)) as u8;
}
// Takes reference on mpol.
*__slate_slot_6 = huge_node(vma, addr, gfp, core::ptr::addr_of_mut!(*__slate_slot_9), core::ptr::addr_of_mut!(*__slate_slot_8));
*__slate_slot_11 = *{
__slate_compound_19 = {
let mut __t0: mempolicy_interpreted = core::mem::zeroed();
__t0.nid = *__slate_slot_6;
__t0.nodemask = *__slate_slot_8;
__t0.mode = (if !(*__slate_slot_9).is_null() { (*(*__slate_slot_9)).mode as i32 } else { 0i32 }) as u32;
__t0
};
core::ptr::addr_of_mut!(__slate_compound_19)
};
*__slate_slot_16 = hugetlb_alloc_folio(h, core::ptr::addr_of_mut!(*__slate_slot_11), alloc_flags);
mpol_cond_put(*__slate_slot_9);
if IS_ERR(*__slate_slot_16 as *const ()) {
*__slate_slot_5 = PTR_ERR(*__slate_slot_16 as *const ()) as i32;
if (*__slate_slot_12 as u32) != 0 && *__slate_slot_14 == 0 {
*__slate_slot_13 = hugepage_subpool_put_pages(spool, 1i64);
hugetlb_acct_memory(h, (*__slate_slot_13).wrapping_neg());
}
// put page to subpool iff the quota of subpool's rsv_hpages is used
// during hugepage_subpool_get_pages.
} else {
hugetlb_set_folio_subpool(*__slate_slot_16, spool);
if (*__slate_slot_12 as u32) != 2 {
// commit() is only needed if the map_chg is not enforced
*__slate_slot_15 = vma_commit_reservation(h, vma, addr);
// Check for possible race conditions. When it happens..
// The page was added to the reservation map between
// vma_needs_reservation and vma_commit_reservation.
// This indicates a race with hugetlb_reserve_pages.
// Adjust for the subpool count incremented above AND
// in hugetlb_reserve_pages for the same page.	Also,
// the reservation count added in hugetlb_reserve_pages
// no longer applies.
if unlikely!((*__slate_slot_12 as u32) == 1 && *__slate_slot_15 == 0) != 0 {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = hugepage_subpool_put_pages(spool, 1i64);
hugetlb_acct_memory(h, (*__slate_slot_0).wrapping_neg());
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
hugetlb_cgroup_uncharge_folio_rsvd(hstate_index(h), pages_per_huge_page(h as *const hstate) as u64, *__slate_slot_16);
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
}
}
return *__slate_slot_16;
}
}
if (*__slate_slot_12 as u32) != 2 {
vma_end_reservation(h, vma, addr);
}
return ERR_PTR(*__slate_slot_5 as i64) as *mut folio;
}
}

#[unsafe(link_section = ".init.text")]
fn alloc_bootmem(h: *mut hstate, nid: i32, node_exact: bool) -> *mut () {
if hugetlb_early_cma(h) {
return hugetlb_cma_alloc_bootmem(h, nid, node_exact);
}
unsafe { memblock_alloc_hugetlb(huge_page_size(h as *const hstate), nid, node_exact) }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn __alloc_bootmem_huge_page(h: *mut hstate, nid: i32) -> *mut () {
let mut nr_nodes: i32 = 0;
let mut node: i32 = nid;
// do node specific alloc
if nid != 1i32.wrapping_neg() {
return alloc_bootmem(h, node, true);
}
// allocate from next node when distributing huge pages
nr_nodes = nodes_weight!(*unsafe { core::ptr::addr_of_mut!(hugetlb_bootmem_nodes) });
loop {
let __v0: bool = if nr_nodes > 0 {
let __v1: i32 = hstate_next_node_to_alloc(unsafe { core::ptr::addr_of_mut!((*h).next_nid_to_alloc) }, unsafe { core::ptr::addr_of_mut!(hugetlb_bootmem_nodes) });
node = __v1;
__v1 != 0 || true
} else {
false
};
if !__v0 {
break;
}
return alloc_bootmem(h, node, false);
}
core::ptr::null_mut::<()>()
}

#[unsafe(link_section = ".init.text")]
fn alloc_bootmem_huge_page(h: *mut hstate, mut nid: i32) -> bool {
let mut pfn: u64 = 0;
let nid_request: u32 = nid as u32;
let m: *mut huge_bootmem_page = __alloc_bootmem_huge_page(h, nid) as *mut huge_bootmem_page;
if m.is_null() {
return false;
}
pfn = PHYS_PFN!(__pa!(m));
nid = unsafe { early_pfn_to_nid(pfn) };
// Use the beginning of the huge page to store the huge_bootmem_page
// struct (until gather_bootmem puts them into the mem_map).
//
// Put them into a private list first because mem_map is not up yet.
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*m).list) });
unsafe {
(*m).hstate = h;
}
unsafe {
(*m).flags = (if hugetlb_early_cma(h) { 4i32 } else { 0i32 }) as u64;
}
// CMA pages: zone-crossing is validated in hugetlb_cma_reserve().
let __v0: bool = !hugetlb_early_cma(h) && unsafe { pfn_range_intersects_zones(nid, pfn, pages_per_huge_page(h as *const hstate) as u64) };
if __v0 {
// If the allocated page is on a different node than requested
// (e.g., on PowerPC LPARs), put it on the requested node's list,
// because hugetlb_free_cross_zone_pages() only frees cross-zone
// pages belonging to the requested node.
let mut __v1: i64;
let __ret_warn_on: i32 = (nid_request != (1i32.wrapping_neg() as u32) && (nid as u32) != nid_request) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("712: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 712b - ., 3; .popsection\n# {0}", const 712 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_alloc_bootmem_huge_page_2, sym __slate_asm_string_alloc_bootmem_huge_page_3, const 3144 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("713: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 713b - ., 4; .popsection\n# {0}", const 713 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
list_add(unsafe { core::ptr::addr_of_mut!((*m).list) }, unsafe { core::ptr::addr_of_mut!(huge_boot_pages.0) as *mut list_head }.wrapping_offset(nid_request as isize));
} else {
list_add(unsafe { core::ptr::addr_of_mut!((*m).list) }, unsafe { core::ptr::addr_of_mut!(huge_boot_pages.0) as *mut list_head }.wrapping_offset(nid as isize));
}
} else {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*m).list) }, unsafe { core::ptr::addr_of_mut!(huge_boot_pages.0) as *mut list_head }.wrapping_offset(nid as isize));
unsafe {
(*m).flags = (*m).flags | 2;
}
// Only initialize the head struct page in memmap_init_reserved_pages,
// rest of the struct pages will be initialized by the HugeTLB
// subsystem itself.
// The head struct page is used to get folio information by the HugeTLB
// subsystem like zone id and node id.
unsafe { memblock_reserved_mark_noinit(__pa!((m as *const u8).wrapping_offset((_AC!() << 12i32) as isize) as *mut ()), huge_page_size(h as *const hstate).wrapping_sub(_AC!() << 12i32)) };
}
true
}

/// Initialize [start_page:end_page_number] tail struct pages of a hugepage
#[unsafe(link_section = ".init.text")]
fn hugetlb_folio_init_tail_vmemmap(folio: *mut folio, h: *mut hstate, start_page_number: u64, end_page_number: u64) {
let zone: u32 = folio_zonenum(folio as *const folio);
let nid: i32 = folio_nid(folio as *const folio);
let mut page: *mut page = folio_page!(folio, start_page_number);
let head_pfn: u64 = folio_pfn(folio as *const folio);
let end_pfn: u64 = head_pfn.wrapping_add(end_page_number);
let order: u32 = huge_page_order(h);
// As we marked all tail pages with memblock_reserved_mark_noinit(),
// we must initialize them ourselves here.
for pfn in head_pfn.wrapping_add(start_page_number)..end_pfn {
unsafe { __init_single_page(page, pfn, (zone as u32) as u64, nid) };
prep_compound_tail(page, (unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }) as *const page, order);
set_page_count(page, 0i32);
let __v1: *mut page = page;
page = __v1.wrapping_offset(1 as isize);
}
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_folio_init_vmemmap(folio: *mut folio, h: *mut hstate, nr_pages: u64) {
let mut ret: i32 = 0;
// This is an open-coded prep_compound_page() whereby we avoid
// walking pages twice by initializing/preparing+freezing them in the
// same go.
__folio_clear_reserved(folio);
__folio_set_head(folio);
ret = folio_ref_freeze(folio, 1i32);
VM_BUG_ON!();
hugetlb_folio_init_tail_vmemmap(folio, h, 1u64, nr_pages);
prep_compound_head(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, huge_page_order(h));
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_bootmem_page_prehvo(m: *mut huge_bootmem_page) -> bool {
(unsafe { (*m).flags }) & 1 != 0
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_bootmem_page_earlycma(m: *mut huge_bootmem_page) -> bool {
(unsafe { (*m).flags }) & 4 != 0
}

/// memblock-allocated pageblocks might not have the migrate type set
/// if marked with the 'noinit' flag. Set it to the default (MIGRATE_MOVABLE)
/// here, or MIGRATE_CMA if this was a page allocated through an early CMA
/// reservation.
///
/// In case of vmemmap optimized folios, the tail vmemmap pages are mapped
/// read-only, but that's ok - for sparse vmemmap this does not write to
/// the page structure.
#[unsafe(link_section = ".init.text")]
fn hugetlb_bootmem_init_migratetype(folio: *mut folio, h: *mut hstate) {
let nr_pages: u64 = pages_per_huge_page(h as *const hstate) as u64;
let mut i: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!pageblock_aligned!(folio_pfn(folio as *const folio)))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("714: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 714b - ., 3; .popsection\n# {0}", const 714 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_hugetlb_bootmem_init_migratetype_1, sym __slate_asm_string_hugetlb_bootmem_init_migratetype_2, const 3233 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("715: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 715b - ., 4; .popsection\n# {0}", const 715 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
while i < nr_pages {
if folio_test_hugetlb_cma(folio) {
init_cma_pageblock(folio_page!(folio, i));
} else {
unsafe { init_pageblock_migratetype(folio_page!(folio, i), 1u32, false) };
}
let __v6: u64 = i;
i = __v6.wrapping_add(1u64 << MIN_T__2!(u32, 9i32, 10i32));
}
}

#[unsafe(link_section = ".init.text")]
fn prep_and_add_bootmem_folios(h: *mut hstate, folio_list: *mut list_head) {
let mut flags: u64 = 0;
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let mut tmp_f: *mut folio = unsafe { core::mem::zeroed() };
// Send list for bulk vmemmap optimization processing
unsafe { hugetlb_vmemmap_optimize_bootmem_folios(h, folio_list) };
folio = list_first_entry!(folio_list);
tmp_f = list_next_entry!(folio, lru);
while list_entry_is_head__2!(folio, folio_list, lru) == 0 {
if !folio_test_hugetlb_vmemmap_optimized(folio) {
// If HVO fails, initialize all tail struct pages
// We do not worry about potential long lock hold
// time as this is early in boot and there should
// be no contention.
hugetlb_folio_init_tail_vmemmap(folio, h, (_AC!() << 12i32) / (core::mem::size_of::<page>() as u64), pages_per_huge_page(h as *const hstate) as u64);
}
hugetlb_bootmem_init_migratetype(folio, h);
// Subdivide locks to achieve better parallel performance
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) }, flags);
account_new_hugetlb_folio(h, folio);
enqueue_hugetlb_folio(h, folio);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) }, flags);
folio = tmp_f;
tmp_f = list_next_entry!(tmp_f, lru);
}
}

/// Put bootmem huge pages into the standard lists after mem_map is up.
/// Note: This only applies to gigantic (order > MAX_PAGE_ORDER) pages.
#[unsafe(link_section = ".init.text")]
fn gather_bootmem_prealloc_node(nid: u64) {
let mut folio_list: list_head = unsafe { core::mem::zeroed() };
folio_list = list_head { next: core::ptr::addr_of_mut!(folio_list), prev: core::ptr::addr_of_mut!(folio_list) };
let mut m: *mut huge_bootmem_page = unsafe { core::mem::zeroed() };
let mut tm: *mut huge_bootmem_page = unsafe { core::mem::zeroed() };
let mut h: *mut hstate = core::ptr::null_mut::<hstate>();
let mut prev_h: *mut hstate = core::ptr::null_mut::<hstate>();
m = list_first_entry__13!(unsafe { core::ptr::addr_of_mut!(huge_boot_pages.0) as *mut list_head }.wrapping_offset(nid as isize));
tm = list_next_entry__9!(m, list);
while list_entry_is_head!(m, unsafe { core::ptr::addr_of_mut!(huge_boot_pages.0) as *mut list_head }.wrapping_offset(nid as isize), list) == 0 {
let page: *mut page = virt_to_page!(m);
let folio: *mut folio = page as *mut folio;
h = unsafe { (*m).hstate };
// It is possible to have multiple huge page sizes (hstates)
// in this list.  If so, process each size separately.
if h != prev_h && !prev_h.is_null() {
prep_and_add_bootmem_folios(prev_h, core::ptr::addr_of_mut!(folio_list));
}
prev_h = h;
VM_BUG_ON!();
let mut __v3: i64;
let __ret_warn_on: i32 = (folio_ref_count(folio as *const folio) != 1) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("716: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 716b - ., 3; .popsection\n# {0}", const 716 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_gather_bootmem_prealloc_node_4, sym __slate_asm_string_gather_bootmem_prealloc_node_5, const 3298 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("717: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 717b - ., 4; .popsection\n# {0}", const 717 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
hugetlb_folio_init_vmemmap(folio, h, (_AC!() << 12i32) / (core::mem::size_of::<page>() as u64));
init_new_hugetlb_folio(folio);
if hugetlb_bootmem_page_prehvo(m) {
folio_set_hugetlb_vmemmap_optimized(folio);
}
// If pre-HVO was done, just set the
// flag, the HVO code will then skip
// this folio.
if hugetlb_bootmem_page_earlycma(m) {
folio_set_hugetlb_cma(folio);
}
list_add(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) }, core::ptr::addr_of_mut!(folio_list));
// We need to restore the 'stolen' pages to totalram_pages
// in order to fix confusing memory reports from free(1) and
// other side-effects, like CommitLimit going negative.
//
// For CMA pages, this is done in init_cma_pageblock
// (via hugetlb_bootmem_init_migratetype), so skip it here.
if !folio_test_hugetlb_cma(folio) {
unsafe { adjust_managed_page_count(page, pages_per_huge_page(h as *const hstate) as i64) };
}
cond_resched__50!();
m = tm;
tm = list_next_entry__9!(tm, list);
}
prep_and_add_bootmem_folios(h, core::ptr::addr_of_mut!(folio_list));
}

#[unsafe(link_section = ".init.text")]
extern "C" fn gather_bootmem_prealloc_parallel(start: u64, end: u64, arg: *mut ()) {
let mut nid: i32 = start as i32;
while (nid as u64) < end {
gather_bootmem_prealloc_node(nid as u64);
nid = nid.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn hugetlb_bootmem_struct_page_init() {
let mut job: padata_mt_job = {
let mut __t0: padata_mt_job = unsafe { core::mem::zeroed() };
__t0.thread_fn = Some(gather_bootmem_prealloc_parallel);
__t0.fn_arg = core::ptr::null_mut::<()>();
__t0.start = 0u64;
__t0.size = (unsafe { nr_node_ids }) as u64;
__t0.align = 1u64;
__t0.min_chunk = 1u64;
__t0.max_threads = num_node_state(3u32);
__t0.numa_aware = true;
__t0
};
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
let mut i: i32 = 0;
while i < if ilog2__2!(17179869184u64).wrapping_sub(12).wrapping_sub(ilog2__2!(8192u64 / (core::mem::size_of::<page>() as u64))).wrapping_add(1) > 0 { ilog2__2!(17179869184u64).wrapping_sub(12).wrapping_sub(ilog2__2!(8192u64 / (core::mem::size_of::<page>() as u64))).wrapping_add(1) } else { 0 } {
let mut order: u32 = 0;
let tail: *mut page = unsafe { *((*zone).vmemmap_tails.as_mut_ptr() as *mut *mut page).wrapping_offset(i as isize) };
if !tail.is_null() {
order = i.wrapping_add(ilog2__2!(8192u64 / (core::mem::size_of::<page>() as u64))) as u32;
let p: *mut page = page_to_virt__2!(tail) as *mut page;
// prep_and_add_bootmem_folios() can access pageblock
// flags on bootmem HugeTLB pages, so initialize the
// shared tail struct pages here before bootmem folios
// start using them.
let mut j: i32 = 0;
while (j as u64) < (_AC!() << 12i32) / (core::mem::size_of::<page>() as u64) {
init_compound_tail(p.wrapping_offset(j as isize), core::ptr::null::<page>(), order, zone);
j = j.wrapping_add(1);
}
}
i = i.wrapping_add(1);
}
zone = unsafe { next_zone(zone) };
}
unsafe { padata_do_multithreaded(core::ptr::addr_of_mut!(job)) };
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_free_cross_zone_pages(h: *mut hstate, nid: i32) -> u64 {
let mut freed: u64 = 0;
let mut m: *mut huge_bootmem_page = unsafe { core::mem::zeroed() };
let mut tmp: *mut huge_bootmem_page = unsafe { core::mem::zeroed() };
if !hstate_is_gigantic(h) {
return freed;
}
m = list_first_entry__13!(unsafe { core::ptr::addr_of_mut!(huge_boot_pages.0) as *mut list_head }.wrapping_offset(nid as isize));
tmp = list_next_entry__9!(m, list);
while list_entry_is_head!(m, unsafe { core::ptr::addr_of_mut!(huge_boot_pages.0) as *mut list_head }.wrapping_offset(nid as isize), list) == 0 {
if (unsafe { (*m).flags }) & 2 != 0 {
break;
}
list_del(unsafe { core::ptr::addr_of_mut!((*m).list) });
unsafe { memblock_free(m as *mut (), huge_page_size(h as *const hstate)) };
freed = freed.wrapping_add(1);
m = tmp;
tmp = list_next_entry__9!(tmp, list);
}
if freed != 0 {
let mut buf: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8; 32]);
unsafe { string_get_size(huge_page_size(h as *const hstate), 1u64, 1u32, buf.0.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 32]>() as i32) };
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x014HugeTLB: freed %lu cross-zone hugepages of size %s on node %d.\n".as_ptr() as *const u8, freed, buf.0.as_mut_ptr() as *mut u8, nid) };
}
freed
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_hstate_alloc_pages_onenode(h: *mut hstate, nid: i32) {
let mut i: u64 = 0;
let mut buf: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8; 32]);
let mut folio_list: list_head = unsafe { core::mem::zeroed() };
folio_list = list_head { next: core::ptr::addr_of_mut!(folio_list), prev: core::ptr::addr_of_mut!(folio_list) };
while i < ((unsafe { *((*h).max_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize) }) as u64) {
if hstate_is_gigantic(h) {
if !alloc_bootmem_huge_page(h, nid) {
break;
}
} else {
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let gfp_mask: u32 = htlb_alloc_mask(h) | (BIT!(21i32) as u32);
folio = only_alloc_fresh_hugetlb_folio(h, gfp_mask, nid, unsafe { core::ptr::addr_of_mut!(node_states) as *mut __SlateRecord_nodemask_types_h_17 }.wrapping_offset(3 as isize), core::ptr::null_mut::<__SlateRecord_nodemask_types_h_17>());
let __v1: bool = folio.is_null() && list_empty(core::ptr::addr_of_mut!(folio_list) as *const list_head) == 0;
let __v2: bool = __v1 && hugetlb_vmemmap_optimizable_size(h as *const hstate) != 0;
if __v2 {
prep_and_add_allocated_folios(h, core::ptr::addr_of_mut!(folio_list));
INIT_LIST_HEAD(core::ptr::addr_of_mut!(folio_list));
folio = only_alloc_fresh_hugetlb_folio(h, gfp_mask, nid, unsafe { core::ptr::addr_of_mut!(node_states) as *mut __SlateRecord_nodemask_types_h_17 }.wrapping_offset(3 as isize), core::ptr::null_mut::<__SlateRecord_nodemask_types_h_17>());
}
if folio.is_null() {
break;
}
list_add(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) }, core::ptr::addr_of_mut!(folio_list));
}
cond_resched__56!();
i = i.wrapping_add(1);
}
let __v6: u64 = hugetlb_free_cross_zone_pages(h, nid);
i = i.wrapping_sub(__v6);
if list_empty(core::ptr::addr_of_mut!(folio_list) as *const list_head) == 0 {
prep_and_add_allocated_folios(h, core::ptr::addr_of_mut!(folio_list));
}
if i == ((unsafe { *((*h).max_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize) }) as u64) {
return;
}
unsafe { string_get_size(huge_page_size(h as *const hstate), 1u64, 1u32, buf.0.as_mut_ptr() as *mut u8, 32i32) };
let mut __v9: i32;
while false {
}
__v9 = unsafe { _printk(c"\x014HugeTLB: allocating %u of page size %s failed node%d.  Only allocated %lu hugepages.\n".as_ptr() as *const u8, *((*h).max_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize), buf.0.as_mut_ptr() as *mut u8, nid, i) };
let __v12: u64 = unsafe { (*h).max_huge_pages };
let __v13: u64 = __v12.wrapping_sub(((unsafe { *((*h).max_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize) }) as u64).wrapping_sub(i));
unsafe {
(*h).max_huge_pages = __v13;
}
unsafe {
*((*h).max_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize) = i as u32;
}
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_hstate_alloc_pages_specific_nodes(h: *mut hstate) -> bool {
let mut i: i32 = first_node!(node_states[1]) as i32;
let mut node_specific_alloc: bool = false;
while i < 64i32 {
if (unsafe { *((*h).max_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) }) > 0 {
hugetlb_hstate_alloc_pages_onenode(h, i);
node_specific_alloc = true;
}
i = next_node!(i, node_states[1]) as i32;
}
node_specific_alloc
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_hstate_alloc_pages_errcheck(allocated: u64, h: *mut hstate) {
if allocated < unsafe { (*h).max_huge_pages } {
let mut buf: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8; 32]);
unsafe { string_get_size(huge_page_size(h as *const hstate), 1u64, 1u32, buf.0.as_mut_ptr() as *mut u8, 32i32) };
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014HugeTLB: allocating %lu of page size %s failed.  Only allocated %lu hugepages.\n".as_ptr() as *const u8, (*h).max_huge_pages, buf.0.as_mut_ptr() as *mut u8, allocated) };
unsafe {
(*h).max_huge_pages = allocated;
}
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn hugetlb_pages_alloc_boot_node(start: u64, end: u64, arg: *mut ()) {
let h: *mut hstate = arg as *mut hstate;
let num: i32 = end.wrapping_sub(start) as i32;
let mut node_alloc_noretry: __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(node_alloc_noretry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut folio_list: list_head = unsafe { core::mem::zeroed() };
folio_list = list_head { next: core::ptr::addr_of_mut!(folio_list), prev: core::ptr::addr_of_mut!(folio_list) };
let mut next_node: i32 = first_node!(node_states[1]) as i32;
// Bit mask controlling how hard we retry per-node allocations.
nodes_clear__2!(node_alloc_noretry);
for i in 0..num {
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let __v0: bool = hugetlb_vmemmap_optimizable_size(h as *const hstate) != 0 && (unsafe { si_mem_available() }) == 0;
let __v1: bool = __v0 && list_empty(core::ptr::addr_of_mut!(folio_list) as *const list_head) == 0;
if __v1 {
prep_and_add_allocated_folios(h, core::ptr::addr_of_mut!(folio_list));
INIT_LIST_HEAD(core::ptr::addr_of_mut!(folio_list));
}
folio = alloc_pool_huge_folio(h, unsafe { core::ptr::addr_of_mut!(node_states) as *mut __SlateRecord_nodemask_types_h_17 }.wrapping_offset(3 as isize), core::ptr::addr_of_mut!(node_alloc_noretry), core::ptr::addr_of_mut!(next_node));
if folio.is_null() {
break;
}
list_move(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) }, core::ptr::addr_of_mut!(folio_list));
cond_resched__65!();
}
prep_and_add_allocated_folios(h, core::ptr::addr_of_mut!(folio_list));
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_gigantic_pages_alloc_boot(h: *mut hstate) -> u64 {
let mut nid: i32 = 0;
let mut i: u64 = 0;
while i < unsafe { (*h).max_huge_pages } {
if !alloc_bootmem_huge_page(h, 1i32.wrapping_neg()) {
break;
}
cond_resched__10!();
i = i.wrapping_add(1);
}
nid = first_node!(node_states[0]) as i32;
while nid < 64i32 {
let __v4: u64 = hugetlb_free_cross_zone_pages(h, nid);
i = i.wrapping_sub(__v4);
nid = next_node!(nid, node_states[0]) as i32;
}
i
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_pages_alloc_boot(h: *mut hstate) -> u64 {
let mut job: padata_mt_job = {
let mut __t0: padata_mt_job = unsafe { core::mem::zeroed() };
__t0.fn_arg = h as *mut ();
__t0.align = 1u64;
__t0.numa_aware = true;
__t0
};
let mut jiffies_start: u64 = 0;
let mut jiffies_end: u64 = 0;
let mut remaining: u64 = 0;
job.thread_fn = Some(hugetlb_pages_alloc_boot_node);
// job.max_threads is 25% of the available cpu threads by default.
//
// On large servers with terabytes of memory, huge page allocation
// can consume a considerably amount of time.
//
// Tests below show how long it takes to allocate 1 TiB of memory with 2MiB huge pages.
// 2MiB huge pages. Using more threads can significantly improve allocation time.
//
// | threads               |   8   |   16  |   32  |   64  |   128 |
// | skylake      144 cpus |   44s |   22s |   16s |   19s |   20s |
// | cascade lake 192 cpus |   39s |   20s |   11s |   10s |    9s |
if (unsafe { hugepage_allocation_threads }) == 0 {
unsafe {
hugepage_allocation_threads = (num_online_cpus() / 4) as u64;
}
let mut __v0: u64;
let __UNIQUE_ID_x__718: u64 = unsafe { hugepage_allocation_threads };
let __UNIQUE_ID_y__719: i32 = 1;
loop {
if __types_ok__7!(__UNIQUE_ID_x__718, __UNIQUE_ID_y__719) == 0 {
unsafe { __compiletime_assert_720() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__718 > (__UNIQUE_ID_y__719 as u64) { __UNIQUE_ID_x__718 } else { __UNIQUE_ID_y__719 as u64 };
unsafe {
hugepage_allocation_threads = __v0;
}
}
job.max_threads = (unsafe { hugepage_allocation_threads }) as i32;
jiffies_start = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
loop {
remaining = unsafe { (*h).max_huge_pages }.wrapping_sub(unsafe { (*h).nr_huge_pages });
job.start = unsafe { (*h).nr_huge_pages };
job.size = remaining;
job.min_chunk = remaining / unsafe { hugepage_allocation_threads };
unsafe { padata_do_multithreaded(core::ptr::addr_of_mut!(job)) };
if (unsafe { (*h).nr_huge_pages }) == unsafe { (*h).max_huge_pages } {
break;
}
// Retry only if the vmemmap optimization might have been able to free
// some memory back to the system.
if !hugetlb_vmemmap_optimizable(h as *const hstate) {
break;
}
// Continue if progress was made in last iteration
if remaining == unsafe { (*h).max_huge_pages }.wrapping_sub(unsafe { (*h).nr_huge_pages }) {
break;
}
}
jiffies_end = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x016HugeTLB: allocation took %dms with hugepage_allocation_threads=%ld\n".as_ptr() as *const u8, jiffies_to_msecs(jiffies_end.wrapping_sub(jiffies_start)), hugepage_allocation_threads) };
unsafe { (*h).nr_huge_pages }
}

/// NOTE: this routine is called in different contexts for gigantic and
/// non-gigantic pages.
/// - For gigantic pages, this is called early in the boot process and
///   pages are allocated from memblock allocated or something similar.
///   Gigantic pages are actually added to pools later with the routine
///   hugetlb_bootmem_struct_page_init.
/// - For non-gigantic pages, this is called later in the boot process after
///   all of mm is up and functional.  Pages are allocated from buddy and
///   then added to hugetlb pools.
#[unsafe(link_section = ".init.text")]
fn hugetlb_hstate_alloc_pages(h: *mut hstate) {
let mut allocated: u64 = 0;
// Skip gigantic hugepages allocation if early CMA
// reservations are not available.
let __v0: bool = hstate_is_gigantic(h) && hugetlb_cma_total_size() != 0;
let __v1: bool = __v0 && !hugetlb_early_cma(h);
if __v1 {
let mut __v2: i64;
let __ret_do_once: bool = true;
let mut __v3: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely!(__ret_cond) != 0 && unlikely!(!unsafe { __already_done__3620 }) != 0 {
unsafe {
__already_done__3620 = true;
}
__ret_once = true;
}
__v3 = __ret_once as i64;
if __v3 != 0 {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x014HugeTLB: hugetlb_cma is enabled, skip boot time allocation\n".as_ptr() as *const u8) };
}
__v2 = __ret_do_once as i64;
return;
}
if (unsafe { (*h).max_huge_pages }) == 0 {
return;
}
// do node specific alloc
if hugetlb_hstate_alloc_pages_specific_nodes(h) {
return;
}
// below will do all node balanced alloc
if hstate_is_gigantic(h) {
allocated = hugetlb_gigantic_pages_alloc_boot(h);
} else {
allocated = hugetlb_pages_alloc_boot(h);
}
hugetlb_hstate_alloc_pages_errcheck(allocated, h);
}

#[unsafe(link_section = ".data..once")]
static mut __already_done__3620: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn hugetlb_init_hstates() {
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut h2: *mut hstate = unsafe { core::mem::zeroed() };
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
// Always reset to first_memory_node here, even if
// next_nid_to_alloc was set before - we can't
// reference hugetlb_bootmem_nodes after init, and
// first_memory_node is right for all further allocations.
unsafe {
(*h).next_nid_to_alloc = first_node!(node_states[3]) as i32;
}
unsafe {
(*h).next_nid_to_free = first_node!(node_states[3]) as i32;
}
// oversize hugepages were init'ed in early boot
if !hstate_is_gigantic(h) {
hugetlb_hstate_alloc_pages(h);
}
// Set demote order for each hstate.  Note that
// h->demote_order is initially 0.
// - We can not demote gigantic pages if runtime freeing
//   is not supported, so skip this.
// - If CMA allocation is possible, we can not demote
//   HUGETLB_PAGE_ORDER or smaller size pages.
if !hstate_is_gigantic_no_runtime(h) {
if !(hugetlb_cma_total_size() != 0 && (unsafe { (*h).order }) <= 9u32) {
h2 = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h2 < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
if h2 != h {
if (unsafe { (*h2).order }) < unsafe { (*h).order } && (unsafe { (*h2).order }) > unsafe { (*h).demote_order } {
unsafe {
(*h).demote_order = (*h2).order;
}
}
}
let __v2: *mut hstate = h2;
h2 = __v2.wrapping_offset(1 as isize);
}
}
}
let __v4: *mut hstate = h;
h = __v4.wrapping_offset(1 as isize);
}
}

#[unsafe(link_section = ".init.text")]
fn report_hugepages() {
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
let mut buf: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8; 32]);
unsafe { string_get_size(huge_page_size(h as *const hstate), 1u64, 1u32, buf.0.as_mut_ptr() as *mut u8, 32i32) };
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016HugeTLB: registered %s page size, pre-allocated %ld pages\n".as_ptr() as *const u8, buf.0.as_mut_ptr() as *mut u8, (*h).nr_huge_pages) };
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016HugeTLB: %d KiB vmemmap can be freed for a %s page\n".as_ptr() as *const u8, hugetlb_vmemmap_optimizable_size(h as *const hstate) / 1024, buf.0.as_mut_ptr() as *mut u8) };
let __v5: *mut hstate = h;
h = __v5.wrapping_offset(1 as isize);
}
}

fn try_to_free_low(h: *mut hstate, count: u64, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17) {
}

/// Increment or decrement surplus_huge_pages.  Keep node-specific counters
/// balanced by operating on them in a round-robin fashion.
/// Returns 1 if an adjustment was made.
fn adjust_pool_surplus(h: *mut hstate, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17, delta: i32) -> i32 {
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __v11: bool = unsafe { core::mem::zeroed() };
let mut node: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(node) as *mut i32;
let mut nr_nodes: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(nr_nodes) as *mut i32;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
lockdep_assert_held!(core::ptr::addr_of_mut!(hugetlb_lock.0));
VM_BUG_ON!();
if delta < 0 {
*__slate_slot_15 = nodes_weight!(*nodes_allowed);
loop {
if *__slate_slot_15 > 0 {
let __v6: i32 = hstate_next_node_to_alloc(core::ptr::addr_of_mut!((*h).next_nid_to_alloc), nodes_allowed);
*__slate_slot_14 = __v6;
__v7 = __v6 != 0 || true;
} else {
__v7 = false;
}
if __v7 {
if *((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_14 as isize) != 0 {
break '__join_0;
} else {
let __v9: i32 = *__slate_slot_15;
*__slate_slot_15 = __v9.wrapping_sub(1);
}
} else {
break;
}
}
} else {
*__slate_slot_15 = nodes_weight!(*nodes_allowed);
loop {
if *__slate_slot_15 > 0 {
let __v10: i32 = hstate_next_node_to_free(h, nodes_allowed);
*__slate_slot_14 = __v10;
__v11 = __v10 != 0 || true;
} else {
__v11 = false;
}
if __v11 {
if *((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_14 as isize) < *((*h).nr_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_14 as isize) {
break '__join_0;
} else {
let __v13: i32 = *__slate_slot_15;
*__slate_slot_15 = __v13.wrapping_sub(1);
}
} else {
break;
}
}
}
return 0;
}
let __v4: u64 = (*h).surplus_huge_pages;
(*h).surplus_huge_pages = __v4.wrapping_add(delta as u64);
let __v2: *mut u32 = ((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_14 as isize);
let __v1: u32 = *__v2;
*__v2 = __v1.wrapping_add(delta as u32);
return 1;
}
}

fn set_max_huge_pages(h: *mut hstate, mut count: u64, nid: i32, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17) -> i32 {
let mut __UNIQUE_ID_y__722: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__721: u64 = unsafe { core::mem::zeroed() };
let mut node_alloc_noretry: *mut __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };
let mut _node_alloc_noretry: core::mem::MaybeUninit<__SlateRecord_nodemask_types_h_17> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut __SlateRecord_nodemask_types_h_17 = core::ptr::addr_of_mut!(_node_alloc_noretry) as *mut __SlateRecord_nodemask_types_h_17;
let mut page_list: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut list_head = core::ptr::addr_of_mut!(page_list) as *mut list_head;
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut allocated: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(allocated) as *mut u64;
let mut min_count: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u64 = core::ptr::addr_of_mut!(min_count) as *mut u64;
let mut persistent_free_count: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u64 = core::ptr::addr_of_mut!(persistent_free_count) as *mut u64;
unsafe {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_12, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_12), prev: core::ptr::addr_of_mut!(*__slate_slot_12) });
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
node_alloc_noretry = core::ptr::addr_of_mut!(*__slate_slot_11);
// Bit mask controlling how hard we retry per-node allocations.
// If we can not allocate the bit mask, do not attempt to allocate
// the requested huge pages.
if !node_alloc_noretry.is_null() {
nodes_clear!(*node_alloc_noretry);
// resize_lock mutex prevents concurrent adjustments to number of
// pages in hstate via the proc/sysfs interfaces.
mutex_lock(core::ptr::addr_of_mut!((*h).resize_lock));
flush_free_hpage_work(h);
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
// Check for a node specific request.
// Changing node specific huge page count may require a corresponding
// change to the global count.  In any case, the passed node mask
// (nodes_allowed) will restrict alloc/free to the specified node.
if nid != 1i32.wrapping_neg() {
let old_count: u64 = count;
let __v8: u64 = count;
count = __v8.wrapping_add(persistent_huge_pages!(h).wrapping_sub(unsafe { *((*h).nr_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize) }.wrapping_sub(*((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize)) as u64));
// User may have specified a large count value which caused the
// above calculation to overflow.  In this case, they wanted
// to allocate as many huge pages as possible.  Set count to
// largest possible value to align with their intention.
if count < old_count {
count = !0;
}
}
// Gigantic pages runtime allocation depend on the capability for large
// page range allocation.
// If the system does not provide this feature, return an error when
// the user tries to allocate gigantic pages but let the user free the
// boottime allocated gigantic pages.
if hstate_is_gigantic(h) && IS_ENABLED!() == 0 {
if count > persistent_huge_pages!(h) {
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
mutex_unlock(core::ptr::addr_of_mut!((*h).resize_lock));
NODEMASK_FREE!();
return 22i32.wrapping_neg();
}
// Fall through to decrease pool
}
// Increase the pool size
// First take pages out of surplus state.  Then make up the
// remaining difference by allocating fresh huge pages.
//
// We might race with alloc_surplus_hugetlb_folio() here and be unable
// to convert a surplus huge page to a normal huge page. That is
// not critical, though, it just means the overall size of the
// pool might be one hugepage larger than it needs to be, but
// within all the constraints specified by the sysctls.
while (*h).surplus_huge_pages != 0 && count > persistent_huge_pages!(h) {
if adjust_pool_surplus(h, nodes_allowed, 1i32.wrapping_neg()) == 0 {
break;
}
}
*__slate_slot_14 = 0;
'__join_0: {
'__join_20: {
'__join_18: {
while count > persistent_huge_pages!(h).wrapping_add(*__slate_slot_14) {
// If this allocation races such that we no longer need the
// page, free_huge_folio will handle it by freeing the page
// and reducing the surplus.
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
// yield cpu to avoid soft lockup
cond_resched__29!();
*__slate_slot_13 = alloc_pool_huge_folio(h, nodes_allowed, node_alloc_noretry, core::ptr::addr_of_mut!((*h).next_nid_to_alloc));
if (*__slate_slot_13).is_null() {
break '__join_20;
} else {
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_13)).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru), core::ptr::addr_of_mut!(*__slate_slot_12));
let __v6: u64 = *__slate_slot_14;
*__slate_slot_14 = __v6.wrapping_add(1);
// Bail for signals. Probably ctrl-c from user
if signal_pending(get_current()) != 0 {
break '__join_18;
} else {
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
}
}
}
// Add allocated pages to the pool
if list_empty(core::ptr::addr_of_mut!(*__slate_slot_12) as *const list_head) == 0 {
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
prep_and_add_allocated_folios(h, core::ptr::addr_of_mut!(*__slate_slot_12));
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
}
// Decrease the pool size
// First return free pages to the buddy allocator (being careful
// to keep enough around to satisfy reservations).  Then place
// pages into surplus state as needed so the pool will shrink
// to the desired size as pages become free.
//
// By placing pages into the surplus state independent of the
// overcommit value, we are allowing the surplus pool size to
// exceed overcommit. There are few sane options here. Since
// alloc_surplus_hugetlb_folio() is checking the global counter,
// though, we'll note that we're not allowed to exceed surplus
// and won't grow the pool anywhere else. Not until one of the
// sysctls are changed, or the surplus pages go out of use.
//
// min_count is the expected number of persistent pages, we
// shouldn't calculate min_count by using
// resv_huge_pages + persistent_huge_pages() - free_huge_pages,
// because there may exist free surplus huge pages, and this will
// lead to subtracting twice. Free surplus huge pages come from HVO
// failing to restore vmemmap, see comments in the callers of
// hugetlb_vmemmap_restore_folio(). Thus, we should calculate
// persistent free count first.
*__slate_slot_16 = (*h).free_huge_pages;
if (*h).free_huge_pages > persistent_huge_pages!(h) {
if (*h).free_huge_pages > (*h).surplus_huge_pages {
let __v4: u64 = *__slate_slot_16;
*__slate_slot_16 = __v4.wrapping_sub((*h).surplus_huge_pages);
} else {
*__slate_slot_16 = 0;
}
}
*__slate_slot_15 = unsafe { (*h).resv_huge_pages }.wrapping_add(persistent_huge_pages!(h)).wrapping_sub(*__slate_slot_16);
__UNIQUE_ID_x__721 = count;
__UNIQUE_ID_y__722 = *__slate_slot_15;
while false {
}
*__slate_slot_15 = if __UNIQUE_ID_x__721 > __UNIQUE_ID_y__722 { __UNIQUE_ID_x__721 } else { __UNIQUE_ID_y__722 };
try_to_free_low(h, *__slate_slot_15, nodes_allowed);
// Collect pages to be removed on list without dropping lock
while *__slate_slot_15 < persistent_huge_pages!(h) {
*__slate_slot_13 = remove_pool_hugetlb_folio(h, nodes_allowed, false);
if (*__slate_slot_13).is_null() {
break;
} else {
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_13)).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru), core::ptr::addr_of_mut!(*__slate_slot_12));
}
}
// free the pages after dropping lock
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
update_and_free_pages_bulk(h, core::ptr::addr_of_mut!(*__slate_slot_12));
flush_free_hpage_work(h);
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
loop {
if count < persistent_huge_pages!(h) {
if adjust_pool_surplus(h, nodes_allowed, 1i32) == 0 {
break '__join_0;
}
} else {
break '__join_0;
}
}
}
prep_and_add_allocated_folios(h, core::ptr::addr_of_mut!(*__slate_slot_12));
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
break '__join_0;
}
prep_and_add_allocated_folios(h, core::ptr::addr_of_mut!(*__slate_slot_12));
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
}
(*h).max_huge_pages = persistent_huge_pages!(h);
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
mutex_unlock(core::ptr::addr_of_mut!((*h).resize_lock));
NODEMASK_FREE!();
return 0;
} else {
return 12i32.wrapping_neg();
}
}
}

fn demote_free_hugetlb_folios(src: *mut hstate, dst: *mut hstate, src_list: *mut list_head) -> i64 {
let mut rc: i64 = 0;
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let mut next: *mut folio = unsafe { core::mem::zeroed() };
let mut dst_list: list_head = unsafe { core::mem::zeroed() };
dst_list = list_head { next: core::ptr::addr_of_mut!(dst_list), prev: core::ptr::addr_of_mut!(dst_list) };
let mut ret_list: list_head = unsafe { core::mem::zeroed() };
ret_list = list_head { next: core::ptr::addr_of_mut!(ret_list), prev: core::ptr::addr_of_mut!(ret_list) };
rc = unsafe { hugetlb_vmemmap_restore_folios(src as *const hstate, src_list, core::ptr::addr_of_mut!(ret_list)) };
list_splice_init(core::ptr::addr_of_mut!(ret_list), src_list);
// Taking target hstate mutex synchronizes with set_max_huge_pages.
// Without the mutex, pages added to target hstate could be marked
// as surplus.
//
// Note that we already hold src->resize_lock.  To prevent deadlock,
// use the convention of always taking larger size hstate mutex first.
unsafe { mutex_lock(core::ptr::addr_of_mut!((*dst).resize_lock)) };
folio = list_first_entry!(src_list);
next = list_next_entry!(folio, lru);
while list_entry_is_head__2!(folio, src_list, lru) == 0 {
let mut i: i32 = 0;
let mut cma: bool = false;
if !folio_test_hugetlb_vmemmap_optimized(folio) {
cma = folio_test_hugetlb_cma(folio);
list_del(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) });
split_page_owner(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, huge_page_order(src) as i32, huge_page_order(dst) as i32);
pgalloc_tag_split(folio, huge_page_order(src) as i32, huge_page_order(dst) as i32);
i = 0;
while (i as u32) < pages_per_huge_page(src as *const hstate) {
let page: *mut page = folio_page!(folio, i);
// Careful: see __split_huge_page_tail()
let new_folio: *mut folio = page as *mut folio;
clear_compound_head(page);
unsafe { prep_compound_page(page, (*dst).order) };
unsafe {
(*new_folio).__slate_anon_0.__slate_anon_0.mapping = core::ptr::null_mut::<address_space>();
}
init_new_hugetlb_folio(new_folio);
// Copy the CMA flag so that it is freed correctly
if cma {
folio_set_hugetlb_cma(new_folio);
}
list_add(unsafe { core::ptr::addr_of_mut!((*new_folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) }, core::ptr::addr_of_mut!(dst_list));
let __v5: u32 = pages_per_huge_page(dst as *const hstate);
i = (i as u32).wrapping_add(__v5) as i32;
}
}
folio = next;
next = list_next_entry!(next, lru);
}
prep_and_add_allocated_folios(dst, core::ptr::addr_of_mut!(dst_list));
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*dst).resize_lock)) };
rc
}

#[unsafe(no_mangle)]
extern "C" fn demote_pool_huge_page(src: *mut hstate, nodes_allowed: *mut __SlateRecord_nodemask_types_h_17, nr_to_demote: u64) -> i64 {
let mut nr_nodes: i32 = 0;
let mut node: i32 = 0;
let mut rc: i64 = 0;
let mut nr_demoted: i64 = 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
// We should never get here if no demote order
if (unsafe { (*src).demote_order }) == 0 {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014HugeTLB: NULL demote order passed to demote_pool_huge_page.\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg() as i64; // internal error
}
let dst: *mut hstate = size_to_hstate(_AC!() << 12i32 << unsafe { (*src).demote_order });
nr_nodes = nodes_weight!(*nodes_allowed);
loop {
let __v20: bool = if nr_nodes > 0 {
let __v21: i32 = hstate_next_node_to_free(src, nodes_allowed);
node = __v21;
__v21 != 0 || true
} else {
false
};
if !__v20 {
break;
}
let mut list: list_head = unsafe { core::mem::zeroed() };
list = list_head { next: core::ptr::addr_of_mut!(list), prev: core::ptr::addr_of_mut!(list) };
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let mut next: *mut folio = unsafe { core::mem::zeroed() };
folio = list_first_entry!(unsafe { (*src).hugepage_freelists.as_mut_ptr() as *mut list_head }.wrapping_offset(node as isize));
next = list_next_entry!(folio, lru);
while list_entry_is_head__2!(folio, unsafe { (*src).hugepage_freelists.as_mut_ptr() as *mut list_head }.wrapping_offset(node as isize), lru) == 0 {
if !folio_test_hwpoison(folio as *const folio) {
remove_hugetlb_folio(src, folio, false);
list_add(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) }, core::ptr::addr_of_mut!(list));
let __v8: i64 = nr_demoted.wrapping_add(1);
nr_demoted = __v8;
if (__v8 as u64) == nr_to_demote {
break;
}
}
folio = next;
next = list_next_entry!(next, lru);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
rc = demote_free_hugetlb_folios(src, dst, core::ptr::addr_of_mut!(list));
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
folio = list_first_entry!(core::ptr::addr_of_mut!(list));
next = list_next_entry!(folio, lru);
while list_entry_is_head__2!(folio, core::ptr::addr_of_mut!(list), lru) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) });
add_hugetlb_folio(src, folio, false);
nr_demoted = nr_demoted.wrapping_sub(1);
folio = next;
next = list_next_entry!(next, lru);
}
if rc < 0 || (nr_demoted as u64) == nr_to_demote {
break;
}
nr_nodes = nr_nodes.wrapping_sub(1);
}
// Not absolutely necessary, but for consistency update max_huge_pages
// based on pool changes for the demoted page.
let __v25: u64 = unsafe { (*src).max_huge_pages };
unsafe {
(*src).max_huge_pages = __v25.wrapping_sub(nr_demoted as u64);
}
let __v27: i64 = nr_demoted << huge_page_order(src).wrapping_sub(huge_page_order(dst));
let __v29: u64 = unsafe { (*dst).max_huge_pages };
unsafe {
(*dst).max_huge_pages = __v29.wrapping_add(__v27 as u64);
}
if rc < 0 {
return rc;
}
if nr_demoted != 0 {
return nr_demoted;
}
// Only way to get here is if all pages on free lists are poisoned.
// Return -EBUSY so that caller will not retry.
16i32.wrapping_neg() as i64
}

#[unsafe(no_mangle)]
extern "C" fn __nr_hugepages_store_common(obey_mempolicy: bool, h: *mut hstate, nid: i32, count: u64, len: u64) -> i64 {
let mut err: i32 = 0;
let mut nodes_allowed: __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(nodes_allowed) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut n_mask: *mut __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };
if hstate_is_gigantic_no_runtime(h) {
return 22i32.wrapping_neg() as i64;
}
if nid == 1i32.wrapping_neg() {
// global hstate attribute
let __v0: bool = obey_mempolicy && unsafe { init_nodemask_of_mempolicy(core::ptr::addr_of_mut!(nodes_allowed)) };
if !__v0 {
n_mask = unsafe { core::ptr::addr_of_mut!(node_states) as *mut __SlateRecord_nodemask_types_h_17 }.wrapping_offset(3 as isize);
} else {
n_mask = core::ptr::addr_of_mut!(nodes_allowed);
}
} else {
// Node specific request.  count adjustment happens in
// set_max_huge_pages() after acquiring hugetlb_lock.
init_nodemask_of_node(core::ptr::addr_of_mut!(nodes_allowed), nid);
n_mask = core::ptr::addr_of_mut!(nodes_allowed);
}
err = set_max_huge_pages(h, count, nid, n_mask);
(if err != 0 { err as u64 } else { len }) as i64
}

#[unsafe(link_section = ".init.text")]
extern "C" fn hugetlb_init() -> i32 {
let mut __slate_compound_17: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
loop {
if sizeof_field!().wrapping_mul(8) < 7 {
unsafe { __compiletime_assert_724() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
BUILD_BUG_ON_INVALID!();
if hugepages_supported!() == 0 {
if (unsafe { hugetlb_max_hstate }) != 0 || (unsafe { default_hstate_max_huge_pages }) != 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014HugeTLB: huge pages not supported, ignoring associated command-line parameters\n".as_ptr() as *const u8) };
}
return 0;
}
// Make sure HPAGE_SIZE (HUGETLB_PAGE_ORDER) hstate exists.  Some
// architectures depend on setup being done here.
hugetlb_add_hstate(9u32);
if !unsafe { parsed_default_hugepagesz } {
// If we did not parse a default huge page size, set
// default_hstate_idx to HPAGE_SIZE hstate. And, if the
// number of huge pages for this default size was implicitly
// specified, set that here as well.
// Note that the implicit setting will overwrite an explicit
// setting.  A warning will be printed in this case.
unsafe {
default_hstate_idx = hstate_index(size_to_hstate(_AC!() << 21i32)) as u32;
}
if (unsafe { default_hstate_max_huge_pages }) != 0 {
if (unsafe { hstates.0[default_hstate_idx as usize].max_huge_pages }) != 0 {
let mut buf: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8; 32]);
unsafe { string_get_size(huge_page_size((core::ptr::addr_of_mut!(hstates.0) as *mut hstate).wrapping_offset(default_hstate_idx as isize) as *const hstate), 1u64, 1u32, buf.0.as_mut_ptr() as *mut u8, 32i32) };
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x014HugeTLB: Ignoring hugepages=%lu associated with %s page size\n".as_ptr() as *const u8, hstates.0[default_hstate_idx as usize].max_huge_pages, buf.0.as_mut_ptr() as *mut u8) };
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x014HugeTLB: Using hugepages=%lu for number of default huge pages\n".as_ptr() as *const u8, default_hstate_max_huge_pages) };
}
unsafe {
hstates.0[default_hstate_idx as usize].max_huge_pages = default_hstate_max_huge_pages;
}
i = first_node!(node_states[1]) as i32;
while i < 64i32 {
unsafe {
hstates.0[default_hstate_idx as usize].max_huge_pages_node[i as usize] = default_hugepages_in_node.0[i as usize];
}
i = next_node!(i, node_states[1]) as i32;
}
}
}
hugetlb_init_hstates();
report_hugepages();
unsafe { hugetlb_sysfs_init() };
unsafe { hugetlb_cgroup_file_init() };
unsafe { hugetlb_sysctl_init() };
unsafe {
num_fault_mutexes = roundup_pow_of_two__3!(8u32.wrapping_mul(num_possible_cpus())) as i32;
}
let mut __v14: *mut mutex;
let __obj_size: u64 = size_mul(core::mem::size_of::<mutex>() as u64, (unsafe { num_fault_mutexes }) as u64);
let mut __v15: *mut ();
let mut __v16: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_17 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_17)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
} });
}
__v16 = _res;
__v15 = __v16;
__v14 = __v15 as *mut mutex;
unsafe {
hugetlb_fault_mutex_table = __v14;
}
loop {
if unlikely!(unsafe { hugetlb_fault_mutex_table }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("725: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 725b - ., 3; .popsection\n# {0}", const 725 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_hugetlb_init_20, sym __slate_asm_string_hugetlb_init_21, const 4181 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
i = 0;
while i < unsafe { num_fault_mutexes } {
loop {
__mutex_init(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(i as isize), c"&hugetlb_fault_mutex_table[i]".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__4184) });
if true {
break;
}
}
i = i.wrapping_add(1);
}
0
}

static mut __key__4184: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hugetlb_init_727: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(hugetlb_init)) };

/// Overwritten by architectures with more huge page sizes
#[unsafe(no_mangle)]
#[linkage = "weak"]
#[unsafe(link_section = ".init.text")]
extern "C" fn arch_hugetlb_valid_size(size: u64) -> bool {
size == _AC!() << 21i32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn hugetlb_add_hstate(order: u32) {
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
if !size_to_hstate(_AC!() << 12i32 << order).is_null() {
return;
}
loop {
if unlikely!((unsafe { hugetlb_max_hstate }) >= 2) != 0 {
loop {
unsafe {
core::arch::asm!("728: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 728b - ., 3; .popsection\n# {0}", const 728 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_hugetlb_add_hstate_0, sym __slate_asm_string_hugetlb_add_hstate_1, const 4203 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
loop {
if unlikely!(order < (order_base_2__2!(3i32) as u32)) != 0 {
loop {
unsafe {
core::arch::asm!("729: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 729b - ., 3; .popsection\n# {0}", const 729 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_hugetlb_add_hstate_5, sym __slate_asm_string_hugetlb_add_hstate_6, const 4204 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
let mut __v10: i64;
let __ret_warn_on: i32 = (order > (ilog2__2!(17179869184u64).wrapping_sub(12) as u32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("730: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 730b - ., 3; .popsection\n# {0}", const 730 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_hugetlb_add_hstate_11, sym __slate_asm_string_hugetlb_add_hstate_12, const 4205 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("731: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 731b - ., 4; .popsection\n# {0}", const 731 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v10 = (__ret_warn_on != 0) as i64;
let __v15: i32 = unsafe { hugetlb_max_hstate };
unsafe {
hugetlb_max_hstate = __v15.wrapping_add(1);
}
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset(__v15 as isize);
__mutex_init(unsafe { core::ptr::addr_of_mut!((*h).resize_lock) }, c"resize mutex".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!((*h).resize_key) });
unsafe {
(*h).order = order;
}
unsafe {
(*h).mask = !huge_page_size(h as *const hstate).wrapping_sub(1);
}
for i in 0..64u64 {
INIT_LIST_HEAD(unsafe { (*h).hugepage_freelists.as_mut_ptr() as *mut list_head }.wrapping_offset(i as isize));
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*h).hugepage_activelist) });
unsafe { snprintf((*h).name.as_mut_ptr() as *mut u8, 32u64, c"hugepages-%lukB".as_ptr() as *const u8, huge_page_size(h as *const hstate) / 1024) };
unsafe {
parsed_hstate = h;
}
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
#[unsafe(link_section = ".init.text")]
extern "C" fn hugetlb_node_alloc_supported() -> bool {
true
}

#[unsafe(link_section = ".init.text")]
fn hugepages_clear_pages_in_node() {
if (unsafe { hugetlb_max_hstate }) == 0 {
unsafe {
default_hstate_max_huge_pages = 0;
}
unsafe { memset((core::ptr::addr_of_mut!(default_hugepages_in_node.0) as *mut u32) as *mut (), 0i32, core::mem::size_of::<[u32; 64]>() as u64) };
} else {
unsafe {
(*parsed_hstate).max_huge_pages = 0;
}
unsafe { memset(((*parsed_hstate).max_huge_pages_node.as_mut_ptr() as *mut u32) as *mut (), 0i32, core::mem::size_of::<[u32; 64]>() as u64) };
}
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_add_param(s: *mut u8, setup: Option<unsafe extern "C" fn(*mut u8) -> i32>) -> i32 {
let mut len: u64 = 0;
if s.is_null() {
return 22i32.wrapping_neg();
}
if (unsafe { hugetlb_param_index }) >= 5i32 {
return 22i32.wrapping_neg();
}
len = unsafe { strlen(s as *const u8) }.wrapping_add(1);
if len.wrapping_add((unsafe { hstate_cmdline_index }) as u64) > (core::mem::size_of::<[u8; 2048]>() as u64) {
return 22i32.wrapping_neg();
}
let p: *mut u8 = unsafe { core::ptr::addr_of_mut!(hstate_cmdline_buf.0) as *mut u8 }.wrapping_offset((unsafe { hstate_cmdline_index }) as isize);
unsafe { memcpy(p as *mut (), s as *const (), len) };
unsafe {
hstate_cmdline_index = (hstate_cmdline_index as u64).wrapping_add(len) as i32;
}
unsafe {
hugetlb_params.0[hugetlb_param_index as usize].val = p;
}
unsafe {
hugetlb_params.0[hugetlb_param_index as usize].setup = setup;
}
let __v2: i32 = unsafe { hugetlb_param_index };
unsafe {
hugetlb_param_index = __v2.wrapping_add(1);
}
0
}

#[unsafe(link_section = ".init.text")]
fn hugetlb_parse_params() {
let mut i: i32 = 0;
while i < unsafe { hugetlb_param_index } {
let hcp: *mut hugetlb_cmdline = unsafe { core::ptr::addr_of_mut!(hugetlb_params.0) as *mut hugetlb_cmdline }.wrapping_offset(i as isize);
unsafe { unsafe { (*hcp).setup }.unwrap()((*hcp).val) };
i = i.wrapping_add(1);
}
hugetlb_cma_validate_params();
}

// hugepages command line processing
// hugepages normally follows a valid hugepagsz or default_hugepagsz
// specification.  If not, ignore the hugepages value.  hugepages can also
// be the first huge page command line  option in which case it implicitly
// specifies the number of huge pages for the default size.
#[unsafe(link_section = ".init.text")]
extern "C" fn hugepages_setup(s: *mut u8) -> i32 {
let mut __mask: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u64 = core::ptr::addr_of_mut!(__mask) as *mut u64;
let mut __sz: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(__sz) as *mut u64;
let mut __idx: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(__idx) as *mut u64;
let mut _mask: u64 = unsafe { core::mem::zeroed() };
let mut _i: u64 = unsafe { core::mem::zeroed() };
let mut __v16: bool = unsafe { core::mem::zeroed() };
let mut p: *mut u8 = unsafe { core::mem::zeroed() };
let mut tmp: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u64 = core::ptr::addr_of_mut!(tmp) as *mut u64;
let mut count: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut i32 = core::ptr::addr_of_mut!(count) as *mut i32;
let mut node: i32 = unsafe { core::mem::zeroed() };
let mut mhp: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut u64 = core::ptr::addr_of_mut!(mhp) as *mut *mut u64;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
node = 1i32.wrapping_neg();
let __slate_auto_bytes_2 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
p = s;
if !parsed_valid_hugepagesz {
while false {
}
_printk(c"\x014HugeTLB: hugepages=%s does not follow a valid hugepagesz, ignoring\n".as_ptr() as *const u8, s);
parsed_valid_hugepagesz = true;
return 22i32.wrapping_neg();
} else {
if hugetlb_max_hstate == 0 {
*__slate_slot_25 = core::ptr::addr_of_mut!(default_hstate_max_huge_pages);
} else {
*__slate_slot_25 = core::ptr::addr_of_mut!((*parsed_hstate).max_huge_pages);
}
// !hugetlb_max_hstate means we haven't parsed a hugepagesz= parameter
// yet, so this hugepages= parameter goes to the "default hstate".
// Otherwise, it goes with the previously parsed hugepagesz or
// default_hugepagesz.
if *__slate_slot_25 == last_mhp {
while false {
}
_printk(c"\x014HugeTLB: hugepages= specified twice without interleaving hugepagesz=, ignoring hugepages=%s\n".as_ptr() as *const u8, s);
return 1;
} else {
'__join_1: {
'__join_18: {
'__join_2: {
loop {
if *p != 0 {
*__slate_slot_23 = 0;
if sscanf(p as *const u8, c"%lu%n".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_22), core::ptr::addr_of_mut!(*__slate_slot_23)) != 1 {
break '__join_1;
} else {
// Parameter is node format
if (*p.wrapping_offset(*__slate_slot_23 as isize) as i32) == 58 {
if !hugetlb_node_alloc_supported() {
break '__join_18;
} else {
if *__slate_slot_22 >= 64u64 {
__v16 = true;
} else {
__v16 = node_online__2!(*__slate_slot_22) == 0;
}
if __v16 {
break '__join_1;
} else {
_i = *__slate_slot_22;
let _s: i32 = 64i32;
core::ptr::write(__slate_slot_10, _i);
core::ptr::write(__slate_slot_9, _s as u64);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("cmp {1:r},{2:r}; sbb {0:r},{0:r}", lateout(reg) __t0, in(reg) *__slate_slot_9 as u64, in(reg) *__slate_slot_10 as u64, options(nostack, att_syntax));
*__slate_slot_8 = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
_mask = *__slate_slot_8;
while false {
}
while false {
}
node = (_i & _mask) as i32;
let __v7: *mut u8 = p;
p = __v7.wrapping_offset((*__slate_slot_23).wrapping_add(1) as isize);
// Parse hugepages
if sscanf(p as *const u8, c"%lu%n".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_22), core::ptr::addr_of_mut!(*__slate_slot_23)) != 1 {
break '__join_1;
} else {
if hugetlb_max_hstate == 0 {
default_hugepages_in_node.0[node as usize] = *__slate_slot_22 as u32;
} else {
*((*parsed_hstate).max_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(node as isize) = *__slate_slot_22 as u32;
}
let __v5: *mut u64 = *__slate_slot_25;
let __v4: u64 = *__v5;
*__v5 = __v4.wrapping_add(*__slate_slot_22);
// Go to parse next node
if (*p.wrapping_offset(*__slate_slot_23 as isize) as i32) == 44 {
let __v2: *mut u8 = p;
p = __v2.wrapping_offset((*__slate_slot_23).wrapping_add(1) as isize);
} else {
break '__join_2;
}
}
}
}
} else {
break;
}
}
} else {
break '__join_2;
}
}
if p != s {
break '__join_1;
} else {
*(*__slate_slot_25) = *__slate_slot_22;
}
}
last_mhp = *__slate_slot_25;
return 0;
}
while false {
}
_printk(c"\x014HugeTLB: architecture can't support node specific alloc, ignoring!\n".as_ptr() as *const u8);
return 1;
}
while false {
}
_printk(c"\x014HugeTLB: Invalid hugepages parameter %s\n".as_ptr() as *const u8, p);
hugepages_clear_pages_in_node();
return 22i32.wrapping_neg();
}
}
}
}

static mut last_mhp: *mut u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
extern "C" fn hugepages_setupargs(s: *mut u8) -> i32 {
hugetlb_add_param(s, Some(hugepages_setup))
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_hugepages_setupargs: [u8; 10] = [104, 117, 103, 101, 112, 97, 103, 101, 115, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_hugepages_setupargs: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_hugepages_setupargs) as *const u8 };
__t0.setup_func = Some(hugepages_setupargs);
__t0.early = 1i32;
__t0
};

/// hugepagesz command line processing
/// A specific huge page size can only be specified once with hugepagesz.
/// hugepagesz is followed by hugepages on the command line.  The global
/// variable 'parsed_valid_hugepagesz' is used to determine if prior
/// hugepagesz argument was valid.
#[unsafe(link_section = ".init.text")]
extern "C" fn hugepagesz_setup(s: *mut u8) -> i32 {
let mut size: u64 = 0;
if hugepages_supported!() == 0 {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014HugeTLB: hugepages unsupported, ignoring hugepagesz=%s cmdline\n".as_ptr() as *const u8, s) };
return 0;
}
unsafe {
parsed_valid_hugepagesz = false;
}
size = unsafe { memparse(s as *const u8, core::ptr::null_mut::<*mut u8>()) };
if !arch_hugetlb_valid_size(size) {
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x013HugeTLB: unsupported hugepagesz=%s\n".as_ptr() as *const u8, s) };
return 22i32.wrapping_neg();
}
let h: *mut hstate = size_to_hstate(size);
if !h.is_null() {
// hstate for this size already exists.  This is normally
// an error, but is allowed if the existing hstate is the
// default hstate.  More specifically, it is only allowed if
// the number of huge pages for the default hstate was not
// previously specified.
if !unsafe { parsed_default_hugepagesz } || h != unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { default_hstate_idx }) as isize) || (unsafe { hstates.0[default_hstate_idx as usize].max_huge_pages }) != 0 {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x014HugeTLB: hugepagesz=%s specified twice, ignoring\n".as_ptr() as *const u8, s) };
return 22i32.wrapping_neg();
}
// No need to call hugetlb_add_hstate() as hstate already
// exists.  But, do set parsed_hstate so that a following
// hugepages= parameter will be applied to this hstate.
unsafe {
parsed_hstate = h;
}
unsafe {
parsed_valid_hugepagesz = true;
}
return 0;
}
hugetlb_add_hstate(ilog2!(size).wrapping_sub(12) as u32);
unsafe {
parsed_valid_hugepagesz = true;
}
0
}

#[unsafe(link_section = ".init.text")]
extern "C" fn hugepagesz_setupargs(s: *mut u8) -> i32 {
hugetlb_add_param(s, Some(hugepagesz_setup))
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_hugepagesz_setupargs: [u8; 11] = [104, 117, 103, 101, 112, 97, 103, 101, 115, 122, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_hugepagesz_setupargs: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_hugepagesz_setupargs) as *const u8 };
__t0.setup_func = Some(hugepagesz_setupargs);
__t0.early = 1i32;
__t0
};

/// default_hugepagesz command line input
/// Only one instance of default_hugepagesz allowed on command line.
#[unsafe(link_section = ".init.text")]
extern "C" fn default_hugepagesz_setup(s: *mut u8) -> i32 {
let mut size: u64 = 0;
if hugepages_supported!() == 0 {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014HugeTLB: hugepages unsupported, ignoring default_hugepagesz=%s cmdline\n".as_ptr() as *const u8, s) };
return 0;
}
unsafe {
parsed_valid_hugepagesz = false;
}
if unsafe { parsed_default_hugepagesz } {
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x013HugeTLB: default_hugepagesz previously specified, ignoring %s\n".as_ptr() as *const u8, s) };
return 22i32.wrapping_neg();
}
size = unsafe { memparse(s as *const u8, core::ptr::null_mut::<*mut u8>()) };
if !arch_hugetlb_valid_size(size) {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013HugeTLB: unsupported default_hugepagesz=%s\n".as_ptr() as *const u8, s) };
return 22i32.wrapping_neg();
}
hugetlb_add_hstate(ilog2!(size).wrapping_sub(12) as u32);
unsafe {
parsed_valid_hugepagesz = true;
}
unsafe {
parsed_default_hugepagesz = true;
}
unsafe {
default_hstate_idx = hstate_index(size_to_hstate(size)) as u32;
}
// The number of default huge pages (for this size) could have been
// specified as the first hugetlb parameter: hugepages=X.  If so,
// then default_hstate_max_huge_pages is set.  If the default huge
// page size is gigantic (> MAX_PAGE_ORDER), then the pages must be
// allocated here from bootmem allocator.
if (unsafe { default_hstate_max_huge_pages }) != 0 {
unsafe {
hstates.0[default_hstate_idx as usize].max_huge_pages = default_hstate_max_huge_pages;
}
// Since this is an early parameter, we can't check
// NUMA node state yet, so loop through MAX_NUMNODES.
for i in 0..64i32 {
if (unsafe { default_hugepages_in_node.0[i as usize] }) != 0 {
unsafe {
hstates.0[default_hstate_idx as usize].max_huge_pages_node[i as usize] = default_hugepages_in_node.0[i as usize];
}
}
}
unsafe {
default_hstate_max_huge_pages = 0;
}
}
0
}

#[unsafe(link_section = ".init.text")]
extern "C" fn default_hugepagesz_setupargs(s: *mut u8) -> i32 {
hugetlb_add_param(s, Some(default_hugepagesz_setup))
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_default_hugepagesz_setupargs: [u8; 19] = [100, 101, 102, 97, 117, 108, 116, 95, 104, 117, 103, 101, 112, 97, 103, 101, 115, 122, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_default_hugepagesz_setupargs: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_default_hugepagesz_setupargs) as *const u8 };
__t0.setup_func = Some(default_hugepagesz_setupargs);
__t0.early = 1i32;
__t0
};

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn hugetlb_bootmem_set_nodes() {
let mut i: i32 = 0;
let mut nid: i32 = 0;
if !nodes_empty__2!(hugetlb_bootmem_nodes) {
return;
}
i = 1i32.wrapping_neg();
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::null_mut::<u64>(), core::ptr::null_mut::<u64>(), core::ptr::addr_of_mut!(nid)) };
while i >= 0 {
node_set!(nid, hugetlb_bootmem_nodes);
unsafe { __next_mem_pfn_range(core::ptr::addr_of_mut!(i), 64i32, core::ptr::null_mut::<u64>(), core::ptr::null_mut::<u64>(), core::ptr::addr_of_mut!(nid)) };
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn hugetlb_bootmem_alloc() {
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
hugetlb_bootmem_set_nodes();
for i in 0..64i32 {
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!(huge_boot_pages.0) as *mut list_head }.wrapping_offset(i as isize));
}
hugetlb_parse_params();
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
unsafe {
(*h).next_nid_to_alloc = first_node!(node_states[1]) as i32;
}
if hstate_is_gigantic(h) {
hugetlb_hstate_alloc_pages(h);
}
let __v4: *mut hstate = h;
h = __v4.wrapping_offset(1 as isize);
}
}

/// hugepage_alloc_threads command line parsing.
///
/// When set, use this specific number of threads for the boot
/// allocation of hugepages.
#[unsafe(link_section = ".init.text")]
extern "C" fn hugepage_alloc_threads_setup(s: *mut u8) -> i32 {
let mut allocation_threads: u64 = 0;
if kstrtoul(s as *const u8, 0u32, core::ptr::addr_of_mut!(allocation_threads)) != 0 {
return 1;
}
if allocation_threads == 0 {
return 1;
}
unsafe {
hugepage_allocation_threads = allocation_threads;
}
1
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_hugepage_alloc_threads_setup: [u8; 24] = [104, 117, 103, 101, 112, 97, 103, 101, 95, 97, 108, 108, 111, 99, 95, 116, 104, 114, 101, 97, 100, 115, 61, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_hugepage_alloc_threads_setup: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_hugepage_alloc_threads_setup) as *const u8 };
__t0.setup_func = Some(hugepage_alloc_threads_setup);
__t0.early = 0i32;
__t0
};

fn allowed_mems_nr(h: *mut hstate) -> u32 {
let mut node: i32 = 0;
let mut nr: u32 = 0;
let mut mbind_nodemask: *mut __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };
let array: *mut u32 = unsafe { (*h).free_huge_pages_node.as_mut_ptr() as *mut u32 };
let gfp_mask: u32 = htlb_alloc_mask(h);
mbind_nodemask = policy_mbind_nodemask(gfp_mask);
node = first_node!((*get_current()).mems_allowed) as i32;
while node < 64i32 {
let __v0: bool = mbind_nodemask.is_null() || node_isset__2!(node, *mbind_nodemask) != 0;
if __v0 {
nr = nr.wrapping_add(unsafe { *array.wrapping_offset(node as isize) });
}
node = next_node!(node, (*get_current()).mems_allowed) as i32;
}
nr
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_report_meminfo(m: *mut seq_file) {
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut total: u64 = 0;
if hugepages_supported!() == 0 {
return;
}
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
let count: u64 = unsafe { (*h).nr_huge_pages };
let __v1: u64 = huge_page_size(h as *const hstate).wrapping_mul(count);
total = total.wrapping_add(__v1);
if h == unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { default_hstate_idx }) as isize) {
unsafe { seq_printf(m, c"HugePages_Total:   %5lu\nHugePages_Free:    %5lu\nHugePages_Rsvd:    %5lu\nHugePages_Surp:    %5lu\nHugepagesize:   %8lu kB\n".as_ptr() as *const u8, count, (*h).free_huge_pages, (*h).resv_huge_pages, (*h).surplus_huge_pages, huge_page_size(h as *const hstate) / 1024) };
}
let __v5: *mut hstate = h;
h = __v5.wrapping_offset(1 as isize);
}
unsafe { seq_printf(m, c"Hugetlb:        %8lu kB\n".as_ptr() as *const u8, total / 1024) };
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_report_node_meminfo(buf: *mut u8, len: i32, nid: i32) -> i32 {
let h: *mut hstate = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { default_hstate_idx }) as isize);
if hugepages_supported!() == 0 {
return 0;
}
unsafe { sysfs_emit_at(buf, len, c"Node %d HugePages_Total: %5u\nNode %d HugePages_Free:  %5u\nNode %d HugePages_Surp:  %5u\n".as_ptr() as *const u8, nid, *((*h).nr_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize), nid, *((*h).free_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize), nid, *((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize)) }
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_show_meminfo_node(nid: i32) {
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
if hugepages_supported!() == 0 {
return;
}
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"Node %d hugepages_total=%u hugepages_free=%u hugepages_surp=%u hugepages_size=%lukB\n".as_ptr() as *const u8, nid, *((*h).nr_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize), *((*h).free_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize), *((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(nid as isize), huge_page_size(h as *const hstate) / 1024) };
let __v4: *mut hstate = h;
h = __v4.wrapping_offset(1 as isize);
}
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_report_usage(m: *mut seq_file, mm: *mut mm_struct) {
unsafe { seq_printf(m, c"HugetlbPages:\t%8lu kB\n".as_ptr() as *const u8, K!(atomic_long_read(core::ptr::addr_of_mut!((*mm).__slate_anon_0.hugetlb_usage) as *const __SlateRecord_types_h_197))) };
}

/// Return the number pages of memory we physically have, in PAGE_SIZE units.
#[unsafe(no_mangle)]
extern "C" fn hugetlb_total_pages() -> u64 {
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut nr_total_pages: u64 = 0;
h = unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate };
while h < unsafe { core::ptr::addr_of_mut!(hstates.0) as *mut hstate }.wrapping_offset((unsafe { hugetlb_max_hstate }) as isize) {
let __v0: u64 = unsafe { (*h).nr_huge_pages }.wrapping_mul(pages_per_huge_page(h as *const hstate) as u64);
nr_total_pages = nr_total_pages.wrapping_add(__v0);
let __v4: *mut hstate = h;
h = __v4.wrapping_offset(1 as isize);
}
nr_total_pages
}

/// Forward declaration
fn hugetlb_acct_memory(h: *mut hstate, delta: i64) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
unsafe {
ret = 12i32.wrapping_neg();
if delta == 0 {
return 0;
} else {
'__join_0: {
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
// When cpuset is configured, it breaks the strict hugetlb page
// reservation as the accounting is done on a global variable. Such
// reservation is completely rubbish in the presence of cpuset because
// the reservation is not checked against page availability for the
// current cpuset. Application can still potentially OOM'ed by kernel
// with lack of free htlb page in cpuset that the task is in.
// Attempt to enforce strict accounting with cpuset is almost
// impossible (or too ugly) because cpuset is too fluid that
// task or memory node can be dynamically moved between cpusets.
//
// The change of semantics for shared hugetlb mapping with cpuset is
// undesirable. However, in order to preserve some of the semantics,
// we fall back to check against current free page availability as
// a best attempt and hopefully to minimize the impact of changing
// semantics that cpuset has.
//
// Apart from cpuset, we also have memory policy mechanism that
// also determines from which node the kernel will allocate memory
// in a NUMA system. So similar to cpuset, we also should consider
// the memory policy of the current task. Similar to the description
// above.
if delta > 0 {
if gather_surplus_pages(h, delta) < 0 {
break '__join_0;
} else {
if delta > (allowed_mems_nr(h) as i64) {
return_unused_surplus_pages(h, delta as u64);
break '__join_0;
}
}
}
ret = 0;
if delta < 0 {
return_unused_surplus_pages(h, delta.wrapping_neg() as u64);
}
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
return ret;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.hugetlb.hugetlb_vm_op_open")]
extern "C" fn hugetlb_vm_op_open(vma: *mut vm_area_struct) {
let resv: *mut resv_map = vma_resv_map(vma);
// HPAGE_RESV_OWNER indicates a private mapping.
// This new VMA should share its siblings reservation map if present.
// The VMA will only ever have a valid reservation map pointer where
// it is being copied for another still existing VMA.  As that VMA
// has a reference to the reservation map it cannot disappear until
// after this open call completes.  It is therefore safe to take a
// new reference here without additional locking.
let __v0: bool = !resv.is_null() && is_vma_resv_set(vma, 1u64) != 0;
if __v0 {
resv_map_dup_hugetlb_cgroup_uncharge_info(resv);
kref_get(unsafe { core::ptr::addr_of_mut!((*resv).refs) });
}
// vma_lock structure for sharable mappings is vma specific.
// Clear old pointer (if copied via vm_area_dup) and allocate
// new structure.  Before clearing, make sure vma_lock is not
// for this vma.
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) != 0 {
let vma_lock: *mut hugetlb_vma_lock = (unsafe { (*vma).vm_private_data }) as *mut hugetlb_vma_lock;
if !vma_lock.is_null() {
if (unsafe { (*vma_lock).vma }) != vma {
unsafe {
(*vma).vm_private_data = core::ptr::null_mut::<()>();
}
hugetlb_vma_lock_alloc(vma);
} else {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014HugeTLB: vma_lock already exists in %s.\n".as_ptr() as *const u8, c"hugetlb_vm_op_open".as_ptr() as *mut u8) };
}
} else {
hugetlb_vma_lock_alloc(vma);
}
}
}

#[unsafe(link_section = ".text.slate_distinct.hugetlb.hugetlb_vm_op_close")]
extern "C" fn hugetlb_vm_op_close(vma: *mut vm_area_struct) {
let h: *mut hstate = hstate_vma(vma);
let mut resv: *mut resv_map = unsafe { core::mem::zeroed() };
let spool: *mut hugepage_subpool = subpool_vma(vma);
let mut reserve: u64 = 0;
let mut start: u64 = 0;
let mut end: u64 = 0;
let mut gbl_reserve: i64 = 0;
hugetlb_vma_lock_free(vma);
resv = vma_resv_map(vma);
let __v0: bool = resv.is_null() || is_vma_resv_set(vma, 1u64) == 0;
if __v0 {
return;
}
start = vma_hugecache_offset(h, vma, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start });
end = vma_hugecache_offset(h, vma, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end });
reserve = end.wrapping_sub(start).wrapping_sub(region_count(resv, start as i64, end as i64) as u64);
unsafe { hugetlb_cgroup_uncharge_counter(resv, start, end) };
if reserve != 0 {
// Decrement reserve counts.  The global reserve count may be
// adjusted if the subpool has a minimum size.
gbl_reserve = hugepage_subpool_put_pages(spool, reserve as i64);
hugetlb_acct_memory(h, gbl_reserve.wrapping_neg());
}
kref_put(unsafe { core::ptr::addr_of_mut!((*resv).refs) }, Some(resv_map_release));
}

#[unsafe(link_section = ".text.slate_distinct.hugetlb.hugetlb_vm_op_split")]
extern "C" fn hugetlb_vm_op_split(vma: *mut vm_area_struct, addr: u64) -> i32 {
if addr & !huge_page_mask(hstate_vma(vma)) != 0 {
return 22i32.wrapping_neg();
}
0
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_split(vma: *mut vm_area_struct, addr: u64) {
// PMD sharing is only possible for PUD_SIZE-aligned address ranges
// in HugeTLB VMAs. If we will lose PUD_SIZE alignment due to this
// split, unshare PMDs in the PUD_SIZE interval surrounding addr now.
// This function is called in the middle of a VMA split operation, with
// MM, VMA and rmap all write-locked to prevent concurrent page table
// walks (except hardware and gup_fast()).
vma_assert_write_locked(vma);
i_mmap_assert_write_locked(unsafe { (*(*vma).vm_file).f_mapping });
if addr & !(!(_AC!() << 30i32).wrapping_sub(1)) != 0 {
let floor: u64 = addr & !(_AC!() << 30i32).wrapping_sub(1);
let ceil: u64 = floor.wrapping_add(_AC!() << 30i32);
if floor >= unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start } && ceil <= unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end } {
// Locking:
// Use take_locks=false here.
// The file rmap lock is already held.
// The hugetlb VMA lock can't be taken when we already
// hold the file rmap lock, and we don't need it because
// its purpose is to synchronize against concurrent page
// table walks, which are not possible thanks to the
// locks held by our caller.
hugetlb_unshare_pmds(vma, floor, ceil, false); // take_locks =
}
}
}

#[unsafe(link_section = ".text.slate_distinct.hugetlb.hugetlb_vm_op_pagesize")]
extern "C" fn hugetlb_vm_op_pagesize(vma: *mut vm_area_struct) -> u64 {
huge_page_size(hstate_vma(vma) as *const hstate)
}

/// We cannot handle pagefaults against hugetlb pages at all.  They cause
/// handle_mm_fault() to try to instantiate regular-sized pages in the
/// hugepage VMA.  do_page_fault() is supposed to trap this, so BUG is we get
/// this far.
#[unsafe(link_section = ".text.slate_distinct.hugetlb.hugetlb_vm_op_fault")]
extern "C" fn hugetlb_vm_op_fault(vmf: *mut vm_fault) -> u32 {
loop {
unsafe {
core::arch::asm!("734: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 734b - ., 3; .popsection\n# {0}", const 734 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_hugetlb_vm_op_fault_0, sym __slate_asm_string_hugetlb_vm_op_fault_1, const 4804 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
0
}

/// When a new function is introduced to vm_operations_struct and added
/// to hugetlb_vm_ops, please consider adding the function to shm_vm_ops.
/// This is because under System V memory model, mappings created via
/// shmget/shmat with "huge page" specified are backed by hugetlbfs files,
/// their original vm_ops are overwritten with shm_vm_ops.
#[unsafe(no_mangle)]
static hugetlb_vm_ops: __SlateConst<vm_operations_struct> = __SlateConst({
let mut __t0: vm_operations_struct = unsafe { core::mem::zeroed() };
__t0.open = Some(hugetlb_vm_op_open);
__t0.close = Some(hugetlb_vm_op_close);
__t0.may_split = Some(hugetlb_vm_op_split);
__t0.fault = Some(hugetlb_vm_op_fault);
__t0.pagesize = Some(hugetlb_vm_op_pagesize);
__t0
});

fn make_huge_pte(vma: *mut vm_area_struct, folio: *mut folio, try_mkwrite: bool) -> __SlateRecord_pgtable_64_types_h_21 {
let mut entry: __SlateRecord_pgtable_64_types_h_21 = folio_mk_pte(folio as *const folio, unsafe { (*vma).vm_page_prot });
let shift: u32 = huge_page_shift(hstate_vma(vma));
if try_mkwrite && (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(1i32) != 0 {
entry = pte_mkwrite_novma(pte_mkdirty(entry));
} else {
entry = pte_wrprotect(entry);
}
entry = pte_mkyoung(entry);
entry = arch_make_huge_pte(entry, shift, unsafe { (*vma).__slate_anon_3.vm_flags });
return entry;
}

fn set_huge_ptep_writable(vma: *mut vm_area_struct, address: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) {
let mut entry: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
entry = huge_pte_mkwrite(huge_pte_mkdirty(huge_ptep_get(unsafe { (*vma).vm_mm }, address, ptep)));
if huge_ptep_set_access_flags(vma, address, ptep, entry, 1i32) != 0 {
update_mmu_cache(vma, address, ptep);
}
}

fn set_huge_ptep_maybe_writable(vma: *mut vm_area_struct, address: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) {
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(1i32) != 0 {
set_huge_ptep_writable(vma, address, ptep);
}
}

fn hugetlb_install_folio(vma: *mut vm_area_struct, ptep: *mut __SlateRecord_pgtable_64_types_h_21, addr: u64, new_folio: *mut folio, old: __SlateRecord_pgtable_64_types_h_21, sz: u64) {
let mut __slate_compound_1: pgprot = unsafe { core::mem::zeroed() };
let mut newpte: __SlateRecord_pgtable_64_types_h_21 = make_huge_pte(vma, new_folio, true);
__folio_mark_uptodate(new_folio);
unsafe { hugetlb_add_new_anon_rmap(new_folio, vma, addr) };
let __v0: bool = userfaultfd_protected(vma) && huge_pte_uffd(old) != 0;
if __v0 {
newpte = huge_pte_mkuffd(newpte);
// Restore PAGE_NONE so the RWP marker keeps trapping.
if userfaultfd_rwp(vma) {
let shift: u32 = huge_page_shift(hstate_vma(vma));
newpte = huge_pte_modify(newpte, unsafe { *{
__slate_compound_1 = pgprot { pgprot: 0u64 | _AT!(u64, 1i32) << 5i32 | 0 | 0 | 0 | _AT!(u64, 1i32) << 8i32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
newpte = arch_make_huge_pte(newpte, shift, unsafe { (*vma).__slate_anon_3.vm_flags });
}
}
set_huge_pte_at(unsafe { (*vma).vm_mm }, addr, ptep, newpte, sz);
hugetlb_count_add(pages_per_huge_page(hstate_vma(vma) as *const hstate) as i64, unsafe { (*vma).vm_mm });
folio_set_hugetlb_migratable(new_folio);
}

#[unsafe(no_mangle)]
extern "C" fn copy_hugetlb_page_range(dst: *mut mm_struct, src: *mut mm_struct, dst_vma: *mut vm_area_struct, src_vma: *mut vm_area_struct) -> i32 {
let mut uffd: bool = unsafe { core::mem::zeroed() };
let mut marker: u64 = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut new_folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut folio = core::ptr::addr_of_mut!(new_folio) as *mut *mut folio;
let mut src_pte_old: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(src_pte_old) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut dst_ptl: core::mem::MaybeUninit<*mut spinlock> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut spinlock = core::ptr::addr_of_mut!(dst_ptl) as *mut *mut spinlock;
let mut src_ptl: core::mem::MaybeUninit<*mut spinlock> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut spinlock = core::ptr::addr_of_mut!(src_ptl) as *mut *mut spinlock;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut softleaf: core::mem::MaybeUninit<__SlateRecord_mm_types_h_288> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut __SlateRecord_mm_types_h_288 = core::ptr::addr_of_mut!(softleaf) as *mut __SlateRecord_mm_types_h_288;
let mut last_addr_mask: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u64 = core::ptr::addr_of_mut!(last_addr_mask) as *mut u64;
let mut range: core::mem::MaybeUninit<mmu_notifier_range> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut mmu_notifier_range = core::ptr::addr_of_mut!(range) as *mut mmu_notifier_range;
let mut npages: u64 = unsafe { core::mem::zeroed() };
let mut sz: u64 = unsafe { core::mem::zeroed() };
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut cow: bool = unsafe { core::mem::zeroed() };
let mut addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u64 = core::ptr::addr_of_mut!(addr) as *mut u64;
let mut pte_folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut folio = core::ptr::addr_of_mut!(pte_folio) as *mut *mut folio;
let mut entry: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(entry) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut dst_pte: core::mem::MaybeUninit<*mut __SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(dst_pte) as *mut *mut __SlateRecord_pgtable_64_types_h_21;
let mut src_pte: core::mem::MaybeUninit<*mut __SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(src_pte) as *mut *mut __SlateRecord_pgtable_64_types_h_21;
unsafe {
{
let __slate_auto_bytes_3 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
cow = vma_is_cow_mapping(src_vma as *const vm_area_struct);
h = hstate_vma(src_vma);
sz = huge_page_size(h as *const hstate);
npages = pages_per_huge_page(h as *const hstate) as u64;
let __slate_auto_bytes_8 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 40usize);
let __slate_auto_bytes_9 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
if cow {
mmu_notifier_range_init(&mut (*core::ptr::addr_of_mut!(*__slate_slot_16)), 1u32, 0u32, src, (*src_vma).__slate_anon_0.__slate_anon_0.vm_start, (*src_vma).__slate_anon_0.__slate_anon_0.vm_end);
mmu_notifier_invalidate_range_start(core::ptr::addr_of_mut!(*__slate_slot_16));
vma_assert_write_locked(src_vma);
loop {
if seqprop_preemptible!(core::ptr::addr_of_mut!((*src).__slate_anon_0.write_protect_seq)) {
loop {
preempt_count_inc!();
core::arch::asm!("", options(nostack, att_syntax));
if true {
break;
}
}
}
do_raw_write_seqcount_begin(seqprop_ptr!(core::ptr::addr_of_mut!((*src).__slate_anon_0.write_protect_seq)));
if true {
break;
}
}
} else {
// For shared mappings the vma lock must be held before
// calling hugetlb_walk() in the src vma. Otherwise, the
// returned ptep could go away if part of a shared pmd and
// another thread calls huge_pmd_unshare.
hugetlb_vma_lock_read(src_vma);
}
}
*__slate_slot_15 = hugetlb_mask_last_page(h);
*__slate_slot_21 = (*src_vma).__slate_anon_0.__slate_anon_0.vm_start;
'__join_8: {
'__join_50: {
'__join_38: {
'__loop_9: loop {
if *__slate_slot_21 < (*src_vma).__slate_anon_0.__slate_anon_0.vm_end {
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
*__slate_slot_25 = hugetlb_walk(src_vma, *__slate_slot_21, sz);
if (*__slate_slot_25).is_null() {
*__slate_slot_21 |= *__slate_slot_15;
} else {
*__slate_slot_24 = huge_pte_alloc(dst, dst_vma, *__slate_slot_21, sz);
if (*__slate_slot_24).is_null() {
break '__join_50;
} else {
// If the pagetables are shared, there is nothing to do
if ptdesc_pmd_is_shared(virt_to_ptdesc(*__slate_slot_24 as *const ())) {
*__slate_slot_21 |= *__slate_slot_15;
} else {
*__slate_slot_11 = huge_pte_lock(h, dst, *__slate_slot_24);
*__slate_slot_12 = huge_pte_lockptr(h, src, *__slate_slot_25);
spin_lock_nested!(*__slate_slot_12, 1i32);
*__slate_slot_23 = huge_ptep_get((*src_vma).vm_mm, *__slate_slot_21, *__slate_slot_25);
'__join_11: {
'__join_45: {
'__join_12: {
'__join_19: {
'__join_21: {
'__join_31: {
'__join_32: {
loop {
if huge_pte_none(*__slate_slot_23) != 0 {
break '__join_45;
} else {
*__slate_slot_14 = softleaf_from_pte(*__slate_slot_23);
if unlikely!(softleaf_is_hwpoison(*__slate_slot_14)) != 0 {
break '__join_12;
} else {
if unlikely!(softleaf_is_migration(*__slate_slot_14)) != 0 {
break '__join_19;
} else {
if unlikely!(pte_is_marker(*__slate_slot_23)) != 0 {
break '__join_21;
} else {
*__slate_slot_23 = huge_ptep_get((*src_vma).vm_mm, *__slate_slot_21, *__slate_slot_25);
*__slate_slot_22 = page_folio!((vmemmap_base as *mut page).wrapping_offset(pte_pfn(*__slate_slot_23) as isize));
folio_get(*__slate_slot_22);
// Failing to duplicate the anon rmap is a rare case
// where we see pinned hugetlb pages while they're
// prone to COW. We need to do the COW earlier during
// fork.
//
// When pre-allocating the page or copying data, we
// need to be without the pgtable locks since we could
// sleep during the process.
if !folio_test_anon(*__slate_slot_22 as *const folio) {
break '__join_32;
} else {
if hugetlb_try_dup_anon_rmap(*__slate_slot_22, src_vma) != 0 {
core::ptr::write(__slate_slot_6, *__slate_slot_23);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_unlock(*__slate_slot_12);
spin_unlock(*__slate_slot_11);
// Do not use reserve as it's private owned
*__slate_slot_5 = alloc_hugetlb_folio(dst_vma, *__slate_slot_21, false);
if IS_ERR(*__slate_slot_5 as *const ()) {
break '__join_38;
} else {
ret = copy_user_large_folio(*__slate_slot_5, *__slate_slot_22, *__slate_slot_21, dst_vma);
folio_put(*__slate_slot_22);
if ret != 0 {
break '__loop_9;
} else {
// Install the new hugetlb folio if src pte stable
*__slate_slot_11 = huge_pte_lock(h, dst, *__slate_slot_24);
*__slate_slot_12 = huge_pte_lockptr(h, src, *__slate_slot_25);
spin_lock_nested!(*__slate_slot_12, 1i32);
*__slate_slot_23 = huge_ptep_get((*src_vma).vm_mm, *__slate_slot_21, *__slate_slot_25);
if pte_same(*__slate_slot_6, *__slate_slot_23) == 0 {
restore_reserve_on_error(h, dst_vma, *__slate_slot_21, *__slate_slot_5);
folio_put(*__slate_slot_5);
// huge_ptep of dst_pte won't change as in child
} else {
break;
}
}
}
} else {
break '__join_31;
}
}
}
}
}
}
}
hugetlb_install_folio(dst_vma, *__slate_slot_24, *__slate_slot_21, *__slate_slot_5, *__slate_slot_6, sz);
break '__join_11;
}
hugetlb_add_file_rmap(*__slate_slot_22);
}
// See __copy_present_ptes(): restore accessible protection.
if !userfaultfd_protected(dst_vma) {
if userfaultfd_rwp(src_vma) {
__v4 = huge_pte_uffd(*__slate_slot_23) != 0;
} else {
__v4 = false;
}
if __v4 {
*__slate_slot_23 = huge_pte_modify(*__slate_slot_23, (*dst_vma).vm_page_prot);
*__slate_slot_23 = arch_make_huge_pte(*__slate_slot_23, huge_page_shift(h), (*dst_vma).__slate_anon_3.vm_flags);
}
*__slate_slot_23 = huge_pte_clear_uffd(*__slate_slot_23);
}
if cow {
// No need to notify as we are downgrading page
// table protection not changing it to point
// to a new page.
//
// See Documentation/mm/mmu_notifier.rst
huge_ptep_set_wrprotect(src, *__slate_slot_21, *__slate_slot_25);
*__slate_slot_23 = huge_pte_wrprotect(*__slate_slot_23);
}
set_huge_pte_at(dst, *__slate_slot_21, *__slate_slot_24, *__slate_slot_23, sz);
hugetlb_count_add(npages as i64, dst);
break '__join_11;
}
marker = copy_pte_marker(*__slate_slot_14, dst_vma);
if marker != 0 {
set_huge_pte_at(dst, *__slate_slot_21, *__slate_slot_24, make_pte_marker(marker), sz);
break '__join_11;
} else {
break '__join_11;
}
}
uffd = pte_swp_uffd(*__slate_slot_23) != 0;
if !softleaf_is_migration_read(*__slate_slot_14) && cow {
// COW mappings require pages in both
// parent and child to be set to read.
*__slate_slot_14 = make_readable_migration_entry(swp_offset(*__slate_slot_14));
*__slate_slot_23 = swp_entry_to_pte(*__slate_slot_14);
if userfaultfd_protected(src_vma) && uffd {
*__slate_slot_23 = pte_swp_mkuffd(*__slate_slot_23);
}
set_huge_pte_at(src, *__slate_slot_21, *__slate_slot_25, *__slate_slot_23, sz);
}
if !userfaultfd_protected(dst_vma) {
*__slate_slot_23 = pte_swp_clear_uffd(*__slate_slot_23);
}
set_huge_pte_at(dst, *__slate_slot_21, *__slate_slot_24, *__slate_slot_23, sz);
break '__join_11;
}
// A hwpoison entry never carries the uffd-wp bit: it is
// installed fresh by make_hwpoison_entry() and
// hugetlb_change_protection() leaves it untouched, so
// there is nothing to clear for the child.
set_huge_pte_at(dst, *__slate_slot_21, *__slate_slot_24, *__slate_slot_23, sz);
break '__join_11;
}
// Skip if src entry none.
}
spin_unlock(*__slate_slot_12);
spin_unlock(*__slate_slot_11);
}
}
}
let __v1: u64 = *__slate_slot_21;
*__slate_slot_21 = __v1.wrapping_add(sz);
} else {
break '__join_8;
}
}
restore_reserve_on_error(h, dst_vma, *__slate_slot_21, *__slate_slot_5);
folio_put(*__slate_slot_5);
break '__join_8;
}
folio_put(*__slate_slot_22);
ret = PTR_ERR(*__slate_slot_5 as *const ()) as i32;
break '__join_8;
}
ret = 12i32.wrapping_neg();
}
if cow {
loop {
do_raw_write_seqcount_end(seqprop_ptr!(core::ptr::addr_of_mut!((*src).__slate_anon_0.write_protect_seq)));
if seqprop_preemptible!(core::ptr::addr_of_mut!((*src).__slate_anon_0.write_protect_seq)) {
loop {
core::arch::asm!("", options(nostack, att_syntax));
if unlikely!(preempt_count_dec_and_test!()) != 0 {
loop {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
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
if true {
break;
}
}
mmu_notifier_invalidate_range_end(core::ptr::addr_of_mut!(*__slate_slot_16));
} else {
hugetlb_vma_unlock_read(src_vma);
}
return ret;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_735: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

fn move_huge_pte(vma: *mut vm_area_struct, old_addr: u64, new_addr: u64, src_pte: *mut __SlateRecord_pgtable_64_types_h_21, dst_pte: *mut __SlateRecord_pgtable_64_types_h_21, sz: u64) {
let need_clear_uffd_wp: bool = vma_has_uffd_without_event_remap(vma);
let h: *mut hstate = hstate_vma(vma);
let mm: *mut mm_struct = unsafe { (*vma).vm_mm };
let mut pte: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(pte) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let dst_ptl: *mut spinlock = huge_pte_lock(h, mm, dst_pte);
let src_ptl: *mut spinlock = huge_pte_lockptr(h, mm, src_pte);
// We don't have to worry about the ordering of src and dst ptlocks
// because exclusive mmap_lock (or the i_mmap_lock) prevents deadlock.
if src_ptl != dst_ptl {
spin_lock_nested!(src_ptl, 1i32);
}
pte = huge_ptep_get_and_clear(mm, old_addr, src_pte, sz);
let __v3: bool = need_clear_uffd_wp && pte_is_uffd_wp_marker(pte);
if __v3 {
huge_pte_clear(mm, new_addr, dst_pte, sz);
} else {
if need_clear_uffd_wp {
if pte_present(pte) != 0 {
// See __copy_present_ptes(): normalise the RWP
// marker so the destination starts accessible
// instead of taking a numa-hinting fault on
// first access. Only the marker (protnone + uffd)
// needs it; leave other present PTEs untouched.
let __v4: bool = userfaultfd_rwp(vma) && huge_pte_uffd(pte) != 0;
if __v4 {
pte = huge_pte_modify(pte, unsafe { (*vma).vm_page_prot });
pte = arch_make_huge_pte(pte, huge_page_shift(h), unsafe { (*vma).__slate_anon_3.vm_flags });
}
pte = huge_pte_clear_uffd(pte);
} else {
pte = pte_swp_clear_uffd(pte);
}
}
set_huge_pte_at(mm, new_addr, dst_pte, pte, sz);
}
if src_ptl != dst_ptl {
spin_unlock(src_ptl);
}
spin_unlock(dst_ptl);
}

#[unsafe(no_mangle)]
extern "C" fn move_hugetlb_page_tables(vma: *mut vm_area_struct, new_vma: *mut vm_area_struct, mut old_addr: u64, mut new_addr: u64, len: u64) -> i32 {
let h: *mut hstate = hstate_vma(vma);
let mapping: *mut address_space = unsafe { (*(*vma).vm_file).f_mapping };
let sz: u64 = huge_page_size(h as *const hstate);
let mm: *mut mm_struct = unsafe { (*vma).vm_mm };
let old_end: u64 = old_addr.wrapping_add(len);
let mut last_addr_mask: u64 = 0;
let mut range: mmu_notifier_range = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(range) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut tlb: mmu_gather = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tlb) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 128usize);
}
mmu_notifier_range_init(unsafe { &mut (*core::ptr::addr_of_mut!(range)) }, 1u32, 0u32, mm, old_addr, old_end);
adjust_range_if_pmd_sharing_possible(vma, core::ptr::addr_of_mut!(range.start), core::ptr::addr_of_mut!(range.end));
// In case of shared PMDs, we should cover the maximum possible
// range.
flush_cache_range(vma, range.start, range.end);
unsafe { tlb_gather_mmu_vma(core::ptr::addr_of_mut!(tlb), vma) };
mmu_notifier_invalidate_range_start(core::ptr::addr_of_mut!(range));
last_addr_mask = hugetlb_mask_last_page(h);
// Prevent race with file truncation
hugetlb_vma_lock_write(vma);
i_mmap_lock_write(mapping);
while old_addr < old_end {
let offset_to_last_entry: u64 = (old_addr | last_addr_mask).wrapping_sub(old_addr);
let src_pte: *mut __SlateRecord_pgtable_64_types_h_21 = hugetlb_walk(vma, old_addr, sz);
if src_pte.is_null() {
old_addr = old_addr.wrapping_add(offset_to_last_entry);
new_addr = new_addr.wrapping_add(offset_to_last_entry);
} else {
if huge_pte_none(huge_ptep_get(mm, old_addr, src_pte)) == 0 {
if huge_pmd_unshare(core::ptr::addr_of_mut!(tlb), vma, old_addr, src_pte) != 0 {
old_addr = old_addr.wrapping_add(offset_to_last_entry);
new_addr = new_addr.wrapping_add(offset_to_last_entry);
} else {
let dst_pte: *mut __SlateRecord_pgtable_64_types_h_21 = huge_pte_alloc(mm, new_vma, new_addr, sz);
if dst_pte.is_null() {
break;
}
move_huge_pte(vma, old_addr, new_addr, src_pte, dst_pte, sz);
tlb_remove_huge_tlb_entry!(h, core::ptr::addr_of_mut!(tlb), src_pte, old_addr);
}
}
}
old_addr = old_addr.wrapping_add(sz);
new_addr = new_addr.wrapping_add(sz);
}
tlb_flush_mmu_tlbonly(core::ptr::addr_of_mut!(tlb));
huge_pmd_unshare_flush(core::ptr::addr_of_mut!(tlb), vma);
mmu_notifier_invalidate_range_end(core::ptr::addr_of_mut!(range));
i_mmap_unlock_write(mapping);
hugetlb_vma_unlock_write(vma);
unsafe { tlb_finish_mmu(core::ptr::addr_of_mut!(tlb)) };
len.wrapping_add(old_addr).wrapping_sub(old_end) as i32
}

#[unsafe(no_mangle)]
extern "C" fn __unmap_hugepage_range(tlb: *mut mmu_gather, vma: *mut vm_area_struct, start: u64, end: u64, mut folio: *mut folio, zap_flags: u32) {
let mm: *mut mm_struct = unsafe { (*vma).vm_mm };
let folio_provided: bool = !folio.is_null();
let mut address: u64 = 0;
let mut pte: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(pte) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let h: *mut hstate = hstate_vma(vma);
let sz: u64 = huge_page_size(h as *const hstate);
let mut adjust_reservation: bool = false;
let mut last_addr_mask: u64 = 0;
i_mmap_assert_write_locked(unsafe { (*(*vma).vm_file).f_mapping });
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!is_vm_hugetlb_page(vma as *const vm_area_struct))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("736: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 736b - ., 3; .popsection\n# {0}", const 736 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___unmap_hugepage_range_1, sym __slate_asm_string___unmap_hugepage_range_2, const 5226 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v0 = (__ret_warn_on != 0) as i64;
loop {
if unlikely__4!(start & !huge_page_mask(h)) != 0 {
loop {
unsafe {
core::arch::asm!("738: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 738b - ., 3; .popsection\n# {0}", const 738 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___unmap_hugepage_range_5, sym __slate_asm_string___unmap_hugepage_range_6, const 5227 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
loop {
if unlikely__4!(end & !huge_page_mask(h)) != 0 {
loop {
unsafe {
core::arch::asm!("739: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 739b - ., 3; .popsection\n# {0}", const 739 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___unmap_hugepage_range_10, sym __slate_asm_string___unmap_hugepage_range_11, const 5228 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
// This is a hugetlb vma, all the pte entries should point
// to huge page.
tlb_change_page_size(tlb, sz as u32);
tlb_start_vma(tlb, vma);
last_addr_mask = hugetlb_mask_last_page(h);
address = start;
'__slate_break_17: while address < end {
'__slate_continue_17: {
let ptep: *mut __SlateRecord_pgtable_64_types_h_21 = hugetlb_walk(vma, address, sz);
if ptep.is_null() {
address |= last_addr_mask;
} else {
let ptl: *mut spinlock = huge_pte_lock(h, mm, ptep);
if huge_pmd_unshare(tlb, vma, address, ptep) != 0 {
spin_unlock(ptl);
address |= last_addr_mask;
} else {
pte = huge_ptep_get(mm, address, ptep);
if huge_pte_none(pte) != 0 {
spin_unlock(ptl);
} else {
// Migrating hugepage or HWPoisoned hugepage is already
// unmapped and its refcount is dropped, so just clear pte here.
if unlikely!(pte_present(pte) == 0) != 0 {
// If the pte was wr-protected by uffd-wp in any of the
// swap forms, meanwhile the caller does not want to
// drop the uffd-wp bit in this zap, then replace the
// pte with a marker.
if pte_swp_uffd_any(pte) && zap_flags & (BIT!(0i32) as u32) == 0 {
set_huge_pte_at(mm, address, ptep, make_pte_marker(BIT!(0i32)), sz);
} else {
huge_pte_clear(mm, address, ptep, sz);
}
spin_unlock(ptl);
} else {
// If a folio is supplied, it is because a specific
// folio is being unmapped, not a range. Ensure the folio we
// are about to unmap is the actual folio of interest.
if folio_provided {
if folio != page_folio!(((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pte_pfn(pte) as isize)) {
spin_unlock(ptl);
break '__slate_continue_17;
}
// Mark the VMA as having unmapped its page so that
// future faults in this VMA will fail rather than
// looking like data was lost
set_vma_resv_flags(vma, 2u64);
} else {
folio = page_folio!(((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pte_pfn(pte) as isize));
}
pte = huge_ptep_get_and_clear(mm, address, ptep, sz);
tlb_remove_huge_tlb_entry!(h, tlb, ptep, address);
if huge_pte_dirty(pte) != 0 {
unsafe { folio_mark_dirty(folio) };
}
// Leave a uffd-wp pte marker if needed
if huge_pte_uffd(pte) != 0 && zap_flags & (BIT!(0i32) as u32) == 0 {
set_huge_pte_at(mm, address, ptep, make_pte_marker(BIT!(0i32)), sz);
}
hugetlb_count_sub(pages_per_huge_page(h as *const hstate) as i64, mm);
hugetlb_remove_rmap(folio);
spin_unlock(ptl);
// Restore the reservation for anonymous page, otherwise the
// backing page could be stolen by someone. Restore only on the
// last unmap, otherwise the owner could empty its resv map
// while the folio is still mapped by a child. Note that holding
// i_mmap_lock_write is needed to check the number of mappings.
// If there we are freeing a surplus, do not set the restore
// reservation bit.
adjust_reservation = false;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
let __v21: bool = (unsafe { (*h).surplus_huge_pages }) == 0 && __vma_private_lock(vma);
let __v22: bool = __v21 && !folio_mapped(folio as *const folio);
let __v23: bool = __v22 && folio_test_anon(folio as *const folio);
if __v23 {
folio_set_hugetlb_restore_reserve(folio);
// Reservation to be adjusted after the spin lock
adjust_reservation = true;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
// Adjust the reservation for the region that will have the
// reserve restored. Keep in mind that vma_needs_reservation() changes
// resv->adds_in_progress if it succeeds. If this is not done,
// do_exit() will not see it, and will keep the reservation
// forever.
if adjust_reservation {
let rc: i32 = vma_needs_reservation(h, vma, address) as i32;
if rc < 0 {
folio_clear_hugetlb_restore_reserve(folio);
} else {
if rc != 0 {
vma_add_reservation(h, vma, address);
}
}
// Pressumably allocate_file_region_entries failed
// to allocate a file_region struct. Clear
// hugetlb_restore_reserve so that global reserve
// count will not be incremented by free_huge_folio.
// Act as if we consumed the reservation.
}
tlb_remove_page_size(tlb, folio_page!(folio, 0i32), folio_size(folio as *const folio) as i32);
// If we were instructed to unmap a specific folio, we're done.
if folio_provided {
break '__slate_break_17;
}
}
}
}
}
}
address = address.wrapping_add(sz);
}
tlb_end_vma(tlb, vma);
huge_pmd_unshare_flush(tlb, vma);
}

#[unsafe(no_mangle)]
extern "C" fn __hugetlb_zap_begin(vma: *mut vm_area_struct, start: *mut u64, end: *mut u64) {
if unsafe { (*vma).vm_file }.is_null() {
return;
}
// hugetlbfs_file_mmap error
adjust_range_if_pmd_sharing_possible(vma, start, end);
hugetlb_vma_lock_write(vma);
if !unsafe { (*vma).vm_file }.is_null() {
i_mmap_lock_write(unsafe { (*(*vma).vm_file).f_mapping });
}
}

#[unsafe(no_mangle)]
extern "C" fn __hugetlb_zap_end(vma: *mut vm_area_struct, details: *mut zap_details) {
let zap_flags: u32 = if !details.is_null() { unsafe { (*details).zap_flags } } else { 0 };
if unsafe { (*vma).vm_file }.is_null() {
return;
}
// hugetlbfs_file_mmap error
if zap_flags & (BIT!(1i32) as u32) != 0 {
// final unmap
// Unlock and free the vma lock before releasing i_mmap_rwsem.
// When the vma_lock is freed, this makes the vma ineligible
// for pmd sharing.  And, i_mmap_rwsem is required to set up
// pmd sharing.  This is important as page tables for this
// unmapped range will be asynchrously deleted.  If the page
// tables are shared, there will be issues when accessed by
// someone else.
__hugetlb_vma_unlock_write_free(vma);
} else {
hugetlb_vma_unlock_write(vma);
}
if !unsafe { (*vma).vm_file }.is_null() {
i_mmap_unlock_write(unsafe { (*(*vma).vm_file).f_mapping });
}
}

#[unsafe(no_mangle)]
extern "C" fn unmap_hugepage_range(vma: *mut vm_area_struct, start: u64, end: u64, folio: *mut folio, zap_flags: u32) {
let mut range: mmu_notifier_range = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(range) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut tlb: mmu_gather = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tlb) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 128usize);
}
mmu_notifier_range_init(unsafe { &mut (*core::ptr::addr_of_mut!(range)) }, 1u32, 0u32, unsafe { (*vma).vm_mm }, start, end);
adjust_range_if_pmd_sharing_possible(vma, core::ptr::addr_of_mut!(range.start), core::ptr::addr_of_mut!(range.end));
mmu_notifier_invalidate_range_start(core::ptr::addr_of_mut!(range));
unsafe { tlb_gather_mmu(core::ptr::addr_of_mut!(tlb), (*vma).vm_mm) };
__unmap_hugepage_range(core::ptr::addr_of_mut!(tlb), vma, start, end, folio, zap_flags);
mmu_notifier_invalidate_range_end(core::ptr::addr_of_mut!(range));
unsafe { tlb_finish_mmu(core::ptr::addr_of_mut!(tlb)) };
}

/// This is called when the original mapper is failing to COW a MAP_PRIVATE
/// mapping it owns the reserve page for. The intention is to unmap the page
/// from other VMAs and let the children be SIGKILLed if they are faulting the
/// same region.
fn unmap_ref_private(mm: *mut mm_struct, vma: *mut vm_area_struct, folio: *mut folio, mut address: u64) {
let h: *mut hstate = hstate_vma(vma);
let mut iter_vma: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut pgoff: u64 = 0;
// vm_pgoff is in PAGE_SIZE units, hence the different calculation
// from page cache lookup which is in HPAGE_SIZE units.
address &= huge_page_mask(h);
pgoff = linear_page_index(vma as *const vm_area_struct, address);
let mapping: *mut address_space = unsafe { (*(*vma).vm_file).f_mapping };
// Take the mapping lock for the duration of the table walk. As
// this mapping should be shared between all the VMAs,
// __unmap_hugepage_range() is called as the lock is already held
i_mmap_lock_write(mapping);
iter_vma = unsafe { mapping_rmap_tree_iter_first(mapping, pgoff, pgoff) };
while !iter_vma.is_null() {
// Do not unmap the current VMA
if iter_vma != vma {
// Shared VMAs have their own reserves and do not affect
// MAP_PRIVATE accounting but it is possible that a shared
// VMA is using the same page so check and skip such VMAs.
if (unsafe { (*iter_vma).__slate_anon_3.vm_flags }) & BIT!(7i32) == 0 {
// Unmap the page from other VMAs without their own reserves.
// They get marked to be SIGKILLed if they fault in these
// areas. This is because a future no-page fault on this VMA
// could insert a zeroed page instead of the data existing
// from the time of fork. This would look like data corruption
if is_vma_resv_set(iter_vma, 1u64) == 0 {
unmap_hugepage_range(iter_vma, address, address.wrapping_add(huge_page_size(h as *const hstate)), folio, 0u32);
}
}
}
iter_vma = unsafe { mapping_rmap_tree_iter_next(iter_vma, pgoff, pgoff) };
}
i_mmap_unlock_write(mapping);
}

/// hugetlb_wp() should be called with page lock of the original hugepage held.
/// Called with hugetlb_fault_mutex_table held and pte_page locked so we
/// cannot race with other handlers or page migration.
/// Keep the pte_same checks anyway to make transition from the mutex easier.
fn hugetlb_wp(vmf: *mut vm_fault) -> u32 {
let mut newpte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(newpte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut hash: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(hash) as *mut u32;
let mut idx: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(idx) as *mut u64;
// If the process that created a MAP_PRIVATE mapping is about to perform
// a COW due to a shared page count, attempt to satisfy the allocation
// without using the existing reserves.
// In order to determine where this is a COW on a MAP_PRIVATE mapping it
// is enough to check whether the old_folio is anonymous. This means that
// the reserve for this address was consumed. If reserves were used, a
// partial faulted mapping at the fime of fork() could consume its reserves
// on COW instead of the full address range.
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut __v7: bool = unsafe { core::mem::zeroed() };
// Never handle CoW for uffd-wp protected pages.  It should be only
// handled when the uffd-wp protection is removed.
//
// Note that only the CoW optimization path (in hugetlb_no_page())
// can trigger this, because hugetlb_fault() will always resolve
// uffd-wp bit first.
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut range: core::mem::MaybeUninit<mmu_notifier_range> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut mmu_notifier_range = core::ptr::addr_of_mut!(range) as *mut mmu_notifier_range;
let mut ret: u32 = unsafe { core::mem::zeroed() };
let mut cow_from_owner: bool = unsafe { core::mem::zeroed() };
let mut new_folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut folio = core::ptr::addr_of_mut!(new_folio) as *mut *mut folio;
let mut old_folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut folio = core::ptr::addr_of_mut!(old_folio) as *mut *mut folio;
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut pte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(pte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut unshare: bool = unsafe { core::mem::zeroed() };
let mut mm: *mut mm_struct = unsafe { core::mem::zeroed() };
let mut vma: *mut vm_area_struct = unsafe { core::mem::zeroed() };
unsafe {
{
vma = (*vmf).__slate_anon_0.vma;
mm = (*vma).vm_mm;
unshare = ((*vmf).flags as u32) & 1024 != 0;
core::ptr::write(__slate_slot_15, huge_ptep_get(mm, (*vmf).__slate_anon_0.address, (*vmf).pte));
h = hstate_vma(vma);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
cow_from_owner = false;
ret = 0;
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
if !unshare {
__v8 = huge_pte_uffd(*__slate_slot_15) != 0;
} else {
__v8 = false;
}
}
if __v8 {
return 0;
} else {
// Let's take out MAP_SHARED mappings first.
if (*vma).__slate_anon_3.vm_flags & BIT!(7i32) != 0 {
set_huge_ptep_writable(vma, (*vmf).__slate_anon_0.address, (*vmf).pte);
return 0;
} else {
*__slate_slot_13 = page_folio!((vmemmap_base as *mut page).wrapping_offset(pte_pfn(*__slate_slot_15) as isize));
delayacct_wpcopy_start();
'__join_32: {
'__join_0: {
'__join_15: {
loop {
if folio_mapcount(*__slate_slot_13 as *const folio) == 1 {
__v7 = folio_test_anon(*__slate_slot_13 as *const folio);
} else {
__v7 = false;
}
if __v7 {
break '__join_32;
} else {
// If no-one else is actually using this page, we're the exclusive
// owner and can reuse this page.
//
// Note that we don't rely on the (safer) folio refcount here, because
// copying the hugetlb folio when there are unexpected (temporary)
// folio references could harm simple fork()+exit() users when
// we run out of free hugetlb folios: we would have to kill processes
// in scenarios that used to work. As a side effect, there can still
// be leaks between processes, for example, with FOLL_GET users.
VM_BUG_ON_PAGE!();
if is_vma_resv_set(vma, 1u64) != 0 {
__v6 = folio_test_anon(*__slate_slot_13 as *const folio);
} else {
__v6 = false;
}
if __v6 {
cow_from_owner = true;
}
folio_get(*__slate_slot_13);
// Drop page table lock as buddy allocator may be called. It will
// be acquired again before returning to the caller, as expected.
spin_unlock((*vmf).ptl);
*__slate_slot_12 = alloc_hugetlb_folio(vma, (*vmf).__slate_anon_0.address, cow_from_owner);
if IS_ERR(*__slate_slot_12 as *const ()) {
// If a process owning a MAP_PRIVATE mapping fails to COW,
// it is due to references held by a child and an insufficient
// huge page pool. To guarantee the original mappers
// reliability, unmap the page from child processes. The child
// may get SIGKILLed if it later faults.
if cow_from_owner {
let mapping: *mut address_space = (*(*vma).vm_file).f_mapping;
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
folio_put(*__slate_slot_13);
// Drop hugetlb_fault_mutex and vma_lock before
// unmapping.  unmapping needs to hold vma_lock
// in write mode.  Dropping vma_lock in read mode
// here is OK as COW mappings do not interact with
// PMD sharing.
//
// Reacquire both after unmap operation.
*__slate_slot_4 = vma_hugecache_offset(h, vma, (*vmf).__slate_anon_0.address);
*__slate_slot_3 = hugetlb_fault_mutex_hash(mapping, *__slate_slot_4);
hugetlb_vma_unlock_read(vma);
mutex_unlock(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(*__slate_slot_3 as isize));
unmap_ref_private(mm, vma, *__slate_slot_13, (*vmf).__slate_anon_0.address);
mutex_lock(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(*__slate_slot_3 as isize));
hugetlb_vma_lock_read(vma);
spin_lock((*vmf).ptl);
(*vmf).pte = hugetlb_walk(vma, (*vmf).__slate_anon_0.address, huge_page_size(h as *const hstate));
if !unsafe { (*vmf).pte }.is_null() {
__v2 = pte_same(huge_ptep_get(mm, (*vmf).__slate_anon_0.address, (*vmf).pte), *__slate_slot_15) != 0;
} else {
__v2 = false;
}
if !__v2 {
// race occurs while re-acquiring page table
// lock, and our job is done.
delayacct_wpcopy_end();
return 0;
}
} else {
break '__join_15;
}
} else {
break;
}
}
}
// When the original hugepage is shared one, it does not have
// anon_vma prepared.
ret = __vmf_anon_prepare(vmf);
if unlikely__3!(ret) == 0 {
if copy_user_large_folio(*__slate_slot_12, *__slate_slot_13, (*vmf).__slate_anon_0.real_address, vma) != 0 {
ret = 32 | VM_FAULT_SET_HINDEX!(hstate_index(h));
} else {
__folio_mark_uptodate(*__slate_slot_12);
mmu_notifier_range_init(&mut (*core::ptr::addr_of_mut!(*__slate_slot_9)), 1u32, 0u32, mm, (*vmf).__slate_anon_0.address, unsafe { (*vmf).__slate_anon_0.address }.wrapping_add(huge_page_size(h as *const hstate)));
mmu_notifier_invalidate_range_start(core::ptr::addr_of_mut!(*__slate_slot_9));
// Retake the page table lock to check for racing updates
// before the page tables are altered
spin_lock((*vmf).ptl);
(*vmf).pte = hugetlb_walk(vma, (*vmf).__slate_anon_0.address, huge_page_size(h as *const hstate));
if !unsafe { (*vmf).pte }.is_null() {
__v1 = pte_same(huge_ptep_get(mm, (*vmf).__slate_anon_0.address, (*vmf).pte), *__slate_slot_15) != 0;
} else {
__v1 = false;
}
if __v1 {
core::ptr::write(__slate_slot_0, make_huge_pte(vma, *__slate_slot_12, !unshare));
// Break COW or unshare
huge_ptep_clear_flush(vma, (*vmf).__slate_anon_0.address, (*vmf).pte);
hugetlb_remove_rmap(*__slate_slot_13);
hugetlb_add_new_anon_rmap(*__slate_slot_12, vma, (*vmf).__slate_anon_0.address);
if huge_pte_uffd(*__slate_slot_15) != 0 {
*__slate_slot_0 = huge_pte_mkuffd(*__slate_slot_0);
}
set_huge_pte_at(mm, (*vmf).__slate_anon_0.address, (*vmf).pte, *__slate_slot_0, huge_page_size(h as *const hstate));
folio_set_hugetlb_migratable(*__slate_slot_12);
// Make the old page be freed below
*__slate_slot_12 = *__slate_slot_13;
}
spin_unlock((*vmf).ptl);
mmu_notifier_invalidate_range_end(core::ptr::addr_of_mut!(*__slate_slot_9));
}
}
if *__slate_slot_12 != *__slate_slot_13 {
restore_reserve_on_error(h, vma, (*vmf).__slate_anon_0.address, *__slate_slot_12);
}
// No restore in case of successful pagetable update (Break COW or
// unshare)
folio_put(*__slate_slot_12);
break '__join_0;
}
ret = vmf_error(PTR_ERR(*__slate_slot_12 as *const ()) as i32);
}
folio_put(*__slate_slot_13);
spin_lock((*vmf).ptl); // Caller expects lock to be held
delayacct_wpcopy_end();
return ret;
}
if PageAnonExclusive(core::ptr::addr_of_mut!((*(*__slate_slot_13)).__slate_anon_0.page) as *const page) == 0 {
folio_move_anon_rmap(*__slate_slot_13, vma);
SetPageAnonExclusive(core::ptr::addr_of_mut!((*(*__slate_slot_13)).__slate_anon_0.page));
}
if likely!(!unshare) != 0 {
set_huge_ptep_maybe_writable(vma, (*vmf).__slate_anon_0.address, (*vmf).pte);
}
delayacct_wpcopy_end();
return 0;
}
}
}
}

/// Return whether there is a pagecache page to back given address within VMA.
#[unsafe(no_mangle)]
extern "C" fn hugetlbfs_pagecache_present(h: *mut hstate, vma: *mut vm_area_struct, address: u64) -> bool {
let mapping: *mut address_space = unsafe { (*(*vma).vm_file).f_mapping };
let idx: u64 = linear_page_index(vma as *const vm_area_struct, address);
let folio: *mut folio = filemap_get_folio(mapping, idx);
if IS_ERR(folio as *const ()) {
return false;
}
folio_put(folio);
true
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_add_to_page_cache(folio: *mut folio, mapping: *mut address_space, mut idx: u64) -> i32 {
let inode: *mut inode = unsafe { (*mapping).host };
let h: *mut hstate = hstate_inode(inode);
let mut err: i32 = 0;
let __v0: u32 = huge_page_order(h);
idx <<= __v0;
__folio_set_locked(folio);
err = unsafe { __filemap_add_folio(mapping, folio, idx, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), core::ptr::null_mut::<*mut ()>()) };
if unlikely__2!(err) != 0 {
__folio_clear_locked(folio);
return err;
}
folio_clear_hugetlb_restore_reserve(folio);
// mark folio dirty so that it will not be removed from cache/file
// by non-hugetlbfs specific code paths.
unsafe { folio_mark_dirty(folio) };
spin_lock(unsafe { core::ptr::addr_of_mut!((*inode).i_lock) });
let __v3: u32 = blocks_per_huge_page(h);
let __v5: u64 = unsafe { (*inode).i_blocks };
unsafe {
(*inode).i_blocks = __v5.wrapping_add(__v3 as u64);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*inode).i_lock) });
0
}

fn hugetlb_handle_userfault(vmf: *mut vm_fault, mapping: *mut address_space, reason: u64) -> u32 {
let mut hash: u32 = 0;
// vma_lock and hugetlb_fault_mutex must be dropped before handling
// userfault. Also mmap_lock could be dropped due to handling
// userfault, any vma operation should be careful from here.
hugetlb_vma_unlock_read(unsafe { (*vmf).__slate_anon_0.vma });
hash = hugetlb_fault_mutex_hash(mapping, unsafe { (*vmf).__slate_anon_0.pgoff });
unsafe { mutex_unlock(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(hash as isize)) };
handle_userfault(vmf, reason)
}

/// Recheck pte with pgtable lock.  Returns true if pte didn't change, or
/// false if pte changed or is changing.
fn hugetlb_pte_stable(h: *mut hstate, mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21, old_pte: __SlateRecord_pgtable_64_types_h_21) -> bool {
let mut same: bool = false;
let ptl: *mut spinlock = huge_pte_lock(h, mm, ptep);
same = pte_same(huge_ptep_get(mm, addr, ptep), old_pte) != 0;
spin_unlock(ptl);
same
}

fn hugetlb_no_page(mapping: *mut address_space, vmf: *mut vm_fault) -> u32 {
let mut new_pte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(new_pte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(size) as *mut u64;
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut folio_locked: bool = unsafe { core::mem::zeroed() };
let mut ret: u32 = unsafe { core::mem::zeroed() };
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut mm: *mut mm_struct = unsafe { core::mem::zeroed() };
let mut vma: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut new_anon_folio: bool = unsafe { core::mem::zeroed() };
let mut new_folio: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut bool = core::ptr::addr_of_mut!(new_folio) as *mut bool;
let mut hash: u32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_6: {
hash = hugetlb_fault_mutex_hash(mapping, (*vmf).__slate_anon_0.pgoff);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
new_anon_folio = false;
vma = (*vmf).__slate_anon_0.vma;
mm = (*vma).vm_mm;
h = hstate_vma(vma);
ret = 2;
folio_locked = true;
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Currently, we are forced to kill the process in the event the
// original mapper has unmapped pages from the child due to a failed
// COW/unsharing. Warn that such a situation has occurred as it may not
// be obvious.
if is_vma_resv_set(vma, 2u64) != 0 {
if __ratelimit__15!(core::ptr::addr_of_mut!(_rs)) != 0 {
while false {
}
_printk(c"\x014PID %d killed due to inadequate hugepage pool\n".as_ptr() as *const u8, (*get_current()).pid);
}
} else {
'__join_2: {
// Use page lock to guard against racing truncation
// before we get page_table_lock.
*__slate_slot_11 = false;
*__slate_slot_4 = filemap_lock_hugetlb_folio(h, mapping, (*vmf).__slate_anon_0.pgoff);
if IS_ERR(*__slate_slot_4 as *const ()) {
*__slate_slot_3 = (i_size_read((*mapping).host as *const inode) >> huge_page_shift(h)) as u64;
if (*vmf).__slate_anon_0.pgoff >= *__slate_slot_3 {
break '__join_6;
} else {
// Check for page in userfault range
if userfaultfd_missing(vma) {
// Since hugetlb_no_page() was examining pte
// without pgtable lock, we need to re-test under
// lock because the pte may not be stable and could
// have changed from under us.  Try to detect
// either changed or during-changing ptes and retry
// properly when needed.
//
// Note that userfaultfd is actually fine with
// false positives (e.g. caused by pte changed),
// but not wrong logical events (e.g. caused by
// reading a pte during changing).  The latter can
// confuse the userspace, so the strictness is very
// much preferred.  E.g., MISSING event should
// never happen on the page after UFFDIO_COPY has
// correctly installed the page and returned.
if !hugetlb_pte_stable(h, mm, (*vmf).__slate_anon_0.address, (*vmf).pte, (*vmf).__slate_anon_4.orig_pte) {
ret = 0;
break '__join_6;
} else {
return hugetlb_handle_userfault(vmf, mapping, BIT!(9i32));
}
} else {
if (*vma).__slate_anon_3.vm_flags & BIT!(7i32) == 0 {
ret = __vmf_anon_prepare(vmf);
if unlikely__3!(ret) != 0 {
break '__join_6;
}
}
*__slate_slot_4 = alloc_hugetlb_folio(vma, (*vmf).__slate_anon_0.address, false);
if IS_ERR(*__slate_slot_4 as *const ()) {
// Returning error will result in faulting task being
// sent SIGBUS.  The hugetlb fault mutex prevents two
// tasks from racing to fault in the same page which
// could result in false unable to allocate errors.
// Page migration does not take the fault mutex, but
// does a clear then write of pte's under page table
// lock.  Page fault code could race with migration,
// notice the clear pte and try to allocate a page
// here.  Before returning error, get ptl and make
// sure there really is no pte entry.
if hugetlb_pte_stable(h, mm, (*vmf).__slate_anon_0.address, (*vmf).pte, (*vmf).__slate_anon_4.orig_pte) {
ret = vmf_error(PTR_ERR(*__slate_slot_4 as *const ()) as i32);
break '__join_6;
} else {
ret = 0;
break '__join_6;
}
} else {
folio_zero_user(*__slate_slot_4, (*vmf).__slate_anon_0.real_address);
__folio_mark_uptodate(*__slate_slot_4);
*__slate_slot_11 = true;
if (*vma).__slate_anon_3.vm_flags & BIT!(7i32) != 0 {
let err: i32 = hugetlb_add_to_page_cache(*__slate_slot_4, mapping, (*vmf).__slate_anon_0.pgoff);
if err != 0 {
// err can't be -EEXIST which implies someone
// else consumed the reservation since hugetlb
// fault mutex is held when add a hugetlb page
// to the page cache. So it's safe to call
// restore_reserve_on_error() here.
restore_reserve_on_error(h, vma, (*vmf).__slate_anon_0.address, *__slate_slot_4);
folio_put(*__slate_slot_4);
ret = 2;
break '__join_6;
}
} else {
new_anon_folio = true;
folio_lock(*__slate_slot_4);
}
}
}
}
} else {
// If memory error occurs between mmap() and fault, some process
// don't have hwpoisoned swap entry for errored virtual address.
// So we need to block hugepage fault by PG_hwpoison bit check.
if unlikely!(folio_test_hwpoison(*__slate_slot_4 as *const folio)) != 0 {
ret = 32 | VM_FAULT_SET_HINDEX!(hstate_index(h));
break '__join_2;
} else {
// Check for page in userfault range.
if userfaultfd_minor(vma) {
folio_unlock(*__slate_slot_4);
folio_put(*__slate_slot_4);
// See comment in userfaultfd_missing() block above
if !hugetlb_pte_stable(h, mm, (*vmf).__slate_anon_0.address, (*vmf).pte, (*vmf).__slate_anon_4.orig_pte) {
ret = 0;
break '__join_6;
} else {
return hugetlb_handle_userfault(vmf, mapping, 0u64);
}
}
}
}
// If we are going to COW a private mapping later, we examine the
// pending reservations for this page now. This will ensure that
// any allocations necessary to record that reservation occur outside
// the spinlock.
if ((*vmf).flags as u32) & 1 != 0 && (*vma).__slate_anon_3.vm_flags & BIT!(3i32) == 0 {
if vma_needs_reservation(h, vma, (*vmf).__slate_anon_0.address) < 0 {
ret = 1;
break '__join_2;
} else {
// Just decrements count, does not deallocate
vma_end_reservation(h, vma, (*vmf).__slate_anon_0.address);
}
}
(*vmf).ptl = huge_pte_lock(h, mm, (*vmf).pte);
ret = 0;
// If pte changed from under us, retry
if pte_same(huge_ptep_get(mm, (*vmf).__slate_anon_0.address, (*vmf).pte), (*vmf).__slate_anon_4.orig_pte) == 0 {
spin_unlock((*vmf).ptl);
} else {
if new_anon_folio {
hugetlb_add_new_anon_rmap(*__slate_slot_4, vma, (*vmf).__slate_anon_0.address);
} else {
hugetlb_add_file_rmap(*__slate_slot_4);
}
*__slate_slot_2 = make_huge_pte(vma, *__slate_slot_4, (*vma).__slate_anon_3.vm_flags & BIT!(3i32) != 0);
// If this pte was previously wr-protected, keep it wr-protected even
// if populated.
if unlikely!(pte_is_uffd_wp_marker((*vmf).__slate_anon_4.orig_pte)) != 0 {
*__slate_slot_2 = huge_pte_mkuffd(*__slate_slot_2);
}
set_huge_pte_at(mm, (*vmf).__slate_anon_0.address, (*vmf).pte, *__slate_slot_2, huge_page_size(h as *const hstate));
hugetlb_count_add(pages_per_huge_page(h as *const hstate) as i64, mm);
if ((*vmf).flags as u32) & 1 != 0 && (*vma).__slate_anon_3.vm_flags & BIT!(3i32) == 0 {
// No need to keep file folios locked. See comment in
// hugetlb_fault().
if !new_anon_folio {
folio_locked = false;
folio_unlock(*__slate_slot_4);
}
// Optimization, do the COW without a second fault
ret = hugetlb_wp(vmf);
}
spin_unlock((*vmf).ptl);
// Only set hugetlb_migratable in newly allocated pages.  Existing pages
// found in the pagecache may not have hugetlb_migratable if they have
// been isolated for migration.
if *__slate_slot_11 {
folio_set_hugetlb_migratable(*__slate_slot_4);
}
if folio_locked {
folio_unlock(*__slate_slot_4);
break '__join_6;
} else {
break '__join_6;
}
}
}
if new_anon_folio {
restore_reserve_on_error(h, vma, (*vmf).__slate_anon_0.address, *__slate_slot_4);
}
// We only need to restore reservations for private mappings
folio_unlock(*__slate_slot_4);
folio_put(*__slate_slot_4);
}
}
hugetlb_vma_unlock_read(vma);
// We must check to release the per-VMA lock. __vmf_anon_prepare() is
// the only way ret can be set to VM_FAULT_RETRY.
if unlikely__3!(ret & 1024) != 0 {
vma_end_read(vma);
}
mutex_unlock(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(hash as isize));
return ret;
}
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

#[unsafe(no_mangle)]
extern "C" fn hugetlb_fault_mutex_hash(mapping: *mut address_space, idx: u64) -> u32 {
let mut key: __SlateAlign16<[u64; 2]> = __SlateAlign16([0u64; 2]);
let mut hash: u32 = 0;
key.0[0] = mapping as u64;
key.0[1] = idx;
hash = jhash2((core::ptr::addr_of_mut!(key.0) as *mut u32) as *const u32, ((core::mem::size_of::<[u64; 2]>() as u64) / (core::mem::size_of::<u32>() as u64)) as u32, 0u32);
hash & (unsafe { num_fault_mutexes }.wrapping_sub(1) as u32)
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_fault(mm: *mut mm_struct, vma: *mut vm_area_struct, address: u64, flags: u32) -> u32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
// Handle userfault-wp first, before trying to lock more pages
let mut __v2: bool = unsafe { core::mem::zeroed() };
// If we are going to COW/unshare the mapping later, we examine the
// pending reservations for this page now. This will ensure that any
// allocations necessary to record that reservation occur outside the
// spinlock.
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut pte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(pte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut ptl: core::mem::MaybeUninit<*mut spinlock> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut spinlock = core::ptr::addr_of_mut!(ptl) as *mut *mut spinlock;
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut __v9: bool = unsafe { core::mem::zeroed() };
// Protnone hugetlb PTEs with the uffd bit are used by
// userfaultfd RWP for access tracking. Plain PROT_NONE (without the
// marker) is not an RWP fault and is not expected on hugetlb (no
// NUMA hinting), so let normal hugetlb fault handling proceed.
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut softleaf: core::mem::MaybeUninit<__SlateRecord_mm_types_h_288> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut __SlateRecord_mm_types_h_288 = core::ptr::addr_of_mut!(softleaf) as *mut __SlateRecord_mm_types_h_288;
// Not present, either a migration or a hwpoisoned entry
let mut __v12: bool = unsafe { core::mem::zeroed() };
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut marker: u64 = unsafe { core::mem::zeroed() };
// TODO: Track hugetlb faults using vm_fault
// Some fields may not be initialized, be careful as it may
// be hard to debug if called functions make assumptions
let mut vmf: core::mem::MaybeUninit<vm_fault> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut vm_fault = core::ptr::addr_of_mut!(vmf) as *mut vm_fault;
let mut need_wait_lock: bool = unsafe { core::mem::zeroed() };
let mut mapping: core::mem::MaybeUninit<*mut address_space> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut address_space = core::ptr::addr_of_mut!(mapping) as *mut *mut address_space;
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let mut hash: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(hash) as *mut u32;
let mut ret: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(ret) as *mut u32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
folio = core::ptr::null_mut::<folio>();
h = hstate_vma(vma);
let __slate_auto_bytes_3 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
need_wait_lock = false;
core::ptr::write(__slate_slot_17, {
let mut __t0: vm_fault = core::mem::zeroed();
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_mm_h_754 = core::mem::zeroed();
__t1.vma = vma;
__t1.pgoff = vma_hugecache_offset(h, vma, address & huge_page_mask(h));
__t1.address = address & huge_page_mask(h);
__t1.real_address = address;
__t1
};
__t0.flags = flags as u32;
__t0
});
// Serialize hugepage allocation and instantiation, so that we don't
// get spurious allocation failures if two CPUs race to instantiate
// the same page in the page cache.
*__slate_slot_19 = (*(*vma).vm_file).f_mapping;
*__slate_slot_22 = hugetlb_fault_mutex_hash(*__slate_slot_19, (*__slate_slot_17).__slate_anon_0.pgoff);
mutex_lock(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(*__slate_slot_22 as isize));
// Acquire vma lock before calling huge_pte_alloc and hold
// until finished with vmf.pte.  This prevents huge_pmd_unshare from
// being called elsewhere and making the vmf.pte no longer valid.
hugetlb_vma_lock_read(vma);
(*__slate_slot_17).pte = huge_pte_alloc(mm, vma, (*__slate_slot_17).__slate_anon_0.address, huge_page_size(h as *const hstate));
if (*__slate_slot_17).pte.is_null() {
hugetlb_vma_unlock_read(vma);
mutex_unlock(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(*__slate_slot_22 as isize));
return 1;
} else {
(*__slate_slot_17).__slate_anon_4.orig_pte = huge_ptep_get(mm, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte);
if huge_pte_none((*__slate_slot_17).__slate_anon_4.orig_pte) != 0 {
return hugetlb_no_page(*__slate_slot_19, core::ptr::addr_of_mut!(*__slate_slot_17));
} else {
'__join_4: {
// hugetlb_no_page will drop vma lock and hugetlb fault
// mutex internally, which make us return immediately.
if pte_is_marker((*__slate_slot_17).__slate_anon_4.orig_pte) {
marker = softleaf_to_marker(softleaf_from_pte((*__slate_slot_17).__slate_anon_4.orig_pte));
if marker & BIT!(1i32) != 0 {
*__slate_slot_23 = 32 | VM_FAULT_SET_HINDEX!(hstate_index(h));
} else {
__ret_warn_on = (marker & BIT!(2i32) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("740: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 740b - ., 3; .popsection\n# {0}", const 740 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_hugetlb_fault_24, sym __slate_asm_string_hugetlb_fault_25, const 6075 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("741: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 741b - ., 4; .popsection\n# {0}", const 741 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v15: i64 = (__ret_warn_on != 0) as i64;
if __v15 != 0 {
// This isn't supported in hugetlb.
*__slate_slot_23 = 64;
} else {
return hugetlb_no_page(*__slate_slot_19, core::ptr::addr_of_mut!(*__slate_slot_17));
}
}
} else {
*__slate_slot_23 = 0;
if pte_present((*__slate_slot_17).__slate_anon_4.orig_pte) == 0 {
__v12 = huge_pte_none((*__slate_slot_17).__slate_anon_4.orig_pte) == 0;
} else {
__v12 = false;
}
if __v12 {
core::ptr::write(__slate_slot_11, softleaf_from_pte((*__slate_slot_17).__slate_anon_4.orig_pte));
if softleaf_is_migration(*__slate_slot_11) {
// Release the hugetlb fault lock now, but retain
// the vma lock, because it is needed to guard the
// huge_pte_lockptr() later in
// migration_entry_wait_huge(). The vma lock will
// be released there.
mutex_unlock(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(*__slate_slot_22 as isize));
migration_entry_wait_huge(vma, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte);
return 0;
} else {
if softleaf_is_hwpoison(*__slate_slot_11) {
*__slate_slot_23 = 32 | VM_FAULT_SET_HINDEX!(hstate_index(h));
}
}
} else {
if pte_protnone((*__slate_slot_17).__slate_anon_4.orig_pte) != 0 {
__v10 = vma_is_accessible(vma as *const vm_area_struct);
} else {
__v10 = false;
}
if __v10 {
__v9 = userfaultfd_rwp(vma);
} else {
__v9 = false;
}
if __v9 {
__v8 = huge_pte_uffd((*__slate_slot_17).__slate_anon_4.orig_pte) != 0;
} else {
__v8 = false;
}
if __v8 {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Sync: drop hugetlb locks before blocking in handle_userfault()
if !userfaultfd_rwp_async(vma) {
return hugetlb_handle_userfault(core::ptr::addr_of_mut!(*__slate_slot_17), *__slate_slot_19, 0u64);
} else {
*__slate_slot_7 = huge_pte_lock(h, mm, (*__slate_slot_17).pte);
*__slate_slot_6 = huge_ptep_get(mm, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte);
if pte_protnone(*__slate_slot_6) != 0 {
__v5 = huge_pte_uffd(*__slate_slot_6) != 0;
} else {
__v5 = false;
}
if __v5 {
let shift: u32 = huge_page_shift(h);
*__slate_slot_6 = huge_pte_modify(*__slate_slot_6, (*vma).vm_page_prot);
*__slate_slot_6 = arch_make_huge_pte(*__slate_slot_6, shift, (*vma).__slate_anon_3.vm_flags);
// huge_pte_modify() preserves _PAGE_UFFD; drop it on resolution
*__slate_slot_6 = huge_pte_clear_uffd(*__slate_slot_6);
*__slate_slot_6 = pte_mkyoung(*__slate_slot_6);
// Unlike do_uffd_rwp(), do not upgrade to writable
// here. Hugetlb lacks a can_change_huge_pte_writable()
// equivalent, so a write access will take a separate
// COW fault — acceptable for the rare private hugetlb
// case.
set_huge_pte_at(mm, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte, *__slate_slot_6, huge_page_size(h as *const hstate));
update_mmu_cache(vma, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte);
}
spin_unlock(*__slate_slot_7);
*__slate_slot_23 = 0;
}
} else {
if flags & 1025u32 != 0 && (*vma).__slate_anon_3.vm_flags & BIT!(7i32) == 0 {
__v3 = huge_pte_write((*__slate_slot_17).__slate_anon_4.orig_pte) == 0;
} else {
__v3 = false;
}
if __v3 {
if vma_needs_reservation(h, vma, (*__slate_slot_17).__slate_anon_0.address) < 0 {
*__slate_slot_23 = 1;
break '__join_4;
} else {
// Just decrements count, does not deallocate
vma_end_reservation(h, vma, (*__slate_slot_17).__slate_anon_0.address);
}
}
'__join_5: {
(*__slate_slot_17).ptl = huge_pte_lock(h, mm, (*__slate_slot_17).pte);
// Check for a racing update before calling hugetlb_wp()
if unlikely!(pte_same((*__slate_slot_17).__slate_anon_4.orig_pte, huge_ptep_get(mm, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte)) == 0) == 0 {
if userfaultfd_wp(vma) {
__v2 = huge_pte_uffd(huge_ptep_get(mm, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte)) != 0;
} else {
__v2 = false;
}
if __v2 && flags & 1 != 0 {
__v1 = huge_pte_write((*__slate_slot_17).__slate_anon_4.orig_pte) == 0;
} else {
__v1 = false;
}
if __v1 {
if !userfaultfd_wp_async(vma) {
spin_unlock((*__slate_slot_17).ptl);
hugetlb_vma_unlock_read(vma);
mutex_unlock(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(*__slate_slot_22 as isize));
return handle_userfault(core::ptr::addr_of_mut!(*__slate_slot_17), BIT!(12i32));
} else {
(*__slate_slot_17).__slate_anon_4.orig_pte = huge_pte_clear_uffd((*__slate_slot_17).__slate_anon_4.orig_pte);
set_huge_pte_at(mm, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte, (*__slate_slot_17).__slate_anon_4.orig_pte, huge_page_size(hstate_vma(vma) as *const hstate));
// Fallthrough to CoW
}
}
if flags & 1025u32 != 0 {
if huge_pte_write((*__slate_slot_17).__slate_anon_4.orig_pte) == 0 {
// Anonymous folios need to be lock since hugetlb_wp()
// checks whether we can re-use the folio exclusively
// for us in case we are the only user of it.
folio = page_folio!((vmemmap_base as *mut page).wrapping_offset(pte_pfn((*__slate_slot_17).__slate_anon_4.orig_pte) as isize));
if folio_test_anon(folio as *const folio) {
__v0 = !folio_trylock(folio);
} else {
__v0 = false;
}
if __v0 {
need_wait_lock = true;
break '__join_5;
} else {
folio_get(folio);
*__slate_slot_23 = hugetlb_wp(core::ptr::addr_of_mut!(*__slate_slot_17));
if folio_test_anon(folio as *const folio) {
folio_unlock(folio);
}
folio_put(folio);
break '__join_5;
}
} else {
if likely__4!(flags & 1) != 0 {
(*__slate_slot_17).__slate_anon_4.orig_pte = huge_pte_mkdirty((*__slate_slot_17).__slate_anon_4.orig_pte);
}
}
}
(*__slate_slot_17).__slate_anon_4.orig_pte = pte_mkyoung((*__slate_slot_17).__slate_anon_4.orig_pte);
if huge_ptep_set_access_flags(vma, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte, (*__slate_slot_17).__slate_anon_4.orig_pte, (flags & 1) as i32) != 0 {
update_mmu_cache(vma, (*__slate_slot_17).__slate_anon_0.address, (*__slate_slot_17).pte);
}
}
}
spin_unlock((*__slate_slot_17).ptl);
}
}
}
}
hugetlb_vma_unlock_read(vma);
// We must check to release the per-VMA lock. __vmf_anon_prepare() in
// hugetlb_wp() is the only way ret can be set to VM_FAULT_RETRY.
if unlikely__3!(*__slate_slot_23 & 1024) != 0 {
vma_end_read(vma);
}
mutex_unlock(unsafe { hugetlb_fault_mutex_table }.wrapping_offset(*__slate_slot_22 as isize));
// hugetlb_wp drops all the locks, but the folio lock, before trying to
// unmap the folio from other processes. During that window, if another
// process mapping that folio faults in, it will take the mutex and then
// it will wait on folio_lock, causing an ABBA deadlock.
// Use trylock instead and bail out if we fail.
//
// Ideally, we should hold a refcount on the folio we wait for, but we do
// not want to use the folio after it becomes unlocked, but rather just
// wait for it to become unlocked, so hopefully next fault successes on
// the trylock.
if need_wait_lock {
folio_wait_locked(folio);
}
return *__slate_slot_23;
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_change_protection(vma: *mut vm_area_struct, mut address: u64, end: u64, newprot: pgprot, cp_flags: u64) -> i64 {
let mut __slate_compound_40: pgprot = unsafe { core::mem::zeroed() };
let mut newpte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(newpte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
// Preserve RWP protection across mprotect()
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut __v9: bool = unsafe { core::mem::zeroed() };
// Already protnone with uffd bit set? Nothing to do.
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut shift: u32 = unsafe { core::mem::zeroed() };
let mut old_pte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(old_pte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut ptl: core::mem::MaybeUninit<*mut spinlock> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut spinlock = core::ptr::addr_of_mut!(ptl) as *mut *mut spinlock;
let mut entry: core::mem::MaybeUninit<__SlateRecord_mm_types_h_288> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut __SlateRecord_mm_types_h_288 = core::ptr::addr_of_mut!(entry) as *mut __SlateRecord_mm_types_h_288;
let mut tlb: core::mem::MaybeUninit<mmu_gather> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut mmu_gather = core::ptr::addr_of_mut!(tlb) as *mut mmu_gather;
let mut uffd_rwp_resolve: bool = unsafe { core::mem::zeroed() };
let mut uffd_rwp: bool = unsafe { core::mem::zeroed() };
let mut uffd_wp_resolve: bool = unsafe { core::mem::zeroed() };
let mut uffd_wp: bool = unsafe { core::mem::zeroed() };
let mut last_addr_mask: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut u64 = core::ptr::addr_of_mut!(last_addr_mask) as *mut u64;
let mut range: core::mem::MaybeUninit<mmu_notifier_range> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut mmu_notifier_range = core::ptr::addr_of_mut!(range) as *mut mmu_notifier_range;
let mut psize: i64 = unsafe { core::mem::zeroed() };
let mut pages: i64 = unsafe { core::mem::zeroed() };
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut pte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(pte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut ptep: core::mem::MaybeUninit<*mut __SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(ptep) as *mut *mut __SlateRecord_pgtable_64_types_h_21;
let mut mm: *mut mm_struct = unsafe { core::mem::zeroed() };
unsafe {
mm = (*vma).vm_mm;
let start: u64 = address;
let __slate_auto_bytes_2 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
h = hstate_vma(vma);
pages = 0;
psize = huge_page_size(h as *const hstate) as i64;
let __slate_auto_bytes_4 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 40usize);
let __slate_auto_bytes_5 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
uffd_wp = cp_flags & 4u64 != 0;
uffd_wp_resolve = cp_flags & 8u64 != 0;
uffd_rwp = cp_flags & 16u64 != 0;
uffd_rwp_resolve = cp_flags & 32u64 != 0;
let __slate_auto_bytes_6 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 128usize);
// In the case of shared PMDs, the area to flush could be beyond
// start/end.  Set range.start/range.end to cover the maximum possible
// range if PMD sharing is possible.
mmu_notifier_range_init(&mut (*core::ptr::addr_of_mut!(*__slate_slot_30)), 2u32, 0u32, mm, start, end);
adjust_range_if_pmd_sharing_possible(vma, core::ptr::addr_of_mut!((*__slate_slot_30).start), core::ptr::addr_of_mut!((*__slate_slot_30).end));
'__join_57: {
loop {
if unlikely!(address >= end) != 0 {
break '__join_57;
} else {
if true {
break;
}
}
}
flush_cache_range(vma, (*__slate_slot_30).start, (*__slate_slot_30).end);
tlb_gather_mmu_vma(core::ptr::addr_of_mut!(*__slate_slot_24), vma);
mmu_notifier_invalidate_range_start(core::ptr::addr_of_mut!(*__slate_slot_30));
hugetlb_vma_lock_write(vma);
i_mmap_lock_write((*(*vma).vm_file).f_mapping);
*__slate_slot_29 = hugetlb_mask_last_page(h);
'__join_0: {
'__loop_1: loop {
if address < end {
'__join_2: {
{
let __slate_auto_bytes_1 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_35 = hugetlb_walk(vma, address, psize as u64);
if (*__slate_slot_35).is_null() {
// uffd_wp installs a pte marker on the unpopulated
// entry; uffd_rwp does not install markers so the
// allocation is unnecessary for it.
if !uffd_wp {
address |= *__slate_slot_29;
break '__join_2;
} else {
// Userfaultfd wr-protect requires pgtable
// pre-allocations to install pte markers.
*__slate_slot_35 = huge_pte_alloc(mm, vma, address, psize as u64);
if (*__slate_slot_35).is_null() {
break '__loop_1;
}
}
}
}
*__slate_slot_22 = huge_pte_lock(h, mm, *__slate_slot_35);
if huge_pmd_unshare(core::ptr::addr_of_mut!(*__slate_slot_24), vma, address, *__slate_slot_35) != 0 {
let __ret_warn_on: i32 = (uffd_wp || uffd_wp_resolve || uffd_rwp || uffd_rwp_resolve) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("743: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 743b - ., 3; .popsection\n# {0}", const 743 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_hugetlb_change_protection_41, sym __slate_asm_string_hugetlb_change_protection_42, const 6565 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("744: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 744b - ., 4; .popsection\n# {0}", const 744 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// When uffd-wp is enabled on the vma, unshare
// shouldn't happen at all.  Warn about it if it
// happened due to some reason.
pages = pages.wrapping_add(1);
spin_unlock(*__slate_slot_22);
address |= *__slate_slot_29;
} else {
*__slate_slot_34 = huge_ptep_get(mm, address, *__slate_slot_35);
if huge_pte_none(*__slate_slot_34) != 0 {
if unlikely!(uffd_wp) != 0 {
set_huge_pte_at(mm, address, *__slate_slot_35, make_pte_marker(BIT!(0i32)), psize as u64);
}
// Safe to modify directly (none->non-present).
} else {
*__slate_slot_23 = softleaf_from_pte(*__slate_slot_34);
if unlikely!(softleaf_is_hwpoison(*__slate_slot_23)) != 0 {
// Nothing to do.
} else {
if unlikely!(softleaf_is_migration(*__slate_slot_23)) != 0 {
folio = softleaf_to_folio(*__slate_slot_23);
core::ptr::write(__slate_slot_4, *__slate_slot_34);
if softleaf_is_migration_write(*__slate_slot_23) {
if folio_test_anon(folio as *const folio) {
*__slate_slot_23 = make_readable_exclusive_migration_entry(swp_offset(*__slate_slot_23));
} else {
*__slate_slot_23 = make_readable_migration_entry(swp_offset(*__slate_slot_23));
}
*__slate_slot_4 = swp_entry_to_pte(*__slate_slot_23);
pages = pages.wrapping_add(1);
}
if uffd_wp || uffd_rwp {
*__slate_slot_4 = pte_swp_mkuffd(*__slate_slot_4);
} else {
if uffd_wp_resolve || uffd_rwp_resolve {
*__slate_slot_4 = pte_swp_clear_uffd(*__slate_slot_4);
}
}
if pte_same(*__slate_slot_34, *__slate_slot_4) == 0 {
set_huge_pte_at(mm, address, *__slate_slot_35, *__slate_slot_4, psize as u64);
}
} else {
if unlikely!(pte_is_marker(*__slate_slot_34)) != 0 {
// Do nothing on a poison marker; page is
// corrupted, permissions do not apply. Here
// pte_marker_uffd_wp()==true implies !poison
// because they're mutual exclusive.
if pte_is_uffd_wp_marker(*__slate_slot_34) && (uffd_wp_resolve || uffd_rwp_resolve) {
huge_pte_clear(mm, address, *__slate_slot_35, psize as u64);
}
// Safe to modify directly (non-present->none).
} else {
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
shift = huge_page_shift(hstate_vma(vma));
if uffd_rwp {
__v10 = pte_protnone(*__slate_slot_34) != 0;
} else {
__v10 = false;
}
if __v10 {
__v9 = huge_pte_uffd(*__slate_slot_34) != 0;
} else {
__v9 = false;
}
if !__v9 {
*__slate_slot_12 = huge_ptep_modify_prot_start(vma, address, *__slate_slot_35);
*__slate_slot_34 = huge_pte_modify(*__slate_slot_12, newprot);
*__slate_slot_34 = arch_make_huge_pte(*__slate_slot_34, shift, (*vma).__slate_anon_3.vm_flags);
if uffd_wp || uffd_rwp {
*__slate_slot_34 = huge_pte_mkuffd(*__slate_slot_34);
} else {
if uffd_wp_resolve || uffd_rwp_resolve {
*__slate_slot_34 = huge_pte_clear_uffd(*__slate_slot_34);
}
}
if userfaultfd_rwp(vma) {
__v8 = huge_pte_uffd(*__slate_slot_34) != 0;
} else {
__v8 = false;
}
if __v8 {
*__slate_slot_34 = huge_pte_modify(*__slate_slot_34, *{
__slate_compound_40 = pgprot { pgprot: 0u64 | _AT!(u64, 1i32) << 5i32 | 0 | 0 | 0 | _AT!(u64, 1i32) << 8i32 };
core::ptr::addr_of_mut!(__slate_compound_40)
});
*__slate_slot_34 = arch_make_huge_pte(*__slate_slot_34, shift, (*vma).__slate_anon_3.vm_flags);
}
huge_ptep_modify_prot_commit(vma, address, *__slate_slot_35, *__slate_slot_12, *__slate_slot_34);
pages = pages.wrapping_add(1);
tlb_remove_huge_tlb_entry!(h, core::ptr::addr_of_mut!(*__slate_slot_24), *__slate_slot_35, address);
}
}
}
}
}
spin_unlock(*__slate_slot_22);
cond_resched__107!();
}
}
address = address.wrapping_add(psize as u64);
} else {
break '__join_0;
}
}
pages = 12i32.wrapping_neg() as i64;
}
tlb_flush_mmu_tlbonly(core::ptr::addr_of_mut!(*__slate_slot_24));
huge_pmd_unshare_flush(core::ptr::addr_of_mut!(*__slate_slot_24), vma);
// No need to call mmu_notifier_arch_invalidate_secondary_tlbs() we are
// downgrading page table protection not changing it to point to a new
// page.
//
// See Documentation/mm/mmu_notifier.rst
i_mmap_unlock_write((*(*vma).vm_file).f_mapping);
hugetlb_vma_unlock_write(vma);
mmu_notifier_invalidate_range_end(core::ptr::addr_of_mut!(*__slate_slot_30));
tlb_finish_mmu(core::ptr::addr_of_mut!(*__slate_slot_24));
return if pages > 0 { pages << (*h).order } else { pages };
}
core::arch::asm!("742: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 742b - ., 3; .popsection\n# {0}", const 742 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_hugetlb_change_protection_43, sym __slate_asm_string_hugetlb_change_protection_44, const 6524 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

// Update the reservation map for the range [from, to].
//
// Returns the number of entries that would be added to the reservation map
// associated with the range [from, to].  This number is greater or equal to
// zero. -EINVAL or -ENOMEM is returned in case of any errors.
#[unsafe(no_mangle)]
extern "C" fn hugetlb_reserve_pages(inode: *mut inode, from: i64, to: i64, vma: *mut vm_area_struct, mut vma_flags: __SlateRecord_mm_types_h_871) -> i64 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
// pages in this range were added to the reserve
// map between region_chg and region_add.  This
// indicates a race with alloc_hugetlb_folio.  Adjust
// the subpool and reserve counts modified above
// based on the difference.
let mut rsv_adjust: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i64 = core::ptr::addr_of_mut!(rsv_adjust) as *mut i64;
// Account for the reservations made. Shared mappings record regions
// that have reservations as they are shared by multiple VMAs.
// When the last VMA disappears, the region map says how much
// the reservation was and the page cache tells how much of
// the reservation was consumed. Private mappings are per-VMA and
// only the consumed reservations are tracked. When the VMA
// disappears, the original reservation is the VMA size and the
// consumed reservations are stored in the map. Hence, nothing
// else has to be done for private mappings here
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
// Shared mappings base their reservation on the number of pages that
// are already allocated on behalf of the file. Private mappings need
// to reserve the full area even if read-only as mprotect() may be
// called to make the mapping read-write. Assume !vma is a shm mapping
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut regions_needed: i64 = unsafe { core::mem::zeroed() };
let mut gbl_reserve: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i64 = core::ptr::addr_of_mut!(gbl_reserve) as *mut i64;
let mut h_cg: *mut hugetlb_cgroup = unsafe { core::mem::zeroed() };
let mut resv_map: core::mem::MaybeUninit<*mut resv_map> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut resv_map = core::ptr::addr_of_mut!(resv_map) as *mut *mut resv_map;
let mut spool: *mut hugepage_subpool = unsafe { core::mem::zeroed() };
let mut h: *mut hstate = unsafe { core::mem::zeroed() };
let mut gbl_resv: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i64 = core::ptr::addr_of_mut!(gbl_resv) as *mut i64;
let mut spool_resv: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i64 = core::ptr::addr_of_mut!(spool_resv) as *mut i64;
let mut add: i64 = unsafe { core::mem::zeroed() };
let mut chg: i64 = unsafe { core::mem::zeroed() };
unsafe {
chg = 1i32.wrapping_neg() as i64;
add = 1i32.wrapping_neg() as i64;
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
h = hstate_inode(inode);
spool = subpool_inode(inode);
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
h_cg = core::ptr::null_mut::<hugetlb_cgroup>();
let __slate_auto_bytes_4 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
regions_needed = 0;
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
// This should never happen
if from > to {
BUILD_BUG_ON_INVALID!();
return 22i32.wrapping_neg() as i64;
} else {
// vma specific semaphore used for pmd sharing and fault/truncation
// synchronization
hugetlb_vma_lock_alloc(vma);
// Only apply hugepage reservation if asked. At fault time, an
// attempt will be made for VM_NORESERVE to allocate a page
// without using reserves
if vma_flags_test(core::ptr::addr_of_mut!(vma_flags) as *const __SlateRecord_mm_types_h_871, 21i32) {
return 0;
} else {
if vma.is_null() {
__v9 = true;
} else {
__v9 = vma_test(vma as *const vm_area_struct, 7i32);
}
'__join_11: {
if __v9 {
// resv_map can not be NULL as hugetlb_reserve_pages is only
// called for inodes for which resv_maps were created (see
// hugetlbfs_get_inode).
*__slate_slot_14 = inode_resv_map(inode);
chg = region_chg(*__slate_slot_14, from, to, core::ptr::addr_of_mut!(regions_needed));
} else {
// Private mapping.
*__slate_slot_14 = resv_map_alloc();
if (*__slate_slot_14).is_null() {
*__slate_slot_10 = 12i32.wrapping_neg();
break '__join_11;
} else {
chg = to.wrapping_sub(from);
set_vma_resv_map(vma, *__slate_slot_14);
set_vma_resv_flags(vma, 1u64);
}
}
if chg < 0 {
// region_chg() above can return -ENOMEM
*__slate_slot_10 = if chg == (12i32.wrapping_neg() as i64) { 12i32.wrapping_neg() } else { 22i32.wrapping_neg() };
} else {
*__slate_slot_10 = hugetlb_cgroup_charge_cgroup_rsvd(hstate_index(h), chg.wrapping_mul(pages_per_huge_page(h as *const hstate) as i64) as u64, core::ptr::addr_of_mut!(h_cg));
if *__slate_slot_10 >= 0 {
if !vma.is_null() {
__v8 = !vma_test(vma as *const vm_area_struct, 7i32);
} else {
__v8 = false;
}
if __v8 && !h_cg.is_null() {
// For private mappings, the hugetlb_cgroup uncharge info hangs
// of the resv_map.
resv_map_set_hugetlb_cgroup_uncharge_info(*__slate_slot_14, h_cg, h);
}
// There must be enough pages in the subpool for the mapping. If
// the subpool has a minimum size, there may be some global
// reservations already in place (gbl_reserve).
*__slate_slot_12 = hugepage_subpool_get_pages(spool, chg);
if *__slate_slot_12 < 0 {
*__slate_slot_10 = *__slate_slot_12 as i32;
} else {
'__join_18: {
// Check enough hugepages are available for the reservation.
// Hand the pages back to the subpool if there are not
*__slate_slot_10 = hugetlb_acct_memory(h, *__slate_slot_12);
if *__slate_slot_10 >= 0 {
if vma.is_null() {
__v7 = true;
} else {
__v7 = vma_test(vma as *const vm_area_struct, 7i32);
}
if __v7 {
add = region_add(*__slate_slot_14, from, to, regions_needed, h, h_cg);
if unlikely!(add < 0) != 0 {
hugetlb_acct_memory(h, (*__slate_slot_12).wrapping_neg());
*__slate_slot_10 = add as i32;
break '__join_18;
} else {
if unlikely!(chg > add) != 0 {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// hugetlb_cgroup_uncharge_cgroup_rsvd() will put the
// reference to h_cg->css. See comment below for detail.
hugetlb_cgroup_uncharge_cgroup_rsvd(hstate_index(h), chg.wrapping_sub(add).wrapping_mul(pages_per_huge_page(h as *const hstate) as i64) as u64, h_cg);
*__slate_slot_6 = hugepage_subpool_put_pages(spool, chg.wrapping_sub(add));
hugetlb_acct_memory(h, (*__slate_slot_6).wrapping_neg());
} else {
if !h_cg.is_null() {
// The file_regions will hold their own reference to
// h_cg->css. So we should release the reference held
// via hugetlb_cgroup_charge_cgroup_rsvd() when we are
// done.
hugetlb_cgroup_put_rsvd_cgroup(h_cg);
}
}
}
}
return chg;
}
}
*__slate_slot_18 = chg.wrapping_sub(*__slate_slot_12);
if *__slate_slot_18 != 0 {
// put sub pool's reservation back, chg - gbl_reserve
*__slate_slot_17 = hugepage_subpool_put_pages(spool, *__slate_slot_18);
// subpool's reserved pages can not be put back due to race,
// return to hstate.
hugetlb_acct_memory(h, (*__slate_slot_17).wrapping_neg());
}
// Restore used_hpages for pages that failed global reservation
if *__slate_slot_12 != 0 && !spool.is_null() {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*spool).lock), *__slate_slot_5);
if (*spool).max_hpages != (1i32.wrapping_neg() as i64) {
let __v3: i64 = (*spool).used_hpages;
(*spool).used_hpages = __v3.wrapping_sub(*__slate_slot_12);
}
unlock_or_release_subpool(spool, *__slate_slot_5);
}
}
hugetlb_cgroup_uncharge_cgroup_rsvd(hstate_index(h), chg.wrapping_mul(pages_per_huge_page(h as *const hstate) as i64) as u64, h_cg);
}
}
}
hugetlb_vma_lock_free(vma);
if vma.is_null() {
__v1 = true;
} else {
__v1 = vma_test(vma as *const vm_area_struct, 7i32);
}
if __v1 {
if chg >= 0 && add < 0 {
region_abort(*__slate_slot_14, from, to, regions_needed);
}
}
// Only call region_abort if the region_chg succeeded but the
// region_add failed or didn't run.
if !vma.is_null() {
__v0 = is_vma_resv_set(vma, 1u64) != 0;
} else {
__v0 = false;
}
if __v0 {
kref_put(core::ptr::addr_of_mut!((*(*__slate_slot_14)).refs), Some(resv_map_release));
set_vma_resv_map(vma, core::ptr::null_mut::<resv_map>());
}
return *__slate_slot_10 as i64;
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn hugetlb_unreserve_pages(inode: *mut inode, start: i64, end: i64, freed: i64) -> i64 {
let h: *mut hstate = hstate_inode(inode);
let resv_map: *mut resv_map = inode_resv_map(inode);
let mut chg: i64 = 0;
let spool: *mut hugepage_subpool = subpool_inode(inode);
let mut gbl_reserve: i64 = 0;
// Since this routine can be called in the evict inode path for all
// hugetlbfs inodes, resv_map could be NULL.
if !resv_map.is_null() {
chg = region_del(resv_map, start, end);
// region_del() can fail in the rare case where a region
// must be split and another region descriptor can not be
// allocated.  If end == LONG_MAX, it will not fail.
if chg < 0 {
return chg;
}
}
spin_lock(unsafe { core::ptr::addr_of_mut!((*inode).i_lock) });
let __v0: i64 = (blocks_per_huge_page(h) as i64).wrapping_mul(freed);
let __v2: u64 = unsafe { (*inode).i_blocks };
unsafe {
(*inode).i_blocks = __v2.wrapping_sub(__v0 as u64);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*inode).i_lock) });
// If the subpool has a minimum size, the number of global
// reservations to be released may be adjusted.
//
// Note that !resv_map implies freed == 0. So (chg - freed)
// won't go negative.
gbl_reserve = hugepage_subpool_put_pages(spool, chg.wrapping_sub(freed));
hugetlb_acct_memory(h, gbl_reserve.wrapping_neg());
0
}

fn page_table_shareable(svma: *mut vm_area_struct, vma: *mut vm_area_struct, addr: u64, idx: u64) -> u64 {
let saddr: u64 = (idx.wrapping_sub(vma_start_pgoff(svma as *const vm_area_struct)) << 12i32).wrapping_add(unsafe { (*svma).__slate_anon_0.__slate_anon_0.vm_start });
let sbase: u64 = saddr & !(_AC!() << 30i32).wrapping_sub(1);
let s_end: u64 = sbase.wrapping_add(_AC!() << 30i32);
// Allow segments to share if only one is marked locked
let vm_flags: u64 = (unsafe { (*vma).__slate_anon_3.vm_flags }) & !(BIT!(13i32) | BIT!(19i32));
let svm_flags: u64 = (unsafe { (*svma).__slate_anon_3.vm_flags }) & !(BIT!(13i32) | BIT!(19i32));
// match the virtual addresses, permission and the alignment of the
// page table page.
//
// Also, vma_lock (vm_private_data) is required for sharing.
let __v0: bool = pmd_index(addr) != pmd_index(saddr) || vm_flags != svm_flags || !range_in_vma(svma as *const vm_area_struct, sbase, s_end);
if __v0 || unsafe { (*svma).vm_private_data }.is_null() {
return 0;
}
saddr
}

#[unsafe(no_mangle)]
extern "C" fn want_pmd_share(vma: *mut vm_area_struct, addr: u64) -> bool {
let start: u64 = addr & !(_AC!() << 30i32).wrapping_sub(1);
let end: u64 = start.wrapping_add(_AC!() << 30i32);
// check on proper vm_flags and page table alignment
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) == 0 {
return false;
}
if unsafe { (*vma).vm_private_data }.is_null() {
return false;
}
// vma lock required for sharing
if !range_in_vma(vma as *const vm_area_struct, start, end) {
return false;
}
true
}

/// Determine if start,end range within vma could be mapped by shared pmd.
/// If yes, adjust start and end to cover range associated with possible
/// shared pmd mappings.
#[unsafe(no_mangle)]
extern "C" fn adjust_range_if_pmd_sharing_possible(vma: *mut vm_area_struct, start: *mut u64, end: *mut u64) {
let v_start: u64 = ALIGN!(unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }, 1073741824u64);
let v_end: u64 = ALIGN_DOWN!(unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }, 1073741824u64);
// vma needs to span at least one aligned PUD size, and the range
// must be at least partially within in.
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) == 0 || v_end <= v_start || (unsafe { *end }) <= v_start || (unsafe { *start }) >= v_end {
return;
}
// Extend the range to be PUD aligned for a worst case scenario
if (unsafe { *start }) > v_start {
unsafe {
*start = ALIGN_DOWN!(unsafe { *start }, 1073741824u64);
}
}
if (unsafe { *end }) < v_end {
unsafe {
*end = ALIGN!(unsafe { *end }, 1073741824u64);
}
}
}

/// Search for a shareable pmd page for hugetlb. In any case calls pmd_alloc()
/// and returns the corresponding pte. While this is not necessary for the
/// !shared pmd case because we can allocate the pmd later as well, it makes the
/// code much cleaner. pmd allocation is essential for the shared case because
/// pud has to be populated inside the same i_mmap_rwsem section - otherwise
/// racing tasks could either miss the sharing (see huge_pte_offset) or select a
/// bad pmd for sharing.
#[unsafe(no_mangle)]
extern "C" fn huge_pmd_share(mm: *mut mm_struct, vma: *mut vm_area_struct, addr: u64, pud: *mut __SlateRecord_pgtable_types_h_368) -> *mut __SlateRecord_pgtable_64_types_h_21 {
let mut pte: core::mem::MaybeUninit<*mut __SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(pte) as *mut *mut __SlateRecord_pgtable_64_types_h_21;
let mut spte: *mut __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
let mut saddr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(saddr) as *mut u64;
let mut svma: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(svma) as *mut *mut vm_area_struct;
let mut idx: u64 = unsafe { core::mem::zeroed() };
let mut mapping: *mut address_space = unsafe { core::mem::zeroed() };
unsafe {
mapping = (*(*vma).vm_file).f_mapping;
idx = linear_page_index(vma as *const vm_area_struct, addr);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
spte = core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
i_mmap_lock_read(mapping);
*__slate_slot_3 = mapping_rmap_tree_iter_first(mapping, idx, idx);
'__join_5: {
loop {
if !(*__slate_slot_3).is_null() {
if *__slate_slot_3 != vma {
*__slate_slot_2 = page_table_shareable(*__slate_slot_3, vma, addr, idx);
if *__slate_slot_2 != 0 {
spte = hugetlb_walk(*__slate_slot_3, *__slate_slot_2, vma_mmu_pagesize(*__slate_slot_3));
if !spte.is_null() {
break;
}
}
}
*__slate_slot_3 = mapping_rmap_tree_iter_next(*__slate_slot_3, idx, idx);
} else {
break '__join_5;
}
}
ptdesc_pmd_pts_inc(virt_to_ptdesc(spte as *const ()));
}
if !spte.is_null() {
spin_lock(core::ptr::addr_of_mut!((*mm).__slate_anon_0.page_table_lock));
if pud_none(*pud) != 0 {
pud_populate(mm, pud, ((spte as u64) & !(_AC!() << 12i32).wrapping_sub(1)) as *mut __SlateRecord_pgtable_64_types_h_22);
mm_inc_nr_pmds(mm);
} else {
ptdesc_pmd_pts_dec(virt_to_ptdesc(spte as *const ()));
}
spin_unlock(core::ptr::addr_of_mut!((*mm).__slate_anon_0.page_table_lock));
}
*__slate_slot_0 = pmd_alloc(mm, pud, addr) as *mut __SlateRecord_pgtable_64_types_h_21;
i_mmap_unlock_read(mapping);
return *__slate_slot_0;
}
}

fn __huge_pmd_unshare(tlb: *mut mmu_gather, vma: *mut vm_area_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21, check_locks: bool) -> i32 {
let sz: u64 = huge_page_size(hstate_vma(vma) as *const hstate);
let mm: *mut mm_struct = unsafe { (*vma).vm_mm };
let pgd: *mut __SlateRecord_pgtable_types_h_295 = pgd_offset!(mm, addr);
let p4d: *mut __SlateRecord_pgtable_types_h_342 = p4d_offset(pgd, addr);
let pud: *mut __SlateRecord_pgtable_types_h_368 = pud_offset(p4d, addr);
if sz != _AC!() << 21i32 {
return 0;
}
if !ptdesc_pmd_is_shared(virt_to_ptdesc(ptep as *const ())) {
return 0;
}
i_mmap_assert_write_locked(unsafe { (*(*vma).vm_file).f_mapping });
if check_locks {
hugetlb_vma_assert_locked(vma);
}
pud_clear!(pud);
tlb_unshare_pmd_ptdesc(tlb, virt_to_ptdesc(ptep as *const ()), addr);
mm_dec_nr_pmds(mm);
1
}

/// huge_pmd_unshare - Unmap a pmd table if it is shared by multiple users
/// @tlb: the current mmu_gather.
/// @vma: the vma covering the pmd table.
/// @addr: the address we are trying to unshare.
/// @ptep: pointer into the (pmd) page table.
///
/// Called with the page table lock held, the i_mmap_rwsem held in write mode
/// and the hugetlb vma lock held in write mode.
///
/// Note: The caller must call huge_pmd_unshare_flush() before dropping the
/// i_mmap_rwsem.
///
/// Returns: 1 if it was a shared PMD table and it got unmapped, or 0 if it
/// 	    was not a shared PMD table.
#[unsafe(no_mangle)]
extern "C" fn huge_pmd_unshare(tlb: *mut mmu_gather, vma: *mut vm_area_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21) -> i32 {
return __huge_pmd_unshare(tlb, vma, addr, ptep, true); // check_locks=
}

/// huge_pmd_unshare_flush - Complete a sequence of huge_pmd_unshare() calls
/// @tlb: the current mmu_gather.
/// @vma: the vma covering the pmd table.
///
/// Perform necessary TLB flushes or IPI broadcasts to synchronize PMD table
/// unsharing with concurrent page table walkers.
///
/// This function must be called after a sequence of huge_pmd_unshare()
/// calls while still holding the i_mmap_rwsem.
#[unsafe(no_mangle)]
extern "C" fn huge_pmd_unshare_flush(tlb: *mut mmu_gather, vma: *mut vm_area_struct) {
// We must synchronize page table unsharing such that nobody will
// try reusing a previously-shared page table while it might still
// be in use by previous sharers (TLB, GUP_fast).
i_mmap_assert_write_locked(unsafe { (*(*vma).vm_file).f_mapping });
tlb_flush_unshared_tables(tlb);
}

#[unsafe(no_mangle)]
extern "C" fn huge_pte_alloc(mm: *mut mm_struct, vma: *mut vm_area_struct, addr: u64, sz: u64) -> *mut __SlateRecord_pgtable_64_types_h_21 {
let mut pte: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
let pgd: *mut __SlateRecord_pgtable_types_h_295 = pgd_offset!(mm, addr);
let p4d: *mut __SlateRecord_pgtable_types_h_342 = p4d_alloc(mm, pgd, addr);
if p4d.is_null() {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
let pud: *mut __SlateRecord_pgtable_types_h_368 = pud_alloc(mm, p4d, addr);
if !pud.is_null() {
if sz == _AC!() << 30i32 {
pte = pud as *mut __SlateRecord_pgtable_64_types_h_21;
} else {
loop {
if unlikely!(sz != _AC!() << 21i32) != 0 {
loop {
unsafe {
core::arch::asm!("745: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 745b - ., 3; .popsection\n# {0}", const 745 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_huge_pte_alloc_0, sym __slate_asm_string_huge_pte_alloc_1, const 7156 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
let __v5: bool = want_pmd_share(vma, addr) && pud_none(unsafe { *pud }) != 0;
if __v5 {
pte = huge_pmd_share(mm, vma, addr, pud);
} else {
pte = pmd_alloc(mm, pud, addr) as *mut __SlateRecord_pgtable_64_types_h_21;
}
}
}
if !pte.is_null() {
let pteval: __SlateRecord_pgtable_64_types_h_21 = ptep_get_lockless(pte);
loop {
let __v6: bool = pte_present(pteval) != 0 && pte_huge(pteval) == 0;
if __v6 {
loop {
unsafe {
core::arch::asm!("746: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 746b - ., 3; .popsection\n# {0}", const 746 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_huge_pte_alloc_7, sym __slate_asm_string_huge_pte_alloc_8, const 7167 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
pte
}

/// huge_pte_offset() - Walk the page table to resolve the hugepage
/// entry at address @addr
///
/// Return: Pointer to page table entry (PUD or PMD) for
/// address @addr, or NULL if a !p*d_present() entry is encountered and the
/// size @sz doesn't match the hugepage size at this level of the page
/// table.
#[unsafe(no_mangle)]
extern "C" fn huge_pte_offset(mm: *mut mm_struct, addr: u64, sz: u64) -> *mut __SlateRecord_pgtable_64_types_h_21 {
let pgd: *mut __SlateRecord_pgtable_types_h_295 = pgd_offset!(mm, addr);
if pgd_present(unsafe { *pgd }) == 0 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
let p4d: *mut __SlateRecord_pgtable_types_h_342 = p4d_offset(pgd, addr);
if p4d_present(unsafe { *p4d }) == 0 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
let pud: *mut __SlateRecord_pgtable_types_h_368 = pud_offset(p4d, addr);
if sz == _AC!() << 30i32 {
return pud as *mut __SlateRecord_pgtable_64_types_h_21;
}
// must be pud huge, non-present or none
if pud_present(unsafe { *pud }) == 0 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
// must have a valid entry and size to go further
let pmd: *mut __SlateRecord_pgtable_64_types_h_22 = pmd_offset(pud, addr);
// must be pmd huge, non-present or none
pmd as *mut __SlateRecord_pgtable_64_types_h_21
}

/// Return a mask that can be used to update an address to the last huge
/// page in a page table page mapping size.  Used to skip non-present
/// page table entries when linearly scanning address ranges.  Architectures
/// with unique huge page to page table relationships can define their own
/// version of this routine.
#[unsafe(no_mangle)]
extern "C" fn hugetlb_mask_last_page(h: *mut hstate) -> u64 {
let hp_size: u64 = huge_page_size(h as *const hstate);
if hp_size == _AC!() << 30i32 {
return (_AC!() << 39i32).wrapping_sub(_AC!() << 30i32);
} else {
if hp_size == _AC!() << 21i32 {
return (_AC!() << 30i32).wrapping_sub(_AC!() << 21i32);
} else {
return 0;
}
}
}

/// folio_isolate_hugetlb - try to isolate an allocated hugetlb folio
/// @folio: the folio to isolate
/// @list: the list to add the folio to on success
///
/// Isolate an allocated (refcount > 0) hugetlb folio, marking it as
/// isolated/non-migratable, and moving it from the active list to the
/// given list.
///
/// Isolation will fail if @folio is not an allocated hugetlb folio, or if
/// it is already isolated/non-migratable.
///
/// On success, an additional folio reference is taken that must be dropped
/// using folio_putback_hugetlb() to undo the isolation.
///
/// Return: True if isolation worked, otherwise False.
#[unsafe(no_mangle)]
extern "C" fn folio_isolate_hugetlb(folio: *mut folio, list: *mut list_head) -> bool {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut ret: bool = unsafe { core::mem::zeroed() };
unsafe {
ret = true;
spin_lock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
if !folio_test_hugetlb(folio as *const folio) {
__v1 = true;
} else {
__v1 = !folio_test_hugetlb_migratable(folio);
}
if __v1 {
__v0 = true;
} else {
__v0 = !folio_try_get(folio);
}
if __v0 {
ret = false;
} else {
folio_clear_hugetlb_migratable(folio);
list_move_tail(core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru), list);
}
spin_unlock_irq(core::ptr::addr_of_mut!(hugetlb_lock.0));
return ret;
}
}

#[unsafe(no_mangle)]
extern "C" fn get_hwpoison_hugetlb_folio(folio: *mut folio, hugetlb: *mut bool, unpoison: bool) -> i32 {
let mut ret: i32 = 0;
unsafe {
*hugetlb = false;
}
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
if folio_test_hugetlb(folio as *const folio) {
unsafe {
*hugetlb = true;
}
if folio_test_hugetlb_freed(folio) {
ret = 0;
} else {
if folio_test_hugetlb_migratable(folio) || unpoison {
ret = folio_try_get(folio) as i32;
} else {
ret = 16i32.wrapping_neg();
}
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
ret
}

/// folio_putback_hugetlb - unisolate a hugetlb folio
/// @folio: the isolated hugetlb folio
///
/// Putback/un-isolate the hugetlb folio that was previous isolated using
/// folio_isolate_hugetlb(): marking it non-isolated/migratable and putting it
/// back onto the active list.
///
/// Will drop the additional folio reference obtained through
/// folio_isolate_hugetlb().
#[unsafe(no_mangle)]
extern "C" fn folio_putback_hugetlb(folio: *mut folio) {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
folio_set_hugetlb_migratable(folio);
list_move_tail(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) }, unsafe { core::ptr::addr_of_mut!((*folio_hstate(folio)).hugepage_activelist) });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
folio_put(folio);
}

#[unsafe(no_mangle)]
extern "C" fn move_hugetlb_state(old_folio: *mut folio, new_folio: *mut folio, reason: u32) {
let h: *mut hstate = folio_hstate(old_folio);
unsafe { hugetlb_cgroup_migrate(old_folio, new_folio) };
folio_set_owner_migrate_reason(new_folio, reason);
// transfer temporary state of the new hugetlb folio. This is
// reverse to other transitions because the newpage is going to
// be final while the old one will be freed so it takes over
// the temporary status.
//
// Also note that we have to transfer the per-node surplus state
// here as well otherwise the global surplus count will not match
// the per-node's.
if folio_test_hugetlb_temporary(new_folio) {
let old_nid: i32 = folio_nid(old_folio as *const folio);
let new_nid: i32 = folio_nid(new_folio as *const folio);
folio_set_hugetlb_temporary(old_folio);
folio_clear_hugetlb_temporary(new_folio);
// There is no need to transfer the per-node surplus state
// when we do not cross the node.
if new_nid != old_nid {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
if (unsafe { *((*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32).wrapping_offset(old_nid as isize) }) != 0 {
let __v0: *mut u32 = unsafe { (*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(old_nid as isize);
let __v1: u32 = unsafe { *__v0 };
unsafe {
*__v0 = __v1.wrapping_sub(1);
}
let __v3: *mut u32 = unsafe { (*h).surplus_huge_pages_node.as_mut_ptr() as *mut u32 }.wrapping_offset(new_nid as isize);
let __v4: u32 = unsafe { *__v3 };
unsafe {
*__v3 = __v4.wrapping_add(1);
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
}
}
// Our old folio is isolated and has "migratable" cleared until it
// is putback. As migration succeeded, set the new folio "migratable"
// and add it to the active list.
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
folio_set_hugetlb_migratable(new_folio);
list_move_tail(unsafe { core::ptr::addr_of_mut!((*new_folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.lru) }, unsafe { core::ptr::addr_of_mut!((*folio_hstate(new_folio)).hugepage_activelist) });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(hugetlb_lock.0) });
}

/// If @take_locks is false, the caller must ensure that no concurrent page table
/// access can happen (except for gup_fast() and hardware page walks).
/// If @take_locks is true, we take the hugetlb VMA lock (to lock out things like
/// concurrent page fault handling) and the file rmap lock.
fn hugetlb_unshare_pmds(vma: *mut vm_area_struct, start: u64, end: u64, take_locks: bool) {
let h: *mut hstate = hstate_vma(vma);
let sz: u64 = huge_page_size(h as *const hstate);
let mm: *mut mm_struct = unsafe { (*vma).vm_mm };
let mut range: mmu_notifier_range = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(range) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut tlb: mmu_gather = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tlb) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 128usize);
}
let mut address: u64 = 0;
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(7i32) == 0 {
return;
}
if start >= end {
return;
}
flush_cache_range(vma, start, end);
unsafe { tlb_gather_mmu_vma(core::ptr::addr_of_mut!(tlb), vma) };
// No need to call adjust_range_if_pmd_sharing_possible(), because
// we have already done the PUD_SIZE alignment.
mmu_notifier_range_init(unsafe { &mut (*core::ptr::addr_of_mut!(range)) }, 1u32, 0u32, mm, start, end);
mmu_notifier_invalidate_range_start(core::ptr::addr_of_mut!(range));
if take_locks {
hugetlb_vma_lock_write(vma);
i_mmap_lock_write(unsafe { (*(*vma).vm_file).f_mapping });
} else {
i_mmap_assert_write_locked(unsafe { (*(*vma).vm_file).f_mapping });
}
address = start;
while address < end {
let ptep: *mut __SlateRecord_pgtable_64_types_h_21 = hugetlb_walk(vma, address, sz);
if !ptep.is_null() {
let ptl: *mut spinlock = huge_pte_lock(h, mm, ptep);
__huge_pmd_unshare(core::ptr::addr_of_mut!(tlb), vma, address, ptep, take_locks);
spin_unlock(ptl);
}
let __v1: u64 = address;
address = __v1.wrapping_add(_AC!() << 30i32);
}
huge_pmd_unshare_flush(core::ptr::addr_of_mut!(tlb), vma);
if take_locks {
i_mmap_unlock_write(unsafe { (*(*vma).vm_file).f_mapping });
hugetlb_vma_unlock_write(vma);
}
// No need to call mmu_notifier_arch_invalidate_secondary_tlbs(), see
// Documentation/mm/mmu_notifier.rst.
mmu_notifier_invalidate_range_end(core::ptr::addr_of_mut!(range));
unsafe { tlb_finish_mmu(core::ptr::addr_of_mut!(tlb)) };
}

/// This function will unconditionally remove all the shared pmd pgtable entries
/// within the specific vma for a hugetlbfs memory range.
#[unsafe(no_mangle)]
extern "C" fn hugetlb_unshare_all_pmds(vma: *mut vm_area_struct) {
hugetlb_unshare_pmds(vma, ALIGN!(unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }, 1073741824u64), ALIGN_DOWN!(unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }, 1073741824u64), true); // take_locks =
}

/// For hugetlb, mremap() is an odd edge case - while the VMA copying is
/// performed, we permit both the old and new VMAs to reference the same
/// reservation.
///
/// We fix this up after the operation succeeds, or if a newly allocated VMA
/// is closed as a result of a failure to allocate memory.
#[unsafe(no_mangle)]
extern "C" fn fixup_hugetlb_reservations(vma: *mut vm_area_struct) {
if is_vm_hugetlb_page(vma as *const vm_area_struct) {
clear_vma_resv_huge_pages(vma);
}
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string___unmap_hugepage_range_1: [u8; 1] = *b"\0";

static __slate_asm_string___unmap_hugepage_range_10: [u8; 1] = *b"\0";

static __slate_asm_string___unmap_hugepage_range_11: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string___unmap_hugepage_range_2: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string___unmap_hugepage_range_5: [u8; 1] = *b"\0";

static __slate_asm_string___unmap_hugepage_range_6: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string___vma_reservation_common_1: [u8; 1] = *b"\0";

static __slate_asm_string___vma_reservation_common_2: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_alloc_bootmem_huge_page_2: [u8; 1] = *b"\0";

static __slate_asm_string_alloc_bootmem_huge_page_3: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_folio_set_order_2: [u8; 1] = *b"\0";

static __slate_asm_string_folio_set_order_3: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/internal.h\0";

static __slate_asm_string_gather_bootmem_prealloc_node_4: [u8; 1] = *b"\0";

static __slate_asm_string_gather_bootmem_prealloc_node_5: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_huge_pte_alloc_0: [u8; 1] = *b"\0";

static __slate_asm_string_huge_pte_alloc_1: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_huge_pte_alloc_7: [u8; 1] = *b"\0";

static __slate_asm_string_huge_pte_alloc_8: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugepage_put_subpool_5: [u8; 1] = *b"\0";

static __slate_asm_string_hugepage_put_subpool_6: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugetlb_add_hstate_0: [u8; 1] = *b"\0";

static __slate_asm_string_hugetlb_add_hstate_1: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugetlb_add_hstate_11: [u8; 1] = *b"\0";

static __slate_asm_string_hugetlb_add_hstate_12: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugetlb_add_hstate_5: [u8; 1] = *b"\0";

static __slate_asm_string_hugetlb_add_hstate_6: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugetlb_bootmem_init_migratetype_1: [u8; 1] = *b"\0";

static __slate_asm_string_hugetlb_bootmem_init_migratetype_2: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugetlb_change_protection_41: [u8; 1] = *b"\0";

static __slate_asm_string_hugetlb_change_protection_42: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugetlb_change_protection_43: [u8; 1] = *b"\0";

static __slate_asm_string_hugetlb_change_protection_44: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugetlb_fault_24: [u8; 1] = *b"\0";

static __slate_asm_string_hugetlb_fault_25: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugetlb_init_20: [u8; 1] = *b"\0";

static __slate_asm_string_hugetlb_init_21: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_hugetlb_vm_op_fault_0: [u8; 1] = *b"\0";

static __slate_asm_string_hugetlb_vm_op_fault_1: [u8; 42] = *b"/home/takashi/c-corpus/linux/mm/hugetlb.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_pfn_pte_1: [u8; 1] = *b"\0";

static __slate_asm_string_pfn_pte_2: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/include/asm/pgtable.h\0";

static __slate_asm_string_rwsem_assert_held_write_nolockdep_1: [u8; 1] = *b"\0";

static __slate_asm_string_rwsem_assert_held_write_nolockdep_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/include/linux/rwsem.h\0";
