
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
//! USB hub driver.
//!
//! (C) Copyright 1999 Linus Torvalds
//! (C) Copyright 1999 Johannes Erdfelt
//! (C) Copyright 1999 Gregory P. Smith
//! (C) Copyright 2001 Brad Hards (bhards@bigpond.net.au)
//!
//! Released under the GPLv2 only.
//! @a is a power of 2 value
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
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
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
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! restart by calling sys_restart_syscall
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
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
//! writeable
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
//! The first word is the work queue pointer and the flags rolled into
//! one
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
//! USB directions
//!
//! This bit flag is used in endpoint descriptors' bEndpointAddress field.
//! It's also one of three fields in control requests bRequestType.
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! Generate SD flag indexes
//! Resource type
//! Runtime PM flag argument bits
//! This is arbitrary.
//! From USB 2.0 spec Table 11-13, offset 7, a hub can
//! have up to 255 ports. The most yet reported is 10.
//! Upcoming hardware might raise that limit.
//! Because the arrays need to add a bit for hub status data, we
//! use 31, so plus one evens out to four bytes.
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
//! device can't be reset(e.g morph devices), don't use reset
//! FIELD_GET() - extract a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_reg:  value of entire bitfield
//!
//! FIELD_GET() extracts the field specified by @_mask from the
//! bitfield passed in as @_reg by masking and shifting it down.
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! See the comment in asm-generic/io.h about ioremap_np().
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
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
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
// warning conditions
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
// "acpi" ACPI via MSR
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Is a directory
// Invalid argument
// Text file busy
// No space left on device
// Broken pipe
// Invalid system call number
// Protocol error
// Value too large for defined data type
// Message too long
// Operation not supported on transport endpoint
// Connection reset by peer
// Transport endpoint is not connected
// Cannot send after transport endpoint shutdown
// Connection timed out
// Operation now in progress
// Memory page has hardware error
// Operation is not supported
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// Resource extended types
// No address assigned yet
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
// device needs a pause during initialization, after we read the device
// descriptor
// device can't handle device_qualifier descriptor requests
// device generates spurious wakeup, ignore remote wakeup capability
// device can't handle Link Power Management
// Hub needs extra delay after resetting its port.
// short SET_ADDRESS request timeout
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
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
macro_rules! ACPI_COMPANION {
($dev:expr) => {
to_acpi_device_node!(unsafe { (*$dev).fwnode })
};
}

macro_rules! ACPI_HANDLE {
($dev:expr) => {
acpi_device_handle(ACPI_COMPANION!($dev))
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

macro_rules! GENMASK_INPUT_CHECK__12 {
() => {
core::mem::size_of::<__SlateRecord_find_h_186>() as i32
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
macro_rules! GENMASK_TYPE__3 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__12!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

macro_rules! GENMASK__5 {
($h:expr, $l:expr) => {
GENMASK_TYPE__3!(u64, $h, $l)
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

macro_rules! MKDEV {
($ma:expr, $mi:expr) => {
($ma) << 20i32 | $mi
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PMSG_IS_AUTO {
($msg:expr) => {
$msg.event & 1024 != 0
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

macro_rules! USB_EXT_PORT_RX_LANES {
($p:expr) => {
(($p) & 3840) >> 8i32
};
}

macro_rules! USB_EXT_PORT_TX_LANES {
($p:expr) => {
(($p) & 61440) >> 12i32
};
}

macro_rules! USB_PORT_LPM_TIMEOUT {
($p:expr) => {
(($p) & 255) << 8i32
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

macro_rules! __WRITE_ONCE__6 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u32, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__2 {
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
macro_rules! __native_word__7 {
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

macro_rules! __ratelimit__2 {
($state:expr) => {
unsafe { ___ratelimit($state, c"hub_port_reset".as_ptr() as *const u8) }
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
macro_rules! __sign_use {
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
macro_rules! __sign_use__2 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
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

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
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

macro_rules! arch_atomic_fetch_sub {
($i:expr, $v:expr) => {
arch_atomic_fetch_add($i.wrapping_neg(), $v)
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
let mut __v: *mut usb_hub;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__73!() as isize)) }) as *mut ()) as *mut usb_hub;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__39 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__28!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__40 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__41!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__62 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__78!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// ffz - find first zero bit in word
/// @word: The word to search
///
/// Undefined if no zero exists, so code should check against ~0UL first.
macro_rules! ffz {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ((!$word).trailing_zeros() as i32) as u64 } else { variable_ffz($word) }
};
}

macro_rules! interface_to_usbdev {
($intf:expr) => {
__intf_to_usbdev($intf)
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
macro_rules! list_entry__43 {
($ptr:expr, $type:ty) => {
container_of__40!($ptr, $type)
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

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__6 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, fwnode) as u64
};
}

macro_rules! offsetof__23 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, post_resume_work.work) as u64
};
}

macro_rules! offsetof__28 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, kref) as u64
};
}

macro_rules! offsetof__41 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, clear_list) as u64
};
}

macro_rules! offsetof__46 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, leds.work) as u64
};
}

macro_rules! offsetof__3 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, driver) as u64
};
}

macro_rules! offsetof__61 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, tt.clear_work) as u64
};
}

macro_rules! offsetof__73 {
() => {
core::mem::offset_of!(usb_hub, irq_urb_retry) as u64
};
}

macro_rules! offsetof__78 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, events) as u64
};
}

macro_rules! offsetof__2 {
() => {
0u64
};
}

macro_rules! offsetof__14 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, init_work.work) as u64
};
}

macro_rules! printk_ratelimit {
() => {
unsafe { __printk_ratelimit(c"hub_port_connect".as_ptr() as *const u8) }
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

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits {
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

macro_rules! test_bit__2 {
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

macro_rules! timer_container_of__4 {
($callback_timer:expr) => {
container_of__23!($callback_timer)
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

macro_rules! to_acpi_device_node {
($__fwnode:expr) => {
{
let mut __v: *mut acpi_device;
let __v2: *mut fwnode_handle = $__fwnode;
let __v3: *mut acpi_device = if unsafe { is_acpi_device_node(__v2 as *const fwnode_handle) } {
let mut __v4: *mut acpi_device;
__v4 = ((unsafe { (__v2 as *const u8).wrapping_offset(-(offsetof__6!(acpi_device) as isize)) }) as *mut ()) as *mut acpi_device;
__v4
} else {
core::ptr::null_mut::<acpi_device>()
};
__v = __v3;
__v
}
};
}

macro_rules! to_usb_device__2 {
($__dev:expr) => {
{
let mut __v: *mut usb_device;
__v = ((unsafe { ((($__dev) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(usb_device) as isize)) }) as *mut ()) as *mut usb_device;
__v as *const usb_device
}
};
}

macro_rules! to_usb_device {
($__dev:expr) => {
{
let mut __v: *mut usb_device;
__v = ((unsafe { ((($__dev) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(usb_device) as isize)) }) as *mut ()) as *mut usb_device;
__v
}
};
}

macro_rules! to_usb_driver {
($d:expr) => {
{
let mut __v: *mut usb_driver;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!(usb_driver) as isize)) }) as *mut ()) as *mut usb_driver;
__v
}
};
}

macro_rules! to_usb_interface {
($__dev:expr) => {
{
let mut __v: *mut usb_interface;
__v = ((unsafe { ((($__dev) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(usb_interface) as isize)) }) as *mut ()) as *mut usb_interface;
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

/// USB device locking
macro_rules! usb_lock_device {
($udev:expr) => {
device_lock(unsafe { core::ptr::addr_of_mut!((*$udev).dev) });
};
}

macro_rules! usb_pipecontrol__2 {
($pipe:expr) => {
usb_pipetype__2!($pipe) == 2
};
}

macro_rules! usb_pipeendpoint {
($pipe:expr) => {
($pipe) >> 15i32 & 15
};
}

macro_rules! usb_pipeendpoint__2 {
($pipe:expr) => {
($pipe) >> 15i32 & 15
};
}

macro_rules! usb_pipein__2 {
($pipe:expr) => {
($pipe) & 128
};
}

macro_rules! usb_pipein {
($pipe:expr) => {
($pipe) & 128
};
}

macro_rules! usb_pipetype__2 {
($pipe:expr) => {
($pipe) >> 30i32 & 3
};
}

macro_rules! usb_rcvctrlpipe {
($dev:expr, $endpoint:expr) => {
2147483648u32 | __create_pipe($dev, ($endpoint) as u32) | 128
};
}

macro_rules! usb_rcvintpipe {
($dev:expr, $endpoint:expr) => {
1073741824u32 | __create_pipe($dev, ($endpoint) as u32) | 128
};
}

/// use a define to avoid include chaining to get THIS_MODULE & friends
macro_rules! usb_register {
($driver:expr) => {
unsafe { usb_register_driver($driver, core::ptr::null_mut::<module>(), c"usbcore".as_ptr() as *const u8) }
};
}

/// Create various pipes...
macro_rules! usb_sndctrlpipe {
($dev:expr, $endpoint:expr) => {
2147483648u32 | __create_pipe($dev, ($endpoint) as u32)
};
}

macro_rules! usb_unlock_device {
($udev:expr) => {
device_unlock(unsafe { core::ptr::addr_of_mut!((*$udev).dev) });
};
}

/// work_pending - Find out whether a work item is currently pending
/// @work: The work item in question
macro_rules! work_pending {
($work:expr) => {
test_bit__2!(0i32, (unsafe { core::ptr::addr_of_mut!((*$work).data) }) as *mut u64)
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __invalid_size_argument_for_IOC: u32;
    static mut __cpu_online_mask: cpumask;
    static mut cpuinfo_op: seq_operations;
    static mut cpu_number: i32;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut system_power_efficient_wq: *mut workqueue_struct;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_int: kernel_param_ops;
    static mut param_ops_bool: kernel_param_ops;
    static mut kstat: kernel_stat;
    static mut usbcore_name: *const u8;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_usb_set_device_state: static_call_key;
    static mut __tracepoint_usb_set_device_state: tracepoint;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __printk_ratelimit(arg0: *const u8) -> i32;
    fn kasprintf(arg0: u32, arg1: *const u8, ...) -> *mut u8;
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __compiletime_assert_7();
    fn __compiletime_assert_10();
    fn __compiletime_assert_11();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_34();
    fn ____wrong_branch_error() -> bool;
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn _find_next_zero_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn strlen(arg0: *const u8) -> u64;
    fn add_device_randomness(arg0: *const (), arg1: u64);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn rcu_is_watching() -> bool;
    fn down_read(arg0: *mut rw_semaphore);
    fn up_read(arg0: *mut rw_semaphore);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn ktime_get_mono_fast_ns() -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn timer_delete_sync(arg0: *mut timer_list) -> i32;
    fn alloc_workqueue_noprof(arg0: *const u8, arg1: u32, arg2: i32, ...) -> *mut workqueue_struct;
    fn destroy_workqueue(arg0: *mut workqueue_struct);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn queue_delayed_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut delayed_work, arg3: u64) -> bool;
    fn flush_work(arg0: *mut work_struct) -> bool;
    fn flush_delayed_work(arg0: *mut delayed_work) -> bool;
    fn cancel_delayed_work_sync(arg0: *mut delayed_work) -> bool;
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn kernfs_notify(arg0: *mut kernfs_node);
    fn sysfs_create_link(arg0: *mut kobject, arg1: *mut kobject, arg2: *const u8) -> i32;
    fn sysfs_remove_link(arg0: *mut kobject, arg1: *const u8);
    fn sysfs_notify(arg0: *mut kobject, arg1: *const u8, arg2: *const u8);
    fn kobject_get_path(arg0: *const kobject, arg1: u32) -> *mut u8;
    fn kobject_uevent_env(arg0: *mut kobject, arg1: u32, arg2: *mut *mut u8) -> i32;
    fn usb_speed_string(arg0: u32) -> *const u8;
    fn msleep(arg0: u32);
    fn usleep_range_state(arg0: u64, arg1: u64, arg2: u32);
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_notice(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn device_wakeup_enable(arg0: *mut device) -> i32;
    fn device_wakeup_disable(arg0: *mut device);
    fn device_set_wakeup_capable(arg0: *mut device, arg1: bool);
    fn pm_wakeup_dev_event(arg0: *mut device, arg1: u32, arg2: bool);
    fn device_add(arg0: *mut device) -> i32;
    fn device_del(arg0: *mut device);
    fn put_device(arg0: *mut device);
    fn device_link_del(arg0: *mut device_link);
    fn __pm_runtime_idle(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_suspend(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_resume(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_set_status(arg0: *mut device, arg1: u32) -> i32;
    fn pm_runtime_barrier(arg0: *mut device);
    fn pm_runtime_enable(arg0: *mut device);
    fn __pm_runtime_disable(arg0: *mut device, arg1: bool);
    fn __pm_runtime_use_autosuspend(arg0: *mut device, arg1: bool);
    fn pm_runtime_set_autosuspend_delay(arg0: *mut device, arg1: i32);
    fn __compiletime_assert_459();
    fn usb_get_intf(arg0: *mut usb_interface) -> *mut usb_interface;
    fn usb_put_intf(arg0: *mut usb_interface);
    fn usb_get_dev(arg0: *mut usb_device) -> *mut usb_device;
    fn usb_put_dev(arg0: *mut usb_device);
    fn usb_enable_autosuspend(arg0: *mut usb_device);
    fn usb_disable_autosuspend(arg0: *mut usb_device);
    fn usb_autopm_get_interface(arg0: *mut usb_interface) -> i32;
    fn usb_autopm_put_interface(arg0: *mut usb_interface);
    fn usb_autopm_put_interface_async(arg0: *mut usb_interface);
    fn usb_autopm_get_interface_no_resume(arg0: *mut usb_interface);
    fn usb_autopm_put_interface_no_suspend(arg0: *mut usb_interface);
    fn usb_register_driver(arg0: *mut usb_driver, arg1: *mut module, arg2: *const u8) -> i32;
    fn usb_deregister(arg0: *mut usb_driver);
    fn __compiletime_assert_463();
    fn __compiletime_assert_464();
    fn usb_alloc_urb(arg0: i32, arg1: u32) -> *mut urb;
    fn usb_free_urb(arg0: *mut urb);
    fn usb_submit_urb(arg0: *mut urb, arg1: u32) -> i32;
    fn usb_kill_urb(arg0: *mut urb);
    fn usb_control_msg(arg0: *mut usb_device, arg1: u32, arg2: u8, arg3: u8, arg4: u16, arg5: u16, arg6: *mut (), arg7: u16, arg8: i32) -> i32;
    fn usb_get_descriptor(arg0: *mut usb_device, arg1: u8, arg2: u8, arg3: *mut (), arg4: i32) -> i32;
    fn usb_get_status(arg0: *mut usb_device, arg1: i32, arg2: i32, arg3: i32, arg4: *mut ()) -> i32;
    fn usb_string(arg0: *mut usb_device, arg1: i32, arg2: *mut u8, arg3: u64) -> i32;
    fn usb_cache_string(arg0: *mut usb_device, arg1: i32) -> *mut u8;
    fn usb_set_interface(arg0: *mut usb_device, arg1: i32, arg2: i32) -> i32;
    fn usb_choose_configuration(arg0: *mut usb_device) -> i32;
    fn usb_set_configuration(arg0: *mut usb_device, arg1: i32) -> i32;
    fn usb_hcd_synchronize_unlinks(arg0: *mut usb_device);
    fn usb_hcd_alloc_bandwidth(arg0: *mut usb_device, arg1: *mut usb_host_config, arg2: *mut usb_host_interface, arg3: *mut usb_host_interface) -> i32;
    fn usb_alloc_dev(arg0: *mut usb_device, arg1: *mut usb_bus, arg2: u32) -> *mut usb_device;
    fn usb_get_configuration(arg0: *mut usb_device) -> i32;
    fn is_acpi_device_node(arg0: *const fwnode_handle) -> bool;
    fn usb_create_ep_devs(arg0: *mut device, arg1: *mut usb_host_endpoint, arg2: *mut usb_device) -> i32;
    fn usb_remove_ep_devs(arg0: *mut usb_host_endpoint);
    fn usb_enable_endpoint(arg0: *mut usb_device, arg1: *mut usb_host_endpoint, arg2: bool);
    fn usb_enable_interface(arg0: *mut usb_device, arg1: *mut usb_interface, arg2: bool);
    fn usb_disable_endpoint(arg0: *mut usb_device, arg1: u32, arg2: bool);
    fn usb_disable_interface(arg0: *mut usb_device, arg1: *mut usb_interface, arg2: bool);
    fn usb_disable_device(arg0: *mut usb_device, arg1: i32);
    fn usb_detect_quirks(arg0: *mut usb_device);
    fn usb_detect_interface_quirks(arg0: *mut usb_device);
    fn usb_get_device_descriptor(arg0: *mut usb_device) -> *mut usb_device_descriptor;
    fn usb_set_isoch_delay(arg0: *mut usb_device) -> i32;
    fn usb_get_bos_descriptor(arg0: *mut usb_device) -> i32;
    fn usb_release_bos_descriptor(arg0: *mut usb_device);
    fn usb_forced_unbind_intf(arg0: *mut usb_interface);
    fn usb_unbind_and_rebind_marked_interfaces(arg0: *mut usb_device);
    fn usb_autosuspend_device(arg0: *mut usb_device);
    fn usb_autoresume_device(arg0: *mut usb_device) -> i32;
    fn usb_enable_usb2_hardware_lpm(arg0: *mut usb_device) -> i32;
    fn usb_disable_usb2_hardware_lpm(arg0: *mut usb_device) -> i32;
    fn usb_hub_create_port_device(arg0: *mut usb_hub, arg1: i32) -> i32;
    fn usb_hub_remove_port_device(arg0: *mut usb_hub, arg1: i32);
    fn __compiletime_assert_632();
    fn usb_phy_roothub_notify_connect(arg0: *mut usb_phy_roothub, arg1: i32) -> i32;
    fn usb_phy_roothub_notify_disconnect(arg0: *mut usb_phy_roothub, arg1: i32) -> i32;
    fn __traceiter_usb_set_device_state(arg0: *mut (), arg1: *mut usb_device) -> i32;
    fn __SCT__tp_func_usb_set_device_state(arg0: *mut (), arg1: *mut usb_device) -> i32;
    fn __compiletime_assert_644();
    fn __compiletime_assert_665();
    fn __compiletime_assert_668();
    fn __compiletime_assert_671();
    fn __compiletime_assert_680();
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
    fn __compiletime_assert_698();
    fn __compiletime_assert_699();
    fn __compiletime_assert_700();
    fn __compiletime_assert_701();
    fn __compiletime_assert_702();
    fn __compiletime_assert_703();
    fn __compiletime_assert_704();
    fn __compiletime_assert_705();
    fn __compiletime_assert_706();
    fn __compiletime_assert_720();
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
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
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

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[inline(always)]
fn arch_test_and_clear_bit(mut nr: i64, mut addr: *mut u64) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock  btrq  {2:r}, ({0:r})\nsetc {1}", in(reg) addr, lateout(reg_byte) __t0, in(reg) nr as u64, options(nostack, att_syntax));
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

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
#[inline(always)]
fn variable_ffz(word: u64) -> u64 {
variable__ffs(!word)
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


#[inline(always)]
fn test_and_set_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit(nr, addr)
}

#[inline(always)]
fn test_and_clear_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_clear_bit(nr, addr)
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

fn INIT_LIST_HEAD(list: *mut list_head) {
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
__WRITE_ONCE!((*list).next, list);
if true {
break;
}
}
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
__WRITE_ONCE!((*list).prev, list);
if true {
break;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_186 {
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
unsafe { __compiletime_assert_7() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_10() };
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

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_11() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


// No need to mask any bits for !PAE
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits122U0,
}


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_usb_set_device_state, const 0 as u8, label {
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
    __slate_bits_0: __slate_bits::__SlateBits129U0,
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__2!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
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

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[inline(always)]
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, new: i32) -> bool {
arch_atomic_try_cmpxchg(v, old, new)
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
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
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
fn atomic_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_add_unless(v, a, u)
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
}

#[inline(always)]
fn find_next_zero_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits!(size) {
let mut val: u64 = 0;
if unlikely!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) | !GENMASK__5!(size.wrapping_sub(1), offset);
return if val == !0 { size } else { ffz!(val) };
}
unsafe { _find_next_zero_bit(addr, size, offset) }
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
struct workqueue_struct {
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
struct rseq {
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
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
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
struct sysv_sem {
    undo_list: *mut sem_undo_list,
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
struct sysv_shm {
    shm_clist: list_head,
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
struct plist_head {
    node_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
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
struct gendisk {
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
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits290U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
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
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits71U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits71U1,
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
    __slate_bits_2: __slate_bits::__SlateBits71U2,
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct irq_domain {
}


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
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
struct folio_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


fn kref_init(kref: *mut kref) {
refcount_set(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }, 1i32);
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


fn kref_get(kref: *mut kref) {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*kref).refcount) });
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sg_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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


fn __init_work(work: *mut work_struct, onstack: i32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


fn queue_delayed_work(wq: *mut workqueue_struct, dwork: *mut delayed_work, delay: u64) -> bool {
unsafe { queue_delayed_work_on(64i32, wq, dwork, delay) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct pidfs_attr {
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct atomic_notifier_head {
    lock: spinlock,
    head: *mut notifier_block,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blocking_notifier_head {
    rwsem: rw_semaphore,
    head: *mut notifier_block,
}


// Suits me
// Don't call further
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
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
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct rhash_head {
    next: *mut rhash_head,
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
struct poll_table_struct {
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
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


fn memalloc_noio_save() -> u32 {
memalloc_flags_save(524288u32)
}

fn memalloc_noio_restore(flags: u32) {
memalloc_flags_restore(flags);
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
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
struct proc_dir_entry {
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
struct wake_irq {
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
    __slate_bits_0: __slate_bits::__SlateBits604U0,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
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
struct ep_device {
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
struct mon_bus {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_dev_state {
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
struct scatterlist {
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
struct usb_phy_roothub {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gen_pool {
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
    __slate_bits_0: __slate_bits::__SlateBits673U0,
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


fn sysfs_notify_dirent(kn: *mut kernfs_node) {
unsafe { kernfs_notify(kn) };
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
    __slate_bits_0: __slate_bits::__SlateBits616U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_property_h_397 {
    pointer: *const (),
    value: __SlateRecord_property_h_399,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_property_h_399 {
    u8_data: [u8; 8],
    u16_data: [u16; 4],
    u32_data: [u32; 2],
    u64_data: [u64; 1],
    str: [*const u8; 1],
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
struct cgroup_taskset {
}


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


fn str_enable_disable(v: bool) -> *const u8 {
(if v { c"enable".as_ptr() as *mut u8 } else { c"disable".as_ptr() as *mut u8 }) as *const u8
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


fn str_plural(num: u64) -> *const u8 {
(if num == 1 { c"".as_ptr() as *mut u8 } else { c"s".as_ptr() as *mut u8 }) as *const u8
}

fn kcov_remote_stop() {
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


fn kcov_remote_start_usb(id: u64) {
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
struct usb_device_id {
    match_flags: u16,
    idVendor: u16,
    idProduct: u16,
    bcdDevice_lo: u16,
    bcdDevice_hi: u16,
    bDeviceClass: u8,
    bDeviceSubClass: u8,
    bDeviceProtocol: u8,
    bInterfaceClass: u8,
    bInterfaceSubClass: u8,
    bInterfaceProtocol: u8,
    bInterfaceNumber: u8,
    driver_info: u64,
}


// Some useful macros to use to create struct usb_device_id
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


// to device
// to host
// Standard requests, for the bRequest field of a SETUP packet.
//
// These are qualified by the bRequestType field, so that for example
// TYPE_CLASS or TYPE_VENDOR specific feature flags could be retrieved
// by a GET_STATUS request.
// USB feature flags are written using USB_REQ_{CLEAR,SET}_FEATURE, and
// are read as a bit array returned by USB_REQ_GET_STATUS.  (So there
// are at most sixteen features of each type.)  Hubs may also support a
// new USB_REQ_TEST_AND_SET_FEATURE to put ports into L1 suspend.
// (read only)
// dev may initiate wakeup
// Status Type
// New Feature Selectors as added by USB 3.0
// See USB 3.0 spec Table 9-7
// dev may initiate U1 transition
// dev may initiate U2 transition
// dev may send LTM
// function suspend
// Suspend Options, Table 9-8 USB 3.0 spec
// Interface status, Figure 9-5 USB 3.0 spec
// Descriptor types ... USB 2.0 spec table 9.5
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
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulator {
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


// self powered
// can wakeup
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
fn usb_endpoint_dir_in(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bEndpointAddress }) as i32) & 128 == 128) as i32
}

fn usb_endpoint_xfer_int(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 3) as i32
}

fn usb_endpoint_xfer_isoc(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 1) as i32
}

fn usb_endpoint_is_int_in(epd: *const usb_endpoint_descriptor) -> i32 {
let __v0: bool = usb_endpoint_xfer_int(epd) != 0 && usb_endpoint_dir_in(epd) != 0;
__v0 as i32
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


#[repr(C)]
#[derive(Clone, Copy)]
struct extcon_dev {
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_qualifier_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bcdUSB: u16,
    bDeviceClass: u8,
    bDeviceSubClass: u8,
    bDeviceProtocol: u8,
    bMaxPacketSize0: u8,
    bNumConfigurations: u8,
    bRESERVED: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_gadget {
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
union __SlateRecord_pm_qos_h_109 {
    pnode: plist_node,
    flr: pm_qos_flags_request,
    freq: freq_qos_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_ptm_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_set_sel_req {
    u1_sel: u8,
    u1_pel: u8,
    u2_sel: u16,
    u2_pel: u16,
}


// The Set System Exit Latency control transfer provides one byte each for
// U1 SEL and U1 PEL, so the max exit latency is 0xFF.  U2 SEL and U2 PEL each
// are two bytes long.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_910 {
    r#type: u32,
    value: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_915 {
    r#type: u32,
    length: u32,
    pointer: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_921 {
    r#type: u32,
    length: u32,
    pointer: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_927 {
    r#type: u32,
    count: u32,
    elements: *mut acpi_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_933 {
    r#type: u32,
    actual_type: u32,
    handle: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_939 {
    r#type: u32,
    proc_id: u32,
    pblk_address: u64,
    pblk_length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_946 {
    r#type: u32,
    system_level: u32,
    resource_order: u32,
}


fn usleep_range(min: u64, max: u64) {
unsafe { usleep_range_state(min, max, 2u32) };
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
    __slate_bits_0: __slate_bits::__SlateBits751U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits751U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits751U2,
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
    __slate_bits_3: __slate_bits::__SlateBits751U3,
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
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits960U0,
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
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits964U0,
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
struct acpi_gpio_mapping {
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
    __slate_bits_0: __slate_bits::__SlateBits748U0,
}


fn pm_wakeup_event(dev: *mut device, msec: u32) {
unsafe { pm_wakeup_dev_event(dev, msec, false) };
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hub_h_28 {
    hub: usb_hub_status,
    port: usb_port_status,
}


fn device_init_wakeup(dev: *mut device, enable: bool) -> i32 {
if enable {
unsafe { device_set_wakeup_capable(dev, true) };
return unsafe { device_wakeup_enable(dev) };
}
unsafe { device_wakeup_disable(dev) };
unsafe { device_set_wakeup_capable(dev, false) };
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_96 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_96_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_96_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_96_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_96_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_96_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_105 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_105_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_105_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_105_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_105_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_105_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_125 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_125_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_125_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_125_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_125_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_125_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_130 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_130_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_130_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_130_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_130_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hub_c_130_6 {
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
    pub struct __SlateBits572U0 {
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
    pub struct __SlateBits604U0 {
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
    pub struct __SlateBits71U0 {
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
    pub struct __SlateBits71U1 {
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
    pub struct __SlateBits432U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits673U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits865U0 {
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
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits71U2 {
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
    pub struct __SlateBits130U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits751U0 {
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
    pub struct __SlateBits748U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits751U1 {
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
    pub struct __SlateBits751U2 {
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
    pub struct __SlateBits751U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits616U0 {
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
    pub struct __SlateBits129U0 {
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
    pub struct __SlateBits799U0 {
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
    pub struct __SlateBits795U0 {
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
    pub struct __SlateBits788U0 {
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
    pub struct __SlateBits788U1 {
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
    pub struct __SlateBits788U2 {
        #[bits(1)]
        pub use_generic_driver: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits810U0 {
        #[bits(1)]
        pub poisoned: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits789U0 {
        #[bits(1)]
        pub no_dynamic_id: u32,
        #[bits(1)]
        pub supports_autosuspend: u32,
        #[bits(1)]
        pub disable_hub_initiated_lpm: u32,
        #[bits(1)]
        pub soft_unbind: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits977U0 {
        #[bits(1)]
        pub limited_power: u32,
        #[bits(1)]
        pub quiescing: u32,
        #[bits(1)]
        pub disconnected: u32,
        #[bits(1)]
        pub in_reset: u32,
        #[bits(1)]
        pub quirk_disable_autosuspend: u32,
        #[bits(1)]
        pub quirk_check_port_auto_suspend: u32,
        #[bits(1)]
        pub has_indicators: u32,
        #[bits(1, access = na)]
        pub __slate_pad_7: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits979U0 {
        #[bits(1)]
        pub early_stop: u32,
        #[bits(1)]
        pub ignore_event: u32,
        #[bits(1)]
        pub is_superspeed: u32,
        #[bits(1)]
        pub usb3_lpm_u1_permit: u32,
        #[bits(1)]
        pub usb3_lpm_u2_permit: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits899U0 {
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
    pub struct __SlateBits816U0 {
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
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits951U0 {
        #[bits(1)]
        pub present: u32,
        #[bits(1)]
        pub enabled: u32,
        #[bits(1)]
        pub show_in_ui: u32,
        #[bits(1)]
        pub functional: u32,
        #[bits(1)]
        pub battery_present: u32,
        #[bits(27)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits952U0 {
        #[bits(1)]
        pub dynamic_status: u32,
        #[bits(1)]
        pub removable: u32,
        #[bits(1)]
        pub ejectable: u32,
        #[bits(1)]
        pub power_manageable: u32,
        #[bits(1)]
        pub initialized: u32,
        #[bits(1)]
        pub visited: u32,
        #[bits(1)]
        pub hotplug_notify: u32,
        #[bits(1)]
        pub is_dock_station: u32,
        #[bits(1)]
        pub of_compatible_ok: u32,
        #[bits(1)]
        pub coherent_dma: u32,
        #[bits(1)]
        pub cca_seen: u32,
        #[bits(1)]
        pub enumeration_by_parent: u32,
        #[bits(1)]
        pub honor_deps: u32,
        #[bits(19)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits956U0 {
        #[bits(1)]
        pub hardware_id: u32,
        #[bits(1)]
        pub bus_address: u32,
        #[bits(1)]
        pub platform_id: u32,
        #[bits(1)]
        pub backlight: u32,
        #[bits(28)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits958U0 {
        #[bits(1)]
        pub explicit_get: u32,
        #[bits(1)]
        pub power_resources: u32,
        #[bits(1)]
        pub inrush_current: u32,
        #[bits(1)]
        pub power_removed: u32,
        #[bits(1)]
        pub ignore_parent: u32,
        #[bits(1)]
        pub dsw_present: u32,
        #[bits(26)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits960U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub explicit_set: u8,
        #[bits(6)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits966U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub notifier_present: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits962U0 {
        #[bits(8)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits964U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(7)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits945U0 {
        #[bits(1)]
        pub enabled: bool,
        #[bits(1)]
        pub demand_offline: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
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

fn dev_set_drvdata(dev: *mut device, data: *mut ()) {
unsafe {
(*dev).driver_data = data;
}
}

fn device_enable_async_suspend(dev: *mut device) {
if ((unsafe { (*dev).power.__slate_bits_0.__get_is_prepared() }) as i32) == 0 {
unsafe {
let __t0 = true;
(*dev).power.__slate_bits_0.__set_async_suspend(__t0);
}
}
}

fn device_lock(dev: *mut device) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*dev).mutex)) };
}

fn device_unlock(dev: *mut device) {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*dev).mutex)) };
}

fn dev_set_removable(dev: *mut device, removable: u32) {
unsafe {
(*dev).removable = removable;
}
}

// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
fn pm_suspend_ignore_children(dev: *mut device, enable: bool) {
unsafe {
(*dev).power.__slate_bits_2.__set_ignore_children(enable);
}
}

fn pm_runtime_get_noresume(dev: *mut device) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*dev).power.usage_count) });
}

fn pm_runtime_put_noidle(dev: *mut device) {
atomic_add_unless(unsafe { core::ptr::addr_of_mut!((*dev).power.usage_count) }, 1i32.wrapping_neg(), 0i32);
}

fn pm_runtime_active(dev: *mut device) -> bool {
((unsafe { (*dev).power.runtime_status }) as i32) == 0 || ((unsafe { (*dev).power.__slate_bits_2.__get_disable_depth() }) as i32) != 0
}

fn pm_runtime_mark_last_busy(dev: *mut device) {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_459() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*dev).power.last_busy, unsafe { ktime_get_mono_fast_ns() });
if true {
break;
}
}
}

fn pm_runtime_get_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 4i32) }
}

fn pm_runtime_get_active(dev: *mut device, rpmflags: i32) -> i32 {
let ret: i32 = unsafe { __pm_runtime_resume(dev, 4 | rpmflags) };
if ret < 0 {
pm_runtime_put_noidle(dev);
return ret;
}
0
}

fn pm_runtime_resume_and_get(dev: *mut device) -> i32 {
pm_runtime_get_active(dev, 0i32)
}

fn pm_runtime_put(dev: *mut device) {
unsafe { __pm_runtime_idle(dev, 5i32) };
}

fn pm_runtime_put_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_idle(dev, 4i32) }
}

fn pm_runtime_put_sync_autosuspend(dev: *mut device) -> i32 {
pm_runtime_mark_last_busy(dev);
unsafe { __pm_runtime_suspend(dev, 12i32) }
}

fn pm_runtime_set_active(dev: *mut device) -> i32 {
unsafe { __pm_runtime_set_status(dev, 0u32) }
}

fn pm_runtime_set_suspended(dev: *mut device) -> i32 {
unsafe { __pm_runtime_set_status(dev, 2u32) }
}

fn pm_runtime_disable(dev: *mut device) {
unsafe { __pm_runtime_disable(dev, true) };
}

fn pm_runtime_use_autosuspend(dev: *mut device) {
unsafe { __pm_runtime_use_autosuspend(dev, true) };
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
    __slate_bits_0: __slate_bits::__SlateBits795U0,
    wireless_status: u32,
    wireless_status_work: work_struct,
    dev: device,
    usb_dev: *mut device,
    reset_ws: work_struct,
}


fn usb_get_intfdata(intf: *mut usb_interface) -> *mut () {
dev_get_drvdata((unsafe { core::ptr::addr_of_mut!((*intf).dev) }) as *const device)
}

fn usb_set_intfdata(intf: *mut usb_interface, data: *mut ()) {
dev_set_drvdata(unsafe { core::ptr::addr_of_mut!((*intf).dev) }, data);
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
    __slate_bits_0: __slate_bits::__SlateBits799U0,
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


// For the given port, prefer the old (faster) enumeration scheme.
// Decrease TRSTRCY to 10ms during device enumeration.
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
    __slate_bits_0: __slate_bits::__SlateBits788U0,
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
    __slate_bits_1: __slate_bits::__SlateBits788U1,
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
    __slate_bits_2: __slate_bits::__SlateBits788U2,
}


fn __intf_to_usbdev(intf: *mut usb_interface) -> *mut usb_device {
to_usb_device!(unsafe { (*intf).dev.parent })
}

fn __intf_to_usbdev_const(intf: *const usb_interface) -> *const usb_device {
to_usb_device__2!((unsafe { (*intf).dev.parent }) as *const device)
}

fn usb_mark_last_busy(udev: *mut usb_device) {
pm_runtime_mark_last_busy(unsafe { core::ptr::addr_of_mut!((*udev).dev) });
}

fn usb_device_supports_ltm(udev: *mut usb_device) -> bool {
if ((unsafe { (*udev).speed }) as u32) < 5 || unsafe { (*udev).bos }.is_null() || unsafe { (*(*udev).bos).ss_cap }.is_null() {
return false;
}
((unsafe { (*(*(*udev).bos).ss_cap).bmAttributes }) as i32) & 2i32 != 0
}

// USB_DEVICE_INFO - macro used to describe a class of usb devices
// @cl: bDeviceClass value
// @sc: bDeviceSubClass value
// @pr: bDeviceProtocol value
//
// This macro is used to create a struct usb_device_id that matches a
// specific class of devices.
#[repr(C)]
#[derive(Clone, Copy)]
struct usb_dynids {
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_driver {
    name: *const u8,
    probe: Option<unsafe extern "C" fn(*mut usb_interface, *const usb_device_id) -> i32>,
    disconnect: Option<unsafe extern "C" fn(*mut usb_interface)>,
    unlocked_ioctl: Option<unsafe extern "C" fn(*mut usb_interface, u32, *mut ()) -> i32>,
    suspend: Option<unsafe extern "C" fn(*mut usb_interface, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut usb_interface) -> i32>,
    reset_resume: Option<unsafe extern "C" fn(*mut usb_interface) -> i32>,
    pre_reset: Option<unsafe extern "C" fn(*mut usb_interface) -> i32>,
    post_reset: Option<unsafe extern "C" fn(*mut usb_interface) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut usb_interface)>,
    id_table: *const usb_device_id,
    dev_groups: *mut *const attribute_group,
    dynids: usb_dynids,
    driver: device_driver,
    __slate_bits_0: __slate_bits::__SlateBits789U0,
}


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
    __slate_bits_0: __slate_bits::__SlateBits810U0,
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


fn usb_fill_int_urb(urb: *mut urb, dev: *mut usb_device, pipe: u32, transfer_buffer: *mut (), buffer_length: i32, complete_fn: Option<unsafe extern "C" fn(*mut urb)>, context: *mut (), mut interval: i32) {
unsafe {
(*urb).dev = dev;
}
unsafe {
(*urb).pipe = pipe;
}
unsafe {
(*urb).transfer_buffer = transfer_buffer;
}
unsafe {
(*urb).transfer_buffer_length = buffer_length as u32;
}
unsafe {
(*urb).complete = complete_fn;
}
unsafe {
(*urb).context = context;
}
if ((unsafe { (*dev).speed }) as u32) == 3 || ((unsafe { (*dev).speed }) as u32) >= 5 {
// make sure interval is within allowed range
let mut __v0: i32;
let __UNIQUE_ID_v__460: i32 = interval;
let __UNIQUE_ID_l__461: i32 = 1;
let __UNIQUE_ID_h__462: i32 = 16;
loop {
if !(!statically_true!(__UNIQUE_ID_l__461 > __UNIQUE_ID_h__462)) {
unsafe { __compiletime_assert_463() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3!(__UNIQUE_ID_v__460, __UNIQUE_ID_l__461, __UNIQUE_ID_h__462) == 0 {
unsafe { __compiletime_assert_464() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_v__460 >= __UNIQUE_ID_h__462 { __UNIQUE_ID_h__462 } else { if __UNIQUE_ID_v__460 <= __UNIQUE_ID_l__461 { __UNIQUE_ID_l__461 } else { __UNIQUE_ID_v__460 } };
interval = __v0;
unsafe {
(*urb).interval = 1i32 << interval.wrapping_sub(1);
}
} else {
unsafe {
(*urb).interval = interval;
}
}
unsafe {
(*urb).start_frame = 1i32.wrapping_neg();
}
}

fn usb_get_std_status(dev: *mut usb_device, recip: i32, target: i32, data: *mut ()) -> i32 {
unsafe { usb_get_status(dev, recip, 0i32, target, data) }
}

// timeouts, in milliseconds, used for sending/receiving control messages
// they typically complete within a few frames (msec) after they're issued
// USB identifies 5 second timeouts, maybe more in a few cases, and a few
// slow devices (like some MGE Ellipse UPSes) actually push that limit.
fn __create_pipe(dev: *mut usb_device, endpoint: u32) -> u32 {
(((unsafe { (*dev).devnum }) << 8i32) as u32) | endpoint << 15i32
}

fn usb_pipe_endpoint(dev: *mut usb_device, pipe: u32) -> *mut usb_host_endpoint {
let eps: *mut *mut usb_host_endpoint = if usb_pipein!(pipe) != 0 { unsafe { (*dev).ep_in.as_mut_ptr() as *mut *mut usb_host_endpoint } } else { unsafe { (*dev).ep_out.as_mut_ptr() as *mut *mut usb_host_endpoint } };
unsafe { *eps.wrapping_offset(usb_pipeendpoint!(pipe) as isize) }
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
struct usbdevfs_hub_portinfo {
    nports: u8,
    port: [u8; 127],
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
    __slate_bits_0: __slate_bits::__SlateBits816U0,
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


fn bus_to_hcd(bus: *mut usb_bus) -> *mut usb_hcd {
container_of!(bus, usb_hcd)
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


// See USB 3.1 spec Table 10-5
// Port status type for GetPortStatus requests added in USB 3.1
// See USB 3.1 spec Table 10-12
// Hub class requests
// See USB 2.0 spec Table 11-16
// Hub class additional requests defined by USB 3.0 spec
// See USB 3.0 spec Table 10-6
// Hub Class feature numbers
// See USB 2.0 spec Table 11-17
// L2 suspend
// Port feature selectors added by USB 3.0 spec.
// See USB 3.0 spec Table 10-7
// USB 3.0 hub remote wake mask bits, see table 10-14
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_port_status {
    wPortStatus: u16,
    wPortChange: u16,
    dwExtPortStatus: u32,
}


// wPortStatus bit field
// See USB 2.0 spec Table 11-21
// bits 6 to 7 are reserved
// Additions to wPortStatus bit field from USB 3.0
// See USB 3.0 spec Table 10-10
// Definitions for PORT_LINK_STATE values
// (bits 5-8) in wPortStatus
// wPortChange bit field
// See USB 2.0 spec Table 11-22 and USB 2.0 LPM ECN Table-4.10
// Bits 0 to 5 shown, bits 6 to 15 are reserved
// USB 3.0 wPortChange bit fields
// See USB 3.0 spec Table 10-11
// wHubCharacteristics (masks)
// See USB 2.0 spec Table 11-13, offset 3
// Logical Power Switching Mode mask
// All ports power control at once
// per-port power control
// no power switching
// hub is part of a compound device
// Over-Current Protection Mode mask
// All ports Over-Current reporting
// per-port Over-current reporting
// No Over-current Protection support
// TT Think Time mask
// per-port indicators (LEDs)
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_hub_status {
    wHubStatus: u16,
    wHubChange: u16,
}


// Hub Status & Hub Change bit masks
// See USB 2.0 spec Table 11-19 and Table 11-20
// Bits 0 and 1 for wHubStatus and wHubChange
// Bits 2 to 15 are reserved for both
// Full speed hub
// Hi-speed hub with single TT
// Hi-speed hub with multiple TT
// Super speed hub
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


// port indicator status selectors, tables 11-7 and 11-25
// Transaction Translator Think Times, in bits
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


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_tt_clear {
    clear_list: list_head,
    tt: u32,
    devinfo: u16,
    hcd: *mut usb_hcd,
    ep: *mut usb_host_endpoint,
}


fn onboard_dev_create_pdevs(parent_dev: *mut usb_device, pdev_list: *mut list_head) {
}

fn onboard_dev_destroy_pdevs(pdev_list: *mut list_head) {
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_ehci_cf_port_reset_rwsem: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_wakeup_notification: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_hub_clear_tt_buffer: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_hub_claim_port: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_hub_release_port: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_set_device_state: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_disable_ltm: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_enable_ltm: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_wakeup_enabled_descendants: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_root_hub_lost_power: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_disable_lpm: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_unlocked_disable_lpm: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_enable_lpm: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_unlocked_enable_lpm: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_ep0_reinit: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_reset_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_queue_reset_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_hub_find_child: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous", sym ehci_cf_port_reset_rwsem, sym usb_wakeup_notification, sym usb_hub_clear_tt_buffer, sym usb_hub_claim_port, sym usb_hub_release_port, sym usb_set_device_state, sym usb_disable_ltm, sym usb_enable_ltm, sym usb_wakeup_enabled_descendants, sym usb_root_hub_lost_power, sym usb_disable_lpm, sym usb_unlocked_disable_lpm, sym usb_enable_lpm, sym usb_unlocked_enable_lpm, sym usb_ep0_reinit, sym usb_reset_device, sym usb_queue_reset_device, sym usb_hub_find_child, options(att_syntax));

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
struct software_node_ref_args {
    swnode: *const software_node,
    fwnode: *mut fwnode_handle,
    nargs: u32,
    args: [u64; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct property_entry {
    name: *const u8,
    length: u64,
    is_inline: bool,
    r#type: u32,
    __slate_anon_4: __SlateRecord_property_h_397,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct software_node {
    name: *const u8,
    parent: *const software_node,
    properties: *const property_entry,
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
    __slate_bits_0: __slate_bits::__SlateBits865U0,
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
    __slate_bits_0: __slate_bits::__SlateBits572U0,
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
    __slate_bits_0: __slate_bits::__SlateBits899U0,
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


fn usb_phy_notify_connect(x: *mut usb_phy, speed: u32) -> i32 {
if !x.is_null() && (unsafe { (*x).notify_connect }) != None {
return unsafe { unsafe { (*x).notify_connect }.unwrap()(x, speed) };
} else {
return 0;
}
}

fn usb_phy_notify_disconnect(x: *mut usb_phy, speed: u32) -> i32 {
if !x.is_null() && (unsafe { (*x).notify_disconnect }) != None {
return unsafe { unsafe { (*x).notify_disconnect }.unwrap()(x, speed) };
} else {
return 0;
}
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
struct pm_qos_constraints {
    list: plist_head,
    target_value: i32,
    default_value: i32,
    no_constraint_value: i32,
    r#type: u32,
    notifiers: *mut blocking_notifier_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_flags_request {
    node: list_head,
    flags: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_flags {
    list: list_head,
    effective_flags: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct freq_constraints {
    min_freq: pm_qos_constraints,
    min_freq_notifiers: blocking_notifier_head,
    max_freq: pm_qos_constraints,
    max_freq_notifiers: blocking_notifier_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct freq_qos_request {
    r#type: u32,
    pnode: plist_node,
    qos: *mut freq_constraints,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos_request {
    r#type: u32,
    data: __SlateRecord_pm_qos_h_109,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
    resume_latency: pm_qos_constraints,
    latency_tolerance: pm_qos_constraints,
    freq: freq_constraints,
    flags: pm_qos_flags,
    resume_latency_req: *mut dev_pm_qos_request,
    latency_tolerance_req: *mut dev_pm_qos_request,
    flags_req: *mut dev_pm_qos_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct typec_connector {
    attach: Option<unsafe extern "C" fn(*mut typec_connector, *mut device)>,
    deattach: Option<unsafe extern "C" fn(*mut typec_connector, *mut device)>,
}


fn typec_attach(con: *mut typec_connector, dev: *mut device) {
if !con.is_null() && (unsafe { (*con).attach }) != None {
unsafe { unsafe { (*con).attach }.unwrap()(con, dev) };
}
}

fn typec_deattach(con: *mut typec_connector, dev: *mut device) {
if !con.is_null() && (unsafe { (*con).deattach }) != None {
unsafe { unsafe { (*con).deattach }.unwrap()(con, dev) };
}
}

// Fixed by ACPI spec
#[repr(C)]
#[derive(Clone, Copy)]
union acpi_object {
    r#type: u32,
    integer: __SlateRecord_actypes_h_910,
    string: __SlateRecord_actypes_h_915,
    buffer: __SlateRecord_actypes_h_921,
    package: __SlateRecord_actypes_h_927,
    reference: __SlateRecord_actypes_h_933,
    processor: __SlateRecord_actypes_h_939,
    power_resource: __SlateRecord_actypes_h_946,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_hotplug_profile {
    kobj: kobject,
    scan_dependent: Option<unsafe extern "C" fn(*mut acpi_device) -> i32>,
    notify_online: Option<unsafe extern "C" fn(*mut acpi_device)>,
    __slate_bits_0: __slate_bits::__SlateBits945U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_scan_handler {
    list_node: list_head,
    ids: *const acpi_device_id,
    r#match: Option<unsafe extern "C" fn(*const u8, *mut *const acpi_device_id) -> bool>,
    attach: Option<unsafe extern "C" fn(*mut acpi_device, *const acpi_device_id) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut acpi_device)>,
    post_eject: Option<unsafe extern "C" fn(*mut acpi_device)>,
    bind: Option<unsafe extern "C" fn(*mut device)>,
    unbind: Option<unsafe extern "C" fn(*mut device)>,
    hotplug: acpi_hotplug_profile,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_hotplug_context {
    self_: *mut acpi_device,
    notify: Option<unsafe extern "C" fn(*mut acpi_device, u32) -> i32>,
    uevent: Option<unsafe extern "C" fn(*mut acpi_device, u32)>,
    fixup: Option<unsafe extern "C" fn(*mut acpi_device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_status {
    __slate_bits_0: __slate_bits::__SlateBits951U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_flags {
    __slate_bits_0: __slate_bits::__SlateBits952U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_dir {
    entry: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_type {
    __slate_bits_0: __slate_bits::__SlateBits956U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_pnp {
    bus_id: [u8; 8],
    instance_no: i32,
    r#type: acpi_pnp_type,
    bus_address: u64,
    unique_id: *mut u8,
    ids: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power_flags {
    __slate_bits_0: __slate_bits::__SlateBits958U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power_state {
    resources: list_head,
    flags: __SlateRecord_acpi_bus_h_249,
    power: i32,
    latency: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power {
    state: i32,
    flags: acpi_device_power_flags,
    states: [acpi_device_power_state; 5],
    state_for_enumeration: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf_flags {
    __slate_bits_0: __slate_bits::__SlateBits962U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf_state {
    flags: __SlateRecord_acpi_bus_h_281,
    power: u8,
    performance: u8,
    latency: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf {
    state: i32,
    flags: acpi_device_perf_flags,
    state_count: i32,
    states: *mut acpi_device_perf_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_flags {
    __slate_bits_0: __slate_bits::__SlateBits966U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_context {
    func: Option<unsafe extern "C" fn(*mut acpi_device_wakeup_context)>,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup {
    gpe_device: *mut (),
    gpe_number: u64,
    sleep_state: u64,
    resources: list_head,
    flags: acpi_device_wakeup_flags,
    context: acpi_device_wakeup_context,
    ws: *mut wakeup_source,
    prepare_count: i32,
    enable_count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_data {
    pointer: *const acpi_object,
    properties: list_head,
    of_compatible: *const acpi_object,
    subnodes: list_head,
}


// The maximum expected number of CSI-2 data lanes.
//
// This number is not expected to ever have to be equal to or greater than the
// number of bits in an unsigned long variable, but if it needs to be increased
// above that limit, code will need to be adjusted accordingly.
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_software_node_port {
    port_name: [u8; 9],
    data_lanes: [u32; 8],
    lane_polarities: [u32; 9],
    link_frequencies: [u64; 8],
    port_nr: u32,
    crs_csi2_local: bool,
    port_props: [property_entry; 2],
    ep_props: [property_entry; 8],
    remote_ep: [software_node_ref_args; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_software_nodes {
    dev_props: [property_entry; 6],
    nodes: *mut software_node,
    nodeptrs: *mut *const software_node,
    ports: *mut acpi_device_software_node_port,
    num_ports: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device {
    pld_crc: u32,
    device_type: i32,
    handle: *mut (),
    fwnode: fwnode_handle,
    wakeup_list: list_head,
    del_list: list_head,
    status: acpi_device_status,
    flags: acpi_device_flags,
    pnp: acpi_device_pnp,
    power: acpi_device_power,
    wakeup: acpi_device_wakeup,
    performance: acpi_device_perf,
    dir: acpi_device_dir,
    data: acpi_device_data,
    handler: *mut acpi_scan_handler,
    hp: *mut acpi_hotplug_context,
    swnodes: *mut acpi_device_software_nodes,
    driver_gpios: *const acpi_gpio_mapping,
    driver_data: *mut (),
    dev: device,
    physical_node_count: u32,
    dep_unmet: u32,
    physical_node_list: list_head,
    physical_node_lock: mutex,
    remove: Option<unsafe extern "C" fn(*mut acpi_device)>,
}


fn acpi_device_handle(adev: *mut acpi_device) -> *mut () {
if !adev.is_null() { unsafe { (*adev).handle } } else { core::ptr::null_mut::<()>() }
}

// DWORD 3
// DWORD 5
fn usb_get_max_power(udev: *mut usb_device, c: *mut usb_host_config) -> u32 {
// SuperSpeed power is in 8 mA units; others are in 2 mA units
let mul: u32 = (if ((unsafe { (*udev).speed }) as u32) >= 5 { 8i32 } else { 2i32 }) as u32;
(((unsafe { (*c).desc.bMaxPower }) as i32) as u32).wrapping_mul(mul)
}

fn is_root_hub(udev: *mut usb_device) -> i32 {
unsafe { (*udev).parent }.is_null() as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct usb_hub {
    intfdev: *mut device,
    hdev: *mut usb_device,
    kref: kref,
    urb: *mut urb,
    buffer: *mut [u8; 8],
    status: *mut __SlateRecord_hub_h_28,
    status_mutex: mutex,
    error: i32,
    nerrors: i32,
    event_bits: [u64; 1],
    change_bits: [u64; 1],
    removed_bits: [u64; 1],
    wakeup_bits: [u64; 1],
    power_bits: [u64; 1],
    child_usage_bits: [u64; 1],
    warm_reset_bits: [u64; 1],
    descriptor: *mut usb_hub_descriptor,
    tt: usb_tt,
    mA_per_port: u32,
    wakeup_enabled_descendants: u32,
    __slate_bits_0: __slate_bits::__SlateBits977U0,
    indicator: [u8; 31],
    leds: delayed_work,
    init_work: delayed_work,
    post_resume_work: delayed_work,
    events: work_struct,
    irq_urb_lock: spinlock,
    irq_urb_retry: timer_list,
    ports: *mut *mut usb_port,
    onboard_devs: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_port {
    child: *mut usb_device,
    dev: device,
    port_owner: *mut usb_dev_state,
    peer: *mut usb_port,
    connector: *mut typec_connector,
    req: *mut dev_pm_qos_request,
    connect_type: u32,
    state: u32,
    state_kn: *mut kernfs_node,
    location: u32,
    status_lock: mutex,
    over_current_count: u32,
    portnum: u8,
    quirks: u32,
    __slate_bits_0: __slate_bits::__SlateBits979U0,
}


fn hub_is_port_power_switchable(hub: *mut usb_hub) -> bool {
let mut hcs: u16 = 0;
if hub.is_null() {
return false;
}
hcs = unsafe { (*(*hub).descriptor).wHubCharacteristics };
(hcs as i32) & 3 < 2
}

fn hub_is_superspeed(hdev: *mut usb_device) -> i32 {
(((unsafe { (*hdev).descriptor.bDeviceProtocol }) as i32) == 3) as i32
}

fn hub_is_superspeedplus(hdev: *mut usb_device) -> i32 {
(((unsafe { (*hdev).descriptor.bDeviceProtocol }) as i32) == 3 && ((unsafe { (*hdev).descriptor.bcdUSB }) as i32) >= 784 && !unsafe { (*hdev).bos }.is_null() && !unsafe { (*(*hdev).bos).ssp_cap }.is_null()) as i32
}

fn hub_power_on_good_delay(hub: *mut usb_hub) -> u32 {
let delay: u32 = ((unsafe { (*(*hub).descriptor).bPwrOn2PwrGood }) as i32).wrapping_mul(2) as u32;
if unsafe { (*(*hub).hdev).parent }.is_null() {
return delay;
} else {
let mut __v0: u32;
let __UNIQUE_ID_x__630: u32 = delay;
let __UNIQUE_ID_y__631: u32 = 100;
loop {
if __types_ok!(__UNIQUE_ID_x__630, __UNIQUE_ID_y__631) == 0 {
unsafe { __compiletime_assert_632() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__630 > __UNIQUE_ID_y__631 { __UNIQUE_ID_x__630 } else { __UNIQUE_ID_y__631 };
return __v0;
}
// root hub
// Wait at least 100 msec for power to become stable
}

fn hub_port_debounce_be_stable(hub: *mut usb_hub, port1: i32) -> i32 {
hub_port_debounce(hub, port1, false)
}

static mut productlist_table: __SlateAlign16<[usb_device_id; 6]> = __SlateAlign16([{
let mut __t0: usb_device_id = unsafe { core::mem::zeroed() };
__t0.match_flags = 112u16;
__t0.bDeviceClass = 9u8;
__t0.bDeviceSubClass = 0u8;
__t0.bDeviceProtocol = 0u8;
__t0
}, {
let mut __t1: usb_device_id = unsafe { core::mem::zeroed() };
__t1.match_flags = 112u16;
__t1.bDeviceClass = 9u8;
__t1.bDeviceSubClass = 0u8;
__t1.bDeviceProtocol = 1u8;
__t1
}, {
let mut __t2: usb_device_id = unsafe { core::mem::zeroed() };
__t2.match_flags = 112u16;
__t2.bDeviceClass = 7u8;
__t2.bDeviceSubClass = 1u8;
__t2.bDeviceProtocol = 1u8;
__t2
}, {
let mut __t3: usb_device_id = unsafe { core::mem::zeroed() };
__t3.match_flags = 112u16;
__t3.bDeviceClass = 7u8;
__t3.bDeviceSubClass = 1u8;
__t3.bDeviceProtocol = 2u8;
__t3
}, {
let mut __t4: usb_device_id = unsafe { core::mem::zeroed() };
__t4.match_flags = 112u16;
__t4.bDeviceClass = 7u8;
__t4.bDeviceSubClass = 1u8;
__t4.bDeviceProtocol = 3u8;
__t4
}, {
let mut __t5: usb_device_id = unsafe { core::mem::zeroed() };
__t5
}]);

fn is_targeted(dev: &mut usb_device) -> i32 {
let mut id: *mut usb_device_id = unsafe { core::ptr::addr_of_mut!(productlist_table.0) as *mut usb_device_id };
// HNP test device is _never_ targeted (see OTG spec 6.6.6)
if ((unsafe { (*(dev as *mut usb_device)).descriptor.idVendor }) as i32) == 6666 && ((unsafe { (*(dev as *mut usb_device)).descriptor.idProduct }) as i32) == 47837 {
return 0;
}
// OTG PET device is always targeted (see OTG 2.0 ECN 6.4.2)
if ((unsafe { (*(dev as *mut usb_device)).descriptor.idVendor }) as i32) == 6666 && ((unsafe { (*(dev as *mut usb_device)).descriptor.idProduct }) as i32) == 512 {
return 1;
}
// NOTE: can't use usb_match_id() since interface caches
// aren't set up yet. this is cut/paste from that code.
id = unsafe { core::ptr::addr_of_mut!(productlist_table.0) as *mut usb_device_id };
while (unsafe { (*id).match_flags }) != 0 {
if !(((unsafe { (*id).match_flags }) as i32) & 1 != 0 && ((unsafe { (*id).idVendor }) as i32) != ((unsafe { (*(dev as *mut usb_device)).descriptor.idVendor }) as i32)) {
if !(((unsafe { (*id).match_flags }) as i32) & 2 != 0 && ((unsafe { (*id).idProduct }) as i32) != ((unsafe { (*(dev as *mut usb_device)).descriptor.idProduct }) as i32)) {
// No need to test id->bcdDevice_lo != 0, since 0 is never
// greater than any unsigned number.
if !(((unsafe { (*id).match_flags }) as i32) & 4 != 0 && ((unsafe { (*id).bcdDevice_lo }) as i32) > ((unsafe { (*(dev as *mut usb_device)).descriptor.bcdDevice }) as i32)) {
if !(((unsafe { (*id).match_flags }) as i32) & 8 != 0 && ((unsafe { (*id).bcdDevice_hi }) as i32) < ((unsafe { (*(dev as *mut usb_device)).descriptor.bcdDevice }) as i32)) {
if !(((unsafe { (*id).match_flags }) as i32) & 16 != 0 && ((unsafe { (*id).bDeviceClass }) as i32) != ((unsafe { (*(dev as *mut usb_device)).descriptor.bDeviceClass }) as i32)) {
if !(((unsafe { (*id).match_flags }) as i32) & 32 != 0 && ((unsafe { (*id).bDeviceSubClass }) as i32) != ((unsafe { (*(dev as *mut usb_device)).descriptor.bDeviceSubClass }) as i32)) {
if !(((unsafe { (*id).match_flags }) as i32) & 64 != 0 && ((unsafe { (*id).bDeviceProtocol }) as i32) != ((unsafe { (*(dev as *mut usb_device)).descriptor.bDeviceProtocol }) as i32)) {
return 1;
}
}
}
}
}
}
}
let __v1: *mut usb_device_id = id;
id = __v1.wrapping_offset(1 as isize);
}
// add other match criteria here ...
// OTG MESSAGE: report errors here, customize to match your product
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(dev as *mut usb_device)).dev) as *const device, c"device v%04x p%04x is not supported\n".as_ptr() as *const u8, (*(dev as *mut usb_device)).descriptor.idVendor as i32, (*(dev as *mut usb_device)).descriptor.idProduct as i32) };
0
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
fn __do_trace_usb_set_device_state(udev: *mut usb_device) {
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
let mut __UNIQUE_ID_guard_641: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_641) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_642: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_642) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_644() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_usb_set_device_state)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_643: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_643;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut usb_device) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut usb_device) -> i32>>(__SCT__tp_func_usb_set_device_state as *const ()) };
unsafe { __v12.unwrap()(__data, udev) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_usb_set_device_state: __SlateAlign16<[u8; 21]> = __SlateAlign16([117u8, 115u8, 98u8, 95u8, 115u8, 101u8, 116u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 95u8, 115u8, 116u8, 97u8, 116u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_usb_set_device_state_645: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_usb_set_device_state) }) as *mut ();

fn trace_usb_set_device_state(udev: *mut usb_device) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_usb_set_device_state.key) }) != 0 {
__do_trace_usb_set_device_state(udev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_646: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// ns
// ns
// ns
// The SET_ADDRESS request timeout will be 500 ms when
// USB_QUIRK_SHORT_SET_ADDRESS_REQ_TIMEOUT quirk flag is set.
// ms
// Give SS hubs 200ms time after wake to train downstream links before
// assuming no port activity and allowing hub to runtime suspend back.
// Root hubs have no upstream hub whose wake propagation needs to be
// accounted for, so they need less time, use 120ms for them.
// ms
// ms
/// Protect struct usb_device->state and ->children members
/// Note: Both are also protected by ->dev.sem, except that ->state can
/// change to USB_STATE_NOTATTACHED even when the semaphore isn't held.
static mut device_state_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

/// workqueue to process hub events
static mut hub_wq: *mut workqueue_struct = unsafe { core::mem::zeroed() };

/// synchronize hub-port add/remove and peering operations
#[unsafe(no_mangle)]
static mut usb_port_peer_mutex: mutex = {
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

/// cycle leds on hubs that aren't blinking for attention
static mut blinkenlights: bool = unsafe { core::mem::zeroed() };

fn __check_blinkenlights() -> *mut bool {
unsafe { core::ptr::addr_of_mut!(blinkenlights) }
}

static __param_str_blinkenlights: __SlateAlign16<[u8; 22]> = __SlateAlign16([117u8, 115u8, 98u8, 99u8, 111u8, 114u8, 101u8, 46u8, 98u8, 108u8, 105u8, 110u8, 107u8, 101u8, 110u8, 108u8, 105u8, 103u8, 104u8, 116u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_blinkenlights: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_blinkenlights.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_bool) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_hub_c_96>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_96_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_96_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_96_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_96_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_96_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(blinkenlights) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_654: [u8; 36] = [117, 115, 98, 99, 111, 114, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 98, 108, 105, 110, 107, 101, 110, 108, 105, 103, 104, 116, 115, 58, 98, 111, 111, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_655: [u8; 54] = [117, 115, 98, 99, 111, 114, 101, 46, 112, 97, 114, 109, 61, 98, 108, 105, 110, 107, 101, 110, 108, 105, 103, 104, 116, 115, 58, 116, 114, 117, 101, 32, 116, 111, 32, 99, 121, 99, 108, 101, 32, 108, 101, 100, 115, 32, 111, 110, 32, 104, 117, 98, 115, 0];

/// Device SATA8000 FW1.0 from DATAST0R Technology Corp requires about
/// 10 seconds to send reply for the initial 64-byte descriptor request.
///
/// define initial 64-byte descriptor request timeout in milliseconds
static mut initial_descriptor_timeout: i32 = 5000;

fn __check_initial_descriptor_timeout() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(initial_descriptor_timeout) }
}

static __param_str_initial_descriptor_timeout: __SlateAlign16<[u8; 35]> = __SlateAlign16([117u8, 115u8, 98u8, 99u8, 111u8, 114u8, 101u8, 46u8, 105u8, 110u8, 105u8, 116u8, 105u8, 97u8, 108u8, 95u8, 100u8, 101u8, 115u8, 99u8, 114u8, 105u8, 112u8, 116u8, 111u8, 114u8, 95u8, 116u8, 105u8, 109u8, 101u8, 111u8, 117u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_initial_descriptor_timeout: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_initial_descriptor_timeout.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_hub_c_105>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_105_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_105_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_105_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_105_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_105_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(initial_descriptor_timeout) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_656: [u8; 48] = [117, 115, 98, 99, 111, 114, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 105, 110, 105, 116, 105, 97, 108, 95, 100, 101, 115, 99, 114, 105, 112, 116, 111, 114, 95, 116, 105, 109, 101, 111, 117, 116, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_657: [u8; 128] = [117, 115, 98, 99, 111, 114, 101, 46, 112, 97, 114, 109, 61, 105, 110, 105, 116, 105, 97, 108, 95, 100, 101, 115, 99, 114, 105, 112, 116, 111, 114, 95, 116, 105, 109, 101, 111, 117, 116, 58, 105, 110, 105, 116, 105, 97, 108, 32, 54, 52, 45, 98, 121, 116, 101, 32, 100, 101, 115, 99, 114, 105, 112, 116, 111, 114, 32, 114, 101, 113, 117, 101, 115, 116, 32, 116, 105, 109, 101, 111, 117, 116, 32, 105, 110, 32, 109, 105, 108, 108, 105, 115, 101, 99, 111, 110, 100, 115, 32, 40, 100, 101, 102, 97, 117, 108, 116, 32, 53, 48, 48, 48, 32, 45, 32, 53, 46, 48, 32, 115, 101, 99, 111, 110, 100, 115, 41, 0];

/// As of 2.6.10 we introduce a new USB device initialization scheme which
/// closely resembles the way Windows works.  Hopefully it will be compatible
/// with a wider range of devices than the old scheme.  However some previously
/// working devices may start giving rise to "device not accepting address"
/// errors; if that happens the user can try the old scheme by adjusting the
/// following module parameters.
///
/// For maximum flexibility there are two boolean parameters to control the
/// hub driver's behavior.  On the first initialization attempt, if the
/// "old_scheme_first" parameter is set then the old scheme will be used,
/// otherwise the new scheme is used.  If that fails and "use_both_schemes"
/// is set, then the driver will make another attempt, using the other scheme.
static mut old_scheme_first: bool = unsafe { core::mem::zeroed() };

fn __check_old_scheme_first() -> *mut bool {
unsafe { core::ptr::addr_of_mut!(old_scheme_first) }
}

static __param_str_old_scheme_first: __SlateAlign16<[u8; 25]> = __SlateAlign16([117u8, 115u8, 98u8, 99u8, 111u8, 114u8, 101u8, 46u8, 111u8, 108u8, 100u8, 95u8, 115u8, 99u8, 104u8, 101u8, 109u8, 101u8, 95u8, 102u8, 105u8, 114u8, 115u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_old_scheme_first: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_old_scheme_first.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_bool) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_hub_c_125>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_125_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_125_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_125_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_125_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_125_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(old_scheme_first) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_658: [u8; 39] = [117, 115, 98, 99, 111, 114, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 111, 108, 100, 95, 115, 99, 104, 101, 109, 101, 95, 102, 105, 114, 115, 116, 58, 98, 111, 111, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_659: [u8; 78] = [117, 115, 98, 99, 111, 114, 101, 46, 112, 97, 114, 109, 61, 111, 108, 100, 95, 115, 99, 104, 101, 109, 101, 95, 102, 105, 114, 115, 116, 58, 115, 116, 97, 114, 116, 32, 119, 105, 116, 104, 32, 116, 104, 101, 32, 111, 108, 100, 32, 100, 101, 118, 105, 99, 101, 32, 105, 110, 105, 116, 105, 97, 108, 105, 122, 97, 116, 105, 111, 110, 32, 115, 99, 104, 101, 109, 101, 0];

static mut use_both_schemes: bool = 1i32 != 0i32;

fn __check_use_both_schemes() -> *mut bool {
unsafe { core::ptr::addr_of_mut!(use_both_schemes) }
}

static __param_str_use_both_schemes: __SlateAlign16<[u8; 25]> = __SlateAlign16([117u8, 115u8, 98u8, 99u8, 111u8, 114u8, 101u8, 46u8, 117u8, 115u8, 101u8, 95u8, 98u8, 111u8, 116u8, 104u8, 95u8, 115u8, 99u8, 104u8, 101u8, 109u8, 101u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_use_both_schemes: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_use_both_schemes.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_bool) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_hub_c_130>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_130_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_130_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_130_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_130_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_hub_c_130_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(use_both_schemes) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_660: [u8; 39] = [117, 115, 98, 99, 111, 114, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 117, 115, 101, 95, 98, 111, 116, 104, 95, 115, 99, 104, 101, 109, 101, 115, 58, 98, 111, 111, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_661: [u8; 96] = [117, 115, 98, 99, 111, 114, 101, 46, 112, 97, 114, 109, 61, 117, 115, 101, 95, 98, 111, 116, 104, 95, 115, 99, 104, 101, 109, 101, 115, 58, 116, 114, 121, 32, 116, 104, 101, 32, 111, 116, 104, 101, 114, 32, 100, 101, 118, 105, 99, 101, 32, 105, 110, 105, 116, 105, 97, 108, 105, 122, 97, 116, 105, 111, 110, 32, 115, 99, 104, 101, 109, 101, 32, 105, 102, 32, 116, 104, 101, 32, 102, 105, 114, 115, 116, 32, 111, 110, 101, 32, 102, 97, 105, 108, 115, 0];

/// Mutual exclusion for EHCI CF initialization.  This interferes with
/// port reset on some companion controllers.
#[unsafe(no_mangle)]
static mut ehci_cf_port_reset_rwsem: rw_semaphore = rw_semaphore { count: __SlateRecord_types_h_197 { counter: 0i64 }, owner: __SlateRecord_types_h_197 { counter: 0i64 }, osq: optimistic_spin_queue { tail: __SlateRecord_types_h_190 { counter: 0i32 } }, wait_lock: raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t0.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t0
} } }, first_waiter: core::ptr::null_mut::<rwsem_waiter>() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_cf_port_reset_rwsem_662: *mut () = (unsafe { core::ptr::addr_of_mut!(ehci_cf_port_reset_rwsem) }) as *mut ();

fn portspeed(hub: &usb_hub, portstatus: i32) -> *mut u8 {
if hub_is_superspeedplus(unsafe { (*(hub as *const usb_hub)).hdev }) != 0 {
return c"10.0 Gb/s".as_ptr() as *mut u8;
}
if hub_is_superspeed(unsafe { (*(hub as *const usb_hub)).hdev }) != 0 {
return c"5.0 Gb/s".as_ptr() as *mut u8;
}
if portstatus & 1024 != 0 {
return c"480 Mb/s".as_ptr() as *mut u8;
} else {
if portstatus & 512 != 0 {
return c"1.5 Mb/s".as_ptr() as *mut u8;
} else {
return c"12 Mb/s".as_ptr() as *mut u8;
}
}
}

/// Note that hdev or one of its children must be locked!
#[unsafe(no_mangle)]
extern "C" fn usb_hub_to_struct_hub(hdev: *mut usb_device) -> *mut usb_hub {
if hdev.is_null() || unsafe { (*hdev).actconfig }.is_null() || (unsafe { (*hdev).maxchild }) == 0 {
return core::ptr::null_mut::<usb_hub>();
}
usb_get_intfdata(unsafe { *((*(*hdev).actconfig).interface.as_mut_ptr() as *mut *mut usb_interface).wrapping_offset(0 as isize) }) as *mut usb_hub
}

#[unsafe(no_mangle)]
extern "C" fn usb_device_supports_lpm(udev: *mut usb_device) -> i32 {
// Some devices have trouble with LPM
if ((unsafe { (*udev).quirks }) as u64) & BIT!(10i32) != 0 {
return 0;
}
// Skip if the device BOS descriptor couldn't be read
if unsafe { (*udev).bos }.is_null() {
return 0;
}
// USB 2.1 (and greater) devices indicate LPM support through
// their USB 2.0 Extended Capabilities BOS descriptor.
if ((unsafe { (*udev).speed }) as u32) == 3 || ((unsafe { (*udev).speed }) as u32) == 2 {
if !unsafe { (*(*udev).bos).ext_cap }.is_null() && 2u32 & unsafe { (*(*(*udev).bos).ext_cap).bmAttributes } != 0 {
return 1;
}
return 0;
}
// According to the USB 3.0 spec, all USB 3.0 devices must support LPM.
// However, there are some that don't, and they set the U1/U2 exit
// latencies to zero.
if unsafe { (*(*udev).bos).ss_cap }.is_null() {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"No LPM exit latency info found, disabling LPM.\n".as_ptr() as *const u8) };
return 0;
}
if ((unsafe { (*(*(*udev).bos).ss_cap).bU1devExitLat }) as i32) == 0 && ((unsafe { (*(*(*udev).bos).ss_cap).bU2DevExitLat }) as i32) == 0 {
if !unsafe { (*udev).parent }.is_null() {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"LPM exit latency is zeroed, disabling LPM.\n".as_ptr() as *const u8) };
} else {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"We don't know the algorithms for LPM for this host, disabling LPM.\n".as_ptr() as *const u8) };
}
return 0;
}
if unsafe { (*udev).parent }.is_null() || ((unsafe { (*(*udev).parent).__slate_bits_0.__get_lpm_capable() }) as i32) != 0 {
return 1;
}
0
}

// Set the Maximum Exit Latency (MEL) for the host to wakup up the path from
// U1/U2, send a PING to the device and receive a PING_RESPONSE.
// See USB 3.1 section C.1.5.2
fn usb_set_lpm_mel(udev: *mut usb_device, udev_lpm_params: *mut usb3_lpm_parameters, __UNIQUE_ID_x__663: u32, hub: *mut usb_hub, hub_lpm_params: *mut usb3_lpm_parameters, __UNIQUE_ID_y__664: u32) {
let mut total_mel: u32 = 0;
// tMEL1. time to transition path from host to device into U0.
// MEL for parent already contains the delay up to parent, so only add
// the exit latency for the last link (pick the slower exit latency),
// and the hub header decode latency. See USB 3.1 section C 2.2.1
// Store MEL in nanoseconds
let mut __v0: u32;
loop {
if __types_ok!(__UNIQUE_ID_x__663, __UNIQUE_ID_y__664) == 0 {
unsafe { __compiletime_assert_665() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__663 > __UNIQUE_ID_y__664 { __UNIQUE_ID_x__663 } else { __UNIQUE_ID_y__664 };
total_mel = unsafe { (*hub_lpm_params).mel }.wrapping_add(__v0.wrapping_mul(1000)).wrapping_add(((unsafe { (*(*hub).descriptor).u.ss.bHubHdrDecLat }) as i32).wrapping_mul(100) as u32);
// tMEL2. Time to submit PING packet. Sum of tTPTransmissionDelay for
// each link + wHubDelay for each hub. Add only for last link.
// tMEL4, the time for PING_RESPONSE to traverse upstream is similar.
// Multiply by 2 to include it as well.
let __v2: u32 = total_mel;
total_mel = __v2.wrapping_add((__le16_to_cpu!(unsafe { (*(*hub).descriptor).u.ss.wHubDelay }) as i32).wrapping_add(40).wrapping_mul(2) as u32);
// tMEL3, tPingResponse. Time taken by device to generate PING_RESPONSE
// after receiving PING. Also add 2100ns as stated in USB 3.1 C 1.5.2.4
// to cover the delay if the PING_RESPONSE is queued behind a Max Packet
// Size DP.
// Note these delays should be added only once for the entire path, so
// add them to the MEL of the device connected to the roothub.
if unsafe { (*(*hub).hdev).parent }.is_null() {
total_mel = total_mel.wrapping_add(2500u32);
}
unsafe {
(*udev_lpm_params).mel = total_mel;
}
}

// Set the maximum Device to Host Exit Latency (PEL) for the device to initiate
// a transition from either U1 or U2.
fn usb_set_lpm_pel(udev: *mut usb_device, udev_lpm_params: *mut usb3_lpm_parameters, __UNIQUE_ID_x__666: u32, hub: *mut usb_hub, hub_lpm_params: *mut usb3_lpm_parameters, __UNIQUE_ID_y__667: u32, port_to_port_exit_latency: u32) {
let mut first_link_pel: u32 = 0;
let mut hub_pel: u32 = 0;
// First, the device sends an LFPS to transition the link between the
// device and the parent hub into U0.  The exit latency is the bigger of
// the device exit latency or the hub exit latency.
let mut __v0: u32;
loop {
if __types_ok!(__UNIQUE_ID_x__666, __UNIQUE_ID_y__667) == 0 {
unsafe { __compiletime_assert_668() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__666 > __UNIQUE_ID_y__667 { __UNIQUE_ID_x__666 } else { __UNIQUE_ID_y__667 };
first_link_pel = __v0.wrapping_mul(1000);
// When the hub starts to receive the LFPS, there is a slight delay for
// it to figure out that one of the ports is sending an LFPS.  Then it
// will forward the LFPS to its upstream link.  The exit latency is the
// delay, plus the PEL that we calculated for this hub.
hub_pel = port_to_port_exit_latency.wrapping_mul(1000).wrapping_add(unsafe { (*hub_lpm_params).pel });
// According to figure C-7 in the USB 3.0 spec, the PEL for this device
// is the greater of the two exit latencies.
let mut __v2: u32;
let __UNIQUE_ID_x__669: u32 = first_link_pel;
let __UNIQUE_ID_y__670: u32 = hub_pel;
loop {
if __types_ok!(__UNIQUE_ID_x__669, __UNIQUE_ID_y__670) == 0 {
unsafe { __compiletime_assert_671() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__669 > __UNIQUE_ID_y__670 { __UNIQUE_ID_x__669 } else { __UNIQUE_ID_y__670 };
unsafe {
(*udev_lpm_params).pel = __v2;
}
}

/// Set the System Exit Latency (SEL) to indicate the total worst-case time from
/// when a device initiates a transition to U0, until when it will receive the
/// first packet from the host controller.
///
/// Section C.1.5.1 describes the four components to this:
///  - t1: device PEL
///  - t2: time for the ERDY to make it from the device to the host.
///  - t3: a host-specific delay to process the ERDY.
///  - t4: time for the packet to make it from the host to the device.
///
/// t3 is specific to both the xHCI host and the platform the host is integrated
/// into.  The Intel HW folks have said it's negligible, FIXME if a different
/// vendor says otherwise.
fn usb_set_lpm_sel(udev: *mut usb_device, udev_lpm_params: *mut usb3_lpm_parameters) {
let mut parent: *mut usb_device = unsafe { core::mem::zeroed() };
let mut num_hubs: u32 = 0;
let mut total_sel: u32 = 0;
// t1 = device PEL
total_sel = unsafe { (*udev_lpm_params).pel };
// How many external hubs are in between the device & the root port.
parent = unsafe { (*udev).parent };
while !unsafe { (*parent).parent }.is_null() {
num_hubs = num_hubs.wrapping_add(1);
parent = unsafe { (*parent).parent };
}
// t2 = 2.1us + 250ns * (num_hubs - 1)
if num_hubs > 0 {
total_sel = total_sel.wrapping_add(2100u32.wrapping_add(250u32.wrapping_mul(num_hubs.wrapping_sub(1))));
}
// t4 = 250ns * num_hubs
total_sel = total_sel.wrapping_add(250u32.wrapping_mul(num_hubs));
unsafe {
(*udev_lpm_params).sel = total_sel;
}
}

fn usb_set_lpm_parameters(udev: *mut usb_device) {
// Appendix C, section C.2.2.2, says that there is a slight delay from
// when the parent hub notices the downstream port is trying to
// transition to U0 to when the hub initiates a U0 transition on its
// upstream port.  The section says the delays are tPort2PortU1EL and
// tPort2PortU2EL, but it doesn't define what they are.
//
// The hub chapter, sections 10.4.2.4 and 10.4.2.5 seem to be talking
// about the same delays.  Use the maximum delay calculations from those
// sections.  For U1, it's tHubPort2PortExitLat, which is 1us max.  For
// U2, it's tHubPort2PortExitLat + U2DevExitLat - U1DevExitLat.  I
// assume the device exit latencies they are talking about are the hub
// exit latencies.
//
// What do we do if the U2 exit latency is less than the U1 exit
// latency?  It's possible, although not likely...
let mut port_to_port_delay: u32 = 1;
let mut udev_u1_del: u32 = 0;
let mut udev_u2_del: u32 = 0;
let mut hub_u1_del: u32 = 0;
let mut hub_u2_del: u32 = 0;
if ((unsafe { (*udev).__slate_bits_0.__get_lpm_capable() }) as i32) == 0 || ((unsafe { (*udev).speed }) as u32) < 5 {
return;
}
// Skip if the device BOS descriptor couldn't be read
if unsafe { (*udev).bos }.is_null() {
return;
}
let hub: *mut usb_hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
// It doesn't take time to transition the roothub into U0, since it
// doesn't have an upstream link.
if hub.is_null() {
return;
}
udev_u1_del = (unsafe { (*(*(*udev).bos).ss_cap).bU1devExitLat }) as u32;
udev_u2_del = (unsafe { (*(*(*udev).bos).ss_cap).bU2DevExitLat }) as u32;
hub_u1_del = (unsafe { (*(*(*(*udev).parent).bos).ss_cap).bU1devExitLat }) as u32;
hub_u2_del = (unsafe { (*(*(*(*udev).parent).bos).ss_cap).bU2DevExitLat }) as u32;
usb_set_lpm_mel(udev, unsafe { core::ptr::addr_of_mut!((*udev).u1_params) }, udev_u1_del, hub, unsafe { core::ptr::addr_of_mut!((*(*udev).parent).u1_params) }, hub_u1_del);
usb_set_lpm_mel(udev, unsafe { core::ptr::addr_of_mut!((*udev).u2_params) }, udev_u2_del, hub, unsafe { core::ptr::addr_of_mut!((*(*udev).parent).u2_params) }, hub_u2_del);
usb_set_lpm_pel(udev, unsafe { core::ptr::addr_of_mut!((*udev).u1_params) }, udev_u1_del, hub, unsafe { core::ptr::addr_of_mut!((*(*udev).parent).u1_params) }, hub_u1_del, port_to_port_delay);
if hub_u2_del > hub_u1_del {
port_to_port_delay = 1u32.wrapping_add(hub_u2_del).wrapping_sub(hub_u1_del);
} else {
port_to_port_delay = 1u32.wrapping_add(hub_u1_del);
}
usb_set_lpm_pel(udev, unsafe { core::ptr::addr_of_mut!((*udev).u2_params) }, udev_u2_del, hub, unsafe { core::ptr::addr_of_mut!((*(*udev).parent).u2_params) }, hub_u2_del, port_to_port_delay);
// Now that we've got PEL, calculate SEL.
usb_set_lpm_sel(udev, unsafe { core::ptr::addr_of_mut!((*udev).u1_params) });
usb_set_lpm_sel(udev, unsafe { core::ptr::addr_of_mut!((*udev).u2_params) });
}

/// USB 2.0 spec Section 11.24.4.5
fn get_hub_descriptor(hdev: *mut usb_device, desc: *mut usb_hub_descriptor) -> i32 {
let mut ret: i32 = 0;
let mut size: i32 = 0;
let mut dtype: u32 = 0;
if hub_is_superspeed(hdev) != 0 {
dtype = 42u32;
size = 12;
} else {
dtype = 41u32;
size = core::mem::size_of::<usb_hub_descriptor>() as i32;
}
for i in 0i32..3i32 {
ret = unsafe { usb_control_msg(hdev, usb_rcvctrlpipe!(hdev, 0i32), 6u8, 160u8, (dtype << 8i32) as u16, 0u16, desc as *mut (), size as u16, 5000i32) };
if hub_is_superspeed(hdev) != 0 {
if ret == size {
return ret;
}
} else {
if ret >= 9i32 {
// Make sure we have the DeviceRemovable field.
size = 7i32.wrapping_add(((unsafe { (*desc).bNbrPorts }) as i32) / 8).wrapping_add(1);
if ret < size {
return 90i32.wrapping_neg();
}
return ret;
}
}
}
22i32.wrapping_neg()
}

/// USB 2.0 spec Section 11.24.2.1
fn clear_hub_feature(hdev: *mut usb_device, feature: i32) -> i32 {
unsafe { usb_control_msg(hdev, usb_sndctrlpipe!(hdev, 0i32), 1u8, 32u8, feature as u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 1000i32) }
}

/// USB 2.0 spec Section 11.24.2.2
#[unsafe(no_mangle)]
extern "C" fn usb_clear_port_feature(hdev: *mut usb_device, port1: i32, feature: i32) -> i32 {
unsafe { usb_control_msg(hdev, usb_sndctrlpipe!(hdev, 0i32), 1u8, 35u8, feature as u16, port1 as u16, core::ptr::null_mut::<()>(), 0u16, 1000i32) }
}

/// USB 2.0 spec Section 11.24.2.13
fn set_port_feature(hdev: *mut usb_device, port1: i32, feature: i32) -> i32 {
unsafe { usb_control_msg(hdev, usb_sndctrlpipe!(hdev, 0i32), 3u8, 35u8, feature as u16, port1 as u16, core::ptr::null_mut::<()>(), 0u16, 1000i32) }
}

fn to_led_name(selector: i32) -> *mut u8 {
match selector {
1 => {
return c"amber".as_ptr() as *mut u8;
}
2 => {
return c"green".as_ptr() as *mut u8;
}
3 => {
return c"off".as_ptr() as *mut u8;
}
0 => {
return c"auto".as_ptr() as *mut u8;
}
_ => {
return c"??".as_ptr() as *mut u8;
}
}
unsafe { core::mem::zeroed() }
}

/// USB 2.0 spec Section 11.24.2.7.1.10 and table 11-7
/// for info about using port indicators
fn set_port_led(hub: *mut usb_hub, port1: i32, selector: i32) {
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize) };
let status: i32 = set_port_feature(unsafe { (*hub).hdev }, selector << 8i32 | port1, 22i32);
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"indicator %s status %d\n".as_ptr() as *const u8, to_led_name(selector), status) };
}
}

#[unsafe(link_section = ".text.slate_distinct.hub.led_work")]
extern "C" fn led_work(work: *mut work_struct) {
let mut hub: *mut usb_hub = unsafe { core::mem::zeroed() };
let mut __v0: *mut usb_hub;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__46!(usb_hub) as isize)) }) as *mut ()) as *mut usb_hub;
hub = __v0;
let hdev: *mut usb_device = unsafe { (*hub).hdev };
let mut i: u32 = 0;
let mut changed: u32 = 0;
let mut cursor: i32 = 1i32.wrapping_neg();
if ((unsafe { (*hdev).state }) as u32) != 7 || ((unsafe { (*hub).__slate_bits_0.__get_quiescing() }) as i32) != 0 {
return;
}
while i < ((unsafe { (*hdev).maxchild }) as u32) {
'__slate_continue_2: {
let mut selector: u32 = 0;
let mut mode: u32 = 0;
// 30%-50% duty cycle
'__slate_break_1: {
match (unsafe { *((*hub).indicator.as_mut_ptr() as *mut u8).wrapping_offset(i as isize) }) as i32 {
1 => {
// cycle marker
cursor = i as i32;
selector = 0;
mode = 0;
break '__slate_break_1;
// blinking green = sw attention
}
2 => {
selector = 2;
mode = 3;
}
3 => {
selector = 3;
mode = 2;
break '__slate_break_1;
// blinking amber = hw attention
}
4 => {
selector = 1;
mode = 5;
}
5 => {
selector = 3;
mode = 4;
break '__slate_break_1;
// blink green/amber = reserved
}
6 => {
selector = 2;
mode = 7;
}
7 => {
selector = 1;
mode = 6;
}
_ => {
break '__slate_continue_2;
}
}
}
if selector != 0 {
changed = 1;
}
set_port_led(hub, i.wrapping_add(1) as i32, selector as i32);
unsafe {
*((*hub).indicator.as_mut_ptr() as *mut u8).wrapping_offset(i as isize) = mode as u8;
}
}
i = i.wrapping_add(1);
}
if changed == 0 && unsafe { blinkenlights } {
cursor = cursor.wrapping_add(1);
cursor %= unsafe { (*hdev).maxchild };
set_port_led(hub, cursor.wrapping_add(1), 2i32);
unsafe {
*((*hub).indicator.as_mut_ptr() as *mut u8).wrapping_offset(cursor as isize) = 1;
}
changed = changed.wrapping_add(1);
}
if changed != 0 {
queue_delayed_work(unsafe { system_power_efficient_wq }, unsafe { core::ptr::addr_of_mut!((*hub).leds) }, 666u64);
}
}

// use a short timeout for hub/port status fetches
/// USB 2.0 spec Section 11.24.2.6
fn get_hub_status(hdev: *mut usb_device, data: *mut usb_hub_status) -> i32 {
let mut i: i32 = 0;
let mut status: i32 = 110i32.wrapping_neg();
while i < 5 && (status == 110i32.wrapping_neg() || status == 32i32.wrapping_neg()) {
status = unsafe { usb_control_msg(hdev, usb_rcvctrlpipe!(hdev, 0i32), 0u8, 160u8, 0u16, 0u16, data as *mut (), core::mem::size_of::<usb_hub_status>() as u16, 1000i32) };
i = i.wrapping_add(1);
}
status
}

/// USB 2.0 spec Section 11.24.2.7
/// USB 3.1 takes into use the wValue and wLength fields, spec Section 10.16.2.6
fn get_port_status(hdev: *mut usb_device, port1: i32, data: *mut (), value: u16, length: u16) -> i32 {
let mut i: i32 = 0;
let mut status: i32 = 110i32.wrapping_neg();
while i < 5 && (status == 110i32.wrapping_neg() || status == 32i32.wrapping_neg()) {
status = unsafe { usb_control_msg(hdev, usb_rcvctrlpipe!(hdev, 0i32), 0u8, 163u8, value, port1 as u16, data, length, 1000i32) };
i = i.wrapping_add(1);
}
status
}

fn hub_ext_port_status(hub: *mut usb_hub, port1: i32, r#type: i32, status: *mut u16, change: *mut u16, ext_status: *mut u32) -> i32 {
let mut ret: i32 = 0;
let mut len: i32 = 4;
if r#type != 0 {
len = 8;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*hub).status_mutex)) };
ret = get_port_status(unsafe { (*hub).hdev }, port1, (unsafe { core::ptr::addr_of_mut!((*(*hub).status).port) }) as *mut (), r#type as u16, len as u16);
if ret < len {
if ret >= 0 {
ret = 5i32.wrapping_neg();
}
if ret != 19i32.wrapping_neg() {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*hub).intfdev as *const device, c"get_port_status failed: err = %d\n".as_ptr() as *const u8, ret) };
}
}
} else {
unsafe {
*status = (*(*hub).status).port.wPortStatus;
}
unsafe {
*change = (*(*hub).status).port.wPortChange;
}
if r#type != 0 && !ext_status.is_null() {
unsafe {
*ext_status = (*(*hub).status).port.dwExtPortStatus;
}
}
ret = 0;
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*hub).status_mutex)) };
// There is no need to lock status_mutex here, because status_mutex
// protects hub->status, and the phy driver only checks the port
// status without changing the status.
if ret == 0 {
let hdev: *mut usb_device = unsafe { (*hub).hdev };
// Only roothub will be notified of connection changes,
// since the USB PHY only cares about changes at the next
// level.
if is_root_hub(hdev) != 0 {
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*hdev).bus });
let mut connect: bool = false;
let mut connect_change: bool = false;
connect_change = ((unsafe { *change }) as i32) & 1 != 0;
connect = ((unsafe { *status }) as i32) & 1 != 0;
if connect_change && connect {
unsafe { usb_phy_roothub_notify_connect((*hcd).phy_roothub, port1.wrapping_sub(1)) };
} else {
if connect_change {
unsafe { usb_phy_roothub_notify_disconnect((*hcd).phy_roothub, port1.wrapping_sub(1)) };
}
}
}
}
ret
}

#[unsafe(no_mangle)]
extern "C" fn usb_hub_port_status(hub: *mut usb_hub, port1: i32, status: *mut u16, change: *mut u16) -> i32 {
hub_ext_port_status(hub, port1, 0i32, status, change, core::ptr::null_mut::<u32>())
}

fn hub_resubmit_irq_urb(hub: *mut usb_hub) {
let mut flags: u64 = 0;
let mut status: i32 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*hub).irq_urb_lock) }, flags);
if ((unsafe { (*hub).__slate_bits_0.__get_quiescing() }) as i32) != 0 {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*hub).irq_urb_lock) }, flags);
return;
}
status = unsafe { usb_submit_urb((*hub).urb, (BIT!(5i32) as u32) | (BIT!(11i32) as u32)) };
if status != 0 && status != 19i32.wrapping_neg() && status != 1i32.wrapping_neg() && status != 108i32.wrapping_neg() {
while false {
}
unsafe { _dev_err((*hub).intfdev as *const device, c"resubmit --> %d\n".as_ptr() as *const u8, status) };
unsafe { mod_timer(core::ptr::addr_of_mut!((*hub).irq_urb_retry), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(1000)) };
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*hub).irq_urb_lock) }, flags);
}

#[unsafe(link_section = ".text.slate_distinct.hub.hub_retry_irq_urb")]
extern "C" fn hub_retry_irq_urb(t: *mut timer_list) {
let hub: *mut usb_hub = timer_container_of__4!(t);
hub_resubmit_irq_urb(hub);
}

fn kick_hub_wq(hub: *mut usb_hub) {
let __v0: bool = ((unsafe { (*hub).__slate_bits_0.__get_disconnected() }) as i32) != 0 || work_pending!(unsafe { core::ptr::addr_of_mut!((*hub).events) }) != 0;
if __v0 {
return;
}
// Suppress autosuspend until the event is proceed.
//
// Be careful and make sure that the symmetric operation is
// always called. We are here only when there is no pending
// work for this hub. Therefore put the interface either when
// the new work is called or when it is canceled.
let intf: *mut usb_interface = to_usb_interface!(unsafe { (*hub).intfdev });
unsafe { usb_autopm_get_interface_no_resume(intf) };
hub_get(hub);
if queue_work(unsafe { hub_wq }, unsafe { core::ptr::addr_of_mut!((*hub).events) }) {
return;
}
// the work has already been scheduled
unsafe { usb_autopm_put_interface_async(intf) };
hub_put(hub);
}

#[unsafe(no_mangle)]
extern "C" fn usb_kick_hub_wq(hdev: *mut usb_device) {
let hub: *mut usb_hub = usb_hub_to_struct_hub(hdev);
if !hub.is_null() {
kick_hub_wq(hub);
}
}

/// Let the USB core know that a USB 3.0 device has sent a Function Wake Device
/// Notification, which indicates it had initiated remote wakeup.
///
/// USB 3.0 hubs do not report the port link state change from U3 to U0 when the
/// device initiates resume, so the USB core will not receive notice of the
/// resume through the normal hub interrupt URB.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_wakeup_notification")]
extern "C" fn usb_wakeup_notification(hdev: *mut usb_device, portnum: u32) {
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
if hdev.is_null() {
return;
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(device_state_lock) }, flags);
let hub: *mut usb_hub = usb_hub_to_struct_hub(hdev);
if !hub.is_null() {
port_dev = unsafe { *unsafe { (*hub).ports }.wrapping_offset(portnum.wrapping_sub(1) as isize) };
if !port_dev.is_null() && !unsafe { (*port_dev).child }.is_null() {
pm_wakeup_event(unsafe { core::ptr::addr_of_mut!((*(*port_dev).child).dev) }, 0u32);
}
set_bit(portnum as i64, (unsafe { (*hub).wakeup_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
kick_hub_wq(hub);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(device_state_lock) }, flags);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_wakeup_notification_672: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device, u32)>, *mut ()>(Some(usb_wakeup_notification)) };

/// completion function, fires on port status changes and various faults
#[unsafe(link_section = ".text.slate_distinct.hub.hub_irq")]
extern "C" fn hub_irq(urb: *mut urb) {
let mut bits: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(bits) as *mut u64;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut status: i32 = unsafe { core::mem::zeroed() };
let mut hub: *mut usb_hub = unsafe { core::mem::zeroed() };
unsafe {
'__slate_dispatch: {
'__join_7: {
'__join_0: {
hub = (*urb).context as *mut usb_hub;
status = (*urb).status;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: i32 = status;
if __t0 == -2 {
break '__join_7;
} else {
if __t0 == -104 {
break '__join_7;
} else {
if __t0 == -108 {
break '__join_7;
} else {
if __t0 != 0 {
// presumably an error
// Cause a hub reset after 10 consecutive errors
let __v5: i32 = (*hub).nerrors;
let __v4: i32 = __v5.wrapping_add(1);
(*hub).nerrors = __v4;
if __v4 < 10 || (*hub).error != 0 {
break '__join_0;
} else {
(*hub).error = status;
// let hub_wq handle things
}
}
}
}
}
*__slate_slot_7 = 0; // we got data:  port status changed
*__slate_slot_8 = 0;
while *__slate_slot_8 < (*urb).actual_length {
let __v3: u64 = *__slate_slot_7;
let __v2: u64 = __v3 | (*((*(*hub).buffer).as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_8 as isize) as u64) << (*__slate_slot_8).wrapping_mul(8);
*__slate_slot_7 = __v2;
let __v1: u32 = *__slate_slot_8;
*__slate_slot_8 = __v1.wrapping_add(1);
}
*((*hub).event_bits.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = *__slate_slot_7;
(*hub).nerrors = 0;
// Something happened, let hub_wq figure it out
kick_hub_wq(hub);
}
hub_resubmit_irq_urb(hub);
break '__slate_dispatch;
}
return;
}
}
// synchronous unlink
// async unlink
// hardware going away
}

/// USB 2.0 spec Section 11.24.2.3
fn hub_clear_tt_buffer(hdev: *mut usb_device, devinfo: u16, tt: u16) -> i32 {
// Need to clear both directions for control ep
if (devinfo as i32) >> 11i32 & 3 == 0 {
let status: i32 = unsafe { usb_control_msg(hdev, usb_sndctrlpipe!(hdev, 0i32), 8u8, 35u8, ((devinfo as i32) ^ 32768) as u16, tt, core::ptr::null_mut::<()>(), 0u16, 1000i32) };
if status != 0 {
return status;
}
}
unsafe { usb_control_msg(hdev, usb_sndctrlpipe!(hdev, 0i32), 8u8, 35u8, devinfo, tt, core::ptr::null_mut::<()>(), 0u16, 1000i32) }
}

/// enumeration blocks hub_wq for a long time. we use keventd instead, since
/// long blocking there is the exception, not the rule.  accordingly, HCDs
/// talking to TTs must queue control transfers (not just bulk and iso), so
/// both can talk to the same hub concurrently.
#[unsafe(link_section = ".text.slate_distinct.hub.hub_tt_work")]
extern "C" fn hub_tt_work(work: *mut work_struct) {
let mut hub: *mut usb_hub = unsafe { core::mem::zeroed() };
let mut __v0: *mut usb_hub;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__61!(usb_hub) as isize)) }) as *mut ()) as *mut usb_hub;
hub = __v0;
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*hub).tt.lock) }, flags);
while list_empty((unsafe { core::ptr::addr_of_mut!((*hub).tt.clear_list) }) as *const list_head) == 0 {
let mut clear: *mut usb_tt_clear = unsafe { core::mem::zeroed() };
let hdev: *mut usb_device = unsafe { (*hub).hdev };
let mut status: i32 = 0;
let next: *mut list_head = unsafe { (*hub).tt.clear_list.next };
clear = list_entry__43!(next, usb_tt_clear);
list_del(unsafe { core::ptr::addr_of_mut!((*clear).clear_list) });
// drop lock so HCD can concurrently report other TT errors
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*hub).tt.lock) }, flags);
status = hub_clear_tt_buffer(hdev, unsafe { (*clear).devinfo }, (unsafe { (*clear).tt }) as u16);
if status != 0 && status != 19i32.wrapping_neg() {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*hdev).dev) as *const device, c"clear tt %d (%04x) error %d\n".as_ptr() as *const u8, (*clear).tt, (*clear).devinfo as i32, status) };
}
// Tell the HCD, even if the operation failed
let drv: *const hc_driver = unsafe { (*(*clear).hcd).driver };
if (unsafe { (*drv).clear_tt_buffer_complete }) != None {
unsafe { unsafe { (*drv).clear_tt_buffer_complete }.unwrap()((*clear).hcd, (*clear).ep) };
}
unsafe { kfree(clear as *const ()) };
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*hub).tt.lock) }, flags);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*hub).tt.lock) }, flags);
}

/// usb_hub_set_port_power - control hub port's power state
/// @hdev: USB device belonging to the usb hub
/// @hub: target hub
/// @port1: port index
/// @set: expected status
///
/// call this function to control port's power via setting or
/// clearing the port's PORT_POWER feature.
///
/// Return: 0 if successful. A negative error code otherwise.
#[unsafe(no_mangle)]
extern "C" fn usb_hub_set_port_power(hdev: *mut usb_device, hub: *mut usb_hub, port1: i32, set: bool) -> i32 {
let ret: i32 = if set {
set_port_feature(hdev, port1, 8i32)
} else {
usb_clear_port_feature(hdev, port1, 8i32)
};
if ret != 0 {
return ret;
}
if set {
set_bit(port1 as i64, (unsafe { (*hub).power_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
} else {
clear_bit(port1 as i64, (unsafe { (*hub).power_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
}
0
}

/// usb_hub_clear_tt_buffer - clear control/bulk TT state in high speed hub
/// @urb: an URB associated with the failed or incomplete split transaction
///
/// High speed HCDs use this to tell the hub driver that some split control or
/// bulk transaction failed in a way that requires clearing internal state of
/// a transaction translator.  This is normally detected (and reported) from
/// interrupt context.
///
/// It may not be possible for that hub to handle additional full (or low)
/// speed transactions until that state is fully cleared out.
///
/// Return: 0 if successful. A negative error code otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_hub_clear_tt_buffer")]
extern "C" fn usb_hub_clear_tt_buffer(urb: *mut urb) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let udev: *mut usb_device = unsafe { (*urb).dev };
let pipe: i32 = (unsafe { (*urb).pipe }) as i32;
let tt: *mut usb_tt = unsafe { (*udev).tt };
let mut flags: u64 = 0;
let mut clear: *mut usb_tt_clear = unsafe { core::mem::zeroed() };
// we've got to cope with an arbitrary number of pending TT clears,
// since each TT has "at least two" buffers that can need it (and
// there can be many TTs per hub).  even if they're uncommon.
let mut __v0: *mut usb_tt_clear;
let __obj_size: u64 = size_mul(core::mem::size_of::<usb_tt_clear>() as u64, 1u64);
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
__v0 = __v1 as *mut usb_tt_clear;
clear = __v0;
if clear.is_null() {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"can't save CLEAR_TT_BUFFER state\n".as_ptr() as *const u8) };
// FIXME recover somehow ... RESET_TT?
return 12i32.wrapping_neg();
}
// info that CLEAR_TT_BUFFER needs
unsafe {
(*clear).tt = (if (*tt).multi != 0 { (*udev).ttport } else { 1i32 }) as u32;
}
unsafe {
(*clear).devinfo = usb_pipeendpoint__2!(pipe) as u16;
}
unsafe {
(*clear).devinfo = (((*clear).devinfo as i32) | (((*udev).devaddr as u16) as i32) << 4i32) as u16;
}
let __v11: u16 = unsafe { (*clear).devinfo };
let __v12: u16 = ((__v11 as i32) | if usb_pipecontrol__2!(pipe) { 0i32 } else { 4096i32 }) as u16;
unsafe {
(*clear).devinfo = __v12;
}
if usb_pipein__2!(pipe) != 0 {
unsafe {
(*clear).devinfo = (((*clear).devinfo as i32) | 32768i32) as u16;
}
}
// info for completion callback
unsafe {
(*clear).hcd = bus_to_hcd((*udev).bus);
}
unsafe {
(*clear).ep = (*urb).ep;
}
// tell keventd to clear state for this TT
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*tt).lock) }, flags);
list_add_tail(unsafe { core::ptr::addr_of_mut!((*clear).clear_list) }, unsafe { core::ptr::addr_of_mut!((*tt).clear_list) });
schedule_work(unsafe { core::ptr::addr_of_mut!((*tt).clear_work) });
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*tt).lock) }, flags);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_hub_clear_tt_buffer_673: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut urb) -> i32>, *mut ()>(Some(usb_hub_clear_tt_buffer)) };

fn hub_power_on(hub: *mut usb_hub, do_delay: bool) {
let mut port1: i32 = 1;
// Enable power on each port.  Some hubs have reserved values
// of LPSM (> 2) in their descriptors, even though they are
// USB 2.0 hubs.  Some hubs do not implement port-power switching
// but only emulate it.  In all cases, the ports won't work
// unless we send these messages to the hub.
if hub_is_port_power_switchable(hub) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*hub).intfdev as *const device, c"enabling power on all ports\n".as_ptr() as *const u8) };
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*hub).intfdev as *const device, c"trying to enable port power on non-switchable hub\n".as_ptr() as *const u8) };
}
}
while port1 <= unsafe { (*(*hub).hdev).maxchild } {
if test_bit!(port1, unsafe { (*hub).power_bits.as_mut_ptr() as *mut u64 }) != 0 {
set_port_feature(unsafe { (*hub).hdev }, port1, 8i32);
} else {
usb_clear_port_feature(unsafe { (*hub).hdev }, port1, 8i32);
}
port1 = port1.wrapping_add(1);
}
if do_delay {
unsafe { msleep(hub_power_on_good_delay(hub)) };
}
}

fn hub_hub_status(hub: *mut usb_hub, status: &mut u16, change: &mut u16) -> i32 {
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*hub).status_mutex)) };
ret = get_hub_status(unsafe { (*hub).hdev }, unsafe { core::ptr::addr_of_mut!((*(*hub).status).hub) });
if ret < (core::mem::size_of::<usb_hub_status>() as i32) {
if ret >= 0 {
ret = 5i32.wrapping_neg();
}
if ret != 19i32.wrapping_neg() {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*hub).intfdev as *const device, c"get_hub_status failed: err = %d\n".as_ptr() as *const u8, ret) };
}
}
} else {
unsafe {
*(status as *mut u16) = (*(*hub).status).hub.wHubStatus;
}
unsafe {
*(change as *mut u16) = (*(*hub).status).hub.wHubChange;
}
ret = 0;
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*hub).status_mutex)) };
ret
}

fn hub_set_port_link_state(hub: *mut usb_hub, port1: i32, link_status: u32) -> i32 {
set_port_feature(unsafe { (*hub).hdev }, ((port1 as u32) | link_status << 3i32) as i32, 5i32)
}

/// Disable a port and mark a logical connect-change event, so that some
/// time later hub_wq will disconnect() any existing usb_device on the port
/// and will re-enumerate if there actually is a device attached.
fn hub_port_logical_disconnect(hub: *mut usb_hub, port1: i32) {
hub_port_disable(hub, port1, 1i32);
// FIXME let caller ask to power down the port:
//  - some devices won't enumerate without a VBUS power cycle
//  - SRP saves power that way
//  - ... new call, TBD ...
// That's easy if this hub can switch power per-port, and
// hub_wq reactivates the port later (timer, SRP, etc).
// Powerdown must be optional, because of reset/DFU.
set_bit(port1 as i64, (unsafe { (*hub).change_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
kick_hub_wq(hub);
}

/// usb_remove_device - disable a device's port on its parent hub
/// @udev: device to be disabled and removed
/// Context: @udev locked, must be able to sleep.
///
/// After @udev's port has been disabled, hub_wq is notified and it will
/// see that the device has been disconnected.  When the device is
/// physically unplugged and something is plugged in, the events will
/// be received and processed normally.
///
/// Return: 0 if successful. A negative error code otherwise.
#[unsafe(no_mangle)]
extern "C" fn usb_remove_device(udev: *mut usb_device) -> i32 {
let mut ret: i32 = 0;
if unsafe { (*udev).parent }.is_null() {
return 22i32.wrapping_neg();
}
// Can't remove a root hub
let hub: *mut usb_hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
let intf: *mut usb_interface = to_usb_interface!(unsafe { (*hub).intfdev });
ret = unsafe { usb_autopm_get_interface(intf) };
if ret < 0 {
return ret;
}
set_bit((unsafe { (*udev).portnum }) as i64, (unsafe { (*hub).removed_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
hub_port_logical_disconnect(hub, (unsafe { (*udev).portnum }) as i32);
unsafe { usb_autopm_put_interface(intf) };
0
}

fn hub_activate(hub: *mut usb_hub, r#type: u32) {
let mut __slate_compound_23: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __v0: bool = unsafe { core::mem::zeroed() };
// The power session apparently survived the resume.
// If there was an overcurrent or suspend change
// (i.e., remote wakeup request), have hub_wq
// take care of it.  Look at the port link state
// for USB 3.0 hubs, since they don't have a suspend
// change bit, and they don't set the port link change
// bit on device-initiated resume.
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut port_resumed: bool = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
// Make sure a warm-reset request is handled by port_event
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut portchange: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u16 = core::ptr::addr_of_mut!(portchange) as *mut u16;
let mut portstatus: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u16 = core::ptr::addr_of_mut!(portstatus) as *mut u16;
let mut udev: *mut usb_device = unsafe { core::mem::zeroed() };
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut delay: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u32 = core::ptr::addr_of_mut!(delay) as *mut u32;
let mut need_debounce_delay: bool = unsafe { core::mem::zeroed() };
let mut status: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(status) as *mut i32;
let mut port1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(port1) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut hcd: core::mem::MaybeUninit<*mut usb_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut usb_hcd = core::ptr::addr_of_mut!(hcd) as *mut *mut usb_hcd;
let mut hdev: *mut usb_device = unsafe { core::mem::zeroed() };
unsafe {
'__join_5: {
'__join_6: {
'__join_8: {
'__join_14: {
{
hdev = (*hub).hdev;
let __slate_auto_bytes_1 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
need_debounce_delay = false;
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Continue a partial initialization
if (r#type as u32) == 1 || (r#type as u32) == 2 {
device_lock(core::ptr::addr_of_mut!((*hdev).dev));
// Was the hub disconnected while we were waiting?
if ((*hub).__slate_bits_0.__get_disconnected() as i32) != 0 {
break '__join_6;
} else {
if (r#type as u32) != 1 {
break '__join_14;
}
}
} else {
hub_get(hub);
// The superspeed hub except for root hub has to use Hub Depth
// value as an offset into the route string to locate the bits
// it uses to determine the downstream port number. So hub driver
// should send a set hub depth request to superspeed hub after
// the superspeed hub is set configuration in initialization or
// reset procedure.
//
// After a resume, port power should still be on.
// For any other type of activation, turn it on.
if (r#type as u32) != 4 {
if !unsafe { (*hdev).parent }.is_null() {
__v13 = hub_is_superspeed(hdev) != 0;
} else {
__v13 = false;
}
if __v13 {
*__slate_slot_18 = usb_control_msg(hdev, usb_sndctrlpipe!(hdev, 0i32), 12u8, 32u8, ((*hdev).level as i32).wrapping_sub(1) as u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 5000i32);
if *__slate_slot_18 < 0 {
while false {
}
_dev_err((*hub).intfdev as *const device, c"set hub depth failed\n".as_ptr() as *const u8);
}
}
// Speed up system boot by using a delayed_work for the
// hub's initial power-up delays.  This is pretty awkward
// and the implementation looks like a home-brewed sort of
// setjmp/longjmp, but it saves at least 100 ms for each
// root hub (assuming usbcore is compiled into the kernel
// rather than as a module).  It adds up.
//
// This can't be done for HUB_RESUME or HUB_RESET_RESUME
// because for those activation types the ports have to be
// operational when we return.  In theory this could be done
// for HUB_POST_RESET, but it's easier not to.
if (r#type as u32) == 0 {
*__slate_slot_14 = hub_power_on_good_delay(hub);
hub_power_on(hub, false);
loop {
__init_work(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work), 0i32);
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work)).data = *{
__slate_compound_26 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_26)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work)).entry));
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work)).func = Some(hub_init_func2);
if true {
if true {
__timer_init!(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).timer), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
}
}
queue_delayed_work(system_power_efficient_wq, core::ptr::addr_of_mut!((*hub).init_work), msecs_to_jiffies(*__slate_slot_14));
// Suppress autosuspend until init is done
usb_autopm_get_interface_no_resume(to_usb_interface!(unsafe { (*hub).intfdev }));
return;
} else {
if (r#type as u32) == 5 {
// The internal host controller state for the hub device
// may be gone after a host power loss on system resume.
// Update the device's info so the HW knows it's a hub.
*__slate_slot_19 = bus_to_hcd((*hdev).bus);
if (*(*(*__slate_slot_19)).driver).update_hub_device != None {
*__slate_slot_18 = unsafe { (*(*(*__slate_slot_19)).driver).update_hub_device }.unwrap()(*__slate_slot_19, hdev, core::ptr::addr_of_mut!((*hub).tt), (BIT!(10i32) | BIT!(11i32)) as u32);
if *__slate_slot_18 < 0 {
while false {
}
_dev_err((*hub).intfdev as *const device, c"Host not accepting hub info update\n".as_ptr() as *const u8);
while false {
}
_dev_err((*hub).intfdev as *const device, c"LS/FS devices and hubs may not work under this hub\n".as_ptr() as *const u8);
}
}
hub_power_on(hub, true);
} else {
hub_power_on(hub, true);
}
// Give some time on remote wakeup to let links to transit to U0
}
} else {
if hub_is_superspeed((*hub).hdev) != 0 {
msleep(20u32);
}
}
}
}
*__slate_slot_17 = 1;
while *__slate_slot_17 <= (*hdev).maxchild {
let port_dev: *mut usb_port = *unsafe { (*hub).ports }.wrapping_offset((*__slate_slot_17).wrapping_sub(1) as isize);
udev = (*port_dev).child;
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_9 = 0;
*__slate_slot_10 = 0;
*__slate_slot_16 = usb_hub_port_status(hub, *__slate_slot_17, core::ptr::addr_of_mut!(*__slate_slot_10), core::ptr::addr_of_mut!(*__slate_slot_9));
if *__slate_slot_16 != 0 {
break '__join_8;
} else {
if !udev.is_null() || (*__slate_slot_10 as i32) & 1 != 0 {
}
// After anything other than HUB_RESUME (i.e., initialization
// or any sort of reset), every port should be disabled.
// Unconnected ports should likewise be disabled (paranoia),
// and so should ports for which we have no usb_device.
if (*__slate_slot_10 as i32) & 2 != 0 && ((r#type as u32) != 4 || (*__slate_slot_10 as i32) & 1 == 0 || udev.is_null() || ((*udev).state as u32) == 0) {
// USB3 protocol ports will automatically transition
// to Enabled state when detect an USB3.0 device attach.
// Do not disable USB3 protocol ports, just pretend
// power was lost
let __v7: u16 = ((*__slate_slot_10 as i32) & !2) as u16;
*__slate_slot_10 = __v7;
if hub_is_superspeed(hdev) == 0 {
usb_clear_port_feature(hdev, *__slate_slot_17, 1i32);
}
}
if (r#type as u32) == 4 {
__v6 = hub_port_warm_reset_required(hub, *__slate_slot_17, *__slate_slot_10);
} else {
__v6 = false;
}
if __v6 {
set_bit(*__slate_slot_17 as i64, ((*hub).event_bits.as_mut_ptr() as *mut u64) as *mut u64);
}
// Add debounce if USB3 link is in polling/link training state.
// Link will automatically transition to Enabled state after
// link training completes.
if hub_is_superspeed(hdev) != 0 && (*__slate_slot_10 as i32) & 480 == 224 {
need_debounce_delay = true;
}
// Clear status-change flags; we'll debounce later
if (*__slate_slot_9 as i32) & 1 != 0 {
need_debounce_delay = true;
usb_clear_port_feature((*hub).hdev, *__slate_slot_17, 16i32);
}
if (*__slate_slot_9 as i32) & 2 != 0 {
need_debounce_delay = true;
usb_clear_port_feature((*hub).hdev, *__slate_slot_17, 17i32);
}
if (*__slate_slot_9 as i32) & 16 != 0 {
need_debounce_delay = true;
usb_clear_port_feature((*hub).hdev, *__slate_slot_17, 20i32);
}
if (*__slate_slot_9 as i32) & 32 != 0 {
__v5 = hub_is_superspeed((*hub).hdev) != 0;
} else {
__v5 = false;
}
if __v5 {
need_debounce_delay = true;
usb_clear_port_feature((*hub).hdev, *__slate_slot_17, 29i32);
}
// We can forget about a "removed" device when there's a
// physical disconnect or the connect status changes.
if (*__slate_slot_10 as i32) & 1 == 0 || (*__slate_slot_9 as i32) & 1 != 0 {
clear_bit(*__slate_slot_17 as i64, ((*hub).removed_bits.as_mut_ptr() as *mut u64) as *mut u64);
}
if udev.is_null() || ((*udev).state as u32) == 0 {
// Tell hub_wq to disconnect the device or
// check for a new connection or over current condition.
// Based on USB2.0 Spec Section 11.12.5,
// C_PORT_OVER_CURRENT could be set while
// PORT_OVER_CURRENT is not. So check for any of them.
if !udev.is_null() || (*__slate_slot_10 as i32) & 1 != 0 || (*__slate_slot_9 as i32) & 1 != 0 || (*__slate_slot_10 as i32) & 8 != 0 || (*__slate_slot_9 as i32) & 8 != 0 {
set_bit(*__slate_slot_17 as i64, ((*hub).change_bits.as_mut_ptr() as *mut u64) as *mut u64);
}
} else {
if (*__slate_slot_10 as i32) & 2 != 0 {
port_resumed = (*__slate_slot_10 as i32) & 480 == 0;
if *__slate_slot_9 != 0 {
__v3 = true;
} else {
__v3 = hub_is_superspeed((*hub).hdev) != 0 && port_resumed;
}
if __v3 {
set_bit(*__slate_slot_17 as i64, ((*hub).event_bits.as_mut_ptr() as *mut u64) as *mut u64);
}
} else {
if ((*udev).__slate_bits_0.__get_persist_enabled() as i32) != 0 {
(*udev).__slate_bits_1.__set_reset_resume(1u32);
// Don't set the change_bits when the device
// was powered off.
if test_bit!(*__slate_slot_17, (*hub).power_bits.as_mut_ptr() as *mut u64) != 0 {
set_bit(*__slate_slot_17 as i64, ((*hub).change_bits.as_mut_ptr() as *mut u64) as *mut u64);
}
} else {
// The power session is gone; tell hub_wq
usb_set_device_state(udev, 0u32);
set_bit(*__slate_slot_17 as i64, ((*hub).change_bits.as_mut_ptr() as *mut u64) as *mut u64);
}
}
}
let __v2: i32 = *__slate_slot_17;
*__slate_slot_17 = __v2.wrapping_add(1);
}
}
// Check each port and set hub->change_bits to let hub_wq know
// which ports need attention.
// If no port-status-change flags were set, we don't need any
// debouncing.  If flags were set we can try to debounce the
// ports all at once right now, instead of letting hub_wq do them
// one at a time later on.
//
// If any port-status changes do occur during this delay, hub_wq
// will see them later and handle them normally.
if need_debounce_delay {
*__slate_slot_14 = 100;
// Don't do a long sleep inside a workqueue routine
if (r#type as u32) == 1 {
loop {
__init_work(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work), 0i32);
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work)).data = *{
__slate_compound_23 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_23)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work)).entry));
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work)).func = Some(hub_init_func3);
if true {
if true {
__timer_init!(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).timer), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
}
}
queue_delayed_work(system_power_efficient_wq, core::ptr::addr_of_mut!((*hub).init_work), msecs_to_jiffies(*__slate_slot_14));
device_unlock(core::ptr::addr_of_mut!((*hdev).dev));
return;
} else {
msleep(*__slate_slot_14);
}
}
}
(*hub).__slate_bits_0.__set_quiescing(0u32);
*__slate_slot_16 = usb_submit_urb((*hub).urb, (BIT!(10i32) | BIT!(11i32)) as u32);
if *__slate_slot_16 < 0 {
while false {
}
_dev_err((*hub).intfdev as *const device, c"activate --> %d\n".as_ptr() as *const u8, *__slate_slot_16);
}
if ((*hub).__slate_bits_0.__get_has_indicators() as i32) != 0 && blinkenlights {
queue_delayed_work(system_power_efficient_wq, core::ptr::addr_of_mut!((*hub).leds), 666u64);
}
// Scan all ports that need attention
kick_hub_wq(hub);
}
if (r#type as u32) == 1 || (r#type as u32) == 2 {
// Allow autosuspend if it was suppressed
} else {
break '__join_5;
}
}
usb_autopm_put_interface_async(to_usb_interface!(unsafe { (*hub).intfdev }));
device_unlock(core::ptr::addr_of_mut!((*hdev).dev));
}
if (r#type as u32) == 4 {
__v0 = hub_is_superspeed((*hub).hdev) != 0;
} else {
__v0 = false;
}
if __v0 {
// give usb3 downstream links training time after hub resume
usb_autopm_get_interface_no_resume(to_usb_interface!(unsafe { (*hub).intfdev }));
queue_delayed_work(system_power_efficient_wq, core::ptr::addr_of_mut!((*hub).post_resume_work), msecs_to_jiffies((if !unsafe { (*hdev).parent }.is_null() { 200i32 } else { 120i32 }) as u32));
return;
} else {
hub_put(hub);
}
}
// Continues at init2: below
// Continues at init3: below
}

static mut __key__1148: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1327: lock_class_key = unsafe { core::mem::zeroed() };

/// Implement the continuations for the delays above
#[unsafe(link_section = ".text.slate_distinct.hub.hub_init_func2")]
extern "C" fn hub_init_func2(ws: *mut work_struct) {
let mut __v0: *mut usb_hub;
__v0 = ((unsafe { (ws as *const u8).wrapping_offset(-(offsetof__14!(usb_hub) as isize)) }) as *mut ()) as *mut usb_hub;
let hub: *mut usb_hub = __v0;
hub_activate(hub, 1u32);
}

#[unsafe(link_section = ".text.slate_distinct.hub.hub_init_func3")]
extern "C" fn hub_init_func3(ws: *mut work_struct) {
let mut __v0: *mut usb_hub;
__v0 = ((unsafe { (ws as *const u8).wrapping_offset(-(offsetof__14!(usb_hub) as isize)) }) as *mut ()) as *mut usb_hub;
let hub: *mut usb_hub = __v0;
hub_activate(hub, 2u32);
}

#[unsafe(link_section = ".text.slate_distinct.hub.hub_post_resume")]
extern "C" fn hub_post_resume(ws: *mut work_struct) {
let mut __v0: *mut usb_hub;
__v0 = ((unsafe { (ws as *const u8).wrapping_offset(-(offsetof__23!(usb_hub) as isize)) }) as *mut ()) as *mut usb_hub;
let hub: *mut usb_hub = __v0;
unsafe { usb_autopm_put_interface_async(to_usb_interface!(unsafe { (*hub).intfdev })) };
hub_put(hub);
}

fn hub_quiesce(hub: *mut usb_hub, r#type: u32) {
let hdev: *mut usb_device = unsafe { (*hub).hdev };
let mut flags: u64 = 0;
let mut i: i32 = 0;
// hub_wq and related activity won't re-trigger
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*hub).irq_urb_lock) }, flags);
unsafe {
(*hub).__slate_bits_0.__set_quiescing(1u32);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*hub).irq_urb_lock) }, flags);
if (r#type as u32) != 2 {
// Disconnect all the children
i = 0;
while i < unsafe { (*hdev).maxchild } {
if !unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child }.is_null() {
usb_disconnect(unsafe { core::ptr::addr_of_mut!((*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child) });
}
i = i.wrapping_add(1);
}
}
// Stop hub_wq and related activity
unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*hub).irq_urb_retry)) };
unsafe { flush_delayed_work(core::ptr::addr_of_mut!((*hub).post_resume_work)) };
unsafe { usb_kill_urb((*hub).urb) };
if ((unsafe { (*hub).__slate_bits_0.__get_has_indicators() }) as i32) != 0 {
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*hub).leds)) };
}
if !unsafe { (*hub).tt.hub }.is_null() {
unsafe { flush_work(core::ptr::addr_of_mut!((*hub).tt.clear_work)) };
}
}

fn hub_pm_barrier_for_all_ports(hub: *mut usb_hub) {
let mut i: i32 = 0;
while i < unsafe { (*(*hub).hdev).maxchild } {
unsafe { pm_runtime_barrier(core::ptr::addr_of_mut!((*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).dev)) };
i = i.wrapping_add(1);
}
}

/// caller has locked the hub device
#[unsafe(link_section = ".text.slate_distinct.hub.hub_pre_reset")]
extern "C" fn hub_pre_reset(intf: *mut usb_interface) -> i32 {
let hub: *mut usb_hub = usb_get_intfdata(intf) as *mut usb_hub;
hub_quiesce(hub, 1u32);
unsafe {
(*hub).__slate_bits_0.__set_in_reset(1u32);
}
hub_pm_barrier_for_all_ports(hub);
0
}

/// caller has locked the hub device
#[unsafe(link_section = ".text.slate_distinct.hub.hub_post_reset")]
extern "C" fn hub_post_reset(intf: *mut usb_interface) -> i32 {
let hub: *mut usb_hub = usb_get_intfdata(intf) as *mut usb_hub;
unsafe {
(*hub).__slate_bits_0.__set_in_reset(0u32);
}
hub_pm_barrier_for_all_ports(hub);
hub_activate(hub, 3u32);
0
}

fn hub_configure(hub: *mut usb_hub, endpoint: *mut usb_endpoint_descriptor) -> i32 {
let mut __slate_compound_54: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_55: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_56: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_58: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_65: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_67: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_68: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_70: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_71: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_73: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
let mut portstr: core::mem::MaybeUninit<__SlateAlign16<[u8; 32]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut [u8; 32] = core::ptr::addr_of_mut!(portstr) as *mut [u8; 32];
// FIXME for USB 3.0, skip for now
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut delay: u32 = unsafe { core::mem::zeroed() };
// Accumulate wHubDelay + 40ns for every hub in the tree of devices.
// The resulting value will be used for SetIsochDelay() request.
let mut __v21: bool = unsafe { core::mem::zeroed() };
let mut _old_1: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old_1) as *mut *mut alloc_tag;
let mut _res_1: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut () = core::ptr::addr_of_mut!(_res_1) as *mut *mut ();
let mut __obj_size_1: u64 = unsafe { core::mem::zeroed() };
let mut _old_2: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old_2) as *mut *mut alloc_tag;
let mut _res_2: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut () = core::ptr::addr_of_mut!(_res_2) as *mut *mut ();
let mut __obj_size_2: u64 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size_3: u64 = unsafe { core::mem::zeroed() };
let mut maxchild: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut u32 = core::ptr::addr_of_mut!(maxchild) as *mut u32;
let mut full_load: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut u32 = core::ptr::addr_of_mut!(full_load) as *mut u32;
let mut unit_load: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut u32 = core::ptr::addr_of_mut!(unit_load) as *mut u32;
let mut message: *mut u8 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_45: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut maxp: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_46: *mut i32 = core::ptr::addr_of_mut!(maxp) as *mut i32;
let mut pipe: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_47: *mut u32 = core::ptr::addr_of_mut!(pipe) as *mut u32;
let mut wHubCharacteristics: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut u16 = core::ptr::addr_of_mut!(wHubCharacteristics) as *mut u16;
let mut hubchange: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_49: *mut u16 = core::ptr::addr_of_mut!(hubchange) as *mut u16;
let mut hubstatus: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_50: *mut u16 = core::ptr::addr_of_mut!(hubstatus) as *mut u16;
let mut hub_dev: *mut device = unsafe { core::mem::zeroed() };
let mut hdev: *mut usb_device = unsafe { core::mem::zeroed() };
let mut hcd: core::mem::MaybeUninit<*mut usb_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_53: *mut *mut usb_hcd = core::ptr::addr_of_mut!(hcd) as *mut *mut usb_hcd;
unsafe {
let __slate_auto_bytes_4 = __slate_slot_53 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
hdev = (*hub).hdev;
hub_dev = (*hub).intfdev;
let __slate_auto_bytes_5 = __slate_slot_50 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_6 = __slate_slot_49 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_7 = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_8 = __slate_slot_47 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_46 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_45 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_11 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 4usize);
message = c"out of memory".as_ptr() as *mut u8;
let __slate_auto_bytes_12 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_13 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_14 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 4usize);
__obj_size_3 = size_mul(core::mem::size_of::<[u8; 8]>() as u64, 1u64);
let __slate_auto_bytes_15 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_15.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_34 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_35 = _kmalloc_noprof(__obj_size_3, 3264u32, *{
__slate_compound_71 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_71)
});
alloc_tag_restore!();
} else {
*__slate_slot_35 = _kmalloc_noprof(__obj_size_3, 3264u32, *{
__slate_compound_73 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_73)
});
}
'__join_1: {
(*hub).buffer = *__slate_slot_35 as *mut [u8; 8];
if unsafe { (*hub).buffer }.is_null() {
*__slate_slot_45 = 12i32.wrapping_neg();
} else {
__obj_size_2 = size_mul(core::mem::size_of::<__SlateRecord_hub_h_28>() as u64, 1u64);
let __slate_auto_bytes_3 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_28 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_29 = _kmalloc_noprof(__obj_size_2, 3264u32, *{
__slate_compound_68 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_68)
});
alloc_tag_restore!();
} else {
*__slate_slot_29 = _kmalloc_noprof(__obj_size_2, 3264u32, *{
__slate_compound_70 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_70)
});
}
(*hub).status = *__slate_slot_29 as *mut __SlateRecord_hub_h_28;
if unsafe { (*hub).status }.is_null() {
*__slate_slot_45 = 12i32.wrapping_neg();
} else {
loop {
__mutex_init(core::ptr::addr_of_mut!((*hub).status_mutex), c"&hub->status_mutex".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1485));
if true {
break;
}
}
__obj_size_1 = size_mul(core::mem::size_of::<usb_hub_descriptor>() as u64, 1u64);
let __slate_auto_bytes_2 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_22 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_23 = _kzalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_65 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_65)
});
alloc_tag_restore!();
} else {
*__slate_slot_23 = _kzalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_67 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_67)
});
}
(*hub).descriptor = *__slate_slot_23 as *mut usb_hub_descriptor;
if unsafe { (*hub).descriptor }.is_null() {
*__slate_slot_45 = 12i32.wrapping_neg();
} else {
// Request the entire hub descriptor.
// hub->descriptor can handle USB_MAXCHILDREN ports,
// but a (non-SS) hub can/will return fewer bytes here.
*__slate_slot_45 = get_hub_descriptor(hdev, (*hub).descriptor);
if *__slate_slot_45 < 0 {
message = c"can't read hub descriptor".as_ptr() as *mut u8;
} else {
*__slate_slot_40 = 31;
if hub_is_superspeed(hdev) != 0 {
*__slate_slot_40 = min_t__2!(u32, *__slate_slot_40, 15i32);
}
if (((*(*hub).descriptor).bNbrPorts as i32) as u32) > *__slate_slot_40 {
message = c"hub has too many ports!".as_ptr() as *mut u8;
*__slate_slot_45 = 19i32.wrapping_neg();
} else {
if ((*(*hub).descriptor).bNbrPorts as i32) == 0 {
message = c"hub doesn't have any ports!".as_ptr() as *mut u8;
*__slate_slot_45 = 19i32.wrapping_neg();
} else {
if hub_is_superspeed(hdev) != 0 {
__v21 = true;
} else {
__v21 = hub_is_superspeedplus(hdev) != 0;
}
if __v21 {
delay = __le16_to_cpu!(unsafe { (*(*hub).descriptor).u.ss.wHubDelay }) as u32;
if !unsafe { (*hdev).parent }.is_null() {
delay = delay.wrapping_add(((*(*hdev).parent).hub_delay as i32) as u32);
}
delay = delay.wrapping_add(40);
(*hdev).hub_delay = min_t__2!(u32, delay, 65535i32) as u16;
}
*__slate_slot_40 = (*(*hub).descriptor).bNbrPorts as u32;
while false {
}
_dev_info(hub_dev as *const device, c"%d port%s detected\n".as_ptr() as *const u8, *__slate_slot_40, str_plural(*__slate_slot_40 as u64));
__obj_size = size_mul(core::mem::size_of::<*mut usb_port>() as u64, *__slate_slot_40 as u64);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_10 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_11 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_56 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_56)
});
alloc_tag_restore!();
} else {
*__slate_slot_11 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_58 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_58)
});
}
(*hub).ports = *__slate_slot_11 as *mut *mut usb_port;
if unsafe { (*hub).ports }.is_null() {
*__slate_slot_45 = 12i32.wrapping_neg();
} else {
*__slate_slot_48 = (*(*hub).descriptor).wHubCharacteristics;
if hub_is_superspeed(hdev) != 0 {
*__slate_slot_42 = 150;
*__slate_slot_41 = 900;
} else {
*__slate_slot_42 = 100;
*__slate_slot_41 = 500;
}
if (*__slate_slot_48 as i32) & 4 != 0 {
__v9 = hub_is_superspeed(hdev) == 0;
} else {
__v9 = false;
}
if __v9 {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
*__slate_slot_44 = 0;
while (*__slate_slot_44 as u32) < *__slate_slot_40 {
(*__slate_slot_8)[*__slate_slot_44 as usize] = (if ((core::ptr::addr_of_mut!((*(*hub).descriptor).u.hs.DeviceRemovable) as *mut u8).wrapping_offset(((*__slate_slot_44).wrapping_add(1) / 8) as isize).read_unaligned() as i32) & 1i32 << (*__slate_slot_44).wrapping_add(1) % 8 != 0 { 70i32 } else { 82i32 }) as u8;
let __v7: i32 = *__slate_slot_44;
*__slate_slot_44 = __v7.wrapping_add(1);
}
(*__slate_slot_8)[*__slate_slot_40 as usize] = 0;
}
loop {
spinlock_check(core::ptr::addr_of_mut!((*hub).tt.lock));
*core::ptr::addr_of_mut!((*hub).tt.lock) = *{
__slate_compound_55 = spinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t2.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t3: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t3.val = __SlateRecord_types_h_190 { counter: 0 };
__t3
} } };
__t2
} };
core::ptr::addr_of_mut!(__slate_compound_55)
};
if true {
break;
}
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*hub).tt.clear_list));
loop {
__init_work(core::ptr::addr_of_mut!((*hub).tt.clear_work), 0i32);
(*core::ptr::addr_of_mut!((*hub).tt.clear_work)).data = *{
__slate_compound_54 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_54)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).tt.clear_work)).entry));
(*core::ptr::addr_of_mut!((*hub).tt.clear_work)).func = Some(hub_tt_work);
if true {
if true {
break;
}
}
}
let __t1: i32 = (*hdev).descriptor.bDeviceProtocol as i32;
if __t1 != 0 {
if __t1 == 1 {
(*hub).tt.hub = hdev;
} else {
if __t1 == 2 {
*__slate_slot_45 = usb_set_interface(hdev, 0i32, 1i32);
if *__slate_slot_45 == 0 {
(*hub).tt.multi = 1;
} else {
while false {
}
_dev_err(hub_dev as *const device, c"Using single TT (err %d)\n".as_ptr() as *const u8, *__slate_slot_45);
}
(*hub).tt.hub = hdev;
}
}
}
// Note 8 FS bit times == (8 bits / 12000000 bps) ~= 666ns
let __t0: i32 = (*__slate_slot_48 as i32) & 96;
if __t0 == 0 {
if ((*hdev).descriptor.bDeviceProtocol as i32) != 0 {
(*hub).tt.think_time = 666;
}
} else {
if __t0 == 32 {
(*hub).tt.think_time = 1332u32;
} else {
if __t0 == 64 {
(*hub).tt.think_time = 1998u32;
} else {
if __t0 == 96 {
(*hub).tt.think_time = 2664u32;
}
}
}
}
// probe() zeroes hub->indicator[]
if (*__slate_slot_48 as i32) & 128 != 0 {
(*hub).__slate_bits_0.__set_has_indicators(1u32);
}
// power budgeting mostly matters with bus-powered hubs,
// and battery-powered root hubs (may provide just 8 mA).
*__slate_slot_45 = usb_get_std_status(hdev, 0i32, 0i32, core::ptr::addr_of_mut!(*__slate_slot_50) as *mut ());
if *__slate_slot_45 != 0 {
message = c"can't get hub status".as_ptr() as *mut u8;
} else {
*__slate_slot_53 = bus_to_hcd((*hdev).bus);
if hdev == (*(*hdev).bus).root_hub {
if (*(*__slate_slot_53)).power_budget > 0 {
(*hdev).bus_mA = (*(*__slate_slot_53)).power_budget as u16;
} else {
(*hdev).bus_mA = (*__slate_slot_41).wrapping_mul(*__slate_slot_40) as u16;
}
if (((*hdev).bus_mA as i32) as u32) >= *__slate_slot_41 {
(*hub).mA_per_port = *__slate_slot_41;
} else {
(*hub).mA_per_port = (*hdev).bus_mA as u32;
(*hub).__slate_bits_0.__set_limited_power(1u32);
}
} else {
if (*__slate_slot_50 as i32) & 1i32 == 0 {
let remaining: i32 = ((*hdev).bus_mA as i32).wrapping_sub((*(*hub).descriptor).bHubContrCurrent as i32);
(*hub).__slate_bits_0.__set_limited_power(1u32);
if (remaining as u32) < (*__slate_slot_40).wrapping_mul(*__slate_slot_42) {
while false {
}
_dev_warn(hub_dev as *const device, c"insufficient power available to use all downstream ports\n".as_ptr() as *const u8);
}
(*hub).mA_per_port = *__slate_slot_42;
// 7.2.1
} else {
// Self-powered external hub
// FIXME: What about battery-powered external hubs that
// provide less current per port?
(*hub).mA_per_port = *__slate_slot_41;
}
}
if (*hub).mA_per_port < *__slate_slot_41 {
}
*__slate_slot_45 = hub_hub_status(hub, &mut (*core::ptr::addr_of_mut!(*__slate_slot_50)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_49)));
if *__slate_slot_45 < 0 {
message = c"can't get hub status".as_ptr() as *mut u8;
} else {
// local power status reports aren't always correct
if ((*(*hdev).actconfig).desc.bmAttributes as i32) & 64i32 != 0 {
}
if (*__slate_slot_48 as i32) & 24 != 0 {
}
// set up the interrupt endpoint
// We use the EP's maxpacket size instead of (PORTS+1+7)/8
// bytes as USB2.0[11.12.3] says because some hubs are known
// to send more data (and thus cause overflow). For root hubs,
// maxpktsize is defined in hcd.c's fake endpoint descriptors
// to be big enough for at least USB_MAXCHILDREN ports.
*__slate_slot_47 = usb_rcvintpipe!(hdev, unsafe { (*endpoint).bEndpointAddress });
*__slate_slot_46 = usb_maxpacket(hdev, *__slate_slot_47 as i32) as i32;
if (*__slate_slot_46 as u64) > (core::mem::size_of::<[u8; 8]>() as u64) {
*__slate_slot_46 = core::mem::size_of::<[u8; 8]>() as i32;
}
(*hub).urb = usb_alloc_urb(0i32, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if unsafe { (*hub).urb }.is_null() {
*__slate_slot_45 = 12i32.wrapping_neg();
} else {
usb_fill_int_urb((*hub).urb, hdev, *__slate_slot_47, ((*(*hub).buffer).as_mut_ptr() as *mut u8) as *mut (), *__slate_slot_46, Some(hub_irq), hub as *mut (), (*endpoint).bInterval as i32);
// maybe cycle the hub leds
if ((*hub).__slate_bits_0.__get_has_indicators() as i32) != 0 && blinkenlights {
*((*hub).indicator.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 1;
}
mutex_lock(core::ptr::addr_of_mut!(usb_port_peer_mutex));
*__slate_slot_44 = 0;
'__join_9: {
loop {
if (*__slate_slot_44 as u32) < *__slate_slot_40 {
*__slate_slot_45 = usb_hub_create_port_device(hub, (*__slate_slot_44).wrapping_add(1));
if *__slate_slot_45 < 0 {
break;
} else {
let __v4: i32 = *__slate_slot_44;
*__slate_slot_44 = __v4.wrapping_add(1);
}
} else {
break '__join_9;
}
}
while false {
}
_dev_err((*hub).intfdev as *const device, c"couldn't create port%d device.\n".as_ptr() as *const u8, (*__slate_slot_44).wrapping_add(1));
}
(*hdev).maxchild = *__slate_slot_44;
*__slate_slot_44 = 0;
while *__slate_slot_44 < (*hdev).maxchild {
port_dev = *unsafe { (*hub).ports }.wrapping_offset(*__slate_slot_44 as isize);
pm_runtime_put(core::ptr::addr_of_mut!((*port_dev).dev));
let __v1: i32 = *__slate_slot_44;
*__slate_slot_44 = __v1.wrapping_add(1);
}
mutex_unlock(core::ptr::addr_of_mut!(usb_port_peer_mutex));
if *__slate_slot_45 >= 0 {
// Update the HCD's internal representation of this hub before hub_wq
// starts getting port status changes for devices under the hub.
if (*(*(*__slate_slot_53)).driver).update_hub_device != None {
*__slate_slot_45 = unsafe { (*(*(*__slate_slot_53)).driver).update_hub_device }.unwrap()(*__slate_slot_53, hdev, core::ptr::addr_of_mut!((*hub).tt), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if *__slate_slot_45 < 0 {
message = c"can't update HCD hub info".as_ptr() as *mut u8;
break '__join_1;
}
}
usb_hub_adjust_deviceremovable(hdev, (*hub).descriptor);
hub_activate(hub, 0u32);
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
}
}
}
while false {
}
_dev_err(hub_dev as *const device, c"config failed, %s (err %d)\n".as_ptr() as *const u8, message, *__slate_slot_45);
// hub_disconnect() frees urb and descriptor
return *__slate_slot_45;
}
// USB 3.0 hubs don't have a TT
}

static mut __key__1485: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1592: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.hub.hub_release")]
extern "C" fn hub_release(kref: *mut kref) {
let hub: *mut usb_hub = container_of__39!(kref, usb_hub);
unsafe { usb_put_dev((*hub).hdev) };
unsafe { usb_put_intf(to_usb_interface!(unsafe { (*hub).intfdev })) };
unsafe { kfree(hub as *const ()) };
}

#[unsafe(no_mangle)]
extern "C" fn hub_get(hub: *mut usb_hub) {
kref_get(unsafe { core::ptr::addr_of_mut!((*hub).kref) });
}

#[unsafe(no_mangle)]
extern "C" fn hub_put(hub: *mut usb_hub) {
kref_put(unsafe { core::ptr::addr_of_mut!((*hub).kref) }, Some(hub_release));
}

static mut highspeed_hubs: u32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.hub.hub_disconnect")]
extern "C" fn hub_disconnect(intf: *mut usb_interface) {
let hub: *mut usb_hub = usb_get_intfdata(intf) as *mut usb_hub;
let hdev: *mut usb_device = interface_to_usbdev!(intf);
let mut port1: i32 = 0;
// Stop adding new hub events. We do not want to block here and thus
// will not try to remove any pending work item.
unsafe {
(*hub).__slate_bits_0.__set_disconnected(1u32);
}
// Disconnect all children and quiesce the hub
unsafe {
(*hub).error = 0;
}
hub_quiesce(hub, 0u32);
unsafe { mutex_lock(core::ptr::addr_of_mut!(usb_port_peer_mutex)) };
// Avoid races with recursively_mark_NOTATTACHED()
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(device_state_lock) });
port1 = unsafe { (*hdev).maxchild };
unsafe {
(*hdev).maxchild = 0;
}
usb_set_intfdata(intf, core::ptr::null_mut::<()>());
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(device_state_lock) });
while port1 > 0 {
unsafe { usb_hub_remove_port_device(hub, port1) };
port1 = port1.wrapping_sub(1);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(usb_port_peer_mutex)) };
if ((unsafe { (*(*hub).hdev).speed }) as u32) == 3 {
let __v3: u32 = unsafe { highspeed_hubs };
unsafe {
highspeed_hubs = __v3.wrapping_sub(1);
}
}
unsafe { usb_free_urb((*hub).urb) };
unsafe { kfree((*hub).ports as *const ()) };
unsafe { kfree((*hub).descriptor as *const ()) };
unsafe { kfree((*hub).status as *const ()) };
unsafe { kfree((*hub).buffer as *const ()) };
pm_suspend_ignore_children(unsafe { core::ptr::addr_of_mut!((*intf).dev) }, false);
if ((unsafe { (*hub).__slate_bits_0.__get_quirk_disable_autosuspend() }) as i32) != 0 {
unsafe { usb_autopm_put_interface(intf) };
}
onboard_dev_destroy_pdevs(unsafe { core::ptr::addr_of_mut!((*hub).onboard_devs) });
hub_put(hub);
}

fn hub_descriptor_is_sane(desc: *mut usb_host_interface) -> bool {
// Some hubs have a subclass of 1, which AFAICT according to the
//
//  specs is not defined, but it works
if ((unsafe { (*desc).desc.bInterfaceSubClass }) as i32) != 0 && ((unsafe { (*desc).desc.bInterfaceSubClass }) as i32) != 1 {
return false;
}
// Multiple endpoints? What kind of mutant ninja-hub is this?
if ((unsafe { (*desc).desc.bNumEndpoints }) as i32) != 1 {
return false;
}
// If the first endpoint is not interrupt IN, we'd better punt!
if usb_endpoint_is_int_in((unsafe { core::ptr::addr_of_mut!((*unsafe { (*desc).endpoint }.wrapping_offset(0 as isize)).desc) }) as *const usb_endpoint_descriptor) == 0 {
return false;
}
true
}

#[unsafe(link_section = ".text.slate_distinct.hub.hub_probe")]
extern "C" fn hub_probe(intf: *mut usb_interface, id: *const usb_device_id) -> i32 {
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_25: spinlock = unsafe { core::mem::zeroed() };
let mut desc: *mut usb_host_interface = unsafe { core::mem::zeroed() };
let mut hdev: *mut usb_device = unsafe { core::mem::zeroed() };
let mut hub: *mut usb_hub = unsafe { core::mem::zeroed() };
desc = unsafe { (*intf).cur_altsetting };
hdev = interface_to_usbdev!(intf);
// The USB 2.0 spec prohibits hubs from having more than one
// configuration or interface, and we rely on this prohibition.
// Refuse to accept a device that violates it.
if ((unsafe { (*hdev).descriptor.bNumConfigurations }) as i32) > 1 || ((unsafe { (*(*hdev).actconfig).desc.bNumInterfaces }) as i32) > 1 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*intf).dev) as *const device, c"Invalid hub with more than one config or interface\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
// Set default autosuspend delay as 0 to speedup bus suspend,
// based on the below considerations:
//
// - Unlike other drivers, the hub driver does not rely on the
//   autosuspend delay to provide enough time to handle a wakeup
//   event, and the submitted status URB is just to check future
//   change on hub downstream ports, so it is safe to do it.
//
// - The patch might cause one or more auto supend/resume for
//   below very rare devices when they are plugged into hub
//   first time:
//
//   	devices having trouble initializing, and disconnect
//   	themselves from the bus and then reconnect a second
//   	or so later
//
//   	devices just for downloading firmware, and disconnects
//   	themselves after completing it
//
//   For these quite rare devices, their drivers may change the
//   autosuspend delay of their parent hub in the probe() to one
//   appropriate value to avoid the subtle problem if someone
//   does care it.
//
// - The patch may cause one or more auto suspend/resume on
//   hub during running 'lsusb', but it is probably too
//   infrequent to worry about.
//
// - Change autosuspend delay of hub can avoid unnecessary auto
//   suspend timer for hub, also may decrease power consumption
//   of USB bus.
//
// - If user has indicated to prevent autosuspend by passing
//   usbcore.autosuspend = -1 then keep autosuspend disabled.
if (unsafe { (*hdev).dev.power.autosuspend_delay }) >= 0 {
unsafe { pm_runtime_set_autosuspend_delay(core::ptr::addr_of_mut!((*hdev).dev), 0i32) };
}
// Hubs have proper suspend/resume support, except for root hubs
// where the controller driver doesn't have bus_suspend and
// bus_resume methods.
if !unsafe { (*hdev).parent }.is_null() {
// normal device
unsafe { usb_enable_autosuspend(hdev) };
} else {
// root hub
let drv: *const hc_driver = unsafe { (*bus_to_hcd((*hdev).bus)).driver };
if (unsafe { (*drv).bus_suspend }) != None && (unsafe { (*drv).bus_resume }) != None {
unsafe { usb_enable_autosuspend(hdev) };
}
}
if ((unsafe { (*hdev).level }) as i32) == 6 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*intf).dev) as *const device, c"Unsupported bus topology: hub nested too deep\n".as_ptr() as *const u8) };
return 7i32.wrapping_neg();
}
if !hub_descriptor_is_sane(desc) {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*intf).dev) as *const device, c"bad descriptor, ignoring hub\n".as_ptr() as *const u8) };
return 5i32.wrapping_neg();
}
// We found a hub
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*intf).dev) as *const device, c"USB hub found\n".as_ptr() as *const u8) };
let mut __v4: *mut usb_hub;
let __obj_size: u64 = size_mul(core::mem::size_of::<usb_hub>() as u64, 1u64);
let mut __v5: *mut ();
let mut __v6: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
}
__v6 = _res;
__v5 = __v6;
__v4 = __v5 as *mut usb_hub;
hub = __v4;
if hub.is_null() {
return 12i32.wrapping_neg();
}
kref_init(unsafe { core::ptr::addr_of_mut!((*hub).kref) });
unsafe {
(*hub).intfdev = core::ptr::addr_of_mut!((*intf).dev);
}
unsafe {
(*hub).hdev = hdev;
}
'__slate_break_13: loop {
'__slate_break_12: loop {
'__slate_break_11: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).leds)).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).leds)).work)).data = *{
__slate_compound_10 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_10)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).leds)).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).leds)).work)).func = Some(led_work);
}
if true {
break;
}
}
if true {
break;
}
}
__timer_init!(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).leds)).timer) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
'__slate_break_17: loop {
'__slate_break_16: loop {
'__slate_break_15: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work)).data = *{
__slate_compound_14 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_14)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).work)).func = None;
}
if true {
break;
}
}
if true {
break;
}
}
__timer_init!(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).init_work)).timer) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
'__slate_break_21: loop {
'__slate_break_20: loop {
'__slate_break_19: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).post_resume_work)).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).post_resume_work)).work)).data = *{
__slate_compound_18 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_18)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).post_resume_work)).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).post_resume_work)).work)).func = Some(hub_post_resume);
}
if true {
break;
}
}
if true {
break;
}
}
__timer_init!(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).post_resume_work)).timer) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
'__slate_break_24: loop {
'__slate_break_23: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*hub).events) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*hub).events)).data = *{
__slate_compound_22 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_22)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*hub).events)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*hub).events)).func = Some(hub_event);
}
if true {
break;
}
}
if true {
break;
}
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*hub).onboard_devs) });
'__slate_break_26: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*hub).irq_urb_lock) });
unsafe {
*core::ptr::addr_of_mut!((*hub).irq_urb_lock) = *{
__slate_compound_25 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_25)
};
}
if true {
break;
}
}
timer_setup!(unsafe { core::ptr::addr_of_mut!((*hub).irq_urb_retry) }, Some(hub_retry_irq_urb), 0i32);
unsafe { usb_get_intf(intf) };
unsafe { usb_get_dev(hdev) };
usb_set_intfdata(intf, hub as *mut ());
unsafe {
(*intf).__slate_bits_0.__set_needs_remote_wakeup(1u32);
}
pm_suspend_ignore_children(unsafe { core::ptr::addr_of_mut!((*intf).dev) }, true);
if ((unsafe { (*hdev).speed }) as u32) == 3 {
let __v27: u32 = unsafe { highspeed_hubs };
unsafe {
highspeed_hubs = __v27.wrapping_add(1);
}
}
if (unsafe { (*id).driver_info }) & BIT!(0i32) != 0 {
unsafe {
(*hub).__slate_bits_0.__set_quirk_check_port_auto_suspend(1u32);
}
}
if (unsafe { (*id).driver_info }) & BIT!(1i32) != 0 {
unsafe {
(*hub).__slate_bits_0.__set_quirk_disable_autosuspend(1u32);
}
unsafe { usb_autopm_get_interface_no_resume(intf) };
}
if (unsafe { (*id).driver_info }) & BIT!(2i32) != 0 && ((unsafe { (*unsafe { (*desc).endpoint }.wrapping_offset(0 as isize)).desc.bInterval }) as i32) > 9 {
unsafe {
(*unsafe { (*desc).endpoint }.wrapping_offset(0 as isize)).desc.bInterval = 9;
}
// Tell the HCD about the interrupt ep's new bInterval
unsafe { usb_set_interface(hdev, 0i32, 0i32) };
}
if hub_configure(hub, unsafe { core::ptr::addr_of_mut!((*unsafe { (*desc).endpoint }.wrapping_offset(0 as isize)).desc) }) >= 0 {
onboard_dev_create_pdevs(hdev, unsafe { core::ptr::addr_of_mut!((*hub).onboard_devs) });
return 0;
}
hub_disconnect(intf);
19i32.wrapping_neg()
}

static mut __key__1978: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1979: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1980: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1981: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.hub.hub_ioctl")]
extern "C" fn hub_ioctl(intf: *mut usb_interface, code: u32, user_data: *mut ()) -> i32 {
let hdev: *mut usb_device = interface_to_usbdev!(intf);
let hub: *mut usb_hub = usb_hub_to_struct_hub(hdev);
// assert ifno == 0 (part of hub spec)
match code {
2155894035 => {
let mut i: i32 = 0;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(device_state_lock) });
if (unsafe { (*hdev).devnum }) <= 0 {
unsafe {
(*(user_data as *mut usbdevfs_hub_portinfo)).nports = 0;
}
} else {
unsafe {
(*(user_data as *mut usbdevfs_hub_portinfo)).nports = (*hdev).maxchild as u8;
}
i = 0;
while i < ((unsafe { (*(user_data as *mut usbdevfs_hub_portinfo)).nports }) as i32) {
if unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child }.is_null() {
unsafe {
*((*(user_data as *mut usbdevfs_hub_portinfo)).port.as_mut_ptr() as *mut u8).wrapping_offset(i as isize) = 0;
}
} else {
unsafe {
*((*(user_data as *mut usbdevfs_hub_portinfo)).port.as_mut_ptr() as *mut u8).wrapping_offset(i as isize) = (*(*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child).devnum as u8;
}
}
i = i.wrapping_add(1);
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(device_state_lock) });
return ((unsafe { (*(user_data as *mut usbdevfs_hub_portinfo)).nports }) as i32).wrapping_add(1);
}
_ => {
return 38i32.wrapping_neg();
}
}
unsafe { core::mem::zeroed() }
}

/// Allow user programs to claim ports on a hub.  When a device is attached
/// to one of these "claimed" ports, the program will "own" the device.
fn find_port_owner(hdev: *mut usb_device, port1: u32, ppowner: &mut *mut *mut usb_dev_state) -> i32 {
let hub: *mut usb_hub = usb_hub_to_struct_hub(hdev);
if ((unsafe { (*hdev).state }) as u32) == 0 {
return 19i32.wrapping_neg();
}
if port1 == 0 || port1 > ((unsafe { (*hdev).maxchild }) as u32) {
return 22i32.wrapping_neg();
}
// Devices not managed by the hub driver
// will always have maxchild equal to 0.
unsafe {
*(ppowner as *mut *mut *mut usb_dev_state) = core::ptr::addr_of_mut!((*(*unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize))).port_owner);
}
0
}

/// In the following three functions, the caller must hold hdev's lock
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_hub_claim_port")]
extern "C" fn usb_hub_claim_port(hdev: *mut usb_device, port1: u32, owner: *mut usb_dev_state) -> i32 {
let mut rc: i32 = 0;
let mut powner: *mut *mut usb_dev_state = unsafe { core::mem::zeroed() };
rc = find_port_owner(hdev, port1, unsafe { &mut (*core::ptr::addr_of_mut!(powner)) });
if rc != 0 {
return rc;
}
if !unsafe { *powner }.is_null() {
return 16i32.wrapping_neg();
}
unsafe {
*powner = owner;
}
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_hub_claim_port_678: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device, u32, *mut usb_dev_state) -> i32>, *mut ()>(Some(usb_hub_claim_port)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_hub_release_port")]
extern "C" fn usb_hub_release_port(hdev: *mut usb_device, port1: u32, owner: *mut usb_dev_state) -> i32 {
let mut rc: i32 = 0;
let mut powner: *mut *mut usb_dev_state = unsafe { core::mem::zeroed() };
rc = find_port_owner(hdev, port1, unsafe { &mut (*core::ptr::addr_of_mut!(powner)) });
if rc != 0 {
return rc;
}
if (unsafe { *powner }) != owner {
return 2i32.wrapping_neg();
}
unsafe {
*powner = core::ptr::null_mut::<usb_dev_state>();
}
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_hub_release_port_679: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device, u32, *mut usb_dev_state) -> i32>, *mut ()>(Some(usb_hub_release_port)) };

#[unsafe(no_mangle)]
extern "C" fn usb_hub_release_all_ports(hdev: *mut usb_device, owner: *mut usb_dev_state) {
let hub: *mut usb_hub = usb_hub_to_struct_hub(hdev);
let mut n: i32 = 0;
while n < unsafe { (*hdev).maxchild } {
if (unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(n as isize))).port_owner }) == owner {
unsafe {
(*(*unsafe { (*hub).ports }.wrapping_offset(n as isize))).port_owner = core::ptr::null_mut::<usb_dev_state>();
}
}
n = n.wrapping_add(1);
}
}

/// The caller must hold udev's lock
#[unsafe(no_mangle)]
extern "C" fn usb_device_is_owned(udev: *mut usb_device) -> bool {
if ((unsafe { (*udev).state }) as u32) == 0 || unsafe { (*udev).parent }.is_null() {
return false;
}
let hub: *mut usb_hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
!unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize))).port_owner }.is_null()
}

fn update_port_device_state(udev: *mut usb_device) {
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
if !unsafe { (*udev).parent }.is_null() {
let hub: *mut usb_hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
// The Link Layer Validation System Driver (lvstest)
// has a test step to unbind the hub before running the
// rest of the procedure. This triggers hub_disconnect
// which will set the hub's maxchild to 0, further
// resulting in usb_hub_to_struct_hub returning NULL.
if !hub.is_null() {
port_dev = unsafe { *unsafe { (*hub).ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize) };
loop {
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_680() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__6!((*port_dev).state, unsafe { (*udev).state });
if true {
break;
}
}
sysfs_notify_dirent(unsafe { (*port_dev).state_kn });
}
}
}

fn update_usb_device_state(udev: *mut usb_device, new_state: u32) {
if ((unsafe { (*udev).state }) as u32) == 8 && (new_state as u32) != 8 {
let __v1: u64 = unsafe { (*udev).active_duration };
let __v2: u64 = __v1.wrapping_sub(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) });
unsafe {
(*udev).active_duration = __v2;
}
} else {
if (new_state as u32) == 8 && ((unsafe { (*udev).state }) as u32) != 8 {
let __v4: u64 = unsafe { (*udev).active_duration };
let __v5: u64 = __v4.wrapping_add(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) });
unsafe {
(*udev).active_duration = __v5;
}
}
}
unsafe {
(*udev).state = new_state;
}
update_port_device_state(udev);
trace_usb_set_device_state(udev);
}

fn recursively_mark_NOTATTACHED(udev: *mut usb_device) {
let hub: *mut usb_hub = usb_hub_to_struct_hub(udev);
let mut i: i32 = 0;
while i < unsafe { (*udev).maxchild } {
if !unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child }.is_null() {
recursively_mark_NOTATTACHED(unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child });
}
i = i.wrapping_add(1);
}
update_usb_device_state(udev, 0u32);
}

/// usb_set_device_state - change a device's current state (usbcore, hcds)
/// @udev: pointer to device whose state should be changed
/// @new_state: new state value to be stored
///
/// udev->state is _not_ fully protected by the device lock.  Although
/// most transitions are made only while holding the lock, the state can
/// can change to USB_STATE_NOTATTACHED at almost any time.  This
/// is so that devices can be marked as disconnected as soon as possible,
/// without having to wait for any semaphores to be released.  As a result,
/// all changes to any device's state must be protected by the
/// device_state_lock spinlock.
///
/// Once a device has been added to the device tree, all changes to its state
/// should be made using this routine.  The state should _not_ be set directly.
///
/// If udev->state is already USB_STATE_NOTATTACHED then no change is made.
/// Otherwise udev->state is set to new_state, and if new_state is
/// USB_STATE_NOTATTACHED then all of udev's descendants' states are also set
/// to USB_STATE_NOTATTACHED.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_set_device_state")]
extern "C" fn usb_set_device_state(udev: *mut usb_device, new_state: u32) {
let mut flags: u64 = 0;
let mut wakeup: i32 = 1i32.wrapping_neg();
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(device_state_lock) }, flags);
if ((unsafe { (*udev).state }) as u32) != 0 {
if (new_state as u32) != 0 {
// root hub wakeup capabilities are managed out-of-band
// and may involve silicon errata ... ignore them here.
if !unsafe { (*udev).parent }.is_null() {
if !(((unsafe { (*udev).state }) as u32) == 8 || (new_state as u32) == 8) {
if (new_state as u32) == 7 {
wakeup = if ((unsafe { (*udev).quirks }) as u64) & BIT!(9i32) != 0 { 0 } else { ((unsafe { (*(*udev).actconfig).desc.bmAttributes }) as i32) & 32i32 };
} else {
wakeup = 0;
}
}
// No change to wakeup settings
}
update_usb_device_state(udev, new_state);
} else {
recursively_mark_NOTATTACHED(udev);
}
}
// do nothing
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(device_state_lock) }, flags);
if wakeup >= 0 {
unsafe { device_set_wakeup_capable(core::ptr::addr_of_mut!((*udev).dev), wakeup != 0) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_set_device_state_681: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device, u32)>, *mut ()>(Some(usb_set_device_state)) };

/// Choose a device number.
///
/// Device numbers are used as filenames in usbfs.  On USB-1.1 and
/// USB-2.0 buses they are also used as device addresses, however on
/// USB-3.0 buses the address is assigned by the controller hardware
/// and it usually is not the same as the device number.
///
/// Devices connected under xHCI are not as simple.  The host controller
/// supports virtualization, so the hardware assigns device addresses and
/// the HCD must setup data structures before issuing a set address
/// command to the hardware.
fn choose_devnum(udev: *mut usb_device) {
let mut devnum: i32 = 0;
let bus: *mut usb_bus = unsafe { (*udev).bus };
// be safe when more hub events are proceed in parallel
unsafe { mutex_lock(core::ptr::addr_of_mut!((*bus).devnum_next_mutex)) };
// Try to allocate the next devnum beginning at bus->devnum_next.
devnum = find_next_zero_bit((unsafe { (*bus).devmap.as_mut_ptr() as *mut u64 }) as *const u64, 128u64, (unsafe { (*bus).devnum_next }) as u64) as i32;
if devnum >= 128 {
devnum = find_next_zero_bit((unsafe { (*bus).devmap.as_mut_ptr() as *mut u64 }) as *const u64, 128u64, 1u64) as i32;
}
unsafe {
(*bus).devnum_next = if devnum >= 127 { 1 } else { devnum.wrapping_add(1) };
}
if devnum < 128 {
set_bit(devnum as i64, (unsafe { (*bus).devmap.as_mut_ptr() as *mut u64 }) as *mut u64);
unsafe {
(*udev).devnum = devnum;
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*bus).devnum_next_mutex)) };
}

fn release_devnum(udev: *mut usb_device) {
if (unsafe { (*udev).devnum }) > 0 {
clear_bit((unsafe { (*udev).devnum }) as i64, (unsafe { (*(*udev).bus).devmap.as_mut_ptr() as *mut u64 }) as *mut u64);
unsafe {
(*udev).devnum = 1i32.wrapping_neg();
}
}
}

fn update_devnum(udev: *mut usb_device, devnum: i32) {
unsafe {
(*udev).devnum = devnum;
}
if (unsafe { (*udev).devaddr }) == 0 {
unsafe {
(*udev).devaddr = devnum as u8;
}
}
}

fn hub_free_dev(udev: *mut usb_device) {
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*udev).bus });
// Root hubs aren't real devices, so don't free HCD resources
if (unsafe { (*(*hcd).driver).free_dev }) != None && !unsafe { (*udev).parent }.is_null() {
unsafe { unsafe { (*(*hcd).driver).free_dev }.unwrap()(hcd, udev) };
}
}

fn hub_disconnect_children(udev: *mut usb_device) {
let hub: *mut usb_hub = usb_hub_to_struct_hub(udev);
let mut i: i32 = 0;
// Free up all the children before we remove this device
while i < unsafe { (*udev).maxchild } {
if !unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child }.is_null() {
usb_disconnect(unsafe { core::ptr::addr_of_mut!((*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child) });
}
i = i.wrapping_add(1);
}
}

/// usb_disconnect - disconnect a device (usbcore-internal)
/// @pdev: pointer to device being disconnected
///
/// Context: task context, might sleep
///
/// Something got disconnected. Get rid of it and all of its children.
///
/// If *pdev is a normal device then the parent hub must already be locked.
/// If *pdev is a root hub then the caller must hold the usb_bus_idr_lock,
/// which protects the set of root hubs as well as the list of buses.
///
/// Only hub drivers (including virtual root hub drivers for host
/// controllers) should ever call this.
///
/// This call is synchronous, and may not be used in an interrupt context.
#[unsafe(no_mangle)]
extern "C" fn usb_disconnect(pdev: *mut *mut usb_device) {
let mut port_dev: *mut usb_port = core::ptr::null_mut::<usb_port>();
let udev: *mut usb_device = unsafe { *pdev };
let mut hub: *mut usb_hub = core::ptr::null_mut::<usb_hub>();
let mut port1: i32 = 1;
// mark the device as inactive, so any further urb submissions for
// this device (and any of its children) will fail immediately.
// this quiesces everything except pending urbs.
usb_set_device_state(udev, 0u32);
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"USB disconnect, device number %d\n".as_ptr() as *const u8, (*udev).devnum) };
// Ensure that the pm runtime code knows that the USB device
// is in the process of being disconnected.
unsafe { pm_runtime_barrier(core::ptr::addr_of_mut!((*udev).dev)) };
usb_lock_device!(udev);
hub_disconnect_children(udev);
// deallocate hcd/hardware state ... nuking all pending urbs and
// cleaning up all state associated with the current configuration
// so that the hardware is now fully quiesced.
unsafe { usb_disable_device(udev, 0i32) };
unsafe { usb_hcd_synchronize_unlinks(udev) };
if !unsafe { (*udev).parent }.is_null() {
port1 = (unsafe { (*udev).portnum }) as i32;
hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
port_dev = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize) };
unsafe { sysfs_remove_link(core::ptr::addr_of_mut!((*udev).dev.kobj), c"port".as_ptr() as *const u8) };
unsafe { sysfs_remove_link(core::ptr::addr_of_mut!((*port_dev).dev.kobj), c"device".as_ptr() as *const u8) };
// As usb_port_runtime_resume() de-references udev, make
// sure no resumes occur during removal
if !test_and_set_bit(port1 as i64, (unsafe { (*hub).child_usage_bits.as_mut_ptr() as *mut u64 }) as *mut u64) {
pm_runtime_get_sync(unsafe { core::ptr::addr_of_mut!((*port_dev).dev) });
}
typec_deattach(unsafe { (*port_dev).connector }, unsafe { core::ptr::addr_of_mut!((*udev).dev) });
}
unsafe { usb_remove_ep_devs(core::ptr::addr_of_mut!((*udev).ep0)) };
usb_unlock_device!(udev);
if !unsafe { (*udev).usb4_link }.is_null() {
unsafe { device_link_del((*udev).usb4_link) };
}
// Unregister the device.  The device driver is responsible
// for de-configuring the device and invoking the remove-device
// notifier chain (used by usbfs and possibly others).
unsafe { device_del(core::ptr::addr_of_mut!((*udev).dev)) };
// Free the device number and delete the parent's children[]
// (or root_hub) pointer.
release_devnum(udev);
// Avoid races with recursively_mark_NOTATTACHED()
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(device_state_lock) });
unsafe {
*pdev = core::ptr::null_mut::<usb_device>();
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(device_state_lock) });
let __v1: bool = !port_dev.is_null() && test_and_clear_bit(port1 as i64, (unsafe { (*hub).child_usage_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
if __v1 {
pm_runtime_put(unsafe { core::ptr::addr_of_mut!((*port_dev).dev) });
}
hub_free_dev(udev);
unsafe { put_device(core::ptr::addr_of_mut!((*udev).dev)) };
}

fn show_string(udev: *mut usb_device, id: *mut u8, string: *mut u8) {
if string.is_null() {
return;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"%s: %s\n".as_ptr() as *const u8, id, string) };
}

fn announce_device_ids(udev: *mut usb_device) {
let bcdDevice: u16 = unsafe { (*udev).descriptor.bcdDevice };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"New USB device found, idVendor=%04x, idProduct=%04x, bcdDevice=%2x.%02x\n".as_ptr() as *const u8, (*udev).descriptor.idVendor as i32, (*udev).descriptor.idProduct as i32, (bcdDevice as i32) >> 8i32, (bcdDevice as i32) & 255) };
}

fn announce_device_strings(udev: *mut usb_device) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"New USB device strings: Mfr=%d, Product=%d, SerialNumber=%d\n".as_ptr() as *const u8, (*udev).descriptor.iManufacturer as i32, (*udev).descriptor.iProduct as i32, (*udev).descriptor.iSerialNumber as i32) };
show_string(udev, c"Product".as_ptr() as *mut u8, unsafe { (*udev).product });
show_string(udev, c"Manufacturer".as_ptr() as *mut u8, unsafe { (*udev).manufacturer });
show_string(udev, c"SerialNumber".as_ptr() as *mut u8, unsafe { (*udev).serial });
}

/// usb_enumerate_device_otg - FIXME (usbcore-internal)
/// @udev: newly addressed device (in ADDRESS state)
///
/// Finish enumeration for On-The-Go devices
///
/// Return: 0 if successful. A negative error code otherwise.
fn usb_enumerate_device_otg(udev: *mut usb_device) -> i32 {
let err: i32 = 0;
err
}

/// usb_enumerate_device - Read device configs/intfs/otg (usbcore-internal)
/// @udev: newly addressed device (in ADDRESS state)
///
/// This is only called by usb_new_device() -- all comments that apply there
/// apply here wrt to environment.
///
/// If the device is WUSB and not authorized, we don't attempt to read
/// the string descriptors, as they will be errored out by the device
/// until it has been authorized.
///
/// Return: 0 if successful. A negative error code otherwise.
fn usb_enumerate_device(udev: *mut usb_device) -> i32 {
let mut err: i32 = 0;
bus_to_hcd(unsafe { (*udev).bus });
if unsafe { (*udev).config }.is_null() {
err = unsafe { usb_get_configuration(udev) };
if err < 0 {
if err != 19i32.wrapping_neg() {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"can't read configurations, error %d\n".as_ptr() as *const u8, err) };
}
return err;
}
}
// read the standard strings and cache them if present
unsafe {
(*udev).product = usb_cache_string(udev, (*udev).descriptor.iProduct as i32);
}
unsafe {
(*udev).manufacturer = usb_cache_string(udev, (*udev).descriptor.iManufacturer as i32);
}
unsafe {
(*udev).serial = usb_cache_string(udev, (*udev).descriptor.iSerialNumber as i32);
}
err = usb_enumerate_device_otg(udev);
if err < 0 {
return err;
}
unsafe { usb_detect_interface_quirks(udev) };
0
}

fn set_usb_port_removable(udev: *mut usb_device) {
let hdev: *mut usb_device = unsafe { (*udev).parent };
let mut hub: *mut usb_hub = unsafe { core::mem::zeroed() };
let port: u8 = unsafe { (*udev).portnum };
let mut wHubCharacteristics: u16 = 0;
let mut removable: bool = true;
dev_set_removable(unsafe { core::ptr::addr_of_mut!((*udev).dev) }, 1u32);
if hdev.is_null() {
return;
}
hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
// If the platform firmware has provided information about a port,
// use that to determine whether it's removable.
match (unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize))).connect_type }) as u32 {
1 => {
dev_set_removable(unsafe { core::ptr::addr_of_mut!((*udev).dev) }, 3u32);
return;
}
2 | 3 => {
dev_set_removable(unsafe { core::ptr::addr_of_mut!((*udev).dev) }, 2u32);
return;
}
_ => {
}
}
// Otherwise, check whether the hub knows whether a port is removable
// or not
wHubCharacteristics = unsafe { (*(*hub).descriptor).wHubCharacteristics };
if (wHubCharacteristics as i32) & 4 == 0 {
return;
}
if hub_is_superspeed(hdev) != 0 {
if ((unsafe { (*(*hub).descriptor).u.ss.DeviceRemovable }) as i32) & 1i32 << (port as i32) != 0 {
removable = false;
}
} else {
if ((unsafe { (core::ptr::addr_of_mut!((*(*hub).descriptor).u.hs.DeviceRemovable) as *mut u8).wrapping_offset(((port as i32) / 8) as isize).read_unaligned() }) as i32) & 1i32 << (port as i32) % 8 != 0 {
removable = false;
}
}
if removable {
dev_set_removable(unsafe { core::ptr::addr_of_mut!((*udev).dev) }, 3u32);
} else {
dev_set_removable(unsafe { core::ptr::addr_of_mut!((*udev).dev) }, 2u32);
}
}

/// usb_new_device - perform initial device setup (usbcore-internal)
/// @udev: newly addressed device (in ADDRESS state)
///
/// This is called with devices which have been detected but not fully
/// enumerated.  The device descriptor is available, but not descriptors
/// for any device configuration.  The caller must have locked either
/// the parent hub (if udev is a normal device) or else the
/// usb_bus_idr_lock (if udev is a root hub).  The parent's pointer to
/// udev has already been installed, but udev is not yet visible through
/// sysfs or other filesystem code.
///
/// This call is synchronous, and may not be used in an interrupt context.
///
/// Only the hub driver or root-hub registrar should ever call this.
///
/// Return: Whether the device is configured properly or not. Zero if the
/// interface was registered with the driver core; else a negative errno
/// value.
#[unsafe(no_mangle)]
extern "C" fn usb_new_device(udev: *mut usb_device) -> i32 {
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
let mut port1: i32 = unsafe { core::mem::zeroed() };
let mut hub: *mut usb_hub = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !unsafe { (*udev).parent }.is_null() {
// Initialize non-root-hub device wakeup to disabled;
// device (un)configuration controls wakeup capable
// sysfs power/wakeup controls wakeup enabled/disabled
device_init_wakeup(core::ptr::addr_of_mut!((*udev).dev), false);
}
'__join_0: {
// Announce the device identity
announce_device_ids(udev);
// Tell the runtime-PM framework the device is active
pm_runtime_set_active(core::ptr::addr_of_mut!((*udev).dev));
pm_runtime_get_noresume(core::ptr::addr_of_mut!((*udev).dev));
pm_runtime_use_autosuspend(core::ptr::addr_of_mut!((*udev).dev));
pm_runtime_enable(core::ptr::addr_of_mut!((*udev).dev));
// By default, forbid autosuspend for all devices.  It will be
// allowed for hubs during binding.
usb_disable_autosuspend(udev);
*__slate_slot_3 = usb_enumerate_device(udev); // Read descriptors
if *__slate_slot_3 >= 0 {
// export the usbdev device-node for libusb
(*udev).dev.devt = MKDEV!(189i32, unsafe { (*(*udev).bus).busnum }.wrapping_sub(1).wrapping_mul(128).wrapping_add(unsafe { (*udev).devnum }.wrapping_sub(1))) as u32;
// Announce the device's product, manufacturer and serial number
announce_device_strings(udev);
if !unsafe { (*udev).serial }.is_null() {
add_device_randomness((*udev).serial as *const (), strlen((*udev).serial as *const u8));
}
if !unsafe { (*udev).product }.is_null() {
add_device_randomness((*udev).product as *const (), strlen((*udev).product as *const u8));
}
if !unsafe { (*udev).manufacturer }.is_null() {
add_device_randomness((*udev).manufacturer as *const (), strlen((*udev).manufacturer as *const u8));
}
device_enable_async_suspend(core::ptr::addr_of_mut!((*udev).dev));
// check whether the hub or firmware marks this port as non-removable
set_usb_port_removable(udev);
// Register the device.  The device driver is responsible
// for configuring the device and invoking the add-device
// notifier chain (used by usbfs and possibly others).
*__slate_slot_3 = device_add(core::ptr::addr_of_mut!((*udev).dev));
if *__slate_slot_3 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"can't device_add, error %d\n".as_ptr() as *const u8, *__slate_slot_3);
} else {
'__join_2: {
// Create link files between child device and usb port device.
if !unsafe { (*udev).parent }.is_null() {
hub = usb_hub_to_struct_hub((*udev).parent);
port1 = (*udev).portnum as i32;
port_dev = *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize);
*__slate_slot_3 = sysfs_create_link(core::ptr::addr_of_mut!((*udev).dev.kobj), core::ptr::addr_of_mut!((*port_dev).dev.kobj), c"port".as_ptr() as *const u8);
if *__slate_slot_3 == 0 {
*__slate_slot_3 = sysfs_create_link(core::ptr::addr_of_mut!((*port_dev).dev.kobj), core::ptr::addr_of_mut!((*udev).dev.kobj), c"device".as_ptr() as *const u8);
if *__slate_slot_3 != 0 {
sysfs_remove_link(core::ptr::addr_of_mut!((*udev).dev.kobj), c"port".as_ptr() as *const u8);
} else {
if !test_and_set_bit(port1 as i64, ((*hub).child_usage_bits.as_mut_ptr() as *mut u64) as *mut u64) {
pm_runtime_get_sync(core::ptr::addr_of_mut!((*port_dev).dev));
}
typec_attach((*port_dev).connector, core::ptr::addr_of_mut!((*udev).dev));
break '__join_2;
}
}
device_del(core::ptr::addr_of_mut!((*udev).dev));
break '__join_0;
}
}
usb_create_ep_devs(core::ptr::addr_of_mut!((*udev).dev), core::ptr::addr_of_mut!((*udev).ep0), udev);
usb_mark_last_busy(udev);
pm_runtime_put_sync_autosuspend(core::ptr::addr_of_mut!((*udev).dev));
return *__slate_slot_3;
}
}
}
usb_set_device_state(udev, 0u32);
pm_runtime_disable(core::ptr::addr_of_mut!((*udev).dev));
pm_runtime_set_suspended(core::ptr::addr_of_mut!((*udev).dev));
return *__slate_slot_3;
}
}

/// usb_deauthorize_device - deauthorize a device (usbcore-internal)
/// @usb_dev: USB device
///
/// Move the USB device to a very basic state where interfaces are disabled
/// and the device is in fact unconfigured and unusable.
///
/// We share a lock (that we have) with device_del(), so we need to
/// defer its call.
///
/// Return: 0.
#[unsafe(no_mangle)]
extern "C" fn usb_deauthorize_device(usb_dev: *mut usb_device) -> i32 {
unsafe {
usb_lock_device!(usb_dev);
if ((*usb_dev).__slate_bits_0.__get_authorized() as i32) != 0 {
(*usb_dev).__slate_bits_0.__set_authorized(0u32);
usb_set_configuration(usb_dev, 1i32.wrapping_neg());
}
usb_unlock_device!(usb_dev);
return 0;
}
}

#[unsafe(no_mangle)]
extern "C" fn usb_authorize_device(usb_dev: *mut usb_device) -> i32 {
let mut c: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(c) as *mut i32;
let mut result: i32 = unsafe { core::mem::zeroed() };
unsafe {
result = 0;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
usb_lock_device!(usb_dev);
if ((*usb_dev).__slate_bits_0.__get_authorized() as i32) != 1 {
result = usb_autoresume_device(usb_dev);
if result < 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*usb_dev).dev) as *const device, c"can't autoresume for authorization: %d\n".as_ptr() as *const u8, result);
} else {
(*usb_dev).__slate_bits_0.__set_authorized(1u32);
// Choose and set the configuration.  This registers the interfaces
// with the driver core and lets interface drivers bind to them.
*__slate_slot_0 = usb_choose_configuration(usb_dev);
if *__slate_slot_0 >= 0 {
result = usb_set_configuration(usb_dev, *__slate_slot_0);
if result != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*usb_dev).dev) as *const device, c"can't set config #%d, error %d\n".as_ptr() as *const u8, *__slate_slot_0, result);
// This need not be fatal.  The user can try to
// set other configurations.
}
}
while false {
}
_dev_info(core::ptr::addr_of_mut!((*usb_dev).dev) as *const device, c"authorized to connect\n".as_ptr() as *const u8);
usb_autosuspend_device(usb_dev);
}
}
usb_unlock_device!(usb_dev); // complements locktree
return result;
}
}

// get_port_ssp_rate - Match the extended port status to SSP rate
// @hdev: The hub device
// @ext_portstatus: extended port status
//
// Match the extended port status speed id to the SuperSpeed Plus sublink speed
// capability attributes. Base on the number of connected lanes and speed,
// return the corresponding enum usb_ssp_rate.
fn get_port_ssp_rate(hdev: *mut usb_device, ext_portstatus: u32) -> u32 {
let mut r#type: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u8 = core::ptr::addr_of_mut!(r#type) as *mut u8;
let mut lse: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u8 = core::ptr::addr_of_mut!(lse) as *mut u8;
let mut mantissa: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u16 = core::ptr::addr_of_mut!(mantissa) as *mut u16;
let mut ssid: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u8 = core::ptr::addr_of_mut!(ssid) as *mut u8;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut lanes: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u8 = core::ptr::addr_of_mut!(lanes) as *mut u8;
let mut ssac: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u8 = core::ptr::addr_of_mut!(ssac) as *mut u8;
let mut speed_id: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u8 = core::ptr::addr_of_mut!(speed_id) as *mut u8;
let mut attr: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u32 = core::ptr::addr_of_mut!(attr) as *mut u32;
let mut ssp_cap: core::mem::MaybeUninit<*mut usb_ssp_cap_descriptor> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut usb_ssp_cap_descriptor = core::ptr::addr_of_mut!(ssp_cap) as *mut *mut usb_ssp_cap_descriptor;
unsafe {
'__join_0: {
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_5 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_6 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_7 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
if !unsafe { (*hdev).bos }.is_null() {
*__slate_slot_25 = (*(*hdev).bos).ssp_cap;
if !(*__slate_slot_25).is_null() {
*__slate_slot_23 = (ext_portstatus & 15) as u8;
*__slate_slot_21 = USB_EXT_PORT_RX_LANES!(ext_portstatus).wrapping_add(1) as u8;
*__slate_slot_22 = ((*(*__slate_slot_25)).bmAttributes & 31u32) as u8;
*__slate_slot_20 = 0;
loop {
if *__slate_slot_20 <= (*__slate_slot_22 as i32) {
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_24 = (core::ptr::addr_of_mut!((*(*__slate_slot_25)).__slate_anon_7.__slate_anon_1.bmSublinkSpeedAttr) as *mut u32).wrapping_offset(*__slate_slot_20 as isize).read_unaligned();
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(15i32 >> (15u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_685();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 15u64.wrapping_add(1u64 << (15u64.trailing_zeros() as i32)) & 15u64.wrapping_add(1u64 << (15u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_686();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_19 = (((*__slate_slot_24 & 15) >> (15u64.trailing_zeros() as i32)) as i32) as u8;
if (*__slate_slot_23 as i32) == (*__slate_slot_19 as i32) {
break;
} else {
let __v1: i32 = *__slate_slot_20;
*__slate_slot_20 = __v1.wrapping_add(1);
}
} else {
break '__join_0;
}
}
let __slate_auto_bytes_8 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(192i32 >> (192u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_690();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 192u64.wrapping_add(1u64 << (192u64.trailing_zeros() as i32)) & 192u64.wrapping_add(1u64 << (192u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_691();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
let __v12: i32 = ((*__slate_slot_24 & 192u32) >> (192u64.trailing_zeros() as i32)) as i32;
// Note: currently asymmetric lane types are only
// applicable for SSIC operate in SuperSpeed protocol
*__slate_slot_14 = __v12 as u8;
if !((*__slate_slot_14 as i32) == 1 || (*__slate_slot_14 as i32) == 3) {
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(49152i32 >> (49152u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_695();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 49152u64.wrapping_add(1u64 << (49152u64.trailing_zeros() as i32)) & 49152u64.wrapping_add(1u64 << (49152u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_696();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
let __v11: i32 = ((*__slate_slot_24 & 49152u32) >> (49152u64.trailing_zeros() as i32)) as i32;
if __v11 == 1 {
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(48i32 >> (48u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_700();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 48u64.wrapping_add(1u64 << (48u64.trailing_zeros() as i32)) & 48u64.wrapping_add(1u64 << (48u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_701();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_15 = (((*__slate_slot_24 & 48u32) >> (48u64.trailing_zeros() as i32)) as i32) as u8;
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(16711680i32 >> (16711680u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_705();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)) & 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_706();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_16 = (((*__slate_slot_24 & 16711680u32) >> (16711680u64.trailing_zeros() as i32)) as i32) as u16;
// Convert to Gbps
while (*__slate_slot_15 as i32) < 3 {
*__slate_slot_16 = ((*__slate_slot_16 as i32) / 1000) as u16;
*__slate_slot_15 = (*__slate_slot_15 as i32).wrapping_add(1) as u8;
}
if (*__slate_slot_16 as i32) >= 10 && (*__slate_slot_21 as i32) == 1 {
return 1;
} else {
if (*__slate_slot_16 as i32) >= 10 && (*__slate_slot_21 as i32) == 2 {
return 3;
} else {
if (*__slate_slot_16 as i32) >= 5 && (*__slate_slot_21 as i32) == 2 {
return 2;
}
}
}
}
}
}
}
}
return 0;
}
}

// times are in msec
fn use_new_scheme(udev: *mut usb_device, retry: i32, port_dev: *mut usb_port) -> bool {
let old_scheme_first_port: i32 = (((unsafe { (*port_dev).quirks }) as u64) & BIT!(0i32) != 0 || unsafe { old_scheme_first }) as i32;
// "New scheme" enumeration causes an extra state transition to be
// exposed to an xhci host and causes USB3 devices to receive control
// commands in the default state.  This has been seen to cause
// enumeration failures, so disable this enumeration scheme for USB3
// devices.
if ((unsafe { (*udev).speed }) as u32) >= 5 {
return false;
}
// If use_both_schemes is set, use the first scheme (whichever
// it is) for the larger half of the retries, then use the other
// scheme.  Otherwise, use the first scheme for all the retries.
if (unsafe { use_both_schemes }) && retry >= 2i32 {
return old_scheme_first_port != 0;
}
// Second half
return old_scheme_first_port == 0; // First half or all
}

/// Is a USB 3.0 port in the Inactive or Compliance Mode state?
/// Port warm reset is required to recover
fn hub_port_warm_reset_required(hub: *mut usb_hub, port1: i32, portstatus: u16) -> bool {
let mut link_state: u16 = 0;
if hub_is_superspeed(unsafe { (*hub).hdev }) == 0 {
return false;
}
if test_bit!(port1, unsafe { (*hub).warm_reset_bits.as_mut_ptr() as *mut u64 }) != 0 {
return true;
}
link_state = ((portstatus as i32) & 480) as u16;
(link_state as i32) == 192 || (link_state as i32) == 320
}

fn hub_port_wait_reset(hub: *mut usb_hub, port1: i32, udev: *mut usb_device, mut delay: u32, warm: bool) -> i32 {
let mut delay_time: i32 = 0;
let mut ret: i32 = 0;
let mut portstatus: u16 = 0;
let mut portchange: u16 = 0;
let mut ext_portstatus: u32 = 0;
while delay_time < 800 {
// wait to give the device a chance to reset
unsafe { msleep(delay) };
// read and decode port status
if hub_is_superspeedplus(unsafe { (*hub).hdev }) != 0 {
ret = hub_ext_port_status(hub, port1, 2i32, core::ptr::addr_of_mut!(portstatus), core::ptr::addr_of_mut!(portchange), core::ptr::addr_of_mut!(ext_portstatus));
} else {
ret = usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(portstatus), core::ptr::addr_of_mut!(portchange));
}
if ret < 0 {
return ret;
}
// The port state is unknown until the reset completes.
//
// On top of that, some chips may require additional time
// to re-establish a connection after the reset is complete,
// so also wait for the connection to be re-established.
if (portstatus as i32) & 16 == 0 && (portstatus as i32) & 1 != 0 {
break;
}
// switch to the long delay after two short delay failures
if delay_time >= 20i32 {
delay = 200;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize))).dev) as *const device, c"not %sreset yet, waiting %dms\n".as_ptr() as *const u8, if warm { c"warm ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, delay) };
}
delay_time = (delay_time as u32).wrapping_add(delay) as i32;
}
if (portstatus as i32) & 16 != 0 {
return 16i32.wrapping_neg();
}
if hub_port_warm_reset_required(hub, port1, portstatus) {
return 107i32.wrapping_neg();
}
// Device went away?
if (portstatus as i32) & 1 == 0 {
return 107i32.wrapping_neg();
}
// Retry if connect change is set but status is still connected.
// A USB 3.0 connection may bounce if multiple warm resets were issued,
// but the device may have successfully re-connected. Ignore it.
if hub_is_superspeed(unsafe { (*hub).hdev }) == 0 && (portchange as i32) & 1 != 0 {
usb_clear_port_feature(unsafe { (*hub).hdev }, port1, 16i32);
return 11i32.wrapping_neg();
}
if (portstatus as i32) & 2 == 0 {
return 16i32.wrapping_neg();
}
if udev.is_null() {
return 0;
}
if hub_is_superspeedplus(unsafe { (*hub).hdev }) != 0 {
// extended portstatus Rx and Tx lane count are zero based
unsafe {
(*udev).rx_lanes = USB_EXT_PORT_RX_LANES!(ext_portstatus).wrapping_add(1);
}
unsafe {
(*udev).tx_lanes = USB_EXT_PORT_TX_LANES!(ext_portstatus).wrapping_add(1);
}
unsafe {
(*udev).ssp_rate = get_port_ssp_rate((*hub).hdev, ext_portstatus);
}
} else {
unsafe {
(*udev).rx_lanes = 1;
}
unsafe {
(*udev).tx_lanes = 1;
}
unsafe {
(*udev).ssp_rate = 0;
}
}
if ((unsafe { (*udev).ssp_rate }) as u32) != 0 {
unsafe {
(*udev).speed = 6;
}
} else {
if hub_is_superspeed(unsafe { (*hub).hdev }) != 0 {
unsafe {
(*udev).speed = 5;
}
} else {
if (portstatus as i32) & 1024 != 0 {
unsafe {
(*udev).speed = 3;
}
} else {
if (portstatus as i32) & 512 != 0 {
unsafe {
(*udev).speed = 1;
}
} else {
unsafe {
(*udev).speed = 2;
}
}
}
}
}
0
}

/// Handle port reset and port warm(BH) reset (for USB3 protocol ports)
fn hub_port_reset(hub: *mut usb_hub, port1: i32, udev: *mut usb_device, mut delay: u32, mut warm: bool) -> i32 {
let mut hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
let mut reset_recovery_time: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(reset_recovery_time) as *mut i32;
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
let mut portstatus: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u16 = core::ptr::addr_of_mut!(portstatus) as *mut u16;
let mut portchange: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u16 = core::ptr::addr_of_mut!(portchange) as *mut u16;
let mut status: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(status) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 2usize);
port_dev = *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if hub_is_superspeed((*hub).hdev) == 0 {
if warm {
while false {
}
_dev_err((*hub).intfdev as *const device, c"only USB3 hub support warm reset\n".as_ptr() as *const u8);
return 22i32.wrapping_neg();
} else {
// Block EHCI CF initialization during the port reset.
// Some companion controllers don't like it when they mix.
down_read(core::ptr::addr_of_mut!(ehci_cf_port_reset_rwsem));
}
} else {
if !warm {
// If the caller hasn't explicitly requested a warm reset,
// double check and see if one is needed.
if usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(*__slate_slot_7), core::ptr::addr_of_mut!(*__slate_slot_8)) == 0 {
if hub_port_warm_reset_required(hub, port1, *__slate_slot_7) {
warm = true;
}
}
}
}
clear_bit(port1 as i64, ((*hub).warm_reset_bits.as_mut_ptr() as *mut u64) as *mut u64);
// Reset the port
*__slate_slot_10 = 0;
'__join_14: {
while *__slate_slot_10 < 5 {
*__slate_slot_9 = set_port_feature((*hub).hdev, port1, if warm { 28i32 } else { 4i32 });
if *__slate_slot_9 == 19i32.wrapping_neg() {
// The hub is gone
} else {
if *__slate_slot_9 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"cannot %sreset (err = %d)\n".as_ptr() as *const u8, if warm { c"warm ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, *__slate_slot_9);
} else {
*__slate_slot_9 = hub_port_wait_reset(hub, port1, udev, delay, warm);
if *__slate_slot_9 != 0 && *__slate_slot_9 != 107i32.wrapping_neg() && *__slate_slot_9 != 19i32.wrapping_neg() {
}
}
}
// Check for disconnect or reset, and bail out after several
// reset attempts to avoid warm reset loop.
if *__slate_slot_9 == 0 || *__slate_slot_9 == 107i32.wrapping_neg() || *__slate_slot_9 == 19i32.wrapping_neg() || *__slate_slot_9 == 16i32.wrapping_neg() && *__slate_slot_10 == 4i32 {
usb_clear_port_feature((*hub).hdev, port1, 20i32);
if hub_is_superspeed((*hub).hdev) == 0 {
break '__join_14;
} else {
usb_clear_port_feature((*hub).hdev, port1, 29i32);
usb_clear_port_feature((*hub).hdev, port1, 25i32);
if !udev.is_null() {
usb_clear_port_feature((*hub).hdev, port1, 16i32);
}
// If a USB 3.0 device migrates from reset to an error
// state, re-issue the warm reset.
if usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(*__slate_slot_7), core::ptr::addr_of_mut!(*__slate_slot_8)) < 0 {
break '__join_14;
} else {
if !hub_port_warm_reset_required(hub, port1, *__slate_slot_7) {
break '__join_14;
} else {
// If the port is in SS.Inactive or Compliance Mode, the
// hot or warm reset failed.  Try another warm reset.
if !warm {
warm = true;
}
}
}
}
}
delay = 200;
let __v4: i32 = *__slate_slot_10;
*__slate_slot_10 = __v4.wrapping_add(1);
}
loop {
if __ratelimit__2!(core::ptr::addr_of_mut!(_rs)) != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"Cannot enable. Maybe the USB cable is bad?\n".as_ptr() as *const u8);
}
if true {
break;
}
}
}
if *__slate_slot_9 == 0 {
if ((*port_dev).quirks as u64) & BIT!(1i32) != 0 {
usleep_range(10000u64, 12000u64);
} else {
// TRSTRCY = 10 ms; plus some extra
*__slate_slot_5 = 50i32;
// Hub needs extra delay after resetting its port.
if ((*(*hub).hdev).quirks as u64) & BIT!(14i32) != 0 {
let __v2: i32 = *__slate_slot_5;
*__slate_slot_5 = __v2.wrapping_add(100);
}
msleep(*__slate_slot_5 as u32);
}
if !udev.is_null() {
hcd = bus_to_hcd((*udev).bus);
update_devnum(udev, 0i32);
// The xHC may think the device is already reset,
// so ignore the status.
if (*(*hcd).driver).reset_device != None {
unsafe { (*(*hcd).driver).reset_device }.unwrap()(hcd, udev);
}
usb_set_device_state(udev, 5u32);
}
} else {
if !udev.is_null() {
usb_set_device_state(udev, 0u32);
}
}
if hub_is_superspeed((*hub).hdev) == 0 {
up_read(core::ptr::addr_of_mut!(ehci_cf_port_reset_rwsem));
}
return *__slate_slot_9;
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

/// hub_port_stop_enumerate - stop USB enumeration or ignore port events
/// @hub: target hub
/// @port1: port num of the port
/// @retries: port retries number of hub_port_init()
///
/// Return:
///    true: ignore port actions/events or give up connection attempts.
///    false: keep original behavior.
///
/// This function will be based on retries to check whether the port which is
/// marked with early_stop attribute would stop enumeration or ignore events.
///
/// Note:
/// This function didn't change anything if early_stop is not set, and it will
/// prevent all connection attempts when early_stop is set and the attempts of
/// the port are more than 1.
fn hub_port_stop_enumerate(hub: *mut usb_hub, port1: i32, retries: i32) -> bool {
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize) };
if ((unsafe { (*port_dev).__slate_bits_0.__get_early_stop() }) as i32) != 0 {
if ((unsafe { (*port_dev).__slate_bits_0.__get_ignore_event() }) as i32) != 0 {
return true;
}
// We want unsuccessful attempts to fail quickly.
// Since some devices may need one failure during
// port initialization, we allow two tries but no
// more.
if retries < 2 {
return false;
}
unsafe {
(*port_dev).__slate_bits_0.__set_ignore_event(1u32);
}
} else {
unsafe {
(*port_dev).__slate_bits_0.__set_ignore_event(0u32);
}
}
((unsafe { (*port_dev).__slate_bits_0.__get_ignore_event() }) as i32) != 0
}

/// Check if a port is power on
#[unsafe(no_mangle)]
extern "C" fn usb_port_is_power_on(hub: *mut usb_hub, portstatus: u32) -> i32 {
let mut ret: i32 = 0;
if hub_is_superspeed(unsafe { (*hub).hdev }) != 0 {
if portstatus & 512 != 0 {
ret = 1;
}
} else {
if portstatus & 256 != 0 {
ret = 1;
}
}
ret
}

fn usb_lock_port(port_dev: *mut usb_port) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*port_dev).status_lock)) };
__acquire!();
}

fn usb_unlock_port(port_dev: *mut usb_port) {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*port_dev).status_lock)) };
__release!();
}

/// Check if a port is suspended(USB2.0 port) or in U3 state(USB3.0 port)
fn port_is_suspended(hub: *mut usb_hub, portstatus: u32) -> i32 {
let mut ret: i32 = 0;
if hub_is_superspeed(unsafe { (*hub).hdev }) != 0 {
if portstatus & 480 == 96 {
ret = 1;
}
} else {
if portstatus & 4 != 0 {
ret = 1;
}
}
ret
}

/// Determine whether the device on a port is ready for a normal resume,
/// is ready for a reset-resume, or should be disconnected.
fn check_port_resume_type(udev: *mut usb_device, hub: *mut usb_hub, port1: i32, mut status: i32, mut portchange: u16, mut portstatus: u16) -> i32 {
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
unsafe {
retries = 3;
'__join_7: {
'__join_8: {
'__join_10: {
'__join_11: {
loop {
if status == 0 && ((*udev).__slate_bits_1.__get_reset_resume() as i32) != 0 {
__v4 = hub_port_warm_reset_required(hub, port1, portstatus);
} else {
__v4 = false;
}
if __v4 {
break '__join_8;
} else {
if status != 0 {
__v3 = true;
} else {
__v3 = port_is_suspended(hub, portstatus as u32) != 0;
}
if __v3 {
__v2 = true;
} else {
__v2 = usb_port_is_power_on(hub, portstatus as u32) == 0;
}
if __v2 {
break '__join_10;
} else {
if (portstatus as i32) & 1 == 0 {
let __v1: i32 = retries;
retries = __v1.wrapping_sub(1);
if __v1 != 0 {
usleep_range(200u64, 300u64);
status = usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(portstatus), core::ptr::addr_of_mut!(portchange));
} else {
break '__join_11;
}
} else {
break;
}
}
}
}
if (portstatus as i32) & 2 == 0 && ((*udev).__slate_bits_1.__get_reset_resume() as i32) == 0 {
if ((*udev).__slate_bits_0.__get_persist_enabled() as i32) != 0 {
(*udev).__slate_bits_1.__set_reset_resume(1u32);
break '__join_7;
} else {
status = 19i32.wrapping_neg();
break '__join_7;
}
} else {
break '__join_7;
}
}
status = 19i32.wrapping_neg();
break '__join_7;
}
if status >= 0 {
status = 19i32.wrapping_neg();
break '__join_7;
} else {
break '__join_7;
}
}
// pass
}
// Is a warm reset needed to recover the connection?
// Is the device still present?
// Can't do a normal resume if the port isn't enabled,
// so try a reset-resume instead.
if status == 0 {
if ((*udev).__slate_bits_1.__get_reset_resume() as i32) != 0 {
// Late port handoff can set status-change bits
if (portchange as i32) & 1 != 0 {
usb_clear_port_feature((*hub).hdev, port1, 16i32);
}
if (portchange as i32) & 2 != 0 {
usb_clear_port_feature((*hub).hdev, port1, 17i32);
}
// Whatever made this reset-resume necessary may have
// turned on the port1 bit in hub->change_bits.  But after
// a successful reset-resume we want the bit to be clear;
// if it was on it would indicate that something happened
// following the reset-resume.
clear_bit(port1 as i64, ((*hub).change_bits.as_mut_ptr() as *mut u64) as *mut u64);
}
}
return status;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_disable_ltm")]
extern "C" fn usb_disable_ltm(udev: *mut usb_device) -> i32 {
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*udev).bus });
// Check if the roothub and device supports LTM.
let __v0: bool = !usb_device_supports_ltm(unsafe { (*hcd).self_.root_hub }) || !usb_device_supports_ltm(udev);
if __v0 {
return 0;
}
// Clear Feature LTM Enable can only be sent if the device is
// configured.
if unsafe { (*udev).actconfig }.is_null() {
return 0;
}
unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 1u8, 0u8, 50u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 5000i32) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_disable_ltm_707: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device) -> i32>, *mut ()>(Some(usb_disable_ltm)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_enable_ltm")]
extern "C" fn usb_enable_ltm(udev: *mut usb_device) {
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*udev).bus });
// Check if the roothub and device supports LTM.
let __v0: bool = !usb_device_supports_ltm(unsafe { (*hcd).self_.root_hub }) || !usb_device_supports_ltm(udev);
if __v0 {
return;
}
// Set Feature LTM Enable can only be sent if the device is
// configured.
if unsafe { (*udev).actconfig }.is_null() {
return;
}
unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 3u8, 0u8, 50u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 5000i32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_enable_ltm_708: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device)>, *mut ()>(Some(usb_enable_ltm)) };

/// usb_enable_remote_wakeup - enable remote wakeup for a device
/// @udev: target device
///
/// For USB-2 devices: Set the device's remote wakeup feature.
///
/// For USB-3 devices: Assume there's only one function on the device and
/// enable remote wake for the first interface.  FIXME if the interface
/// association descriptor shows there's more than one function.
fn usb_enable_remote_wakeup(udev: *mut usb_device) -> i32 {
if ((unsafe { (*udev).speed }) as u32) < 5 {
return unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 3u8, 0u8, 1u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 5000i32) };
} else {
return unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 3u8, 1u8, 0u16, 768u16, core::ptr::null_mut::<()>(), 0u16, 5000i32) };
}
}

/// usb_disable_remote_wakeup - disable remote wakeup for a device
/// @udev: target device
///
/// For USB-2 devices: Clear the device's remote wakeup feature.
///
/// For USB-3 devices: Assume there's only one function on the device and
/// disable remote wake for the first interface.  FIXME if the interface
/// association descriptor shows there's more than one function.
fn usb_disable_remote_wakeup(udev: *mut usb_device) -> i32 {
if ((unsafe { (*udev).speed }) as u32) < 5 {
return unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 1u8, 0u8, 1u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 5000i32) };
} else {
return unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 3u8, 1u8, 0u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 5000i32) };
}
}

/// Count of wakeup-enabled devices at or below udev
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_wakeup_enabled_descendants")]
extern "C" fn usb_wakeup_enabled_descendants(udev: *mut usb_device) -> u32 {
let hub: *mut usb_hub = usb_hub_to_struct_hub(udev);
(((unsafe { (*udev).__slate_bits_1.__get_do_remote_wakeup() }) as i32) as u32).wrapping_add(if !hub.is_null() { unsafe { (*hub).wakeup_enabled_descendants } } else { 0 })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_wakeup_enabled_descendants_709: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device) -> u32>, *mut ()>(Some(usb_wakeup_enabled_descendants)) };

/// usb_port_suspend - suspend a usb device's upstream port
/// @udev: device that's no longer in active use, not a root hub
/// Context: must be able to sleep; device not locked; pm locks held
///
/// Suspends a USB device that isn't in active use, conserving power.
/// Devices may wake out of a suspend, if anything important happens,
/// using the remote wakeup mechanism.  They may also be taken out of
/// suspend by the host, using usb_port_resume().  It's also routine
/// to disconnect devices while they are suspended.
///
/// This only affects the USB hardware for a device; its interfaces
/// (and, for hubs, child devices) must already have been suspended.
///
/// Selective port suspend reduces power; most suspended devices draw
/// less than 500 uA.  It's also used in OTG, along with remote wakeup.
/// All devices below the suspended port are also suspended.
///
/// Devices leave suspend state when the host wakes them up.  Some devices
/// also support "remote wakeup", where the device can activate the USB
/// tree above them to deliver data, such as a keypress or packet.  In
/// some cases, this wakes the USB host.
///
/// Suspending OTG devices may trigger HNP, if that's been enabled
/// between a pair of dual-role devices.  That will change roles, such
/// as from A-Host to A-Peripheral or from B-Host back to B-Peripheral.
///
/// Devices on USB hub ports have only one "suspend" state, corresponding
/// to ACPI D2, "may cause the device to lose some context".
/// State transitions include:
///
///   - suspend, resume ... when the VBUS power link stays live
///   - suspend, disconnect ... VBUS lost
///
/// Once VBUS drop breaks the circuit, the port it's using has to go through
/// normal re-enumeration procedures, starting with enabling VBUS power.
/// Other than re-initializing the hub (plug/unplug, except for root hubs),
/// Linux (2.6) currently has NO mechanisms to initiate that:  no hub_wq
/// timer, no SRP, no requests through sysfs.
///
/// If Runtime PM isn't enabled or used, non-SuperSpeed devices may not get
/// suspended until their bus goes into global suspend (i.e., the root
/// hub is suspended).  Nevertheless, we change @udev->state to
/// USB_STATE_SUSPENDED as this is the device's "logical" state.  The actual
/// upstream port setting is stored in @udev->port_is_suspended.
///
/// Returns 0 on success, else negative errno.
#[unsafe(no_mangle)]
extern "C" fn usb_port_suspend(udev: *mut usb_device, msg: pm_message) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut portchange: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u16 = core::ptr::addr_of_mut!(portchange) as *mut u16;
let mut portstatus: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u16 = core::ptr::addr_of_mut!(portstatus) as *mut u16;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut really_suspend: bool = unsafe { core::mem::zeroed() };
let mut status: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(status) as *mut i32;
let mut port1: i32 = unsafe { core::mem::zeroed() };
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
let mut hub: *mut usb_hub = unsafe { core::mem::zeroed() };
unsafe {
'__join_5: {
'__join_8: {
{
hub = usb_hub_to_struct_hub((*udev).parent);
port_dev = *unsafe { (*hub).ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize);
port1 = (*udev).portnum as i32;
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
really_suspend = true;
usb_lock_port(port_dev);
// enable remote wakeup when appropriate; this lets the device
// wake up the upstream hub (including maybe the root hub).
//
// NOTE:  OTG devices may issue remote wakeup (or SRP) even when
// we don't explicitly enable it here.
if ((*udev).__slate_bits_1.__get_do_remote_wakeup() as i32) != 0 {
*__slate_slot_7 = usb_enable_remote_wakeup(udev);
if *__slate_slot_7 != 0 {
// bail if autosuspend is requested
if PMSG_IS_AUTO!(msg) {
break '__join_8;
}
}
}
}
'__join_10: {
// disable USB2 hardware LPM
usb_disable_usb2_hardware_lpm(udev);
if usb_disable_ltm(udev) != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Failed to disable LTM before suspend\n".as_ptr() as *const u8);
*__slate_slot_7 = 12i32.wrapping_neg();
if PMSG_IS_AUTO!(msg) {
break '__join_10;
}
}
// see 7.1.7.6
if hub_is_superspeed((*hub).hdev) != 0 {
*__slate_slot_7 = hub_set_port_link_state(hub, port1, 96u32);
} else {
if PMSG_IS_AUTO!(msg) {
__v5 = true;
} else {
__v5 = usb_wakeup_enabled_descendants(udev) > 0;
}
if __v5 {
*__slate_slot_7 = set_port_feature((*hub).hdev, port1, 2i32);
} else {
really_suspend = false;
*__slate_slot_7 = 0;
}
}
'__join_20: {
// For system suspend, we do not need to enable the suspend feature
// on individual USB-2 ports.  The devices will automatically go
// into suspend a few ms after the root hub stops sending packets.
// The USB 2.0 spec calls this "global suspend".
//
// However, many USB hubs have a bug: They don't relay wakeup requests
// from a downstream port if the port's suspend feature isn't on.
// Therefore we will turn on the suspend feature if udev or any of its
// descendants is enabled for remote wakeup.
if *__slate_slot_7 != 0 {
// Check if the port has been suspended for the timeout case
// to prevent the suspended port from incorrect handling.
if *__slate_slot_7 == 110i32.wrapping_neg() {
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_2 = 0;
*__slate_slot_3 = 0;
*__slate_slot_4 = usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(*__slate_slot_3), core::ptr::addr_of_mut!(*__slate_slot_2));
if *__slate_slot_4 == 0 {
__v1 = port_is_suspended(hub, *__slate_slot_3 as u32) != 0;
} else {
__v1 = false;
}
if __v1 {
*__slate_slot_7 = 0;
break '__join_20;
}
}
// Try to enable USB3 LTM again
usb_enable_ltm(udev);
break '__join_10;
}
}
if really_suspend {
(*udev).__slate_bits_1.__set_port_is_suspended(1u32);
// device has up to 10 msec to fully suspend
msleep(10u32);
}
usb_set_device_state(udev, 8u32);
break '__join_5;
}
usb_enable_usb2_hardware_lpm(udev); // Try to enable USB2 hardware LPM again
if ((*udev).__slate_bits_1.__get_do_remote_wakeup() as i32) != 0 {
usb_disable_remote_wakeup(udev);
}
}
if !PMSG_IS_AUTO!(msg) {
*__slate_slot_7 = 0;
}
// System sleep transitions should never fail
}
if *__slate_slot_7 == 0 && ((*udev).__slate_bits_1.__get_do_remote_wakeup() as i32) == 0 && ((*udev).__slate_bits_0.__get_persist_enabled() as i32) != 0 {
__v0 = test_and_clear_bit(port1 as i64, ((*hub).child_usage_bits.as_mut_ptr() as *mut u64) as *mut u64);
} else {
__v0 = false;
}
if __v0 {
pm_runtime_put_sync(core::ptr::addr_of_mut!((*port_dev).dev));
}
usb_mark_last_busy((*hub).hdev);
usb_unlock_port(port_dev);
return *__slate_slot_7;
}
}

/// If the USB "suspend" state is in use (rather than "global suspend"),
/// many devices will be individually taken out of suspend state using
/// special "resume" signaling.  This routine kicks in shortly after
/// hardware resume signaling is finished, either because of selective
/// resume (by host) or remote wakeup (by device) ... now see what changed
/// in the tree that's rooted at this device.
///
/// If @udev->reset_resume is set then the device is reset before the
/// status check is done.
fn finish_port_resume(udev: *mut usb_device) -> i32 {
let mut devstatus: u16 = unsafe { core::mem::zeroed() };
let mut status: i32 = unsafe { core::mem::zeroed() };
let mut __slate_state: usize = 18usize;
unsafe {
status = 0;
devstatus = 0;
// caller owns the udev device lock
// usb ch9 identifies four variants of SUSPENDED, based on what
// state the device resumes to.  Linux currently won't see the
// first two on the host side; they'd be inside hub_port_init()
// during many timeouts, but hub_wq can't suspend until later.
usb_set_device_state(udev, (if !unsafe { (*udev).actconfig }.is_null() { 7i32 } else { 6i32 }) as u32);
// 10.5.4.5 says not to reset a suspended port if the attached
// device is enabled for remote wakeup.  Hence the reset
// operation is carried out here, after the port has been
// resumed.
if ((*udev).__slate_bits_1.__get_reset_resume() as i32) != 0 {
// If the device morphs or switches modes when it is reset,
// we don't want to perform a reset-resume.  We'll fail the
// resume, which will cause a logical disconnect, and then
// the device will be rediscovered.
__slate_state = 16;
} else {
__slate_state = 13;
}
loop {
match __slate_state {
13 => {
// 10.5.4.5 says be sure devices in the tree are still there.
// For now let's assume the device didn't go crazy on resume,
// and device drivers will know about any resume quirks.
if status == 0 {
devstatus = 0;
status = usb_get_std_status(udev, 0i32, 0i32, core::ptr::addr_of_mut!(devstatus) as *mut ());
// If a normal resume failed, try doing a reset-resume
if status != 0 && ((*udev).__slate_bits_1.__get_reset_resume() as i32) == 0 && ((*udev).__slate_bits_0.__get_persist_enabled() as i32) != 0 {
(*udev).__slate_bits_1.__set_reset_resume(1u32);
__slate_state = 16;
} else {
break;
}
} else {
break;
}
}
_ => {
if ((*udev).quirks as u64) & BIT!(4i32) != 0 {
status = 19i32.wrapping_neg();
__slate_state = 13;
} else {
status = usb_reset_and_verify_device(udev);
__slate_state = 13;
}
}
}
}
if status != 0 {
// There are a few quirky devices which violate the standard
// by claiming to have remote wakeup enabled after a reset,
// which crash if the feature is cleared, hence check for
// udev->reset_resume
} else {
if !unsafe { (*udev).actconfig }.is_null() && ((*udev).__slate_bits_1.__get_reset_resume() as i32) == 0 {
if ((*udev).speed as u32) < 5 {
if (devstatus as i32) & 2i32 != 0 {
status = usb_disable_remote_wakeup(udev);
}
} else {
status = usb_get_std_status(udev, 1i32, 0i32, core::ptr::addr_of_mut!(devstatus) as *mut ());
if status == 0 && (devstatus as i32) & (1 | 2) != 0 {
status = usb_disable_remote_wakeup(udev);
}
}
status = 0;
}
}
return status;
}
}

/// There are some SS USB devices which take longer time for link training.
/// XHCI specs 4.19.4 says that when Link training is successful, port
/// sets CCS bit to 1. So if SW reads port status before successful link
/// training, then it will not find device to be present.
/// USB Analyzer log with such buggy devices show that in some cases
/// device switch on the RX termination after long delay of host enabling
/// the VBUS. In few other cases it has been seen that device fails to
/// negotiate link training in first attempt. It has been
/// reported till now that few devices take as long as 2000 ms to train
/// the link after host enabling its VBUS and termination. Following
/// routine implements a 2000 ms timeout for link training. If in a case
/// link trains before timeout, loop will exit earlier.
///
/// There are also some 2.0 hard drive based devices and 3.0 thumb
/// drives that, when plugged into a 2.0 only port, take a long
/// time to set CCS after VBUS enable.
///
/// FIXME: If a device was connected before suspend, but was removed
/// while system was asleep, then the loop in the following routine will
/// only exit at timeout.
///
/// This routine should only be called when persist is enabled.
fn wait_for_connected(udev: *mut usb_device, hub: *mut usb_hub, port1: i32, portchange: *mut u16, portstatus: *mut u16) -> i32 {
let mut status: i32 = 0;
let mut delay_ms: i32 = 0;
while delay_ms < 2000 {
if status != 0 || ((unsafe { *portstatus }) as i32) & 1 != 0 {
break;
}
if usb_port_is_power_on(hub, (unsafe { *portstatus }) as u32) == 0 {
status = 19i32.wrapping_neg();
break;
}
unsafe { msleep(20u32) };
delay_ms = delay_ms.wrapping_add(20);
status = usb_hub_port_status(hub, port1, portstatus, portchange);
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Waited %dms for CONNECT\n".as_ptr() as *const u8, delay_ms) };
}
status
}

/// usb_port_resume - re-activate a suspended usb device's upstream port
/// @udev: device to re-activate, not a root hub
/// Context: must be able to sleep; device not locked; pm locks held
///
/// This will re-activate the suspended device, increasing power usage
/// while letting drivers communicate again with its endpoints.
/// USB resume explicitly guarantees that the power session between
/// the host and the device is the same as it was when the device
/// suspended.
///
/// If @udev->reset_resume is set then this routine won't check that the
/// port is still enabled.  Furthermore, finish_port_resume() above will
/// reset @udev.  The end result is that a broken power session can be
/// recovered and @udev will appear to persist across a loss of VBUS power.
///
/// For example, if a host controller doesn't maintain VBUS suspend current
/// during a system sleep or is reset when the system wakes up, all the USB
/// power sessions below it will be broken.  This is especially troublesome
/// for mass-storage devices containing mounted filesystems, since the
/// device will appear to have disconnected and all the memory mappings
/// to it will be lost.  Using the USB_PERSIST facility, the device can be
/// made to appear as if it had not disconnected.
///
/// This facility can be dangerous.  Although usb_reset_and_verify_device() makes
/// every effort to insure that the same device is present after the
/// reset as before, it cannot provide a 100% guarantee.  Furthermore it's
/// quite possible for a device to remain unaltered but its media to be
/// changed.  If the user replaces a flash memory card while the system is
/// asleep, he will have only himself to blame when the filesystem on the
/// new card is corrupted and the system crashes.
///
/// Returns 0 on success, else negative errno.
#[unsafe(no_mangle)]
extern "C" fn usb_port_resume(udev: *mut usb_device, msg: pm_message) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut portstatus: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u16 = core::ptr::addr_of_mut!(portstatus) as *mut u16;
let mut portchange: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u16 = core::ptr::addr_of_mut!(portchange) as *mut u16;
let mut status: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(status) as *mut i32;
let mut port1: i32 = unsafe { core::mem::zeroed() };
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
let mut hub: *mut usb_hub = unsafe { core::mem::zeroed() };
unsafe {
hub = usb_hub_to_struct_hub((*udev).parent);
port_dev = *unsafe { (*hub).ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize);
port1 = (*udev).portnum as i32;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
if !test_and_set_bit(port1 as i64, ((*hub).child_usage_bits.as_mut_ptr() as *mut u64) as *mut u64) {
*__slate_slot_3 = pm_runtime_resume_and_get(core::ptr::addr_of_mut!((*port_dev).dev));
if *__slate_slot_3 < 0 {
return *__slate_slot_3;
}
}
usb_lock_port(port_dev);
// Skip the initial Clear-Suspend step for a remote wakeup
*__slate_slot_3 = usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(*__slate_slot_1), core::ptr::addr_of_mut!(*__slate_slot_2));
if *__slate_slot_3 == 0 {
__v0 = port_is_suspended(hub, *__slate_slot_1 as u32) == 0;
} else {
__v0 = false;
}
if __v0 {
if (*__slate_slot_2 as i32) & 4 != 0 {
pm_wakeup_event(core::ptr::addr_of_mut!((*udev).dev), 0u32);
}
} else {
// see 7.1.7.7; affects power usage, but not budgeting
if hub_is_superspeed((*hub).hdev) != 0 {
*__slate_slot_3 = hub_set_port_link_state(hub, port1, 0u32);
} else {
*__slate_slot_3 = usb_clear_port_feature((*hub).hdev, port1, 2i32);
}
if *__slate_slot_3 == 0 {
// drive resume for USB_RESUME_TIMEOUT msec
msleep(40u32);
// Virtual root hubs can trigger on GET_PORT_STATUS to
// stop resume signaling.  Then finish the resume
// sequence.
*__slate_slot_3 = usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(*__slate_slot_1), core::ptr::addr_of_mut!(*__slate_slot_2));
}
}
if *__slate_slot_3 == 0 {
(*udev).__slate_bits_1.__set_port_is_suspended(0u32);
if hub_is_superspeed((*hub).hdev) != 0 {
if (*__slate_slot_2 as i32) & 64 != 0 {
usb_clear_port_feature((*hub).hdev, port1, 25i32);
}
} else {
if (*__slate_slot_2 as i32) & 4 != 0 {
usb_clear_port_feature((*hub).hdev, port1, 18i32);
}
}
// TRSMRCY = 10 msec
msleep(10u32);
}
if ((*udev).__slate_bits_0.__get_persist_enabled() as i32) != 0 {
*__slate_slot_3 = wait_for_connected(udev, hub, port1, core::ptr::addr_of_mut!(*__slate_slot_2), core::ptr::addr_of_mut!(*__slate_slot_1));
}
*__slate_slot_3 = check_port_resume_type(udev, hub, port1, *__slate_slot_3, *__slate_slot_2, *__slate_slot_1);
if *__slate_slot_3 == 0 {
*__slate_slot_3 = finish_port_resume(udev);
}
if *__slate_slot_3 < 0 {
hub_port_logical_disconnect(hub, port1);
} else {
// Try to enable USB2 hardware LPM
usb_enable_usb2_hardware_lpm(udev);
// Try to enable USB3 LTM
usb_enable_ltm(udev);
}
usb_unlock_port(port_dev);
return *__slate_slot_3;
}
}

#[unsafe(no_mangle)]
extern "C" fn usb_remote_wakeup(udev: *mut usb_device) -> i32 {
let mut status: i32 = 0;
usb_lock_device!(udev);
if ((unsafe { (*udev).state }) as u32) == 8 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"usb %sresume\n".as_ptr() as *const u8, c"wakeup-".as_ptr() as *mut u8) };
}
status = unsafe { usb_autoresume_device(udev) };
if status == 0 {
// Let the drivers do their thing, then...
unsafe { usb_autosuspend_device(udev) };
}
}
usb_unlock_device!(udev);
status
}

/// Returns 1 if there was a remote wakeup and a connect status change.
fn hub_handle_remote_wakeup(hub: *mut usb_hub, port: u32, portstatus: u16, portchange: u16) -> i32 {
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port.wrapping_sub(1) as isize) };
let mut connect_change: i32 = 0;
let mut link_state: u16 = 0;
let mut ret: i32 = 0;
let hdev: *mut usb_device = unsafe { (*hub).hdev };
let udev: *mut usb_device = unsafe { (*port_dev).child };
if hub_is_superspeed(hdev) == 0 {
if (portchange as i32) & 4 == 0 {
return 0;
}
usb_clear_port_feature(hdev, port as i32, 18i32);
} else {
link_state = ((portstatus as i32) & 480) as u16;
if udev.is_null() || ((unsafe { (*udev).state }) as u32) != 8 || (link_state as i32) != 0 && (link_state as i32) != 32 && (link_state as i32) != 64 {
return 0;
}
}
if !udev.is_null() {
// TRSMRCY = 10 msec
unsafe { msleep(10u32) };
usb_unlock_port(port_dev);
ret = usb_remote_wakeup(udev);
usb_lock_port(port_dev);
if ret < 0 {
connect_change = 1;
}
} else {
ret = 19i32.wrapping_neg();
hub_port_disable(hub, port as i32, 1i32);
}
connect_change
}

fn check_ports_changed(hub: *mut usb_hub) -> i32 {
let mut port1: i32 = 1;
while port1 <= unsafe { (*(*hub).hdev).maxchild } {
let mut portstatus: u16 = 0;
let mut portchange: u16 = 0;
let status: i32 = usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(portstatus), core::ptr::addr_of_mut!(portchange));
if status == 0 && portchange != 0 {
return 1;
}
port1 = port1.wrapping_add(1);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.hub.hub_suspend")]
extern "C" fn hub_suspend(intf: *mut usb_interface, msg: pm_message) -> i32 {
let hub: *mut usb_hub = usb_get_intfdata(intf) as *mut usb_hub;
let hdev: *mut usb_device = unsafe { (*hub).hdev };
let mut port1: u32 = 1;
// Warn if children aren't already suspended.
// Also, add up the number of wakeup-enabled descendants.
unsafe {
(*hub).wakeup_enabled_descendants = 0;
}
while port1 <= ((unsafe { (*hdev).maxchild }) as u32) {
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize) };
let udev: *mut usb_device = unsafe { (*port_dev).child };
if !udev.is_null() && ((unsafe { (*udev).__slate_bits_0.__get_can_submit() }) as i32) != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"device %s not suspended yet\n".as_ptr() as *const u8, dev_name(core::ptr::addr_of_mut!((*udev).dev) as *const device)) };
if PMSG_IS_AUTO!(msg) {
return 16i32.wrapping_neg();
}
}
if !udev.is_null() {
let __v1: u32 = usb_wakeup_enabled_descendants(udev);
let __v3: u32 = unsafe { (*hub).wakeup_enabled_descendants };
unsafe {
(*hub).wakeup_enabled_descendants = __v3.wrapping_add(__v1);
}
}
port1 = port1.wrapping_add(1);
}
if ((unsafe { (*hdev).__slate_bits_1.__get_do_remote_wakeup() }) as i32) != 0 && ((unsafe { (*hub).__slate_bits_0.__get_quirk_check_port_auto_suspend() }) as i32) != 0 {
// check if there are changes pending on hub ports
if check_ports_changed(hub) != 0 {
if PMSG_IS_AUTO!(msg) {
return 16i32.wrapping_neg();
}
pm_wakeup_event(unsafe { core::ptr::addr_of_mut!((*hdev).dev) }, 2000u32);
}
}
if hub_is_superspeed(hdev) != 0 && ((unsafe { (*hdev).__slate_bits_1.__get_do_remote_wakeup() }) as i32) != 0 {
// Enable hub to send remote wakeup for all ports.
port1 = 1;
while port1 <= ((unsafe { (*hdev).maxchild }) as u32) {
set_port_feature(hdev, (port1 | 256u32 | 512u32 | 1024u32) as i32, 27i32);
port1 = port1.wrapping_add(1);
}
}
// stop hub_wq and related activity
hub_quiesce(hub, 2u32);
0
}

/// Report wakeup requests from the ports of a resuming root hub
fn report_wakeup_requests(hub: *mut usb_hub) {
let hdev: *mut usb_device = unsafe { (*hub).hdev };
let mut udev: *mut usb_device = unsafe { core::mem::zeroed() };
let mut resuming_ports: u64 = 0;
let mut i: i32 = 0;
if !unsafe { (*hdev).parent }.is_null() {
return;
}
// Not a root hub
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*hdev).bus });
if (unsafe { (*(*hcd).driver).get_resuming_ports }) != None {
// The get_resuming_ports() method returns a bitmap (origin 0)
// of ports which have started wakeup signaling but have not
// yet finished resuming.  During system resume we will
// resume all the enabled ports, regardless of any wakeup
// signals, which means the wakeup requests would be lost.
// To prevent this, report them to the PM core here.
resuming_ports = unsafe { unsafe { (*(*hcd).driver).get_resuming_ports }.unwrap()(hcd) };
i = 0;
while i < unsafe { (*hdev).maxchild } {
if test_bit!(i, core::ptr::addr_of_mut!(resuming_ports)) != 0 {
udev = unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child };
if !udev.is_null() {
pm_wakeup_event(unsafe { core::ptr::addr_of_mut!((*udev).dev) }, 0u32);
}
}
i = i.wrapping_add(1);
}
}
}

#[unsafe(link_section = ".text.slate_distinct.hub.hub_resume")]
extern "C" fn hub_resume(intf: *mut usb_interface) -> i32 {
let hub: *mut usb_hub = usb_get_intfdata(intf) as *mut usb_hub;
hub_activate(hub, 4u32);
// This should be called only for system resume, not runtime resume.
// We can't tell the difference here, so some wakeup requests will be
// reported at the wrong time or more than once.  This shouldn't
// matter much, so long as they do get reported.
report_wakeup_requests(hub);
0
}

#[unsafe(link_section = ".text.slate_distinct.hub.hub_reset_resume")]
extern "C" fn hub_reset_resume(intf: *mut usb_interface) -> i32 {
let hub: *mut usb_hub = usb_get_intfdata(intf) as *mut usb_hub;
hub_activate(hub, 5u32);
0
}

/// usb_root_hub_lost_power - called by HCD if the root hub lost Vbus power
/// @rhdev: struct usb_device for the root hub
///
/// The USB host controller driver calls this function when its root hub
/// is resumed and Vbus power has been interrupted or the controller
/// has been reset.  The routine marks @rhdev as having lost power.
/// When the hub driver is resumed it will take notice and carry out
/// power-session recovery for all the "USB-PERSIST"-enabled child devices;
/// the others will be disconnected.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_root_hub_lost_power")]
extern "C" fn usb_root_hub_lost_power(rhdev: *mut usb_device) {
while false {
}
unsafe { _dev_notice(core::ptr::addr_of_mut!((*rhdev).dev) as *const device, c"root hub lost power or was reset\n".as_ptr() as *const u8) };
unsafe {
(*rhdev).__slate_bits_1.__set_reset_resume(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_root_hub_lost_power_710: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device)>, *mut ()>(Some(usb_root_hub_lost_power)) };

static usb3_lpm_names: __SlateAlign16<[*const u8; 4]> = __SlateAlign16([(b"U0\0".as_ptr() as *mut u8) as *const u8, (b"U1\0".as_ptr() as *mut u8) as *const u8, (b"U2\0".as_ptr() as *mut u8) as *const u8, (b"U3\0".as_ptr() as *mut u8) as *const u8]);

/// Send a Set SEL control transfer to the device, prior to enabling
/// device-initiated U1 or U2.  This lets the device know the exit latencies from
/// the time the device initiates a U1 or U2 exit, to the time it will receive a
/// packet from the host.
///
/// This function will fail if the SEL or PEL values for udev are greater than
/// the maximum allowed values for the link state to be enabled.
fn usb_req_set_sel(udev: *mut usb_device) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut u1_sel: u64 = 0;
let mut u1_pel: u64 = 0;
let mut u2_sel: u64 = 0;
let mut u2_pel: u64 = 0;
let mut ret: i32 = 0;
if unsafe { (*udev).parent }.is_null() || ((unsafe { (*udev).speed }) as u32) < 5 || ((unsafe { (*udev).__slate_bits_0.__get_lpm_capable() }) as i32) == 0 {
return 0;
}
// Convert SEL and PEL stored in ns to us
u1_sel = (unsafe { (*udev).u1_params.sel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64;
u1_pel = (unsafe { (*udev).u1_params.pel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64;
u2_sel = (unsafe { (*udev).u2_params.sel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64;
u2_pel = (unsafe { (*udev).u2_params.pel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64;
// Make sure that the calculated SEL and PEL values for the link
// state we're enabling aren't bigger than the max SEL/PEL
// value that will fit in the SET SEL control transfer.
// Otherwise the device would get an incorrect idea of the exit
// latency for the link state, and could start a device-initiated
// U1/U2 when the exit latencies are too high.
if u1_sel > 255 || u1_pel > 255 || u2_sel > 65535 || u2_pel > 65535 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Device-initiated U1/U2 disabled due to long SEL or PEL\n".as_ptr() as *const u8) };
}
return 22i32.wrapping_neg();
}
// usb_enable_lpm() can be called as part of a failed device reset,
// which may be initiated by an error path of a mass storage driver.
// Therefore, use GFP_NOIO.
let mut __v0: *mut usb_set_sel_req;
let __obj_size: u64 = size_mul(core::mem::size_of::<usb_set_sel_req>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3072u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3072u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut usb_set_sel_req;
let sel_values: *mut usb_set_sel_req = __v0;
if sel_values.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*sel_values).u1_sel = u1_sel as u8;
}
unsafe {
(*sel_values).u1_pel = u1_pel as u8;
}
unsafe {
(*sel_values).u2_sel = u2_sel as u16;
}
unsafe {
(*sel_values).u2_pel = u2_pel as u16;
}
ret = unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 48u8, 0u8, 0u16, 0u16, sel_values as *mut (), core::mem::size_of::<usb_set_sel_req>() as u16, 5000i32) };
unsafe { kfree(sel_values as *const ()) };
if ret > 0 {
unsafe {
(*udev).__slate_bits_0.__set_lpm_devinit_allow(1u32);
}
}
ret
}

/// Enable or disable device-initiated U1 or U2 transitions.
fn usb_set_device_initiated_lpm(udev: *mut usb_device, state: u32, enable: bool) -> i32 {
let mut ret: i32 = 0;
let mut feature: i32 = 0;
match state as u32 {
1 => {
feature = 48;
}
2 => {
feature = 49;
}
_ => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"%s: Can't %s non-U1 or U2 state.\n".as_ptr() as *const u8, c"usb_set_device_initiated_lpm".as_ptr() as *mut u8, str_enable_disable(enable)) };
return 22i32.wrapping_neg();
}
}
if ((unsafe { (*udev).state }) as u32) != 7 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"%s: Can't %s %s state for unconfigured device.\n".as_ptr() as *const u8, c"usb_set_device_initiated_lpm".as_ptr() as *mut u8, str_enable_disable(enable), usb3_lpm_names.0[(state as u32) as usize]) };
}
return 22i32.wrapping_neg();
}
if enable {
// Now send the control transfer to enable device-initiated LPM
// for either U1 or U2.
ret = unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 3u8, 0u8, feature as u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 5000i32) };
} else {
ret = unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 1u8, 0u8, feature as u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 5000i32) };
}
if ret < 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"%s of device-initiated %s failed.\n".as_ptr() as *const u8, str_enable_disable(enable), usb3_lpm_names.0[(state as u32) as usize]) };
return 16i32.wrapping_neg();
}
0
}

fn usb_set_lpm_timeout(udev: *mut usb_device, state: u32, timeout: i32) -> i32 {
let mut ret: i32 = 0;
let mut feature: i32 = 0;
match state as u32 {
1 => {
feature = 23;
}
2 => {
feature = 24;
}
_ => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"%s: Can't set timeout for non-U1 or U2 state.\n".as_ptr() as *const u8, c"usb_set_lpm_timeout".as_ptr() as *mut u8) };
return 22i32.wrapping_neg();
}
}
if (state as u32) == 1 && timeout > 127 && timeout != 255 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Failed to set %s timeout to 0x%x, which is a reserved value.\n".as_ptr() as *const u8, usb3_lpm_names.0[(state as u32) as usize], timeout) };
return 22i32.wrapping_neg();
}
ret = set_port_feature(unsafe { (*udev).parent }, USB_PORT_LPM_TIMEOUT!(timeout) | ((unsafe { (*udev).portnum }) as i32), feature);
if ret < 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Failed to set %s timeout to 0x%x,error code %i\n".as_ptr() as *const u8, usb3_lpm_names.0[(state as u32) as usize], timeout, ret) };
return 16i32.wrapping_neg();
}
if (state as u32) == 1 {
unsafe {
(*udev).u1_params.timeout = timeout;
}
} else {
unsafe {
(*udev).u2_params.timeout = timeout;
}
}
0
}

/// Don't allow device intiated U1/U2 if device isn't in the configured state,
/// or the system exit latency + one bus interval is greater than the minimum
/// service interval of any active periodic endpoint. See USB 3.2 section 9.4.9
fn usb_device_may_initiate_lpm(udev: *mut usb_device, state: u32) -> bool {
let mut sel: u32 = 0; // us
let mut i: i32 = 0;
let mut j: i32 = 0;
if ((unsafe { (*udev).__slate_bits_0.__get_lpm_devinit_allow() }) as i32) == 0 || unsafe { (*udev).actconfig }.is_null() {
return false;
}
if (state as u32) == 1 {
sel = unsafe { (*udev).u1_params.sel }.wrapping_add(1000).wrapping_sub(1) / 1000;
} else {
if (state as u32) == 2 {
sel = unsafe { (*udev).u2_params.sel }.wrapping_add(1000).wrapping_sub(1) / 1000;
} else {
return false;
}
}
while i < ((unsafe { (*(*udev).actconfig).desc.bNumInterfaces }) as i32) {
let mut intf: *mut usb_interface = unsafe { core::mem::zeroed() };
let mut interval: u32 = 0;
intf = unsafe { *((*(*udev).actconfig).interface.as_mut_ptr() as *mut *mut usb_interface).wrapping_offset(i as isize) };
if !intf.is_null() {
j = 0;
while j < ((unsafe { (*(*intf).cur_altsetting).desc.bNumEndpoints }) as i32) {
let desc: *mut usb_endpoint_descriptor = unsafe { core::ptr::addr_of_mut!((*unsafe { (*(*intf).cur_altsetting).endpoint }.wrapping_offset(j as isize)).desc) };
let __v1: bool = usb_endpoint_xfer_int(desc as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_isoc(desc as *const usb_endpoint_descriptor) != 0;
if __v1 {
interval = (1i32 << ((unsafe { (*desc).bInterval }) as i32).wrapping_sub(1)).wrapping_mul(125) as u32;
if sel.wrapping_add(125) > interval {
return false;
}
}
j = j.wrapping_add(1);
}
}
i = i.wrapping_add(1);
}
true
}

/// Enable the hub-initiated U1/U2 idle timeouts, and enable device-initiated
/// U1/U2 entry.
///
/// We will attempt to enable U1 or U2, but there are no guarantees that the
/// control transfers to set the hub timeout or enable device-initiated U1/U2
/// will be successful.
///
/// If the control transfer to enable device-initiated U1/U2 entry fails, then
/// hub-initiated U1/U2 will be disabled.
///
/// If we cannot set the parent hub U1/U2 timeout, we attempt to let the xHCI
/// driver know about it.  If that call fails, it should be harmless, and just
/// take up more slightly more bus bandwidth for unnecessary U1/U2 exit latency.
fn usb_enable_link_state(hcd: *mut usb_hcd, udev: *mut usb_device, state: u32) -> i32 {
let mut timeout: i32 = 0;
let mut u1_mel: u8 = 0;
let mut u2_mel: u16 = 0;
// Skip if the device BOS descriptor couldn't be read
if unsafe { (*udev).bos }.is_null() {
return 22i32.wrapping_neg();
}
u1_mel = unsafe { (*(*(*udev).bos).ss_cap).bU1devExitLat };
u2_mel = unsafe { (*(*(*udev).bos).ss_cap).bU2DevExitLat };
// If the device says it doesn't have *any* exit latency to come out of
// U1 or U2, it's probably lying.  Assume it doesn't implement that link
// state.
if (state as u32) == 1 && (u1_mel as i32) == 0 || (state as u32) == 2 && (u2_mel as i32) == 0 {
return 22i32.wrapping_neg();
}
// We allow the host controller to set the U1/U2 timeout internally
// first, so that it can change its schedule to account for the
// additional latency to send data to a device in a lower power
// link state.
timeout = unsafe { unsafe { (*(*hcd).driver).enable_usb3_lpm_timeout }.unwrap()(hcd, udev, state) };
// xHCI host controller doesn't want to enable this LPM state.
if timeout == 0 {
return 22i32.wrapping_neg();
}
if timeout < 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Could not enable %s link state, xHCI error %i.\n".as_ptr() as *const u8, usb3_lpm_names.0[(state as u32) as usize], timeout) };
return timeout;
}
if usb_set_lpm_timeout(udev, state, timeout) != 0 {
// If we can't set the parent hub U1/U2 timeout,
// device-initiated LPM won't be allowed either, so let the xHCI
// host know that this link state won't be enabled.
unsafe { unsafe { (*(*hcd).driver).disable_usb3_lpm_timeout }.unwrap()(hcd, udev, state) };
return 16i32.wrapping_neg();
}
if (state as u32) == 1 {
unsafe {
(*udev).__slate_bits_0.__set_usb3_lpm_u1_enabled(1u32);
}
} else {
if (state as u32) == 2 {
unsafe {
(*udev).__slate_bits_0.__set_usb3_lpm_u2_enabled(1u32);
}
}
}
0
}

/// Disable the hub-initiated U1/U2 idle timeouts, and disable device-initiated
/// U1/U2 entry.
///
/// If this function returns -EBUSY, the parent hub will still allow U1/U2 entry.
/// If zero is returned, the parent will not allow the link to go into U1/U2.
///
/// If zero is returned, device-initiated U1/U2 entry may still be enabled, but
/// it won't have an effect on the bus link state because the parent hub will
/// still disallow device-initiated U1/U2 entry.
///
/// If zero is returned, the xHCI host controller may still think U1/U2 entry is
/// possible.  The result will be slightly more bus bandwidth will be taken up
/// (to account for U1/U2 exit latency), but it should be harmless.
fn usb_disable_link_state(hcd: *mut usb_hcd, udev: *mut usb_device, state: u32) -> i32 {
match state as u32 {
1 | 2 => {
}
_ => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"%s: Can't disable non-U1 or U2 state.\n".as_ptr() as *const u8, c"usb_disable_link_state".as_ptr() as *mut u8) };
return 22i32.wrapping_neg();
}
}
if usb_set_lpm_timeout(udev, state, 0i32) != 0 {
return 16i32.wrapping_neg();
}
if (unsafe { unsafe { (*(*hcd).driver).disable_usb3_lpm_timeout }.unwrap()(hcd, udev, state) }) != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Could not disable xHCI %s timeout, bus schedule bandwidth may be impacted.\n".as_ptr() as *const u8, usb3_lpm_names.0[(state as u32) as usize]) };
}
// As soon as usb_set_lpm_timeout(0) return 0, hub initiated LPM
// is disabled. Hub will disallows link to enter U1/U2 as well,
// even device is initiating LPM. Hence LPM is disabled if hub LPM
// timeout set to 0, no matter device-initiated LPM is disabled or
// not.
if (state as u32) == 1 {
unsafe {
(*udev).__slate_bits_0.__set_usb3_lpm_u1_enabled(0u32);
}
} else {
if (state as u32) == 2 {
unsafe {
(*udev).__slate_bits_0.__set_usb3_lpm_u2_enabled(0u32);
}
}
}
0
}

/// Disable hub-initiated and device-initiated U1 and U2 entry.
/// Caller must own the bandwidth_mutex.
///
/// This will call usb_enable_lpm() on failure, which will decrement
/// lpm_disable_count, and will re-enable LPM if lpm_disable_count reaches zero.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_disable_lpm")]
extern "C" fn usb_disable_lpm(udev: *mut usb_device) -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut hcd: core::mem::MaybeUninit<*mut usb_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut usb_hcd = core::ptr::addr_of_mut!(hcd) as *mut *mut usb_hcd;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if udev.is_null() || unsafe { (*udev).parent }.is_null() || ((*udev).speed as u32) < 5 || ((*udev).__slate_bits_0.__get_lpm_capable() as i32) == 0 || ((*udev).state as u32) < 7 {
return 0;
} else {
*__slate_slot_7 = bus_to_hcd((*udev).bus);
if (*__slate_slot_7).is_null() || (*(*(*__slate_slot_7)).driver).disable_usb3_lpm_timeout == None {
return 0;
} else {
let __v4: u32 = (*udev).lpm_disable_count;
(*udev).lpm_disable_count = __v4.wrapping_add(1);
if (*udev).u1_params.timeout == 0 && (*udev).u2_params.timeout == 0 {
return 0;
} else {
// If LPM is enabled, attempt to disable it.
if usb_disable_link_state(*__slate_slot_7, udev, 1u32) == 0 {
if usb_disable_link_state(*__slate_slot_7, udev, 2u32) == 0 {
*__slate_slot_6 = usb_set_device_initiated_lpm(udev, 1u32, false);
if *__slate_slot_6 == 0 {
usb_set_device_initiated_lpm(udev, 2u32, false);
}
return 0;
}
}
let __v1: u32 = (*udev).lpm_disable_count;
(*udev).lpm_disable_count = __v1.wrapping_sub(1);
return 16i32.wrapping_neg();
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_disable_lpm_711: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device) -> i32>, *mut ()>(Some(usb_disable_lpm)) };

/// Grab the bandwidth_mutex before calling usb_disable_lpm()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_unlocked_disable_lpm")]
extern "C" fn usb_unlocked_disable_lpm(udev: *mut usb_device) -> i32 {
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*udev).bus });
let mut ret: i32 = 0;
if hcd.is_null() {
return 22i32.wrapping_neg();
}
unsafe { mutex_lock((*hcd).bandwidth_mutex) };
ret = usb_disable_lpm(udev);
unsafe { mutex_unlock((*hcd).bandwidth_mutex) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_unlocked_disable_lpm_712: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device) -> i32>, *mut ()>(Some(usb_unlocked_disable_lpm)) };

/// Attempt to enable device-initiated and hub-initiated U1 and U2 entry.  The
/// xHCI host policy may prevent U1 or U2 from being enabled.
///
/// Other callers may have disabled link PM, so U1 and U2 entry will be disabled
/// until the lpm_disable_count drops to zero.  Caller must own the
/// bandwidth_mutex.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_enable_lpm")]
extern "C" fn usb_enable_lpm(udev: *mut usb_device) {
if udev.is_null() || unsafe { (*udev).parent }.is_null() || ((unsafe { (*udev).speed }) as u32) < 5 || ((unsafe { (*udev).__slate_bits_0.__get_lpm_capable() }) as i32) == 0 || ((unsafe { (*udev).state }) as u32) < 7 {
return;
}
let __v1: u32 = unsafe { (*udev).lpm_disable_count };
unsafe {
(*udev).lpm_disable_count = __v1.wrapping_sub(1);
}
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*udev).bus });
// Double check that we can both enable and disable LPM.
// Device must be configured to accept set feature U1/U2 timeout.
if hcd.is_null() || (unsafe { (*(*hcd).driver).enable_usb3_lpm_timeout }) == None || (unsafe { (*(*hcd).driver).disable_usb3_lpm_timeout }) == None {
return;
}
if (unsafe { (*udev).lpm_disable_count }) > 0 {
return;
}
let hub: *mut usb_hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
if hub.is_null() {
return;
}
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize) };
if ((unsafe { (*port_dev).__slate_bits_0.__get_usb3_lpm_u1_permit() }) as i32) != 0 {
if usb_enable_link_state(hcd, udev, 1u32) != 0 {
return;
}
}
if ((unsafe { (*port_dev).__slate_bits_0.__get_usb3_lpm_u2_permit() }) as i32) != 0 {
if usb_enable_link_state(hcd, udev, 2u32) != 0 {
return;
}
}
// Enable device initiated U1/U2 with a SetFeature(U1/U2_ENABLE) request
// if system exit latency is short enough and device is configured
if usb_device_may_initiate_lpm(udev, 1u32) {
if usb_set_device_initiated_lpm(udev, 1u32, true) != 0 {
return;
}
if usb_device_may_initiate_lpm(udev, 2u32) {
usb_set_device_initiated_lpm(udev, 2u32, true);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_enable_lpm_713: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device)>, *mut ()>(Some(usb_enable_lpm)) };

/// Grab the bandwidth_mutex before calling usb_enable_lpm()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_unlocked_enable_lpm")]
extern "C" fn usb_unlocked_enable_lpm(udev: *mut usb_device) {
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*udev).bus });
if hcd.is_null() {
return;
}
unsafe { mutex_lock((*hcd).bandwidth_mutex) };
usb_enable_lpm(udev);
unsafe { mutex_unlock((*hcd).bandwidth_mutex) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_unlocked_enable_lpm_714: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device)>, *mut ()>(Some(usb_unlocked_enable_lpm)) };

/// usb3 devices use U3 for disabled, make sure remote wakeup is disabled
fn hub_usb3_port_prepare_disable(hub: *mut usb_hub, port_dev: *mut usb_port) {
let udev: *mut usb_device = unsafe { (*port_dev).child };
let mut ret: i32 = 0;
if !udev.is_null() && ((unsafe { (*udev).__slate_bits_1.__get_port_is_suspended() }) as i32) != 0 && ((unsafe { (*udev).__slate_bits_1.__get_do_remote_wakeup() }) as i32) != 0 {
ret = hub_set_port_link_state(hub, (unsafe { (*port_dev).portnum }) as i32, 0u32);
if ret == 0 {
unsafe { msleep(40u32) };
ret = usb_disable_remote_wakeup(udev);
}
if ret != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Port disable: can't disable remote wake\n".as_ptr() as *const u8) };
}
unsafe {
(*udev).__slate_bits_1.__set_do_remote_wakeup(0u32);
}
}
}

/// USB-3 does not have a similar link state as USB-2 that will avoid negotiating
/// a connection with a plugged-in cable but will signal the host when the cable
/// is unplugged. Disable remote wake and set link state to U3 for USB-3 devices
fn hub_port_disable(hub: *mut usb_hub, port1: i32, set_state: i32) -> i32 {
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize) };
let hdev: *mut usb_device = unsafe { (*hub).hdev };
let mut ret: i32 = 0;
if (unsafe { (*hub).error }) == 0 {
if hub_is_superspeed(unsafe { (*hub).hdev }) != 0 {
hub_usb3_port_prepare_disable(hub, port_dev);
ret = hub_set_port_link_state(hub, (unsafe { (*port_dev).portnum }) as i32, 96u32);
} else {
ret = usb_clear_port_feature(hdev, port1, 1i32);
}
}
if !unsafe { (*port_dev).child }.is_null() && set_state != 0 {
usb_set_device_state(unsafe { (*port_dev).child }, 0u32);
}
if ret != 0 && ret != 19i32.wrapping_neg() {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"cannot disable (err = %d)\n".as_ptr() as *const u8, ret) };
}
ret
}

/// usb_port_disable - disable a usb device's upstream port
/// @udev: device to disable
/// Context: @udev locked, must be able to sleep.
///
/// Disables a USB device that isn't in active use.
#[unsafe(no_mangle)]
extern "C" fn usb_port_disable(udev: *mut usb_device) -> i32 {
let hub: *mut usb_hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
hub_port_disable(hub, (unsafe { (*udev).portnum }) as i32, 0i32)
}

/// USB 2.0 spec, 7.1.7.3 / fig 7-29:
///
/// Between connect detection and reset signaling there must be a delay
/// of 100ms at least for debounce and power-settling.  The corresponding
/// timer shall restart whenever the downstream port detects a disconnect.
///
/// Apparently there are some bluetooth and irda-dongles and a number of
/// low-speed devices for which this debounce period may last over a second.
/// Not covered by the spec - but easy to deal with.
///
/// This implementation uses a 1500ms total debounce timeout; if the
/// connection isn't stable by then it returns -ETIMEDOUT.  It checks
/// every 25ms for transient disconnects.  When the port status has been
/// unchanged for 100ms it returns the port status.
#[unsafe(no_mangle)]
extern "C" fn hub_port_debounce(hub: *mut usb_hub, port1: i32, must_be_connected: bool) -> i32 {
let mut ret: i32 = 0;
let mut portchange: u16 = 0;
let mut portstatus: u16 = 0;
let mut connection: u32 = 65535;
let mut total_time: i32 = 0;
let mut stable_time: i32 = 0;
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize) };
loop {
ret = usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(portstatus), core::ptr::addr_of_mut!(portchange));
if ret < 0 {
return ret;
}
if (portchange as i32) & 1 == 0 && (((portstatus as i32) & 1) as u32) == connection {
if !must_be_connected || connection == 1 {
stable_time = stable_time.wrapping_add(25);
}
if stable_time >= 100 {
break;
}
} else {
stable_time = 0;
connection = ((portstatus as i32) & 1) as u32;
}
if (portchange as i32) & 1 != 0 {
usb_clear_port_feature(unsafe { (*hub).hdev }, port1, 16i32);
}
if total_time >= 2000 {
break;
}
unsafe { msleep(25u32) };
total_time = total_time.wrapping_add(25);
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"debounce total %dms stable %dms status 0x%x\n".as_ptr() as *const u8, total_time, stable_time, portstatus as i32) };
}
if stable_time < 100 {
return 110i32.wrapping_neg();
}
portstatus as i32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_ep0_reinit")]
extern "C" fn usb_ep0_reinit(udev: *mut usb_device) {
unsafe { usb_disable_endpoint(udev, 128u32, true) };
unsafe { usb_disable_endpoint(udev, 0u32, true) };
unsafe { usb_enable_endpoint(udev, core::ptr::addr_of_mut!((*udev).ep0), true) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_ep0_reinit_715: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device)>, *mut ()>(Some(usb_ep0_reinit)) };

fn hub_set_address(udev: *mut usb_device, devnum: i32) -> i32 {
let mut retval: i32 = 0;
let mut timeout_ms: u32 = 5000;
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*udev).bus });
let hub: *mut usb_hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
if ((unsafe { (*(*hub).hdev).quirks }) as u64) & BIT!(16i32) != 0 {
timeout_ms = 500;
}
// The host controller will choose the device address,
// instead of the core having chosen it earlier
if (unsafe { (*(*hcd).driver).address_device }) == None && devnum <= 1 {
return 22i32.wrapping_neg();
}
if ((unsafe { (*udev).state }) as u32) == 6 {
return 0;
}
if ((unsafe { (*udev).state }) as u32) != 5 {
return 22i32.wrapping_neg();
}
if (unsafe { (*(*hcd).driver).address_device }) != None {
retval = unsafe { unsafe { (*(*hcd).driver).address_device }.unwrap()(hcd, udev, timeout_ms) };
} else {
retval = unsafe { usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 5u8, 0u8, devnum as u16, 0u16, core::ptr::null_mut::<()>(), 0u16, timeout_ms as i32) };
}
if retval == 0 {
update_devnum(udev, devnum);
// Device now using proper address.
usb_set_device_state(udev, 6u32);
usb_ep0_reinit(udev);
}
retval
}

/// There are reports of USB 3.0 devices that say they support USB 2.0 Link PM
/// when they're plugged into a USB 2.0 port, but they don't work when LPM is
/// enabled.
///
/// Only enable USB 2.0 Link PM if the port is internal (hardwired), or the
/// device says it supports the new USB 2.0 Link PM errata by setting the BESL
/// support bit in the BOS descriptor.
fn hub_set_initial_usb2_lpm_policy(udev: *mut usb_device) {
let hub: *mut usb_hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
let mut connect_type: i32 = 0;
if ((unsafe { (*udev).__slate_bits_0.__get_usb2_hw_lpm_capable() }) as i32) == 0 || unsafe { (*udev).bos }.is_null() {
return;
}
if !hub.is_null() {
connect_type = ((unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize))).connect_type }) as u32) as i32;
}
if (unsafe { (*(*(*udev).bos).ext_cap).bmAttributes }) & 4u32 != 0 || connect_type == 2 {
unsafe {
(*udev).__slate_bits_0.__set_usb2_hw_lpm_allowed(1u32);
}
unsafe { usb_enable_usb2_hardware_lpm(udev) };
}
}

fn hub_enable_device(udev: *mut usb_device) -> i32 {
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*udev).bus });
if (unsafe { (*(*hcd).driver).enable_device }) == None {
return 0;
}
if ((unsafe { (*udev).state }) as u32) == 6 {
return 0;
}
if ((unsafe { (*udev).state }) as u32) != 5 {
return 22i32.wrapping_neg();
}
unsafe { unsafe { (*(*hcd).driver).enable_device }.unwrap()(hcd, udev) }
}

/// Get the bMaxPacketSize0 value during initialization by reading the
/// device's device descriptor.  Since we don't already know this value,
/// the transfer is unsafe and it ignores I/O errors, only testing for
/// reasonable received values.
///
/// For "old scheme" initialization, size will be 8 so we read just the
/// start of the device descriptor, which should work okay regardless of
/// the actual bMaxPacketSize0 value.  For "new scheme" initialization,
/// size will be 64 (and buf will point to a sufficiently large buffer),
/// which might not be kosher according to the USB spec but it's what
/// Windows does and what many devices expect.
///
/// Returns: bMaxPacketSize0 or a negative error code.
fn get_bMaxPacketSize0(udev: *mut usb_device, buf: *mut usb_device_descriptor, size: i32, first_time: bool) -> i32 {
let mut rc: i32 = 0;
// Retry on all errors; some devices are flakey.
// 255 is for WUSB devices, we actually need to use
// 512 (WUSB1.0[4.8.1]).
for i in 0i32..3i32 {
// Start with invalid values in case the transfer fails
unsafe {
(*buf).bMaxPacketSize0 = 0;
}
unsafe {
(*buf).bDescriptorType = 0;
}
rc = unsafe { usb_control_msg(udev, usb_rcvctrlpipe!(udev, 0i32), 6u8, 128u8, 256u16, 0u16, buf as *mut (), size as u16, initial_descriptor_timeout) };
match (unsafe { (*buf).bMaxPacketSize0 }) as i32 {
8 | 9 | 16 | 32 | 64 => {
if ((unsafe { (*buf).bDescriptorType }) as i32) == 1 {
rc = (unsafe { (*buf).bMaxPacketSize0 }) as i32;
} else {
if rc >= 0 {
rc = 71i32.wrapping_neg();
}
}
}
_ => {
if rc >= 0 {
rc = 71i32.wrapping_neg();
}
}
}
// Some devices time out if they are powered on
// when already connected. They need a second
// reset, so return early. But only on the first
// attempt, lest we get into a time-out/reset loop.
if rc > 0 || rc == 110i32.wrapping_neg() && first_time && ((unsafe { (*udev).speed }) as u32) > 2 {
break;
}
}
rc
}

/// Reset device, (re)assign address, get device descriptor.
/// Device connection must be stable, no more debouncing needed.
/// Returns device in USB_STATE_ADDRESS, except on error.
///
/// If this is called for an already-existing device (as part of
/// usb_reset_and_verify_device), the caller must own the device lock and
/// the port lock.  For a newly detected device that is not accessible
/// through any global pointers, it's not necessary to lock the device,
/// but it is still necessary to lock the port.
///
/// For a newly detected device, @dev_descr must be NULL.  The device
/// descriptor retrieved from the device will then be stored in
/// @udev->descriptor.  For an already existing device, @dev_descr
/// must be non-NULL.  The device descriptor will be stored there,
/// not in @udev->descriptor, because descriptors for registered
/// devices are meant to be immutable.
fn hub_port_init(hub: *mut usb_hub, udev: *mut usb_device, port1: i32, retry_counter: i32, dev_descr: *mut usb_device_descriptor) -> i32 {
let mut __slate_compound_29: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_31: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut delay: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(delay) as *mut u32;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut descr: core::mem::MaybeUninit<*mut usb_device_descriptor> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut usb_device_descriptor = core::ptr::addr_of_mut!(descr) as *mut *mut usb_device_descriptor;
let mut buf: core::mem::MaybeUninit<*mut usb_device_descriptor> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut usb_device_descriptor = core::ptr::addr_of_mut!(buf) as *mut *mut usb_device_descriptor;
let mut maxp0: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(maxp0) as *mut i32;
let mut initial: bool = unsafe { core::mem::zeroed() };
let mut do_new_scheme: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut bool = core::ptr::addr_of_mut!(do_new_scheme) as *mut bool;
let mut driver_name: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *const u8 = core::ptr::addr_of_mut!(driver_name) as *mut *const u8;
let mut devnum: i32 = unsafe { core::mem::zeroed() };
let mut speed: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *const u8 = core::ptr::addr_of_mut!(speed) as *mut *const u8;
let mut oldspeed: u32 = unsafe { core::mem::zeroed() };
let mut delay: u32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut operations: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut i32 = core::ptr::addr_of_mut!(operations) as *mut i32;
let mut retries: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i32 = core::ptr::addr_of_mut!(retries) as *mut i32;
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
let mut hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
let mut hdev: *mut usb_device = unsafe { core::mem::zeroed() };
unsafe {
hdev = (*hub).hdev;
hcd = bus_to_hcd((*hdev).bus);
port_dev = *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize);
let __slate_auto_bytes_1 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
delay = 10;
oldspeed = (*udev).speed;
let __slate_auto_bytes_5 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
devnum = (*udev).devnum;
let __slate_auto_bytes_6 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 1usize);
initial = dev_descr.is_null();
let __slate_auto_bytes_8 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_10 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_11 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_5 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_6 = _kmalloc_noprof(64u64, 3072u32, *{
__slate_compound_29 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_29)
});
alloc_tag_restore!();
} else {
*__slate_slot_6 = _kmalloc_noprof(64u64, 3072u32, *{
__slate_compound_31 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_31)
});
}
*__slate_slot_10 = *__slate_slot_6 as *mut usb_device_descriptor;
if (*__slate_slot_10).is_null() {
return 12i32.wrapping_neg();
} else {
// root hub ports have a slightly longer reset period
// (from USB 2.0 spec, section 7.1.7.5)
if unsafe { (*hdev).parent }.is_null() {
delay = 60;
if port1 == ((*(*hdev).bus).otg_port as i32) {
(*(*hdev).bus).__slate_bits_0.__set_b_hnp_enable(0u32);
}
}
// Some low speed devices have problems with the quick delay, so
//
//  be a bit pessimistic with those devices. RHbug #23670
if (oldspeed as u32) == 1 {
delay = 200;
}
'__join_2: {
// Reset the device; full speed may morph to high speed
//
// FIXME a USB 2.0 device may morph into SuperSpeed on reset.
*__slate_slot_20 = hub_port_reset(hub, port1, udev, delay, false);
if *__slate_slot_20 >= 0 {
// error or disconnect
// success, speed is known
*__slate_slot_20 = 19i32.wrapping_neg();
// Don't allow speed changes at reset, except usb 3.0 to faster
if !((oldspeed as u32) != 0 && (oldspeed as u32) != ((*udev).speed as u32) && !((oldspeed as u32) == 5 && ((*udev).speed as u32) > (oldspeed as u32))) {
'__join_90: {
oldspeed = (*udev).speed;
if initial {
// USB 2.0 section 5.5.3 talks about ep0 maxpacket ...
// it's fixed size except for full speed devices.
let __t1: u32 = (*udev).speed as u32;
if __t1 != 6 {
if __t1 != 5 {
if __t1 == 3 {
(*udev).ep0.desc.wMaxPacketSize = 64;
// fixed at 64
break '__join_90;
} else {
if __t1 == 2 {
(*udev).ep0.desc.wMaxPacketSize = 64;
// 8, 16, 32, or 64
// to determine the ep0 maxpacket size, try to read
// the device descriptor to get bMaxPacketSize0 and
// then correct our initial guess.
break '__join_90;
} else {
if __t1 == 1 {
(*udev).ep0.desc.wMaxPacketSize = 8;
// fixed at 8
break '__join_90;
} else {
break '__join_2;
}
}
}
}
}
(*udev).ep0.desc.wMaxPacketSize = 512;
}
}
*__slate_slot_16 = usb_speed_string((*udev).speed);
// The controller driver may be NULL if the controller device
// is the middle device between platform device and roothub.
// This middle device may not need a device driver due to
// all hardware control can be at platform device driver, this
// platform device is usually a dual-role USB controller device.
if !unsafe { (*(*(*udev).bus).controller).driver }.is_null() {
*__slate_slot_14 = (*(*(*(*udev).bus).controller).driver).name;
} else {
*__slate_slot_14 = (*(*(*(*udev).bus).sysdev).driver).name;
}
if ((*udev).speed as u32) < 5 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"%s %s USB device number %d using %s\n".as_ptr() as *const u8, if initial { c"new".as_ptr() as *mut u8 } else { c"reset".as_ptr() as *mut u8 }, *__slate_slot_16, devnum, *__slate_slot_14);
}
if initial {
// Set up TT records, if needed
if !unsafe { (*hdev).tt }.is_null() {
(*udev).tt = (*hdev).tt;
(*udev).ttport = (*hdev).ttport;
} else {
if ((*udev).speed as u32) != 3 && ((*hdev).speed as u32) == 3 {
if unsafe { (*hub).tt.hub }.is_null() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"parent hub has no TT\n".as_ptr() as *const u8);
*__slate_slot_20 = 22i32.wrapping_neg();
break '__join_2;
} else {
(*udev).tt = core::ptr::addr_of_mut!((*hub).tt);
(*udev).ttport = port1;
}
}
}
}
// Why interleave GET_DESCRIPTOR and SET_ADDRESS this way?
// Because device hardware and firmware is sometimes buggy in
// this area, and this is how Linux has done it for ages.
// Change it cautiously.
//
// NOTE:  If use_new_scheme() is true we will start by issuing
// a 64-byte GET_DESCRIPTOR request.  This is what Windows does,
// so it may help with some non-standards-compliant devices.
// Otherwise we start with SET_ADDRESS and then try to read the
// first 8 bytes of the device descriptor to get the ep0 maxpacket
// value.
*__slate_slot_13 = use_new_scheme(udev, retry_counter, port_dev);
*__slate_slot_22 = 0;
'__join_34: {
'__join_74: {
'__join_71: {
'__join_68: {
'__join_64: {
'__join_53: {
'__loop_35: loop {
if *__slate_slot_22 < 2 {
if hub_port_stop_enumerate(hub, port1, *__slate_slot_22) {
break '__join_74;
} else {
'__join_36: {
if *__slate_slot_13 {
*__slate_slot_20 = hub_enable_device(udev);
if *__slate_slot_20 < 0 {
break '__join_71;
} else {
*__slate_slot_11 = get_bMaxPacketSize0(udev, *__slate_slot_10, 64i32, *__slate_slot_22 == 0);
if *__slate_slot_11 > 0 && !initial && *__slate_slot_11 != ((*udev).descriptor.bMaxPacketSize0 as i32) {
break '__join_68;
} else {
*__slate_slot_20 = hub_port_reset(hub, port1, udev, delay, false);
if *__slate_slot_20 < 0 {
break '__join_2;
} else {
// error or disconnect
if (oldspeed as u32) != ((*udev).speed as u32) {
break '__join_64;
} else {
if *__slate_slot_11 < 0 {
if *__slate_slot_11 != 19i32.wrapping_neg() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"device descriptor read/64, error %d\n".as_ptr() as *const u8, *__slate_slot_11);
}
*__slate_slot_20 = *__slate_slot_11;
break '__join_36;
}
}
}
}
}
}
*__slate_slot_21 = 0;
while *__slate_slot_21 < 2 {
*__slate_slot_20 = hub_set_address(udev, devnum);
if *__slate_slot_20 >= 0 {
break;
} else {
msleep(200u32);
let __v4: i32 = *__slate_slot_21;
*__slate_slot_21 = __v4.wrapping_add(1);
}
}
if *__slate_slot_20 < 0 {
break '__join_53;
} else {
if ((*udev).speed as u32) >= 5 {
devnum = (*udev).devnum;
while false {
}
_dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"%s SuperSpeed%s%s USB device number %d using %s\n".as_ptr() as *const u8, if !unsafe { (*udev).config }.is_null() { c"reset".as_ptr() as *mut u8 } else { c"new".as_ptr() as *mut u8 }, if ((*udev).speed as u32) == 6 { c" Plus".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ((*udev).ssp_rate as u32) == 3 { c" Gen 2x2".as_ptr() as *mut u8 } else { if ((*udev).ssp_rate as u32) == 1 { c" Gen 2x1".as_ptr() as *mut u8 } else { if ((*udev).ssp_rate as u32) == 2 { c" Gen 1x2".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 } } }, devnum, *__slate_slot_14);
}
// cope with hardware quirkiness:
//  - let SET_ADDRESS settle, some device hardware wants it
//  - read ep0 maxpacket even for high and low speed,
msleep(10u32);
if *__slate_slot_13 {
break '__join_34;
} else {
*__slate_slot_11 = get_bMaxPacketSize0(udev, *__slate_slot_10, 8i32, *__slate_slot_22 == 0);
if *__slate_slot_11 < 0 {
*__slate_slot_20 = *__slate_slot_11;
if *__slate_slot_20 != 19i32.wrapping_neg() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"device descriptor read/8, error %d\n".as_ptr() as *const u8, *__slate_slot_20);
}
} else {
break '__loop_35;
}
}
}
}
let __v1: i32 = *__slate_slot_22;
*__slate_slot_22 = __v1.wrapping_add(1);
msleep(100u32);
}
} else {
break '__join_34;
}
}
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !initial && *__slate_slot_11 != ((*udev).descriptor.bMaxPacketSize0 as i32) {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"device reset changed ep0 maxpacket size!\n".as_ptr() as *const u8);
*__slate_slot_20 = 19i32.wrapping_neg();
break '__join_2;
} else {
*__slate_slot_2 = (*(*udev).parent).hub_delay as u32;
(*udev).hub_delay = min_t__2!(u32, *__slate_slot_2, 65535i32) as u16;
*__slate_slot_20 = usb_set_isoch_delay(udev);
if *__slate_slot_20 != 0 {
*__slate_slot_20 = 0;
break '__join_34;
} else {
break '__join_34;
}
}
}
if *__slate_slot_20 != 19i32.wrapping_neg() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"device not accepting address %d, error %d\n".as_ptr() as *const u8, devnum, *__slate_slot_20);
break '__join_2;
} else {
break '__join_2;
}
}
*__slate_slot_20 = 19i32.wrapping_neg();
break '__join_2;
}
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"device reset changed ep0 maxpacket size!\n".as_ptr() as *const u8);
*__slate_slot_20 = 19i32.wrapping_neg();
break '__join_2;
}
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"hub failed to enable device, error %d\n".as_ptr() as *const u8, *__slate_slot_20);
break '__join_2;
}
*__slate_slot_20 = 19i32.wrapping_neg();
}
if *__slate_slot_20 == 0 {
// Check the ep0 maxpacket guess and correct it if necessary.
// maxp0 is the value stored in the device descriptor;
// i is the value it encodes (logarithmic for SuperSpeed or greater).
*__slate_slot_19 = *__slate_slot_11;
if ((*udev).speed as u32) >= 5 {
if *__slate_slot_11 <= 16 {
*__slate_slot_19 = 1i32 << *__slate_slot_11;
} else {
*__slate_slot_19 = 0;
}
// Invalid
}
if usb_endpoint_maxp(core::ptr::addr_of_mut!((*udev).ep0.desc) as *const usb_endpoint_descriptor) == *__slate_slot_19 {
// Initial ep0 maxpacket guess is right
} else {
if (((*udev).speed as u32) == 2 || ((*udev).speed as u32) == 3) && (*__slate_slot_19 == 8 || *__slate_slot_19 == 16 || *__slate_slot_19 == 32 || *__slate_slot_19 == 64) || ((*udev).speed as u32) >= 5 && *__slate_slot_19 > 0 {
// Initial guess is wrong; use the descriptor's value
if ((*udev).speed as u32) != 2 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Using ep0 maxpacket: %d\n".as_ptr() as *const u8, *__slate_slot_19);
}
(*udev).ep0.desc.wMaxPacketSize = *__slate_slot_19 as u16;
usb_ep0_reinit(udev);
} else {
// Initial guess is wrong and descriptor's value is invalid
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Invalid ep0 maxpacket: %d\n".as_ptr() as *const u8, *__slate_slot_11);
*__slate_slot_20 = 90i32.wrapping_neg();
break '__join_2;
}
}
*__slate_slot_9 = usb_get_device_descriptor(udev);
if IS_ERR(*__slate_slot_9 as *const ()) {
*__slate_slot_20 = PTR_ERR(*__slate_slot_9 as *const ()) as i32;
if *__slate_slot_20 != 19i32.wrapping_neg() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"device descriptor read/all, error %d\n".as_ptr() as *const u8, *__slate_slot_20);
}
} else {
if initial {
(*udev).descriptor = *(*__slate_slot_9);
} else {
*dev_descr = *(*__slate_slot_9);
}
kfree(*__slate_slot_9 as *const ());
// Some superspeed devices have finished the link training process
// and attached to a superspeed hub port, but the device descriptor
// got from those devices show they aren't superspeed devices. Warm
// reset the port attached by the devices can fix them.
if ((*udev).speed as u32) >= 5 && ((*udev).descriptor.bcdUSB as i32) < 768 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"got a wrong device descriptor, warm reset device\n".as_ptr() as *const u8);
hub_port_reset(hub, port1, udev, 50u32, true);
*__slate_slot_20 = 22i32.wrapping_neg();
} else {
usb_detect_quirks(udev);
if ((*udev).descriptor.bcdUSB as i32) >= 513 {
*__slate_slot_20 = usb_get_bos_descriptor(udev);
if *__slate_slot_20 == 0 {
let __t0 = usb_device_supports_lpm(udev) as u32;
(*udev).__slate_bits_0.__set_lpm_capable(__t0);
(*udev).lpm_disable_count = 1;
usb_set_lpm_parameters(udev);
usb_req_set_sel(udev);
}
}
*__slate_slot_20 = 0;
// notify HCD that we have a device connected and addressed
if (*(*hcd).driver).update_device != None {
unsafe { (*(*hcd).driver).update_device }.unwrap()(hcd, udev);
}
hub_set_initial_usb2_lpm_policy(udev);
}
}
}
}
}
}
if *__slate_slot_20 != 0 {
hub_port_disable(hub, port1, 0i32);
update_devnum(udev, devnum); // for disconnect processing
}
kfree(*__slate_slot_10 as *const ());
return *__slate_slot_20;
}
}
}

fn check_highspeed(hub: *mut usb_hub, udev: *mut usb_device, port1: i32) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut status: i32 = 0;
if ((unsafe { (*udev).quirks }) as u64) & BIT!(8i32) != 0 {
return;
}
let mut __v0: *mut usb_qualifier_descriptor;
let __obj_size: u64 = size_mul(core::mem::size_of::<usb_qualifier_descriptor>() as u64, 1u64);
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
__v0 = __v1 as *mut usb_qualifier_descriptor;
let qual: *mut usb_qualifier_descriptor = __v0;
if qual.is_null() {
return;
}
status = unsafe { usb_get_descriptor(udev, 6u8, 0u8, qual as *mut (), core::mem::size_of::<usb_qualifier_descriptor>() as i32) };
if (status as u64) == (core::mem::size_of::<usb_qualifier_descriptor>() as u64) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"not running at top speed; connect to a high speed hub\n".as_ptr() as *const u8) };
// hub LEDs are probably harder to miss than syslog
if ((unsafe { (*hub).__slate_bits_0.__get_has_indicators() }) as i32) != 0 {
unsafe {
*((*hub).indicator.as_mut_ptr() as *mut u8).wrapping_offset(port1.wrapping_sub(1) as isize) = 2;
}
queue_delayed_work(unsafe { system_power_efficient_wq }, unsafe { core::ptr::addr_of_mut!((*hub).leds) }, 0u64);
}
}
unsafe { kfree(qual as *const ()) };
}

fn hub_power_remaining(hub: *mut usb_hub) -> u32 {
let hdev: *mut usb_device = unsafe { (*hub).hdev };
let mut remaining: i32 = 0;
let mut port1: i32 = 1;
if ((unsafe { (*hub).__slate_bits_0.__get_limited_power() }) as i32) == 0 {
return 0;
}
remaining = ((unsafe { (*hdev).bus_mA }) as i32).wrapping_sub((unsafe { (*(*hub).descriptor).bHubContrCurrent }) as i32);
while port1 <= unsafe { (*hdev).maxchild } {
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize) };
let udev: *mut usb_device = unsafe { (*port_dev).child };
let mut unit_load: u32 = 0;
let mut delta: i32 = 0;
if !udev.is_null() {
if hub_is_superspeed(udev) != 0 {
unit_load = 150;
} else {
unit_load = 100;
}
// Unconfigured devices may not use more than one unit load,
// or 8mA for OTG ports
if !unsafe { (*udev).actconfig }.is_null() {
delta = usb_get_max_power(udev, unsafe { (*udev).actconfig }) as i32;
} else {
if port1 != ((unsafe { (*(*udev).bus).otg_port }) as i32) || !unsafe { (*hdev).parent }.is_null() {
delta = unit_load as i32;
} else {
delta = 8;
}
}
if (delta as u32) > unsafe { (*hub).mA_per_port } {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"%dmA is over %umA budget!\n".as_ptr() as *const u8, delta, (*hub).mA_per_port) };
}
remaining = remaining.wrapping_sub(delta);
}
port1 = port1.wrapping_add(1);
}
if remaining < 0 {
while false {
}
unsafe { _dev_warn((*hub).intfdev as *const device, c"%dmA over power budget!\n".as_ptr() as *const u8, remaining.wrapping_neg()) };
remaining = 0;
}
remaining as u32
}

fn descriptors_changed(udev: *mut usb_device, new_device_descriptor: *mut usb_device_descriptor, old_bos: *mut usb_host_bos) -> i32 {
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut changed: i32 = 0;
let mut index: u32 = 0;
let mut serial_len: u32 = 0;
let mut len: u32 = 0;
let mut old_length: u32 = 0;
let mut length: i32 = 0;
if (unsafe { memcmp(core::ptr::addr_of_mut!((*udev).descriptor) as *const (), new_device_descriptor as *const (), core::mem::size_of::<usb_device_descriptor>() as u64) }) != 0 {
return 1;
}
if !old_bos.is_null() && unsafe { (*udev).bos }.is_null() || old_bos.is_null() && !unsafe { (*udev).bos }.is_null() {
return 1;
}
if !unsafe { (*udev).bos }.is_null() {
len = (unsafe { (*(*(*udev).bos).desc).wTotalLength }) as u32;
if len != (((unsafe { (*(*old_bos).desc).wTotalLength }) as i32) as u32) {
return 1;
}
if (unsafe { memcmp((*(*udev).bos).desc as *const (), (*old_bos).desc as *const (), len as u64) }) != 0 {
return 1;
}
}
// Since the idVendor, idProduct, and bcdDevice values in the
// device descriptor haven't changed, we will assume the
// Manufacturer and Product strings haven't changed either.
// But the SerialNumber string could be different (e.g., a
// different flash card of the same brand).
if !unsafe { (*udev).serial }.is_null() {
serial_len = unsafe { strlen((*udev).serial as *const u8) }.wrapping_add(1) as u32;
}
len = serial_len;
while index < (((unsafe { (*udev).descriptor.bNumConfigurations }) as i32) as u32) {
old_length = (unsafe { (*unsafe { (*udev).config }.wrapping_offset(index as isize)).desc.wTotalLength }) as u32;
let mut __v0: u32;
let __UNIQUE_ID_x__718: u32 = len;
let __UNIQUE_ID_y__719: u32 = old_length;
loop {
if __types_ok!(__UNIQUE_ID_x__718, __UNIQUE_ID_y__719) == 0 {
unsafe { __compiletime_assert_720() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__718 > __UNIQUE_ID_y__719 { __UNIQUE_ID_x__718 } else { __UNIQUE_ID_y__719 };
len = __v0;
index = index.wrapping_add(1);
}
let mut __v5: *mut ();
let mut __v6: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(len as u64, 3072u32, unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(len as u64, 3072u32, unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
}
__v6 = _res;
__v5 = __v6;
let buf: *mut u8 = __v5 as *mut u8;
if buf.is_null() {
return 1;
}
// assume the worst
index = 0;
while index < (((unsafe { (*udev).descriptor.bNumConfigurations }) as i32) as u32) {
old_length = (unsafe { (*unsafe { (*udev).config }.wrapping_offset(index as isize)).desc.wTotalLength }) as u32;
length = unsafe { usb_get_descriptor(udev, 2u8, index as u8, buf as *mut (), old_length as i32) };
if (length as u32) != old_length {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"config index %d, error %d\n".as_ptr() as *const u8, index, length) };
}
changed = 1;
break;
}
if (unsafe { memcmp(buf as *const (), *unsafe { (*udev).rawdescriptors }.wrapping_offset(index as isize) as *const (), old_length as u64) }) != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"config index %d changed (#%d)\n".as_ptr() as *const u8, index, (*(buf as *mut usb_config_descriptor)).bConfigurationValue as i32) };
}
changed = 1;
break;
}
index = index.wrapping_add(1);
}
if changed == 0 && serial_len != 0 {
length = unsafe { usb_string(udev, (*udev).descriptor.iSerialNumber as i32, buf, serial_len as u64) };
if (length.wrapping_add(1) as u32) != serial_len {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"serial string error %d\n".as_ptr() as *const u8, length) };
}
changed = 1;
} else {
if (unsafe { memcmp(buf as *const (), (*udev).serial as *const (), length as u64) }) != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"serial string changed\n".as_ptr() as *const u8) };
}
changed = 1;
}
}
}
unsafe { kfree(buf as *const ()) };
changed
}

fn hub_port_connect(hub: *mut usb_hub, port1: i32, mut portstatus: u16, portchange: u16) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut devstat: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u16 = core::ptr::addr_of_mut!(devstat) as *mut u16;
// maybe switch power back on (e.g. root hub was reset)
// but only if the port isn't owned by someone else.
let mut __v4: bool = unsafe { core::mem::zeroed() };
// Return now if debouncing failed or nothing is connected or
// the device was "removed".
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut retry_locked: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut bool = core::ptr::addr_of_mut!(retry_locked) as *mut bool;
let mut udev: *mut usb_device = unsafe { core::mem::zeroed() };
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
let mut hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
let mut hdev: *mut usb_device = unsafe { core::mem::zeroed() };
let mut unit_load: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u32 = core::ptr::addr_of_mut!(unit_load) as *mut u32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut status: i32 = unsafe { core::mem::zeroed() };
unsafe {
{
status = 19i32.wrapping_neg();
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
hdev = (*hub).hdev;
hcd = bus_to_hcd((*hdev).bus);
port_dev = *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize);
udev = (*port_dev).child;
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
// Disconnect any existing devices under this port
if !udev.is_null() {
if !unsafe { (*hcd).usb_phy }.is_null() && unsafe { (*hdev).parent }.is_null() {
usb_phy_notify_disconnect((*hcd).usb_phy, (*udev).speed);
}
usb_disconnect(core::ptr::addr_of_mut!((*port_dev).child));
}
}
// We can forget about a "removed" device when there's a physical
// disconnect or the connect status changes.
if (portstatus as i32) & 1 == 0 || (portchange as i32) & 1 != 0 {
clear_bit(port1 as i64, ((*hub).removed_bits.as_mut_ptr() as *mut u64) as *mut u64);
}
if (portchange as i32) & (1 | 2) != 0 {
status = hub_port_debounce_be_stable(hub, port1);
if status < 0 {
if status != 19i32.wrapping_neg() && port1 != unreliable_port {
__v8 = printk_ratelimit!() != 0;
} else {
__v8 = false;
}
if __v8 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"connect-debounce failed\n".as_ptr() as *const u8);
}
let __v6: u16 = ((portstatus as i32) & !1) as u16;
portstatus = __v6;
unreliable_port = port1;
} else {
portstatus = status as u16;
}
}
if (portstatus as i32) & 1 == 0 {
__v5 = true;
} else {
__v5 = test_bit!(port1, (*hub).removed_bits.as_mut_ptr() as *mut u64) != 0;
}
'__join_3: {
if __v5 {
if hub_is_port_power_switchable(hub) {
__v4 = usb_port_is_power_on(hub, portstatus as u32) == 0;
} else {
__v4 = false;
}
if __v4 && unsafe { (*port_dev).port_owner }.is_null() {
set_port_feature(hdev, port1, 8i32);
}
if (portstatus as i32) & 2 == 0 {
return;
}
} else {
if hub_is_superspeed((*hub).hdev) != 0 {
*__slate_slot_14 = 150;
} else {
*__slate_slot_14 = 100;
}
status = 0;
*__slate_slot_15 = 0;
'__join_10: {
'__join_51: {
'__join_49: {
'__loop_11: loop {
if *__slate_slot_15 < 4 {
if hub_port_stop_enumerate(hub, port1, *__slate_slot_15) {
break '__join_51;
} else {
usb_lock_port(port_dev);
mutex_lock((*hcd).address0_mutex);
*__slate_slot_9 = true;
// reallocate for each attempt, since references
// to the previous one can escape in various ways
udev = usb_alloc_dev(hdev, (*hdev).bus, port1 as u32);
if udev.is_null() {
break '__join_49;
} else {
usb_set_device_state(udev, 2u32);
(*udev).bus_mA = (*hub).mA_per_port as u16;
(*udev).level = ((*hdev).level as i32).wrapping_add(1) as u8;
// Devices connected to SuperSpeed hubs are USB 3.0 or later
if hub_is_superspeed((*hub).hdev) != 0 {
(*udev).speed = 5;
} else {
(*udev).speed = 0;
}
choose_devnum(udev);
if (*udev).devnum <= 0 {
status = 107i32.wrapping_neg(); // Don't retry
} else {
// reset (non-USB 3.0 devices) and get descriptor
status = hub_port_init(hub, udev, port1, *__slate_slot_15, core::ptr::null_mut::<usb_device_descriptor>());
if status >= 0 {
mutex_unlock((*hcd).address0_mutex);
usb_unlock_port(port_dev);
*__slate_slot_9 = false;
if ((*udev).quirks as u64) & BIT!(6i32) != 0 {
msleep(2000u32);
}
'__join_19: {
// consecutive bus-powered hubs aren't reliable; they can
// violate the voltage drop budget.  if the new child has
// a "powered" LED, users should notice we didn't enable it
// (without reading syslog), even without per-port LEDs
// on the parent.
if ((*udev).descriptor.bDeviceClass as i32) == 9 && (((*udev).bus_mA as i32) as u32) <= *__slate_slot_14 {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
status = usb_get_std_status(udev, 0i32, 0i32, core::ptr::addr_of_mut!(*__slate_slot_3) as *mut ());
if status != 0 {
break '__join_19;
} else {
if (*__slate_slot_3 as i32) & 1i32 == 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"can't connect bus-powered hub to this port\n".as_ptr() as *const u8);
if ((*hub).__slate_bits_0.__get_has_indicators() as i32) != 0 {
*((*hub).indicator.as_mut_ptr() as *mut u8).wrapping_offset(port1.wrapping_sub(1) as isize) = 4;
queue_delayed_work(system_power_efficient_wq, core::ptr::addr_of_mut!((*hub).leds), 0u64);
}
status = 107i32.wrapping_neg(); // Don't retry
break '__join_19;
}
}
}
// check for devices running slower than they could
if ((*udev).descriptor.bcdUSB as i32) >= 512 && ((*udev).speed as u32) == 2 && highspeed_hubs != 0 {
check_highspeed(hub, udev, port1);
}
// Store the parent's children[] pointer.  At this point
// udev becomes globally accessible, although presumably
// no one will look at it until hdev is unlocked.
status = 0;
mutex_lock(core::ptr::addr_of_mut!(usb_port_peer_mutex));
// We mustn't add new devices if the parent hub has
// been disconnected; we would race with the
// recursively_mark_NOTATTACHED() routine.
spin_lock_irq(core::ptr::addr_of_mut!(device_state_lock));
if ((*hdev).state as u32) == 0 {
status = 107i32.wrapping_neg();
} else {
(*port_dev).child = udev;
}
spin_unlock_irq(core::ptr::addr_of_mut!(device_state_lock));
mutex_unlock(core::ptr::addr_of_mut!(usb_port_peer_mutex));
// Run it through the hoops (find a driver, etc)
if status == 0 {
status = usb_new_device(udev);
if status != 0 {
mutex_lock(core::ptr::addr_of_mut!(usb_port_peer_mutex));
spin_lock_irq(core::ptr::addr_of_mut!(device_state_lock));
(*port_dev).child = core::ptr::null_mut::<usb_device>();
spin_unlock_irq(core::ptr::addr_of_mut!(device_state_lock));
mutex_unlock(core::ptr::addr_of_mut!(usb_port_peer_mutex));
} else {
if !unsafe { (*hcd).usb_phy }.is_null() && unsafe { (*hdev).parent }.is_null() {
usb_phy_notify_connect((*hcd).usb_phy, (*udev).speed);
}
}
}
if status == 0 {
break '__loop_11;
}
}
hub_port_disable(hub, port1, 1i32);
}
}
usb_ep0_reinit(udev);
release_devnum(udev);
hub_free_dev(udev);
if *__slate_slot_9 {
mutex_unlock((*hcd).address0_mutex);
usb_unlock_port(port_dev);
}
usb_put_dev(udev);
if status == 107i32.wrapping_neg() || status == 524i32.wrapping_neg() {
break '__join_10;
} else {
// When halfway through our retry count, power-cycle the port
if *__slate_slot_15 == 1i32 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"attempt power cycle\n".as_ptr() as *const u8);
usb_hub_set_port_power(hdev, hub, port1, false);
msleep(2u32.wrapping_mul(hub_power_on_good_delay(hub)));
usb_hub_set_port_power(hdev, hub, port1, true);
msleep(hub_power_on_good_delay(hub));
}
let __v2: i32 = *__slate_slot_15;
*__slate_slot_15 = __v2.wrapping_add(1);
}
}
}
} else {
break '__join_10;
}
}
status = hub_power_remaining(hub) as i32;
return;
}
while false {
}
_dev_err(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"couldn't allocate usb_device\n".as_ptr() as *const u8);
mutex_unlock((*hcd).address0_mutex);
usb_unlock_port(port_dev);
break '__join_3;
}
status = 19i32.wrapping_neg();
}
if !unsafe { (*(*hub).hdev).parent }.is_null() || (*(*hcd).driver).port_handed_over == None {
__v0 = true;
} else {
__v0 = unsafe { (*(*hcd).driver).port_handed_over }.unwrap()(hcd, port1) == 0;
}
if __v0 {
if status != 107i32.wrapping_neg() && status != 19i32.wrapping_neg() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"unable to enumerate USB device\n".as_ptr() as *const u8);
}
}
}
}
hub_port_disable(hub, port1, 1i32);
if (*(*hcd).driver).relinquish_port != None && unsafe { (*(*hub).hdev).parent }.is_null() {
if status != 107i32.wrapping_neg() && status != 19i32.wrapping_neg() {
unsafe { (*(*hcd).driver).relinquish_port }.unwrap()(hcd, port1);
}
}
}
}

static mut unreliable_port: i32 = 1i32.wrapping_neg();

/// Handle physical or logical connection change events.
/// This routine is called when:
/// 	a port connection-change occurs;
/// 	a port enable-change occurs (often caused by EMI);
/// 	usb_reset_and_verify_device() encounters changed descriptors (as from
/// 		a firmware download)
/// caller already locked the hub
fn hub_port_connect_change(hub: *mut usb_hub, port1: i32, portstatus: u16, portchange: u16) {
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize) };
let udev: *mut usb_device = unsafe { (*port_dev).child };
let mut status: i32 = 19i32.wrapping_neg();
if ((unsafe { (*hub).__slate_bits_0.__get_has_indicators() }) as i32) != 0 {
set_port_led(hub, port1, 0i32);
unsafe {
*((*hub).indicator.as_mut_ptr() as *mut u8).wrapping_offset(port1.wrapping_sub(1) as isize) = 0;
}
}
// Try to resuscitate an existing device
if (portstatus as i32) & 1 != 0 && !udev.is_null() && ((unsafe { (*udev).state }) as u32) != 0 {
if (portstatus as i32) & 2 != 0 {
// USB-3 connections are initialized automatically by
// the hostcontroller hardware. Therefore check for
// changed device descriptors before resuscitating the
// device.
let descr: *mut usb_device_descriptor = unsafe { usb_get_device_descriptor(udev) };
if !IS_ERR(descr as *const ()) {
if descriptors_changed(udev, descr, unsafe { (*udev).bos }) == 0 {
status = 0; // Nothing to do
}
unsafe { kfree(descr as *const ()) };
}
} else {
if ((unsafe { (*udev).state }) as u32) == 8 && ((unsafe { (*udev).__slate_bits_0.__get_persist_enabled() }) as i32) != 0 {
// For a suspended device, treat this as a
// remote wakeup event.
usb_unlock_port(port_dev);
status = usb_remote_wakeup(udev);
usb_lock_port(port_dev);
}
// Don't resuscitate
}
}
clear_bit(port1 as i64, (unsafe { (*hub).change_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
// successfully revalidated the connection
if status == 0 {
return;
}
usb_unlock_port(port_dev);
hub_port_connect(hub, port1, portstatus, portchange);
usb_lock_port(port_dev);
}

/// Handle notifying userspace about hub over-current events
fn port_over_current_notify(port_dev: *mut usb_port) {
let mut port_dev_path: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut u8 = core::ptr::addr_of_mut!(port_dev_path) as *mut *mut u8;
let mut hub_dev: core::mem::MaybeUninit<*mut device> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut device = core::ptr::addr_of_mut!(hub_dev) as *mut *mut device;
let mut envp: core::mem::MaybeUninit<__SlateAlign16<[*mut u8; 3]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut [*mut u8; 3] = core::ptr::addr_of_mut!(envp) as *mut [*mut u8; 3];
unsafe {
core::ptr::write(__slate_slot_2, [core::ptr::null_mut::<u8>(), core::ptr::null_mut::<u8>(), core::ptr::null_mut::<u8>()]);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
sysfs_notify(core::ptr::addr_of_mut!((*port_dev).dev.kobj), core::ptr::null::<u8>(), c"over_current_count".as_ptr() as *const u8);
*__slate_slot_1 = (*port_dev).dev.parent;
if (*__slate_slot_1).is_null() {
return;
} else {
*__slate_slot_0 = kobject_get_path(core::ptr::addr_of_mut!((*port_dev).dev.kobj) as *const kobject, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if (*__slate_slot_0).is_null() {
return;
} else {
(*__slate_slot_2)[0] = kasprintf(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), c"OVER_CURRENT_PORT=%s".as_ptr() as *const u8, *__slate_slot_0);
if !(*__slate_slot_2)[0].is_null() {
(*__slate_slot_2)[1] = kasprintf(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), c"OVER_CURRENT_COUNT=%u".as_ptr() as *const u8, (*port_dev).over_current_count);
if !(*__slate_slot_2)[1].is_null() {
kobject_uevent_env(core::ptr::addr_of_mut!((*(*__slate_slot_1)).kobj), 2u32, (*__slate_slot_2).as_mut_ptr() as *mut *mut u8);
}
}
kfree((*__slate_slot_2)[1] as *const ());
kfree((*__slate_slot_2)[0] as *const ());
kfree(*__slate_slot_0 as *const ());
}
}
}
}

fn port_event(hub: *mut usb_hub, port1: i32) {
let mut connect_change: i32 = 0;
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize) };
let udev: *mut usb_device = unsafe { (*port_dev).child };
let hdev: *mut usb_device = unsafe { (*hub).hdev };
let mut portstatus: u16 = 0;
let mut portchange: u16 = 0;
let mut i: i32 = 0;
let mut err: i32 = 0;
connect_change = test_bit!(port1, unsafe { (*hub).change_bits.as_mut_ptr() as *mut u64 });
clear_bit(port1 as i64, (unsafe { (*hub).event_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
clear_bit(port1 as i64, (unsafe { (*hub).wakeup_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
if usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(portstatus), core::ptr::addr_of_mut!(portchange)) < 0 {
return;
}
if (portchange as i32) & 1 != 0 {
usb_clear_port_feature(hdev, port1, 16i32);
connect_change = 1;
}
if (portchange as i32) & 2 != 0 {
usb_clear_port_feature(hdev, port1, 17i32);
// EM interference sometimes causes badly shielded USB devices
// to be shutdown by the hub, this hack enables them again.
// Works at least with mouse driver.
if (portstatus as i32) & 2 == 0 && connect_change == 0 && !udev.is_null() {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"disabled by hub (EMI?), re-enabling...\n".as_ptr() as *const u8) };
connect_change = 1;
}
}
if (portchange as i32) & 8 != 0 {
let mut status: u16 = 0;
let mut unused: u16 = 0;
let __v3: u32 = unsafe { (*port_dev).over_current_count };
unsafe {
(*port_dev).over_current_count = __v3.wrapping_add(1);
}
port_over_current_notify(port_dev);
usb_clear_port_feature(hdev, port1, 19i32);
unsafe { msleep(100u32) }; // Cool down
hub_power_on(hub, true);
usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(status), core::ptr::addr_of_mut!(unused));
if (status as i32) & 8 != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"over-current condition\n".as_ptr() as *const u8) };
}
}
if (portchange as i32) & 16 != 0 {
usb_clear_port_feature(hdev, port1, 20i32);
}
let __v6: bool = (portchange as i32) & 32 != 0 && hub_is_superspeed(hdev) != 0;
if __v6 {
usb_clear_port_feature(hdev, port1, 29i32);
}
if (portchange as i32) & 64 != 0 {
usb_clear_port_feature(hdev, port1, 25i32);
}
if (portchange as i32) & 128 != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"config error\n".as_ptr() as *const u8) };
usb_clear_port_feature(hdev, port1, 26i32);
}
// skip port actions that require the port to be powered on
if !pm_runtime_active(unsafe { core::ptr::addr_of_mut!((*port_dev).dev) }) {
return;
}
// skip port actions if ignore_event and early_stop are true
if ((unsafe { (*port_dev).__slate_bits_0.__get_ignore_event() }) as i32) != 0 && ((unsafe { (*port_dev).__slate_bits_0.__get_early_stop() }) as i32) != 0 {
return;
}
if hub_handle_remote_wakeup(hub, port1 as u32, portstatus, portchange) != 0 {
connect_change = 1;
}
// Avoid trying to recover a USB3 SS.Inactive port with a warm reset if
// the device was disconnected. A 12ms disconnect detect timer in
// SS.Inactive state transitions the port to RxDetect automatically.
// SS.Inactive link error state is common during device disconnect.
'__slate_break_10: while hub_port_warm_reset_required(hub, port1, portstatus) {
'__slate_continue_10: {
let __v8: i32 = i;
i = __v8.wrapping_add(1);
if __v8 < 5 && !udev.is_null() {
let mut unused: u16 = 0;
unsafe { msleep(20u32) };
usb_hub_port_status(hub, port1, core::ptr::addr_of_mut!(portstatus), core::ptr::addr_of_mut!(unused));
break '__slate_continue_10;
} else {
if udev.is_null() || (portstatus as i32) & 1 == 0 || ((unsafe { (*udev).state }) as u32) == 0 {
err = hub_port_reset(hub, port1, core::ptr::null_mut::<usb_device>(), 50u32, true);
if udev.is_null() && err == 107i32.wrapping_neg() {
connect_change = 0;
} else {
if err < 0 {
hub_port_disable(hub, port1, 1i32);
}
}
} else {
usb_unlock_port(port_dev);
usb_lock_device!(udev);
usb_reset_device(udev);
usb_unlock_device!(udev);
usb_lock_port(port_dev);
connect_change = 0;
}
}
break '__slate_break_10;
}
}
if connect_change != 0 {
hub_port_connect_change(hub, port1, portstatus, portchange);
}
}

#[unsafe(link_section = ".text.slate_distinct.hub.hub_event")]
extern "C" fn hub_event(work: *mut work_struct) {
let mut unused: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u16 = core::ptr::addr_of_mut!(unused) as *mut u16;
let mut status: u16 = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut port_dev: *mut usb_port = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut hubchange: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u16 = core::ptr::addr_of_mut!(hubchange) as *mut u16;
let mut hubstatus: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u16 = core::ptr::addr_of_mut!(hubstatus) as *mut u16;
let mut hub_dev: core::mem::MaybeUninit<*mut device> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut device = core::ptr::addr_of_mut!(hub_dev) as *mut *mut device;
let mut hub: core::mem::MaybeUninit<*mut usb_hub> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut usb_hub = core::ptr::addr_of_mut!(hub) as *mut *mut usb_hub;
let mut intf: core::mem::MaybeUninit<*mut usb_interface> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut usb_interface = core::ptr::addr_of_mut!(intf) as *mut *mut usb_interface;
let mut hdev: core::mem::MaybeUninit<*mut usb_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut usb_device = core::ptr::addr_of_mut!(hdev) as *mut *mut usb_device;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_6 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_7 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
*__slate_slot_12 = container_of__62!(work, usb_hub);
*__slate_slot_14 = (*(*__slate_slot_12)).hdev;
*__slate_slot_11 = (*(*__slate_slot_12)).intfdev;
*__slate_slot_13 = to_usb_interface!(*__slate_slot_11);
kcov_remote_start_usb((*(*(*__slate_slot_14)).bus).busnum as u64);
// NOTE: expects max 15 ports...
// Lock the device, then check to see if we were
// disconnected while waiting for the lock to succeed.
usb_lock_device!(*__slate_slot_14);
if unlikely__2!((*(*__slate_slot_12)).__slate_bits_0.__get_disconnected() as i32) == 0 {
// If the hub has died, clean up after it
if ((*(*__slate_slot_14)).state as u32) == 0 {
(*(*__slate_slot_12)).error = 19i32.wrapping_neg();
hub_quiesce(*__slate_slot_12, 0u32);
} else {
// Autoresume
*__slate_slot_7 = usb_autopm_get_interface(*__slate_slot_13);
if *__slate_slot_7 == 0 {
'__join_1: {
// If this is an inactive hub, do nothing
if ((*(*__slate_slot_12)).__slate_bits_0.__get_quiescing() as i32) == 0 {
if (*(*__slate_slot_12)).error != 0 {
*__slate_slot_7 = usb_reset_device(*__slate_slot_14);
if *__slate_slot_7 != 0 {
break '__join_1;
} else {
(*(*__slate_slot_12)).nerrors = 0;
(*(*__slate_slot_12)).error = 0;
}
}
// deal with port status changes
*__slate_slot_8 = 1;
while *__slate_slot_8 <= (*(*__slate_slot_14)).maxchild {
port_dev = *unsafe { (*(*__slate_slot_12)).ports }.wrapping_offset((*__slate_slot_8).wrapping_sub(1) as isize);
if test_bit!(*__slate_slot_8, (*(*__slate_slot_12)).event_bits.as_mut_ptr() as *mut u64) != 0 {
__v5 = true;
} else {
__v5 = test_bit!(*__slate_slot_8, (*(*__slate_slot_12)).change_bits.as_mut_ptr() as *mut u64) != 0;
}
if __v5 {
__v4 = true;
} else {
__v4 = test_bit!(*__slate_slot_8, (*(*__slate_slot_12)).wakeup_bits.as_mut_ptr() as *mut u64) != 0;
}
if __v4 {
// The get_noresume and barrier ensure that if
// the port was in the process of resuming, we
// flush that work and keep the port active for
// the duration of the port_event().  However,
// if the port is runtime pm suspended
// (powered-off), we leave it in that state, run
// an abbreviated port_event(), and move on.
pm_runtime_get_noresume(core::ptr::addr_of_mut!((*port_dev).dev));
pm_runtime_barrier(core::ptr::addr_of_mut!((*port_dev).dev));
usb_lock_port(port_dev);
port_event(*__slate_slot_12, *__slate_slot_8);
usb_unlock_port(port_dev);
pm_runtime_put_sync(core::ptr::addr_of_mut!((*port_dev).dev));
}
let __v3: i32 = *__slate_slot_8;
*__slate_slot_8 = __v3.wrapping_add(1);
}
// deal with hub status changes
if (test_and_clear_bit(0i64, ((*(*__slate_slot_12)).event_bits.as_mut_ptr() as *mut u64) as *mut u64) as i32) != 0 {
if hub_hub_status(*__slate_slot_12, &mut (*core::ptr::addr_of_mut!(*__slate_slot_10)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_9))) < 0 {
while false {
}
_dev_err(*__slate_slot_11 as *const device, c"get_hub_status failed\n".as_ptr() as *const u8);
} else {
if (*__slate_slot_9 as i32) & 1 != 0 {
clear_hub_feature(*__slate_slot_14, 0i32);
if (*__slate_slot_10 as i32) & 1 != 0 {
(*(*__slate_slot_12)).__slate_bits_0.__set_limited_power(1u32);
} else {
(*(*__slate_slot_12)).__slate_bits_0.__set_limited_power(0u32);
}
// FIXME: Is this always true?
}
if (*__slate_slot_9 as i32) & 2 != 0 {
status = 0;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
clear_hub_feature(*__slate_slot_14, 1i32);
msleep(500u32); // Cool down
hub_power_on(*__slate_slot_12, true);
hub_hub_status(*__slate_slot_12, &mut (*core::ptr::addr_of_mut!(status)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_0)));
if (status as i32) & 2 != 0 {
while false {
}
_dev_err(*__slate_slot_11 as *const device, c"over-current condition\n".as_ptr() as *const u8);
}
}
}
}
// do nothing
}
}
usb_autopm_put_interface_no_suspend(*__slate_slot_13); // Balance the usb_autopm_get_interface() above
}
}
}
usb_unlock_device!(*__slate_slot_14);
// Balance the stuff in kick_hub_wq() and allow autosuspend
usb_autopm_put_interface(*__slate_slot_13);
hub_put(*__slate_slot_12);
kcov_remote_stop();
}
}

/// Terminating entry
static hub_id_table: __SlateAlign16<[usb_device_id; 11]> = __SlateAlign16([{
let mut __t0: usb_device_id = unsafe { core::mem::zeroed() };
__t0.match_flags = 131u16;
__t0.idVendor = 1060u16;
__t0.idProduct = 21812u16;
__t0.bInterfaceClass = 9u8;
__t0.driver_info = 2u64;
__t0
}, {
let mut __t1: usb_device_id = unsafe { core::mem::zeroed() };
__t1.match_flags = 3u16;
__t1.idVendor = 1204u16;
__t1.idProduct = 25968u16;
__t1.driver_info = 2u64;
__t1
}, {
let mut __t2: usb_device_id = unsafe { core::mem::zeroed() };
__t2.match_flags = 129u16;
__t2.idVendor = 1507u16;
__t2.bInterfaceClass = 9u8;
__t2.driver_info = 1u64;
__t2
}, {
let mut __t3: usb_device_id = unsafe { core::mem::zeroed() };
__t3.match_flags = 3u16;
__t3.idVendor = 1105u16;
__t3.idProduct = 33090u16;
__t3.driver_info = 2u64;
__t3
}, {
let mut __t4: usb_device_id = unsafe { core::mem::zeroed() };
__t4.match_flags = 3u16;
__t4.idVendor = 1105u16;
__t4.idProduct = 33088u16;
__t4.driver_info = 2u64;
__t4
}, {
let mut __t5: usb_device_id = unsafe { core::mem::zeroed() };
__t5.match_flags = 3u16;
__t5.idVendor = 1060u16;
__t5.idProduct = 18707u16;
__t5.driver_info = 4u64;
__t5
}, {
let mut __t6: usb_device_id = unsafe { core::mem::zeroed() };
__t6.match_flags = 3u16;
__t6.idVendor = 1060u16;
__t6.idProduct = 18708u16;
__t6.driver_info = 4u64;
__t6
}, {
let mut __t7: usb_device_id = unsafe { core::mem::zeroed() };
__t7.match_flags = 3u16;
__t7.idVendor = 1060u16;
__t7.idProduct = 18709u16;
__t7.driver_info = 4u64;
__t7
}, {
let mut __t8: usb_device_id = unsafe { core::mem::zeroed() };
__t8.match_flags = 16u16;
__t8.bDeviceClass = 9u8;
__t8
}, {
let mut __t9: usb_device_id = unsafe { core::mem::zeroed() };
__t9.match_flags = 128u16;
__t9.bInterfaceClass = 9u8;
__t9
}, {
let mut __t10: usb_device_id = unsafe { core::mem::zeroed() };
__t10
}]);

static mut hub_driver: usb_driver = {
let mut __t0: usb_driver = unsafe { core::mem::zeroed() };
__t0.name = (b"hub\0".as_ptr() as *mut u8) as *const u8;
__t0.probe = Some(hub_probe);
__t0.disconnect = Some(hub_disconnect);
__t0.unlocked_ioctl = Some(hub_ioctl);
__t0.suspend = Some(hub_suspend);
__t0.resume = Some(hub_resume);
__t0.reset_resume = Some(hub_reset_resume);
__t0.pre_reset = Some(hub_pre_reset);
__t0.post_reset = Some(hub_post_reset);
__t0.id_table = unsafe { core::ptr::addr_of!(hub_id_table.0) as *const usb_device_id };
__t0.__slate_bits_0.__set_supports_autosuspend(1u32);
__t0
};

#[unsafe(no_mangle)]
extern "C" fn usb_hub_init() -> i32 {
if usb_register!(unsafe { core::ptr::addr_of_mut!(hub_driver) }) < 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013%s: can't register hub driver\n".as_ptr() as *const u8, usbcore_name) };
return 1i32.wrapping_neg();
}
// The workqueue needs to be freezable to avoid interfering with
// USB-PERSIST port handover. Otherwise it might see that a full-speed
// device was gone before the EHCI controller had handed its port
// over to the companion full-speed controller.
let mut __v2: *mut workqueue_struct;
let mut __v3: *mut workqueue_struct;
let mut _res: *mut workqueue_struct = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = unsafe { alloc_workqueue_noprof(c"usb_hub_wq".as_ptr() as *const u8, 260u32, 0i32) };
alloc_tag_restore!();
} else {
_res = unsafe { alloc_workqueue_noprof(c"usb_hub_wq".as_ptr() as *const u8, 260u32, 0i32) };
}
__v3 = _res;
__v2 = __v3;
unsafe {
hub_wq = __v2;
}
if !unsafe { hub_wq }.is_null() {
return 0;
}
// Fall through if kernel_thread failed
unsafe { usb_deregister(core::ptr::addr_of_mut!(hub_driver)) };
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013%s: can't allocate workqueue for usb hub\n".as_ptr() as *const u8, usbcore_name) };
1i32.wrapping_neg()
}

/// usb_hub_cleanup()
#[unsafe(no_mangle)]
extern "C" fn usb_hub_cleanup() {
unsafe { destroy_workqueue(hub_wq) };
// Hub resources are freed for us by usb_deregister. It calls
// usb_driver_purge on every device which in turn calls that
// devices disconnect function if it is using this driver.
// The hub_disconnect function takes care of releasing the
// individual hub resources. -greg
unsafe { usb_deregister(core::ptr::addr_of_mut!(hub_driver)) };
}

/// hub_hc_release_resources - clear resources used by host controller
/// @udev: pointer to device being released
///
/// Context: task context, might sleep
///
/// Function releases the host controller resources in correct order before
/// making any operation on resuming usb device. The host controller resources
/// allocated for devices in tree should be released starting from the last
/// usb device in tree toward the root hub. This function is used only during
/// resuming device when usb device require reinitialization – that is, when
/// flag udev->reset_resume is set.
///
/// This call is synchronous, and may not be used in an interrupt context.
fn hub_hc_release_resources(udev: *mut usb_device) {
let hub: *mut usb_hub = usb_hub_to_struct_hub(udev);
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*udev).bus });
// Release up resources for all children before this device
let mut i: i32 = 0;
while i < unsafe { (*udev).maxchild } {
if !unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child }.is_null() {
hub_hc_release_resources(unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(i as isize))).child });
}
i = i.wrapping_add(1);
}
if (unsafe { (*(*hcd).driver).reset_device }) != None {
unsafe { unsafe { (*(*hcd).driver).reset_device }.unwrap()(hcd, udev) };
}
}

/// usb_reset_and_verify_device - perform a USB port reset to reinitialize a device
/// @udev: device to reset (not in SUSPENDED or NOTATTACHED state)
///
/// WARNING - don't use this routine to reset a composite device
/// (one with multiple interfaces owned by separate drivers)!
/// Use usb_reset_device() instead.
///
/// Do a port reset, reassign the device's address, and establish its
/// former operating configuration.  If the reset fails, or the device's
/// descriptors change from their values before the reset, or the original
/// configuration and altsettings cannot be restored, a flag will be set
/// telling hub_wq to pretend the device has been disconnected and then
/// re-connected.  All drivers will be unbound, and the device will be
/// re-enumerated and probed all over again.
///
/// Return: 0 if the reset succeeded, -ENODEV if the device has been
/// flagged for logical disconnection, or some other negative error code
/// if the reset wasn't even attempted.
///
/// Note:
/// The caller must own the device lock and the port lock, the latter is
/// taken by usb_reset_device().  For example, it's safe to use
/// usb_reset_device() from a driver probe() routine after downloading
/// new firmware.  For calls that might not occur during probe(), drivers
/// should lock the device using usb_lock_device_for_reset().
///
/// Locking exception: This routine may also be called from within an
/// autoresume handler.  Such usage won't conflict with other tasks
/// holding the device lock because these tasks should always call
/// usb_autopm_resume_device(), thereby preventing any unwanted
/// autoresume.  The autoresume handler is expected to have already
/// acquired the port lock before calling this routine.
fn usb_reset_and_verify_device(udev: *mut usb_device) -> i32 {
let mut desc: core::mem::MaybeUninit<*mut usb_interface_descriptor> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut usb_interface_descriptor = core::ptr::addr_of_mut!(desc) as *mut *mut usb_interface_descriptor;
let mut port1: i32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut j: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(j) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut bos: core::mem::MaybeUninit<*mut usb_host_bos> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut usb_host_bos = core::ptr::addr_of_mut!(bos) as *mut *mut usb_host_bos;
let mut intf: core::mem::MaybeUninit<*mut usb_interface> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut usb_interface = core::ptr::addr_of_mut!(intf) as *mut *mut usb_interface;
let mut descriptor: core::mem::MaybeUninit<usb_device_descriptor> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut usb_device_descriptor = core::ptr::addr_of_mut!(descriptor) as *mut usb_device_descriptor;
let mut hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
let mut parent_hub: core::mem::MaybeUninit<*mut usb_hub> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut usb_hub = core::ptr::addr_of_mut!(parent_hub) as *mut *mut usb_hub;
let mut parent_hdev: *mut usb_device = unsafe { core::mem::zeroed() };
unsafe {
parent_hdev = (*udev).parent;
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
hcd = bus_to_hcd((*udev).bus);
let __slate_auto_bytes_2 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 18usize);
let __slate_auto_bytes_3 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
ret = 0;
port1 = (*udev).portnum as i32;
if ((*udev).state as u32) == 0 || ((*udev).state as u32) == 8 {
return 22i32.wrapping_neg();
} else {
if parent_hdev.is_null() {
return 21i32.wrapping_neg();
} else {
*__slate_slot_18 = usb_hub_to_struct_hub(parent_hdev);
// Disable USB2 hardware LPM.
// It will be re-enabled by the enumeration process.
usb_disable_usb2_hardware_lpm(udev);
*__slate_slot_14 = (*udev).bos;
(*udev).bos = core::ptr::null_mut::<usb_host_bos>();
if ((*udev).__slate_bits_1.__get_reset_resume() as i32) != 0 {
hub_hc_release_resources(udev);
}
mutex_lock((*hcd).address0_mutex);
*__slate_slot_13 = 0;
'__join_29: {
loop {
if *__slate_slot_13 < 4 {
if hub_port_stop_enumerate(*__slate_slot_18, port1, *__slate_slot_13) {
break;
} else {
// ep0 maxpacket size may change; let the HCD know about it.
// Other endpoints will be handled by re-enumeration.
usb_ep0_reinit(udev);
ret = hub_port_init(*__slate_slot_18, udev, port1, *__slate_slot_13, core::ptr::addr_of_mut!(*__slate_slot_16));
if ret >= 0 || ret == 107i32.wrapping_neg() || ret == 19i32.wrapping_neg() {
break '__join_29;
} else {
let __v9: i32 = *__slate_slot_13;
*__slate_slot_13 = __v9.wrapping_add(1);
}
}
} else {
break '__join_29;
}
}
ret = 19i32.wrapping_neg();
}
'__join_0: {
mutex_unlock((*hcd).address0_mutex);
if ret >= 0 {
// Device might have changed firmware (DFU or similar)
if descriptors_changed(udev, core::ptr::addr_of_mut!(*__slate_slot_16), *__slate_slot_14) != 0 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"device firmware changed\n".as_ptr() as *const u8);
} else {
'__join_1: {
// Restore the device's previous configuration
if !unsafe { (*udev).actconfig }.is_null() {
// Some devices can't handle setting default altsetting 0 with a
// Set-Interface request. Disable host-side endpoints of those
// interfaces here. Enable and reset them back after host has set
// its internal endpoint structures during usb_hcd_alloc_bandwith()
*__slate_slot_13 = 0;
while *__slate_slot_13 < ((*(*udev).actconfig).desc.bNumInterfaces as i32) {
*__slate_slot_15 = *((*(*udev).actconfig).interface.as_mut_ptr() as *mut *mut usb_interface).wrapping_offset(*__slate_slot_13 as isize);
if ((*(*(*__slate_slot_15)).cur_altsetting).desc.bAlternateSetting as i32) == 0 {
usb_disable_interface(udev, *__slate_slot_15, true);
}
let __v7: i32 = *__slate_slot_13;
*__slate_slot_13 = __v7.wrapping_add(1);
}
mutex_lock((*hcd).bandwidth_mutex);
ret = usb_hcd_alloc_bandwidth(udev, (*udev).actconfig, core::ptr::null_mut::<usb_host_interface>(), core::ptr::null_mut::<usb_host_interface>());
if ret < 0 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Busted HC?  Not enough HCD resources for old configuration.\n".as_ptr() as *const u8);
mutex_unlock((*hcd).bandwidth_mutex);
break '__join_0;
} else {
ret = usb_control_msg(udev, usb_sndctrlpipe!(udev, 0i32), 9u8, 0u8, (*(*udev).actconfig).desc.bConfigurationValue as u16, 0u16, core::ptr::null_mut::<()>(), 0u16, 5000i32);
if ret < 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"can't restore configuration #%d (error=%d)\n".as_ptr() as *const u8, (*(*udev).actconfig).desc.bConfigurationValue as i32, ret);
mutex_unlock((*hcd).bandwidth_mutex);
break '__join_0;
} else {
mutex_unlock((*hcd).bandwidth_mutex);
usb_set_device_state(udev, 7u32);
// Put interfaces back into the same altsettings as before.
// Don't bother to send the Set-Interface request for interfaces
// that were already in altsetting 0; besides being unnecessary,
// many devices can't handle it.  Instead just reset the host-side
// endpoint state.
*__slate_slot_13 = 0;
loop {
if *__slate_slot_13 < ((*(*udev).actconfig).desc.bNumInterfaces as i32) {
let config: *mut usb_host_config = (*udev).actconfig;
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_15 = *((*config).interface.as_mut_ptr() as *mut *mut usb_interface).wrapping_offset(*__slate_slot_13 as isize);
*__slate_slot_4 = core::ptr::addr_of_mut!((*(*(*__slate_slot_15)).cur_altsetting).desc);
if ((*(*__slate_slot_4)).bAlternateSetting as i32) == 0 {
usb_enable_interface(udev, *__slate_slot_15, true);
ret = 0;
} else {
// Let the bandwidth allocation function know that this
// device has been reset, and it will have to use
// alternate setting 0 as the current alternate setting.
(*(*__slate_slot_15)).__slate_bits_0.__set_resetting_device(1u32);
ret = usb_set_interface(udev, (*(*__slate_slot_4)).bInterfaceNumber as i32, (*(*__slate_slot_4)).bAlternateSetting as i32);
(*(*__slate_slot_15)).__slate_bits_0.__set_resetting_device(0u32);
}
if ret < 0 {
break;
} else {
// Resetting also frees any allocated streams
*__slate_slot_12 = 0;
while *__slate_slot_12 < ((*(*(*__slate_slot_15)).cur_altsetting).desc.bNumEndpoints as i32) {
(*unsafe { (*(*(*__slate_slot_15)).cur_altsetting).endpoint }.wrapping_offset(*__slate_slot_12 as isize)).streams = 0;
let __v3: i32 = *__slate_slot_12;
*__slate_slot_12 = __v3.wrapping_add(1);
}
let __v1: i32 = *__slate_slot_13;
*__slate_slot_13 = __v1.wrapping_add(1);
}
} else {
break '__join_1;
}
}
while false {
}
_dev_err(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"failed to restore interface %d altsetting %d (error=%d)\n".as_ptr() as *const u8, (*(*__slate_slot_4)).bInterfaceNumber as i32, (*(*__slate_slot_4)).bAlternateSetting as i32, ret);
break '__join_0;
}
}
}
}
usb_enable_usb2_hardware_lpm(udev); // Now that the alt settings are re-installed, enable LTM and LPM.
usb_unlocked_enable_lpm(udev);
usb_enable_ltm(udev);
usb_release_bos_descriptor(udev);
(*udev).bos = *__slate_slot_14;
return 0;
}
}
}
usb_release_bos_descriptor(udev);
(*udev).bos = *__slate_slot_14;
hub_port_logical_disconnect(*__slate_slot_18, port1);
return 19i32.wrapping_neg();
}
}
}
}

/// usb_reset_device - warn interface drivers and perform a USB port reset
/// @udev: device to reset (not in NOTATTACHED state)
///
/// Warns all drivers bound to registered interfaces (using their pre_reset
/// method), performs the port reset, and then lets the drivers know that
/// the reset is over (using their post_reset method).
///
/// Return: The same as for usb_reset_and_verify_device().
/// However, if a reset is already in progress (for instance, if a
/// driver doesn't have pre_reset() or post_reset() callbacks, and while
/// being unbound or re-bound during the ongoing reset its disconnect()
/// or probe() routine tries to perform a second, nested reset), the
/// routine returns -EINPROGRESS.
///
/// Note:
/// The caller must own the device lock.  For example, it's safe to use
/// this from a driver probe() routine after downloading new firmware.
/// For calls that might not occur during probe(), drivers should lock
/// the device using usb_lock_device_for_reset().
///
/// If an interface is currently being probed or disconnected, we assume
/// its driver knows how to handle resets.  For all other interfaces,
/// if the driver doesn't have pre_reset and post_reset methods then
/// we attempt to unbind it and rebind afterward.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_reset_device")]
extern "C" fn usb_reset_device(udev: *mut usb_device) -> i32 {
let mut ret: i32 = 0;
let mut i: i32 = 0;
let mut noio_flag: u32 = 0;
let config: *mut usb_host_config = unsafe { (*udev).actconfig };
let hub: *mut usb_hub = usb_hub_to_struct_hub(unsafe { (*udev).parent });
if ((unsafe { (*udev).state }) as u32) == 0 {
return 22i32.wrapping_neg();
}
if unsafe { (*udev).parent }.is_null() {
// this requires hcd-specific logic; see ohci_restart()
return 21i32.wrapping_neg();
}
if ((unsafe { (*udev).__slate_bits_0.__get_reset_in_progress() }) as i32) != 0 {
return 115i32.wrapping_neg();
}
unsafe {
(*udev).__slate_bits_0.__set_reset_in_progress(1u32);
}
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize) };
// Don't allocate memory with GFP_KERNEL in current
// context to avoid possible deadlock if usb mass
// storage interface or usbnet interface(iSCSI case)
// is included in current configuration. The easist
// approach is to do it for every device reset,
// because the device 'memalloc_noio' flag may have
// not been set before reseting the usb device.
noio_flag = memalloc_noio_save();
// Prevent autosuspend during the reset
unsafe { usb_autoresume_device(udev) };
if !config.is_null() {
i = 0;
while i < ((unsafe { (*config).desc.bNumInterfaces }) as i32) {
let cintf: *mut usb_interface = unsafe { *((*config).interface.as_mut_ptr() as *mut *mut usb_interface).wrapping_offset(i as isize) };
let mut unbind: i32 = 0;
if !unsafe { (*cintf).dev.driver }.is_null() {
let drv: *mut usb_driver = to_usb_driver!(unsafe { (*cintf).dev.driver });
if (unsafe { (*drv).pre_reset }) != None && (unsafe { (*drv).post_reset }) != None {
unbind = unsafe { unsafe { (*drv).pre_reset }.unwrap()(cintf) };
} else {
if ((unsafe { (*cintf).condition }) as u32) == 2 {
unbind = 1;
}
}
if unbind != 0 {
unsafe { usb_forced_unbind_intf(cintf) };
}
}
i = i.wrapping_add(1);
}
}
usb_lock_port(port_dev);
ret = usb_reset_and_verify_device(udev);
usb_unlock_port(port_dev);
if !config.is_null() {
i = ((unsafe { (*config).desc.bNumInterfaces }) as i32).wrapping_sub(1);
while i >= 0 {
let cintf: *mut usb_interface = unsafe { *((*config).interface.as_mut_ptr() as *mut *mut usb_interface).wrapping_offset(i as isize) };
let mut rebind: i32 = (unsafe { (*cintf).__slate_bits_0.__get_needs_binding() }) as i32;
if rebind == 0 && !unsafe { (*cintf).dev.driver }.is_null() {
let drv: *mut usb_driver = to_usb_driver!(unsafe { (*cintf).dev.driver });
if (unsafe { (*drv).post_reset }) != None {
rebind = unsafe { unsafe { (*drv).post_reset }.unwrap()(cintf) };
} else {
if ((unsafe { (*cintf).condition }) as u32) == 2 {
rebind = 1;
}
}
if rebind != 0 {
unsafe {
(*cintf).__slate_bits_0.__set_needs_binding(1u32);
}
}
}
i = i.wrapping_sub(1);
}
// If the reset failed, hub_wq will unbind drivers later
if ret == 0 {
unsafe { usb_unbind_and_rebind_marked_interfaces(udev) };
}
}
unsafe { usb_autosuspend_device(udev) };
memalloc_noio_restore(noio_flag);
unsafe {
(*udev).__slate_bits_0.__set_reset_in_progress(0u32);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_reset_device_721: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device) -> i32>, *mut ()>(Some(usb_reset_device)) };

/// usb_queue_reset_device - Reset a USB device from an atomic context
/// @iface: USB interface belonging to the device to reset
///
/// This function can be used to reset a USB device from an atomic
/// context, where usb_reset_device() won't work (as it blocks).
///
/// Doing a reset via this method is functionally equivalent to calling
/// usb_reset_device(), except for the fact that it is delayed to a
/// workqueue. This means that any drivers bound to other interfaces
/// might be unbound, as well as users from usbfs in user space.
///
/// Corner cases:
///
/// - Scheduling two resets at the same time from two different drivers
///   attached to two different interfaces of the same device is
///   possible; depending on how the driver attached to each interface
///   handles ->pre_reset(), the second reset might happen or not.
///
/// - If the reset is delayed so long that the interface is unbound from
///   its driver, the reset will be skipped.
///
/// - This function can be called during .probe().  It can also be called
///   during .disconnect(), but doing so is pointless because the reset
///   will not occur.  If you really want to reset the device during
///   .disconnect(), call usb_reset_device() directly -- but watch out
///   for nested unbinding issues!
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_queue_reset_device")]
extern "C" fn usb_queue_reset_device(iface: *mut usb_interface) {
if schedule_work(unsafe { core::ptr::addr_of_mut!((*iface).reset_ws) }) {
unsafe { usb_get_intf(iface) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_queue_reset_device_722: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_interface)>, *mut ()>(Some(usb_queue_reset_device)) };

/// usb_hub_find_child - Get the pointer of child device
/// attached to the port which is specified by @port1.
/// @hdev: USB device belonging to the usb hub
/// @port1: port num to indicate which port the child device
/// 	is attached to.
///
/// USB drivers call this function to get hub's child device
/// pointer.
///
/// Return: %NULL if input param is invalid and
/// child's usb_device pointer if non-NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hub.usb_hub_find_child")]
extern "C" fn usb_hub_find_child(hdev: *mut usb_device, port1: i32) -> *mut usb_device {
let hub: *mut usb_hub = usb_hub_to_struct_hub(hdev);
if port1 < 1 || port1 > unsafe { (*hdev).maxchild } {
return core::ptr::null_mut::<usb_device>();
}
unsafe { (*(*unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize))).child }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_hub_find_child_723: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device, i32) -> *mut usb_device>, *mut ()>(Some(usb_hub_find_child)) };

#[unsafe(no_mangle)]
extern "C" fn usb_hub_adjust_deviceremovable(hdev: *mut usb_device, desc: *mut usb_hub_descriptor) {
let hub: *mut usb_hub = usb_hub_to_struct_hub(hdev);
let mut connect_type: u32 = 0;
let mut i: i32 = 0;
if hub.is_null() {
return;
}
if hub_is_superspeed(hdev) == 0 {
i = 1;
while i <= unsafe { (*hdev).maxchild } {
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(i.wrapping_sub(1) as isize) };
connect_type = unsafe { (*port_dev).connect_type };
if (connect_type as u32) == 2 {
let mask: u8 = (1i32 << i % 8) as u8;
if ((unsafe { (core::ptr::addr_of_mut!((*desc).u.hs.DeviceRemovable) as *mut u8).wrapping_offset((i / 8) as isize).read_unaligned() }) as i32) & (mask as i32) == 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"DeviceRemovable is changed to 1 according to platform information.\n".as_ptr() as *const u8) };
}
let __v0: *mut u8 = unsafe { core::ptr::addr_of_mut!((*desc).u.hs.DeviceRemovable) as *mut u8 }.wrapping_offset((i / 8) as isize);
unsafe {
*__v0 = ((*__v0 as i32) | (mask as i32)) as u8;
}
}
}
i = i.wrapping_add(1);
}
} else {
let mut port_removable: u16 = unsafe { (*desc).u.ss.DeviceRemovable };
i = 1;
while i <= unsafe { (*hdev).maxchild } {
let port_dev: *mut usb_port = unsafe { *unsafe { (*hub).ports }.wrapping_offset(i.wrapping_sub(1) as isize) };
connect_type = unsafe { (*port_dev).connect_type };
if (connect_type as u32) == 2 {
let mask: u16 = (1i32 << i) as u16;
if (port_removable as i32) & (mask as i32) == 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*port_dev).dev) as *const device, c"DeviceRemovable is changed to 1 according to platform information.\n".as_ptr() as *const u8) };
}
port_removable = ((port_removable as i32) | (mask as i32)) as u16;
}
}
i = i.wrapping_add(1);
}
unsafe {
(*desc).u.ss.DeviceRemovable = port_removable;
}
}
}

/// usb_get_hub_port_acpi_handle - Get the usb port's acpi handle
/// @hdev: USB device belonging to the usb hub
/// @port1: port num of the port
///
/// Return: Port's acpi handle if successful, %NULL if params are
/// invalid.
#[unsafe(no_mangle)]
extern "C" fn usb_get_hub_port_acpi_handle(hdev: *mut usb_device, port1: i32) -> *mut () {
let hub: *mut usb_hub = usb_hub_to_struct_hub(hdev);
if hub.is_null() {
return core::ptr::null_mut::<()>();
}
ACPI_HANDLE!(unsafe { core::ptr::addr_of_mut!((*(*unsafe { (*hub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize))).dev) })
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
