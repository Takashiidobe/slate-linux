
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
//! xHCI host controller driver
//!
//! Copyright (C) 2008 Intel Corp.
//!
//! Author: Sarah Sharp
//! Some code borrowed from the Linux EHCI driver.
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! USB directions
//!
//! This bit flag is used in endpoint descriptors' bEndpointAddress field.
//! It's also one of three fields in control requests bRequestType.
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! restart by calling sys_restart_syscall
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! SCHED_ISO: reserved but not implemented yet
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! This counts to 15. Any more, it will return 16th argument.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
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
//! Sign Flag
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! Internal kernel timer frequency
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
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
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! signal pending
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! 'empty' nodes are nodes that are known not to be inserted in an rbtree
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
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
//! Keep unconverted code working
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
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
//! Runtime PM flag argument bits
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! FIELD_GET() - extract a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_reg:  value of entire bitfield
//!
//! FIELD_GET() extracts the field specified by @_mask from the
//! bitfield passed in as @_reg by masking and shifting it down.
//! This is arbitrary.
//! From USB 2.0 spec Table 11-13, offset 7, a hub can
//! have up to 255 ports. The most yet reported is 10.
//! Upcoming hardware might raise that limit.
//! Because the arrays need to add a bit for hub status data, we
//! use 31, so plus one evens out to four bytes.
//! HCCPARAMS offset from PCI base address
//! PORTSC - Port Status and Control Register - port_status_base bitmasks
//!
//! true: device connected
//! bits 7:4 - Event Ring Segment Table Max, 2^(n)
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! The maximum length of a string descriptor is 255, because the bLength
//! field in the usb_string_descriptor struct is __u8.  In practice the
//! maximum length is 254, because a string descriptor consists of a 2 byte
//! header followed by UTF-16 characters (2 bytes each). This allows for
//! only 126 characters (code points) in the string, which is where
//! USB_MAX_STRING_LEN comes from.
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
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
// Overflow Flag
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
// Atomic operations are already serializing on x86
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
// clamp_val - return a value clamped to a given range using val's type
// @val: current value
// @lo: minimum allowable value
// @hi: maximum allowable value
//
// This macro does no typechecking and uses temporary variables of whatever
// type the input argument @val is.  This is useful when @val is an unsigned
// type and @lo and @hi are literals that will otherwise be assigned a signed
// integer type.
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// No static inline type checking - see Subtlety (1) above.
// N 32-bit words worth of info
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
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
// disable sas during sighandling
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// Same as rb_first(), but O(1)
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// Set the IDR flag and the IDR_FREE tag
// Generate SD flag bits
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
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
// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
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
// See USB 3.1 spec Table 10-5
// true: port enabled
// bit 2 reserved and zeroed
//
// true: port has an over-current condition
// true: port reset signaling asserted
// Port Link State - bits 5:8
// A read gives the current link PM state of the port,
// a write with Link State Write Strobe set sets the link state.
// true: port has power (see HCC_PPC)
// Bits 20:23 in the Slot Context are the speed for the device
// Port Link State Write Strobe - set this when changing link state
// true: connect status change
// true: port enable change
// true: warm reset for a USB 3.0 device is done.  A "hot" reset puts the port
// into an enabled state, and the device into the default state.  A "warm" reset
// also resets the link, forcing the device through the link training sequence.
// SW can also look at the Port Reset register to see when warm reset is done.
// true: over-current change
// true: reset change - 1 to 0 transition of PORT_RESET
// port link status change - set on some port link state transitions:
// Transition				Reason
// - U3 to Resume			Wakeup signaling from a device
// - Resume to Recovery to U0		USB 3.0 device resume
// - Resume to U0			USB 2.0 device resume
// - U3 to Recovery to U0		Software resume of USB 3.0 device complete
// - U3 to U0				Software resume of USB 2.0 device complete
// - U2 to U0				L1 resume of USB 2.1 device complete
// - U0 to U0 (???)			L1 entry rejection by USB 2.1 device
// - U0 to disabled			L1 entry error with USB 2.1 device
// - Any state to inactive		Error on USB 3.0 port
// port configure error change - port failed to configure its link partner
// Cold Attach Status - xHC can set this bit to report device attached during
// Sx state. Warm port reset should be perfomed to clear this bit and move port
// to connected state.
// wake on connect (enable)
// wake on disconnect (enable)
// wake on over-current (enable)
// bits 28:29 reserved
//
// true: device is non-removable - for USB 3.0 roothub emulation
// Initiate a warm port reset - complete when PORT_WRC is '1'
// We mark duplicate entries with -1
// bit 2 - Context Size
// bit 4 - Large ESIT Payload Capability, true: HC support ESIT payload > 48k
// bit 10 - Rsvd
//
// bit 11 - HC support Double BW on a eUSB2 HS ISOC EP
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

macro_rules! BIT_ULL {
($nr:expr) => {
ULL!() << $nr
};
}

macro_rules! BIT_WORD {
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

macro_rules! CTX_SIZE {
($_hcc:expr) => {
if (($_hcc) as u64) & BIT!(2i32) != 0 { 64i32 } else { 32i32 }
};
}

macro_rules! CTX_TO_EP_INTERVAL {
($p:expr) => {
($p) >> 16i32 & 255
};
}

macro_rules! CTX_TO_EP_MULT {
($p:expr) => {
($p) >> 8i32 & 3
};
}

macro_rules! CTX_TO_EP_TYPE {
($p:expr) => {
($p) >> 3i32 & 7
};
}

macro_rules! CTX_TO_MAX_BURST {
($p:expr) => {
($p) >> 8i32 & 255
};
}

macro_rules! CTX_TO_MAX_ESIT_PAYLOAD {
($p:expr) => {
($p) >> 16i32 & 65535
};
}

/// tx_info bitmasks
macro_rules! EP_AVG_TRB_LENGTH {
($p:expr) => {
($p) & 65535
};
}

/// tx_info bitmasks
macro_rules! EP_AVG_TRB_LENGTH__2 {
($p:expr) => {
($p) & 65535
};
}

/// bits 10:14 are Max Primary Streams
///
/// bit 15 is Linear Stream Array
///
/// Interval - period between requests to an endpoint - 125u increments.
macro_rules! EP_INTERVAL {
($p:expr) => {
(($p) & 255) << 16i32
};
}

macro_rules! EP_IS_ADDED {
($ctrl_ctx:expr, $i:expr) => {
(unsafe { (*$ctrl_ctx).add_flags }) & ((1i32 << $i.wrapping_add(1)) as u32)
};
}

macro_rules! EP_IS_DROPPED {
($ctrl_ctx:expr, $i:expr) => {
(unsafe { (*$ctrl_ctx).drop_flags }) & ((1i32 << $i.wrapping_add(1)) as u32)
};
}

macro_rules! EP_MAXPSTREAMS {
($p:expr) => {
($p) << 10i32 & 31744u32
};
}

macro_rules! EP_MAX_ESIT_PAYLOAD_HI {
($p:expr) => {
(($p) >> 16i32 & 255) << 24i32
};
}

macro_rules! EP_MAX_ESIT_PAYLOAD_LO {
($p:expr) => {
(($p) & 65535) << 16i32
};
}

/// Mult - Max number of burtst within an interval, in EP companion desc.
macro_rules! EP_MULT {
($p:expr) => {
(($p) & 3) << 8i32
};
}

macro_rules! EP_TYPE {
($p:expr) => {
($p) << 3i32
};
}

macro_rules! ERROR_COUNT {
($p:expr) => {
(($p) & 3) << 1i32
};
}

macro_rules! ERROR_COUNT__2 {
($p:expr) => {
(($p) & 3) << 1i32
};
}

macro_rules! GENMASK_INPUT_CHECK__2 {
() => {
core::mem::size_of::<__SlateRecord_xhci_mem_c_2347>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__8 {
() => {
core::mem::size_of::<__SlateRecord_xhci_mem_c_2346>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__22 {
() => {
core::mem::size_of::<__SlateRecord_xhci_mem_c_2032>() as i32
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
(GENMASK_INPUT_CHECK__8!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__18 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__2!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__22 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__22!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK_ULL__6 {
($h:expr, $l:expr) => {
GENMASK_TYPE__22!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__9 {
($h:expr, $l:expr) => {
GENMASK_TYPE__18!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__12 {
($h:expr, $l:expr) => {
GENMASK_TYPE__15!(u64, $h, $l)
};
}

macro_rules! INIT_RADIX_TREE {
($root:expr, $mask:expr) => {
xa_init_flags($root, $mask);
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

macro_rules! LAST_CTX {
($p:expr) => {
($p) << 27i32
};
}

/// bit 6 reserved
///
/// bit 7 is Host Initiate Disable - for disabling stream selection
macro_rules! MAX_BURST {
($p:expr) => {
(($p) & 255) << 8i32
};
}

/// bit 6 reserved
///
/// bit 7 is Host Initiate Disable - for disabling stream selection
macro_rules! MAX_BURST__2 {
($p:expr) => {
(($p) & 255) << 8i32
};
}

macro_rules! MAX_PACKET_DECODED {
($p:expr) => {
($p) >> 16i32 & 65535
};
}

macro_rules! MAX_PACKET__2 {
($p:expr) => {
(($p) & 65535) << 16i32
};
}

macro_rules! MAX_PACKET {
($p:expr) => {
(($p) & 65535) << 16i32
};
}

macro_rules! PASS_BUCKET_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PASS_KMALLOC_PARAMS {
($_size:expr) => {
PASS_BUCKET_PARAMS!($_size)
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

/// Root hub port number that is needed to access the USB device
macro_rules! ROOT_HUB_PORT {
($p:expr) => {
(($p) & 255) << 16i32
};
}

/// Stream Context Types (section 6.4.1) - bits 3:1 of stream ctx deq ptr
macro_rules! SCT_FOR_CTX {
($p:expr) => {
(($p) & 7) << 1i32
};
}

macro_rules! TRB_TYPE {
($p:expr) => {
($p) << 10i32
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

macro_rules! USB_EP_MAXP_MULT {
($m:expr) => {
(($m) & 6144i32) >> 11i32
};
}

/// Extended capability register fields
macro_rules! XHCI_EXT_CAPS_ID {
($p:expr) => {
($p) >> 0i32 & 255
};
}

macro_rules! XHCI_EXT_CAPS_NEXT {
($p:expr) => {
($p) >> 8i32 & 255
};
}

macro_rules! XHCI_EXT_PORT_COUNT {
($x:expr) => {
($x) >> 8i32 & 255
};
}

macro_rules! XHCI_EXT_PORT_LP {
($x:expr) => {
($x) >> 14i32 & 3
};
}

macro_rules! XHCI_EXT_PORT_MAJOR {
($x:expr) => {
($x) >> 24i32 & 255
};
}

macro_rules! XHCI_EXT_PORT_MINOR {
($x:expr) => {
($x) >> 16i32 & 255
};
}

macro_rules! XHCI_EXT_PORT_OFF {
($x:expr) => {
($x) & 255
};
}

macro_rules! XHCI_EXT_PORT_PFD {
($x:expr) => {
($x) >> 8i32 & 1
};
}

macro_rules! XHCI_EXT_PORT_PLT {
($x:expr) => {
($x) >> 6i32 & 3
};
}

macro_rules! XHCI_EXT_PORT_PSIC {
($x:expr) => {
($x) >> 28i32 & 15
};
}

macro_rules! XHCI_EXT_PORT_PSIE {
($x:expr) => {
($x) >> 4i32 & 3
};
}

macro_rules! XHCI_EXT_PORT_PSIM {
($x:expr) => {
($x) >> 16i32 & 65535
};
}

macro_rules! XHCI_EXT_PORT_PSIV {
($x:expr) => {
($x) >> 0i32 & 15
};
}

/// HCCPARAMS contains the first extended capability pointer
macro_rules! XHCI_HCC_EXT_CAPS {
($p:expr) => {
($p) >> 16i32 & 65535
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

macro_rules! __bf_cast_unsigned {
($x:expr) => {
($x) as u32
};
}

macro_rules! __bf_cast_unsigned__2 {
($x:expr) => {
($x) as u64
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
if is_signed_type__4!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u8>() }) as u64) < 4) as i32)) }
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

macro_rules! __types_ok3__2 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use__4!($ux) & __sign_use__4!($uy) & __sign_use__4!($uz)
};
}

macro_rules! __types_ok3__3 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy) & __sign_use__2!($uz)
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

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

/// array_size() - Calculate size of 2-dimensional array.
/// @a: dimension one
/// @b: dimension two
///
/// Calculates size of 2-dimensional array: @a * @b.
///
/// Returns: number of bytes needed to represent the array or SIZE_MAX on
/// overflow.
macro_rules! array_size {
($a:expr, $b:expr) => {
size_mul($a, ($b) as u64)
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
macro_rules! container_of__36 {
($ptr:expr) => {
{
let mut __v: *mut xhci_tt_bw_info;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__36!() as isize)) }) as *mut ()) as *mut xhci_tt_bw_info;
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
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
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
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { __ilog2_u32($n) }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__4 {
() => {
((1i32.wrapping_neg() as u8) as i32) < 1
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
macro_rules! list_entry__38 {
($ptr:expr) => {
container_of__36!($ptr)
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
list_entry__38!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__3 {
($pos:expr, $member:ident) => {
list_entry__38!(unsafe { (*$pos).$member.next })
};
}

macro_rules! offsetof__36 {
() => {
core::mem::offset_of!(xhci_tt_bw_info, tt_list) as u64
};
}

macro_rules! preempt_count_dec_and_test {
() => {
__preempt_count_dec_and_test()
};
}

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
};
}

/// rounddown_pow_of_two - round the given value down to nearest power of two
/// @n: parameter
///
/// round the given value down to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! rounddown_pow_of_two {
($n:expr) => {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { 1u64 << ilog2__5!($n) } else { __rounddown_pow_of_two(($n) as u64) }
};
}

macro_rules! static_branch_unlikely__5 {
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

macro_rules! static_branch_unlikely__6 {
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

macro_rules! static_branch_unlikely__3 {
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

macro_rules! static_branch_unlikely__4 {
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

macro_rules! static_branch_unlikely__7 {
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

macro_rules! static_branch_unlikely__8 {
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

macro_rules! static_branch_unlikely__9 {
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

macro_rules! static_branch_unlikely__10 {
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

unsafe extern "C" {
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __cpu_online_mask: cpumask;
    static mut cpuinfo_op: seq_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut cpu_number: i32;
    static mut radix_tree_preloads: radix_tree_preload;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_xhci_dbg_context_change: static_call_key;
    static mut __tracepoint_xhci_dbg_context_change: tracepoint;
    static mut __SCK__tp_func_xhci_dbg_init: static_call_key;
    static mut __tracepoint_xhci_dbg_init: tracepoint;
    static mut __SCK__tp_func_xhci_dbg_ring_expansion: static_call_key;
    static mut __tracepoint_xhci_dbg_ring_expansion: tracepoint;
    static mut __SCK__tp_func_xhci_free_virt_device: static_call_key;
    static mut __tracepoint_xhci_free_virt_device: tracepoint;
    static mut __SCK__tp_func_xhci_alloc_virt_device: static_call_key;
    static mut __tracepoint_xhci_alloc_virt_device: tracepoint;
    static mut __SCK__tp_func_xhci_setup_addressable_virt_device: static_call_key;
    static mut __tracepoint_xhci_setup_addressable_virt_device: tracepoint;
    static mut __SCK__tp_func_xhci_alloc_stream_info_ctx: static_call_key;
    static mut __tracepoint_xhci_alloc_stream_info_ctx: tracepoint;
    static mut __SCK__tp_func_xhci_ring_alloc: static_call_key;
    static mut __tracepoint_xhci_ring_alloc: tracepoint;
    static mut __SCK__tp_func_xhci_ring_free: static_call_key;
    static mut __tracepoint_xhci_ring_free: tracepoint;
    static mut __SCK__tp_func_xhci_ring_expansion: static_call_key;
    static mut __tracepoint_xhci_ring_expansion: tracepoint;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn ____wrong_branch_error() -> bool;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn rcu_is_watching() -> bool;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn cancel_delayed_work_sync(arg0: *mut delayed_work) -> bool;
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn __init_swait_queue_head(arg0: *mut swait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn __compiletime_assert_168();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn radix_tree_insert(arg0: *mut xarray, arg1: u64, arg2: *mut ()) -> i32;
    fn radix_tree_lookup(arg0: *const xarray, arg1: u64) -> *mut ();
    fn radix_tree_delete(arg0: *mut xarray, arg1: u64) -> *mut ();
    fn radix_tree_preload(arg0: u32) -> i32;
    fn radix_tree_maybe_preload(arg0: u32) -> i32;
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_415();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_cache_node_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: i32, arg3: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_large_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn usb_endpoint_max_periodic_payload(arg0: *mut usb_device, arg1: *const usb_host_endpoint) -> u32;
    fn usb_endpoint_is_hs_isoc_double(arg0: *mut usb_device, arg1: *const usb_host_endpoint) -> bool;
    fn dma_pool_create_node(arg0: *const u8, arg1: *mut device, arg2: u64, arg3: u64, arg4: u64, arg5: i32) -> *mut dma_pool;
    fn dma_pool_destroy(arg0: *mut dma_pool);
    fn dma_pool_alloc(arg0: *mut dma_pool, arg1: u32, arg2: *mut u64) -> *mut ();
    fn dma_pool_free(arg0: *mut dma_pool, arg1: *mut (), arg2: u64);
    fn dma_alloc_attrs(arg0: *mut device, arg1: u64, arg2: *mut u64, arg3: u32, arg4: u64) -> *mut ();
    fn dma_free_attrs(arg0: *mut device, arg1: u64, arg2: *mut (), arg3: u64, arg4: u64);
    fn usb_hcd_is_primary_hcd(arg0: *mut usb_hcd) -> i32;
    fn xhci_dbg_trace(arg0: *mut xhci_hcd, arg1: Option<unsafe extern "C" fn(*mut va_format)>, arg2: *const u8, ...);
    fn xhci_get_endpoint_index(arg0: *mut usb_endpoint_descriptor) -> u32;
    fn xhci_update_tt_active_eps(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: i32);
    fn xhci_skip_sec_intr_events(arg0: *mut xhci_hcd, arg1: *mut xhci_ring, arg2: *mut xhci_interrupter);
    fn xhci_halt(arg0: *mut xhci_hcd) -> i32;
    fn xhci_reset(arg0: *mut xhci_hcd, arg1: u64) -> i32;
    fn xhci_set_interrupter_moderation(arg0: *mut xhci_interrupter, arg1: u32) -> i32;
    fn xhci_trb_virt_to_dma(arg0: *mut xhci_segment, arg1: *mut xhci_trb) -> u64;
    fn xhci_cleanup_command_queue(arg0: *mut xhci_hcd);
    fn xhci_update_erst_dequeue(arg0: *mut xhci_hcd, arg1: *mut xhci_interrupter, arg2: bool);
    fn xhci_usb_endpoint_maxp(arg0: *mut usb_device, arg1: *mut usb_host_endpoint) -> i32;
    fn xhci_get_rhub(arg0: *mut usb_hcd) -> *mut xhci_hub;
    fn __traceiter_xhci_dbg_context_change(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_context_change(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_538();
    fn __traceiter_xhci_dbg_init(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_init(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_566();
    fn __traceiter_xhci_dbg_ring_expansion(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_ring_expansion(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_573();
    fn __traceiter_xhci_free_virt_device(arg0: *mut (), arg1: *mut xhci_virt_device) -> i32;
    fn __SCT__tp_func_xhci_free_virt_device(arg0: *mut (), arg1: *mut xhci_virt_device) -> i32;
    fn __compiletime_assert_636();
    fn __traceiter_xhci_alloc_virt_device(arg0: *mut (), arg1: *mut xhci_virt_device) -> i32;
    fn __SCT__tp_func_xhci_alloc_virt_device(arg0: *mut (), arg1: *mut xhci_virt_device) -> i32;
    fn __compiletime_assert_643();
    fn __traceiter_xhci_setup_addressable_virt_device(arg0: *mut (), arg1: *mut xhci_virt_device) -> i32;
    fn __SCT__tp_func_xhci_setup_addressable_virt_device(arg0: *mut (), arg1: *mut xhci_virt_device) -> i32;
    fn __compiletime_assert_657();
    fn __traceiter_xhci_alloc_stream_info_ctx(arg0: *mut (), arg1: *mut xhci_stream_info, arg2: u32) -> i32;
    fn __SCT__tp_func_xhci_alloc_stream_info_ctx(arg0: *mut (), arg1: *mut xhci_stream_info, arg2: u32) -> i32;
    fn __compiletime_assert_692();
    fn __traceiter_xhci_ring_alloc(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __SCT__tp_func_xhci_ring_alloc(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __compiletime_assert_818();
    fn __traceiter_xhci_ring_free(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __SCT__tp_func_xhci_ring_free(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __compiletime_assert_825();
    fn __traceiter_xhci_ring_expansion(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __SCT__tp_func_xhci_ring_expansion(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __compiletime_assert_832();
    fn xhci_debugfs_remove_root();
    fn xhci_debugfs_remove_slot(arg0: *mut xhci_hcd, arg1: i32);
    fn __compiletime_assert_928();
    fn __compiletime_assert_929();
    fn __compiletime_assert_933();
    fn __compiletime_assert_934();
    fn __compiletime_assert_939();
    fn __compiletime_assert_940();
    fn __compiletime_assert_941();
    fn __compiletime_assert_942();
    fn __compiletime_assert_943();
    fn __compiletime_assert_944();
    fn __compiletime_assert_945();
    fn __compiletime_assert_946();
    fn __compiletime_assert_947();
    fn __compiletime_assert_948();
    fn __compiletime_assert_949();
    fn __compiletime_assert_950();
    fn __compiletime_assert_951();
    fn __compiletime_assert_952();
    fn __compiletime_assert_953();
    fn __compiletime_assert_954();
    fn __compiletime_assert_955();
    fn __compiletime_assert_956();
    fn __compiletime_assert_957();
    fn __compiletime_assert_958();
    fn __compiletime_assert_959();
    fn __compiletime_assert_960();
    fn __compiletime_assert_962();
    fn __compiletime_assert_963();
    fn __compiletime_assert_964();
    fn __compiletime_assert_965();
    fn __compiletime_assert_966();
    fn __compiletime_assert_969();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
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


// USB String descriptors can contain at most 126 characters.
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

fn usb_endpoint_xfer_bulk(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 2) as i32
}

fn usb_endpoint_xfer_control(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 0) as i32
}

fn usb_endpoint_xfer_int(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 3) as i32
}

fn usb_endpoint_xfer_isoc(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 1) as i32
}

fn usb_endpoint_maxp_mult(epd: *const usb_endpoint_descriptor) -> i32 {
let maxp: i32 = __le16_to_cpu!(unsafe { (*epd).wMaxPacketSize }) as i32;
USB_EP_MAXP_MULT!(maxp).wrapping_add(1)
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


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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
struct usb_ptm_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
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
struct lock_class_key {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockdep_map {
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
struct va_format {
    fmt: *const u8,
    va: *mut core::ffi::VaList<'static>,
}


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
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
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
struct perf_event {
}


#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

fn __rounddown_pow_of_two(n: u64) -> u64 {
1u64 << fls_long(n).wrapping_sub(1)
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
    __slate_bits_0: __slate_bits::__SlateBits141U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_context_change, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_init, const 0 as u8, label {
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


#[inline(always)]
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_ring_expansion, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_free_virt_device, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_alloc_virt_device, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_setup_addressable_virt_device, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_alloc_stream_info_ctx, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_ring_alloc, const 0 as u8, label {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
fn arch_static_branch__const9(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_ring_free, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_ring_expansion, const 0 as u8, label {
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
    __slate_bits_0: __slate_bits::__SlateBits148U0,
}


#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
struct seccomp_filter {
}


fn INIT_LIST_HEAD(list: *mut list_head) {
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
__WRITE_ONCE!((*list).next, list);
if true {
break;
}
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

#[inline(always)]
fn list_add(new: *mut list_head, head: *mut list_head) {
__list_add(new, head, unsafe { (*head).next });
}

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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


fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
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

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
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
    __slate_bits_0: __slate_bits::__SlateBits149U0,
    pkru: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
// We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct rb_node_linked {
    node: rb_node,
    prev: *mut rb_node_linked,
    next: *mut rb_node_linked,
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
    __slate_bits_0: __slate_bits::__SlateBits242U0,
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


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct proc_dir_entry {
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
struct uprobe {
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
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
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
}

fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_xarray_h_1175 {
    private_list: list_head,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_xarray_h_1180 {
    tags: [[u64; 1]; 3],
    marks: [[u64; 1]; 3],
}


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
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
struct ns_common {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


fn init_completion(x: *mut completion) {
unsafe {
(*x).done = 0;
}
loop {
unsafe { __init_swait_queue_head(core::ptr::addr_of_mut!((*x).wait), c"&x->wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key)) };
if true {
break;
}
}
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

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
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_168() };
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
struct wake_irq {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
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
struct folio_queue {
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
struct kmem_cache {
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
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits437U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
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
struct key_type {
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
struct local_lock {
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
struct local_trylock {
    acquired: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


fn local_lock_release(l: *mut local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
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
struct xa_node {
    shift: u8,
    offset: u8,
    count: u8,
    nr_values: u8,
    parent: *mut xa_node,
    array: *mut xarray,
    __slate_anon_6: __SlateRecord_xarray_h_1175,
    slots: [*mut (); 64],
    __slate_anon_8: __SlateRecord_xarray_h_1180,
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
#[repr(C)]
#[derive(Clone, Copy)]
struct radix_tree_preload {
    lock: local_lock,
    nr: u32,
    nodes: *mut xa_node,
}


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
fn radix_tree_preload_end() {
loop {
__release!();
loop {
let mut __v1: *mut local_lock;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<local_lock>() as *const ();
if true {
break;
}
}
let mut __v3: *mut local_lock;
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
let __v6: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(radix_tree_preloads.lock) }) as u64);
tcp_ptr__ = __v6;
__v3 = tcp_ptr__ as *mut local_lock;
__v1 = __v3;
let __l: *mut local_lock = __v1;
local_lock_release(__l);
if true {
break;
}
}
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
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_219: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

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
    __slate_bits_0: __slate_bits::__SlateBits492U0,
    vm_ops: *const vm_operations_struct,
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
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
struct pm_message {
    event: i32,
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
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
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
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct module_param_attrs {
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


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct ep_device {
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
struct mon_bus {
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
struct dma_pool {
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
    __slate_bits_0: __slate_bits::__SlateBits645U0,
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
#[inline(always)]
fn _kmalloc_node_noprof(size: u64, flags: u32, node: i32, token: __SlateRecord_slab_h_532) -> *mut () {
if (unsafe { __slate_intrinsic_f217c4720f04f6db(size) }) && size != 0 {
let mut index: u32 = 0;
if size > 8192u64 {
return unsafe { __kmalloc_large_node_noprof(size, flags, node) };
}
index = kmalloc_index!(size);
return unsafe { __kmalloc_cache_node_noprof(kmalloc_caches[(kmalloc_type(flags, token) as u32) as usize][index as usize], flags, node, size) };
}
unsafe { __kmalloc_node_noprof(PASS_KMALLOC_PARAMS!(size), flags, node) }
}

fn _kmalloc_array_node_noprof(n: u64, size: u64, flags: u32, node: i32, token: __SlateRecord_slab_h_532) -> *mut () {
let mut bytes: u64 = 0;
if unlikely!(check_mul_overflow!(n, size, core::ptr::addr_of_mut!(bytes))) != 0 {
return core::ptr::null_mut::<()>();
}
if (unsafe { __slate_intrinsic_f217c4720f04f6db(n) }) && unsafe { __slate_intrinsic_f217c4720f04f6db(size) } {
return _kmalloc_node_noprof(bytes, flags, node, token);
}
unsafe { __kmalloc_node_noprof(PASS_KMALLOC_PARAMS!(bytes), flags, node) }
}

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
struct xhci_sideband {
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
struct fasync_struct {
    fa_lock: rwlock,
    magic: i32,
    fa_fd: i32,
    fa_next: *mut fasync_struct,
    fa_file: *mut file,
    fa_rcu: callback_head,
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
struct clk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reset_control {
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
struct serial_icounter_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_struct {
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
union __SlateRecord_tty_buffer_h_11 {
    next: *mut tty_buffer,
    free: llist_node,
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
struct __SlateRecord_tty_port_h_118 {
    __slate_anon_0: __SlateRecord_tty_port_h_118_2,
    buf: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_tty_port_h_118_2 {
    kfifo: __kfifo,
    r#type: *mut u8,
    const_type: *const u8,
    rectype: *mut [u8; 0],
    ptr: *mut u8,
    ptr_const: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_h_212 {
    lock: spinlock,
    stopped: bool,
    tco_stopped: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_h_218 {
    pgrp: *mut pid,
    session: *mut pid,
    lock: spinlock,
    pktstatus: u8,
    packet: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_21 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_23 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_24 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_27 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1651_28 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_17 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_21 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_23 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_24 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_27 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_1719_28 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2032 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2304_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2304_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2304_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2304_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2304_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2304_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2304_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2304_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2346 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_mem_c_2347 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
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
    pub struct __SlateBits645U0 {
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
    pub struct __SlateBits148U0 {
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
    pub struct __SlateBits437U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits846U0 {
        #[bits(1)]
        pub console: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits141U0 {
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
    pub struct __SlateBits149U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits242U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits757U0 {
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
    pub struct __SlateBits753U0 {
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
    pub struct __SlateBits747U0 {
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
    pub struct __SlateBits747U1 {
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
    pub struct __SlateBits747U2 {
        #[bits(1)]
        pub use_generic_driver: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits765U0 {
        #[bits(1)]
        pub poisoned: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits783U0 {
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
    pub struct __SlateBits810U0 {
        #[bits(1)]
        pub link_inactive: u32,
        #[bits(1)]
        pub connected: u32,
        #[bits(1)]
        pub lpm_incapable: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits830U0 {
        #[bits(1)]
        pub hw_lpm_support: u32,
        #[bits(1)]
        pub broken_suspend: u32,
        #[bits(1)]
        pub allow_single_roothub: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits516U0,
}


fn dev_to_node(dev: *mut device) -> i32 {
unsafe { (*dev).numa_node }
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
    __slate_bits_0: __slate_bits::__SlateBits753U0,
    wireless_status: u32,
    wireless_status_work: work_struct,
    dev: device,
    usb_dev: *mut device,
    reset_ws: work_struct,
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
    __slate_bits_0: __slate_bits::__SlateBits757U0,
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
    __slate_bits_0: __slate_bits::__SlateBits747U0,
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
    __slate_bits_1: __slate_bits::__SlateBits747U1,
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
    __slate_bits_2: __slate_bits::__SlateBits747U2,
}


// urb->transfer_dma valid on submit
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
    __slate_bits_0: __slate_bits::__SlateBits765U0,
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


// These macros should be used after a dma_map_sg call has been done
// to get bus addresses of each of the SG entries and their lengths.
// You should only work with the number of sg entries dma_map_sg
// returns, or alternatively stop on the first sg_dma_len(sg) which
// is 0.
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
    __slate_bits_0: __slate_bits::__SlateBits783U0,
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


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_xhci_initialize_ring_info: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_get_ep_ctx: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_remove_secondary_interrupter: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_create_secondary_interrupter: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous", sym xhci_initialize_ring_info, sym xhci_get_ep_ctx, sym xhci_remove_secondary_interrupter, sym xhci_create_secondary_interrupter, options(att_syntax));

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


fn lo_hi_readq(addr: *const ()) -> u64 {
let p: *const u32 = addr as *const u32;
let low: u32 = readl(p as *const ());
let high: u32 = readl(p.wrapping_offset(1 as isize) as *const ());
(low as u64).wrapping_add((high as u64) << 32i32)
}

fn lo_hi_writeq(val: u64, addr: *mut ()) {
writel(val as u32, addr);
writel((val >> 32i32) as u32, (unsafe { (addr as *const u8).wrapping_offset(4 as isize) }) as *mut ());
}

fn hi_lo_writeq(val: u64, addr: *mut ()) {
writel((val >> 32i32) as u32, (unsafe { (addr as *const u8).wrapping_offset(4 as isize) }) as *mut ());
writel(val as u32, addr);
}

// USB 2.0 xHCI 1.0 hardware LMP capability - section 7.2.2.1.3.2
fn xhci_find_next_ext_cap(base: *mut (), start: u32, id: i32) -> i32 {
let mut val: u32 = 0;
let mut next: u32 = 0;
let mut offset: u32 = start;
if start == 0 || start == 16 {
val = readl(((unsafe { (base as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
if val == 4294967295u32 {
return 0;
}
offset = XHCI_HCC_EXT_CAPS!(val) << 2i32;
if offset == 0 {
return 0;
}
}
loop {
val = readl(((unsafe { (base as *const u8).wrapping_offset(offset as isize) }) as *mut ()) as *const ());
if val == 4294967295u32 {
return 0;
}
if offset != start && (id == 0 || XHCI_EXT_CAPS_ID!(val) == (id as u32)) {
return offset as i32;
}
next = XHCI_EXT_CAPS_NEXT!(val);
offset = offset.wrapping_add(next << 2i32);
if next == 0 {
break;
}
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_cap_regs {
    hc_capbase: u32,
    hcs_params1: u32,
    hcs_params2: u32,
    hcs_params3: u32,
    hcc_params: u32,
    db_off: u32,
    run_regs_off: u32,
    hcc_params2: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_port_regs {
    portsc: u32,
    portpmsc: u32,
    portli: u32,
    porthlmpc: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_op_regs {
    command: u32,
    status: u32,
    page_size: u32,
    reserved1: u32,
    reserved2: u32,
    dev_notification: u32,
    cmd_ring: u64,
    reserved3: [u32; 4],
    dcbaa_ptr: u64,
    config_reg: u32,
    reserved4: [u32; 241],
    port_regs: [xhci_port_regs; 0],
}


// USBSTS - USB status - status bitmasks
//
// HC not running - set to 1 when run/stop bit is cleared.
// serious error, e.g. PCI parity error.  The HC will clear the run/stop bit.
// event interrupt - clear this prior to clearing any IP flags in IR set
// port change detect
// bits 5:7 reserved and zeroed
//
// save state status - '1' means xHC is saving state
// restore state status - '1' means xHC is restoring state
// true: save or restore error
// true: Controller Not Ready to accept doorbell or op reg writes after reset
// true: internal Host Controller Error - SW needs to reset and reinitialize
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_intr_reg {
    iman: u32,
    imod: u32,
    erst_size: u32,
    rsvd: u32,
    erst_base: u64,
    erst_dequeue: u64,
}


// erst_size bitmasks
//
// bits 15:0 - Event Ring Segment Table Size, number of ERST entries
// erst_base bitmasks
//
// bits 63:6 - Event Ring Segment Table Base Address Register
// bits 63:4 - Event Ring Dequeue Pointer
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_run_regs {
    microframe_index: u32,
    rsvd: [u32; 7],
    ir_set: [xhci_intr_reg; 1024],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_doorbell_array {
    doorbell: [u32; 256],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_container_ctx {
    r#type: u32,
    size: i32,
    bytes: *mut u8,
    dma: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_slot_ctx {
    dev_info: u32,
    dev_info2: u32,
    tt_info: u32,
    dev_state: u32,
    reserved: [u32; 4],
}


// dev_info bitmasks
//
// Route String - 0:19
// Device speed - values defined by PORTSC Device Speed field - 20:23
// bit 24 reserved
//
// Is this LS/FS device connected through a HS hub? - bit 25
// Set if the device is a hub - bit 26
// Index of the last valid endpoint context in this device context - 27:31
// dev_info2 bitmasks
//
// Max Exit Latency (ms) - worst case time to wake up all links in dev path
// tt_info bitmasks
//
// TT Hub Slot ID - for low or full speed devices attached to a high-speed hub
// The Slot ID of the hub that isolates the high speed signaling from
// this low or full-speed device.  '0' if attached to root hub port.
// The number of the downstream facing port of the high-speed hub
// '0' if the device is not low or full speed.
// dev_state bitmasks
//
// USB device address - assigned by the HC
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_ep_ctx {
    ep_info: u32,
    ep_info2: u32,
    deq: u64,
    tx_info: u32,
    reserved: [u32; 3],
}


// ep_info bitmasks
//
// Endpoint State - bits 0:2
// 0 - disabled
// 1 - running
// 2 - halted due to halt condition - ok to manipulate endpoint ring
// 3 - stopped
// 4 - TRB error
// 5-7 - reserved
// Endpoint is set up with a Linear Stream Array (vs. Secondary Stream Array)
// hosts with LEC=1 use bits 31:24 as ESIT high bits.
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_input_control_ctx {
    drop_flags: u32,
    add_flags: u32,
    rsvd2: [u32; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_command {
    in_ctx: *mut xhci_container_ctx,
    status: u32,
    comp_param: u32,
    slot_id: i32,
    completion: *mut completion,
    command_trb: *mut xhci_trb,
    cmd_list: list_head,
    timeout_ms: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_stream_ctx {
    stream_ring: u64,
    reserved: [u32; 2],
}


// Primary stream array type, dequeue pointer is to a transfer ring
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_stream_info {
    stream_rings: *mut *mut xhci_ring,
    num_streams: u32,
    stream_ctx_array: *mut xhci_stream_ctx,
    num_stream_ctxs: u32,
    ctx_array_dma: u64,
    trb_address_map: xarray,
    free_streams_command: *mut xhci_command,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_bw_info {
    ep_interval: u32,
    mult: u32,
    num_packets: u32,
    max_packet_size: u32,
    max_esit_payload: u32,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_virt_ep {
    vdev: *mut xhci_virt_device,
    ep_index: u32,
    ring: *mut xhci_ring,
    stream_info: *mut xhci_stream_info,
    new_ring: *mut xhci_ring,
    err_count: u32,
    ep_state: u32,
    cancelled_td_list: list_head,
    queued_deq_seg: *mut xhci_segment,
    queued_deq_ptr: *mut xhci_trb,
    skip: bool,
    bw_info: xhci_bw_info,
    bw_endpoint_list: list_head,
    stop_time: u64,
    next_uframe: i32,
    use_extended_tbc: bool,
    sideband: *mut xhci_sideband,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_interval_bw {
    num_packets: u32,
    endpoints: list_head,
    overhead: [u32; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_interval_bw_table {
    interval0_esit_payload: u32,
    interval_bw: [xhci_interval_bw; 16],
    bw_used: u32,
    ss_bw_in: u32,
    ss_bw_out: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_virt_device {
    slot_id: i32,
    udev: *mut usb_device,
    out_ctx: *mut xhci_container_ctx,
    in_ctx: *mut xhci_container_ctx,
    eps: [xhci_virt_ep; 31],
    rhub_port: *mut xhci_port,
    bw_table: *mut xhci_interval_bw_table,
    tt_info: *mut xhci_tt_bw_info,
    current_mel: u16,
    debugfs_private: *mut (),
    sideband: *mut xhci_sideband,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_root_port_bw_info {
    tts: list_head,
    num_active_tts: u32,
    bw_table: xhci_interval_bw_table,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_tt_bw_info {
    tt_list: list_head,
    slot_id: i32,
    ttport: i32,
    bw_table: xhci_interval_bw_table,
    active_eps: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_device_context_array {
    ctx_array: *mut u64,
    dma: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_transfer_event {
    buffer: u64,
    transfer_len: u32,
    flags: u32,
}


// Transfer event flags bitfield, also for select command completion events
// Endpoint ID 1 - 31
// Transfer event TRB length bit mask
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_link_trb {
    segment_ptr: u64,
    intr_target: u32,
    control: u32,
}


// control bitfields
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_event_cmd {
    cmd_trb: u64,
    status: u32,
    flags: u32,
}


// Address device - disable SetAddress
// Configure Endpoint - Deconfigure
// Stop Ring - Transfer State Preserve
// Force Event
// Set Latency Tolerance Value
// Get Port Bandwidth
// Force Header
// Set TR Dequeue Pointer command TRB fields, 6.4.3.9
// Link TRB specific fields
// Normal TRB fields
//
// transfer_len bitmasks - bits 0:16
// Cycle bit - indicates TRB ownership by HC or HCD
// Force next event data TRB to be evaluated before task switch.
// Used to pass OS data back after a TD completes.
// Interrupt on short packet
// Set PCIe no snoop attribute
// Chain multiple TRBs into a TD
// Interrupt on completion
// The buffer pointer contains immediate data
// TDs smaller than this might use IDT
// Block Event Interrupt
// Isochronous TRB specific fields
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_generic_trb {
    field: [u32; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
union xhci_trb {
    link: xhci_link_trb,
    trans_event: xhci_transfer_event,
    event_cmd: xhci_event_cmd,
    generic: xhci_generic_trb,
}


// TRB bit mask
// TRB type IDs
//
// bulk, interrupt, isoc scatter/gather, and control data stage
// setup stage for control transfers
// data stage for control transfers
// status stage for control transfers
// isoc transfers
// TRB for linking ring segments
// Transfer Ring No-op (not for the command ring)
// Command TRBs
//
// Enable Slot Command
// Disable Slot Command
// Address Device Command
// Configure Endpoint Command
// Evaluate Context Command
// Reset Endpoint Command
// Stop Transfer Ring Command
// Set Transfer Ring Dequeue Pointer Command
// Reset Device Command
// Force Event Command (opt)
// Negotiate Bandwidth Command (opt)
// Set Latency Tolerance Value Command (opt)
// Get port bandwidth Command
// Force Header Command - generate a transaction or link management packet
// No-op Command - not for transfer rings
// TRB IDs 24-31 reserved
//
// Event TRBS
//
// Transfer Event
// Command Completion Event
// Port Status Change Event
// Bandwidth Request Event (opt)
// Doorbell Event (opt)
// Host Controller Event
// Device Notification Event - device sent function wake notification
// MFINDEX Wrap Event - microframe counter wrapped
// Nec vendor-specific command completion event.
// Get NEC firmware revision.
// TRBS_PER_SEGMENT must be a multiple of 4,
// since the command ring is 64-byte aligned.
// It must also be greater than 16.
// Allow two commands + a link TRB, along with any reserved command TRBs
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_segment {
    trbs: *mut xhci_trb,
    next: *mut xhci_segment,
    num: u32,
    dma: u64,
    bounce_dma: u64,
    bounce_buf: *mut (),
    bounce_offs: u32,
    bounce_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_td {
    td_list: list_head,
    cancelled_td_list: list_head,
    status: i32,
    cancel_status: u32,
    urb: *mut urb,
    start_seg: *mut xhci_segment,
    start_trb: *mut xhci_trb,
    end_seg: *mut xhci_segment,
    end_trb: *mut xhci_trb,
    bounce_seg: *mut xhci_segment,
    urb_length_set: bool,
    error_mid_td: bool,
}


// xHCI command default timeout value in milliseconds.
// USB 3.2 spec, section 9.2.6.1
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_ring {
    first_seg: *mut xhci_segment,
    last_seg: *mut xhci_segment,
    enqueue: *mut xhci_trb,
    enq_seg: *mut xhci_segment,
    dequeue: *mut xhci_trb,
    deq_seg: *mut xhci_segment,
    td_list: list_head,
    cycle_state: u32,
    stream_id: u32,
    num_segs: u32,
    bounce_buf_len: u32,
    r#type: u32,
    old_trb_comp_code: u32,
    trb_address_map: *mut xarray,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_erst_entry {
    seg_addr: u64,
    seg_size: u32,
    rsvd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_erst {
    entries: *mut xhci_erst_entry,
    num_entries: u32,
    erst_dma_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_scratchpad {
    sp_array: *mut u64,
    sp_dma: u64,
    sp_buffers: *mut *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct urb_priv {
    num_tds: i32,
    num_tds_done: i32,
    td: [xhci_td; 0],
}


// Number of Event Ring segments to allocate, when amount is not specified. (spec allows 32k)
#[repr(C)]
#[derive(Clone, Copy)]
struct s3_save {
    command: u32,
    dev_nt: u32,
    dcbaa_ptr: u64,
    config_reg: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_bus_state {
    bus_suspended: u64,
    next_statechange: u64,
    port_c_suspend: u32,
    suspended_ports: u32,
    port_remote_wakeup: u32,
    resuming_ports: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_interrupter {
    event_ring: *mut xhci_ring,
    erst: xhci_erst,
    ir_set: *mut xhci_intr_reg,
    intr_num: u32,
    ip_autoclear: bool,
    isoc_bei_interval: u32,
    s3_iman: u32,
    s3_imod: u32,
    s3_erst_size: u32,
    s3_erst_base: u64,
    s3_erst_dequeue: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_port_cap {
    psi: *mut u32,
    psi_count: u8,
    psi_uid_count: u8,
    maj_rev: u8,
    min_rev: u8,
    protocol_caps: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_port {
    port_reg: *mut xhci_port_regs,
    hw_portnum: i32,
    hcd_portnum: i32,
    rhub: *mut xhci_hub,
    port_cap: *mut xhci_port_cap,
    __slate_bits_0: __slate_bits::__SlateBits810U0,
    resume_timestamp: u64,
    rexit_active: bool,
    slot_id: i32,
    rexit_done: completion,
    u3exit_done: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_hub {
    ports: *mut *mut xhci_port,
    num_ports: u32,
    hcd: *mut usb_hcd,
    bus_state: xhci_bus_state,
    maj_rev: u8,
    min_rev: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_hcd {
    main_hcd: *mut usb_hcd,
    shared_hcd: *mut usb_hcd,
    cap_regs: *mut xhci_cap_regs,
    op_regs: *mut xhci_op_regs,
    run_regs: *mut xhci_run_regs,
    dba: *mut xhci_doorbell_array,
    hcs_params2: u32,
    hcs_params3: u32,
    hcc_params: u32,
    hcc_params2: u32,
    lock: spinlock,
    hci_version: u16,
    max_interrupters: u16,
    max_slots: u8,
    max_ports: u8,
    imod_interval: u32,
    page_size: u32,
    dma_mask_bits: u32,
    nvecs: i32,
    clk: *mut clk,
    reg_clk: *mut clk,
    reset: *mut reset_control,
    dcbaa: xhci_device_context_array,
    interrupters: *mut *mut xhci_interrupter,
    cmd_ring: *mut xhci_ring,
    cmd_ring_state: u32,
    cmd_list: list_head,
    cmd_ring_reserved_trbs: u32,
    cmd_timer: delayed_work,
    cmd_ring_stop_completion: completion,
    current_cmd: *mut xhci_command,
    scratchpad: *mut xhci_scratchpad,
    mutex: mutex,
    devs: *mut *mut xhci_virt_device,
    rh_bw: *mut xhci_root_port_bw_info,
    device_pool: *mut dma_pool,
    segment_pool: *mut dma_pool,
    small_streams_pool: *mut dma_pool,
    port_bw_pool: *mut dma_pool,
    medium_streams_pool: *mut dma_pool,
    xhc_state: u32,
    run_graceperiod: u64,
    s3: s3_save,
    quirks: u64,
    num_active_eps: u32,
    limit_active_eps: u32,
    hw_ports: *mut xhci_port,
    usb2_rhub: xhci_hub,
    usb3_rhub: xhci_hub,
    __slate_bits_0: __slate_bits::__SlateBits830U0,
    port_caps: *mut xhci_port_cap,
    num_port_caps: u32,
    comp_mode_recovery_timer: timer_list,
    port_status_u0: u32,
    test_mode: u16,
    debugfs_root: *mut dentry,
    debugfs_slots: *mut dentry,
    regset_list: list_head,
    dbc: *mut (),
    r#priv: [u64; 0],
}


fn hcd_to_xhci(hcd: *mut usb_hcd) -> *mut xhci_hcd {
let mut primary_hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
if (unsafe { usb_hcd_is_primary_hcd(hcd) }) != 0 {
primary_hcd = hcd;
} else {
primary_hcd = unsafe { (*hcd).primary_hcd };
}
(unsafe { core::ptr::addr_of_mut!((*primary_hcd).hcd_priv) as *mut u64 }) as *mut xhci_hcd
}

fn xhci_to_hcd(xhci: *mut xhci_hcd) -> *mut usb_hcd {
unsafe { (*xhci).main_hcd }
}

fn xhci_get_usb3_hcd(xhci: *mut xhci_hcd) -> *mut usb_hcd {
if !unsafe { (*xhci).shared_hcd }.is_null() {
return unsafe { (*xhci).shared_hcd };
}
if (unsafe { (*xhci).usb2_rhub.num_ports }) == 0 {
return unsafe { (*xhci).main_hcd };
}
core::ptr::null_mut::<usb_hcd>()
}

fn xhci_read_64(xhci: *const xhci_hcd, regs: *mut u64) -> u64 {
lo_hi_readq(regs as *const ())
}

fn xhci_write_64(xhci: *mut xhci_hcd, val: u64, regs: *mut u64) {
lo_hi_writeq(val, regs as *mut ());
}

fn xhci_link_chain_quirk(xhci: *mut xhci_hcd, r#type: u32) -> bool {
(unsafe { (*xhci).quirks }) & BIT_ULL!(0i32) != 0 || (r#type as u32) == 1 && (unsafe { (*xhci).quirks }) & (BIT_ULL!(9i32) | BIT_ULL!(2i32)) != 0
}

#[unsafe(no_mangle)]
extern "C" fn xhci_endpoint_init(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, udev: *mut usb_device, ep: *mut usb_host_endpoint, mem_flags: u32) -> i32 {
let ep_index: u32 = unsafe { xhci_get_endpoint_index(core::ptr::addr_of_mut!((*ep).desc)) };
let mut max_packet: u32 = 0;
let mut ring_type: u32 = 0;
let mut max_esit_payload: u32 = 0;
let mut endpoint_type: u32 = 0;
let mut max_burst: u32 = 0;
let mut interval: u32 = 0;
let mut mult: u32 = 0;
let mut avg_trb_len: u32 = 0;
let mut err_count: u32 = 0;
let ep_ctx: *mut xhci_ep_ctx = xhci_get_ep_ctx(xhci, unsafe { (*virt_dev).in_ctx }, ep_index);
endpoint_type = xhci_get_endpoint_type(ep);
if endpoint_type == 0 {
return 22i32.wrapping_neg();
}
ring_type = usb_endpoint_type((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) as u32;
// Ensure host supports double isoc bandwidth for eUSB2 devices
if (unsafe { usb_endpoint_is_hs_isoc_double(udev, ep as *const usb_host_endpoint) }) && ((unsafe { (*xhci).hcc_params2 }) as u64) & BIT!(11i32) == 0 {
return 22i32.wrapping_neg();
}
// Get values to fill the endpoint context, mostly from ep descriptor.
// The average TRB buffer lengt for bulk endpoints is unclear as we
// have no clue on scatter gather list entry size. For Isoc and Int,
// set it to max available. See xHCI 1.1 spec 4.14.1.1 for details.
max_esit_payload = unsafe { usb_endpoint_max_periodic_payload(udev, ep as *const usb_host_endpoint) };
interval = xhci_get_endpoint_interval(udev, ep);
// Periodic endpoint bInterval limit quirk
let __v0: bool = usb_endpoint_xfer_int((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0;
if __v0 {
if (unsafe { (*xhci).quirks }) & BIT_ULL!(50i32) != 0 && interval >= 9 {
interval = 8;
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(26i32) != 0 && ((unsafe { (*udev).speed }) as u32) >= 3 && interval >= 7 {
interval = 6;
}
}
mult = xhci_get_endpoint_mult(xhci, udev, ep);
max_packet = (unsafe { xhci_usb_endpoint_maxp(udev, ep) }) as u32;
max_burst = xhci_get_endpoint_max_burst(udev, ep);
avg_trb_len = max_esit_payload;
// FIXME dig Mult and streams info out of ep companion desc
// Allow 3 retries for everything but isoc, set CErr = 3
if usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) == 0 {
err_count = 3;
}
// HS bulk max packet should be 512, FS bulk supports 8, 16, 32 or 64
if usb_endpoint_xfer_bulk((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 {
if ((unsafe { (*udev).speed }) as u32) == 3 {
max_packet = 512;
}
if ((unsafe { (*udev).speed }) as u32) == 2 {
max_packet = rounddown_pow_of_two!(max_packet) as u32;
let mut __v1: u32;
let __UNIQUE_ID_v__936: u32 = max_packet;
let __UNIQUE_ID_l__937: u32 = 8;
let __UNIQUE_ID_h__938: u32 = 64;
loop {
if !(!statically_true!(__UNIQUE_ID_l__937 > __UNIQUE_ID_h__938)) {
unsafe { __compiletime_assert_939() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
while false {
}
__v1 = if __UNIQUE_ID_v__936 >= __UNIQUE_ID_h__938 { __UNIQUE_ID_h__938 } else { if __UNIQUE_ID_v__936 <= __UNIQUE_ID_l__937 { __UNIQUE_ID_l__937 } else { __UNIQUE_ID_v__936 } };
max_packet = __v1;
}
}
// xHCI 1.0 and 1.1 indicates that ctrl ep avg TRB Length should be 8
if usb_endpoint_xfer_control((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 && ((unsafe { (*xhci).hci_version }) as i32) >= 256 {
avg_trb_len = 8;
}
// Set up the endpoint ring
unsafe {
(*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).new_ring = xhci_ring_alloc(xhci, 2u32, ring_type, max_packet, mem_flags);
}
if unsafe { (*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).new_ring }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).skip = false;
}
unsafe {
(*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).next_uframe = 1i32.wrapping_neg();
}
let ep_ring: *mut xhci_ring = unsafe { (*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).new_ring };
xhci_ring_init(xhci, ep_ring);
// Fill the endpoint context
unsafe {
(*ep_ctx).ep_info = EP_MAX_ESIT_PAYLOAD_HI!(max_esit_payload) | EP_INTERVAL!(interval) | EP_MULT!(mult);
}
unsafe {
(*ep_ctx).ep_info2 = EP_TYPE!(endpoint_type) | MAX_PACKET__2!(max_packet) | MAX_BURST__2!(max_burst) | ERROR_COUNT!(err_count);
}
unsafe {
(*ep_ctx).deq = (*(*ep_ring).first_seg).dma | ((*ep_ring).cycle_state as u64);
}
unsafe {
(*ep_ctx).tx_info = EP_MAX_ESIT_PAYLOAD_LO!(max_esit_payload) | EP_AVG_TRB_LENGTH!(avg_trb_len);
}
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
#[repr(C)]
#[derive(Clone, Copy)]
struct ktermios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 19],
    c_ispeed: u32,
    c_ospeed: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct winsize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
    kobj: kobject,
    owner: *mut module,
    ops: *const file_operations,
    list: list_head,
    dev: u32,
    count: u32,
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
struct tty_operations {
    lookup: Option<unsafe extern "C" fn(*mut tty_driver, *mut file, i32) -> *mut tty_struct>,
    install: Option<unsafe extern "C" fn(*mut tty_driver, *mut tty_struct) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut tty_driver, *mut tty_struct)>,
    open: Option<unsafe extern "C" fn(*mut tty_struct, *mut file) -> i32>,
    close: Option<unsafe extern "C" fn(*mut tty_struct, *mut file)>,
    shutdown: Option<unsafe extern "C" fn(*mut tty_struct)>,
    cleanup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    write: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, u64) -> i64>,
    put_char: Option<unsafe extern "C" fn(*mut tty_struct, u8) -> i32>,
    flush_chars: Option<unsafe extern "C" fn(*mut tty_struct)>,
    write_room: Option<unsafe extern "C" fn(*mut tty_struct) -> u32>,
    chars_in_buffer: Option<unsafe extern "C" fn(*mut tty_struct) -> u32>,
    ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i64>,
    set_termios: Option<unsafe extern "C" fn(*mut tty_struct, *const ktermios)>,
    throttle: Option<unsafe extern "C" fn(*mut tty_struct)>,
    unthrottle: Option<unsafe extern "C" fn(*mut tty_struct)>,
    stop: Option<unsafe extern "C" fn(*mut tty_struct)>,
    start: Option<unsafe extern "C" fn(*mut tty_struct)>,
    hangup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    break_ctl: Option<unsafe extern "C" fn(*mut tty_struct, i32) -> i32>,
    flush_buffer: Option<unsafe extern "C" fn(*mut tty_struct)>,
    ldisc_ok: Option<unsafe extern "C" fn(*mut tty_struct, i32) -> i32>,
    set_ldisc: Option<unsafe extern "C" fn(*mut tty_struct)>,
    wait_until_sent: Option<unsafe extern "C" fn(*mut tty_struct, i32)>,
    send_xchar: Option<unsafe extern "C" fn(*mut tty_struct, u8)>,
    tiocmget: Option<unsafe extern "C" fn(*mut tty_struct) -> i32>,
    tiocmset: Option<unsafe extern "C" fn(*mut tty_struct, u32, u32) -> i32>,
    resize: Option<unsafe extern "C" fn(*mut tty_struct, *mut winsize) -> i32>,
    get_icount: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_icounter_struct) -> i32>,
    get_serial: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_struct) -> i32>,
    set_serial: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_struct) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut tty_struct, *mut seq_file)>,
    proc_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_driver {
    kref: kref,
    cdevs: *mut *mut cdev,
    owner: *mut module,
    driver_name: *const u8,
    name: *const u8,
    name_base: i32,
    major: i32,
    minor_start: i32,
    num: u32,
    r#type: u32,
    subtype: u32,
    init_termios: ktermios,
    flags: u64,
    proc_entry: *mut proc_dir_entry,
    other: *mut tty_driver,
    flip_wq: *mut workqueue_struct,
    ttys: *mut *mut tty_struct,
    ports: *mut *mut tty_port,
    termios: *mut *mut ktermios,
    driver_state: *mut (),
    ops: *const tty_operations,
    tty_drivers: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ld_semaphore {
    count: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    wait_readers: u32,
    read_wait: list_head,
    write_wait: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_ldisc_ops {
    name: *mut u8,
    num: i32,
    open: Option<unsafe extern "C" fn(*mut tty_struct) -> i32>,
    close: Option<unsafe extern "C" fn(*mut tty_struct)>,
    flush_buffer: Option<unsafe extern "C" fn(*mut tty_struct)>,
    read: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *mut u8, u64, *mut *mut (), u64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *const u8, u64) -> i64>,
    ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    set_termios: Option<unsafe extern "C" fn(*mut tty_struct, *const ktermios)>,
    poll: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *mut poll_table_struct) -> u32>,
    hangup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    receive_buf: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64)>,
    write_wakeup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    dcd_change: Option<unsafe extern "C" fn(*mut tty_struct, bool)>,
    receive_buf2: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64) -> u64>,
    lookahead_buf: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64)>,
    owner: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_ldisc {
    ops: *const tty_ldisc_ops,
    tty: *mut tty_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kfifo {
    r#in: u32,
    out: u32,
    mask: u32,
    esize: u32,
    data: *mut (),
}


// DECLARE_KFIFO_PTR - macro to declare a fifo pointer object
// @fifo: name of the declared fifo
// @type: type of the fifo elements
#[repr(C)]
#[derive(Clone, Copy)]
struct tty_buffer {
    __slate_anon_0: __SlateRecord_tty_buffer_h_11,
    used: u32,
    size: u32,
    commit: u32,
    lookahead: u32,
    read: u32,
    flags: bool,
    __slate_pad_7: [u8; 3],
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_bufhead {
    head: *mut tty_buffer,
    flip_wq: *mut workqueue_struct,
    work: work_struct,
    lock: mutex,
    priority: __SlateRecord_types_h_190,
    sentinel: tty_buffer,
    free: llist_head,
    mem_used: __SlateRecord_types_h_190,
    mem_limit: i32,
    tail: *mut tty_buffer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port_operations {
    carrier_raised: Option<unsafe extern "C" fn(*mut tty_port) -> bool>,
    dtr_rts: Option<unsafe extern "C" fn(*mut tty_port, bool)>,
    shutdown: Option<unsafe extern "C" fn(*mut tty_port)>,
    activate: Option<unsafe extern "C" fn(*mut tty_port, *mut tty_struct) -> i32>,
    destruct: Option<unsafe extern "C" fn(*mut tty_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port_client_operations {
    receive_buf: Option<unsafe extern "C" fn(*mut tty_port, *const u8, *const u8, u64) -> u64>,
    lookahead_buf: Option<unsafe extern "C" fn(*mut tty_port, *const u8, *const u8, u64)>,
    write_wakeup: Option<unsafe extern "C" fn(*mut tty_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port {
    buf: tty_bufhead,
    tty: *mut tty_struct,
    itty: *mut tty_struct,
    ops: *const tty_port_operations,
    client_ops: *const tty_port_client_operations,
    lock: spinlock,
    blocked_open: i32,
    count: i32,
    open_wait: wait_queue_head,
    delta_msr_wait: wait_queue_head,
    flags: u64,
    iflags: u64,
    __slate_bits_0: __slate_bits::__SlateBits846U0,
    mutex: mutex,
    buf_mutex: mutex,
    xmit_buf: *mut u8,
    xmit_fifo: __SlateRecord_tty_port_h_118,
    close_delay: u32,
    closing_wait: u32,
    drain_delay: i32,
    kref: kref,
    client_data: *mut (),
}


// tty_port::iflags bits -- use atomic bit ops
// device is initialized
// device is suspended
// device is open
// uart drivers: use the uart_port::status field and the UPSTAT_* defines
// for s/w-based flow control steering and carrier detection status
// h/w flow control enabled
// carrier detect enabled
// device exclusively opened by
// kernel
#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
    kref: kref,
    index: i32,
    dev: *mut device,
    driver: *mut tty_driver,
    port: *mut tty_port,
    ops: *const tty_operations,
    ldisc: *mut tty_ldisc,
    ldisc_sem: ld_semaphore,
    atomic_write_lock: mutex,
    legacy_mutex: mutex,
    throttle_mutex: mutex,
    termios_rwsem: rw_semaphore,
    winsize_mutex: mutex,
    termios: ktermios,
    termios_locked: ktermios,
    name: [u8; 64],
    flags: u64,
    count: i32,
    receive_room: u32,
    winsize: winsize,
    flow: __SlateRecord_tty_h_212,
    ctrl: __SlateRecord_tty_h_218,
    hw_stopped: bool,
    closing: bool,
    flow_change: i32,
    link: *mut tty_struct,
    fasync: *mut fasync_struct,
    write_wait: wait_queue_head,
    read_wait: wait_queue_head,
    break_wait: wait_queue_head,
    hangup_work: work_struct,
    disc_data: *mut (),
    driver_data: *mut (),
    files_lock: spinlock,
    write_cnt: i32,
    write_buf: *mut u8,
    tty_files: list_head,
    SAK_work: work_struct,
}


fn __do_trace_xhci_dbg_context_change(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_535: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_535) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_536: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_536) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_538() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_context_change)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_537: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_537;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_context_change as *const ()) };
unsafe { __v12.unwrap()(__data, vaf) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_dbg_context_change: __SlateAlign16<[u8; 24]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 100u8, 98u8, 103u8, 95u8, 99u8, 111u8, 110u8, 116u8, 101u8, 120u8, 116u8, 95u8, 99u8, 104u8, 97u8, 110u8, 103u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_context_change_539: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_context_change) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci_mem.trace_xhci_dbg_context_change")]
extern "C" fn trace_xhci_dbg_context_change(vaf: *mut va_format) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_context_change.key) }) != 0 {
__do_trace_xhci_dbg_context_change(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_540: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_dbg_init(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_563: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_563) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_564: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_564) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_566() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_init)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_565: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_565;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_init as *const ()) };
unsafe { __v12.unwrap()(__data, vaf) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_dbg_init: [u8; 14] = [120, 104, 99, 105, 95, 100, 98, 103, 95, 105, 110, 105, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_init_567: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_init) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci_mem.trace_xhci_dbg_init")]
extern "C" fn trace_xhci_dbg_init(vaf: *mut va_format) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_init.key) }) != 0 {
__do_trace_xhci_dbg_init(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_568: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_dbg_ring_expansion(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_570: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_570) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_571: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_571) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_573() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_ring_expansion)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_572: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_572;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_ring_expansion as *const ()) };
unsafe { __v12.unwrap()(__data, vaf) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_dbg_ring_expansion: __SlateAlign16<[u8; 24]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 100u8, 98u8, 103u8, 95u8, 114u8, 105u8, 110u8, 103u8, 95u8, 101u8, 120u8, 112u8, 97u8, 110u8, 115u8, 105u8, 111u8, 110u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_ring_expansion_574: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_ring_expansion) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci_mem.trace_xhci_dbg_ring_expansion")]
extern "C" fn trace_xhci_dbg_ring_expansion(vaf: *mut va_format) {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_ring_expansion.key) }) != 0 {
__do_trace_xhci_dbg_ring_expansion(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_575: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_free_virt_device(vdev: *mut xhci_virt_device) {
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
let mut __UNIQUE_ID_guard_633: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_633) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_634: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_634) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_636() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_free_virt_device)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_635: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_635;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_virt_device) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_virt_device) -> i32>>(__SCT__tp_func_xhci_free_virt_device as *const ()) };
unsafe { __v12.unwrap()(__data, vdev) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_free_virt_device: __SlateAlign16<[u8; 22]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 102u8, 114u8, 101u8, 101u8, 95u8, 118u8, 105u8, 114u8, 116u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_free_virt_device_637: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_free_virt_device) }) as *mut ();

fn trace_xhci_free_virt_device(vdev: *mut xhci_virt_device) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_free_virt_device.key) }) != 0 {
__do_trace_xhci_free_virt_device(vdev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_638: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_alloc_virt_device(vdev: *mut xhci_virt_device) {
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
let mut __UNIQUE_ID_guard_640: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_640) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_641: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_641) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_643() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_alloc_virt_device)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_642: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_642;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_virt_device) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_virt_device) -> i32>>(__SCT__tp_func_xhci_alloc_virt_device as *const ()) };
unsafe { __v12.unwrap()(__data, vdev) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_alloc_virt_device: __SlateAlign16<[u8; 23]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 97u8, 108u8, 108u8, 111u8, 99u8, 95u8, 118u8, 105u8, 114u8, 116u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_alloc_virt_device_644: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_alloc_virt_device) }) as *mut ();

fn trace_xhci_alloc_virt_device(vdev: *mut xhci_virt_device) {
if static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_alloc_virt_device.key) }) != 0 {
__do_trace_xhci_alloc_virt_device(vdev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_645: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_setup_addressable_virt_device(vdev: *mut xhci_virt_device) {
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
let mut __UNIQUE_ID_guard_654: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_654) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_655: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_655) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_657() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_setup_addressable_virt_device)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_656: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_656;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_virt_device) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_virt_device) -> i32>>(__SCT__tp_func_xhci_setup_addressable_virt_device as *const ()) };
unsafe { __v12.unwrap()(__data, vdev) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_setup_addressable_virt_device: __SlateAlign16<[u8; 35]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 115u8, 101u8, 116u8, 117u8, 112u8, 95u8, 97u8, 100u8, 100u8, 114u8, 101u8, 115u8, 115u8, 97u8, 98u8, 108u8, 101u8, 95u8, 118u8, 105u8, 114u8, 116u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_setup_addressable_virt_device_658: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_setup_addressable_virt_device) }) as *mut ();

fn trace_xhci_setup_addressable_virt_device(vdev: *mut xhci_virt_device) {
if static_branch_unlikely__8!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_setup_addressable_virt_device.key) }) != 0 {
__do_trace_xhci_setup_addressable_virt_device(vdev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_659: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_alloc_stream_info_ctx(info: *mut xhci_stream_info, stream_id: u32) {
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
let mut __UNIQUE_ID_guard_689: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_689) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_690: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_690) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_692() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_alloc_stream_info_ctx)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_691: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_691;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_stream_info, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_stream_info, u32) -> i32>>(__SCT__tp_func_xhci_alloc_stream_info_ctx as *const ()) };
unsafe { __v12.unwrap()(__data, info, stream_id) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_alloc_stream_info_ctx: __SlateAlign16<[u8; 27]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 97u8, 108u8, 108u8, 111u8, 99u8, 95u8, 115u8, 116u8, 114u8, 101u8, 97u8, 109u8, 95u8, 105u8, 110u8, 102u8, 111u8, 95u8, 99u8, 116u8, 120u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_alloc_stream_info_ctx_693: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_alloc_stream_info_ctx) }) as *mut ();

fn trace_xhci_alloc_stream_info_ctx(info: *mut xhci_stream_info, stream_id: u32) {
if static_branch_unlikely__10!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_alloc_stream_info_ctx.key) }) != 0 {
__do_trace_xhci_alloc_stream_info_ctx(info, stream_id);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_694: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_ring_alloc(ring: *mut xhci_ring) {
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
let mut __UNIQUE_ID_guard_815: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_815) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_816: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_816) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_818() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_ring_alloc)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_817: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_817;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>>(__SCT__tp_func_xhci_ring_alloc as *const ()) };
unsafe { __v12.unwrap()(__data, ring) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_ring_alloc: __SlateAlign16<[u8; 16]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 114u8, 105u8, 110u8, 103u8, 95u8, 97u8, 108u8, 108u8, 111u8, 99u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_ring_alloc_819: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_ring_alloc) }) as *mut ();

fn trace_xhci_ring_alloc(ring: *mut xhci_ring) {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_ring_alloc.key) }) != 0 {
__do_trace_xhci_ring_alloc(ring);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_820: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_ring_free(ring: *mut xhci_ring) {
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
let mut __UNIQUE_ID_guard_822: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_822) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_823: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_823) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_825() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_ring_free)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_824: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_824;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>>(__SCT__tp_func_xhci_ring_free as *const ()) };
unsafe { __v12.unwrap()(__data, ring) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_ring_free: [u8; 15] = [120, 104, 99, 105, 95, 114, 105, 110, 103, 95, 102, 114, 101, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_ring_free_826: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_ring_free) }) as *mut ();

fn trace_xhci_ring_free(ring: *mut xhci_ring) {
if static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_ring_free.key) }) != 0 {
__do_trace_xhci_ring_free(ring);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_827: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_ring_expansion(ring: *mut xhci_ring) {
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
let mut __UNIQUE_ID_guard_829: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_829) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_830: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_830) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_832() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_ring_expansion)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_831: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_831;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>>(__SCT__tp_func_xhci_ring_expansion as *const ()) };
unsafe { __v12.unwrap()(__data, ring) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_ring_expansion: __SlateAlign16<[u8; 20]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 114u8, 105u8, 110u8, 103u8, 95u8, 101u8, 120u8, 112u8, 97u8, 110u8, 115u8, 105u8, 111u8, 110u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_ring_expansion_833: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_ring_expansion) }) as *mut ();

fn trace_xhci_ring_expansion(ring: *mut xhci_ring) {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_ring_expansion.key) }) != 0 {
__do_trace_xhci_ring_expansion(ring);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_834: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// Allocates a generic ring segment from the ring pool, sets the dma address,
/// initializes the segment to zero, and sets the private next pointer to NULL.
///
/// Section 4.11.1.1:
/// "All components of all Command and Transfer TRBs shall be initialized to '0'"
fn xhci_segment_alloc(xhci: *mut xhci_hcd, max_packet: u32, num: u32, flags: u32) -> *mut xhci_segment {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut dma: u64 = 0;
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let mut __v0: *mut ();
let mut __v1: *mut ();
{
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_segment>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_segment>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
}
__v0 = __v1;
let seg: *mut xhci_segment = __v0 as *mut xhci_segment;
if seg.is_null() {
return core::ptr::null_mut::<xhci_segment>();
}
unsafe {
(*seg).trbs = dma_pool_zalloc((*xhci).segment_pool, flags, core::ptr::addr_of_mut!(dma)) as *mut xhci_trb;
}
if unsafe { (*seg).trbs }.is_null() {
unsafe { kfree(seg as *const ()) };
return core::ptr::null_mut::<xhci_segment>();
}
if max_packet != 0 {
let mut __v5: *mut ();
let mut __v6: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_node_noprof(max_packet as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_node_noprof(max_packet as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
}
__v6 = _res;
__v5 = __v6;
unsafe {
(*seg).bounce_buf = __v5;
}
if unsafe { (*seg).bounce_buf }.is_null() {
unsafe { dma_pool_free((*xhci).segment_pool, (*seg).trbs as *mut (), dma) };
unsafe { kfree(seg as *const ()) };
return core::ptr::null_mut::<xhci_segment>();
}
}
unsafe {
(*seg).num = num;
}
unsafe {
(*seg).dma = dma;
}
unsafe {
(*seg).next = core::ptr::null_mut::<xhci_segment>();
}
seg
}

fn xhci_segment_free(xhci: *mut xhci_hcd, seg: *mut xhci_segment) {
if !unsafe { (*seg).trbs }.is_null() {
unsafe { dma_pool_free((*xhci).segment_pool, (*seg).trbs as *mut (), (*seg).dma) };
unsafe {
(*seg).trbs = core::ptr::null_mut::<xhci_trb>();
}
}
unsafe { kfree((*seg).bounce_buf as *const ()) };
unsafe { kfree(seg as *const ()) };
}

fn xhci_ring_segments_free(xhci: *mut xhci_hcd, ring: *mut xhci_ring) {
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
unsafe {
(*(*ring).last_seg).next = core::ptr::null_mut::<xhci_segment>();
}
seg = unsafe { (*ring).first_seg };
while !seg.is_null() {
let next: *mut xhci_segment = unsafe { (*seg).next };
xhci_segment_free(xhci, seg);
seg = next;
}
}

/// Only for transfer and command rings where driver is the producer, not for
/// event rings.
///
/// Change the last TRB in the segment to be a Link TRB which points to the
/// DMA address of the next segment.  The caller needs to set any Link TRB
/// related flags, such as End TRB, Toggle Cycle, and no snoop.
fn xhci_set_link_trb(seg: *mut xhci_segment, chain_links: bool) {
let mut val: u32 = 0;
if seg.is_null() || unsafe { (*seg).next }.is_null() {
return;
}
let trb: *mut xhci_trb = unsafe { (*seg).trbs }.wrapping_offset(255i32 as isize);
// Set the last TRB in the segment to have a TRB type ID of Link TRB
val = unsafe { (*trb).link.control };
let __v1: u32 = val & 4294902783u32;
val = __v1;
let __v2: u32 = val;
val = __v2 | (TRB_TYPE!(6i32) as u32);
if chain_links {
let __v4: u32 = val;
val = ((__v4 as u64) | BIT!(4i32)) as u32;
}
unsafe {
(*trb).link.control = val;
}
unsafe {
(*trb).link.segment_ptr = (*(*seg).next).dma;
}
}

fn xhci_initialize_ring_segments(xhci: *mut xhci_hcd, ring: *mut xhci_ring) {
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
let mut chain_links: bool = false;
if ((unsafe { (*ring).r#type }) as u32) == 6 {
return;
}
chain_links = xhci_link_chain_quirk(xhci, unsafe { (*ring).r#type });
seg = unsafe { (*ring).first_seg };
while !seg.is_null() {
xhci_set_link_trb(seg, chain_links);
seg = if (unsafe { (*seg).next }) != unsafe { (*ring).first_seg } { unsafe { (*seg).next } } else { core::ptr::null_mut::<xhci_segment>() };
}
// See section 4.9.2.1 and 6.4.4.1
let __v1: *mut xhci_trb = unsafe { (*(*ring).last_seg).trbs }.wrapping_offset(255i32 as isize);
let __v2: u32 = unsafe { (*__v1).link.control };
let __v3: u32 = __v2 | (BIT!(1i32) as u32);
unsafe {
(*__v1).link.control = __v3;
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_ring_init(xhci: *mut xhci_hcd, ring: *mut xhci_ring) {
xhci_initialize_ring_segments(xhci, ring);
xhci_initialize_ring_info(ring);
trace_xhci_ring_alloc(ring);
}

/// Link the src ring segments to the dst ring.
/// Set Toggle Cycle for the new ring if needed.
fn xhci_link_rings(xhci: *mut xhci_hcd, src: *mut xhci_ring, dst: *mut xhci_ring) {
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
let mut chain_links: bool = false;
if src.is_null() || dst.is_null() {
return;
}
// If the cycle state is 0, set the cycle bit to 1 for all the TRBs
if (unsafe { (*dst).cycle_state }) == 0 {
seg = unsafe { (*src).first_seg };
while !seg.is_null() {
for i in 0i32..256i32 {
let __v0: *mut xhci_trb = unsafe { (*seg).trbs }.wrapping_offset(i as isize);
let __v1: u32 = unsafe { (*__v0).link.control };
let __v2: u32 = __v1 | (BIT!(0i32) as u32);
unsafe {
(*__v0).link.control = __v2;
}
}
seg = if (unsafe { (*seg).next }) != unsafe { (*src).first_seg } { unsafe { (*seg).next } } else { core::ptr::null_mut::<xhci_segment>() };
}
}
unsafe {
(*(*src).last_seg).next = (*(*dst).enq_seg).next;
}
unsafe {
(*(*dst).enq_seg).next = (*src).first_seg;
}
if ((unsafe { (*dst).r#type }) as u32) != 6 {
chain_links = xhci_link_chain_quirk(xhci, unsafe { (*dst).r#type });
xhci_set_link_trb(unsafe { (*dst).enq_seg }, chain_links);
xhci_set_link_trb(unsafe { (*src).last_seg }, chain_links);
}
let __v6: u32 = unsafe { (*dst).num_segs };
unsafe {
(*dst).num_segs = __v6.wrapping_add((*src).num_segs);
}
if (unsafe { (*dst).enq_seg }) == unsafe { (*dst).last_seg } {
if ((unsafe { (*dst).r#type }) as u32) != 6 {
let __v8: *mut xhci_trb = unsafe { (*(*dst).last_seg).trbs }.wrapping_offset(255i32 as isize);
let __v9: u32 = unsafe { (*__v8).link.control };
let __v10: u32 = __v9 & !(BIT!(1i32) as u32);
unsafe {
(*__v8).link.control = __v10;
}
}
unsafe {
(*dst).last_seg = (*src).last_seg;
}
} else {
if ((unsafe { (*dst).r#type }) as u32) != 6 {
let __v11: *mut xhci_trb = unsafe { (*(*src).last_seg).trbs }.wrapping_offset(255i32 as isize);
let __v12: u32 = unsafe { (*__v11).link.control };
let __v13: u32 = __v12 & !(BIT!(1i32) as u32);
unsafe {
(*__v11).link.control = __v13;
}
}
}
seg = unsafe { (*dst).enq_seg };
while seg != unsafe { (*dst).last_seg } {
unsafe {
(*(*seg).next).num = unsafe { (*seg).num }.wrapping_add(1);
}
seg = unsafe { (*seg).next };
}
}

/// We need a radix tree for mapping physical addresses of TRBs to which stream
/// ID they belong to.  We need to do this because the host controller won't tell
/// us which stream ring the TRB came from.  We could store the stream ID in an
/// event data TRB, but that doesn't help us for the cancellation case, since the
/// endpoint may stop before it reaches that event data TRB.
///
/// The radix tree maps the upper portion of the TRB DMA address to a ring
/// segment that has the same upper portion of DMA addresses.  For example, say I
/// have segments of size 1KB, that are always 1KB aligned.  A segment may
/// start at 0x10c91000 and end at 0x10c913f0.  If I use the upper 10 bits, the
/// key to the stream ID is 0x43244.  I can use the DMA address of the TRB to
/// pass the radix tree a key to get the right stream ID:
///
/// 	0x10c90fff >> 10 = 0x43243
/// 	0x10c912c0 >> 10 = 0x43244
/// 	0x10c91400 >> 10 = 0x43245
///
/// Obviously, only those TRBs with DMA addresses that are within the segment
/// will make the radix tree return the stream ID for that ring.
///
/// Caveats for the radix tree:
///
/// The radix tree uses an unsigned long as a key pair.  On 32-bit systems, an
/// unsigned long will be 32-bits; on a 64-bit system an unsigned long will be
/// 64-bits.  Since we only request 32-bit DMA addresses, we can use that as the
/// key on 32-bit or 64-bit systems (it would also be fine if we asked for 64-bit
/// PCI DMA addresses on a 64-bit system).  There might be a problem on 32-bit
/// extended systems (where the DMA address can be bigger than 32-bits),
/// if we allow the PCI dma mask to be bigger than 32-bits.  So don't do that.
fn xhci_insert_segment_mapping(trb_address_map: *mut xarray, ring: *mut xhci_ring, seg: *mut xhci_segment, mem_flags: u32) -> i32 {
let key: u64 = (unsafe { (*seg).dma }) >> ilog2!(4096i32);
let mut ret: i32 = 0;
// Skip any segments that were already added.
if !unsafe { radix_tree_lookup(trb_address_map as *const xarray, key) }.is_null() {
return 0;
}
ret = unsafe { radix_tree_maybe_preload(mem_flags) };
if ret != 0 {
return ret;
}
ret = unsafe { radix_tree_insert(trb_address_map, key, ring as *mut ()) };
radix_tree_preload_end();
ret
}

fn xhci_remove_segment_mapping(trb_address_map: *mut xarray, seg: *mut xhci_segment) {
let key: u64 = (unsafe { (*seg).dma }) >> ilog2!(4096i32);
if !unsafe { radix_tree_lookup(trb_address_map as *const xarray, key) }.is_null() {
unsafe { radix_tree_delete(trb_address_map, key) };
}
}

fn xhci_update_stream_segment_mapping(trb_address_map: *mut xarray, ring: *mut xhci_ring, first_seg: *mut xhci_segment, mem_flags: u32) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut failed_seg: core::mem::MaybeUninit<*mut xhci_segment> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut xhci_segment = core::ptr::addr_of_mut!(failed_seg) as *mut *mut xhci_segment;
let mut seg: core::mem::MaybeUninit<*mut xhci_segment> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut xhci_segment = core::ptr::addr_of_mut!(seg) as *mut *mut xhci_segment;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__ret_warn_on = trb_address_map.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("919: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 919b - ., 3; .popsection\n# {0}", const 919 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_xhci_update_stream_segment_mapping_6, sym __slate_asm_string_xhci_update_stream_segment_mapping_7, const 256 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("920: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 920b - ., 4; .popsection\n# {0}", const 920 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return 0;
} else {
*__slate_slot_5 = first_seg;
'__join_5: {
while !(*__slate_slot_5).is_null() {
*__slate_slot_3 = xhci_insert_segment_mapping(trb_address_map, ring, *__slate_slot_5, mem_flags);
if *__slate_slot_3 != 0 {
break '__join_5;
} else {
*__slate_slot_5 = if (*(*__slate_slot_5)).next != first_seg { (*(*__slate_slot_5)).next } else { core::ptr::null_mut::<xhci_segment>() };
}
}
return 0;
}
*__slate_slot_4 = *__slate_slot_5;
*__slate_slot_5 = first_seg;
while !(*__slate_slot_5).is_null() {
xhci_remove_segment_mapping(trb_address_map, *__slate_slot_5);
if *__slate_slot_5 == *__slate_slot_4 {
return *__slate_slot_3;
} else {
*__slate_slot_5 = if (*(*__slate_slot_5)).next != first_seg { (*(*__slate_slot_5)).next } else { core::ptr::null_mut::<xhci_segment>() };
}
}
return *__slate_slot_3;
}
}
}

fn xhci_remove_stream_mapping(ring: *mut xhci_ring) {
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
let mut __v0: i64;
let __ret_warn_on: i32 = unsafe { (*ring).trb_address_map }.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("921: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 921b - ., 3; .popsection\n# {0}", const 921 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_xhci_remove_stream_mapping_1, sym __slate_asm_string_xhci_remove_stream_mapping_2, const 283 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("922: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 922b - ., 4; .popsection\n# {0}", const 922 as i32, options(nostack, att_syntax));
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
seg = unsafe { (*ring).first_seg };
while !seg.is_null() {
xhci_remove_segment_mapping(unsafe { (*ring).trb_address_map }, seg);
seg = if (unsafe { (*seg).next }) != unsafe { (*ring).first_seg } { unsafe { (*seg).next } } else { core::ptr::null_mut::<xhci_segment>() };
}
}

fn xhci_update_stream_mapping(ring: *mut xhci_ring, mem_flags: u32) -> i32 {
xhci_update_stream_segment_mapping(unsafe { (*ring).trb_address_map }, ring, unsafe { (*ring).first_seg }, mem_flags)
}

/// XXX: Do we need the hcd structure in all these functions?
#[unsafe(no_mangle)]
extern "C" fn xhci_ring_free(xhci: *mut xhci_hcd, ring: *mut xhci_ring) {
if ring.is_null() {
return;
}
trace_xhci_ring_free(ring);
if !unsafe { (*ring).first_seg }.is_null() {
if ((unsafe { (*ring).r#type }) as u32) == 4 {
xhci_remove_stream_mapping(ring);
}
xhci_ring_segments_free(xhci, ring);
}
unsafe { kfree(ring as *const ()) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci_mem.xhci_initialize_ring_info")]
extern "C" fn xhci_initialize_ring_info(ring: *mut xhci_ring) {
// The ring is empty, so the enqueue pointer == dequeue pointer
unsafe {
(*ring).enqueue = (*(*ring).first_seg).trbs;
}
unsafe {
(*ring).enq_seg = (*ring).first_seg;
}
unsafe {
(*ring).dequeue = (*ring).enqueue;
}
unsafe {
(*ring).deq_seg = (*ring).first_seg;
}
// The ring is initialized to 0. The producer must write 1 to the cycle
// bit to handover ownership of the TRB, so PCS = 1.  The consumer must
// compare CCS to the cycle bit to check ownership, so CCS = 1.
//
// New rings are initialized with cycle state equal to 1; if we are
// handling ring expansion, set the cycle state equal to the old ring.
unsafe {
(*ring).cycle_state = 1;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_initialize_ring_info_923: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xhci_ring)>, *mut ()>(Some(xhci_initialize_ring_info)) };

/// Allocate segments and link them for a ring
fn xhci_alloc_segments_for_ring(xhci: *mut xhci_hcd, ring: *mut xhci_ring, flags: u32) -> i32 {
let mut next: core::mem::MaybeUninit<*mut xhci_segment> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut xhci_segment = core::ptr::addr_of_mut!(next) as *mut *mut xhci_segment;
let mut num: u32 = unsafe { core::mem::zeroed() };
let mut prev: core::mem::MaybeUninit<*mut xhci_segment> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut xhci_segment = core::ptr::addr_of_mut!(prev) as *mut *mut xhci_segment;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
num = 0;
*__slate_slot_6 = xhci_segment_alloc(xhci, (*ring).bounce_buf_len, num, flags);
if (*__slate_slot_6).is_null() {
return 12i32.wrapping_neg();
} else {
num = num.wrapping_add(1);
(*ring).first_seg = *__slate_slot_6;
while num < (*ring).num_segs {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = xhci_segment_alloc(xhci, (*ring).bounce_buf_len, num, flags);
if (*__slate_slot_2).is_null() {
(*ring).last_seg = *__slate_slot_6;
xhci_ring_segments_free(xhci, ring);
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_6)).next = *__slate_slot_2;
*__slate_slot_6 = *__slate_slot_2;
num = num.wrapping_add(1);
}
}
(*ring).last_seg = *__slate_slot_6;
(*(*ring).last_seg).next = (*ring).first_seg;
return 0;
}
}
}

/// Create a new ring with zero or more segments.
///
/// Link each segment together into a ring.
/// Set the end flag and the cycle toggle bit on the last segment.
/// See section 4.9.1 and figures 15 and 16.
#[unsafe(no_mangle)]
extern "C" fn xhci_ring_alloc(xhci: *mut xhci_hcd, num_segs: u32, r#type: u32, max_packet: u32, flags: u32) -> *mut xhci_ring {
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut dev: *mut device = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut ring: core::mem::MaybeUninit<*mut xhci_ring> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut xhci_ring = core::ptr::addr_of_mut!(ring) as *mut *mut xhci_ring;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
dev = (*xhci_to_hcd(xhci)).self_.sysdev;
let __slate_auto_bytes_3 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kmalloc_node_noprof(core::mem::size_of::<xhci_ring>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kmalloc_node_noprof(core::mem::size_of::<xhci_ring>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
});
}
*__slate_slot_6 = *__slate_slot_1 as *mut xhci_ring;
if (*__slate_slot_6).is_null() {
return core::ptr::null_mut::<xhci_ring>();
} else {
(*(*__slate_slot_6)).num_segs = num_segs;
(*(*__slate_slot_6)).bounce_buf_len = max_packet;
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_6)).td_list));
(*(*__slate_slot_6)).r#type = r#type;
if num_segs == 0 {
return *__slate_slot_6;
} else {
*__slate_slot_5 = xhci_alloc_segments_for_ring(xhci, *__slate_slot_6, flags);
if *__slate_slot_5 != 0 {
kfree(*__slate_slot_6 as *const ());
return core::ptr::null_mut::<xhci_ring>();
} else {
return *__slate_slot_6;
}
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_free_endpoint_ring(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, ep_index: u32) {
xhci_ring_free(xhci, unsafe { (*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ring });
unsafe {
(*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ring = core::ptr::null_mut::<xhci_ring>();
}
}

/// Expand an existing ring.
/// Allocate a new ring which has same segment numbers and link the two rings.
#[unsafe(no_mangle)]
extern "C" fn xhci_ring_expansion(xhci: *mut xhci_hcd, ring: *mut xhci_ring, num_new_segs: u32, flags: u32) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut new_ring: core::mem::MaybeUninit<xhci_ring> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut xhci_ring = core::ptr::addr_of_mut!(new_ring) as *mut xhci_ring;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 96usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if num_new_segs == 0 {
return 0;
} else {
(*__slate_slot_1).num_segs = num_new_segs;
(*__slate_slot_1).bounce_buf_len = (*ring).bounce_buf_len;
(*__slate_slot_1).r#type = (*ring).r#type;
*__slate_slot_0 = xhci_alloc_segments_for_ring(xhci, core::ptr::addr_of_mut!(*__slate_slot_1), flags);
if *__slate_slot_0 != 0 {
return 12i32.wrapping_neg();
} else {
xhci_initialize_ring_segments(xhci, core::ptr::addr_of_mut!(*__slate_slot_1));
if ((*ring).r#type as u32) == 4 {
*__slate_slot_0 = xhci_update_stream_segment_mapping((*ring).trb_address_map, ring, (*__slate_slot_1).first_seg, flags);
if *__slate_slot_0 != 0 {
xhci_ring_segments_free(xhci, core::ptr::addr_of_mut!(*__slate_slot_1));
return *__slate_slot_0;
}
}
xhci_link_rings(xhci, core::ptr::addr_of_mut!(*__slate_slot_1), ring);
trace_xhci_ring_expansion(ring);
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_ring_expansion), c"ring expansion succeed, now has %d segments".as_ptr() as *const u8, (*ring).num_segs);
return 0;
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_alloc_container_ctx(xhci: *mut xhci_hcd, r#type: i32, flags: u32) -> *mut xhci_container_ctx {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut ctx: *mut xhci_container_ctx = unsafe { core::mem::zeroed() };
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
if r#type != 1 && r#type != 2 {
return core::ptr::null_mut::<xhci_container_ctx>();
}
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_container_ctx>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_container_ctx>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
ctx = __v0 as *mut xhci_container_ctx;
if ctx.is_null() {
return core::ptr::null_mut::<xhci_container_ctx>();
}
unsafe {
(*ctx).r#type = r#type as u32;
}
unsafe {
(*ctx).size = if ((*xhci).hcc_params as u64) & BIT!(2i32) != 0 { 2048 } else { 1024 };
}
if r#type == 2 {
let __v6: i32 = unsafe { (*ctx).size };
let __v7: i32 = __v6.wrapping_add(CTX_SIZE!(unsafe { (*xhci).hcc_params }));
unsafe {
(*ctx).size = __v7;
}
}
unsafe {
(*ctx).bytes = dma_pool_zalloc((*xhci).device_pool, flags, core::ptr::addr_of_mut!((*ctx).dma)) as *mut u8;
}
if unsafe { (*ctx).bytes }.is_null() {
unsafe { kfree(ctx as *const ()) };
return core::ptr::null_mut::<xhci_container_ctx>();
}
ctx
}

#[unsafe(no_mangle)]
extern "C" fn xhci_free_container_ctx(xhci: *mut xhci_hcd, ctx: *mut xhci_container_ctx) {
if ctx.is_null() {
return;
}
unsafe { dma_pool_free((*xhci).device_pool, (*ctx).bytes as *mut (), (*ctx).dma) };
unsafe { kfree(ctx as *const ()) };
}

#[unsafe(no_mangle)]
extern "C" fn xhci_alloc_port_bw_ctx(xhci: *mut xhci_hcd, flags: u32) -> *mut xhci_container_ctx {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut ctx: *mut xhci_container_ctx = unsafe { core::mem::zeroed() };
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_container_ctx>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_container_ctx>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
ctx = __v0 as *mut xhci_container_ctx;
if ctx.is_null() {
return core::ptr::null_mut::<xhci_container_ctx>();
}
unsafe {
(*ctx).size = 256;
}
unsafe {
(*ctx).bytes = dma_pool_zalloc((*xhci).port_bw_pool, flags, core::ptr::addr_of_mut!((*ctx).dma)) as *mut u8;
}
if unsafe { (*ctx).bytes }.is_null() {
unsafe { kfree(ctx as *const ()) };
return core::ptr::null_mut::<xhci_container_ctx>();
}
ctx
}

#[unsafe(no_mangle)]
extern "C" fn xhci_free_port_bw_ctx(xhci: *mut xhci_hcd, ctx: *mut xhci_container_ctx) {
if ctx.is_null() {
return;
}
unsafe { dma_pool_free((*xhci).port_bw_pool, (*ctx).bytes as *mut (), (*ctx).dma) };
unsafe { kfree(ctx as *const ()) };
}

#[unsafe(no_mangle)]
extern "C" fn xhci_get_input_control_ctx(ctx: *mut xhci_container_ctx) -> *mut xhci_input_control_ctx {
if (unsafe { (*ctx).r#type }) != 2 {
return core::ptr::null_mut::<xhci_input_control_ctx>();
}
(unsafe { (*ctx).bytes }) as *mut xhci_input_control_ctx
}

#[unsafe(no_mangle)]
extern "C" fn xhci_get_slot_ctx(xhci: *mut xhci_hcd, ctx: *mut xhci_container_ctx) -> *mut xhci_slot_ctx {
if (unsafe { (*ctx).r#type }) == 1 {
return (unsafe { (*ctx).bytes }) as *mut xhci_slot_ctx;
}
unsafe { (*ctx).bytes }.wrapping_offset(CTX_SIZE!(unsafe { (*xhci).hcc_params }) as isize) as *mut xhci_slot_ctx
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci_mem.xhci_get_ep_ctx")]
extern "C" fn xhci_get_ep_ctx(xhci: *mut xhci_hcd, ctx: *mut xhci_container_ctx, mut ep_index: u32) -> *mut xhci_ep_ctx {
// increment ep index by offset of start of ep ctx array
ep_index = ep_index.wrapping_add(1);
if (unsafe { (*ctx).r#type }) == 2 {
ep_index = ep_index.wrapping_add(1);
}
unsafe { (*ctx).bytes }.wrapping_offset(ep_index.wrapping_mul(CTX_SIZE!(unsafe { (*xhci).hcc_params }) as u32) as isize) as *mut xhci_ep_ctx
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_get_ep_ctx_924: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xhci_hcd, *mut xhci_container_ctx, u32) -> *mut xhci_ep_ctx>, *mut ()>(Some(xhci_get_ep_ctx)) };

// Streams structures manipulation
fn xhci_free_stream_ctx(xhci: *mut xhci_hcd, num_stream_ctxs: u32, stream_ctx: *mut xhci_stream_ctx, dma: u64) {
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let size: u64 = array_size!(core::mem::size_of::<xhci_stream_ctx>() as u64, num_stream_ctxs);
if size > 1024 {
dma_free_coherent(dev, size, stream_ctx as *mut (), dma);
} else {
if size > 256 {
unsafe { dma_pool_free((*xhci).medium_streams_pool, stream_ctx as *mut (), dma) };
} else {
unsafe { dma_pool_free((*xhci).small_streams_pool, stream_ctx as *mut (), dma) };
}
}
}

/// The stream context array for each endpoint with bulk streams enabled can
/// vary in size, based on:
///  - how many streams the endpoint supports,
///  - the maximum primary stream array size the host controller supports,
///  - and how many streams the device driver asks for.
///
/// The stream context array must be a power of 2, and can be as small as
/// 64 bytes or as large as 1MB.
fn xhci_alloc_stream_ctx(xhci: *mut xhci_hcd, num_stream_ctxs: u32, dma: *mut u64, mem_flags: u32) -> *mut xhci_stream_ctx {
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let size: u64 = array_size!(core::mem::size_of::<xhci_stream_ctx>() as u64, num_stream_ctxs);
if size > 1024 {
return dma_alloc_coherent(dev, size, dma, mem_flags) as *mut xhci_stream_ctx;
}
if size > 256 {
return dma_pool_zalloc(unsafe { (*xhci).medium_streams_pool }, mem_flags, dma) as *mut xhci_stream_ctx;
} else {
return dma_pool_zalloc(unsafe { (*xhci).small_streams_pool }, mem_flags, dma) as *mut xhci_stream_ctx;
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_dma_to_transfer_ring(ep: *mut xhci_virt_ep, address: u64) -> *mut xhci_ring {
if ((unsafe { (*ep).ep_state }) as u64) & BIT!(4i32) != 0 {
return (unsafe { radix_tree_lookup(core::ptr::addr_of_mut!((*(*ep).stream_info).trb_address_map) as *const xarray, address >> ilog2!(4096i32)) }) as *mut xhci_ring;
}
unsafe { (*ep).ring }
}

/// Change an endpoint's internal structure so it supports stream IDs.  The
/// number of requested streams includes stream 0, which cannot be used by device
/// drivers.
///
/// The number of stream contexts in the stream context array may be bigger than
/// the number of streams the driver wants to use.  This is because the number of
/// stream context array entries must be a power of two.
#[unsafe(no_mangle)]
extern "C" fn xhci_alloc_stream_info(xhci: *mut xhci_hcd, num_stream_ctxs: u32, num_streams: u32, max_packet: u32, mem_flags: u32) -> *mut xhci_stream_info {
let mut __slate_compound_24: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_27: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_29: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut dev: *mut device = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u64 = core::ptr::addr_of_mut!(addr) as *mut u64;
let mut cur_ring: core::mem::MaybeUninit<*mut xhci_ring> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut xhci_ring = core::ptr::addr_of_mut!(cur_ring) as *mut *mut xhci_ring;
let mut cur_stream: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(cur_stream) as *mut u32;
let mut stream_info: core::mem::MaybeUninit<*mut xhci_stream_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut xhci_stream_info = core::ptr::addr_of_mut!(stream_info) as *mut *mut xhci_stream_info;
unsafe {
let __slate_auto_bytes_3 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
dev = (*xhci_to_hcd(xhci)).self_.sysdev;
if (*xhci).cmd_ring_reserved_trbs == 253u32 {
return core::ptr::null_mut::<xhci_stream_info>();
} else {
let __v16: u32 = (*xhci).cmd_ring_reserved_trbs;
(*xhci).cmd_ring_reserved_trbs = __v16.wrapping_add(1);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_11 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_12 = _kmalloc_node_noprof(core::mem::size_of::<xhci_stream_info>() as u64, mem_flags | 256u32, dev_to_node(dev), *{
__slate_compound_27 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_27)
});
alloc_tag_restore!();
} else {
*__slate_slot_12 = _kmalloc_node_noprof(core::mem::size_of::<xhci_stream_info>() as u64, mem_flags | 256u32, dev_to_node(dev), *{
__slate_compound_29 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_29)
});
}
*__slate_slot_23 = *__slate_slot_12 as *mut xhci_stream_info;
if !(*__slate_slot_23).is_null() {
(*(*__slate_slot_23)).num_streams = num_streams;
(*(*__slate_slot_23)).num_stream_ctxs = num_stream_ctxs;
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_7 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_8 = _kmalloc_array_node_noprof(num_streams as u64, core::mem::size_of::<*mut xhci_ring>() as u64, mem_flags | 256u32, dev_to_node(dev), *{
__slate_compound_24 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_24)
});
alloc_tag_restore!();
} else {
*__slate_slot_8 = _kmalloc_array_node_noprof(num_streams as u64, core::mem::size_of::<*mut xhci_ring>() as u64, mem_flags | 256u32, dev_to_node(dev), *{
__slate_compound_26 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_26)
});
}
let __v9: *mut () = *__slate_slot_8;
// Initialize the array of virtual pointers to stream rings.
(*(*__slate_slot_23)).stream_rings = __v9 as *mut *mut xhci_ring;
if !unsafe { (*(*__slate_slot_23)).stream_rings }.is_null() {
// Initialize the array of DMA addresses for stream rings for the HW.
(*(*__slate_slot_23)).stream_ctx_array = xhci_alloc_stream_ctx(xhci, num_stream_ctxs, core::ptr::addr_of_mut!((*(*__slate_slot_23)).ctx_array_dma), mem_flags);
if !unsafe { (*(*__slate_slot_23)).stream_ctx_array }.is_null() {
// Allocate everything needed to free the stream rings later
(*(*__slate_slot_23)).free_streams_command = xhci_alloc_command_with_ctx(xhci, true, mem_flags);
if !unsafe { (*(*__slate_slot_23)).free_streams_command }.is_null() {
INIT_RADIX_TREE!(core::ptr::addr_of_mut!((*(*__slate_slot_23)).trb_address_map), (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
// Allocate rings for all the streams that the driver will use,
// and add their segment DMA addresses to the radix tree.
// Stream 0 is reserved.
*__slate_slot_22 = 1;
'__join_9: {
'__join_13: {
while *__slate_slot_22 < num_streams {
*unsafe { (*(*__slate_slot_23)).stream_rings }.wrapping_offset(*__slate_slot_22 as isize) = xhci_ring_alloc(xhci, 2u32, 4u32, max_packet, mem_flags);
*__slate_slot_21 = *unsafe { (*(*__slate_slot_23)).stream_rings }.wrapping_offset(*__slate_slot_22 as isize);
if (*__slate_slot_21).is_null() {
break '__join_9;
} else {
xhci_ring_init(xhci, *__slate_slot_21);
(*(*__slate_slot_21)).stream_id = *__slate_slot_22;
(*(*__slate_slot_21)).trb_address_map = core::ptr::addr_of_mut!((*(*__slate_slot_23)).trb_address_map);
// Set deq ptr, cycle bit, and stream context type
*__slate_slot_20 = (*(*(*__slate_slot_21)).first_seg).dma | (SCT_FOR_CTX!(1i32) as u64) | ((*(*__slate_slot_21)).cycle_state as u64);
(*unsafe { (*(*__slate_slot_23)).stream_ctx_array }.wrapping_offset(*__slate_slot_22 as isize)).stream_ring = *__slate_slot_20;
*__slate_slot_19 = xhci_update_stream_mapping(*__slate_slot_21, mem_flags);
trace_xhci_alloc_stream_info_ctx(*__slate_slot_23, *__slate_slot_22);
if *__slate_slot_19 != 0 {
break '__join_13;
} else {
let __v6: u32 = *__slate_slot_22;
*__slate_slot_22 = __v6.wrapping_add(1);
}
}
}
// Leave the other unused stream ring pointers in the stream context
// array initialized to zero.  This will cause the xHC to give us an
// error if the device asks for a stream ID we don't have setup (if it
// was any other way, the host controller would assume the ring is
// "empty" and wait forever for data to be queued to that stream ID).
return *__slate_slot_23;
}
xhci_ring_free(xhci, *__slate_slot_21);
*unsafe { (*(*__slate_slot_23)).stream_rings }.wrapping_offset(*__slate_slot_22 as isize) = core::ptr::null_mut::<xhci_ring>();
}
*__slate_slot_22 = 1;
while *__slate_slot_22 < num_streams {
*__slate_slot_21 = *unsafe { (*(*__slate_slot_23)).stream_rings }.wrapping_offset(*__slate_slot_22 as isize);
if !(*__slate_slot_21).is_null() {
xhci_ring_free(xhci, *__slate_slot_21);
*unsafe { (*(*__slate_slot_23)).stream_rings }.wrapping_offset(*__slate_slot_22 as isize) = core::ptr::null_mut::<xhci_ring>();
}
let __v4: u32 = *__slate_slot_22;
*__slate_slot_22 = __v4.wrapping_add(1);
}
xhci_free_command(xhci, (*(*__slate_slot_23)).free_streams_command);
}
xhci_free_stream_ctx(xhci, (*(*__slate_slot_23)).num_stream_ctxs, (*(*__slate_slot_23)).stream_ctx_array, (*(*__slate_slot_23)).ctx_array_dma);
}
kfree((*(*__slate_slot_23)).stream_rings as *const ());
}
kfree(*__slate_slot_23 as *const ());
}
let __v1: u32 = (*xhci).cmd_ring_reserved_trbs;
(*xhci).cmd_ring_reserved_trbs = __v1.wrapping_sub(1);
return core::ptr::null_mut::<xhci_stream_info>();
}
}
}

/// Sets the MaxPStreams field and the Linear Stream Array field.
/// Sets the dequeue pointer to the stream context array.
#[unsafe(no_mangle)]
extern "C" fn xhci_setup_streams_ep_input_ctx(xhci: *mut xhci_hcd, ep_ctx: *mut xhci_ep_ctx, stream_info: *mut xhci_stream_info) {
let mut max_primary_streams: u32 = 0;
// MaxPStreams is the number of stream context array entries, not the
// number we're actually using.  Must be in 2^(MaxPstreams + 1) format.
// fls(0) = 0, fls(0x1) = 1, fls(0x10) = 2, fls(0x100) = 3, etc.
max_primary_streams = fls(unsafe { (*stream_info).num_stream_ctxs }).wrapping_sub(2) as u32;
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_context_change), c"Setting number of stream ctx array entries to %u".as_ptr() as *const u8, 1i32 << max_primary_streams.wrapping_add(1)) };
let __v2: u32 = (unsafe { (*ep_ctx).ep_info }) & 4294935551u32;
unsafe {
(*ep_ctx).ep_info = __v2;
}
let __v4: u32 = unsafe { (*ep_ctx).ep_info };
let __v5: u32 = __v4 | (((EP_MAXPSTREAMS!(max_primary_streams) as u64) | BIT!(15i32)) as u32);
unsafe {
(*ep_ctx).ep_info = __v5;
}
unsafe {
(*ep_ctx).deq = (*stream_info).ctx_array_dma;
}
}

/// Sets the MaxPStreams field and the Linear Stream Array field to 0.
/// Reinstalls the "normal" endpoint ring (at its previous dequeue mark,
/// not at the beginning of the ring).
#[unsafe(no_mangle)]
extern "C" fn xhci_setup_no_streams_ep_input_ctx(ep_ctx: *mut xhci_ep_ctx, ep: *mut xhci_virt_ep) {
let mut addr: u64 = 0;
let __v1: u32 = unsafe { (*ep_ctx).ep_info };
let __v2: u32 = __v1 & (!(31744u64 | BIT!(15i32)) as u32);
unsafe {
(*ep_ctx).ep_info = __v2;
}
addr = unsafe { xhci_trb_virt_to_dma((*(*ep).ring).deq_seg, (*(*ep).ring).dequeue) };
unsafe {
(*ep_ctx).deq = addr | ((*(*ep).ring).cycle_state as u64);
}
}

/// Frees all stream contexts associated with the endpoint,
///
/// Caller should fix the endpoint context streams fields.
#[unsafe(no_mangle)]
extern "C" fn xhci_free_stream_info(xhci: *mut xhci_hcd, stream_info: *mut xhci_stream_info) {
let mut cur_stream: i32 = 1;
if stream_info.is_null() {
return;
}
while (cur_stream as u32) < unsafe { (*stream_info).num_streams } {
let cur_ring: *mut xhci_ring = unsafe { *unsafe { (*stream_info).stream_rings }.wrapping_offset(cur_stream as isize) };
if !cur_ring.is_null() {
xhci_ring_free(xhci, cur_ring);
unsafe {
*unsafe { (*stream_info).stream_rings }.wrapping_offset(cur_stream as isize) = core::ptr::null_mut::<xhci_ring>();
}
}
cur_stream = cur_stream.wrapping_add(1);
}
xhci_free_command(xhci, unsafe { (*stream_info).free_streams_command });
let __v4: u32 = unsafe { (*xhci).cmd_ring_reserved_trbs };
unsafe {
(*xhci).cmd_ring_reserved_trbs = __v4.wrapping_sub(1);
}
if !unsafe { (*stream_info).stream_ctx_array }.is_null() {
xhci_free_stream_ctx(xhci, unsafe { (*stream_info).num_stream_ctxs }, unsafe { (*stream_info).stream_ctx_array }, unsafe { (*stream_info).ctx_array_dma });
}
unsafe { kfree((*stream_info).stream_rings as *const ()) };
unsafe { kfree(stream_info as *const ()) };
}

// Device context manipulation
fn xhci_free_tt_info(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, slot_id: i32) {
let mut tt_info: *mut xhci_tt_bw_info = unsafe { core::mem::zeroed() };
let mut next: *mut xhci_tt_bw_info = unsafe { core::mem::zeroed() };
let mut slot_found: bool = false;
// If the device never made it past the Set Address stage,
// it may not have the root hub port pointer set correctly.
if unsafe { (*virt_dev).rhub_port }.is_null() {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Bad rhub port.\n".as_ptr() as *const u8) };
}
return;
}
let tt_list_head: *mut list_head = unsafe { core::ptr::addr_of_mut!((*unsafe { (*xhci).rh_bw }.wrapping_offset((*(*virt_dev).rhub_port).hw_portnum as isize)).tts) };
tt_info = list_first_entry__6!(tt_list_head);
next = list_next_entry__3!(tt_info, tt_list);
while list_entry_is_head!(tt_info, tt_list_head, tt_list) == 0 {
// Multi-TT hubs will have more than one entry
if (unsafe { (*tt_info).slot_id }) == slot_id {
slot_found = true;
list_del(unsafe { core::ptr::addr_of_mut!((*tt_info).tt_list) });
unsafe { kfree(tt_info as *const ()) };
} else {
if slot_found {
break;
}
}
tt_info = next;
next = list_next_entry__3!(next, tt_list);
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_alloc_tt_info(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, hdev: *mut usb_device, tt: *mut usb_tt, mem_flags: u32) -> i32 {
let mut __slate_compound_16: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut bw_table: core::mem::MaybeUninit<*mut xhci_interval_bw_table> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut xhci_interval_bw_table = core::ptr::addr_of_mut!(bw_table) as *mut *mut xhci_interval_bw_table;
let mut dev: *mut device = unsafe { core::mem::zeroed() };
let mut j: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(j) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut num_ports: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u32 = core::ptr::addr_of_mut!(num_ports) as *mut u32;
let mut tt_info: core::mem::MaybeUninit<*mut xhci_tt_bw_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut xhci_tt_bw_info = core::ptr::addr_of_mut!(tt_info) as *mut *mut xhci_tt_bw_info;
unsafe {
{
let __slate_auto_bytes_3 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
dev = (*xhci_to_hcd(xhci)).self_.sysdev;
if (*tt).multi == 0 {
*__slate_slot_14 = 1;
} else {
*__slate_slot_14 = (*hdev).maxchild as u32;
}
}
*__slate_slot_13 = 0;
while (*__slate_slot_13 as u32) < *__slate_slot_14 {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_6 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_7 = _kmalloc_node_noprof(core::mem::size_of::<xhci_tt_bw_info>() as u64, mem_flags | 256u32, dev_to_node(dev), *{
__slate_compound_16 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_16)
});
alloc_tag_restore!();
} else {
*__slate_slot_7 = _kmalloc_node_noprof(core::mem::size_of::<xhci_tt_bw_info>() as u64, mem_flags | 256u32, dev_to_node(dev), *{
__slate_compound_18 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_18)
});
}
*__slate_slot_15 = *__slate_slot_7 as *mut xhci_tt_bw_info;
if (*__slate_slot_15).is_null() {
xhci_free_tt_info(xhci, virt_dev, (*(*virt_dev).udev).slot_id);
return 12i32.wrapping_neg();
} else {
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_15)).tt_list));
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_15)).tt_list), core::ptr::addr_of_mut!((*unsafe { (*xhci).rh_bw }.wrapping_offset((*(*virt_dev).rhub_port).hw_portnum as isize)).tts));
(*(*__slate_slot_15)).slot_id = (*(*virt_dev).udev).slot_id;
if (*tt).multi != 0 {
(*(*__slate_slot_15)).ttport = (*__slate_slot_13).wrapping_add(1);
}
*__slate_slot_10 = core::ptr::addr_of_mut!((*(*__slate_slot_15)).bw_table);
*__slate_slot_12 = 0;
while *__slate_slot_12 < 16 {
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*((*(*__slate_slot_10)).interval_bw.as_mut_ptr() as *mut xhci_interval_bw).wrapping_offset(*__slate_slot_12 as isize)).endpoints));
let __v5: i32 = *__slate_slot_12;
*__slate_slot_12 = __v5.wrapping_add(1);
}
let __v3: i32 = *__slate_slot_13;
*__slate_slot_13 = __v3.wrapping_add(1);
let __v1: *mut xhci_tt_bw_info = *__slate_slot_15;
*__slate_slot_15 = __v1.wrapping_offset(1 as isize);
}
}
return 0;
}
}

/// All the xhci_tds in the ring's TD list should be freed at this point.
/// Should be called with xhci->lock held if there is any chance the TT lists
/// will be manipulated by the configure endpoint, allocate device, or update
/// hub functions while this function is removing the TT entries from the list.
#[unsafe(no_mangle)]
extern "C" fn xhci_free_virt_device(xhci: *mut xhci_hcd, dev: *mut xhci_virt_device, slot_id: i32) {
let mut old_active_eps: i32 = 0;
// Slot ID 0 is reserved
if slot_id == 0 || dev.is_null() {
return;
}
// If device ctx array still points to _this_ device, clear it
if !unsafe { (*dev).out_ctx }.is_null() && (unsafe { *unsafe { (*xhci).dcbaa.ctx_array }.wrapping_offset(slot_id as isize) }) == unsafe { (*(*dev).out_ctx).dma } {
unsafe {
*unsafe { (*xhci).dcbaa.ctx_array }.wrapping_offset(slot_id as isize) = 0;
}
}
trace_xhci_free_virt_device(dev);
if !unsafe { (*dev).tt_info }.is_null() {
old_active_eps = unsafe { (*(*dev).tt_info).active_eps };
}
for i in 0i32..31i32 {
if !unsafe { (*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).ring }.is_null() {
xhci_ring_free(xhci, unsafe { (*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).ring });
}
if !unsafe { (*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).stream_info }.is_null() {
xhci_free_stream_info(xhci, unsafe { (*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).stream_info });
}
// Endpoints are normally deleted from the bandwidth list when
// endpoints are dropped, before device is freed.
// If host is dying or being removed then endpoints aren't
// dropped cleanly, so delete the endpoint from list here.
// Only applicable for hosts with software bandwidth checking.
if list_empty((unsafe { core::ptr::addr_of_mut!((*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).bw_endpoint_list) }) as *const list_head) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).bw_endpoint_list) });
}
}
// If this is a hub, free the TT(s) from the TT list
xhci_free_tt_info(xhci, dev, slot_id);
// If necessary, update the number of active TTs on this root port
unsafe { xhci_update_tt_active_eps(xhci, dev, old_active_eps) };
if !unsafe { (*dev).in_ctx }.is_null() {
xhci_free_container_ctx(xhci, unsafe { (*dev).in_ctx });
}
if !unsafe { (*dev).out_ctx }.is_null() {
xhci_free_container_ctx(xhci, unsafe { (*dev).out_ctx });
}
if !unsafe { (*dev).udev }.is_null() && (unsafe { (*(*dev).udev).slot_id }) != 0 {
unsafe {
(*(*dev).udev).slot_id = 0;
}
}
if !unsafe { (*dev).rhub_port }.is_null() && (unsafe { (*(*dev).rhub_port).slot_id }) == slot_id {
unsafe {
(*(*dev).rhub_port).slot_id = 0;
}
}
if (unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) }) == dev {
unsafe {
*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) = core::ptr::null_mut::<xhci_virt_device>();
}
}
unsafe { kfree(dev as *const ()) };
}

/// Free a virt_device structure.
/// If the virt_device added a tt_info (a hub) and has children pointing to
/// that tt_info, then free the child first. Recursive.
/// We can't rely on udev at this point to find child-parent relationships.
#[unsafe(no_mangle)]
extern "C" fn xhci_free_virt_devices_depth_first(xhci: *mut xhci_hcd, slot_id: i32) {
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut next: core::mem::MaybeUninit<*mut xhci_tt_bw_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut xhci_tt_bw_info = core::ptr::addr_of_mut!(next) as *mut *mut xhci_tt_bw_info;
let mut tt_info: core::mem::MaybeUninit<*mut xhci_tt_bw_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut xhci_tt_bw_info = core::ptr::addr_of_mut!(tt_info) as *mut *mut xhci_tt_bw_info;
let mut tt_list_head: core::mem::MaybeUninit<*mut list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut list_head = core::ptr::addr_of_mut!(tt_list_head) as *mut *mut list_head;
let mut vdev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(vdev) as *mut *mut xhci_virt_device;
unsafe {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_8 = *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize);
if (*__slate_slot_8).is_null() {
return;
} else {
if !unsafe { (*(*__slate_slot_8)).rhub_port }.is_null() {
*__slate_slot_7 = core::ptr::addr_of_mut!((*unsafe { (*xhci).rh_bw }.wrapping_offset((*(*(*__slate_slot_8)).rhub_port).hw_portnum as isize)).tts);
*__slate_slot_6 = list_first_entry__6!(*__slate_slot_7);
let __v3: *mut xhci_tt_bw_info = list_next_entry__3!(*__slate_slot_6, tt_list);
*__slate_slot_5 = __v3;
while list_entry_is_head!(*__slate_slot_6, *__slate_slot_7, tt_list) == 0 {
// is this a hub device that added a tt_info to the tts list
if (*(*__slate_slot_6)).slot_id == slot_id {
// are any devices using this tt_info?
*__slate_slot_4 = 1;
while *__slate_slot_4 < ((*xhci).max_slots as i32) {
*__slate_slot_8 = *unsafe { (*xhci).devs }.wrapping_offset(*__slate_slot_4 as isize);
if !(*__slate_slot_8).is_null() && (*(*__slate_slot_8)).tt_info == *__slate_slot_6 {
xhci_free_virt_devices_depth_first(xhci, *__slate_slot_4);
}
let __v2: i32 = *__slate_slot_4;
*__slate_slot_4 = __v2.wrapping_add(1);
}
}
*__slate_slot_6 = *__slate_slot_5;
let __v0: *mut xhci_tt_bw_info = list_next_entry__3!(*__slate_slot_5, tt_list);
*__slate_slot_5 = __v0;
}
}
xhci_debugfs_remove_slot(xhci, slot_id); // we are now at a leaf device
xhci_free_virt_device(xhci, *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize), slot_id);
}
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_alloc_virt_device(xhci: *mut xhci_hcd, slot_id: i32, udev: *mut usb_device, flags: u32) -> i32 {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut dev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(dev) as *mut *mut xhci_virt_device;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
// Slot ID 0 is reserved
if slot_id == 0 || !unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) }.is_null() {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Bad Slot ID %d\n".as_ptr() as *const u8, slot_id);
return 0;
} else {
__obj_size = size_mul(core::mem::size_of::<xhci_virt_device>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_3 = _kzalloc_noprof(__obj_size, flags, *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
});
alloc_tag_restore!();
} else {
*__slate_slot_3 = _kzalloc_noprof(__obj_size, flags, *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
});
}
*__slate_slot_9 = *__slate_slot_3 as *mut xhci_virt_device;
if (*__slate_slot_9).is_null() {
return 0;
} else {
(*(*__slate_slot_9)).slot_id = slot_id;
// Allocate the (output) device context that will be used in the HC.
(*(*__slate_slot_9)).out_ctx = xhci_alloc_container_ctx(xhci, 1i32, flags);
if !unsafe { (*(*__slate_slot_9)).out_ctx }.is_null() {
// Allocate the (input) device context for address device command
(*(*__slate_slot_9)).in_ctx = xhci_alloc_container_ctx(xhci, 2i32, flags);
if !unsafe { (*(*__slate_slot_9)).in_ctx }.is_null() {
// Initialize the cancellation and bandwidth list for each ep
*__slate_slot_8 = 0;
while *__slate_slot_8 < 31 {
(*((*(*__slate_slot_9)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_8 as isize)).ep_index = *__slate_slot_8 as u32;
(*((*(*__slate_slot_9)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_8 as isize)).vdev = *__slate_slot_9;
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*((*(*__slate_slot_9)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_8 as isize)).cancelled_td_list));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*((*(*__slate_slot_9)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_8 as isize)).bw_endpoint_list));
let __v1: i32 = *__slate_slot_8;
*__slate_slot_8 = __v1.wrapping_add(1);
}
// Allocate endpoint 0 ring
(*((*(*__slate_slot_9)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(0 as isize)).ring = xhci_ring_alloc(xhci, 2u32, 0u32, 0u32, flags);
if !unsafe { (*((*(*__slate_slot_9)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(0 as isize)).ring }.is_null() {
xhci_ring_init(xhci, (*((*(*__slate_slot_9)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(0 as isize)).ring);
(*(*__slate_slot_9)).udev = udev;
// Point to output device context in dcbaa.
*unsafe { (*xhci).dcbaa.ctx_array }.wrapping_offset(slot_id as isize) = (*(*(*__slate_slot_9)).out_ctx).dma;
trace_xhci_alloc_virt_device(*__slate_slot_9);
*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) = *__slate_slot_9;
return 1;
}
}
}
if !unsafe { (*(*__slate_slot_9)).in_ctx }.is_null() {
xhci_free_container_ctx(xhci, (*(*__slate_slot_9)).in_ctx);
}
if !unsafe { (*(*__slate_slot_9)).out_ctx }.is_null() {
xhci_free_container_ctx(xhci, (*(*__slate_slot_9)).out_ctx);
}
kfree(*__slate_slot_9 as *const ());
return 0;
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_copy_ep0_dequeue_into_input_ctx(xhci: *mut xhci_hcd, udev: *mut usb_device) {
let virt_dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) };
let ep0_ctx: *mut xhci_ep_ctx = xhci_get_ep_ctx(xhci, unsafe { (*virt_dev).in_ctx }, 0u32);
let ep_ring: *mut xhci_ring = unsafe { (*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(0 as isize)).ring };
// FIXME we don't keep track of the dequeue pointer very well after a
// Set TR dequeue pointer, so we're setting the dequeue pointer of the
// host to our enqueue pointer.  This should only be called after a
// configured device has reset, so all control transfers should have
// been completed or cancelled before the reset.
unsafe {
(*ep0_ctx).deq = xhci_trb_virt_to_dma((*ep_ring).enq_seg, (*ep_ring).enqueue) | ((*ep_ring).cycle_state as u64);
}
}

/// The xHCI roothub may have ports of differing speeds in any order in the port
/// status registers.
///
/// The xHCI hardware wants to know the roothub port that the USB device
/// is attached to (or the roothub port its ancestor hub is attached to).  All we
/// know is the index of that port under either the USB 2.0 or the USB 3.0
/// roothub, but that doesn't give us the real index into the HW port status
/// registers.
fn xhci_find_rhub_port(xhci: *mut xhci_hcd, udev: *mut usb_device) -> *mut xhci_port {
let mut top_dev: *mut usb_device = unsafe { core::mem::zeroed() };
let mut hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
if ((unsafe { (*udev).speed }) as u32) >= 5 {
hcd = xhci_get_usb3_hcd(xhci);
} else {
hcd = unsafe { (*xhci).main_hcd };
}
top_dev = udev;
while !unsafe { (*top_dev).parent }.is_null() && !unsafe { (*(*top_dev).parent).parent }.is_null() {
top_dev = unsafe { (*top_dev).parent };
}
// Found device below root hub
let rhub: *mut xhci_hub = unsafe { xhci_get_rhub(hcd) };
unsafe { *unsafe { (*rhub).ports }.wrapping_offset(((*top_dev).portnum as i32).wrapping_sub(1) as isize) }
}

/// Setup an xHCI virtual device for a Set Address command
#[unsafe(no_mangle)]
extern "C" fn xhci_setup_addressable_virt_dev(xhci: *mut xhci_hcd, udev: *mut usb_device) -> i32 {
let mut dev: *mut xhci_virt_device = unsafe { core::mem::zeroed() };
let mut max_packets: u32 = 0;
dev = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) };
// Slot ID 0 is reserved
if (unsafe { (*udev).slot_id }) == 0 || dev.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Slot ID %d is not assigned to this device\n".as_ptr() as *const u8, (*udev).slot_id) };
return 22i32.wrapping_neg();
}
let ep0_ctx: *mut xhci_ep_ctx = xhci_get_ep_ctx(xhci, unsafe { (*dev).in_ctx }, 0u32);
let slot_ctx: *mut xhci_slot_ctx = xhci_get_slot_ctx(xhci, unsafe { (*dev).in_ctx });
// 3) Only the control endpoint is valid - one endpoint context
let __v2: u32 = unsafe { (*slot_ctx).dev_info };
let __v3: u32 = __v2 | ((LAST_CTX!(1i32) as u32) | unsafe { (*udev).route });
unsafe {
(*slot_ctx).dev_info = __v3;
}
'__slate_break_7: {
match (unsafe { (*udev).speed }) as u32 {
6 => {
unsafe {
(*slot_ctx).dev_info = (*slot_ctx).dev_info | 5242880u32;
}
max_packets = MAX_PACKET!(512i32) as u32;
}
5 => {
unsafe {
(*slot_ctx).dev_info = (*slot_ctx).dev_info | 4194304u32;
}
max_packets = MAX_PACKET!(512i32) as u32;
}
3 => {
unsafe {
(*slot_ctx).dev_info = (*slot_ctx).dev_info | 3145728u32;
}
max_packets = MAX_PACKET!(64i32) as u32;
break '__slate_break_7;
// USB core guesses at a 64-byte max packet first for FS devices
}
2 => {
unsafe {
(*slot_ctx).dev_info = (*slot_ctx).dev_info | 1048576u32;
}
max_packets = MAX_PACKET!(64i32) as u32;
}
1 => {
unsafe {
(*slot_ctx).dev_info = (*slot_ctx).dev_info | 2097152u32;
}
max_packets = MAX_PACKET!(8i32) as u32;
}
_ => {
return 22i32.wrapping_neg(); // Speed was set earlier, this shouldn't happen.
}
}
}
// Find the root hub port this device is under
unsafe {
(*dev).rhub_port = xhci_find_rhub_port(xhci, udev);
}
if unsafe { (*dev).rhub_port }.is_null() {
return 22i32.wrapping_neg();
}
// Slot ID is set to the device directly below the root hub
if unsafe { (*(*udev).parent).parent }.is_null() {
unsafe {
(*(*dev).rhub_port).slot_id = (*udev).slot_id;
}
}
let __v21: u32 = unsafe { (*slot_ctx).dev_info2 };
let __v22: u32 = __v21 | (ROOT_HUB_PORT!(unsafe { (*(*dev).rhub_port).hw_portnum }.wrapping_add(1)) as u32);
unsafe {
(*slot_ctx).dev_info2 = __v22;
}
// Find the right bandwidth table that this device will be a part of.
// If this is a full speed device attached directly to a root port (or a
// decendent of one), it counts as a primary bandwidth domain, not a
// secondary bandwidth domain under a TT.  An xhci_tt_info structure
// will never be created for the HS root hub.
if unsafe { (*udev).tt }.is_null() || unsafe { (*(*(*udev).tt).hub).parent }.is_null() {
unsafe {
(*dev).bw_table = core::ptr::addr_of_mut!((*unsafe { (*xhci).rh_bw }.wrapping_offset((*(*dev).rhub_port).hw_portnum as isize)).bw_table);
}
} else {
let mut rh_bw: *mut xhci_root_port_bw_info = unsafe { core::mem::zeroed() };
let mut tt_bw: *mut xhci_tt_bw_info = unsafe { core::mem::zeroed() };
rh_bw = unsafe { (*xhci).rh_bw }.wrapping_offset((unsafe { (*(*dev).rhub_port).hw_portnum }) as isize);
// Find the right TT.
tt_bw = list_first_entry__6!(unsafe { core::ptr::addr_of_mut!((*rh_bw).tts) });
while list_entry_is_head!(tt_bw, unsafe { core::ptr::addr_of_mut!((*rh_bw).tts) }, tt_list) == 0 {
if (unsafe { (*tt_bw).slot_id }) == unsafe { (*(*(*udev).tt).hub).slot_id } {
if (unsafe { (*(*(*dev).udev).tt).multi }) == 0 || (unsafe { (*(*udev).tt).multi }) != 0 && (unsafe { (*tt_bw).ttport }) == unsafe { (*(*dev).udev).ttport } {
unsafe {
(*dev).bw_table = core::ptr::addr_of_mut!((*tt_bw).bw_table);
}
unsafe {
(*dev).tt_info = tt_bw;
}
break;
}
}
tt_bw = list_next_entry__3!(tt_bw, tt_list);
}
if unsafe { (*dev).tt_info }.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: Didn't find a matching TT\n".as_ptr() as *const u8) };
}
}
// Is this a LS/FS device under an external HS hub?
if !unsafe { (*udev).tt }.is_null() && !unsafe { (*(*(*udev).tt).hub).parent }.is_null() {
unsafe {
(*slot_ctx).tt_info = ((*(*(*udev).tt).hub).slot_id | (*udev).ttport << 8i32) as u32;
}
if (unsafe { (*(*udev).tt).multi }) != 0 {
let __v28: u32 = unsafe { (*slot_ctx).dev_info };
let __v29: u32 = __v28 | (BIT!(25i32) as u32);
unsafe {
(*slot_ctx).dev_info = __v29;
}
}
}
// Step 4 - ring already allocated
//
// Step 5
unsafe {
(*ep0_ctx).ep_info2 = EP_TYPE!(4i32) as u32;
}
// EP 0 can handle "burst" sizes of 1, so Max Burst Size field is 0
let __v31: u32 = unsafe { (*ep0_ctx).ep_info2 };
let __v32: u32 = __v31 | (((MAX_BURST!(0i32) | ERROR_COUNT__2!(3i32)) as u32) | max_packets);
unsafe {
(*ep0_ctx).ep_info2 = __v32;
}
unsafe {
(*ep0_ctx).deq = (*(*(*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(0 as isize)).ring).first_seg).dma | ((*(*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(0 as isize)).ring).cycle_state as u64);
}
unsafe {
(*ep0_ctx).tx_info = EP_AVG_TRB_LENGTH__2!(8i32) as u32;
}
trace_xhci_setup_addressable_virt_device(dev);
// Steps 7 and 8 were done in xhci_alloc_virt_device()
0
}

// Convert interval expressed as 2^(bInterval - 1) == interval into
// straight exponent value 2^n == interval.
fn xhci_parse_exponent_interval(udev: *mut usb_device, ep: *mut usb_host_endpoint) -> u32 {
let mut interval: u32 = 0;
let mut __v0: i32;
let __UNIQUE_ID_v__925: u8 = unsafe { (*ep).desc.bInterval };
let __UNIQUE_ID_l__926: u8 = 1;
let __UNIQUE_ID_h__927: u8 = 16;
loop {
if !(!statically_true!((__UNIQUE_ID_l__926 as i32) > (__UNIQUE_ID_h__927 as i32))) {
unsafe { __compiletime_assert_928() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3__2!(__UNIQUE_ID_v__925, __UNIQUE_ID_l__926, __UNIQUE_ID_h__927) == 0 {
unsafe { __compiletime_assert_929() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_v__925 as i32) >= (__UNIQUE_ID_h__927 as i32) { __UNIQUE_ID_h__927 as i32 } else { if (__UNIQUE_ID_v__925 as i32) <= (__UNIQUE_ID_l__926 as i32) { __UNIQUE_ID_l__926 as i32 } else { __UNIQUE_ID_v__925 as i32 } };
interval = __v0.wrapping_sub(1) as u32;
if interval != (((unsafe { (*ep).desc.bInterval }) as i32).wrapping_sub(1) as u32) {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"ep %#x - rounding interval to %d %sframes\n".as_ptr() as *const u8, (*ep).desc.bEndpointAddress as i32, 1i32 << interval, if ((*udev).speed as u32) == 2 { c"".as_ptr() as *mut u8 } else { c"micro".as_ptr() as *mut u8 }) };
}
if ((unsafe { (*udev).speed }) as u32) == 2 {
// Full speed isoc endpoints specify interval in frames,
// not microframes. We are using microframes everywhere,
// so adjust accordingly.
interval = interval.wrapping_add(3); // 1 frame = 2^3 uframes
}
interval
}

// Convert bInterval expressed in microframes (in 1-255 range) to exponent of
// microframes, rounded down to nearest power of 2.
fn xhci_microframes_to_exponent(udev: *mut usb_device, ep: *mut usb_host_endpoint, desc_interval: u32, __UNIQUE_ID_l__931: u32, __UNIQUE_ID_h__932: u32) -> u32 {
let mut interval: u32 = fls(desc_interval).wrapping_sub(1) as u32;
let mut __v0: u32;
let __UNIQUE_ID_v__930: u32 = interval;
loop {
if !(!statically_true!(__UNIQUE_ID_l__931 > __UNIQUE_ID_h__932)) {
unsafe { __compiletime_assert_933() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3__3!(__UNIQUE_ID_v__930, __UNIQUE_ID_l__931, __UNIQUE_ID_h__932) == 0 {
unsafe { __compiletime_assert_934() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_v__930 >= __UNIQUE_ID_h__932 { __UNIQUE_ID_h__932 } else { if __UNIQUE_ID_v__930 <= __UNIQUE_ID_l__931 { __UNIQUE_ID_l__931 } else { __UNIQUE_ID_v__930 } };
interval = __v0;
if ((1i32 << interval) as u32) != desc_interval {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"ep %#x - rounding interval to %d microframes, ep desc says %d microframes\n".as_ptr() as *const u8, (*ep).desc.bEndpointAddress as i32, 1i32 << interval, desc_interval) };
}
}
interval
}

fn xhci_parse_microframe_interval(udev: *mut usb_device, ep: *mut usb_host_endpoint) -> u32 {
if ((unsafe { (*ep).desc.bInterval }) as i32) == 0 {
return 0;
}
xhci_microframes_to_exponent(udev, ep, (unsafe { (*ep).desc.bInterval }) as u32, 0u32, 15u32)
}

fn xhci_parse_frame_interval(udev: *mut usb_device, ep: *mut usb_host_endpoint) -> u32 {
xhci_microframes_to_exponent(udev, ep, ((unsafe { (*ep).desc.bInterval }) as i32).wrapping_mul(8) as u32, 3u32, 10u32)
}

/// Return the polling or NAK interval.
///
/// The polling interval is expressed in "microframes".  If xHCI's Interval field
/// is set to N, it will service the endpoint every 2^(Interval)*125us.
///
/// The NAK interval is one NAK per 1 to 255 microframes, or no NAKs if interval
/// is set to 0.
fn xhci_get_endpoint_interval(udev: *mut usb_device, ep: *mut usb_host_endpoint) -> u32 {
let mut interval: u32 = 0;
// Max NAK rate
// SS and HS isoc/int have same decoding
// Fall through for interrupt endpoint interval decoding
// since it uses the same rules as low speed interrupt
// endpoints.
match (unsafe { (*udev).speed }) as u32 {
3 => {
let __v1: bool = usb_endpoint_xfer_control((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_bulk((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0;
if __v1 {
interval = xhci_parse_microframe_interval(udev, ep);
} else {
// Max NAK rate
// SS and HS isoc/int have same decoding
let _v4: bool = usb_endpoint_xfer_int((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0;
if _v4 {
interval = xhci_parse_exponent_interval(udev, ep);
}
}
}
5 | 6 => {
let __v2: bool = usb_endpoint_xfer_int((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0;
if __v2 {
interval = xhci_parse_exponent_interval(udev, ep);
}
}
2 => {
if usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 {
interval = xhci_parse_exponent_interval(udev, ep);
} else {
// Fall through for interrupt endpoint interval decoding
// since it uses the same rules as low speed interrupt
// endpoints.
let _v5: bool = usb_endpoint_xfer_int((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0;
if _v5 {
interval = xhci_parse_frame_interval(udev, ep);
}
}
}
1 => {
let __v3: bool = usb_endpoint_xfer_int((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0;
if __v3 {
interval = xhci_parse_frame_interval(udev, ep);
}
}
_ => {
loop {
unsafe {
core::arch::asm!("935: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 935b - ., 3; .popsection\n# {0}", const 935 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_xhci_get_endpoint_interval_4, sym __slate_asm_string_xhci_get_endpoint_interval_5, const 1330 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
interval
}

/// xHCs without LEC use the "Mult" field in the endpoint context for SuperSpeed
/// isoc eps, and High speed isoc eps that support bandwidth doubling. Standard
/// High speed endpoint descriptors can define "the number of additional
/// transaction opportunities per microframe", but that goes in the Max Burst
/// endpoint context field.
fn xhci_get_endpoint_mult(xhci: *mut xhci_hcd, udev: *mut usb_device, ep: *mut usb_host_endpoint) -> u32 {
let mut lec: bool = false;
// xHCI 1.1 with LEC set does not use mult field, except intel eUSB2
lec = ((unsafe { (*xhci).hci_version }) as i32) > 256 && ((unsafe { (*xhci).hcc_params2 }) as u64) & BIT!(4i32) != 0;
// eUSB2 double isoc bw devices are the only USB2 devices using mult
if (unsafe { usb_endpoint_is_hs_isoc_double(udev, ep as *const usb_host_endpoint) }) && (!lec || (unsafe { (*xhci).quirks }) & BIT_ULL!(12i32) != 0) {
return 1;
}
// SuperSpeed isoc transfers on hosts without LEC uses mult field
let __v0: bool = ((unsafe { (*udev).speed }) as u32) >= 5 && usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0;
if __v0 && !lec {
return (unsafe { (*ep).ss_ep_comp.bmAttributes }) as u32;
}
0
}

fn xhci_get_endpoint_max_burst(udev: *mut usb_device, ep: *mut usb_host_endpoint) -> u32 {
// Super speed and Plus have max burst in ep companion desc
if ((unsafe { (*udev).speed }) as u32) >= 5 {
return (unsafe { (*ep).ss_ep_comp.bMaxBurst }) as u32;
}
let __v0: bool = ((unsafe { (*udev).speed }) as u32) == 3 && (usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_int((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) != 0);
if __v0 {
// USB 2 Isochronous Double IN Bandwidth ECN uses fixed burst
// size and max packets bits 12:11 are invalid.
if unsafe { usb_endpoint_is_hs_isoc_double(udev, ep as *const usb_host_endpoint) } {
return 2;
}
return usb_endpoint_maxp_mult((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor).wrapping_sub(1) as u32;
}
0
}

fn xhci_get_endpoint_type(ep: *mut usb_host_endpoint) -> u32 {
let r#in: i32 = usb_endpoint_dir_in((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor);
match usb_endpoint_type((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor) {
0 => {
return 4;
}
2 => {
return (if r#in != 0 { 6i32 } else { 2i32 }) as u32;
}
1 => {
return (if r#in != 0 { 5i32 } else { 1i32 }) as u32;
}
3 => {
return (if r#in != 0 { 7i32 } else { 3i32 }) as u32;
}
_ => {
}
}
0
}

// Set up an endpoint with one ring segment.  Do not allocate stream rings.
// Drivers will have to call usb_alloc_streams() to do that.
#[unsafe(no_mangle)]
extern "C" fn xhci_endpoint_zero(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, ep: *mut usb_host_endpoint) {
let mut ep_index: u32 = 0;
ep_index = unsafe { xhci_get_endpoint_index(core::ptr::addr_of_mut!((*ep).desc)) };
let ep_ctx: *mut xhci_ep_ctx = xhci_get_ep_ctx(xhci, unsafe { (*virt_dev).in_ctx }, ep_index);
unsafe {
(*ep_ctx).ep_info = 0;
}
unsafe {
(*ep_ctx).ep_info2 = 0;
}
unsafe {
(*ep_ctx).deq = 0;
}
unsafe {
(*ep_ctx).tx_info = 0;
}
// Don't free the endpoint ring until the set interface or configuration
// request succeeds.
}

#[unsafe(no_mangle)]
extern "C" fn xhci_clear_endpoint_bw_info(bw_info: *mut xhci_bw_info) {
unsafe {
(*bw_info).ep_interval = 0;
}
unsafe {
(*bw_info).mult = 0;
}
unsafe {
(*bw_info).num_packets = 0;
}
unsafe {
(*bw_info).max_packet_size = 0;
}
unsafe {
(*bw_info).r#type = 0;
}
unsafe {
(*bw_info).max_esit_payload = 0;
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_update_bw_info(xhci: *mut xhci_hcd, in_ctx: *mut xhci_container_ctx, ctrl_ctx: *mut xhci_input_control_ctx, virt_dev: *mut xhci_virt_device) {
let mut ep_type: u32 = 0;
for i in 1i32..31i32 {
let bw_info: *mut xhci_bw_info = unsafe { core::ptr::addr_of_mut!((*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).bw_info) };
// We can't tell what endpoint type is being dropped, but
// unconditionally clearing the bandwidth info for non-periodic
// endpoints should be harmless because the info will never be
// set in the first place.
if EP_IS_ADDED!(ctrl_ctx, i) == 0 && EP_IS_DROPPED!(ctrl_ctx, i) != 0 {
// Dropped endpoint
xhci_clear_endpoint_bw_info(bw_info);
} else {
if EP_IS_ADDED!(ctrl_ctx, i) != 0 {
let ep_ctx: *mut xhci_ep_ctx = xhci_get_ep_ctx(xhci, in_ctx, i as u32);
ep_type = CTX_TO_EP_TYPE!(unsafe { (*ep_ctx).ep_info2 });
// Ignore non-periodic endpoints
if !(ep_type != 1 && ep_type != 3 && ep_type != 5 && ep_type != 7) {
// Added or changed endpoint
unsafe {
(*bw_info).ep_interval = CTX_TO_EP_INTERVAL!(unsafe { (*ep_ctx).ep_info });
}
// Number of packets and mult are zero-based in the
// input context, but we want one-based for the
// interval table.
unsafe {
(*bw_info).mult = CTX_TO_EP_MULT!(unsafe { (*ep_ctx).ep_info }).wrapping_add(1);
}
unsafe {
(*bw_info).num_packets = CTX_TO_MAX_BURST!(unsafe { (*ep_ctx).ep_info2 }).wrapping_add(1);
}
unsafe {
(*bw_info).max_packet_size = MAX_PACKET_DECODED!(unsafe { (*ep_ctx).ep_info2 });
}
unsafe {
(*bw_info).r#type = ep_type;
}
unsafe {
(*bw_info).max_esit_payload = CTX_TO_MAX_ESIT_PAYLOAD!(unsafe { (*ep_ctx).tx_info });
}
}
}
}
}
}

/// Copy output xhci_ep_ctx to the input xhci_ep_ctx copy.
/// Useful when you want to change one particular aspect of the endpoint and then
/// issue a configure endpoint command.
#[unsafe(no_mangle)]
extern "C" fn xhci_endpoint_copy(xhci: *mut xhci_hcd, in_ctx: *mut xhci_container_ctx, out_ctx: *mut xhci_container_ctx, ep_index: u32) {
let out_ep_ctx: *mut xhci_ep_ctx = xhci_get_ep_ctx(xhci, out_ctx, ep_index);
let in_ep_ctx: *mut xhci_ep_ctx = xhci_get_ep_ctx(xhci, in_ctx, ep_index);
unsafe {
(*in_ep_ctx).ep_info = (*out_ep_ctx).ep_info;
}
unsafe {
(*in_ep_ctx).ep_info2 = (*out_ep_ctx).ep_info2;
}
unsafe {
(*in_ep_ctx).deq = (*out_ep_ctx).deq;
}
unsafe {
(*in_ep_ctx).tx_info = (*out_ep_ctx).tx_info;
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(21i32) != 0 {
unsafe {
*((*in_ep_ctx).reserved.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = *((*out_ep_ctx).reserved.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize);
}
unsafe {
*((*in_ep_ctx).reserved.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize) = *((*out_ep_ctx).reserved.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize);
}
}
}

/// Copy output xhci_slot_ctx to the input xhci_slot_ctx.
/// Useful when you want to change one particular aspect of the endpoint and then
/// issue a configure endpoint command.  Only the context entries field matters,
/// but we'll copy the whole thing anyway.
#[unsafe(no_mangle)]
extern "C" fn xhci_slot_copy(xhci: *mut xhci_hcd, in_ctx: *mut xhci_container_ctx, out_ctx: *mut xhci_container_ctx) {
let in_slot_ctx: *mut xhci_slot_ctx = xhci_get_slot_ctx(xhci, in_ctx);
let out_slot_ctx: *mut xhci_slot_ctx = xhci_get_slot_ctx(xhci, out_ctx);
unsafe {
(*in_slot_ctx).dev_info = (*out_slot_ctx).dev_info;
}
unsafe {
(*in_slot_ctx).dev_info2 = (*out_slot_ctx).dev_info2;
}
unsafe {
(*in_slot_ctx).tt_info = (*out_slot_ctx).tt_info;
}
unsafe {
(*in_slot_ctx).dev_state = (*out_slot_ctx).dev_state;
}
}

// Set up the scratchpad buffer array and scratchpad buffers, if needed.
fn scratchpad_alloc(xhci: *mut xhci_hcd, flags: u32) -> i32 {
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_24: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut buf: *mut () = unsafe { core::mem::zeroed() };
let mut dma: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(dma) as *mut u64;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __v17: u64 = unsafe { core::mem::zeroed() };
let mut num_sp: i32 = unsafe { core::mem::zeroed() };
let mut dev: *mut device = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
let __slate_auto_bytes_3 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
dev = (*xhci_to_hcd(xhci)).self_.sysdev;
while false {
}
while false {
}
while false {
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
__compiletime_assert_944();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_945();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
__v17 = (((*xhci).hcs_params2 as u64) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25))) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).trailing_zeros() as i32);
while false {
}
while false {
}
while false {
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_20>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_21>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
__compiletime_assert_949();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_22>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_23>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_24>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_25>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_950();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
num_sp = (__v17 << 5i32 | (((*xhci).hcs_params2 as u64) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_27>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1651_28>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) as i32;
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Allocating %d scratchpad buffers".as_ptr() as *const u8, num_sp);
if num_sp == 0 {
return 0;
} else {
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_10 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_11 = _kmalloc_node_noprof(core::mem::size_of::<xhci_scratchpad>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_24 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_24)
});
alloc_tag_restore!();
} else {
*__slate_slot_11 = _kmalloc_node_noprof(core::mem::size_of::<xhci_scratchpad>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_26 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_26)
});
}
(*xhci).scratchpad = *__slate_slot_11 as *mut xhci_scratchpad;
if !unsafe { (*xhci).scratchpad }.is_null() {
(*(*xhci).scratchpad).sp_array = dma_alloc_coherent(dev, array_size!(core::mem::size_of::<u64>() as u64, num_sp), core::ptr::addr_of_mut!((*(*xhci).scratchpad).sp_dma), flags) as *mut u64;
if !unsafe { (*(*xhci).scratchpad).sp_array }.is_null() {
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_6 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_7 = _kmalloc_array_node_noprof(num_sp as u64, core::mem::size_of::<*mut ()>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
alloc_tag_restore!();
} else {
*__slate_slot_7 = _kmalloc_array_node_noprof(num_sp as u64, core::mem::size_of::<*mut ()>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_23 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_23)
});
}
(*(*xhci).scratchpad).sp_buffers = *__slate_slot_7 as *mut *mut ();
if !unsafe { (*(*xhci).scratchpad).sp_buffers }.is_null() {
*unsafe { (*xhci).dcbaa.ctx_array }.wrapping_offset(0 as isize) = (*(*xhci).scratchpad).sp_dma;
*__slate_slot_20 = 0;
'__join_4: {
while *__slate_slot_20 < num_sp {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
buf = dma_alloc_coherent(dev, (*xhci).page_size as u64, core::ptr::addr_of_mut!(*__slate_slot_5), flags);
if buf.is_null() {
break '__join_4;
} else {
*unsafe { (*(*xhci).scratchpad).sp_array }.wrapping_offset(*__slate_slot_20 as isize) = *__slate_slot_5;
*unsafe { (*(*xhci).scratchpad).sp_buffers }.wrapping_offset(*__slate_slot_20 as isize) = buf;
let __v3: i32 = *__slate_slot_20;
*__slate_slot_20 = __v3.wrapping_add(1);
}
}
return 0;
}
loop {
let __v1: i32 = *__slate_slot_20;
*__slate_slot_20 = __v1.wrapping_sub(1);
if __v1 != 0 {
dma_free_coherent(dev, (*xhci).page_size as u64, *unsafe { (*(*xhci).scratchpad).sp_buffers }.wrapping_offset(*__slate_slot_20 as isize), *unsafe { (*(*xhci).scratchpad).sp_array }.wrapping_offset(*__slate_slot_20 as isize));
} else {
break;
}
}
kfree((*(*xhci).scratchpad).sp_buffers as *const ());
}
dma_free_coherent(dev, array_size!(core::mem::size_of::<u64>() as u64, num_sp), (*(*xhci).scratchpad).sp_array as *mut (), (*(*xhci).scratchpad).sp_dma);
}
kfree((*xhci).scratchpad as *const ());
(*xhci).scratchpad = core::ptr::null_mut::<xhci_scratchpad>();
}
return 12i32.wrapping_neg();
}
}
}

fn scratchpad_free(xhci: *mut xhci_hcd) {
let mut num_sp: i32 = 0;
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
if unsafe { (*xhci).scratchpad }.is_null() {
return;
}
let mut __v0: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_951() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_952() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)) == 0 {
unsafe { __compiletime_assert_953() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_954() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_955() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (((unsafe { (*xhci).hcs_params2 }) as u64) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25))) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 21i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(25)).trailing_zeros() as i32);
__v0 = __v2;
let mut __v7: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_17>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_956() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v9: u64;
loop {
if false {
unsafe { __compiletime_assert_957() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_19>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) == 0 {
unsafe { __compiletime_assert_958() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_20>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_21>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_959() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_22>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_23>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_24>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_25>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_960() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = (((unsafe { (*xhci).hcs_params2 }) as u64) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_27>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_1719_28>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 27i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32);
__v7 = __v9;
num_sp = (__v0 << 5i32 | __v7) as i32;
for i in 0..num_sp {
dma_free_coherent(dev, (unsafe { (*xhci).page_size }) as u64, unsafe { *unsafe { (*(*xhci).scratchpad).sp_buffers }.wrapping_offset(i as isize) }, unsafe { *unsafe { (*(*xhci).scratchpad).sp_array }.wrapping_offset(i as isize) });
}
unsafe { kfree((*(*xhci).scratchpad).sp_buffers as *const ()) };
dma_free_coherent(dev, array_size!(core::mem::size_of::<u64>() as u64, num_sp), (unsafe { (*(*xhci).scratchpad).sp_array }) as *mut (), unsafe { (*(*xhci).scratchpad).sp_dma });
unsafe { kfree((*xhci).scratchpad as *const ()) };
unsafe {
(*xhci).scratchpad = core::ptr::null_mut::<xhci_scratchpad>();
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_alloc_command(xhci: *mut xhci_hcd, allocate_completion: bool, mem_flags: u32) -> *mut xhci_command {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut command: *mut xhci_command = unsafe { core::mem::zeroed() };
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let mut __v0: *mut ();
let mut __v1: *mut ();
{
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_command>() as u64, mem_flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_command>() as u64, mem_flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
}
__v0 = __v1;
command = __v0 as *mut xhci_command;
if command.is_null() {
return core::ptr::null_mut::<xhci_command>();
}
if allocate_completion {
let mut __v5: *mut ();
let mut __v6: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_node_noprof(core::mem::size_of::<completion>() as u64, mem_flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_node_noprof(core::mem::size_of::<completion>() as u64, mem_flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
}
__v6 = _res;
__v5 = __v6;
unsafe {
(*command).completion = __v5 as *mut completion;
}
if unsafe { (*command).completion }.is_null() {
unsafe { kfree(command as *const ()) };
return core::ptr::null_mut::<xhci_command>();
}
init_completion(unsafe { (*command).completion });
}
unsafe {
(*command).status = 0;
}
// set default timeout to 5000 ms
unsafe {
(*command).timeout_ms = 5000;
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*command).cmd_list) });
command
}

#[unsafe(no_mangle)]
extern "C" fn xhci_alloc_command_with_ctx(xhci: *mut xhci_hcd, allocate_completion: bool, mem_flags: u32) -> *mut xhci_command {
let command: *mut xhci_command = xhci_alloc_command(xhci, allocate_completion, mem_flags);
if command.is_null() {
return core::ptr::null_mut::<xhci_command>();
}
unsafe {
(*command).in_ctx = xhci_alloc_container_ctx(xhci, 2i32, mem_flags);
}
if unsafe { (*command).in_ctx }.is_null() {
unsafe { kfree((*command).completion as *const ()) };
unsafe { kfree(command as *const ()) };
return core::ptr::null_mut::<xhci_command>();
}
command
}

#[unsafe(no_mangle)]
extern "C" fn xhci_urb_free_priv(urb_priv: *mut urb_priv) {
unsafe { kfree(urb_priv as *const ()) };
}

#[unsafe(no_mangle)]
extern "C" fn xhci_free_command(xhci: *mut xhci_hcd, command: *mut xhci_command) {
xhci_free_container_ctx(xhci, unsafe { (*command).in_ctx });
unsafe { kfree((*command).completion as *const ()) };
unsafe { kfree(command as *const ()) };
}

fn xhci_alloc_erst(xhci: *mut xhci_hcd, evt_ring: *mut xhci_ring, erst: *mut xhci_erst, flags: u32) -> i32 {
let mut size: u64 = 0;
let mut val: u32 = 0;
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
size = array_size!(core::mem::size_of::<xhci_erst_entry>() as u64, unsafe { (*evt_ring).num_segs });
unsafe {
(*erst).entries = dma_alloc_coherent((*xhci_to_hcd(xhci)).self_.sysdev, size, core::ptr::addr_of_mut!((*erst).erst_dma_addr), flags) as *mut xhci_erst_entry;
}
if unsafe { (*erst).entries }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*erst).num_entries = (*evt_ring).num_segs;
}
seg = unsafe { (*evt_ring).first_seg };
while val < unsafe { (*evt_ring).num_segs } {
let entry: *mut xhci_erst_entry = unsafe { (*erst).entries }.wrapping_offset(val as isize);
unsafe {
(*entry).seg_addr = (*seg).dma;
}
unsafe {
(*entry).seg_size = 256;
}
unsafe {
(*entry).rsvd = 0;
}
seg = unsafe { (*seg).next };
val = val.wrapping_add(1);
}
0
}

fn xhci_remove_interrupter(xhci: *mut xhci_hcd, ir: *mut xhci_interrupter) {
let mut tmp: u32 = 0;
if ir.is_null() {
return;
}
// Clean out interrupter registers except ERSTBA. Clearing either the
// low or high 32 bits of ERSTBA immediately causes the controller to
// dereference the partially cleared 64 bit address, causing IOMMU error.
if !unsafe { (*ir).ir_set }.is_null() {
tmp = readl((unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_size) }) as *const ());
let __v1: u32 = tmp & 4294901760u32;
tmp = __v1;
writel(tmp, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_size) }) as *mut ());
unsafe { xhci_update_erst_dequeue(xhci, ir, true) };
}
}

fn xhci_free_interrupter(xhci: *mut xhci_hcd, ir: *mut xhci_interrupter) {
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let mut erst_size: u64 = 0;
if ir.is_null() {
return;
}
erst_size = array_size!(core::mem::size_of::<xhci_erst_entry>() as u64, unsafe { (*ir).erst.num_entries });
if !unsafe { (*ir).erst.entries }.is_null() {
dma_free_coherent(dev, erst_size, (unsafe { (*ir).erst.entries }) as *mut (), unsafe { (*ir).erst.erst_dma_addr });
}
unsafe {
(*ir).erst.entries = core::ptr::null_mut::<xhci_erst_entry>();
}
// free interrupter event ring
if !unsafe { (*ir).event_ring }.is_null() {
xhci_ring_free(xhci, unsafe { (*ir).event_ring });
}
unsafe {
(*ir).event_ring = core::ptr::null_mut::<xhci_ring>();
}
unsafe { kfree(ir as *const ()) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci_mem.xhci_remove_secondary_interrupter")]
extern "C" fn xhci_remove_secondary_interrupter(hcd: *mut usb_hcd, ir: *mut xhci_interrupter) {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
let mut intr_num: u32 = 0;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
// interrupter 0 is primary interrupter, don't touch it
if ir.is_null() || (unsafe { (*ir).intr_num }) == 0 || (unsafe { (*ir).intr_num }) >= (((unsafe { (*xhci).max_interrupters }) as i32) as u32) {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
return;
}
// Cleanup secondary interrupter to ensure there are no pending events.
// This also updates event ring dequeue pointer back to the start.
unsafe { xhci_skip_sec_intr_events(xhci, (*ir).event_ring, ir) };
intr_num = unsafe { (*ir).intr_num };
xhci_remove_interrupter(xhci, ir);
unsafe {
*unsafe { (*xhci).interrupters }.wrapping_offset(intr_num as isize) = core::ptr::null_mut::<xhci_interrupter>();
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
xhci_free_interrupter(xhci, ir);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_remove_secondary_interrupter_961: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, *mut xhci_interrupter)>, *mut ()>(Some(xhci_remove_secondary_interrupter)) };

/// Cleanup roothub bandwidth data
#[unsafe(no_mangle)]
extern "C" fn xhci_rh_bw_cleanup(xhci: *mut xhci_hcd) {
let mut rh_bw: *mut xhci_root_port_bw_info = unsafe { core::mem::zeroed() };
let mut tt_info: *mut xhci_tt_bw_info = unsafe { core::mem::zeroed() };
let mut tt_next: *mut xhci_tt_bw_info = unsafe { core::mem::zeroed() };
let mut ep: *mut list_head = unsafe { core::mem::zeroed() };
let mut ep_next: *mut list_head = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < ((unsafe { (*xhci).max_ports }) as i32) {
rh_bw = unsafe { (*xhci).rh_bw }.wrapping_offset(i as isize);
// Clear and free all TT bandwidth entries
tt_info = list_first_entry__6!(unsafe { core::ptr::addr_of_mut!((*rh_bw).tts) });
tt_next = list_next_entry__3!(tt_info, tt_list);
while list_entry_is_head!(tt_info, unsafe { core::ptr::addr_of_mut!((*rh_bw).tts) }, tt_list) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*tt_info).tt_list) });
unsafe { kfree(tt_info as *const ()) };
tt_info = tt_next;
tt_next = list_next_entry__3!(tt_next, tt_list);
}
// Clear per-interval endpoint lists
for j in 0i32..16i32 {
let eps: *mut list_head = unsafe { core::ptr::addr_of_mut!((*((*rh_bw).bw_table.interval_bw.as_mut_ptr() as *mut xhci_interval_bw).wrapping_offset(j as isize)).endpoints) };
ep = unsafe { (*eps).next };
ep_next = unsafe { (*ep).next };
while list_is_head(ep as *const list_head, eps as *const list_head) == 0 {
list_del_init(ep);
ep = ep_next;
ep_next = unsafe { (*ep).next };
}
}
i = i.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_mem_cleanup(xhci: *mut xhci_hcd) {
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let mut i: i32 = 0;
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*xhci).cmd_timer)) };
while !unsafe { (*xhci).interrupters }.is_null() && i < ((unsafe { (*xhci).max_interrupters }) as i32) {
if !unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(i as isize) }.is_null() {
xhci_remove_interrupter(xhci, unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(i as isize) });
xhci_free_interrupter(xhci, unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(i as isize) });
unsafe {
*unsafe { (*xhci).interrupters }.wrapping_offset(i as isize) = core::ptr::null_mut::<xhci_interrupter>();
}
}
i = i.wrapping_add(1);
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Freed interrupters".as_ptr() as *const u8) };
if !unsafe { (*xhci).cmd_ring }.is_null() {
xhci_ring_free(xhci, unsafe { (*xhci).cmd_ring });
}
unsafe {
(*xhci).cmd_ring = core::ptr::null_mut::<xhci_ring>();
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Freed command ring".as_ptr() as *const u8) };
unsafe { xhci_cleanup_command_queue(xhci) };
if !unsafe { (*xhci).devs }.is_null() {
i = (unsafe { (*xhci).max_slots }) as i32;
while i > 0 {
xhci_free_virt_devices_depth_first(xhci, i);
i = i.wrapping_sub(1);
}
unsafe { kfree((*xhci).devs as *const ()) };
}
unsafe { dma_pool_destroy((*xhci).segment_pool) };
unsafe {
(*xhci).segment_pool = core::ptr::null_mut::<dma_pool>();
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Freed segment pool".as_ptr() as *const u8) };
unsafe { dma_pool_destroy((*xhci).device_pool) };
unsafe {
(*xhci).device_pool = core::ptr::null_mut::<dma_pool>();
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Freed device context pool".as_ptr() as *const u8) };
unsafe { dma_pool_destroy((*xhci).small_streams_pool) };
unsafe {
(*xhci).small_streams_pool = core::ptr::null_mut::<dma_pool>();
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Freed small stream array pool".as_ptr() as *const u8) };
unsafe { dma_pool_destroy((*xhci).port_bw_pool) };
unsafe {
(*xhci).port_bw_pool = core::ptr::null_mut::<dma_pool>();
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Freed xhci port bw array pool".as_ptr() as *const u8) };
unsafe { dma_pool_destroy((*xhci).medium_streams_pool) };
unsafe {
(*xhci).medium_streams_pool = core::ptr::null_mut::<dma_pool>();
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Freed medium stream array pool".as_ptr() as *const u8) };
let dcbaa: *mut xhci_device_context_array = unsafe { core::ptr::addr_of_mut!((*xhci).dcbaa) };
if !unsafe { (*dcbaa).ctx_array }.is_null() {
dma_free_coherent(dev, array_size!(core::mem::size_of::<u64>() as u64, ((unsafe { (*xhci).max_slots }) as i32).wrapping_add(1)), (unsafe { (*dcbaa).ctx_array }) as *mut (), unsafe { (*dcbaa).dma });
unsafe {
(*dcbaa).ctx_array = core::ptr::null_mut::<u64>();
}
}
scratchpad_free(xhci);
if !unsafe { (*xhci).rh_bw }.is_null() {
xhci_rh_bw_cleanup(xhci);
}
unsafe {
(*xhci).cmd_ring_reserved_trbs = 0;
}
unsafe {
(*xhci).usb2_rhub.num_ports = 0;
}
unsafe {
(*xhci).usb3_rhub.num_ports = 0;
}
unsafe {
(*xhci).num_active_eps = 0;
}
unsafe { kfree((*xhci).usb2_rhub.ports as *const ()) };
unsafe { kfree((*xhci).usb3_rhub.ports as *const ()) };
unsafe { kfree((*xhci).hw_ports as *const ()) };
unsafe { kfree((*xhci).rh_bw as *const ()) };
i = 0;
while (i as u32) < unsafe { (*xhci).num_port_caps } {
unsafe { kfree((*unsafe { (*xhci).port_caps }.wrapping_offset(i as isize)).psi as *const ()) };
i = i.wrapping_add(1);
}
unsafe { kfree((*xhci).port_caps as *const ()) };
unsafe { kfree((*xhci).interrupters as *const ()) };
unsafe {
(*xhci).num_port_caps = 0;
}
unsafe {
(*xhci).usb2_rhub.ports = core::ptr::null_mut::<*mut xhci_port>();
}
unsafe {
(*xhci).usb3_rhub.ports = core::ptr::null_mut::<*mut xhci_port>();
}
unsafe {
(*xhci).hw_ports = core::ptr::null_mut::<xhci_port>();
}
unsafe {
(*xhci).rh_bw = core::ptr::null_mut::<xhci_root_port_bw_info>();
}
unsafe {
(*xhci).port_caps = core::ptr::null_mut::<xhci_port_cap>();
}
unsafe {
(*xhci).interrupters = core::ptr::null_mut::<*mut xhci_interrupter>();
}
unsafe {
(*xhci).devs = core::ptr::null_mut::<*mut xhci_virt_device>();
}
unsafe {
(*xhci).usb2_rhub.bus_state.bus_suspended = 0;
}
unsafe {
(*xhci).usb3_rhub.bus_state.bus_suspended = 0;
}
}

fn xhci_set_hc_event_deq(xhci: *mut xhci_hcd, ir: *mut xhci_interrupter) {
let deq: u64 = unsafe { xhci_trb_virt_to_dma((*(*ir).event_ring).deq_seg, (*(*ir).event_ring).dequeue) };
if deq == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN something wrong with SW event ring dequeue ptr.\n".as_ptr() as *const u8) };
}
// Update HC event ring dequeue pointer
//
// Don't clear the EHB bit (which is RW1C) because
// there might be more events to service.
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"// Write event ring dequeue pointer, preserving EHB bit".as_ptr() as *const u8) };
xhci_write_64(xhci, deq & GENMASK_ULL__6!(63i32, 4i32), unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_dequeue) });
}

fn xhci_add_in_port(xhci: *mut xhci_hcd, num_ports: u32, addr: *mut u32, max_caps: i32) {
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut temp: u32 = 0;
let mut port_offset: u32 = 0;
let mut port_count: u32 = 0;
let mut i: i32 = 0;
let mut major_revision: u8 = 0;
let mut minor_revision: u8 = 0;
let mut tmp_minor_revision: u8 = 0;
let mut rhub: *mut xhci_hub = unsafe { core::mem::zeroed() };
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
temp = readl(addr as *const ());
major_revision = XHCI_EXT_PORT_MAJOR!(temp) as u8;
minor_revision = XHCI_EXT_PORT_MINOR!(temp) as u8;
if (major_revision as i32) == 3 {
rhub = unsafe { core::ptr::addr_of_mut!((*xhci).usb3_rhub) };
// Some hosts incorrectly use sub-minor version for minor
// version (i.e. 0x02 instead of 0x20 for bcdUSB 0x320 and 0x01
// for bcdUSB 0x310). Since there is no USB release with sub
// minor version 0x301 to 0x309, we can assume that they are
// incorrect and fix it here.
if (minor_revision as i32) > 0 && (minor_revision as i32) < 16 {
minor_revision = ((minor_revision as i32) << 4i32) as u8;
}
// Some zhaoxin's xHCI controller that follow usb3.1 spec
// but only support Gen1.
if (unsafe { (*xhci).quirks }) & BIT_ULL!(46i32) != 0 {
tmp_minor_revision = minor_revision;
minor_revision = 0;
}
} else {
if (major_revision as i32) <= 2 {
rhub = unsafe { core::ptr::addr_of_mut!((*xhci).usb2_rhub) };
} else {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Ignoring unknown port speed, Ext Cap %p, revision = 0x%x\n".as_ptr() as *const u8, addr, major_revision as i32) };
// Ignoring port protocol we can't understand. FIXME
return;
}
}
// Port offset and count in the third dword, see section 7.2
temp = readl(addr.wrapping_offset(2 as isize) as *const ());
port_offset = XHCI_EXT_PORT_OFF!(temp);
port_count = XHCI_EXT_PORT_COUNT!(temp);
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Ext Cap %p, port offset = %u, count = %u, revision = 0x%x".as_ptr() as *const u8, addr, port_offset, port_count, major_revision as i32) };
// Port count includes the current port offset
if port_offset == 0 || port_offset.wrapping_add(port_count).wrapping_sub(1) > num_ports {
return;
}
// WTF? "Valid values are '1' to MaxPorts"
let __v4: u32 = unsafe { (*xhci).num_port_caps };
unsafe {
(*xhci).num_port_caps = __v4.wrapping_add(1);
}
let port_cap: *mut xhci_port_cap = unsafe { (*xhci).port_caps }.wrapping_offset(__v4 as isize);
if (unsafe { (*xhci).num_port_caps }) > (max_caps as u32) {
return;
}
unsafe {
(*port_cap).psi_count = XHCI_EXT_PORT_PSIC!(temp) as u8;
}
if (unsafe { (*port_cap).psi_count }) != 0 {
let mut __v6: *mut ();
let mut __v7: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_array_node_noprof((unsafe { (*port_cap).psi_count }) as u64, core::mem::size_of::<u32>() as u64, 3520u32, dev_to_node(dev), unsafe { *{
__slate_compound_8 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_8)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_array_node_noprof((unsafe { (*port_cap).psi_count }) as u64, core::mem::size_of::<u32>() as u64, 3520u32, dev_to_node(dev), unsafe { *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
} });
}
__v7 = _res;
__v6 = __v7;
unsafe {
(*port_cap).psi = __v6 as *mut u32;
}
if unsafe { (*port_cap).psi }.is_null() {
unsafe {
(*port_cap).psi_count = 0;
}
}
unsafe {
(*port_cap).psi_uid_count = ((*port_cap).psi_uid_count as i32).wrapping_add(1) as u8;
}
i = 0;
while i < ((unsafe { (*port_cap).psi_count }) as i32) {
unsafe {
*unsafe { (*port_cap).psi }.wrapping_offset(i as isize) = readl(addr.wrapping_offset(4 as isize).wrapping_offset(i as isize) as *const ());
}
// count unique ID values, two consecutive entries can
// have the same ID if link is assymetric
if i != 0 && XHCI_EXT_PORT_PSIV!(unsafe { *unsafe { (*port_cap).psi }.wrapping_offset(i as isize) }) != XHCI_EXT_PORT_PSIV!(unsafe { *unsafe { (*port_cap).psi }.wrapping_offset(i.wrapping_sub(1) as isize) }) {
unsafe {
(*port_cap).psi_uid_count = ((*port_cap).psi_uid_count as i32).wrapping_add(1) as u8;
}
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(46i32) != 0 && (major_revision as i32) == 3 && XHCI_EXT_PORT_PSIV!(unsafe { *unsafe { (*port_cap).psi }.wrapping_offset(i as isize) }) >= 5 {
minor_revision = tmp_minor_revision;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"PSIV:%d PSIE:%d PLT:%d PFD:%d LP:%d PSIM:%d\n".as_ptr() as *const u8, XHCI_EXT_PORT_PSIV!(unsafe { *unsafe { (*port_cap).psi }.wrapping_offset(i as isize) }), XHCI_EXT_PORT_PSIE!(unsafe { *unsafe { (*port_cap).psi }.wrapping_offset(i as isize) }), XHCI_EXT_PORT_PLT!(unsafe { *unsafe { (*port_cap).psi }.wrapping_offset(i as isize) }), XHCI_EXT_PORT_PFD!(unsafe { *unsafe { (*port_cap).psi }.wrapping_offset(i as isize) }), XHCI_EXT_PORT_LP!(unsafe { *unsafe { (*port_cap).psi }.wrapping_offset(i as isize) }), XHCI_EXT_PORT_PSIM!(unsafe { *unsafe { (*port_cap).psi }.wrapping_offset(i as isize) })) };
}
i = i.wrapping_add(1);
}
}
unsafe {
(*rhub).maj_rev = major_revision;
}
if ((unsafe { (*rhub).min_rev }) as i32) < (minor_revision as i32) {
unsafe {
(*rhub).min_rev = minor_revision;
}
}
unsafe {
(*port_cap).maj_rev = major_revision;
}
unsafe {
(*port_cap).min_rev = minor_revision;
}
unsafe {
(*port_cap).protocol_caps = temp;
}
if ((unsafe { (*xhci).hci_version }) as i32) >= 256 && (major_revision as i32) != 3 && temp & 524288u32 != 0 {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"xHCI 1.0: support USB2 hardware lpm".as_ptr() as *const u8) };
unsafe {
(*xhci).__slate_bits_0.__set_hw_lpm_support(1u32);
}
}
port_offset = port_offset.wrapping_sub(1);
i = port_offset as i32;
while (i as u32) < port_offset.wrapping_add(port_count) {
let hw_port: *mut xhci_port = unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize);
// Duplicate entry.  Ignore the port if the revisions differ.
if !unsafe { (*hw_port).rhub }.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Duplicate port entry, Ext Cap %p, port %u\n".as_ptr() as *const u8, addr, i) };
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Port was marked as USB %u, duplicated as USB %u\n".as_ptr() as *const u8, (*(*hw_port).rhub).maj_rev as i32, major_revision as i32) };
// Only adjust the roothub port counts if we haven't
// found a similar duplicate.
if (unsafe { (*hw_port).rhub }) != rhub && (unsafe { (*hw_port).hcd_portnum }) != ((1i32.wrapping_neg() as u8) as i32) {
let __v24: *mut xhci_hub = unsafe { (*hw_port).rhub };
let __v25: u32 = unsafe { (*__v24).num_ports };
unsafe {
(*__v24).num_ports = __v25.wrapping_sub(1);
}
unsafe {
(*hw_port).hcd_portnum = (1i32.wrapping_neg() as u8) as i32;
}
}
} else {
unsafe {
(*hw_port).rhub = rhub;
}
unsafe {
(*hw_port).port_cap = port_cap;
}
let __v29: u32 = unsafe { (*rhub).num_ports };
unsafe {
(*rhub).num_ports = __v29.wrapping_add(1);
}
}
i = i.wrapping_add(1);
}
// FIXME: Should we disable ports not in the Extended Capabilities?
}

fn xhci_create_rhub_port_array(xhci: *mut xhci_hcd, rhub: *mut xhci_hub, max_ports: u32, flags: u32) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut port_index: i32 = 0;
let mut i: i32 = 0;
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
if (unsafe { (*rhub).num_ports }) == 0 {
while false {
}
unsafe { _dev_info((*xhci_to_hcd(xhci)).self_.controller as *const device, c"USB%u root hub has no ports\n".as_ptr() as *const u8, (*rhub).maj_rev as i32) };
return;
}
// Place limits on the number of roothub ports so that the hub
// descriptors aren't longer than the USB core will allocate.
if (unsafe { (*rhub).num_ports }) > max_ports {
unsafe {
(*xhci).usb3_rhub.num_ports = max_ports;
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Limiting USB%u root hub ports to %u".as_ptr() as *const u8, (*rhub).maj_rev as i32, max_ports) };
}
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_array_node_noprof((unsafe { (*rhub).num_ports }) as u64, core::mem::size_of::<*mut xhci_port>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_array_node_noprof((unsafe { (*rhub).num_ports }) as u64, core::mem::size_of::<*mut xhci_port>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
unsafe {
(*rhub).ports = __v1 as *mut *mut xhci_port;
}
if unsafe { (*rhub).ports }.is_null() {
return;
}
while i < ((unsafe { (*xhci).max_ports }) as i32) {
if !((unsafe { (*unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize)).rhub }) != rhub || (unsafe { (*unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize)).hcd_portnum }) == ((1i32.wrapping_neg() as u8) as i32)) {
unsafe {
(*unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize)).hcd_portnum = port_index;
}
unsafe {
*unsafe { (*rhub).ports }.wrapping_offset(port_index as isize) = unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize);
}
port_index = port_index.wrapping_add(1);
if (port_index as u32) == unsafe { (*rhub).num_ports } {
break;
}
}
i = i.wrapping_add(1);
}
}

/// Scan the Extended Capabilities for the "Supported Protocol Capabilities" that
/// specify what speeds each port is supposed to be.  We can't count on the port
/// speed bits in the PORTSC register being correct until a device is connected,
/// but we need to set up the two fake roothubs with the correct number of USB
/// 3.0 and USB 2.0 ports at host controller initialization time.
fn xhci_setup_port_arrays(xhci: *mut xhci_hcd, flags: u32) -> i32 {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_25: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_27: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut offset: u32 = 0;
let mut i: i32 = 0;
let mut cap_count: i32 = 0;
let mut cap_start: u32 = 0;
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let mut __v0: *mut ();
let mut __v1: *mut ();
{
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_array_node_noprof((unsafe { (*xhci).max_ports }) as u64, core::mem::size_of::<xhci_port>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_array_node_noprof((unsafe { (*xhci).max_ports }) as u64, core::mem::size_of::<xhci_port>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
}
__v0 = __v1;
unsafe {
(*xhci).hw_ports = __v0 as *mut xhci_port;
}
if unsafe { (*xhci).hw_ports }.is_null() {
return 12i32.wrapping_neg();
}
while i < ((unsafe { (*xhci).max_ports }) as i32) {
unsafe {
(*unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize)).port_reg = (core::ptr::addr_of_mut!((*(*xhci).op_regs).port_regs) as *mut xhci_port_regs).wrapping_offset(i as isize);
}
unsafe {
(*unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize)).hw_portnum = i;
}
init_completion(unsafe { core::ptr::addr_of_mut!((*unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize)).rexit_done) });
init_completion(unsafe { core::ptr::addr_of_mut!((*unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize)).u3exit_done) });
i = i.wrapping_add(1);
}
let mut __v8: *mut ();
let mut __v9: *mut ();
{
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_array_node_noprof((unsafe { (*xhci).max_ports }) as u64, core::mem::size_of::<xhci_root_port_bw_info>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_array_node_noprof((unsafe { (*xhci).max_ports }) as u64, core::mem::size_of::<xhci_root_port_bw_info>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
} });
}
__v9 = _res;
}
__v8 = __v9;
unsafe {
(*xhci).rh_bw = __v8 as *mut xhci_root_port_bw_info;
}
if unsafe { (*xhci).rh_bw }.is_null() {
return 12i32.wrapping_neg();
}
i = 0;
while i < ((unsafe { (*xhci).max_ports }) as i32) {
let mut bw_table: *mut xhci_interval_bw_table = unsafe { core::mem::zeroed() };
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*unsafe { (*xhci).rh_bw }.wrapping_offset(i as isize)).tts) });
bw_table = unsafe { core::ptr::addr_of_mut!((*unsafe { (*xhci).rh_bw }.wrapping_offset(i as isize)).bw_table) };
for j in 0i32..16i32 {
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*((*bw_table).interval_bw.as_mut_ptr() as *mut xhci_interval_bw).wrapping_offset(j as isize)).endpoints) });
}
i = i.wrapping_add(1);
}
let base: *mut () = (unsafe { core::ptr::addr_of_mut!((*(*xhci).cap_regs).hc_capbase) }) as *mut ();
cap_start = xhci_find_next_ext_cap(base, 0u32, 2i32) as u32;
if cap_start == 0 {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"No Extended Capability registers, unable to set up roothub\n".as_ptr() as *const u8) };
return 19i32.wrapping_neg();
}
offset = cap_start;
// count extended protocol capability entries for later caching
while offset != 0 {
cap_count = cap_count.wrapping_add(1);
offset = xhci_find_next_ext_cap(base, offset, 2i32) as u32;
}
let mut __v23: *mut ();
let mut __v24: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_array_node_noprof(cap_count as u64, core::mem::size_of::<xhci_port_cap>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_25 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_25)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_array_node_noprof(cap_count as u64, core::mem::size_of::<xhci_port_cap>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_27 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_27)
} });
}
__v24 = _res;
__v23 = __v24;
unsafe {
(*xhci).port_caps = __v23 as *mut xhci_port_cap;
}
if unsafe { (*xhci).port_caps }.is_null() {
return 12i32.wrapping_neg();
}
offset = cap_start;
while offset != 0 {
xhci_add_in_port(xhci, (unsafe { (*xhci).max_ports }) as u32, ((unsafe { (base as *const u8).wrapping_offset(offset as isize) }) as *mut ()) as *mut u32, cap_count);
if unsafe { (*xhci).usb2_rhub.num_ports }.wrapping_add(unsafe { (*xhci).usb3_rhub.num_ports }) == (((unsafe { (*xhci).max_ports }) as i32) as u32) {
break;
}
offset = xhci_find_next_ext_cap(base, offset, 2i32) as u32;
}
if (unsafe { (*xhci).usb2_rhub.num_ports }) == 0 && (unsafe { (*xhci).usb3_rhub.num_ports }) == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"No ports on the roothubs?\n".as_ptr() as *const u8) };
return 19i32.wrapping_neg();
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Found %u USB 2.0 ports and %u USB 3.0 ports.".as_ptr() as *const u8, (*xhci).usb2_rhub.num_ports, (*xhci).usb3_rhub.num_ports) };
xhci_create_rhub_port_array(xhci, unsafe { core::ptr::addr_of_mut!((*xhci).usb2_rhub) }, 31u32, flags);
xhci_create_rhub_port_array(xhci, unsafe { core::ptr::addr_of_mut!((*xhci).usb3_rhub) }, 15u32, flags);
0
}

fn xhci_alloc_interrupter(xhci: *mut xhci_hcd, mut segs: u32, flags: u32) -> *mut xhci_interrupter {
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let mut ir: *mut xhci_interrupter = unsafe { core::mem::zeroed() };
let mut max_segs: u32 = 0;
let mut ret: i32 = 0;
if segs == 0 {
segs = 2;
}
let mut __v0: u64;
while false {
}
let mut __v2: u64;
while false {
}
while false {
}
loop {
if !((core::mem::size_of::<__SlateRecord_xhci_mem_c_2304_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_2304_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & 0u64 != 0 {
unsafe { __compiletime_assert_965() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_mem_c_2304_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_2304_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_2304_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_mem_c_2304_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_966() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (((unsafe { (*xhci).hcs_params2 }) as u64) & (core::mem::size_of::<__SlateRecord_xhci_mem_c_2304_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))) >> ((core::mem::size_of::<__SlateRecord_xhci_mem_c_2304_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32);
__v0 = __v2;
max_segs = (UL!() << __v0) as u32;
let mut __v7: u32;
let __UNIQUE_ID_x__967: u32 = segs;
let __UNIQUE_ID_y__968: u32 = max_segs;
while false {
}
__v7 = if __UNIQUE_ID_x__967 < __UNIQUE_ID_y__968 { __UNIQUE_ID_x__967 } else { __UNIQUE_ID_y__968 };
segs = __v7;
let mut __v9: *mut ();
let mut __v10: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_interrupter>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_node_noprof(core::mem::size_of::<xhci_interrupter>() as u64, flags | 256u32, dev_to_node(dev), unsafe { *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
} });
}
__v10 = _res;
__v9 = __v10;
ir = __v9 as *mut xhci_interrupter;
if ir.is_null() {
return core::ptr::null_mut::<xhci_interrupter>();
}
unsafe {
(*ir).event_ring = xhci_ring_alloc(xhci, segs, 6u32, 0u32, flags);
}
if unsafe { (*ir).event_ring }.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Failed to allocate interrupter event ring\n".as_ptr() as *const u8) };
unsafe { kfree(ir as *const ()) };
return core::ptr::null_mut::<xhci_interrupter>();
}
ret = xhci_alloc_erst(xhci, unsafe { (*ir).event_ring }, unsafe { core::ptr::addr_of_mut!((*ir).erst) }, flags);
if ret != 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Failed to allocate interrupter erst\n".as_ptr() as *const u8) };
xhci_ring_free(xhci, unsafe { (*ir).event_ring });
unsafe { kfree(ir as *const ()) };
return core::ptr::null_mut::<xhci_interrupter>();
}
ir
}

#[unsafe(no_mangle)]
extern "C" fn xhci_add_interrupter(xhci: *mut xhci_hcd, intr_num: u32) {
let mut ir: *mut xhci_interrupter = unsafe { core::mem::zeroed() };
let mut erst_base: u64 = 0;
let mut erst_size: u32 = 0;
ir = unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(intr_num as isize) };
unsafe {
(*ir).intr_num = intr_num;
}
unsafe {
(*ir).ir_set = ((*(*xhci).run_regs).ir_set.as_mut_ptr() as *mut xhci_intr_reg).wrapping_offset(intr_num as isize);
}
// set ERST count with the number of entries in the segment table
erst_size = readl((unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_size) }) as *const ());
let __v1: u32 = erst_size & 4294901760u32;
erst_size = __v1;
erst_size |= unsafe { (*(*ir).event_ring).num_segs };
writel(erst_size, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_size) }) as *mut ());
erst_base = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_base) });
let __v4: u64 = erst_base;
erst_base = __v4 & !GENMASK_ULL__12!(63i32, 6i32);
let __v6: u64 = erst_base;
erst_base = __v6 | (unsafe { (*ir).erst.erst_dma_addr }) & GENMASK_ULL__9!(63i32, 6i32);
if (unsafe { (*xhci).quirks }) & BIT_ULL!(47i32) != 0 {
hi_lo_writeq(erst_base, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_base) }) as *mut ());
} else {
xhci_write_64(xhci, erst_base, unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_base) });
}
// Set the event ring dequeue address of this interrupter
xhci_set_hc_event_deq(xhci, ir);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci_mem.xhci_create_secondary_interrupter")]
extern "C" fn xhci_create_secondary_interrupter(hcd: *mut usb_hcd, segs: u32, imod_interval: u32, intr_num: u32) -> *mut xhci_interrupter {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
let mut i: u32 = 0;
let mut err: i32 = 28i32.wrapping_neg();
if unsafe { (*xhci).interrupters }.is_null() || ((unsafe { (*xhci).max_interrupters }) as i32) <= 1 || intr_num >= (((unsafe { (*xhci).max_interrupters }) as i32) as u32) {
return core::ptr::null_mut::<xhci_interrupter>();
}
let ir: *mut xhci_interrupter = xhci_alloc_interrupter(xhci, segs, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if ir.is_null() {
return core::ptr::null_mut::<xhci_interrupter>();
}
xhci_ring_init(xhci, unsafe { (*ir).event_ring });
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
if intr_num == 0 {
// Find available secondary interrupter, interrupter 0 is reserved for primary
i = 1;
while i < (((unsafe { (*xhci).max_interrupters }) as i32) as u32) {
if unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(i as isize) }.is_null() {
unsafe {
*unsafe { (*xhci).interrupters }.wrapping_offset(i as isize) = ir;
}
xhci_add_interrupter(xhci, i);
err = 0;
break;
}
i = i.wrapping_add(1);
}
} else {
if unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(intr_num as isize) }.is_null() {
unsafe {
*unsafe { (*xhci).interrupters }.wrapping_offset(intr_num as isize) = ir;
}
xhci_add_interrupter(xhci, intr_num);
err = 0;
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
if err != 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Failed to add secondary interrupter, max interrupters %d\n".as_ptr() as *const u8, (*xhci).max_interrupters as i32) };
xhci_free_interrupter(xhci, ir);
return core::ptr::null_mut::<xhci_interrupter>();
}
unsafe { xhci_set_interrupter_moderation(ir, imod_interval) };
ir
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_create_secondary_interrupter_970: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, u32, u32, u32) -> *mut xhci_interrupter>, *mut ()>(Some(xhci_create_secondary_interrupter)) };

#[unsafe(no_mangle)]
extern "C" fn xhci_mem_init(xhci: *mut xhci_hcd, flags: u32) -> i32 {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut dcbaa: *mut xhci_device_context_array = unsafe { core::mem::zeroed() };
let mut dev: *mut device = unsafe { core::mem::zeroed() };
unsafe {
dev = (*xhci_to_hcd(xhci)).self_.sysdev;
dcbaa = core::ptr::addr_of_mut!((*xhci).dcbaa);
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Starting %s".as_ptr() as *const u8, c"xhci_mem_init".as_ptr() as *mut u8);
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Allocating internal virtual device array".as_ptr() as *const u8);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kmalloc_array_node_noprof(((*xhci).max_slots as i32).wrapping_add(1) as u64, core::mem::size_of::<*mut xhci_virt_device>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kmalloc_array_node_noprof(((*xhci).max_slots as i32).wrapping_add(1) as u64, core::mem::size_of::<*mut xhci_virt_device>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
});
}
(*xhci).devs = *__slate_slot_5 as *mut *mut xhci_virt_device;
if !unsafe { (*xhci).devs }.is_null() {
(*xhci).dcbaa.ctx_array = dma_alloc_coherent(dev, array_size!(core::mem::size_of::<u64>() as u64, ((*xhci).max_slots as i32).wrapping_add(1)), core::ptr::addr_of_mut!((*dcbaa).dma), flags) as *mut u64;
if !unsafe { (*dcbaa).ctx_array }.is_null() {
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Device context base array address = %pad (DMA), %p (virt)".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dcbaa).dma), (*dcbaa).ctx_array);
// Initialize the ring segment pool.  The ring must be a contiguous
// structure comprised of TRBs.  The TRBs must be 16 byte aligned,
// however, the command ring segment needs 64-byte aligned segments
// and our use of dma addresses in the trb_address_map radix tree needs
// TRB_SEGMENT_SIZE alignment, so we pick the greater alignment need.
if (*xhci).quirks & BIT_ULL!(45i32) != 0 {
(*xhci).segment_pool = dma_pool_create(c"xHCI ring segments".as_ptr() as *const u8, dev, 8192u64, 8192u64, unsafe { (*xhci).page_size }.wrapping_mul(2) as u64);
} else {
(*xhci).segment_pool = dma_pool_create(c"xHCI ring segments".as_ptr() as *const u8, dev, 4096u64, 4096u64, (*xhci).page_size as u64);
}
// Buggy HC prefetches beyond segment bounds - allocate dummy space at the end
if !unsafe { (*xhci).segment_pool }.is_null() {
// See Table 46 and Note on Figure 55
(*xhci).device_pool = dma_pool_create(c"xHCI input/output contexts".as_ptr() as *const u8, dev, 2112u64, 64u64, (*xhci).page_size as u64);
if !unsafe { (*xhci).device_pool }.is_null() {
// Linear stream context arrays don't have any boundary restrictions,
// and only need to be 16-byte aligned.
(*xhci).small_streams_pool = dma_pool_create(c"xHCI 256 byte stream ctx arrays".as_ptr() as *const u8, dev, 256u64, 16u64, 0u64);
if !unsafe { (*xhci).small_streams_pool }.is_null() {
// Any stream context array bigger than MEDIUM_STREAM_ARRAY_SIZE will be
// allocated with dma_alloc_coherent().
(*xhci).medium_streams_pool = dma_pool_create(c"xHCI 1KB stream ctx arrays".as_ptr() as *const u8, dev, 1024u64, 16u64, 0u64);
if !unsafe { (*xhci).medium_streams_pool }.is_null() {
// refer to xhci rev1_2 protocol 5.3.3 max ports is 255.
// refer to xhci rev1_2 protocol 6.4.3.14 port bandwidth buffer need
// to be 16-byte aligned.
(*xhci).port_bw_pool = dma_pool_create(c"xHCI 256 port bw ctx arrays".as_ptr() as *const u8, dev, 256u64, 16u64, 0u64);
if !unsafe { (*xhci).port_bw_pool }.is_null() {
// Set up the command ring to have one segments for now.
(*xhci).cmd_ring = xhci_ring_alloc(xhci, 1u32, 5u32, 0u32, flags);
if !unsafe { (*xhci).cmd_ring }.is_null() {
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Allocated command ring at %p".as_ptr() as *const u8, (*xhci).cmd_ring);
// Allocate and set up primary interrupter 0 with an event ring.
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Allocating primary event ring".as_ptr() as *const u8);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kmalloc_array_node_noprof((*xhci).max_interrupters as u64, core::mem::size_of::<*mut xhci_interrupter>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kmalloc_array_node_noprof((*xhci).max_interrupters as u64, core::mem::size_of::<*mut xhci_interrupter>() as u64, flags | 256u32, dev_to_node(dev), *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
});
}
(*xhci).interrupters = *__slate_slot_1 as *mut *mut xhci_interrupter;
if !unsafe { (*xhci).interrupters }.is_null() {
*unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize) = xhci_alloc_interrupter(xhci, 0u32, flags);
if !unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize) }.is_null() {
if scratchpad_alloc(xhci, flags) == 0 {
if xhci_setup_port_arrays(xhci, flags) == 0 {
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Finished %s".as_ptr() as *const u8, c"xhci_mem_init".as_ptr() as *mut u8);
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
xhci_halt(xhci);
xhci_reset(xhci, 250000u64);
xhci_mem_cleanup(xhci);
return 12i32.wrapping_neg();
}
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_xhci_get_endpoint_interval_4: [u8; 1] = *b"\0";

static __slate_asm_string_xhci_get_endpoint_interval_5: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/xhci-mem.c\0";

static __slate_asm_string_xhci_remove_stream_mapping_1: [u8; 1] = *b"\0";

static __slate_asm_string_xhci_remove_stream_mapping_2: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/xhci-mem.c\0";

static __slate_asm_string_xhci_update_stream_segment_mapping_6: [u8; 1] = *b"\0";

static __slate_asm_string_xhci_update_stream_segment_mapping_7: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/xhci-mem.c\0";
