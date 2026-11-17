
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
//! PCI detection and setup code
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
//! @a is a power of 2 value
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
//! upper_32_bits - return bits 32-63 of a number
//! @n: the number we're accessing
//!
//! A basic shift-right of a 64- or 32-bit quantity.  Use this to suppress
//! the "right shift count >= width of type" warning when that quantity is
//! 32-bits.
//! SCHED_ISO: reserved but not implemented yet
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
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
//! Conventional PCI and PCI-X Mode 1 devices have 256 bytes of
//! configuration space.  PCI-X Mode 2 and PCIe devices have 4096 bytes of
//! configuration space.
//! Vendors and devices.  Sort key: vendor first, device next.
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
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
//! DWORD 3
//! AER and DPC capabilities TLP Logging register sizes (PCIe r6.2, sec 7.8.4
//! & 7.9.14).
//! Runtime PM flag argument bits
//! FIELD_PREP() - prepare a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_val:  value to put in the field
//!
//! FIELD_PREP() masks and shifts up the value.  The result should
//! be combined with other fields of the bitfield using logical OR.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Thunderbolt
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// lower_32_bits - return bits 0-31 of a number
// @n: the number we're accessing
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// The inline keyword gives the compiler room to decide to inline, or
// not inline a function as it sees best.  However, as these functions
// are called in both __init and non-__init functions, if they are not
// inlined we will end up with a section mismatch error (of the type of
// freeable items not being freed).  So we must use __always_inline here
// to fix the problem.  If other functions in the future also end up in
// this situation they will also need to be annotated as __always_inline
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
// Set the IDR flag and the IDR_FREE tag
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// Number of standard BARs
// 16 bits
// 16 bits
// Enable response in I/O space
// Enable response in Memory space
// INTx Emulation Disable
// 16 bits
// Immediate Readiness
// High 24 bits are class, low 8 revision
// Reg. Level Programming Interface
// 8 bits
// Multi-Function Device (possible)
// Base addresses specify locations in memory or I/O space.
// Decoded size can be determined by writing a value of
// 0xffffffff to the register, and reading it back.  Only
// 1 bits are decoded.
// 32 bits
// 0 = memory, 1 = I/O
// 32 bit address
// Below 1M [obsolete]
// 64 bit address
// prefetchable?
// Bits 31..11 are address, 10..1 reserved
// 0x35-0x3b are reserved
// 8 bits
// 8 bits
// Header type 1 (PCI-to-PCI bridges)
// Primary bus number
// Highest bus number behind the bridge
// Masks for dword-sized processing of Bus Number and Sec Latency Timer fields
// I/O range behind the bridge
// I/O bridging type
// Standard 4K I/O windows
// Intel 1K I/O windows
// Secondary status register, only bit 14 used
// Memory range behind
// Prefetchable memory range behind
// Upper half of prefetchable memory range
// Upper half of I/O addresses
// 0x34 same as for htype 0
//
// 0x35-0x3b is reserved
// Same as PCI_ROM_ADDRESS, but for htype 1
// 0x3c-0x3d are same as for htype 0
// The same for SERR forwarding
// Report master aborts
// Accelerated Graphics Port
// PCI-X
// Bridge subsystem vendor/device ID
// AGP Target PCI-PCI bridge
// PCI Express
// PCI Enhanced Allocation
// Status register
// Control register
// First EA Entry in List
// EA fixed Secondary and Subordinate bus numbers for Bridge
// PCI-X capabilities
// 266 MHz capable
// 533 MHz capable
// Secondary Status
// 133 MHz capable
// Secondary Bus Mode and Frequency
// PCI-X Capability Version
// Mode 1 or Modes 1 and 2
// 266 MHz capable
// 533 MHz capable
// PCI Bridge subsystem vendor ID
// PCI Bridge subsystem device ID
// Capabilities register
// Device/Port type
// Express Endpoint
// Legacy Endpoint
// Root Port
// Upstream Port
// Downstream Port
// PCI/PCI-X to PCIe Bridge
// Root Complex Integrated Endpoint
// Root Complex Event Collector
// Device capabilities
// Max_Payload_Size
// Extended tags
// Device Control
// Enable relaxed ordering
// Extended Tag Field Enable
// Link Capabilities
// Max Link Speed (prior to PCIe r3.0: Supported Link Speeds)
// ASPM L0s Support
// ASPM L1 Support
// Data Link Layer Link Active Reporting Capable
// Link Control
// Read Completion Boundary
// Link Status
// Current Link Speed
// Negotiated Link Width
// Slot Capabilities
// Hot-Plug Capable
// Root Control
// Config RRS Software Visibility Enable
// Root Capabilities
// Config RRS Software Visibility
// The Device Capabilities 2, Device Status 2, Device Control 2,
// Link Capabilities 2, Link Status 2, Link Control 2,
// Slot Capabilities 2, Slot Status 2, and Slot Control 2 registers
// are only present on devices with PCIe Capability version 2.
// Use pcie_capability_read_word() and similar interfaces to use them
// safely.
// Device Capabilities 2
// End-End TLP Prefix
// Max End-End TLP Prefixes
// Link Control 2
// Supported Speed 2.5GT/s
// Supported Speed 5GT/s
// Supported Speed 8GT/s
// Supported Speed 16GT/s
// Supported Speed 32GT/s
// Supported Speed 64GT/s
// Link Status 2
// Flit Mode Status
// Alternate Routing ID
// Device 3 Capability/Control/Status
// Alternative Routing-ID Interpretation
// ARI Capability Register
// Number of VF BARs
// Device 3 Status Register
// Segment Captured (end-to-end flit-mode detected)
// CXL r4.0, 8.1.8: Flex Bus DVSEC
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// DWORD 5
// FIELD_GET() - extract a bitfield element
// @_mask: shifted mask defining the field's length and position
// @_reg:  value of entire bitfield
//
// FIELD_GET() extracts the field specified by @_mask from the
// bitfield passed in as @_reg by masking and shifting it down.
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
macro_rules! ARRAY_SIZE__3 {
() => {
((core::mem::size_of::<[*const u8; 26]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__5 {
() => {
((core::mem::size_of::<[*const u8; 11]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__6!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__8 {
() => {
((core::mem::size_of::<[u8; 16]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(__must_be_array__21!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__11 {
() => {
((core::mem::size_of::<[u32; 64]>() as u64) / (core::mem::size_of::<u32>() as u64)).wrapping_add(__must_be_array__7!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__14 {
() => {
((core::mem::size_of::<[u32; 64]>() as u64) / (core::mem::size_of::<u32>() as u64)).wrapping_add(__must_be_array__20!() as u64)
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

macro_rules! DMA_BIT_MASK__2 {
($n:expr) => {
GENMASK_ULL!($n.wrapping_sub(1), 0i32)
};
}

macro_rules! GENMASK_INPUT_CHECK__3 {
() => {
core::mem::size_of::<__SlateRecord_probe_c_2053>() as i32
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
(GENMASK_INPUT_CHECK__3!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK_ULL {
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

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

/// Next Function Number
macro_rules! PCI_ARI_CAP_NFN {
($x:expr) => {
(($x) as i32) >> 8i32 & 255
};
}

/// The PCI interface treats multi-function devices as independent
/// devices.  The slot/function address of each device is encoded
/// in a single byte as follows:
///
/// 	7:3 = slot
/// 	2:0 = function
macro_rules! PCI_DEVFN__2 {
($slot:expr, $func:expr) => {
(($slot) & 31) << 3i32 | ((($func) & 7) as u32)
};
}

macro_rules! PCI_FUNC__2 {
($devfn:expr) => {
($devfn) & 7
};
}

macro_rules! PCI_FUNC {
($devfn:expr) => {
($devfn) & 7
};
}

macro_rules! PCI_POSSIBLE_ERROR {
($val:expr) => {
($val) == 4294967295u32
};
}

macro_rules! PCI_SLOT {
($devfn:expr) => {
($devfn) >> 3i32 & 31
};
}

macro_rules! PCI_SLOT__2 {
($devfn:expr) => {
($devfn) >> 3i32 & 31
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

macro_rules! __bf_cast_unsigned__3 {
($x:expr) => {
($x) as u64
};
}

macro_rules! __bf_cast_unsigned__2 {
($x:expr) => {
($x) as u16
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

macro_rules! __must_be_array {
() => {
core::mem::size_of::<__SlateRecord_probe_c_824>() as i32
};
}

macro_rules! __must_be_array__6 {
() => {
core::mem::size_of::<__SlateRecord_probe_c_1988>() as i32
};
}

macro_rules! __must_be_array__7 {
() => {
core::mem::size_of::<__SlateRecord_probe_c_1966>() as i32
};
}

macro_rules! __must_be_array__20 {
() => {
core::mem::size_of::<__SlateRecord_probe_c_1962>() as i32
};
}

macro_rules! __must_be_array__21 {
() => {
core::mem::size_of::<__SlateRecord_probe_c_785>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__10 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
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
macro_rules! __sign_use__3 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
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

macro_rules! __types_ok__3 {
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
($ptr:expr) => {
{
let mut __v: *mut pci_domain_busn_res;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__27!() as isize)) }) as *mut ()) as *mut pci_domain_busn_res;
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
let mut __v: *mut resource_entry;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__9!() as isize)) }) as *mut ()) as *mut resource_entry;
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
($ptr:expr) => {
{
let mut __v: *mut pci_bus;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__5!() as isize)) }) as *mut ()) as *mut pci_bus;
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
($ptr:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut pci_dev;
__v
}
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
macro_rules! list_entry__3 {
($ptr:expr) => {
container_of__4!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__8 {
($ptr:expr) => {
container_of__9!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr) => {
container_of!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__2 {
($ptr:expr) => {
container_of__2!($ptr)
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
macro_rules! list_first_entry__3 {
($ptr:expr) => {
list_entry__3!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__11 {
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
macro_rules! list_first_entry__4 {
($ptr:expr) => {
list_entry__2!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__5 {
($pos:expr, $member:ident) => {
list_entry__2!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__12 {
($pos:expr, $member:ident) => {
list_entry__8!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__3 {
($pos:expr, $member:ident) => {
list_entry__3!(unsafe { (*$pos).$member.next })
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__5 {
() => {
core::mem::offset_of!(pci_bus, node) as u64
};
}

macro_rules! offsetof__27 {
() => {
core::mem::offset_of!(pci_domain_busn_res, list) as u64
};
}

macro_rules! offsetof__2 {
() => {
core::mem::offset_of!(pci_dev, bus_list) as u64
};
}

macro_rules! offsetof__4 {
() => {
0u64
};
}

macro_rules! offsetof__9 {
() => {
core::mem::offset_of!(resource_entry, node) as u64
};
}

macro_rules! pcibus_to_node {
($bus:expr) => {
__pcibus_to_node(($bus) as *const pci_bus)
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

macro_rules! to_pci_bus {
($n:expr) => {
{
let mut __v: *mut pci_bus;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(pci_bus) as isize)) }) as *mut ()) as *mut pci_bus;
__v
}
};
}

macro_rules! to_pci_dev {
($n:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(pci_dev) as isize)) }) as *mut ()) as *mut pci_dev;
__v
}
};
}

macro_rules! to_pci_host_bridge {
($n:expr) => {
{
let mut __v: *mut pci_host_bridge;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut pci_host_bridge;
__v
}
};
}

macro_rules! type_max {
() => {
__type_max!()
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__2 {
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
    static mut nr_node_ids: u32;
    static mut __cpu_online_mask: cpumask;
    static mut cpu_number: i32;
    static mut ioport_resource: resource;
    static mut iomem_resource: resource;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut pci_flags: u32;
    static mut pcie_bus_config: u32;
    static mut pci_bus_type: bus_type;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_pcie_link_event: static_call_key;
    static mut __tracepoint_pcie_link_event: tracepoint;
    static mut pci_early_dump: bool;
    static mut pci_bus_sem: rw_semaphore;
    static mut pci_dev_attr_groups: [*const attribute_group; 0];
    static mut pcibus_groups: [*const attribute_group; 0];
    static mut pci_hotplug_bus_size: u64;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn print_hex_dump(arg0: *const u8, arg1: *const u8, arg2: i32, arg3: i32, arg4: i32, arg5: *const (), arg6: u64, arg7: bool);
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn scnprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_5();
    fn ____wrong_branch_error() -> bool;
    fn bitmap_free(arg0: *const u64);
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn irq_to_desc(arg0: u32) -> *mut irq_desc;
    fn msleep(arg0: u32);
    fn request_resource_conflict(arg0: *mut resource, arg1: *mut resource) -> *mut resource;
    fn release_resource(arg0: *mut resource) -> i32;
    fn adjust_resource(arg0: *mut resource, arg1: u64, arg2: u64) -> i32;
    fn iomem_get_mapping() -> *mut address_space;
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn rcu_is_watching() -> bool;
    fn down_write(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn __compiletime_assert_179();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn bus_sort_breadthfirst(arg0: *const bus_type, arg1: Option<unsafe extern "C" fn(*const device, *const device) -> i32>);
    fn class_register(arg0: *const class) -> i32;
    fn __devm_add_action(arg0: *mut device, arg1: Option<unsafe extern "C" fn(*mut ())>, arg2: *mut (), arg3: *const u8) -> i32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_415();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn dev_set_name(arg0: *mut device, arg1: *const u8, ...) -> i32;
    fn device_register(arg0: *mut device) -> i32;
    fn device_initialize(arg0: *mut device);
    fn device_add(arg0: *mut device) -> i32;
    fn device_del(arg0: *mut device);
    fn get_device(arg0: *mut device) -> *mut device;
    fn put_device(arg0: *mut device);
    fn pci_find_host_bridge(arg0: *mut pci_bus) -> *mut pci_host_bridge;
    fn pcibios_resource_to_bus(arg0: *mut pci_bus, arg1: *mut pci_bus_region, arg2: *mut resource);
    fn pcibios_bus_to_resource(arg0: *mut pci_bus, arg1: *mut resource, arg2: *mut pci_bus_region);
    fn pci_find_bus(arg0: i32, arg1: i32) -> *mut pci_bus;
    fn pci_bus_add_devices(arg0: *const pci_bus);
    fn pci_dev_assign_slot(arg0: *mut pci_dev);
    fn pci_dev_put(arg0: *mut pci_dev);
    fn pci_find_capability(arg0: *mut pci_dev, arg1: i32) -> u8;
    fn pci_find_ext_capability(arg0: *mut pci_dev, arg1: i32) -> u16;
    fn pci_find_vsec_capability(arg0: *mut pci_dev, arg1: u16, arg2: i32) -> u16;
    fn pci_find_dvsec_capability(arg0: *mut pci_dev, arg1: u16, arg2: u16) -> u16;
    fn pci_get_slot(arg0: *mut pci_bus, arg1: u32) -> *mut pci_dev;
    fn pci_bus_read_config_dword(arg0: *mut pci_bus, arg1: u32, arg2: i32, arg3: *mut u32) -> i32;
    fn pci_read_config_byte(arg0: *const pci_dev, arg1: i32, arg2: *mut u8) -> i32;
    fn pci_read_config_word(arg0: *const pci_dev, arg1: i32, arg2: *mut u16) -> i32;
    fn pci_read_config_dword(arg0: *const pci_dev, arg1: i32, arg2: *mut u32) -> i32;
    fn pci_write_config_byte(arg0: *const pci_dev, arg1: i32, arg2: u8) -> i32;
    fn pci_write_config_word(arg0: *const pci_dev, arg1: i32, arg2: u16) -> i32;
    fn pci_write_config_dword(arg0: *const pci_dev, arg1: i32, arg2: u32) -> i32;
    fn pcie_capability_read_word(arg0: *mut pci_dev, arg1: i32, arg2: *mut u16) -> i32;
    fn pcie_capability_read_dword(arg0: *mut pci_dev, arg1: i32, arg2: *mut u32) -> i32;
    fn pcie_capability_clear_and_set_word_unlocked(arg0: *mut pci_dev, arg1: i32, arg2: u16, arg3: u16) -> i32;
    fn pcie_capability_clear_and_set_word_locked(arg0: *mut pci_dev, arg1: i32, arg2: u16, arg3: u16) -> i32;
    fn pcie_get_readrq(arg0: *mut pci_dev) -> i32;
    fn pcie_set_readrq(arg0: *mut pci_dev, arg1: i32) -> i32;
    fn pcie_get_mps(arg0: *mut pci_dev) -> i32;
    fn pcie_set_mps(arg0: *mut pci_dev, arg1: i32) -> i32;
    fn pci_bus_claim_resources(arg0: *mut pci_bus);
    fn pci_assign_unassigned_bridge_resources(arg0: *mut pci_dev);
    fn pci_assign_unassigned_bus_resources(arg0: *mut pci_bus);
    fn pci_assign_unassigned_root_bus_resources(arg0: *mut pci_bus);
    fn pci_add_resource(arg0: *mut list_head, arg1: *mut resource);
    fn pci_free_resource_list(arg0: *mut list_head);
    fn pci_bus_add_resource(arg0: *mut pci_bus, arg1: *mut resource);
    fn pci_bus_resource_n(arg0: *const pci_bus, arg1: i32) -> *mut resource;
    fn pci_bus_remove_resources(arg0: *mut pci_bus);
    fn pci_walk_bus(arg0: *mut pci_bus, arg1: Option<unsafe extern "C" fn(*mut pci_dev, *mut ()) -> i32>, arg2: *mut ());
    fn pci_bus_release_emul_domain_nr(arg0: i32);
    fn pcibios_assign_all_busses() -> u32;
    fn pci_fixup_device(arg0: u32, arg1: *mut pci_dev);
    fn pcibios_device_add(arg0: *mut pci_dev) -> i32;
    fn pcibios_release_device(arg0: *mut pci_dev);
    fn pci_host_bridge_acpi_msi_domain(arg0: *mut pci_bus) -> *mut irq_domain;
    fn arch_pci_dev_is_removable(arg0: *mut pci_dev) -> bool;
    fn irq_find_matching_fwspec(arg0: *mut irq_fwspec, arg1: u32) -> *mut irq_domain;
    fn pci_msi_get_device_domain(arg0: *mut pci_dev) -> *mut irq_domain;
    fn __pm_runtime_idle(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_resume(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_set_status(arg0: *mut device, arg1: u32) -> i32;
    fn pm_runtime_no_callbacks(arg0: *mut device);
    fn devm_pm_runtime_enable(arg0: *mut device) -> i32;
    fn __traceiter_pcie_link_event(arg0: *mut (), arg1: *mut pci_bus, arg2: u32, arg3: u32, arg4: u32) -> i32;
    fn __SCT__tp_func_pcie_link_event(arg0: *mut (), arg1: *mut pci_bus, arg2: u32, arg3: u32, arg4: u32) -> i32;
    fn __compiletime_assert_535();
    fn pci_init_reset_methods(arg0: *mut pci_dev);
    fn pci_allocate_cap_save_buffers(arg0: *mut pci_dev);
    fn pci_free_cap_save_buffers(arg0: *mut pci_dev);
    fn pci_pm_init(arg0: *mut pci_dev);
    fn pci_ea_init(arg0: *mut pci_dev);
    fn pci_msi_init(arg0: *mut pci_dev);
    fn pci_msix_init(arg0: *mut pci_dev);
    fn platform_pci_configure_wake(arg0: *mut pci_dev);
    fn pci_vpd_init(arg0: *mut pci_dev);
    fn pci_cardbus_scan_bridge_extend(arg0: *mut pci_bus, arg1: *mut pci_dev, arg2: u32, arg3: i32, arg4: u32, arg5: i32) -> i32;
    fn pci_configure_ari(arg0: *mut pci_dev);
    fn pci_resource_name(arg0: *mut pci_dev, arg1: u32) -> *const u8;
    fn pci_reassigndev_resource_alignment(arg0: *mut pci_dev);
    fn pci_bus_get(arg0: *mut pci_bus) -> *mut pci_bus;
    fn pci_bus_put(arg0: *mut pci_bus);
    fn pcie_get_supported_speeds(arg0: *mut pci_dev) -> u8;
    fn __pcie_print_link_status(arg0: *mut pci_dev, arg1: bool);
    fn __compiletime_assert_542();
    fn __compiletime_assert_543();
    fn __compiletime_assert_544();
    fn __compiletime_assert_545();
    fn __compiletime_assert_546();
    fn pci_rcec_init(arg0: *mut pci_dev);
    fn pci_rcec_exit(arg0: *mut pci_dev);
    fn pci_ats_init(arg0: *mut pci_dev);
    fn pci_pri_init(arg0: *mut pci_dev);
    fn pci_pasid_init(arg0: *mut pci_dev);
    fn pci_acs_init(arg0: *mut pci_dev);
    fn pcie_failed_link_retrain(arg0: *mut pci_dev) -> i32;
    fn pci_configure_aspm_l1ss(arg0: *mut pci_dev);
    fn pcie_aspm_init_link_state(arg0: *mut pci_dev);
    fn pci_configure_ltr(arg0: *mut pci_dev);
    fn pci_rebar_init(arg0: *mut pci_dev);
    fn pci_acpi_preserve_config(arg0: *mut pci_host_bridge) -> bool;
    fn pci_acpi_program_hp_params(arg0: *mut pci_dev) -> i32;
    fn pci_set_acpi_fwnode(arg0: *mut pci_dev);
    fn __compiletime_assert_554();
    fn __compiletime_assert_555();
    fn __compiletime_assert_556();
    fn __compiletime_assert_557();
    fn __compiletime_assert_558();
    fn __compiletime_assert_559();
    fn __compiletime_assert_560();
    fn __compiletime_assert_561();
    fn __compiletime_assert_562();
    fn __compiletime_assert_563();
    fn __compiletime_assert_564();
    fn __compiletime_assert_572();
    fn __compiletime_assert_573();
    fn __compiletime_assert_574();
    fn __compiletime_assert_575();
    fn __compiletime_assert_576();
    fn __compiletime_assert_584();
    fn __compiletime_assert_585();
    fn __compiletime_assert_586();
    fn __compiletime_assert_587();
    fn __compiletime_assert_588();
    fn __compiletime_assert_589();
    fn __compiletime_assert_590();
    fn __compiletime_assert_591();
    fn __compiletime_assert_592();
    fn __compiletime_assert_593();
    fn __compiletime_assert_594();
    fn __compiletime_assert_595();
    fn __compiletime_assert_596();
    fn __compiletime_assert_597();
    fn __compiletime_assert_598();
    fn __compiletime_assert_599();
    fn __compiletime_assert_600();
    fn __compiletime_assert_601();
    fn __compiletime_assert_602();
    fn __compiletime_assert_603();
    fn __compiletime_assert_604();
    fn __compiletime_assert_605();
    fn __compiletime_assert_606();
    fn __compiletime_assert_607();
    fn __compiletime_assert_608();
    fn __compiletime_assert_609();
    fn __compiletime_assert_610();
    fn __compiletime_assert_611();
    fn __compiletime_assert_612();
    fn __compiletime_assert_613();
    fn __compiletime_assert_614();
    fn __compiletime_assert_615();
    fn __compiletime_assert_616();
    fn __compiletime_assert_617();
    fn __compiletime_assert_618();
    fn __compiletime_assert_619();
    fn __compiletime_assert_620();
    fn __compiletime_assert_621();
    fn __compiletime_assert_622();
    fn __compiletime_assert_623();
    fn __compiletime_assert_625();
    fn __compiletime_assert_626();
    fn __compiletime_assert_627();
    fn __compiletime_assert_628();
    fn __compiletime_assert_629();
    fn __compiletime_assert_630();
    fn __compiletime_assert_631();
    fn __compiletime_assert_632();
    fn __compiletime_assert_633();
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
    fn __compiletime_assert_651();
    fn __compiletime_assert_652();
    fn __compiletime_assert_653();
    fn __compiletime_assert_654();
    fn __compiletime_assert_655();
    fn __compiletime_assert_666();
    fn __compiletime_assert_670();
    fn __compiletime_assert_673();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
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
unlikely__2!(overflow) != 0
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

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
if unlikely!(__ret_warn_on) != 0 {
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
struct static_key_mod {
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
    __slate_bits_0: __slate_bits::__SlateBits118U0,
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_pcie_link_event, const 0 as u8, label {
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
struct perf_event {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits125U0,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
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
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
}

fn list_is_last(list: *const list_head, head: &list_head) -> i32 {
((unsafe { (*list).next }) == ((head as *const list_head) as *mut list_head)) as i32
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
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


fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
    __slate_bits_0: __slate_bits::__SlateBits126U0,
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct workqueue_struct {
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
struct rseq {
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
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
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
    __slate_bits_0: __slate_bits::__SlateBits220U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
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
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
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
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
    __slate_bits_0: __slate_bits::__SlateBits283U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


// Resource type
// PCI/ISA I/O ports
// No side effects
// size indicates alignment
// Resource extended types
// Userland may not map this resource
// No address assigned yet
// PCI ROM control bits (IORESOURCE_BITS)
// ROM is enabled, same as PCI_ROM_ADDRESS_ENABLE
// PCI control bits.  Shares IORESOURCE_BITS with above PCI ROM.
// Do not move resource
fn resource_set_size(res: *mut resource, size: u64) {
unsafe {
(*res).end = unsafe { (*res).start }.wrapping_add(size).wrapping_sub(1);
}
}

fn resource_set_range(res: *mut resource, start: u64, size: u64) {
unsafe {
(*res).start = start;
}
resource_set_size(res, size);
}

fn resource_type(res: *const resource) -> u64 {
(unsafe { (*res).flags }) & 7936
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
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
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
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
if !(__native_word__10!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_179() };
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
struct user_namespace {
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
struct ns_common {
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


#[inline(always)]
fn __class_srcu_fast_notrace_cleanup_ctx(_T: *mut *mut __SlateRecord_srcu_h_645) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct return_consumer {
    cookie: u64,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits410U0,
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
struct wake_irq {
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
struct dev_pm_qos {
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
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
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
    __slate_bits_0: __slate_bits::__SlateBits458U0,
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
struct folio_queue {
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
    __slate_bits_0: __slate_bits::__SlateBits460U0,
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
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits485U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits485U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits485U2,
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
    __slate_bits_3: __slate_bits::__SlateBits485U3,
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


fn __devm_add_action_or_reset(dev: *mut device, action: Option<unsafe extern "C" fn(*mut ())>, data: *mut (), name: *const u8) -> i32 {
let ret: i32 = unsafe { __devm_add_action(dev, action, data, name) };
if ret != 0 {
unsafe { action.unwrap()(data) };
}
ret
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
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
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
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
    __slate_bits_0: __slate_bits::__SlateBits286U0,
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits613U0,
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
struct pcie_bwctrl_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_link_state {
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
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_desc {
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
union __SlateRecord_hw_irq_h_82 {
    ioapic: ioapic_alloc_info,
    uv: uv_alloc_info,
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
union __SlateRecord_msi_h_26 {
    __slate_anon_0: __SlateRecord_msi_h_26_2,
    dmar_subhandle: u32,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_26_2 {
    __slate_bits_0: __slate_bits::__SlateBits802U0,
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
union __SlateRecord_msi_h_49 {
    __slate_anon_0: __SlateRecord_msi_h_49_2,
    __slate_anon_1: __SlateRecord_msi_h_49_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_2 {
    __slate_bits_0: __slate_bits::__SlateBits806U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_3 {
    __slate_bits_0: __slate_bits::__SlateBits807U0,
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
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits816U0,
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
union __SlateRecord_msi_h_202 {
    pci: pci_msi_desc,
    data: msi_desc_data,
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
    __slate_bits_0: __slate_bits::__SlateBits482U0,
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


fn set_dev_node(dev: *mut device, node: i32) {
unsafe {
(*dev).numa_node = node;
}
}

fn dev_get_msi_domain(dev: *const device) -> *mut irq_domain {
unsafe { (*dev).msi.domain }
}

fn dev_set_msi_domain(dev: *mut device, d: *mut irq_domain) {
unsafe {
(*dev).msi.domain = d;
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

fn dev_set_removable(dev: *mut device, removable: u32) {
unsafe {
(*dev).removable = removable;
}
}

fn dev_is_removable(dev: *mut device) -> bool {
((unsafe { (*dev).removable }) as u32) == 3
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
    __slate_bits_0: __slate_bits::__SlateBits722U0,
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
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits852U0,
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
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits856U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct resource_entry {
    node: list_head,
    res: *mut resource,
    offset: u64,
    __res: resource,
}


fn resource_list_del(entry: *mut resource_entry) {
list_del(unsafe { core::ptr::addr_of_mut!((*entry).node) });
}

fn resource_list_free_entry(entry: *mut resource_entry) {
unsafe { kfree(entry as *const ()) };
}

fn resource_list_destroy_entry(entry: *mut resource_entry) {
resource_list_del(entry);
resource_list_free_entry(entry);
}

#[repr(C)]
#[derive(Clone, Copy)]
union msi_instance_cookie {
    value: u64,
    ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpio_mapping {
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
    __slate_bits_0: __slate_bits::__SlateBits731U0,
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
    __slate_bits_0: __slate_bits::__SlateBits724U0,
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
    __slate_bits_1: __slate_bits::__SlateBits724U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits724U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits724U3,
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
    __slate_bits_4: __slate_bits::__SlateBits724U4,
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
struct __SlateRecord_probe_c_785 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_probe_c_824 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_probe_c_1962 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_probe_c_1966 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_probe_c_1988 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_probe_c_2053 {
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
    pub struct __SlateBits283U0 {
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
    pub struct __SlateBits286U0 {
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
    pub struct __SlateBits118U0 {
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
    pub struct __SlateBits126U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits410U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits613U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits458U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits460U0 {
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
    pub struct __SlateBits125U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits485U0 {
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
    pub struct __SlateBits482U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits485U1 {
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
    pub struct __SlateBits485U2 {
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
    pub struct __SlateBits485U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits806U0 {
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
    pub struct __SlateBits807U0 {
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
    pub struct __SlateBits809U0 {
        #[bits(8)]
        pub reserved: u32,
        #[bits(24)]
        pub destid_8_31: u32,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits802U0 {
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
    pub struct __SlateBits722U0 {
        #[bits(1)]
        pub is_managed: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits816U0 {
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
    pub struct __SlateBits778U0 {
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
    pub struct __SlateBits220U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits731U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits724U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits724U1 {
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
    pub struct __SlateBits724U2 {
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
    pub struct __SlateBits724U3 {
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
    pub struct __SlateBits724U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits439U0 {
        #[bits(1)]
        pub is_added: u32,
        #[bits(1)]
        pub unsafe_warn: u32,
        #[bits(1)]
        pub flit_mode: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits750U0 {
        #[bits(1)]
        pub ignore_reset_delay: u32,
        #[bits(1)]
        pub no_ext_tags: u32,
        #[bits(1)]
        pub no_inc_mrrs: u32,
        #[bits(1)]
        pub native_aer: u32,
        #[bits(1)]
        pub native_pcie_hotplug: u32,
        #[bits(1)]
        pub native_shpc_hotplug: u32,
        #[bits(1)]
        pub native_pme: u32,
        #[bits(1)]
        pub native_ltr: u32,
        #[bits(1)]
        pub native_dpc: u32,
        #[bits(1)]
        pub native_cxl_error: u32,
        #[bits(1)]
        pub preserve_config: u32,
        #[bits(1)]
        pub size_windows: u32,
        #[bits(1)]
        pub msi_domain: u32,
        #[bits(1)]
        pub broken_l1ss_resume: u32,
        #[bits(2, access = na)]
        pub __slate_pad_14: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits843U0 {
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
    pub struct __SlateBits844U0 {
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
    pub struct __SlateBits848U0 {
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
    pub struct __SlateBits850U0 {
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
    pub struct __SlateBits852U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub explicit_set: u8,
        #[bits(6)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits858U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub notifier_present: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits854U0 {
        #[bits(8)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits856U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(7)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits837U0 {
        #[bits(1)]
        pub enabled: bool,
        #[bits(1)]
        pub demand_offline: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
}

fn pci_physfn(dev: *mut pci_dev) -> *mut pci_dev {
dev
}

// Currently in ACPI spec, for each PCI host bridge, PCI Segment
// Group number is limited to a 16-bit value, therefore (int)-1 is
// not a valid PCI domain number, and can be used as a sentinel
// value indicating ->domain_nr is not set by the driver (and
// CONFIG_PCI_DOMAINS_GENERIC=y archs will set it with
// pci_bus_find_domain_nr()).
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct pci_host_bridge {
    dev: device,
    bus: *mut pci_bus,
    ops: *mut pci_ops,
    child_ops: *mut pci_ops,
    sysdata: *mut (),
    busnr: i32,
    domain_nr: i32,
    windows: list_head,
    dma_ranges: list_head,
    ports: list_head,
    swizzle_irq: Option<unsafe extern "C" fn(*mut pci_dev, *mut u8) -> u8>,
    map_irq: Option<unsafe extern "C" fn(*const pci_dev, u8, u8) -> i32>,
    release_fn: Option<unsafe extern "C" fn(*mut pci_host_bridge)>,
    enable_device: Option<unsafe extern "C" fn(*mut pci_host_bridge, *mut pci_dev) -> i32>,
    disable_device: Option<unsafe extern "C" fn(*mut pci_host_bridge, *mut pci_dev)>,
    reset_root_port: Option<unsafe extern "C" fn(*mut pci_host_bridge, *mut pci_dev) -> i32>,
    release_data: *mut (),
    __slate_bits_0: __slate_bits::__SlateBits750U0,
    align_resource: Option<unsafe extern "C" fn(*mut pci_dev, *const resource, u64, u64, u64) -> u64>,
    __slate_pad_19: [u8; 40],
    private: [u64; 0],
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
    __slate_bits_0: __slate_bits::__SlateBits439U0,
}


fn pci_is_root_bus(pbus: *mut pci_bus) -> bool {
unsafe { (*pbus).parent }.is_null()
}

fn pci_is_bridge(dev: *mut pci_dev) -> bool {
((unsafe { (*dev).hdr_type }) as i32) == 1 || ((unsafe { (*dev).hdr_type }) as i32) == 2
}

fn pci_upstream_bridge(mut dev: *mut pci_dev) -> *mut pci_dev {
dev = pci_physfn(dev);
if pci_is_root_bus(unsafe { (*dev).bus }) {
return core::ptr::null_mut::<pci_dev>();
}
unsafe { (*(*dev).bus).self_ }
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
struct pci_bus_region {
    start: u64,
    end: u64,
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
fn pci_has_flag(flag: i32) -> i32 {
((unsafe { pci_flags }) & (flag as u32)) as i32
}

fn pcie_capability_clear_and_set_word(dev: *mut pci_dev, pos: i32, clear: u16, set: u16) -> i32 {
match pos {
16 | 28 | 48 => {
return unsafe { pcie_capability_clear_and_set_word_locked(dev, pos, clear, set) };
}
_ => {
return unsafe { pcie_capability_clear_and_set_word_unlocked(dev, pos, clear, set) };
}
}
unsafe { core::mem::zeroed() }
}

fn pcie_capability_set_word(dev: *mut pci_dev, pos: i32, set: u16) -> i32 {
pcie_capability_clear_and_set_word(dev, pos, 0u16, set)
}

fn pcie_capability_clear_word(dev: *mut pci_dev, pos: i32, clear: u16) -> i32 {
pcie_capability_clear_and_set_word(dev, pos, clear, 0u16)
}

#[unsafe(no_mangle)]
extern "C" fn set_pcie_port_type(pdev: *mut pci_dev) {
let pos: i32 = (unsafe { pci_find_capability(pdev, 16i32) }) as i32;
let mut reg16: u16 = 0;
let mut reg32: u32 = 0;
let mut r#type: i32 = 0;
if pos == 0 {
return;
}
unsafe {
(*pdev).pcie_cap = pos as u8;
}
unsafe { pci_read_config_word(pdev as *const pci_dev, pos.wrapping_add(2), core::ptr::addr_of_mut!(reg16)) };
unsafe {
(*pdev).pcie_flags_reg = reg16;
}
r#type = pci_pcie_type(pdev as *const pci_dev);
if r#type == 4 {
pci_enable_rrs_sv(pdev);
}
unsafe { pci_read_config_dword(pdev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!((*pdev).devcap)) };
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(7i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_625() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_626() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_627() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(7i32 >> (7u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_628() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 7u64.wrapping_add(1u64 << (7u64.trailing_zeros() as i32)) & 7u64.wrapping_add(1u64 << (7u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_629() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (((unsafe { (*pdev).devcap }) & 7) >> (7u64.trailing_zeros() as i32)) as i32;
__v0 = __v2;
unsafe {
(*pdev).__slate_bits_0.__set_pcie_mpss(__v0 as u8);
}
unsafe { pcie_capability_read_dword(pdev, 12i32, core::ptr::addr_of_mut!(reg32)) };
if reg32 & 1048576 != 0 {
unsafe {
(*pdev).__slate_bits_3.__set_link_active_reporting(1u32);
}
}
if reg32 & 1024 != 0 {
unsafe {
(*pdev).__slate_bits_2.__set_aspm_l0s_support(1u32);
}
}
if reg32 & 2048 != 0 {
unsafe {
(*pdev).__slate_bits_2.__set_aspm_l1_support(1u32);
}
}
let parent: *mut pci_dev = pci_upstream_bridge(pdev);
if parent.is_null() {
return;
}
// Some systems do not identify their upstream/downstream ports
// correctly so detect impossible configurations here and correct
// the port type accordingly.
if r#type == 6 {
// If pdev claims to be downstream port but the parent
// device is also downstream port assume pdev is actually
// upstream port.
if pcie_downstream_port(parent as *const pci_dev) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"claims to be downstream port but is acting as upstream port, correcting type\n".as_ptr() as *const u8) };
let __v10: u16 = (((unsafe { (*pdev).pcie_flags_reg }) as i32) & !240) as u16;
unsafe {
(*pdev).pcie_flags_reg = __v10;
}
unsafe {
(*pdev).pcie_flags_reg = (((*pdev).pcie_flags_reg as i32) | 5) as u16;
}
}
} else {
if r#type == 5 {
// If pdev claims to be upstream port but the parent
// device is also upstream port assume pdev is actually
// downstream port.
if pci_pcie_type(parent as *const pci_dev) == 5 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"claims to be upstream port but is acting as downstream port, correcting type\n".as_ptr() as *const u8) };
let __v17: u16 = (((unsafe { (*pdev).pcie_flags_reg }) as i32) & !240) as u16;
unsafe {
(*pdev).pcie_flags_reg = __v17;
}
unsafe {
(*pdev).pcie_flags_reg = (((*pdev).pcie_flags_reg as i32) | 6) as u16;
}
}
}
}
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
struct pci_sysdata {
    domain: i32,
    node: i32,
    companion: *mut acpi_device,
    iommu: *mut (),
    fwnode: *mut (),
}


fn to_pci_sysdata(bus: *const pci_bus) -> *mut pci_sysdata {
(unsafe { (*bus).sysdata }) as *mut pci_sysdata
}

fn pci_domain_nr(bus: *mut pci_bus) -> i32 {
unsafe { (*to_pci_sysdata(bus as *const pci_bus)).domain }
}

fn _pci_root_bus_fwnode(bus: *mut pci_bus) -> *mut () {
unsafe { (*to_pci_sysdata(bus as *const pci_bus)).fwnode }
}

fn __pcibus_to_node(bus: *const pci_bus) -> i32 {
unsafe { (*to_pci_sysdata(bus)).node }
}

// pci_bus_for_each_resource - iterate over PCI bus resources
// @bus: the PCI bus
// @res: pointer to the current resource
// @...: optional index of the current resource
//
// Iterate over PCI bus resources. The first part is to go over PCI bus
// resource array, which has at most the %PCI_BRIDGE_RESOURCE_NUM entries.
// After that continue with the separate list of the additional resources,
// if not empty. That's why the Logical OR is being used.
//
// Possible usage:
//
// 	struct pci_bus *bus = ...;
// 	struct resource *res;
// 	unsigned int i;
//
// 	// With optional index
// 	pci_bus_for_each_resource(bus, res, i)
// 		pr_info("PCI bus resource[%u]: %pR\n", i, res);
//
// 	// Without index
// 	pci_bus_for_each_resource(bus, res)
// 		_do_something_(res);
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
fn pci_name(pdev: *const pci_dev) -> *const u8 {
dev_name(unsafe { core::ptr::addr_of!((*pdev).dev) })
}

fn pci_pcie_cap(dev: *mut pci_dev) -> i32 {
(unsafe { (*dev).pcie_cap }) as i32
}

fn pci_is_pcie(dev: *mut pci_dev) -> bool {
pci_pcie_cap(dev) != 0
}

fn pcie_caps_reg(dev: *const pci_dev) -> u16 {
unsafe { (*dev).pcie_flags_reg }
}

fn pci_pcie_type(dev: *const pci_dev) -> i32 {
((pcie_caps_reg(dev) as i32) & 240) >> 4i32
}

fn pcie_find_root_port(mut dev: *mut pci_dev) -> *mut pci_dev {
while !dev.is_null() {
let __v0: bool = pci_is_pcie(dev) && pci_pcie_type(dev as *const pci_dev) == 4;
if __v0 {
return dev;
}
dev = pci_upstream_bridge(dev);
}
core::ptr::null_mut::<pci_dev>()
}

fn pci_host_bridge_of_msi_domain(bus: *mut pci_bus) -> *mut irq_domain {
core::ptr::null_mut::<irq_domain>()
}

fn pci_host_of_has_msi_map(dev: *mut device) -> bool {
false
}

fn pci_ari_enabled(bus: *mut pci_bus) -> bool {
!unsafe { (*bus).self_ }.is_null() && ((unsafe { (*(*bus).self_).__slate_bits_3.__get_ari_enabled() }) as i32) != 0
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
fn dma_set_max_seg_size(dev: *mut device, size: u32) {
let mut __v0: i64;
let __ret_warn_on: i32 = unsafe { (*dev).dma_parms }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("506: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 506b - ., 3; .popsection\n# {0}", const 506 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dma_set_max_seg_size_1, sym __slate_asm_string_dma_set_max_seg_size_2, const 676 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("507: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 507b - ., 4; .popsection\n# {0}", const 507 as i32, options(nostack, att_syntax));
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
(*(*dev).dma_parms).max_segment_size = size;
}
}

fn dma_set_seg_boundary(dev: *mut device, mask: u64) {
let mut __v0: i64;
let __ret_warn_on: i32 = unsafe { (*dev).dma_parms }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("508: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 508b - ., 3; .popsection\n# {0}", const 508 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dma_set_seg_boundary_1, sym __slate_asm_string_dma_set_seg_boundary_2, const 709 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("509: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 509b - ., 4; .popsection\n# {0}", const 509 as i32, options(nostack, att_syntax));
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
(*(*dev).dma_parms).segment_boundary_mask = mask;
}
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


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_pci_root_buses: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section\t\".initcall2.init\", \"a\"\t\t\n__initcall__kmod_probe__552_93_pcibus_class_init2:\t\t\t\n.long\t{1} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_alloc_host_bridge: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_devm_pci_alloc_host_bridge: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_free_host_bridge: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_link_speed: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_get_link_speed: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_speed_string: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_update_link_speed: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_add_new_bus: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_scan_bridge: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_relaxed_ordering_enabled: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_alloc_dev: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_bus_read_dev_vendor_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_scan_single_device: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_scan_slot: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_bus_configure_settings: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_scan_child_bus: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_create_root_bus: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_host_probe: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_scan_root_bus_bridge: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_scan_root_bus: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_scan_bus: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_rescan_bus: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_lock_rescan_remove: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_unlock_rescan_remove: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_hp_add_bridge: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous", sym pci_root_buses, sym pcibus_class_init, sym pci_alloc_host_bridge, sym devm_pci_alloc_host_bridge, sym pci_free_host_bridge, sym pcie_link_speed, sym pcie_get_link_speed, sym pci_speed_string, sym pcie_update_link_speed, sym pci_add_new_bus, sym pci_scan_bridge, sym pcie_relaxed_ordering_enabled, sym pci_alloc_dev, sym pci_bus_read_dev_vendor_id, sym pci_scan_single_device, sym pci_scan_slot, sym pcie_bus_configure_settings, sym pci_scan_child_bus, sym pci_create_root_bus, sym pci_host_probe, sym pci_scan_root_bus_bridge, sym pci_scan_root_bus, sym pci_scan_bus, sym pci_rescan_bus, sym pci_lock_rescan_remove, sym pci_unlock_rescan_remove, sym pci_hp_add_bridge, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct ioapic_alloc_info {
    pin: i32,
    node: i32,
    __slate_bits_0: __slate_bits::__SlateBits778U0,
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


fn irq_find_matching_fwnode(fwnode: *mut fwnode_handle, bus_token: u32) -> *mut irq_domain {
let mut fwspec: irq_fwspec = {
let mut __t0: irq_fwspec = unsafe { core::mem::zeroed() };
__t0.fwnode = fwnode;
__t0
};
unsafe { irq_find_matching_fwspec(core::ptr::addr_of_mut!(fwspec), bus_token) }
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
    __slate_bits_0: __slate_bits::__SlateBits809U0,
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
struct hotplug_slot_ops {
    enable_slot: Option<unsafe extern "C" fn(*mut hotplug_slot) -> i32>,
    disable_slot: Option<unsafe extern "C" fn(*mut hotplug_slot) -> i32>,
    set_attention_status: Option<unsafe extern "C" fn(*mut hotplug_slot, u8) -> i32>,
    hardware_test: Option<unsafe extern "C" fn(*mut hotplug_slot, u32) -> i32>,
    get_power_status: Option<unsafe extern "C" fn(*mut hotplug_slot, *mut u8) -> i32>,
    get_attention_status: Option<unsafe extern "C" fn(*mut hotplug_slot, *mut u8) -> i32>,
    get_latch_status: Option<unsafe extern "C" fn(*mut hotplug_slot, *mut u8) -> i32>,
    get_adapter_status: Option<unsafe extern "C" fn(*mut hotplug_slot, *mut u8) -> i32>,
    reset_slot: Option<unsafe extern "C" fn(*mut hotplug_slot, bool) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hotplug_slot {
    ops: *const hotplug_slot_ops,
    pci_slot: *mut pci_slot,
    owner: *mut module,
    mod_name: *const u8,
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
    __slate_bits_0: __slate_bits::__SlateBits837U0,
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
    __slate_bits_0: __slate_bits::__SlateBits843U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_flags {
    __slate_bits_0: __slate_bits::__SlateBits844U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_dir {
    entry: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_type {
    __slate_bits_0: __slate_bits::__SlateBits848U0,
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
    __slate_bits_0: __slate_bits::__SlateBits850U0,
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
    __slate_bits_0: __slate_bits::__SlateBits854U0,
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
    __slate_bits_0: __slate_bits::__SlateBits858U0,
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


fn jailhouse_paravirt() -> bool {
false
}

fn hypervisor_isolated_pci_functions() -> bool {
if IS_ENABLED!() != 0 {
return true;
}
if IS_ENABLED!() != 0 {
return true;
}
jailhouse_paravirt()
}

// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
fn pm_runtime_get_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 4i32) }
}

fn pm_runtime_put(dev: *mut device) {
unsafe { __pm_runtime_idle(dev, 5i32) };
}

fn pm_runtime_set_active(dev: *mut device) -> i32 {
unsafe { __pm_runtime_set_status(dev, 0u32) }
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
fn __do_trace_pcie_link_event(bus: *mut pci_bus, reason: u32, width: u32, status: u32) {
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
let mut __UNIQUE_ID_guard_532: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_532) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_533: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_533) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_535() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_pcie_link_event)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_534: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_534;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut pci_bus, u32, u32, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut pci_bus, u32, u32, u32) -> i32>>(__SCT__tp_func_pcie_link_event as *const ()) };
unsafe { __v12.unwrap()(__data, bus, reason, width, status) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_pcie_link_event: __SlateAlign16<[u8; 16]> = __SlateAlign16([112u8, 99u8, 105u8, 101u8, 95u8, 108u8, 105u8, 110u8, 107u8, 95u8, 101u8, 118u8, 101u8, 110u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_pcie_link_event_536: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_pcie_link_event) }) as *mut ();

fn trace_pcie_link_event(bus: *mut pci_bus, reason: u32, width: u32, status: u32) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_pcie_link_event.key) }) != 0 {
__do_trace_pcie_link_event(bus, reason, width, status);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_537: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn pci_ea_fixed_busnrs(dev: *mut pci_dev, sec: *mut u8, sub: *mut u8) -> bool {
let mut ea: i32 = 0;
let mut offset: i32 = 0;
let mut dw: u32 = 0;
let mut ea_sec: u8 = 0;
let mut ea_sub: u8 = 0;
if ((unsafe { (*dev).hdr_type }) as i32) != 1 {
return false;
}
// find PCI EA capability in list
ea = (unsafe { pci_find_capability(dev, 20i32) }) as i32;
if ea == 0 {
return false;
}
offset = ea.wrapping_add(4);
unsafe { pci_read_config_dword(dev as *const pci_dev, offset, core::ptr::addr_of_mut!(dw)) };
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(255i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_584() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_585() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_586() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(255i32 >> (255u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_587() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)) & 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_588() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((dw & 255) >> (255u64.trailing_zeros() as i32)) as i32;
__v0 = __v2;
ea_sec = __v0 as u8;
let mut __v7: i32;
loop {
if __bf_cast_unsigned!(65280i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_589() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v9: i32;
loop {
if false {
unsafe { __compiletime_assert_590() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_591() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(65280i32 >> (65280u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_592() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)) & 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_593() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = ((dw & 65280) >> (65280u64.trailing_zeros() as i32)) as i32;
__v7 = __v9;
ea_sub = __v7 as u8;
if (ea_sec as i32) == 0 || (ea_sub as i32) < (ea_sec as i32) {
return false;
}
unsafe {
*sec = ea_sec;
}
unsafe {
*sub = ea_sub;
}
true
}

fn pci_bus_rrs_vendor_id(l: u32) -> bool {
l & 65535 == 1
}

fn pcie_downstream_port(dev: *const pci_dev) -> bool {
let r#type: i32 = pci_pcie_type(dev);
r#type == 4 || r#type == 6 || r#type == 8
}

fn pci_is_cardbus_bridge(dev: *mut pci_dev) -> bool {
((unsafe { (*dev).hdr_type }) as i32) == 2
}

#[unsafe(no_mangle)]
extern "C" fn pci_setup_device(dev: *mut pci_dev) -> i32 {
let mut progif: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u8 = core::ptr::addr_of_mut!(progif) as *mut u8;
let mut res: core::mem::MaybeUninit<*mut resource> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut resource = core::ptr::addr_of_mut!(res) as *mut *mut resource;
let mut region: core::mem::MaybeUninit<pci_bus_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut pci_bus_region = core::ptr::addr_of_mut!(region) as *mut pci_bus_region;
let mut pos: i32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut hdr_type: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u8 = core::ptr::addr_of_mut!(hdr_type) as *mut u8;
let mut cmd: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u16 = core::ptr::addr_of_mut!(cmd) as *mut u16;
let mut class: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(class) as *mut u32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_3 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_4 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
pos = 0;
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_6 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
*__slate_slot_13 = pci_hdr_type(dev);
(*dev).sysdata = (*(*dev).bus).sysdata;
(*dev).dev.parent = (*(*dev).bus).bridge;
(*dev).dev.bus = core::ptr::addr_of!(pci_bus_type);
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(127i32 >> (127u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_643();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 127u64.wrapping_add(1u64 << (127u64.trailing_zeros() as i32)) & 127u64.wrapping_add(1u64 << (127u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_644();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
(*dev).hdr_type = (((*__slate_slot_13 as i32) & 127) >> (127u64.trailing_zeros() as i32)) as u8;
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(128i32 >> (128u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_648();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 128u64.wrapping_add(1u64 << (128u64.trailing_zeros() as i32)) & 128u64.wrapping_add(1u64 << (128u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_649();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
(*dev).__slate_bits_3.__set_multifunction((((*__slate_slot_13 as i32) & 128) >> (128u64.trailing_zeros() as i32)) as u32);
(*dev).error_state = 1;
set_pcie_port_type(dev);
*__slate_slot_12 = pci_set_of_node(dev);
if *__slate_slot_12 != 0 {
return *__slate_slot_12;
} else {
pci_set_acpi_fwnode(dev);
pci_dev_assign_slot(dev);
// Assume 32-bit PCI; let 64-bit PCI cards (which are far rarer)
// set this higher, assuming the system even supports it.
(*dev).dma_mask = 4294967295;
// Assume 64-bit addresses for MSI initially. Will be changed to 32-bit
// if MSI (rather than MSI-X) capability does not have
// PCI_MSI_FLAGS_64BIT. Can also be overridden by driver.
(*dev).msi_addr_mask = DMA_BIT_MASK__2!(64i32);
dev_set_name(core::ptr::addr_of_mut!((*dev).dev), c"%04x:%02x:%02x.%d".as_ptr() as *const u8, pci_domain_nr((*dev).bus), (*(*dev).bus).number as i32, PCI_SLOT!(unsafe { (*dev).devfn }), PCI_FUNC!(unsafe { (*dev).devfn }));
*__slate_slot_15 = pci_class(dev);
(*dev).revision = (*__slate_slot_15 & 255) as u8;
(*dev).class = *__slate_slot_15 >> 8i32;
// upper 3 bytes
if pci_early_dump {
early_dump_pci_device(dev);
}
// Need to have dev->class ready
(*dev).cfg_size = pci_cfg_space_size(dev);
// Need to have dev->cfg_size ready
set_pcie_thunderbolt(dev);
set_pcie_cxl(dev);
set_pcie_untrusted(dev);
if pci_is_pcie(dev) {
(*dev).supported_speeds = pcie_get_supported_speeds(dev);
}
// "Unknown power state"
(*dev).current_state = 5;
// Early fixups, before probing the BARs
pci_fixup_device(0u32, dev);
pci_set_removable(dev);
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"[%04x:%04x] type %02x class %#08x %s\n".as_ptr() as *const u8, (*dev).vendor as i32, (*dev).device as i32, (*dev).hdr_type as i32, (*dev).class, pci_type_str(dev));
// Device class may be changed after fixup
*__slate_slot_15 = (*dev).class >> 8i32;
if ((*dev).__slate_bits_3.__get_non_compliant_bars() as i32) != 0 && ((*dev).__slate_bits_1.__get_mmio_always_on() as i32) == 0 {
pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(*__slate_slot_14));
if (*__slate_slot_14 as i32) & (1 | 2) != 0 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"device has non-compliant BARs; disabling IO/MEM decoding\n".as_ptr() as *const u8);
let __v3: u16 = ((*__slate_slot_14 as i32) & !1) as u16;
*__slate_slot_14 = __v3;
let __v1: u16 = ((*__slate_slot_14 as i32) & !2) as u16;
*__slate_slot_14 = __v1;
pci_write_config_word(dev as *const pci_dev, 4i32, *__slate_slot_14);
}
}
'__join_0: {
let __t1 = pci_intx_mask_broken(dev) as u32;
(*dev).__slate_bits_3.__set_broken_intx_masking(__t1);
let __t2: i32 = (*dev).hdr_type as i32;
if __t2 == 0 {
if *__slate_slot_15 != 1540 {
// standard header
pci_read_irq(dev);
pci_read_bases(dev, 6u32, 48i32);
pci_subsystem_ids(dev, core::ptr::addr_of_mut!((*dev).subsystem_vendor), core::ptr::addr_of_mut!((*dev).subsystem_device));
// Do the ugly legacy mode stuff here rather than broken chip
// quirk code. Legacy mode ATA controllers have fixed
// addresses. These are not always echoed in BAR0-3, and
// BAR0-3 in a few cases contain junk!
if *__slate_slot_15 == 257 {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
pci_read_config_byte(dev as *const pci_dev, 9i32, core::ptr::addr_of_mut!(*__slate_slot_0));
if (*__slate_slot_0 as i32) & 1 == 0 {
(*__slate_slot_10).start = 496;
(*__slate_slot_10).end = 503;
*__slate_slot_9 = ((*dev).resource.as_mut_ptr() as *mut resource).wrapping_offset(0 as isize);
(*(*__slate_slot_9)).flags = 272u64;
pcibios_bus_to_resource((*dev).bus, *__slate_slot_9, core::ptr::addr_of_mut!(*__slate_slot_10));
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"BAR 0 %pR: legacy IDE quirk\n".as_ptr() as *const u8, *__slate_slot_9);
(*__slate_slot_10).start = 1014;
(*__slate_slot_10).end = 1014;
*__slate_slot_9 = ((*dev).resource.as_mut_ptr() as *mut resource).wrapping_offset(1 as isize);
(*(*__slate_slot_9)).flags = 272u64;
pcibios_bus_to_resource((*dev).bus, *__slate_slot_9, core::ptr::addr_of_mut!(*__slate_slot_10));
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"BAR 1 %pR: legacy IDE quirk\n".as_ptr() as *const u8, *__slate_slot_9);
}
if (*__slate_slot_0 as i32) & 4 == 0 {
(*__slate_slot_10).start = 368;
(*__slate_slot_10).end = 375;
*__slate_slot_9 = ((*dev).resource.as_mut_ptr() as *mut resource).wrapping_offset(2 as isize);
(*(*__slate_slot_9)).flags = 272u64;
pcibios_bus_to_resource((*dev).bus, *__slate_slot_9, core::ptr::addr_of_mut!(*__slate_slot_10));
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"BAR 2 %pR: legacy IDE quirk\n".as_ptr() as *const u8, *__slate_slot_9);
(*__slate_slot_10).start = 886;
(*__slate_slot_10).end = 886;
*__slate_slot_9 = ((*dev).resource.as_mut_ptr() as *mut resource).wrapping_offset(3 as isize);
(*(*__slate_slot_9)).flags = 272u64;
pcibios_bus_to_resource((*dev).bus, *__slate_slot_9, core::ptr::addr_of_mut!(*__slate_slot_10));
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"BAR 3 %pR: legacy IDE quirk\n".as_ptr() as *const u8, *__slate_slot_9);
break '__join_0;
} else {
break '__join_0;
}
} else {
break '__join_0;
}
}
} else {
if __t2 == 1 {
pci_read_irq(dev); // bridge header
// The PCI-to-PCI bridge spec requires that subtractive
// decoding (i.e. transparent) bridge must have programming
// interface code of 0x01.
let __t0 = ((*dev).class & 255 == 1) as u32;
(*dev).__slate_bits_3.__set_transparent(__t0);
pci_read_bases(dev, 2u32, 56i32);
pci_read_bridge_windows(dev);
set_pcie_hotplug_bridge(dev);
pos = pci_find_capability(dev, 13i32) as i32;
if pos != 0 {
pci_read_config_word(dev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!((*dev).subsystem_vendor));
pci_read_config_word(dev as *const pci_dev, pos.wrapping_add(6), core::ptr::addr_of_mut!((*dev).subsystem_device));
break '__join_0;
} else {
break '__join_0;
}
} else {
if __t2 == 2 {
if *__slate_slot_15 == 1543 {
// CardBus bridge header
pci_read_irq(dev);
pci_read_bases(dev, 1u32, 0i32);
pci_read_config_word(dev as *const pci_dev, 64i32, core::ptr::addr_of_mut!((*dev).subsystem_vendor));
pci_read_config_word(dev as *const pci_dev, 66i32, core::ptr::addr_of_mut!((*dev).subsystem_device));
break '__join_0;
}
} else {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"unknown header type %02x, ignoring device\n".as_ptr() as *const u8, (*dev).hdr_type as i32); // unknown header
pci_release_of_node(dev);
return 5i32.wrapping_neg();
}
}
}
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ignoring class %#08x (doesn't match header type %02x)\n".as_ptr() as *const u8, (*dev).class, (*dev).hdr_type as i32);
(*dev).class = 0u32;
}
// We found a fine healthy device, go go go...
return 0;
}
}
// header type
}

fn __pcie_update_link_speed(bus: *mut pci_bus, reason: u32, linksta: u16, linksta2: u16) {
unsafe {
(*bus).cur_bus_speed = *(core::ptr::addr_of!(pcie_link_speed.0) as *const u8).wrapping_offset(((linksta as i32) & 15) as isize);
}
unsafe {
let __t0 = (if (linksta2 as i32) & 1024 != 0 { 1i32 } else { 0i32 }) as u32;
(*bus).__slate_bits_0.__set_flit_mode(__t0);
}
let mut __v0: i32;
while false {
}
let mut __v2: i32;
while false {
}
while false {
}
loop {
if (!(1008i32 >> (1008u64.trailing_zeros() as i32)) as u32) & 0u32 != 0 {
unsafe { __compiletime_assert_545() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 1008u64.wrapping_add(1u64 << (1008u64.trailing_zeros() as i32)) & 1008u64.wrapping_add(1u64 << (1008u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_546() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((linksta as i32) & 1008) >> (1008u64.trailing_zeros() as i32);
__v0 = __v2;
trace_pcie_link_event(bus, reason as u32, __v0 as u32, ((linksta as i32) & (16384 | 32768 | 2048 | 8192)) as u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sriov {
    pos: i32,
    nres: i32,
    cap: u32,
    ctrl: u16,
    total_VFs: u16,
    initial_VFs: u16,
    num_VFs: u16,
    offset: u16,
    stride: u16,
    vf_device: u16,
    pgsz: u32,
    link: u8,
    max_VF_buses: u8,
    driver_max_VFs: u16,
    dev: *mut pci_dev,
    self_: *mut pci_dev,
    class: u32,
    hdr_type: u8,
    subsystem_vendor: u16,
    subsystem_device: u16,
    barsz: [u64; 6],
    vf_rebar_cap: u16,
    drivers_autoprobe: bool,
}


fn pci_doe_init(pdev: *mut pci_dev) {
}

fn pci_npem_create(dev: *mut pci_dev) {
}

fn pci_doe_sysfs_init(pdev: *mut pci_dev) {
}

fn pci_ide_init(dev: *mut pci_dev) {
}

fn pci_ide_init_host_bridge(hb: *mut pci_host_bridge) {
}

fn pci_tsm_init(pdev: *mut pci_dev) {
}

// pci_dev priv_flags
fn pci_dev_is_added(dev: *const pci_dev) -> bool {
test_bit__2!(0i32, unsafe { core::ptr::addr_of!((*dev).priv_flags) }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcec_ea {
    nextbusn: u8,
    lastbusn: u8,
    bitmap: u32,
}


fn pci_dpc_init(pdev: *mut pci_dev) {
}

fn pci_iov_init(dev: *mut pci_dev) -> i32 {
19i32.wrapping_neg()
}

fn pci_iov_release(dev: *mut pci_dev) {
}

fn pci_iov_bus_range(bus: *mut pci_bus) -> i32 {
0
}

fn pci_tph_init(dev: *mut pci_dev) {
}

fn pci_ptm_init(dev: *mut pci_dev) {
}

fn of_pci_preserve_config(node: *mut device_node) -> bool {
false
}

fn pci_set_of_node(dev: *mut pci_dev) -> i32 {
0
}

fn pci_release_of_node(dev: *mut pci_dev) {
}

fn pci_set_bus_of_node(bus: *mut pci_bus) {
}

fn pci_release_bus_of_node(bus: *mut pci_bus) {
}

fn devm_of_pci_bridge_init(dev: *mut device, bridge: *mut pci_host_bridge) -> i32 {
0
}

fn of_pci_make_host_bridge_node(bridge: *mut pci_host_bridge) {
}

fn pci_aer_init(d: *mut pci_dev) {
}

fn pci_aer_exit(d: *mut pci_dev) {
}

static mut busn_resource: resource = {
let mut __t0: resource = unsafe { core::mem::zeroed() };
__t0.start = 0u64;
__t0.end = 255u64;
__t0.name = (b"PCI busn\0".as_ptr() as *mut u8) as *const u8;
__t0.flags = 4096u64;
__t0
};

/// Ugh.  Need to stop exporting this to modules.
#[unsafe(no_mangle)]
static mut pci_root_buses: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(pci_root_buses) }, prev: unsafe { core::ptr::addr_of_mut!(pci_root_buses) } };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_root_buses_551: *mut () = (unsafe { core::ptr::addr_of_mut!(pci_root_buses) }) as *mut ();

static mut pci_domain_busn_res_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(pci_domain_busn_res_list) }, prev: unsafe { core::ptr::addr_of_mut!(pci_domain_busn_res_list) } };

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_domain_busn_res {
    list: list_head,
    res: resource,
    domain_nr: i32,
}


fn get_pci_domain_busn_res(domain_nr: i32) -> *mut resource {
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut r: *mut pci_domain_busn_res = unsafe { core::mem::zeroed() };
r = list_first_entry__11!(unsafe { core::ptr::addr_of_mut!(pci_domain_busn_res_list) });
while list_entry_is_head!(r, unsafe { core::ptr::addr_of_mut!(pci_domain_busn_res_list) }, list) == 0 {
if (unsafe { (*r).domain_nr }) == domain_nr {
return unsafe { core::ptr::addr_of_mut!((*r).res) };
}
r = list_next_entry__12!(r, list);
}
let mut __v3: *mut pci_domain_busn_res;
let __obj_size: u64 = size_mul(core::mem::size_of::<pci_domain_busn_res>() as u64, 1u64);
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
__v3 = __v4 as *mut pci_domain_busn_res;
r = __v3;
if r.is_null() {
return core::ptr::null_mut::<resource>();
}
unsafe {
(*r).domain_nr = domain_nr;
}
unsafe {
(*r).res.start = 0;
}
unsafe {
(*r).res.end = 255;
}
unsafe {
(*r).res.flags = 4112u64;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*r).list) }, unsafe { core::ptr::addr_of_mut!(pci_domain_busn_res_list) });
unsafe { core::ptr::addr_of_mut!((*r).res) }
}

/// PCI Bus Class
#[unsafe(link_section = ".text.slate_distinct.probe.release_pcibus_dev")]
extern "C" fn release_pcibus_dev(dev: *mut device) {
let pci_bus: *mut pci_bus = to_pci_bus!(dev);
unsafe { put_device((*pci_bus).bridge) };
unsafe { pci_bus_remove_resources(pci_bus) };
pci_release_bus_of_node(pci_bus);
unsafe { kfree(pci_bus as *const ()) };
}

static pcibus_class: __SlateConst<class> = __SlateConst({
let mut __t0: class = unsafe { core::mem::zeroed() };
__t0.name = (b"pci_bus\0".as_ptr() as *mut u8) as *const u8;
__t0.dev_groups = (unsafe { core::ptr::addr_of_mut!(pcibus_groups) as *mut *const attribute_group }) as *const *const attribute_group;
__t0.dev_release = Some(release_pcibus_dev);
__t0
});

#[unsafe(link_section = ".init.text")]
extern "C" fn pcibus_class_init() -> i32 {
unsafe { class_register(core::ptr::addr_of!(pcibus_class.0)) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcibus_class_init_553: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(pcibus_class_init)) };

fn pci_size(base: u64, maxbase: u64, mask: u64) -> u64 {
let mut size: u64 = mask & maxbase; // Find the significant bits
if size == 0 {
return 0;
}
// Get the lowest of them to find the decode size, and from that
// the extent.
size &= !size.wrapping_sub(1);
// base == maxbase can be valid only if the BAR has already been
// programmed with all 1s.
if base == maxbase && (base | size.wrapping_sub(1)) & mask != mask {
return 0;
}
size
}

fn decode_bar(dev: *mut pci_dev, bar: u32) -> u64 {
let mut mem_type: u32 = 0;
let mut flags: u64 = 0;
if bar & 1 == 1 {
flags = (bar as u64) & !(!3);
flags |= 256;
return flags;
}
flags = (bar as u64) & !(!15);
flags |= 512;
if flags & 8 != 0 {
flags |= 8192;
}
mem_type = bar & 6;
'__slate_break_6: {
match mem_type {
0 => {
}
2 => {
break '__slate_break_6; // 1M mem BAR treated as 32-bit BAR
}
4 => {
flags |= 1048576;
}
_ => {
break '__slate_break_6; // mem unknown type treated as 32-bit BAR
}
}
}
flags
}

/// __pci_size_bars - Read the raw BAR mask for a range of PCI BARs
/// @dev: the PCI device
/// @count: number of BARs to size
/// @pos: starting config space position
/// @sizes: array to store mask values
/// @rom: indicate whether to use ROM mask, which avoids enabling ROM BARs
///
/// Provided @sizes array must be sufficiently sized to store results for
/// @count u32 BARs.  Caller is responsible for disabling decode to specified
/// BAR range around calling this function.  This function is intended to avoid
/// disabling decode around sizing each BAR individually, which can result in
/// non-trivial overhead in virtualized environments with very large PCI BARs.
fn __pci_size_bars(dev: *mut pci_dev, count: i32, mut pos: u32, mut sizes: *mut u32, rom: bool) {
let mut orig: u32 = 0;
let mask: u32 = if rom { !2047 } else { 4294967295u32 };
let mut i: i32 = 0;
while i < count {
unsafe { pci_read_config_dword(dev as *const pci_dev, pos as i32, core::ptr::addr_of_mut!(orig)) };
unsafe { pci_write_config_dword(dev as *const pci_dev, pos as i32, mask) };
unsafe { pci_read_config_dword(dev as *const pci_dev, pos as i32, sizes) };
unsafe { pci_write_config_dword(dev as *const pci_dev, pos as i32, orig) };
i = i.wrapping_add(1);
pos = pos.wrapping_add(4);
let __v5: *mut u32 = sizes;
sizes = __v5.wrapping_offset(1 as isize);
}
}

#[unsafe(no_mangle)]
extern "C" fn __pci_size_stdbars(dev: *mut pci_dev, count: i32, pos: u32, sizes: *mut u32) {
__pci_size_bars(dev, count, pos, sizes, false);
}

fn __pci_size_rom(dev: *mut pci_dev, pos: u32, sizes: *mut u32) {
__pci_size_bars(dev, 1i32, pos, sizes, true);
}

/// __pci_read_base - Read a PCI BAR
/// @dev: the PCI device
/// @type: type of the BAR
/// @res: resource buffer to be filled in
/// @pos: BAR position in the config space
/// @sizes: array of one or more pre-read BAR masks
///
/// Returns 1 if the BAR is 64-bit, or 0 if 32-bit.
#[unsafe(no_mangle)]
extern "C" fn __pci_read_base(dev: *mut pci_dev, r#type: u32, res: *mut resource, pos: u32, sizes: *mut u32) -> i32 {
let mut res_name: *const u8 = unsafe { core::mem::zeroed() };
let mut inverted_region: core::mem::MaybeUninit<pci_bus_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut pci_bus_region = core::ptr::addr_of_mut!(inverted_region) as *mut pci_bus_region;
let mut region: core::mem::MaybeUninit<pci_bus_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut pci_bus_region = core::ptr::addr_of_mut!(region) as *mut pci_bus_region;
let mut mask64: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u64 = core::ptr::addr_of_mut!(mask64) as *mut u64;
let mut sz64: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u64 = core::ptr::addr_of_mut!(sz64) as *mut u64;
let mut l64: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u64 = core::ptr::addr_of_mut!(l64) as *mut u64;
let mut sz: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(sz) as *mut u32;
let mut l: u32 = unsafe { core::mem::zeroed() };
unsafe {
l = 0;
let __slate_auto_bytes = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 16usize);
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
res_name = pci_resource_name(dev, (res.offset_from(((*dev).resource.as_mut_ptr() as *mut resource) as *mut resource) as i64) as u32);
(*res).name = pci_name(dev as *const pci_dev);
pci_read_config_dword(dev as *const pci_dev, pos as i32, core::ptr::addr_of_mut!(l));
*__slate_slot_27 = *sizes.wrapping_offset(0 as isize);
// All bits set in sz means the device isn't working properly.
// If the BAR isn't implemented, all bits must be 0.  If it's a
// memory BAR or a ROM, bit 0 must be clear; if it's an io BAR, bit
// 1 must be clear.
if PCI_POSSIBLE_ERROR!(*__slate_slot_27) {
*__slate_slot_27 = 0;
}
// I don't know how l can have all bits set.  Copied from old code.
// Maybe it fixes a bug on some ancient platform.
if PCI_POSSIBLE_ERROR!(l) {
l = 0;
}
if (r#type as u32) == 0 {
(*res).flags = decode_bar(dev, l);
(*res).flags = (*res).flags | 262144;
if (*res).flags & 256 != 0 {
*__slate_slot_26 = (l as u64) & !3;
*__slate_slot_25 = (*__slate_slot_27 as u64) & !3;
*__slate_slot_24 = !3 & 65535;
} else {
*__slate_slot_26 = (l as u64) & !15;
*__slate_slot_25 = (*__slate_slot_27 as u64) & !15;
*__slate_slot_24 = 4294967280u64;
}
} else {
if l & 1 != 0 {
(*res).flags = (*res).flags | 1u64;
}
*__slate_slot_26 = (l & !2047) as u64;
*__slate_slot_25 = (*__slate_slot_27 & !2047) as u64;
*__slate_slot_24 = 4294965248u64;
}
if (*res).flags & 1048576 != 0 {
pci_read_config_dword(dev as *const pci_dev, pos.wrapping_add(4) as i32, core::ptr::addr_of_mut!(l));
*__slate_slot_27 = *sizes.wrapping_offset(1 as isize);
*__slate_slot_26 |= (l as u64) << 32i32;
*__slate_slot_25 |= (*__slate_slot_27 as u64) << 32i32;
let __v9: u64 = *__slate_slot_24 | 18446744069414584320u64;
*__slate_slot_24 = __v9;
}
'__join_3: {
if *__slate_slot_25 != 0 {
*__slate_slot_25 = pci_size(*__slate_slot_26, *__slate_slot_25, *__slate_slot_24);
if *__slate_slot_25 == 0 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"[Firmware Bug]: %s: invalid; can't size\n".as_ptr() as *const u8, res_name);
} else {
if (*res).flags & 1048576 != 0 {
}
(*__slate_slot_23).start = *__slate_slot_26;
(*__slate_slot_23).end = (*__slate_slot_26).wrapping_add(*__slate_slot_25).wrapping_sub(1);
pcibios_bus_to_resource((*dev).bus, res, core::ptr::addr_of_mut!(*__slate_slot_23));
pcibios_resource_to_bus((*dev).bus, core::ptr::addr_of_mut!(*__slate_slot_22), res);
// If "A" is a BAR value (a bus address), "bus_to_resource(A)" is
// the corresponding resource address (the physical address used by
// the CPU.  Converting that resource address back to a bus address
// should yield the original BAR value:
//
//     resource_to_bus(bus_to_resource(A)) == A
//
// If it doesn't, CPU accesses to "bus_to_resource(A)" will not
// be claimed by the device.
if (*__slate_slot_22).start != (*__slate_slot_23).start {
(*res).flags = (*res).flags | 536870912;
(*res).start = 0;
(*res).end = (*__slate_slot_23).end.wrapping_sub((*__slate_slot_23).start);
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s: initial BAR value %#010llx invalid\n".as_ptr() as *const u8, res_name, (*__slate_slot_23).start);
break '__join_3;
} else {
break '__join_3;
}
}
}
(*res).flags = 0;
}
if (*res).flags != 0 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s %pR\n".as_ptr() as *const u8, res_name, res);
}
return if (*res).flags & 1048576 != 0 { 1 } else { 0 };
}
// Above 32-bit boundary; try to reallocate
}

#[inline(always)]
fn pci_read_bases(dev: *mut pci_dev, howmany: u32, rom: i32) {
let mut rombar: u32 = 0;
let mut stdbars: __SlateAlign16<[u32; 6]> = __SlateAlign16([0u32; 6]);
let mut pos: u32 = 0;
let mut reg: u32 = 0;
let mut orig_cmd: u16 = 0;
loop {
if !(!statically_true!(howmany > 6)) {
unsafe { __compiletime_assert_554() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if ((unsafe { (*dev).__slate_bits_3.__get_non_compliant_bars() }) as i32) != 0 {
return;
}
// Per PCIe r4.0, sec 9.3.4.1.11, the VF BARs are all RO Zero
if ((unsafe { (*dev).__slate_bits_3.__get_is_virtfn() }) as i32) != 0 {
return;
}
// No printks while decoding is disabled!
if ((unsafe { (*dev).__slate_bits_1.__get_mmio_always_on() }) as i32) == 0 {
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(orig_cmd)) };
if (orig_cmd as i32) & (2 | 1) != 0 {
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, ((orig_cmd as i32) & !(2 | 1)) as u16) };
}
}
__pci_size_stdbars(dev, howmany as i32, 16u32, stdbars.0.as_mut_ptr() as *mut u32);
if rom != 0 {
__pci_size_rom(dev, rom as u32, core::ptr::addr_of_mut!(rombar));
}
if ((unsafe { (*dev).__slate_bits_1.__get_mmio_always_on() }) as i32) == 0 && (orig_cmd as i32) & (2 | 1) != 0 {
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, orig_cmd) };
}
while pos < howmany {
let res: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(pos as isize);
reg = 16u32.wrapping_add(pos << 2i32);
let __v1: i32 = __pci_read_base(dev, 0u32, res, reg, (stdbars.0.as_mut_ptr() as *mut u32).wrapping_offset(pos as isize));
pos = pos.wrapping_add(__v1 as u32);
pos = pos.wrapping_add(1);
}
if rom != 0 {
let res: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(6 as isize);
unsafe {
(*dev).rom_base_reg = rom as u8;
}
unsafe {
(*res).flags = 287232u64;
}
__pci_read_base(dev, 2u32, res, rom as u32, core::ptr::addr_of_mut!(rombar));
}
}

fn pci_read_bridge_io(dev: *mut pci_dev, res: *mut resource, log: bool) {
let mut io_base_lo: u8 = 0;
let mut io_limit_lo: u8 = 0;
let mut io_mask: u64 = !15;
let mut io_granularity: u64 = 4096;
let mut base: u64 = 0;
let mut limit: u64 = 0;
let mut region: pci_bus_region = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(region) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
if ((unsafe { (*dev).__slate_bits_3.__get_io_window() }) as i32) == 0 {
return;
}
if ((unsafe { (*dev).__slate_bits_3.__get_io_window_1k() }) as i32) != 0 {
// Support 1K I/O space granularity
io_mask = !3;
io_granularity = 1024;
}
unsafe { pci_read_config_byte(dev as *const pci_dev, 28i32, core::ptr::addr_of_mut!(io_base_lo)) };
unsafe { pci_read_config_byte(dev as *const pci_dev, 29i32, core::ptr::addr_of_mut!(io_limit_lo)) };
base = (((io_base_lo as i32) as u64) & io_mask) << 8i32;
limit = (((io_limit_lo as i32) as u64) & io_mask) << 8i32;
if ((io_base_lo as i32) as u64) & 15 == 1 {
let mut io_base_hi: u16 = 0;
let mut io_limit_hi: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, 48i32, core::ptr::addr_of_mut!(io_base_hi)) };
unsafe { pci_read_config_word(dev as *const pci_dev, 50i32, core::ptr::addr_of_mut!(io_limit_hi)) };
base |= (io_base_hi as u64) << 16i32;
limit |= (io_limit_hi as u64) << 16i32;
}
unsafe {
(*res).flags = ((io_base_lo as i32) as u64) & 15 | 256;
}
if base <= limit {
region.start = base;
region.end = limit.wrapping_add(io_granularity).wrapping_sub(1);
unsafe { pcibios_bus_to_resource((*dev).bus, res, core::ptr::addr_of_mut!(region)) };
if log {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"  bridge window %pR\n".as_ptr() as *const u8, res) };
}
} else {
resource_set_range(res, 0u64, 0u64);
unsafe {
(*res).flags = (*res).flags | 805306368u64;
}
}
}

fn pci_read_bridge_mmio(dev: *mut pci_dev, res: *mut resource, log: bool) {
let mut mem_base_lo: u16 = 0;
let mut mem_limit_lo: u16 = 0;
let mut base: u64 = 0;
let mut limit: u64 = 0;
let mut region: pci_bus_region = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(region) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
unsafe { pci_read_config_word(dev as *const pci_dev, 32i32, core::ptr::addr_of_mut!(mem_base_lo)) };
unsafe { pci_read_config_word(dev as *const pci_dev, 34i32, core::ptr::addr_of_mut!(mem_limit_lo)) };
base = ((mem_base_lo as u64) & !15) << 16i32;
limit = ((mem_limit_lo as u64) & !15) << 16i32;
unsafe {
(*res).flags = ((mem_base_lo as i32) as u64) & 15 | 512;
}
if base <= limit {
region.start = base;
region.end = limit.wrapping_add(1048575);
unsafe { pcibios_bus_to_resource((*dev).bus, res, core::ptr::addr_of_mut!(region)) };
if log {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"  bridge window %pR\n".as_ptr() as *const u8, res) };
}
} else {
resource_set_range(res, 0u64, 0u64);
unsafe {
(*res).flags = (*res).flags | 805306368u64;
}
}
}

fn pci_read_bridge_mmio_pref(dev: *mut pci_dev, res: *mut resource, log: bool) {
let mut mem_base_lo: u16 = 0;
let mut mem_limit_lo: u16 = 0;
let mut base64: u64 = 0;
let mut limit64: u64 = 0;
let mut base: u64 = 0;
let mut limit: u64 = 0;
let mut region: pci_bus_region = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(region) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
if ((unsafe { (*dev).__slate_bits_3.__get_pref_window() }) as i32) == 0 {
return;
}
unsafe { pci_read_config_word(dev as *const pci_dev, 36i32, core::ptr::addr_of_mut!(mem_base_lo)) };
unsafe { pci_read_config_word(dev as *const pci_dev, 38i32, core::ptr::addr_of_mut!(mem_limit_lo)) };
base64 = (((mem_base_lo as i32) as u64) & !15) << 16i32;
limit64 = (((mem_limit_lo as i32) as u64) & !15) << 16i32;
if ((mem_base_lo as i32) as u64) & 15 == 1 {
let mut mem_base_hi: u32 = 0;
let mut mem_limit_hi: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, 40i32, core::ptr::addr_of_mut!(mem_base_hi)) };
unsafe { pci_read_config_dword(dev as *const pci_dev, 44i32, core::ptr::addr_of_mut!(mem_limit_hi)) };
// Some bridges set the base > limit by default, and some
// (broken) BIOSes do not initialize them.  If we find
// this, just assume they are not being used.
if mem_base_hi <= mem_limit_hi {
base64 |= (mem_base_hi as u64) << 32i32;
limit64 |= (mem_limit_hi as u64) << 32i32;
}
}
base = base64;
limit = limit64;
if base != base64 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"can't handle bridge window above 4GB (bus address %#010llx)\n".as_ptr() as *const u8, base64) };
return;
}
unsafe {
(*res).flags = ((mem_base_lo as i32) as u64) & 15 | 512 | 8192;
}
if (unsafe { (*res).flags }) & 1 != 0 {
unsafe {
(*res).flags = (*res).flags | 1048576;
}
}
if base <= limit {
region.start = base;
region.end = limit.wrapping_add(1048575);
unsafe { pcibios_bus_to_resource((*dev).bus, res, core::ptr::addr_of_mut!(region)) };
if log {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"  bridge window %pR\n".as_ptr() as *const u8, res) };
}
} else {
resource_set_range(res, 0u64, 0u64);
unsafe {
(*res).flags = (*res).flags | 805306368u64;
}
}
}

fn pci_read_bridge_windows(bridge: *mut pci_dev) {
let mut buses: u32 = 0;
let mut io: u16 = 0;
let mut pmem: u32 = 0;
let mut tmp: u32 = 0;
let mut res: resource = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(res) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
unsafe { pci_read_config_dword(bridge as *const pci_dev, 24i32, core::ptr::addr_of_mut!(buses)) };
res.flags = 4096;
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(65280i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_555() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_556() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_557() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(65280i32 >> (65280u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_558() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)) & 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_559() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((buses & 65280) >> (65280u64.trailing_zeros() as i32)) as i32;
__v0 = __v2;
res.start = __v0 as u64;
let mut __v7: i32;
loop {
if __bf_cast_unsigned!(16711680i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_560() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v9: i32;
loop {
if false {
unsafe { __compiletime_assert_561() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_562() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(16711680i32 >> (16711680u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_563() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)) & 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_564() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = ((buses & 16711680) >> (16711680u64.trailing_zeros() as i32)) as i32;
__v7 = __v9;
res.end = __v7 as u64;
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*bridge).dev) as *const device, c"PCI bridge to %pR%s\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(res), if ((*bridge).__slate_bits_3.__get_transparent() as i32) != 0 { c" (subtractive decode)".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe { pci_read_config_word(bridge as *const pci_dev, 28i32, core::ptr::addr_of_mut!(io)) };
if io == 0 {
unsafe { pci_write_config_word(bridge as *const pci_dev, 28i32, 57584u16) };
unsafe { pci_read_config_word(bridge as *const pci_dev, 28i32, core::ptr::addr_of_mut!(io)) };
unsafe { pci_write_config_word(bridge as *const pci_dev, 28i32, 0u16) };
}
if io != 0 {
unsafe {
(*bridge).__slate_bits_3.__set_io_window(1u32);
}
pci_read_bridge_io(bridge, core::ptr::addr_of_mut!(res), true);
}
pci_read_bridge_mmio(bridge, core::ptr::addr_of_mut!(res), true);
// DECchip 21050 pass 2 errata: the bridge may miss an address
// disconnect boundary by one PCI data phase.  Workaround: do not
// use prefetching on this device.
if ((unsafe { (*bridge).vendor }) as i32) == 4113 && ((unsafe { (*bridge).device }) as i32) == 1 {
return;
}
unsafe { pci_read_config_dword(bridge as *const pci_dev, 36i32, core::ptr::addr_of_mut!(pmem)) };
if pmem == 0 {
unsafe { pci_write_config_dword(bridge as *const pci_dev, 36i32, 4292935664u32) };
unsafe { pci_read_config_dword(bridge as *const pci_dev, 36i32, core::ptr::addr_of_mut!(pmem)) };
unsafe { pci_write_config_dword(bridge as *const pci_dev, 36i32, 0u32) };
}
if pmem == 0 {
return;
}
unsafe {
(*bridge).__slate_bits_3.__set_pref_window(1u32);
}
if (pmem as u64) & 15 == 1 {
// Bridge claims to have a 64-bit prefetchable memory
// window; verify that the upper bits are actually
// writable.
unsafe { pci_read_config_dword(bridge as *const pci_dev, 40i32, core::ptr::addr_of_mut!(pmem)) };
unsafe { pci_write_config_dword(bridge as *const pci_dev, 40i32, 4294967295u32) };
unsafe { pci_read_config_dword(bridge as *const pci_dev, 40i32, core::ptr::addr_of_mut!(tmp)) };
unsafe { pci_write_config_dword(bridge as *const pci_dev, 40i32, pmem) };
if tmp != 0 {
unsafe {
(*bridge).__slate_bits_3.__set_pref_64_window(1u32);
}
}
}
pci_read_bridge_mmio_pref(bridge, core::ptr::addr_of_mut!(res), true);
}

#[unsafe(no_mangle)]
extern "C" fn pci_read_bridge_bases(child: *mut pci_bus) {
let dev: *mut pci_dev = unsafe { (*child).self_ };
if pci_is_root_bus(child) {
return;
}
// It's a host bus, nothing to read
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"PCI bridge to %pR%s\n".as_ptr() as *const u8, core::ptr::addr_of_mut!((*child).busn_res), if ((*dev).__slate_bits_3.__get_transparent() as i32) != 0 { c" (subtractive decode)".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe { pci_bus_remove_resources(child) };
for i in 0i32..4i32 {
unsafe {
*((*child).resource.as_mut_ptr() as *mut *mut resource).wrapping_offset(i as isize) = ((*dev).resource.as_mut_ptr() as *mut resource).wrapping_offset(7i32.wrapping_add(i) as isize);
}
}
pci_read_bridge_io(unsafe { (*child).self_ }, unsafe { *((*child).resource.as_mut_ptr() as *mut *mut resource).wrapping_offset(0 as isize) }, false);
pci_read_bridge_mmio(unsafe { (*child).self_ }, unsafe { *((*child).resource.as_mut_ptr() as *mut *mut resource).wrapping_offset(1 as isize) }, false);
pci_read_bridge_mmio_pref(unsafe { (*child).self_ }, unsafe { *((*child).resource.as_mut_ptr() as *mut *mut resource).wrapping_offset(2 as isize) }, false);
if ((unsafe { (*dev).__slate_bits_3.__get_transparent() }) as i32) == 0 {
return;
}
let mut __b: u32 = 0;
loop {
let __v6: *mut resource = unsafe { pci_bus_resource_n((*child).parent as *const pci_bus, __b as i32) };
let res: *mut resource = __v6;
if !(!__v6.is_null() || __b < 4) {
break;
}
if !(res.is_null() || (unsafe { (*res).flags }) == 0) {
unsafe { pci_bus_add_resource(child, res) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"  bridge window %pR (subtractive decode)\n".as_ptr() as *const u8, res) };
}
__b = __b.wrapping_add(1);
}
}

fn pci_alloc_bus(parent: *mut pci_bus) -> *mut pci_bus {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut b: *mut pci_bus = unsafe { core::mem::zeroed() };
let mut __v0: *mut pci_bus;
let __obj_size: u64 = size_mul(core::mem::size_of::<pci_bus>() as u64, 1u64);
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
__v0 = __v1 as *mut pci_bus;
b = __v0;
if b.is_null() {
return core::ptr::null_mut::<pci_bus>();
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*b).node) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*b).children) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*b).devices) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*b).slots) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*b).resources) });
unsafe {
(*b).max_bus_speed = 255;
}
unsafe {
(*b).cur_bus_speed = 255;
}
b
}

#[unsafe(link_section = ".text.slate_distinct.probe.pci_release_host_bridge_dev")]
extern "C" fn pci_release_host_bridge_dev(dev: *mut device) {
let bridge: *mut pci_host_bridge = to_pci_host_bridge!(dev);
if (unsafe { (*bridge).release_fn }) != None {
unsafe { unsafe { (*bridge).release_fn }.unwrap()(bridge) };
}
unsafe { pci_free_resource_list(core::ptr::addr_of_mut!((*bridge).windows)) };
unsafe { pci_free_resource_list(core::ptr::addr_of_mut!((*bridge).dma_ranges)) };
// Host bridges only have domain_nr set in the emulation case
if (unsafe { (*bridge).domain_nr }) != 1i32.wrapping_neg() {
unsafe { pci_bus_release_emul_domain_nr((*bridge).domain_nr) };
}
unsafe { kfree(bridge as *const ()) };
}

static mut pci_host_bridge_groups: [*const attribute_group; 1] = [core::ptr::null::<attribute_group>()];

static pci_host_bridge_type: __SlateConst<device_type> = __SlateConst({
let mut __t0: device_type = unsafe { core::mem::zeroed() };
__t0.groups = unsafe { core::ptr::addr_of_mut!(pci_host_bridge_groups) as *mut *const attribute_group };
__t0.release = Some(pci_release_host_bridge_dev);
__t0
});

fn pci_init_host_bridge(bridge: *mut pci_host_bridge) {
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*bridge).windows) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*bridge).dma_ranges) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*bridge).ports) });
// We assume we can manage these PCIe features.  Some systems may
// reserve these for use by the platform itself, e.g., an ACPI BIOS
// may implement its own AER handling and use _OSC to prevent the
// OS from interfering.
unsafe {
(*bridge).__slate_bits_0.__set_native_aer(1u32);
}
unsafe {
(*bridge).__slate_bits_0.__set_native_pcie_hotplug(1u32);
}
unsafe {
(*bridge).__slate_bits_0.__set_native_shpc_hotplug(1u32);
}
unsafe {
(*bridge).__slate_bits_0.__set_native_pme(1u32);
}
unsafe {
(*bridge).__slate_bits_0.__set_native_ltr(1u32);
}
unsafe {
(*bridge).__slate_bits_0.__set_native_dpc(1u32);
}
unsafe {
(*bridge).domain_nr = 1i32.wrapping_neg();
}
unsafe {
(*bridge).__slate_bits_0.__set_native_cxl_error(1u32);
}
unsafe {
(*bridge).dev.r#type = core::ptr::addr_of!(pci_host_bridge_type.0);
}
pci_ide_init_host_bridge(bridge);
unsafe { device_initialize(core::ptr::addr_of_mut!((*bridge).dev)) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_alloc_host_bridge")]
extern "C" fn pci_alloc_host_bridge(r#priv: u64) -> *mut pci_host_bridge {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(960u64.wrapping_add(r#priv), 3264u32, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(960u64.wrapping_add(r#priv), 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
let bridge: *mut pci_host_bridge = __v0 as *mut pci_host_bridge;
if bridge.is_null() {
return core::ptr::null_mut::<pci_host_bridge>();
}
pci_init_host_bridge(bridge);
bridge
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_alloc_host_bridge_565: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64) -> *mut pci_host_bridge>, *mut ()>(Some(pci_alloc_host_bridge)) };

#[unsafe(link_section = ".text.slate_distinct.probe.devm_pci_alloc_host_bridge_release")]
extern "C" fn devm_pci_alloc_host_bridge_release(data: *mut ()) {
pci_free_host_bridge(data as *mut pci_host_bridge);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.devm_pci_alloc_host_bridge")]
extern "C" fn devm_pci_alloc_host_bridge(dev: *mut device, r#priv: u64) -> *mut pci_host_bridge {
let mut ret: i32 = 0;
let bridge: *mut pci_host_bridge = pci_alloc_host_bridge(r#priv);
if bridge.is_null() {
return core::ptr::null_mut::<pci_host_bridge>();
}
unsafe {
(*bridge).dev.parent = dev;
}
ret = __devm_add_action_or_reset(dev, Some(devm_pci_alloc_host_bridge_release), bridge as *mut (), c"devm_pci_alloc_host_bridge_release".as_ptr() as *const u8);
if ret != 0 {
return core::ptr::null_mut::<pci_host_bridge>();
}
ret = devm_of_pci_bridge_init(dev, bridge);
if ret != 0 {
return core::ptr::null_mut::<pci_host_bridge>();
}
bridge
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_devm_pci_alloc_host_bridge_566: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, u64) -> *mut pci_host_bridge>, *mut ()>(Some(devm_pci_alloc_host_bridge)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_free_host_bridge")]
extern "C" fn pci_free_host_bridge(bridge: *mut pci_host_bridge) {
unsafe { put_device(core::ptr::addr_of_mut!((*bridge).dev)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_free_host_bridge_567: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_host_bridge)>, *mut ()>(Some(pci_free_host_bridge)) };

/// Indexed by PCI_X_SSTATUS_FREQ (secondary bus mode and frequency)
/// 0
/// 1
/// 2
/// 3
/// 4
/// 5
/// 6
/// 7
/// 8
/// 9
/// A
/// B
/// C
/// D
/// E
/// F
static pcix_bus_speed: __SlateAlign16<[u8; 16]> = __SlateAlign16([255u8, 2u8, 3u8, 4u8, 255u8, 5u8, 6u8, 7u8, 255u8, 9u8, 10u8, 11u8, 255u8, 17u8, 18u8, 19u8]);

/// Indexed by PCI_EXP_LNKCAP_SLS, PCI_EXP_LNKSTA_CLS
/// 0
/// 1
/// 2
/// 3
/// 4
/// 5
/// 6
/// 7
/// 8
/// 9
/// A
/// B
/// C
/// D
/// E
/// F
#[unsafe(no_mangle)]
static pcie_link_speed: __SlateAlign16<[u8; 16]> = __SlateAlign16([255u8, 20u8, 21u8, 22u8, 23u8, 24u8, 25u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_link_speed_568: *mut () = (unsafe { core::ptr::addr_of!(pcie_link_speed.0) }) as *mut ();

/// pcie_get_link_speed - Get speed value from PCIe generation number
/// @speed: PCIe speed (1-based: 1 = 2.5GT, 2 = 5GT, ...)
///
/// Returns the speed value (e.g., PCIE_SPEED_2_5GT) if @speed is valid,
/// otherwise returns PCI_SPEED_UNKNOWN.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pcie_get_link_speed")]
extern "C" fn pcie_get_link_speed(speed: u32) -> u8 {
if (speed as u64) >= ARRAY_SIZE__8!() {
return 255;
}
unsafe { pcie_link_speed.0[speed as usize] }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_get_link_speed_569: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> u8>, *mut ()>(Some(pcie_get_link_speed)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_speed_string")]
extern "C" fn pci_speed_string(speed: u32) -> *const u8 {
// Indexed by the pci_bus_speed enum
// 0x00
// 0x01
// 0x02
// 0x03
// 0x04
// 0x05
// 0x06
// 0x07
// 0x08
// 0x09
// 0x0a
// 0x0b
// 0x0c
// 0x0d
// 0x0e
// 0x0f
// 0x10
// 0x11
// 0x12
// 0x13
// 0x14
// 0x15
// 0x16
// 0x17
// 0x18
// 0x19
if ((speed as u32) as u64) < ARRAY_SIZE__3!() {
return unsafe { speed_strings.0[(speed as u32) as usize] };
}
c"Unknown".as_ptr() as *const u8
}

static mut speed_strings: __SlateAlign16<[*const u8; 26]> = __SlateAlign16([(b"33 MHz PCI\0".as_ptr() as *mut u8) as *const u8, (b"66 MHz PCI\0".as_ptr() as *mut u8) as *const u8, (b"66 MHz PCI-X\0".as_ptr() as *mut u8) as *const u8, (b"100 MHz PCI-X\0".as_ptr() as *mut u8) as *const u8, (b"133 MHz PCI-X\0".as_ptr() as *mut u8) as *const u8, core::ptr::null::<u8>(), core::ptr::null::<u8>(), core::ptr::null::<u8>(), core::ptr::null::<u8>(), (b"66 MHz PCI-X 266\0".as_ptr() as *mut u8) as *const u8, (b"100 MHz PCI-X 266\0".as_ptr() as *mut u8) as *const u8, (b"133 MHz PCI-X 266\0".as_ptr() as *mut u8) as *const u8, (b"Unknown AGP\0".as_ptr() as *mut u8) as *const u8, (b"1x AGP\0".as_ptr() as *mut u8) as *const u8, (b"2x AGP\0".as_ptr() as *mut u8) as *const u8, (b"4x AGP\0".as_ptr() as *mut u8) as *const u8, (b"8x AGP\0".as_ptr() as *mut u8) as *const u8, (b"66 MHz PCI-X 533\0".as_ptr() as *mut u8) as *const u8, (b"100 MHz PCI-X 533\0".as_ptr() as *mut u8) as *const u8, (b"133 MHz PCI-X 533\0".as_ptr() as *mut u8) as *const u8, (b"2.5 GT/s PCIe\0".as_ptr() as *mut u8) as *const u8, (b"5.0 GT/s PCIe\0".as_ptr() as *mut u8) as *const u8, (b"8.0 GT/s PCIe\0".as_ptr() as *mut u8) as *const u8, (b"16.0 GT/s PCIe\0".as_ptr() as *mut u8) as *const u8, (b"32.0 GT/s PCIe\0".as_ptr() as *mut u8) as *const u8, (b"64.0 GT/s PCIe\0".as_ptr() as *mut u8) as *const u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_speed_string_570: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> *const u8>, *mut ()>(Some(pci_speed_string)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pcie_update_link_speed")]
extern "C" fn pcie_update_link_speed(bus: *mut pci_bus, reason: u32) {
let bridge: *mut pci_dev = unsafe { (*bus).self_ };
let mut linksta: u16 = 0;
let mut linksta2: u16 = 0;
unsafe { pcie_capability_read_word(bridge, 18i32, core::ptr::addr_of_mut!(linksta)) };
unsafe { pcie_capability_read_word(bridge, 50i32, core::ptr::addr_of_mut!(linksta2)) };
__pcie_update_link_speed(bus, reason, linksta, linksta2);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_update_link_speed_571: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus, u32)>, *mut ()>(Some(pcie_update_link_speed)) };

static mut agp_speeds: [u8; 5] = [12, 13, 14, 15, 16];

fn agp_speed(agp3: i32, agpstat: i32) -> u32 {
let mut index: i32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
index = 0;
if agpstat & 4 != 0 {
index = 3;
} else {
if agpstat & 2 != 0 {
index = 2;
} else {
if agpstat & 1 != 0 {
index = 1;
} else {
break '__join_0;
}
}
}
if agp3 != 0 {
index = index.wrapping_add(2);
if index == 5 {
index = 0;
}
}
}
return agp_speeds[index as usize] as u32;
}
}

fn pci_set_bus_speed(bus: *mut pci_bus) {
let bridge: *mut pci_dev = unsafe { (*bus).self_ };
let mut pos: i32 = (unsafe { pci_find_capability(bridge, 2i32) }) as i32;
if pos == 0 {
pos = (unsafe { pci_find_capability(bridge, 14i32) }) as i32;
}
if pos != 0 {
let mut agpstat: u32 = 0;
let mut agpcmd: u32 = 0;
unsafe { pci_read_config_dword(bridge as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(agpstat)) };
unsafe {
(*bus).max_bus_speed = (agp_speed((agpstat & 8) as i32, (agpstat & 7) as i32) as u32) as u8;
}
unsafe { pci_read_config_dword(bridge as *const pci_dev, pos.wrapping_add(8), core::ptr::addr_of_mut!(agpcmd)) };
unsafe {
(*bus).cur_bus_speed = (agp_speed((agpstat & 8) as i32, (agpcmd & 7) as i32) as u32) as u8;
}
}
pos = (unsafe { pci_find_capability(bridge, 7i32) }) as i32;
if pos != 0 {
let mut status: u16 = 0;
let mut max: u32 = 0;
unsafe { pci_read_config_word(bridge as *const pci_dev, pos.wrapping_add(2), core::ptr::addr_of_mut!(status)) };
if (status as i32) & 32768 != 0 {
max = 19;
} else {
if (status as i32) & 16384 != 0 {
max = 11;
} else {
if (status as i32) & 2 != 0 {
if (status as i32) & 12288 == 8192 {
max = 7;
} else {
max = 4;
}
} else {
max = 2;
}
}
}
unsafe {
(*bus).max_bus_speed = (max as u32) as u8;
}
let mut __v0: i32;
while false {
}
let mut __v2: i32;
while false {
}
while false {
}
loop {
if (!(960i32 >> (960u64.trailing_zeros() as i32)) as u32) & 0u32 != 0 {
unsafe { __compiletime_assert_575() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 960u64.wrapping_add(1u64 << (960u64.trailing_zeros() as i32)) & 960u64.wrapping_add(1u64 << (960u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_576() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((status as i32) & 960) >> (960u64.trailing_zeros() as i32);
__v0 = __v2;
unsafe {
(*bus).cur_bus_speed = pcix_bus_speed.0[__v0 as usize];
}
return;
}
if pci_is_pcie(bridge) {
let mut linkcap: u32 = 0;
unsafe { pcie_capability_read_dword(bridge, 12i32, core::ptr::addr_of_mut!(linkcap)) };
unsafe {
(*bus).max_bus_speed = pcie_link_speed.0[(linkcap & 15) as usize];
}
pcie_update_link_speed(bus, 1u32);
}
}

fn pci_host_bridge_msi_domain(bus: *mut pci_bus) -> *mut irq_domain {
let mut d: *mut irq_domain = unsafe { core::mem::zeroed() };
// If the host bridge driver sets a MSI domain of the bridge, use it
d = dev_get_msi_domain((unsafe { (*bus).bridge }) as *const device);
// Any firmware interface that can resolve the msi_domain
// should be called from here.
if d.is_null() {
d = pci_host_bridge_of_msi_domain(bus);
}
if d.is_null() {
d = unsafe { pci_host_bridge_acpi_msi_domain(bus) };
}
// If no IRQ domain was found via the OF tree, try looking it up
// directly through the fwnode_handle.
if d.is_null() {
let fwnode: *mut fwnode_handle = _pci_root_bus_fwnode(bus) as *mut fwnode_handle;
if !fwnode.is_null() {
d = irq_find_matching_fwnode(fwnode, 3u32);
}
}
d
}

fn pci_set_bus_msi_domain(bus: *mut pci_bus) {
let mut d: *mut irq_domain = unsafe { core::mem::zeroed() };
let mut b: *mut pci_bus = unsafe { core::mem::zeroed() };
// The bus can be a root bus, a subordinate bus, or a virtual bus
// created by an SR-IOV device.  Walk up to the first bridge device
// found or derive the domain from the host bridge.
b = bus;
d = core::ptr::null_mut::<irq_domain>();
loop {
let __v0: bool = d.is_null() && !pci_is_root_bus(b);
if !__v0 {
break;
}
if !unsafe { (*b).self_ }.is_null() {
d = dev_get_msi_domain((unsafe { core::ptr::addr_of_mut!((*(*b).self_).dev) }) as *const device);
}
b = unsafe { (*b).parent };
}
if d.is_null() {
d = pci_host_bridge_msi_domain(b);
}
dev_set_msi_domain(unsafe { core::ptr::addr_of_mut!((*bus).dev) }, d);
}

fn pci_preserve_config(host_bridge: *mut pci_host_bridge) -> bool {
if unsafe { pci_acpi_preserve_config(host_bridge) } {
return true;
}
if !unsafe { (*host_bridge).dev.parent }.is_null() && !unsafe { (*(*host_bridge).dev.parent).of_node }.is_null() {
return of_pci_preserve_config(unsafe { (*(*host_bridge).dev.parent).of_node });
}
false
}

fn pci_register_host_bridge(bridge: *mut pci_host_bridge) -> i32 {
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut __v12: bool = unsafe { core::mem::zeroed() };
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut name: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *const u8 = core::ptr::addr_of_mut!(name) as *mut *const u8;
let mut fmt: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut u8 = core::ptr::addr_of_mut!(fmt) as *mut *mut u8;
let mut addr: core::mem::MaybeUninit<__SlateAlign16<[u8; 64]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut [u8; 64] = core::ptr::addr_of_mut!(addr) as *mut [u8; 64];
let mut bus_registered: bool = unsafe { core::mem::zeroed() };
let mut next_res: core::mem::MaybeUninit<*mut resource> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut resource = core::ptr::addr_of_mut!(next_res) as *mut *mut resource;
let mut res: core::mem::MaybeUninit<*mut resource> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut resource = core::ptr::addr_of_mut!(res) as *mut *mut resource;
let mut resources: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut list_head = core::ptr::addr_of_mut!(resources) as *mut list_head;
let mut next_offset: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u64 = core::ptr::addr_of_mut!(next_offset) as *mut u64;
let mut offset: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u64 = core::ptr::addr_of_mut!(offset) as *mut u64;
let mut b: core::mem::MaybeUninit<*mut pci_bus> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut pci_bus = core::ptr::addr_of_mut!(b) as *mut *mut pci_bus;
let mut bus: core::mem::MaybeUninit<*mut pci_bus> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut pci_bus = core::ptr::addr_of_mut!(bus) as *mut *mut pci_bus;
let mut n: core::mem::MaybeUninit<*mut resource_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut *mut resource_entry = core::ptr::addr_of_mut!(n) as *mut *mut resource_entry;
let mut next: core::mem::MaybeUninit<*mut resource_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut resource_entry = core::ptr::addr_of_mut!(next) as *mut *mut resource_entry;
let mut window: core::mem::MaybeUninit<*mut resource_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut resource_entry = core::ptr::addr_of_mut!(window) as *mut *mut resource_entry;
let mut parent: *mut device = unsafe { core::mem::zeroed() };
unsafe {
parent = (*bridge).dev.parent;
let __slate_auto_bytes = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_22, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_22), prev: core::ptr::addr_of_mut!(*__slate_slot_22) });
let __slate_auto_bytes_7 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
bus_registered = false;
let __slate_auto_bytes_9 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 64usize);
let __slate_auto_bytes_10 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_11 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_26 = pci_alloc_bus(core::ptr::null_mut::<pci_bus>());
if (*__slate_slot_26).is_null() {
return 12i32.wrapping_neg();
} else {
(*bridge).bus = *__slate_slot_26;
(*(*__slate_slot_26)).sysdata = (*bridge).sysdata;
(*(*__slate_slot_26)).ops = (*bridge).ops;
let __v14: u64 = (*bridge).busnr as u64;
(*(*__slate_slot_26)).busn_res.start = __v14;
(*(*__slate_slot_26)).number = __v14 as u8;
*__slate_slot_25 = pci_find_bus(pci_domain_nr(*__slate_slot_26), (*bridge).busnr);
if !(*__slate_slot_25).is_null() {
// Ignore it if we already got here via a different bridge
*__slate_slot_15 = 17i32.wrapping_neg();
} else {
dev_set_name(core::ptr::addr_of_mut!((*bridge).dev), c"pci%04x:%02x".as_ptr() as *const u8, pci_domain_nr(*__slate_slot_26), (*bridge).busnr);
*__slate_slot_15 = pcibios_root_bridge_prepare(bridge);
if *__slate_slot_15 == 0 {
// Temporarily move resources off the list
list_splice_init(core::ptr::addr_of_mut!((*bridge).windows), core::ptr::addr_of_mut!(*__slate_slot_22));
*__slate_slot_15 = device_add(core::ptr::addr_of_mut!((*bridge).dev));
if *__slate_slot_15 == 0 {
(*(*__slate_slot_26)).bridge = get_device(core::ptr::addr_of_mut!((*bridge).dev));
device_enable_async_suspend((*(*__slate_slot_26)).bridge);
pci_set_bus_of_node(*__slate_slot_26);
pci_set_bus_msi_domain(*__slate_slot_26);
if ((*bridge).__slate_bits_0.__get_msi_domain() as i32) != 0 {
__v13 = dev_get_msi_domain(core::ptr::addr_of_mut!((*(*__slate_slot_26)).dev) as *const device).is_null();
} else {
__v13 = false;
}
if __v13 {
__v12 = !pci_host_of_has_msi_map(parent);
} else {
__v12 = false;
}
if __v12 {
let __v11: *mut pci_bus = *__slate_slot_26;
(*__v11).bus_flags = (((*__v11).bus_flags as i32) | 1) as u16;
}
if parent.is_null() {
set_dev_node((*(*__slate_slot_26)).bridge, pcibus_to_node!(*__slate_slot_26));
}
(*(*__slate_slot_26)).dev.class = core::ptr::addr_of!(pcibus_class.0);
(*(*__slate_slot_26)).dev.parent = (*(*__slate_slot_26)).bridge;
dev_set_name(core::ptr::addr_of_mut!((*(*__slate_slot_26)).dev), c"%04x:%02x".as_ptr() as *const u8, pci_domain_nr(*__slate_slot_26), (*(*__slate_slot_26)).number as i32);
*__slate_slot_16 = dev_name(core::ptr::addr_of_mut!((*(*__slate_slot_26)).dev) as *const device);
*__slate_slot_15 = device_register(core::ptr::addr_of_mut!((*(*__slate_slot_26)).dev));
bus_registered = true;
if *__slate_slot_15 != 0 {
put_device(core::ptr::addr_of_mut!((*bridge).dev));
device_del(core::ptr::addr_of_mut!((*bridge).dev));
} else {
pcibios_add_bus(*__slate_slot_26);
if (*(*(*__slate_slot_26)).ops).add_bus != None {
*__slate_slot_15 = unsafe { (*(*(*__slate_slot_26)).ops).add_bus }.unwrap()(*__slate_slot_26);
__ret_warn_on = (*__slate_slot_15 < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("577: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 577b - ., 3; .popsection\n# {0}", const 577 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_register_host_bridge_38, sym __slate_asm_string_pci_register_host_bridge_39, const 1073 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("578: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 578b - ., 4; .popsection\n# {0}", const 578 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v8: i64 = (__ret_warn_on != 0) as i64;
if __v8 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*__slate_slot_26)).dev) as *const device, c"failed to add bus: %d\n".as_ptr() as *const u8, *__slate_slot_15);
}
}
if !parent.is_null() {
while false {
}
_dev_info(parent as *const device, c"PCI host bridge to bus %s\n".as_ptr() as *const u8, *__slate_slot_16);
} else {
while false {
}
_printk(c"\x016PCI host bridge to bus %s\n".as_ptr() as *const u8, *__slate_slot_16);
}
if nr_node_ids > 1 {
__v4 = pcibus_to_node!(*__slate_slot_26) == 1i32.wrapping_neg();
} else {
__v4 = false;
}
if __v4 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*(*__slate_slot_26)).dev) as *const device, c"Unknown NUMA node; performance will be reduced\n".as_ptr() as *const u8);
}
// Check if the boot configuration by FW needs to be preserved
let __t0 = pci_preserve_config(bridge) as u32;
(*bridge).__slate_bits_0.__set_preserve_config(__t0);
// Coalesce contiguous windows
*__slate_slot_29 = list_first_entry__3!(core::ptr::addr_of_mut!(*__slate_slot_22));
let __v3: *mut resource_entry = list_next_entry__3!(*__slate_slot_29, node);
*__slate_slot_27 = __v3;
while list_entry_is_head!(*__slate_slot_29, core::ptr::addr_of_mut!(*__slate_slot_22), node) == 0 {
if list_is_last(core::ptr::addr_of_mut!((*(*__slate_slot_29)).node) as *const list_head, &(*(core::ptr::addr_of_mut!(*__slate_slot_22) as *const list_head))) != 0 {
break;
} else {
*__slate_slot_28 = list_next_entry__3!(*__slate_slot_29, node);
*__slate_slot_24 = (*(*__slate_slot_29)).offset;
*__slate_slot_21 = (*(*__slate_slot_29)).res;
*__slate_slot_23 = (*(*__slate_slot_28)).offset;
*__slate_slot_20 = (*(*__slate_slot_28)).res;
if !((*(*__slate_slot_21)).flags != (*(*__slate_slot_20)).flags || *__slate_slot_24 != *__slate_slot_23) {
if unsafe { (*(*__slate_slot_21)).end }.wrapping_add(1) == (*(*__slate_slot_20)).start {
(*(*__slate_slot_20)).start = (*(*__slate_slot_21)).start;
(*(*__slate_slot_21)).end = 0;
(*(*__slate_slot_21)).start = 0;
(*(*__slate_slot_21)).flags = 0;
}
}
*__slate_slot_29 = *__slate_slot_27;
let __v2: *mut resource_entry = list_next_entry__3!(*__slate_slot_27, node);
*__slate_slot_27 = __v2;
}
}
// Add initial resources to the bus
*__slate_slot_29 = list_first_entry__3!(core::ptr::addr_of_mut!(*__slate_slot_22));
let __v1: *mut resource_entry = list_next_entry__3!(*__slate_slot_29, node);
*__slate_slot_27 = __v1;
while list_entry_is_head!(*__slate_slot_29, core::ptr::addr_of_mut!(*__slate_slot_22), node) == 0 {
*__slate_slot_24 = (*(*__slate_slot_29)).offset;
*__slate_slot_21 = (*(*__slate_slot_29)).res;
if (*(*__slate_slot_21)).flags == 0 && (*(*__slate_slot_21)).start == 0 && (*(*__slate_slot_21)).end == 0 {
release_resource(*__slate_slot_21);
resource_list_destroy_entry(*__slate_slot_29);
} else {
list_move_tail(core::ptr::addr_of_mut!((*(*__slate_slot_29)).node), core::ptr::addr_of_mut!((*bridge).windows));
if (*(*__slate_slot_21)).flags & 4096 != 0 {
pci_bus_insert_busn_res(*__slate_slot_26, (*(*__slate_slot_26)).number as i32, (*(*__slate_slot_21)).end as i32);
} else {
pci_bus_add_resource(*__slate_slot_26, *__slate_slot_21);
}
if *__slate_slot_24 != 0 {
if resource_type(*__slate_slot_21 as *const resource) == 256 {
*__slate_slot_17 = c" (bus address [%#06llx-%#06llx])".as_ptr() as *mut u8;
} else {
*__slate_slot_17 = c" (bus address [%#010llx-%#010llx])".as_ptr() as *mut u8;
}
snprintf((*__slate_slot_18).as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 64]>() as u64, *__slate_slot_17 as *const u8, unsafe { (*(*__slate_slot_21)).start }.wrapping_sub(*__slate_slot_24), unsafe { (*(*__slate_slot_21)).end }.wrapping_sub(*__slate_slot_24));
} else {
(*__slate_slot_18)[0] = 0;
}
while false {
}
_dev_info(core::ptr::addr_of_mut!((*(*__slate_slot_26)).dev) as *const device, c"root bus resource %pR%s\n".as_ptr() as *const u8, *__slate_slot_21, (*__slate_slot_18).as_mut_ptr() as *mut u8);
}
*__slate_slot_29 = *__slate_slot_27;
let __v0: *mut resource_entry = list_next_entry__3!(*__slate_slot_27, node);
*__slate_slot_27 = __v0;
}
of_pci_make_host_bridge_node(bridge);
down_write(core::ptr::addr_of_mut!(pci_bus_sem));
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_26)).node), core::ptr::addr_of_mut!(pci_root_buses));
up_write(core::ptr::addr_of_mut!(pci_bus_sem));
return 0;
}
}
}
}
if bus_registered {
put_device(core::ptr::addr_of_mut!((*(*__slate_slot_26)).dev));
} else {
kfree(*__slate_slot_26 as *const ());
}
return *__slate_slot_15;
}
}
}

fn pci_bridge_child_ext_cfg_accessible(bridge: *mut pci_dev) -> bool {
let mut pos: i32 = 0;
let mut status: u32 = 0;
// If extended config space isn't accessible on a bridge's primary
// bus, we certainly can't access it on the secondary bus.
if ((unsafe { (*(*bridge).bus).bus_flags }) as i32) & 8 != 0 {
return false;
}
// PCIe Root Ports and switch ports are PCIe on both sides, so if
// extended config space is accessible on the primary, it's also
// accessible on the secondary.
let __v0: bool = if pci_is_pcie(bridge) {
let __v1: bool = pci_pcie_type(bridge as *const pci_dev) == 4 || pci_pcie_type(bridge as *const pci_dev) == 5;
let __v2: bool = __v1 || pci_pcie_type(bridge as *const pci_dev) == 6;
__v2
} else {
false
};
if __v0 {
return true;
}
// For the other bridge types:
//   - PCI-to-PCI bridges
//   - PCIe-to-PCI/PCI-X forward bridges
//   - PCI/PCI-X-to-PCIe reverse bridges
// extended config space on the secondary side is only accessible
// if the bridge supports PCI-X Mode 2.
pos = (unsafe { pci_find_capability(bridge, 7i32) }) as i32;
if pos == 0 {
return false;
}
unsafe { pci_read_config_dword(bridge as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(status)) };
status & (1073741824 | 2147483648) != 0
}

fn pci_alloc_child_bus(parent: *mut pci_bus, bridge: *mut pci_dev, busnr: i32) -> *mut pci_bus {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut host: core::mem::MaybeUninit<*mut pci_host_bridge> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut pci_host_bridge = core::ptr::addr_of_mut!(host) as *mut *mut pci_host_bridge;
let mut child: core::mem::MaybeUninit<*mut pci_bus> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut pci_bus = core::ptr::addr_of_mut!(child) as *mut *mut pci_bus;
unsafe {
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Allocate a new bus and inherit stuff from the parent
*__slate_slot_15 = pci_alloc_bus(parent);
if (*__slate_slot_15).is_null() {
return core::ptr::null_mut::<pci_bus>();
} else {
(*(*__slate_slot_15)).parent = parent;
(*(*__slate_slot_15)).sysdata = (*parent).sysdata;
(*(*__slate_slot_15)).bus_flags = (*parent).bus_flags;
*__slate_slot_14 = pci_find_host_bridge(parent);
if !unsafe { (*(*__slate_slot_14)).child_ops }.is_null() {
(*(*__slate_slot_15)).ops = (*(*__slate_slot_14)).child_ops;
} else {
(*(*__slate_slot_15)).ops = (*parent).ops;
}
// Initialize some portions of the bus device, but don't register
// it now as the parent is not properly set up yet.
(*(*__slate_slot_15)).dev.class = core::ptr::addr_of!(pcibus_class.0);
dev_set_name(core::ptr::addr_of_mut!((*(*__slate_slot_15)).dev), c"%04x:%02x".as_ptr() as *const u8, pci_domain_nr(*__slate_slot_15), busnr);
// Set up the primary, secondary and subordinate bus numbers
let __v11: u64 = busnr as u64;
(*(*__slate_slot_15)).busn_res.start = __v11;
(*(*__slate_slot_15)).number = __v11 as u8;
(*(*__slate_slot_15)).primary = (*parent).busn_res.start as u8;
(*(*__slate_slot_15)).busn_res.end = 255;
if bridge.is_null() {
(*(*__slate_slot_15)).dev.parent = (*parent).bridge;
} else {
(*(*__slate_slot_15)).self_ = bridge;
(*(*__slate_slot_15)).bridge = get_device(core::ptr::addr_of_mut!((*bridge).dev));
(*(*__slate_slot_15)).dev.parent = (*(*__slate_slot_15)).bridge;
pci_set_bus_of_node(*__slate_slot_15);
pci_set_bus_speed(*__slate_slot_15);
// Check whether extended config space is accessible on the child
// bus.  Note that we currently assume it is always accessible on
// the root bus.
if !pci_bridge_child_ext_cfg_accessible(bridge) {
let __v10: *mut pci_bus = *__slate_slot_15;
(*__v10).bus_flags = (((*__v10).bus_flags as i32) | 8) as u16;
while false {
}
_dev_info(core::ptr::addr_of_mut!((*(*__slate_slot_15)).dev) as *const device, c"extended config space not accessible\n".as_ptr() as *const u8);
}
// Set up default resource pointers and names
*__slate_slot_13 = 0;
while *__slate_slot_13 < 4 {
*((*(*__slate_slot_15)).resource.as_mut_ptr() as *mut *mut resource).wrapping_offset(*__slate_slot_13 as isize) = ((*bridge).resource.as_mut_ptr() as *mut resource).wrapping_offset(7i32.wrapping_add(*__slate_slot_13) as isize);
(*(*((*(*__slate_slot_15)).resource.as_mut_ptr() as *mut *mut resource).wrapping_offset(*__slate_slot_13 as isize))).name = ((*(*__slate_slot_15)).name.as_mut_ptr() as *mut u8) as *const u8;
let __v7: i32 = *__slate_slot_13;
*__slate_slot_13 = __v7.wrapping_add(1);
}
(*bridge).subordinate = *__slate_slot_15;
}
pci_set_bus_msi_domain(*__slate_slot_15);
*__slate_slot_12 = device_register(core::ptr::addr_of_mut!((*(*__slate_slot_15)).dev));
__ret_warn_on_1 = (*__slate_slot_12 < 0) as i32;
if unlikely!(__ret_warn_on_1) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("579: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 579b - ., 3; .popsection\n# {0}", const 579 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_alloc_child_bus_18, sym __slate_asm_string_pci_alloc_child_bus_19, const 1269 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("580: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 580b - ., 4; .popsection\n# {0}", const 580 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v5: i64 = (__ret_warn_on_1 != 0) as i64;
if __v5 != 0 {
put_device(core::ptr::addr_of_mut!((*(*__slate_slot_15)).dev));
return core::ptr::null_mut::<pci_bus>();
} else {
pcibios_add_bus(*__slate_slot_15);
if (*(*(*__slate_slot_15)).ops).add_bus != None {
*__slate_slot_12 = unsafe { (*(*(*__slate_slot_15)).ops).add_bus }.unwrap()(*__slate_slot_15);
__ret_warn_on = (*__slate_slot_12 < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("581: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 581b - ., 3; .popsection\n# {0}", const 581 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_alloc_child_bus_16, sym __slate_asm_string_pci_alloc_child_bus_17, const 1278 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("582: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 582b - ., 4; .popsection\n# {0}", const 582 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*__slate_slot_15)).dev) as *const device, c"failed to add bus: %d\n".as_ptr() as *const u8, *__slate_slot_12);
}
}
return *__slate_slot_15;
}
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_add_new_bus")]
extern "C" fn pci_add_new_bus(parent: *mut pci_bus, dev: *mut pci_dev, busnr: i32) -> *mut pci_bus {
let mut child: *mut pci_bus = unsafe { core::mem::zeroed() };
child = pci_alloc_child_bus(parent, dev, busnr);
if !child.is_null() {
unsafe { down_write(core::ptr::addr_of_mut!(pci_bus_sem)) };
list_add_tail(unsafe { core::ptr::addr_of_mut!((*child).node) }, unsafe { core::ptr::addr_of_mut!((*parent).children) });
unsafe { up_write(core::ptr::addr_of_mut!(pci_bus_sem)) };
}
child
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_add_new_bus_583: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus, *mut pci_dev, i32) -> *mut pci_bus>, *mut ()>(Some(pci_add_new_bus)) };

fn pci_enable_rrs_sv(pdev: *mut pci_dev) {
let mut root_cap: u16 = 0;
// Enable Configuration RRS Software Visibility if supported
unsafe { pcie_capability_read_word(pdev, 30i32, core::ptr::addr_of_mut!(root_cap)) };
if (root_cap as i32) & 1 != 0 {
pcie_capability_set_word(pdev, 28i32, 16u16);
unsafe {
(*pdev).__slate_bits_1.__set_config_rrs_sv(1u32);
}
}
}

fn pci_scan_child_bus_extend(bus: *mut pci_bus, available_buses: u32) -> u32 {
let mut used_buses: u32 = 0;
let mut normal_bridges: u32 = 0;
let mut hotplug_bridges: u32 = 0;
let start: u32 = (unsafe { (*bus).busn_res.start }) as u32;
let mut cmax: u32 = 0;
let mut max: u32 = start;
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
// Go find them, Rover!
for devnr in 0u32..32u32 {
pci_scan_slot(bus, PCI_DEVFN__2!(devnr, 0i32) as i32);
}
// Reserve buses for SR-IOV capability
used_buses = pci_iov_bus_range(bus) as u32;
max = max.wrapping_add(used_buses);
// After performing arch-dependent fixup of the bus, look behind
// all PCI-to-PCI bridges on this bus.
if ((unsafe { (*bus).__slate_bits_0.__get_is_added() }) as i32) == 0 {
pcibios_fixup_bus(bus);
unsafe {
(*bus).__slate_bits_0.__set_is_added(1u32);
}
}
// Calculate how many hotplug bridges and normal bridges there
// are on this bus. We will distribute the additional available
// buses between hotplug bridges.
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*bus).devices) }, bus_list) == 0 {
if pci_is_bridge(dev) {
if ((unsafe { (*dev).__slate_bits_3.__get_is_hotplug_bridge() }) as i32) != 0 {
hotplug_bridges = hotplug_bridges.wrapping_add(1);
} else {
normal_bridges = normal_bridges.wrapping_add(1);
}
}
dev = list_next_entry!(dev, bus_list);
}
// Scan bridges that are already configured. We don't touch them
// unless they are misconfigured (which will be done in the second
// scan below).
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*bus).devices) }, bus_list) == 0 {
if pci_is_bridge(dev) {
cmax = max;
max = pci_scan_bridge_extend(bus, dev, max as i32, 0u32, 0i32) as u32;
// Reserve one bus for each bridge now to avoid extending
// hotplug bridges too much during the second scan below.
used_buses = used_buses.wrapping_add(1);
if max.wrapping_sub(cmax) > 1 {
used_buses = used_buses.wrapping_add(max.wrapping_sub(cmax).wrapping_sub(1));
}
}
dev = list_next_entry!(dev, bus_list);
}
// Scan bridges that need to be reconfigured
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*bus).devices) }, bus_list) == 0 {
if pci_is_bridge(dev) {
let mut buses: u32 = 0;
if hotplug_bridges == 0 && normal_bridges == 1 {
// There is only one bridge on the bus (upstream
// port) so it gets all available buses which it
// can then distribute to the possible hotplug
// bridges below.
buses = available_buses;
} else {
if ((unsafe { (*dev).__slate_bits_3.__get_is_hotplug_bridge() }) as i32) != 0 {
// Distribute the extra buses between hotplug
// bridges if any.
buses = available_buses / hotplug_bridges;
let mut __v20: u32;
let __UNIQUE_ID_x__668: u32 = buses;
let __UNIQUE_ID_y__669: u32 = available_buses.wrapping_sub(used_buses).wrapping_add(1);
while false {
}
__v20 = if __UNIQUE_ID_x__668 < __UNIQUE_ID_y__669 { __UNIQUE_ID_x__668 } else { __UNIQUE_ID_y__669 };
buses = __v20;
}
}
cmax = max;
max = pci_scan_bridge_extend(bus, dev, cmax as i32, buses, 1i32) as u32;
// One bus is already accounted so don't add it again
if max.wrapping_sub(cmax) > 1 {
used_buses = used_buses.wrapping_add(max.wrapping_sub(cmax).wrapping_sub(1));
}
}
dev = list_next_entry!(dev, bus_list);
}
// Make sure a hotplug bridge has at least the minimum requested
// number of buses but allow it to grow up to the maximum available
// bus number if there is room.
if !unsafe { (*bus).self_ }.is_null() && ((unsafe { (*(*bus).self_).__slate_bits_3.__get_is_hotplug_bridge() }) as i32) != 0 {
let mut __v26: u64;
let __UNIQUE_ID_x__671: u32 = available_buses;
let __UNIQUE_ID_y__672: u64 = unsafe { pci_hotplug_bus_size }.wrapping_sub(1);
while false {
}
__v26 = if (__UNIQUE_ID_x__671 as u64) > __UNIQUE_ID_y__672 { __UNIQUE_ID_x__671 as u64 } else { __UNIQUE_ID_y__672 };
used_buses = __v26 as u32;
if max.wrapping_sub(start) < used_buses {
max = start.wrapping_add(used_buses);
// Do not allocate more buses than we have room left
if (max as u64) > unsafe { (*bus).busn_res.end } {
max = (unsafe { (*bus).busn_res.end }) as u32;
}
}
}
// We've scanned the bus and so we know all about what's on
// the other side of any bridges that may be on this bus plus
// any devices.
//
// Return how far we've got finding sub-buses.
max
}

#[unsafe(no_mangle)]
extern "C" fn pbus_validate_busn(bus: *mut pci_bus) {
let mut upstream: *mut pci_bus = unsafe { (*bus).parent };
let bridge: *mut pci_dev = unsafe { (*bus).self_ };
// Check that all devices are accessible
while !unsafe { (*upstream).parent }.is_null() {
if (unsafe { (*bus).busn_res.end }) > unsafe { (*upstream).busn_res.end } || (((unsafe { (*bus).number }) as i32) as u64) > unsafe { (*upstream).busn_res.end } || ((unsafe { (*bus).number }) as i32) < ((unsafe { (*upstream).number }) as i32) || (unsafe { (*bus).busn_res.end }) < (((unsafe { (*upstream).number }) as i32) as u64) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*bridge).dev) as *const device, c"devices behind bridge are unusable because %pR cannot be assigned for them\n".as_ptr() as *const u8, core::ptr::addr_of_mut!((*bus).busn_res)) };
break;
}
upstream = unsafe { (*upstream).parent };
}
}

// pci_ea_fixed_busnrs() - Read fixed Secondary and Subordinate bus
// numbers from EA capability.
// @dev: Bridge
// @sec: updated with secondary bus number from EA
// @sub: updated with subordinate bus number from EA
//
// If @dev is a bridge with EA capability that specifies valid secondary
// and subordinate bus numbers, return true with the bus numbers in @sec
// and @sub.  Otherwise return false.
// pci_scan_bridge_extend() - Scan buses behind a bridge
// @bus: Parent bus the bridge is on
// @dev: Bridge itself
// @max: Starting subordinate number of buses behind this bridge
// @available_buses: Total number of buses available for this bridge and
// 		     the devices below. After the minimal bus space has
// 		     been allocated the remaining buses will be
// 		     distributed equally between hotplug-capable bridges.
// @pass: Either %0 (scan already configured bridges) or %1 (scan bridges
//        that need to be reconfigured.
//
// If it's a bridge, configure it and scan the bus behind it.
// For CardBus bridges, we don't scan behind as the devices will
// be handled by the bridge driver itself.
//
// We need to process bridges in two passes -- first we scan those
// already configured by the BIOS and after we are done with all of
// them, we proceed to assigning numbers to the remaining buses in
// order to avoid overlaps between old and new bus numbers.
//
// Return: New subordinate number covering all buses behind this bridge.
fn pci_scan_bridge_extend(bus: *mut pci_bus, dev: *mut pci_dev, mut max: i32, mut available_buses: u32, pass: i32) -> i32 {
let mut buses: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(buses) as *mut u32;
let mut cmax: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(cmax) as *mut u32;
let mut __v5: i32 = unsafe { core::mem::zeroed() };
let mut __v7: i32 = unsafe { core::mem::zeroed() };
let mut __v12: bool = unsafe { core::mem::zeroed() };
let mut next_busnr: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(next_busnr) as *mut i32;
let mut fixed_sub: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u8 = core::ptr::addr_of_mut!(fixed_sub) as *mut u8;
let mut fixed_sec: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u8 = core::ptr::addr_of_mut!(fixed_sec) as *mut u8;
let mut fixed_buses: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut bool = core::ptr::addr_of_mut!(fixed_buses) as *mut bool;
let mut broken: i32 = unsafe { core::mem::zeroed() };
let mut subordinate: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u8 = core::ptr::addr_of_mut!(subordinate) as *mut u8;
let mut secondary: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u8 = core::ptr::addr_of_mut!(secondary) as *mut u8;
let mut primary: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u8 = core::ptr::addr_of_mut!(primary) as *mut u8;
let mut bctl: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u16 = core::ptr::addr_of_mut!(bctl) as *mut u16;
let mut buses: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(buses) as *mut u32;
let mut child: core::mem::MaybeUninit<*mut pci_bus> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut pci_bus = core::ptr::addr_of_mut!(child) as *mut *mut pci_bus;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_4 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_5 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_6 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 1usize);
broken = 0;
let __slate_auto_bytes_7 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_8 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_9 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_10 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
// Make sure the bridge is powered on to be able to access config
// space of devices below it.
pm_runtime_get_sync(core::ptr::addr_of_mut!((*dev).dev));
pci_read_config_dword(dev as *const pci_dev, 24i32, core::ptr::addr_of_mut!(*__slate_slot_28));
while false {
}
while false {
}
while false {
}
loop {
if (!(255i32 >> (255u64.trailing_zeros() as i32)) as u32) & 0u32 != 0 {
__compiletime_assert_597();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)) & 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_598();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_26 = (((*__slate_slot_28 & 255) >> (255u64.trailing_zeros() as i32)) as i32) as u8;
while false {
}
while false {
}
while false {
}
loop {
if (!(65280i32 >> (65280u64.trailing_zeros() as i32)) as u32) & 0u32 != 0 {
__compiletime_assert_602();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)) & 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_603();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_25 = (((*__slate_slot_28 & 65280) >> (65280u64.trailing_zeros() as i32)) as i32) as u8;
while false {
}
while false {
}
while false {
}
loop {
if (!(16711680i32 >> (16711680u64.trailing_zeros() as i32)) as u32) & 0u32 != 0 {
__compiletime_assert_607();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)) & 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_608();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_24 = (((*__slate_slot_28 & 16711680) >> (16711680u64.trailing_zeros() as i32)) as i32) as u8;
if *__slate_slot_26 == 0 && (*__slate_slot_26 as i32) != ((*bus).number as i32) && *__slate_slot_25 != 0 && *__slate_slot_24 != 0 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Primary bus is hard wired to 0\n".as_ptr() as *const u8);
*__slate_slot_26 = (*bus).number;
}
// Check if setup is sensible at all
if pass == 0 && ((*__slate_slot_26 as i32) != ((*bus).number as i32) || (*__slate_slot_25 as i32) <= ((*bus).number as i32) || (*__slate_slot_25 as i32) > (*__slate_slot_24 as i32)) {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"bridge configuration invalid ([bus %02x-%02x]), reconfiguring\n".as_ptr() as *const u8, *__slate_slot_25 as i32, *__slate_slot_24 as i32);
broken = 1;
}
'__join_0: {
// Disable Master-Abort Mode during probing to avoid reporting of
// bus errors in some architectures.
pci_read_config_word(dev as *const pci_dev, 62i32, core::ptr::addr_of_mut!(*__slate_slot_27));
pci_write_config_word(dev as *const pci_dev, 62i32, ((*__slate_slot_27 as i32) & !32) as u16);
if pci_is_cardbus_bridge(dev) {
max = pci_cardbus_scan_bridge_extend(bus, dev, *__slate_slot_28, max, available_buses, pass);
} else {
if *__slate_slot_25 != 0 || *__slate_slot_24 != 0 {
__v12 = pcibios_assign_all_busses() == 0;
} else {
__v12 = false;
}
if __v12 && broken == 0 {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Bus already configured by firmware, process it in the
// first pass and just note the configuration.
if pass != 0 {
break '__join_0;
} else {
// The bus might already exist for two reasons: Either we
// are rescanning the bus or the bus is reachable through
// more than one bridge. The second case can happen with
// the i450NX chipset.
*__slate_slot_29 = pci_find_bus(pci_domain_nr(bus), *__slate_slot_25 as i32);
if (*__slate_slot_29).is_null() {
*__slate_slot_29 = pci_add_new_bus(bus, dev, *__slate_slot_25 as i32);
if (*__slate_slot_29).is_null() {
break '__join_0;
} else {
(*(*__slate_slot_29)).primary = *__slate_slot_26;
pci_bus_insert_busn_res(*__slate_slot_29, *__slate_slot_25 as i32, *__slate_slot_24 as i32);
(*(*__slate_slot_29)).bridge_ctl = *__slate_slot_27;
}
}
*__slate_slot_0 = (*__slate_slot_24 as i32).wrapping_sub(*__slate_slot_25 as i32) as u32;
*__slate_slot_1 = pci_scan_child_bus_extend(*__slate_slot_29, *__slate_slot_0);
if *__slate_slot_1 > ((*__slate_slot_24 as i32) as u32) {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"bridge has subordinate %02x but max busn %02x\n".as_ptr() as *const u8, *__slate_slot_24 as i32, *__slate_slot_1);
}
// Subordinate should equal child->busn_res.end
if (*__slate_slot_24 as i32) > max {
max = *__slate_slot_24 as i32;
}
}
} else {
// We need to assign a number to this bus which we always
// do in the second pass.
if pass == 0 {
if pcibios_assign_all_busses() != 0 || broken != 0 {
pci_write_config_dword(dev as *const pci_dev, 24i32, *__slate_slot_28 & 4278190080);
}
// Temporarily disable forwarding of the
// configuration cycles on all bridges in
// this bus segment to avoid possible
// conflicts in the second pass between two
// bridges programmed with overlapping bus
// ranges.
break '__join_0;
} else {
// Clear errors
pci_write_config_word(dev as *const pci_dev, 6i32, 65535u16);
// Read bus numbers from EA Capability (if present)
*__slate_slot_22 = pci_ea_fixed_busnrs(dev, core::ptr::addr_of_mut!(*__slate_slot_21), core::ptr::addr_of_mut!(*__slate_slot_20));
if *__slate_slot_22 {
*__slate_slot_19 = *__slate_slot_21 as i32;
} else {
*__slate_slot_19 = max.wrapping_add(1);
}
// Prevent assigning a bus number that already exists.
// This can happen when a bridge is hot-plugged, so in this
// case we only re-scan this bus.
*__slate_slot_29 = pci_find_bus(pci_domain_nr(bus), *__slate_slot_19);
if (*__slate_slot_29).is_null() {
*__slate_slot_29 = pci_add_new_bus(bus, dev, *__slate_slot_19);
if (*__slate_slot_29).is_null() {
break '__join_0;
} else {
pci_bus_insert_busn_res(*__slate_slot_29, *__slate_slot_19, (*bus).busn_res.end as i32);
}
}
max = max.wrapping_add(1);
if available_buses != 0 {
available_buses = available_buses.wrapping_sub(1);
}
while false {
}
while false {
}
while false {
}
loop {
if (if __slate_intrinsic_70b71f185ebb8c4a((*(*__slate_slot_29)).primary) { !(255i32 >> (255u64.trailing_zeros() as i32)) & 0i32.wrapping_add((*(*__slate_slot_29)).primary as i32) } else { 0i32 }) != 0 {
__compiletime_assert_612();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)) & 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_613();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
__v7 = ((*(*__slate_slot_29)).primary as i32) << (255u64.trailing_zeros() as i32) & 255;
while false {
}
while false {
}
while false {
}
loop {
if (if __slate_intrinsic_f217c4720f04f6db((*(*__slate_slot_29)).busn_res.start) { (!(65280i32 >> (65280u64.trailing_zeros() as i32)) as u64) & 0u64.wrapping_add((*(*__slate_slot_29)).busn_res.start) } else { 0u64 }) != 0 {
__compiletime_assert_617();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)) & 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_618();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
__v5 = ((*(*__slate_slot_29)).busn_res.start as i32) << (65280u64.trailing_zeros() as i32) & 65280;
while false {
}
while false {
}
while false {
}
loop {
if (if __slate_intrinsic_f217c4720f04f6db((*(*__slate_slot_29)).busn_res.end) { (!(16711680i32 >> (16711680u64.trailing_zeros() as i32)) as u64) & 0u64.wrapping_add((*(*__slate_slot_29)).busn_res.end) } else { 0u64 }) != 0 {
__compiletime_assert_622();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)) & 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_623();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_28 = *__slate_slot_28 & 4278190080 | (__v7 as u32) | (__v5 as u32) | ((((*(*__slate_slot_29)).busn_res.end as i32) << (16711680u64.trailing_zeros() as i32) & 16711680) as u32);
// We need to blast all three values with a single write
pci_write_config_dword(dev as *const pci_dev, 24i32, *__slate_slot_28);
(*(*__slate_slot_29)).bridge_ctl = *__slate_slot_27;
max = pci_scan_child_bus_extend(*__slate_slot_29, available_buses) as i32;
// Set subordinate bus number to its real value.
// If fixed subordinate bus number exists from EA
// capability then use it.
if *__slate_slot_22 {
max = *__slate_slot_20 as i32;
}
pci_bus_update_busn_res_end(*__slate_slot_29, max);
pci_write_config_byte(dev as *const pci_dev, 26i32, max as u8);
}
}
scnprintf((*(*__slate_slot_29)).name.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 48]>() as u64, c"PCI Bus %04x:%02x".as_ptr() as *const u8, pci_domain_nr(bus), (*(*__slate_slot_29)).number as i32);
pbus_validate_busn(*__slate_slot_29);
}
}
pci_write_config_word(dev as *const pci_dev, 30i32, 65535u16); // Clear errors in the Secondary Status Register
pci_write_config_word(dev as *const pci_dev, 62i32, *__slate_slot_27);
pm_runtime_put(core::ptr::addr_of_mut!((*dev).dev));
return max;
}
}

/// pci_scan_bridge() - Scan buses behind a bridge
/// @bus: Parent bus the bridge is on
/// @dev: Bridge itself
/// @max: Starting subordinate number of buses behind this bridge
/// @pass: Either %0 (scan already configured bridges) or %1 (scan bridges
///        that need to be reconfigured.
///
/// If it's a bridge, configure it and scan the bus behind it.
/// For CardBus bridges, we don't scan behind as the devices will
/// be handled by the bridge driver itself.
///
/// We need to process bridges in two passes -- first we scan those
/// already configured by the BIOS and after we are done with all of
/// them, we proceed to assigning numbers to the remaining buses in
/// order to avoid overlaps between old and new bus numbers.
///
/// Return: New subordinate number covering all buses behind this bridge.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_scan_bridge")]
extern "C" fn pci_scan_bridge(bus: *mut pci_bus, dev: *mut pci_dev, max: i32, pass: i32) -> i32 {
pci_scan_bridge_extend(bus, dev, max, 0u32, pass)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_scan_bridge_624: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus, *mut pci_dev, i32, i32) -> i32>, *mut ()>(Some(pci_scan_bridge)) };

/// Read interrupt line and base address registers.
/// The architecture-dependent code can tweak these, of course.
fn pci_read_irq(dev: *mut pci_dev) {
let mut irq: u8 = 0;
// VFs are not allowed to use INTx, so skip the config reads
if ((unsafe { (*dev).__slate_bits_3.__get_is_virtfn() }) as i32) != 0 {
unsafe {
(*dev).pin = 0;
}
unsafe {
(*dev).irq = 0;
}
return;
}
unsafe { pci_read_config_byte(dev as *const pci_dev, 61i32, core::ptr::addr_of_mut!(irq)) };
unsafe {
(*dev).pin = irq;
}
if irq != 0 {
unsafe { pci_read_config_byte(dev as *const pci_dev, 60i32, core::ptr::addr_of_mut!(irq)) };
}
unsafe {
(*dev).irq = irq as u32;
}
}

#[unsafe(no_mangle)]
extern "C" fn set_pcie_hotplug_bridge(pdev: *mut pci_dev) {
let mut reg32: u32 = 0;
unsafe { pcie_capability_read_dword(pdev, 20i32, core::ptr::addr_of_mut!(reg32)) };
if reg32 & 64 != 0 {
unsafe {
(*pdev).__slate_bits_3.__set_is_pciehp(1u32);
}
unsafe {
let __t0 = 1u32;
(*pdev).__slate_bits_3.__set_is_hotplug_bridge(__t0);
}
}
}

fn set_pcie_thunderbolt(dev: *mut pci_dev) {
let mut vsec: u16 = 0;
// Is the device part of a Thunderbolt controller?
vsec = unsafe { pci_find_vsec_capability(dev, 32902u16, 4660i32) };
if vsec != 0 {
unsafe {
(*dev).__slate_bits_3.__set_is_thunderbolt(1u32);
}
}
}

fn set_pcie_cxl(dev: *mut pci_dev) {
let mut dvsec: u16 = 0;
let mut cap: u16 = 0;
if !pci_is_pcie(dev) {
return;
}
// Update parent's CXL state because alternate protocol training
// may have changed
let bridge: *mut pci_dev = pci_upstream_bridge(dev);
if !bridge.is_null() {
set_pcie_cxl(bridge);
}
dvsec = unsafe { pci_find_dvsec_capability(dev, 7832u16, 7u16) };
if dvsec == 0 {
return;
}
unsafe { pci_read_config_word(dev as *const pci_dev, (dvsec as i32).wrapping_add(14), core::ptr::addr_of_mut!(cap)) };
let mut __v0: u64;
loop {
if __bf_cast_unsigned__3!(1u64) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u64) {
unsafe { __compiletime_assert_630() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_631() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 1u64 == 0 {
unsafe { __compiletime_assert_632() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !(1u64 >> (1u64.trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_633() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 1u64.wrapping_add(1u64 << (1u64.trailing_zeros() as i32)) & 1u64.wrapping_add(1u64 << (1u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_634() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (((cap as i32) as u64) & 1u64) >> (1u64.trailing_zeros() as i32);
__v0 = __v2;
let __v7: bool = if __v0 != 0 {
true
} else {
let mut __v8: u64;
loop {
if __bf_cast_unsigned__3!(4u64) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u64) {
unsafe { __compiletime_assert_635() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v10: u64;
loop {
if false {
unsafe { __compiletime_assert_636() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 4u64 == 0 {
unsafe { __compiletime_assert_637() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !(4u64 >> (4u64.trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_638() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 4u64.wrapping_add(1u64 << (4u64.trailing_zeros() as i32)) & 4u64.wrapping_add(1u64 << (4u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_639() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = (((cap as i32) as u64) & 4u64) >> (4u64.trailing_zeros() as i32);
__v8 = __v10;
__v8 != 0
};
unsafe {
(*dev).__slate_bits_3.__set_is_cxl(__v7 as u32);
}
}

fn set_pcie_untrusted(dev: *mut pci_dev) {
let parent: *mut pci_dev = pci_upstream_bridge(dev);
if parent.is_null() {
return;
}
// If the upstream bridge is untrusted we treat this device as
// untrusted as well.
if ((unsafe { (*parent).__slate_bits_3.__get_untrusted() }) as i32) != 0 {
unsafe {
(*dev).__slate_bits_3.__set_untrusted(1u32);
}
return;
}
if unsafe { arch_pci_dev_is_removable(dev) } {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"marking as untrusted\n".as_ptr() as *const u8) };
}
unsafe {
(*dev).__slate_bits_3.__set_untrusted(1u32);
}
}
}

fn pci_set_removable(dev: *mut pci_dev) {
let parent: *mut pci_dev = pci_upstream_bridge(dev);
if parent.is_null() {
return;
}
// We (only) consider everything tunneled below an external_facing
// device to be removable by the user. We're mainly concerned with
// consumer platforms with user accessible thunderbolt ports that are
// vulnerable to DMA attacks, and we expect those ports to be marked by
// the firmware as external_facing. Devices in traditional hotplug
// slots can technically be removed, but the expectation is that unless
// the port is marked with external_facing, such devices are less
// accessible to user / may not be removed by end user, and thus not
// exposed as "removable" to userspace.
if dev_is_removable(unsafe { core::ptr::addr_of_mut!((*parent).dev) }) {
dev_set_removable(unsafe { core::ptr::addr_of_mut!((*dev).dev) }, 3u32);
return;
}
if unsafe { arch_pci_dev_is_removable(dev) } {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"marking as removable\n".as_ptr() as *const u8) };
}
dev_set_removable(unsafe { core::ptr::addr_of_mut!((*dev).dev) }, 3u32);
}
}

/// pci_ext_cfg_is_aliased - Is ext config space just an alias of std config?
/// @dev: PCI device
///
/// PCI Express to PCI/PCI-X Bridge Specification, rev 1.0, 4.1.4 says that
/// when forwarding a type1 configuration request the bridge must check that
/// the extended register address field is zero.  The bridge is not permitted
/// to forward the transactions and must handle it as an Unsupported Request.
/// Some bridges do not follow this rule and simply drop the extended register
/// bits, resulting in the standard config space being aliased, every 256
/// bytes across the entire configuration space.  Test for this condition by
/// comparing the first dword of each potential alias to the vendor/device ID.
/// Known offenders:
///   ASM1083/1085 PCIe-to-PCI Reversible Bridge (1b21:1080, rev 01 & 03)
///   AMD/ATI SBx00 PCI to PCI Bridge (1002:4384, rev 40)
fn pci_ext_cfg_is_aliased(dev: *mut pci_dev) -> bool {
let mut pos: i32 = 256;
let mut ret: i32 = 0;
let mut header: u32 = 0;
let mut tmp: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, 0i32, core::ptr::addr_of_mut!(header)) };
while pos < 4096 {
ret = unsafe { pci_read_config_dword(dev as *const pci_dev, pos, core::ptr::addr_of_mut!(tmp)) };
if ret != 0 || header != tmp {
return false;
}
pos = pos.wrapping_add(256);
}
true
}

/// pci_cfg_space_size_ext - Get the configuration space size of the PCI device
/// @dev: PCI device
///
/// Regular PCI devices have 256 bytes, but PCI-X 2 and PCI Express devices
/// have 4096 bytes.  Even if the device is capable, that doesn't mean we can
/// access it.  Maybe we don't have a way to generate extended config space
/// accesses, or the device is behind a reverse Express bridge.  So we try
/// reading the dword at 0x100 which must either be 0 or a valid extended
/// capability header.
fn pci_cfg_space_size_ext(dev: *mut pci_dev) -> i32 {
let mut status: u32 = 0;
let pos: i32 = 256;
if (unsafe { pci_read_config_dword(dev as *const pci_dev, pos, core::ptr::addr_of_mut!(status)) }) != 0 {
return 256;
}
let __v0: bool = PCI_POSSIBLE_ERROR!(status) || pci_ext_cfg_is_aliased(dev);
if __v0 {
return 256;
}
4096
}

#[unsafe(no_mangle)]
extern "C" fn pci_cfg_space_size(dev: *mut pci_dev) -> i32 {
let mut pos: i32 = 0;
let mut status: u32 = 0;
let mut class: u16 = 0;
if ((unsafe { (*(*dev).bus).bus_flags }) as i32) & 8 != 0 {
return 256;
}
class = ((unsafe { (*dev).class }) >> 8i32) as u16;
if (class as i32) == 1536 {
return pci_cfg_space_size_ext(dev);
}
if pci_is_pcie(dev) {
return pci_cfg_space_size_ext(dev);
}
pos = (unsafe { pci_find_capability(dev, 7i32) }) as i32;
if pos == 0 {
return 256;
}
unsafe { pci_read_config_dword(dev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(status)) };
if status & (1073741824 | 2147483648) != 0 {
return pci_cfg_space_size_ext(dev);
}
256
}

fn pci_class(dev: *mut pci_dev) -> u32 {
let mut class: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, 8i32, core::ptr::addr_of_mut!(class)) };
class
}

fn pci_subsystem_ids(dev: *mut pci_dev, vendor: *mut u16, device: *mut u16) {
unsafe { pci_read_config_word(dev as *const pci_dev, 44i32, vendor) };
unsafe { pci_read_config_word(dev as *const pci_dev, 46i32, device) };
}

fn pci_hdr_type(dev: *mut pci_dev) -> u8 {
let mut hdr_type: u8 = 0;
unsafe { pci_read_config_byte(dev as *const pci_dev, 14i32, core::ptr::addr_of_mut!(hdr_type)) };
hdr_type
}

/// pci_intx_mask_broken - Test PCI_COMMAND_INTX_DISABLE writability
/// @dev: PCI device
///
/// Test whether PCI_COMMAND_INTX_DISABLE is writable for @dev.  Check this
/// at enumeration-time to avoid modifying PCI_COMMAND at run-time.
fn pci_intx_mask_broken(dev: *mut pci_dev) -> i32 {
let mut orig: u16 = 0;
let mut toggle: u16 = 0;
let mut new: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(orig)) };
toggle = ((orig as i32) ^ 1024) as u16;
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, toggle) };
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(new)) };
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, orig) };
// PCI_COMMAND_INTX_DISABLE was reserved and read-only prior to PCI
// r2.3, so strictly speaking, a device is not *broken* if it's not
// writable.  But we'll live with the misnomer for now.
if (new as i32) != (toggle as i32) {
return 1;
}
0
}

fn early_dump_pci_device(pdev: *mut pci_dev) {
let mut value: __SlateAlign16<[u32; 64]> = __SlateAlign16([0u32; 64]);
let mut i: i32 = 0;
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"config space:\n".as_ptr() as *const u8) };
while (i as u64) < ARRAY_SIZE__14!() {
unsafe { pci_read_config_dword(pdev as *const pci_dev, (i as u64).wrapping_mul(core::mem::size_of::<u32>() as u64) as i32, (value.0.as_mut_ptr() as *mut u32).wrapping_offset(i as isize)) };
i = i.wrapping_add(1);
}
unsafe { print_hex_dump(c"\x016".as_ptr() as *const u8, c"".as_ptr() as *const u8, 2i32, 16i32, 1i32, value.0.as_mut_ptr() as *const (), ARRAY_SIZE__11!().wrapping_mul(core::mem::size_of::<u32>() as u64), false) };
}

fn pci_type_str(dev: *mut pci_dev) -> *const u8 {
let mut r#type: i32 = 0;
if pci_is_pcie(dev) {
r#type = pci_pcie_type(dev as *const pci_dev);
if (r#type as u64) < ARRAY_SIZE__5!() {
return unsafe { str.0[r#type as usize] };
}
return c"PCIe unknown".as_ptr() as *const u8;
}
match (unsafe { (*dev).hdr_type }) as i32 {
0 => {
return c"conventional PCI endpoint".as_ptr() as *const u8;
}
1 => {
return c"conventional PCI bridge".as_ptr() as *const u8;
}
2 => {
return c"CardBus bridge".as_ptr() as *const u8;
}
_ => {
return c"conventional PCI".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

static str: __SlateAlign16<[*const u8; 11]> = __SlateAlign16([(b"PCIe Endpoint\0".as_ptr() as *mut u8) as *const u8, (b"PCIe Legacy Endpoint\0".as_ptr() as *mut u8) as *const u8, (b"PCIe unknown\0".as_ptr() as *mut u8) as *const u8, (b"PCIe unknown\0".as_ptr() as *mut u8) as *const u8, (b"PCIe Root Port\0".as_ptr() as *mut u8) as *const u8, (b"PCIe Switch Upstream Port\0".as_ptr() as *mut u8) as *const u8, (b"PCIe Switch Downstream Port\0".as_ptr() as *mut u8) as *const u8, (b"PCIe to PCI/PCI-X bridge\0".as_ptr() as *mut u8) as *const u8, (b"PCI/PCI-X to PCIe bridge\0".as_ptr() as *mut u8) as *const u8, (b"PCIe Root Complex Integrated Endpoint\0".as_ptr() as *mut u8) as *const u8, (b"PCIe Root Complex Event Collector\0".as_ptr() as *mut u8) as *const u8]);

// pci_setup_device - Fill in class and map information of a device
// @dev: the device structure to fill
//
// Initialize the device structure with information about the device's
// vendor,class,memory and IO-space addresses, IRQ lines etc.
// Called at initialisation of the PCI subsystem and by CardBus services.
// Returns 0 on success and negative if unknown type of device (not normal,
// bridge or CardBus).
fn pci_configure_mps(dev: *mut pci_dev) {
let bridge: *mut pci_dev = pci_upstream_bridge(dev);
let mut mps: i32 = 0;
let mut mpss: i32 = 0;
let mut p_mps: i32 = 0;
let mut rc: i32 = 0;
if !pci_is_pcie(dev) {
return;
}
// MPS and MRRS fields are of type 'RsvdP' for VFs, short-circuit out
if ((unsafe { (*dev).__slate_bits_3.__get_is_virtfn() }) as i32) != 0 {
return;
}
// For Root Complex Integrated Endpoints, program the maximum
// supported value unless limited by the PCIE_BUS_PEER2PEER case.
if pci_pcie_type(dev as *const pci_dev) == 9 {
if ((unsafe { pcie_bus_config }) as u32) == 4 {
mps = 128;
} else {
mps = 128i32 << ((unsafe { (*dev).__slate_bits_0.__get_pcie_mpss() }) as i32);
}
rc = unsafe { pcie_set_mps(dev, mps) };
if rc != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"can't set Max Payload Size to %d; if necessary, use \"pci=pcie_bus_safe\" and report a bug\n".as_ptr() as *const u8, mps) };
}
return;
}
let __v1: bool = bridge.is_null() || !pci_is_pcie(bridge);
if __v1 {
return;
}
mps = unsafe { pcie_get_mps(dev) };
p_mps = unsafe { pcie_get_mps(bridge) };
if mps == p_mps {
return;
}
if ((unsafe { pcie_bus_config }) as u32) == 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Max Payload Size %d, but upstream %s set to %d; if necessary, use \"pci=pcie_bus_safe\" and report a bug\n".as_ptr() as *const u8, mps, pci_name(bridge as *const pci_dev), p_mps) };
return;
}
// Fancier MPS configuration is done later by
// pcie_bus_configure_settings()
if ((unsafe { pcie_bus_config }) as u32) != 1 {
return;
}
mpss = 128i32 << ((unsafe { (*dev).__slate_bits_0.__get_pcie_mpss() }) as i32);
let __v3: bool = mpss < p_mps && pci_pcie_type(bridge as *const pci_dev) == 4;
if __v3 {
unsafe { pcie_set_mps(bridge, mpss) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Upstream bridge's Max Payload Size set to %d (was %d, max %d)\n".as_ptr() as *const u8, mpss, p_mps, 128i32 << ((*bridge).__slate_bits_0.__get_pcie_mpss() as i32)) };
p_mps = unsafe { pcie_get_mps(bridge) };
}
rc = unsafe { pcie_set_mps(dev, p_mps) };
if rc != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"can't set Max Payload Size to %d; if necessary, use \"pci=pcie_bus_safe\" and report a bug\n".as_ptr() as *const u8, p_mps) };
return;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Max Payload Size set to %d (was %d, max %d)\n".as_ptr() as *const u8, p_mps, mps, mpss) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_configure_extended_tags")]
extern "C" fn pci_configure_extended_tags(dev: *mut pci_dev, ign: *mut ()) -> i32 {
let mut cap: u32 = 0;
let mut ctl: u16 = 0;
let mut ret: i32 = 0;
// PCI_EXP_DEVCTL_EXT_TAG is RsvdP in VFs
if !pci_is_pcie(dev) || ((unsafe { (*dev).__slate_bits_3.__get_is_virtfn() }) as i32) != 0 {
return 0;
}
ret = unsafe { pcie_capability_read_dword(dev, 4i32, core::ptr::addr_of_mut!(cap)) };
if ret != 0 {
return 0;
}
if cap & 32 == 0 {
return 0;
}
ret = unsafe { pcie_capability_read_word(dev, 8i32, core::ptr::addr_of_mut!(ctl)) };
if ret != 0 {
return 0;
}
let host: *mut pci_host_bridge = unsafe { pci_find_host_bridge((*dev).bus) };
if host.is_null() {
return 0;
}
// If some device in the hierarchy doesn't handle Extended Tags
// correctly, make sure they're disabled.
if ((unsafe { (*host).__slate_bits_0.__get_no_ext_tags() }) as i32) != 0 {
if (ctl as i32) & 256 != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"disabling Extended Tags\n".as_ptr() as *const u8) };
pcie_capability_clear_word(dev, 8i32, 256u16);
}
return 0;
}
if (ctl as i32) & 256 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"enabling Extended Tags\n".as_ptr() as *const u8) };
pcie_capability_set_word(dev, 8i32, 256u16);
}
0
}

fn pci_dev3_init(pdev: *mut pci_dev) {
let cap: u16 = unsafe { pci_find_ext_capability(pdev, 47i32) };
let mut val: u32 = 0;
if cap == 0 {
return;
}
unsafe { pci_read_config_dword(pdev as *const pci_dev, (cap as i32).wrapping_add(12), core::ptr::addr_of_mut!(val)) };
unsafe {
let __t0 = (val & 8 != 0) as u32;
(*pdev).__slate_bits_3.__set_fm_enabled(__t0);
}
}

/// pcie_relaxed_ordering_enabled - Probe for PCIe relaxed ordering enable
/// @dev: PCI device to query
///
/// Returns true if the device has enabled relaxed ordering attribute.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pcie_relaxed_ordering_enabled")]
extern "C" fn pcie_relaxed_ordering_enabled(dev: *mut pci_dev) -> bool {
let mut v: u16 = 0;
unsafe { pcie_capability_read_word(dev, 8i32, core::ptr::addr_of_mut!(v)) };
(v as i32) & 16 != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_relaxed_ordering_enabled_650: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> bool>, *mut ()>(Some(pcie_relaxed_ordering_enabled)) };

fn pci_configure_relaxed_ordering(dev: *mut pci_dev) {
// PCI_EXP_DEVCTL_RELAX_EN is RsvdP in VFs
if ((unsafe { (*dev).__slate_bits_3.__get_is_virtfn() }) as i32) != 0 {
return;
}
if !pcie_relaxed_ordering_enabled(dev) {
return;
}
// For now, we only deal with Relaxed Ordering issues with Root
// Ports. Peer-to-Peer DMA is another can of worms.
let root: *mut pci_dev = pcie_find_root_port(dev);
if root.is_null() {
return;
}
if ((unsafe { (*root).dev_flags }) as i32) & 2048 != 0 {
pcie_capability_clear_word(dev, 8i32, 16u16);
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Relaxed Ordering disabled because the Root Port didn't support it\n".as_ptr() as *const u8) };
}
}

fn pci_configure_eetlp_prefix(dev: *mut pci_dev) {
let mut eetlp_max: u32 = 0;
let mut pcie_type: i32 = 0;
let mut cap: u32 = 0;
if !pci_is_pcie(dev) {
return;
}
unsafe { pcie_capability_read_dword(dev, 36i32, core::ptr::addr_of_mut!(cap)) };
if cap & 2097152 == 0 {
return;
}
pcie_type = pci_pcie_type(dev as *const pci_dev);
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(12582912i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_651() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_652() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_653() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(12582912i32 >> (12582912u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_654() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 12582912u64.wrapping_add(1u64 << (12582912u64.trailing_zeros() as i32)) & 12582912u64.wrapping_add(1u64 << (12582912u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_655() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((cap & 12582912) >> (12582912u64.trailing_zeros() as i32)) as i32;
__v0 = __v2;
eetlp_max = __v0 as u32;
// 00b means 4
let __v7: u32 = eetlp_max;
eetlp_max = if __v7 != 0 { __v7 } else { 4 };
if pcie_type == 4 || pcie_type == 9 {
unsafe {
(*dev).__slate_bits_2.__set_eetlp_prefix_max(eetlp_max);
}
} else {
let bridge: *mut pci_dev = pci_upstream_bridge(dev);
if !bridge.is_null() && ((unsafe { (*bridge).__slate_bits_2.__get_eetlp_prefix_max() }) as i32) != 0 {
unsafe {
(*dev).__slate_bits_2.__set_eetlp_prefix_max(eetlp_max);
}
}
}
}

fn pci_configure_serr(dev: *mut pci_dev) {
let mut control: u16 = 0;
if ((unsafe { (*dev).hdr_type }) as i32) == 1 {
// A bridge will not forward ERR_ messages coming from an
// endpoint unless SERR# forwarding is enabled.
unsafe { pci_read_config_word(dev as *const pci_dev, 62i32, core::ptr::addr_of_mut!(control)) };
if (control as i32) & 2 == 0 {
control = ((control as i32) | 2) as u16;
unsafe { pci_write_config_word(dev as *const pci_dev, 62i32, control) };
}
}
}

fn pci_configure_rcb(dev: *mut pci_dev) {
let mut rp_lnkctl: u16 = 0;
// Per PCIe r7.0, sec 7.5.3.7, RCB is only meaningful in Root Ports
// (where it is read-only), Endpoints, and Bridges.  It may only be
// set for Endpoints and Bridges if it is set in the Root Port. For
// Endpoints, it is 'RsvdP' for Virtual Functions.
let __v0: bool = !pci_is_pcie(dev) || pci_pcie_type(dev as *const pci_dev) == 4;
let __v1: bool = __v0 || pci_pcie_type(dev as *const pci_dev) == 5;
let __v2: bool = __v1 || pci_pcie_type(dev as *const pci_dev) == 6;
let __v3: bool = __v2 || pci_pcie_type(dev as *const pci_dev) == 10;
if __v3 || ((unsafe { (*dev).__slate_bits_3.__get_is_virtfn() }) as i32) != 0 {
return;
}
// Root Port often not visible to virtualized guests
let rp: *mut pci_dev = pcie_find_root_port(dev);
if rp.is_null() {
return;
}
unsafe { pcie_capability_read_word(rp, 16i32, core::ptr::addr_of_mut!(rp_lnkctl)) };
pcie_capability_clear_and_set_word(dev, 16i32, 8u16, (if (rp_lnkctl as i32) & 8 != 0 { 8i32 } else { 0i32 }) as u16);
}

fn pci_configure_device(dev: *mut pci_dev) {
pci_configure_mps(dev);
pci_configure_extended_tags(dev, core::ptr::null_mut::<()>());
pci_configure_relaxed_ordering(dev);
unsafe { pci_configure_ltr(dev) };
unsafe { pci_configure_aspm_l1ss(dev) };
pci_configure_eetlp_prefix(dev);
pci_configure_serr(dev);
pci_configure_rcb(dev);
unsafe { pci_acpi_program_hp_params(dev) };
}

fn pci_release_capabilities(dev: *mut pci_dev) {
pci_aer_exit(dev);
unsafe { pci_rcec_exit(dev) };
pci_iov_release(dev);
unsafe { pci_free_cap_save_buffers(dev) };
}

/// pci_release_dev - Free a PCI device structure when all users of it are
/// 		     finished
/// @dev: device that's been disconnected
///
/// Will be called only by the device core when all users of this PCI device are
/// done.
#[unsafe(link_section = ".text.slate_distinct.probe.pci_release_dev")]
extern "C" fn pci_release_dev(dev: *mut device) {
let pci_dev: *mut pci_dev = to_pci_dev!(dev);
pci_release_capabilities(pci_dev);
pci_release_of_node(pci_dev);
unsafe { pcibios_release_device(pci_dev) };
unsafe { pci_bus_put((*pci_dev).bus) };
unsafe { bitmap_free((*pci_dev).dma_alias_mask as *const u64) };
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, dev as *const device, c"device released\n".as_ptr() as *const u8) };
}
unsafe { kfree(pci_dev as *const ()) };
}

static pci_dev_type: __SlateConst<device_type> = __SlateConst({
let mut __t0: device_type = unsafe { core::mem::zeroed() };
__t0.groups = unsafe { core::ptr::addr_of_mut!(pci_dev_attr_groups) as *mut *const attribute_group };
__t0
});

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_alloc_dev")]
extern "C" fn pci_alloc_dev(bus: *mut pci_bus) -> *mut pci_dev {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: resource = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: raw_spinlock = unsafe { core::mem::zeroed() };
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
let mut __v0: *mut pci_dev;
let __obj_size: u64 = size_mul(core::mem::size_of::<pci_dev>() as u64, 1u64);
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
__v0 = __v1 as *mut pci_dev;
dev = __v0;
if dev.is_null() {
return core::ptr::null_mut::<pci_dev>();
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*dev).bus_list) });
unsafe {
(*dev).dev.r#type = core::ptr::addr_of!(pci_dev_type.0);
}
unsafe {
(*dev).bus = pci_bus_get(bus);
}
unsafe {
(*dev).driver_exclusive_resource = *{
__slate_compound_6 = {
let mut __t0: resource = core::mem::zeroed();
__t0.start = 0u64;
__t0.end = 1i32.wrapping_neg() as u64;
__t0.name = c"PCI Exclusive".as_ptr() as *const u8;
__t0
};
core::ptr::addr_of_mut!(__slate_compound_6)
};
}
'__slate_break_8: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*dev).pcie_cap_lock) });
unsafe {
*core::ptr::addr_of_mut!((*dev).pcie_cap_lock) = *{
__slate_compound_7 = spinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t1.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t2.val = __SlateRecord_types_h_190 { counter: 0 };
__t2
} } };
__t1
} };
core::ptr::addr_of_mut!(__slate_compound_7)
};
}
if true {
break;
}
}
'__slate_break_10: loop {
unsafe {
*core::ptr::addr_of_mut!((*dev).msi_lock) = *{
__slate_compound_9 = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t3: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t3.val = __SlateRecord_types_h_190 { counter: 0 };
__t3
} } };
core::ptr::addr_of_mut!(__slate_compound_9)
};
}
if true {
break;
}
}
dev
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_alloc_dev_656: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus) -> *mut pci_dev>, *mut ()>(Some(pci_alloc_dev)) };

fn pci_bus_wait_rrs(bus: *mut pci_bus, devfn: i32, l: *mut u32, timeout: i32) -> bool {
let mut delay: i32 = 1;
if !pci_bus_rrs_vendor_id(unsafe { *l }) {
return true;
}
// not a Configuration RRS completion
if timeout == 0 {
return false;
}
// RRS, but caller doesn't want to wait
// We got the reserved Vendor ID that indicates a completion with
// Configuration Request Retry Status (RRS).  Retry until we get a
// valid Vendor ID or we time out.
while pci_bus_rrs_vendor_id(unsafe { *l }) {
if delay > timeout {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014pci %04x:%02x:%02x.%d: not ready after %dms; giving up\n".as_ptr() as *const u8, pci_domain_nr(bus), (*bus).number as i32, PCI_SLOT__2!(devfn), PCI_FUNC__2!(devfn), delay.wrapping_sub(1)) };
return false;
}
if delay >= 1000 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016pci %04x:%02x:%02x.%d: not ready after %dms; waiting\n".as_ptr() as *const u8, pci_domain_nr(bus), (*bus).number as i32, PCI_SLOT__2!(devfn), PCI_FUNC__2!(devfn), delay.wrapping_sub(1)) };
}
unsafe { msleep(delay as u32) };
delay = delay.wrapping_mul(2);
if (unsafe { pci_bus_read_config_dword(bus, devfn as u32, 0i32, l) }) != 0 {
return false;
}
}
if delay >= 1000 {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x016pci %04x:%02x:%02x.%d: ready after %dms\n".as_ptr() as *const u8, pci_domain_nr(bus), (*bus).number as i32, PCI_SLOT__2!(devfn), PCI_FUNC__2!(devfn), delay.wrapping_sub(1)) };
}
true
}

#[unsafe(no_mangle)]
extern "C" fn pci_bus_generic_read_dev_vendor_id(bus: *mut pci_bus, devfn: i32, l: *mut u32, timeout: i32) -> bool {
if (unsafe { pci_bus_read_config_dword(bus, devfn as u32, 0i32, l) }) != 0 {
return false;
}
// Some broken boards return 0 or ~0 (PCI_ERROR_RESPONSE) if a slot is empty:
if PCI_POSSIBLE_ERROR!(unsafe { *l }) || (unsafe { *l }) == 0 || (unsafe { *l }) == 65535 || (unsafe { *l }) == 4294901760 {
return false;
}
if pci_bus_rrs_vendor_id(unsafe { *l }) {
return pci_bus_wait_rrs(bus, devfn, l, timeout);
}
true
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_bus_read_dev_vendor_id")]
extern "C" fn pci_bus_read_dev_vendor_id(bus: *mut pci_bus, devfn: i32, l: *mut u32, timeout: i32) -> bool {
pci_bus_generic_read_dev_vendor_id(bus, devfn, l, timeout)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_bus_read_dev_vendor_id_657: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus, i32, *mut u32, i32) -> bool>, *mut ()>(Some(pci_bus_read_dev_vendor_id)) };

/// Read the config data for a PCI device, sanity-check it,
/// and fill in the dev structure.
fn pci_scan_device(bus: *mut pci_bus, devfn: i32) -> *mut pci_dev {
let mut l: u32 = 0;
if !pci_bus_read_dev_vendor_id(bus, devfn, core::ptr::addr_of_mut!(l), 60000i32) {
return core::ptr::null_mut::<pci_dev>();
}
let dev: *mut pci_dev = pci_alloc_dev(bus);
if dev.is_null() {
return core::ptr::null_mut::<pci_dev>();
}
unsafe {
(*dev).devfn = devfn as u32;
}
unsafe {
(*dev).vendor = (l & 65535) as u16;
}
unsafe {
(*dev).device = (l >> 16i32 & 65535) as u16;
}
if pci_setup_device(dev) != 0 {
unsafe { pci_bus_put((*dev).bus) };
unsafe { kfree(dev as *const ()) };
return core::ptr::null_mut::<pci_dev>();
}
dev
}

#[unsafe(no_mangle)]
extern "C" fn pcie_report_downtraining(dev: *mut pci_dev) {
if !pci_is_pcie(dev) {
return;
}
// Look from the device up to avoid downstream ports with no devices
let __v0: bool = pci_pcie_type(dev as *const pci_dev) != 0 && pci_pcie_type(dev as *const pci_dev) != 1;
let __v1: bool = __v0 && pci_pcie_type(dev as *const pci_dev) != 5;
if __v1 {
return;
}
// Multi-function PCIe devices share the same link/status
if PCI_FUNC!(unsafe { (*dev).devfn }) != 0 || ((unsafe { (*dev).__slate_bits_3.__get_is_virtfn() }) as i32) != 0 {
return;
}
// Print link status only if the device is constrained by the fabric
unsafe { __pcie_print_link_status(dev, false) };
}

fn pci_imm_ready_init(dev: *mut pci_dev) {
let mut status: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, 6i32, core::ptr::addr_of_mut!(status)) };
if (status as i32) & 1 != 0 {
unsafe {
(*dev).__slate_bits_1.__set_imm_ready(1u32);
}
}
}

fn pci_init_capabilities(dev: *mut pci_dev) {
unsafe { pci_ea_init(dev) }; // Enhanced Allocation
unsafe { pci_msi_init(dev) }; // Disable MSI
unsafe { pci_msix_init(dev) }; // Disable MSI-X
// Buffers for saving PCIe and PCI-X capabilities
unsafe { pci_allocate_cap_save_buffers(dev) };
pci_imm_ready_init(dev); // Immediate Readiness
unsafe { pci_pm_init(dev) }; // Power Management
unsafe { pci_vpd_init(dev) }; // Vital Product Data
unsafe { pci_configure_ari(dev) }; // Alternative Routing-ID Forwarding
pci_iov_init(dev); // Single Root I/O Virtualization
unsafe { pci_ats_init(dev) }; // Address Translation Services
unsafe { pci_pri_init(dev) }; // Page Request Interface
unsafe { pci_pasid_init(dev) }; // Process Address Space ID
unsafe { pci_acs_init(dev) }; // Access Control Services
pci_ptm_init(dev); // Precision Time Measurement
pci_aer_init(dev); // Advanced Error Reporting
pci_dpc_init(dev); // Downstream Port Containment
unsafe { pci_rcec_init(dev) }; // Root Complex Event Collector
pci_doe_init(dev); // Data Object Exchange
pci_tph_init(dev); // TLP Processing Hints
unsafe { pci_rebar_init(dev) }; // Resizable BAR
pci_dev3_init(dev); // Device 3 capabilities
pci_ide_init(dev); // Link Integrity and Data Encryption
pcie_report_downtraining(dev);
unsafe { pci_init_reset_methods(dev) };
}

/// This is the equivalent of pci_host_bridge_msi_domain() that acts on
/// devices. Firmware interfaces that can select the MSI domain on a
/// per-device basis should be called from here.
fn pci_dev_msi_domain(dev: *mut pci_dev) -> *mut irq_domain {
let mut d: *mut irq_domain = unsafe { core::mem::zeroed() };
// If a domain has been set through the pcibios_device_add()
// callback, then this is the one (platform code knows best).
d = dev_get_msi_domain((unsafe { core::ptr::addr_of_mut!((*dev).dev) }) as *const device);
if !d.is_null() {
return d;
}
// Let's see if we have a firmware interface able to provide
// the domain.
d = unsafe { pci_msi_get_device_domain(dev) };
if !d.is_null() {
return d;
}
core::ptr::null_mut::<irq_domain>()
}

fn pci_set_msi_domain(dev: *mut pci_dev) {
let mut d: *mut irq_domain = unsafe { core::mem::zeroed() };
// If the platform or firmware interfaces cannot supply a
// device-specific MSI domain, then inherit the default domain
// from the host bridge itself.
d = pci_dev_msi_domain(dev);
if d.is_null() {
d = dev_get_msi_domain((unsafe { core::ptr::addr_of_mut!((*(*dev).bus).dev) }) as *const device);
}
dev_set_msi_domain(unsafe { core::ptr::addr_of_mut!((*dev).dev) }, d);
}

#[unsafe(no_mangle)]
extern "C" fn pci_device_add(dev: *mut pci_dev, bus: *mut pci_bus) {
let mut ret: i32 = 0;
pci_configure_device(dev);
unsafe { device_initialize(core::ptr::addr_of_mut!((*dev).dev)) };
unsafe {
(*dev).dev.release = Some(pci_release_dev);
}
set_dev_node(unsafe { core::ptr::addr_of_mut!((*dev).dev) }, pcibus_to_node!(bus));
unsafe {
(*dev).dev.dma_mask = core::ptr::addr_of_mut!((*dev).dma_mask);
}
unsafe {
(*dev).dev.dma_parms = core::ptr::addr_of_mut!((*dev).dma_parms);
}
unsafe {
(*dev).dev.coherent_dma_mask = 4294967295;
}
dma_set_max_seg_size(unsafe { core::ptr::addr_of_mut!((*dev).dev) }, 65536u32);
dma_set_seg_boundary(unsafe { core::ptr::addr_of_mut!((*dev).dev) }, 4294967295u64);
unsafe { pcie_failed_link_retrain(dev) };
// Fix up broken headers
unsafe { pci_fixup_device(1u32, dev) };
unsafe { pci_reassigndev_resource_alignment(dev) };
pci_init_capabilities(dev);
unsafe { platform_pci_configure_wake(dev) };
// Add the device to our list of discovered devices
// and the bus list for fixup functions, etc.
unsafe { down_write(core::ptr::addr_of_mut!(pci_bus_sem)) };
list_add_tail(unsafe { core::ptr::addr_of_mut!((*dev).bus_list) }, unsafe { core::ptr::addr_of_mut!((*bus).devices) });
unsafe { up_write(core::ptr::addr_of_mut!(pci_bus_sem)) };
ret = unsafe { pcibios_device_add(dev) };
let mut __v0: i64;
{
let __ret_warn_on: i32 = (ret < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("658: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 658b - ., 3; .popsection\n# {0}", const 658 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_device_add_1, sym __slate_asm_string_pci_device_add_2, const 2763 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("659: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 659b - ., 4; .popsection\n# {0}", const 659 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
// Set up MSI IRQ domain
pci_set_msi_domain(dev);
// Notifier could use PCI capabilities
ret = unsafe { device_add(core::ptr::addr_of_mut!((*dev).dev)) };
let mut __v5: i64;
let __ret_warn_on: i32 = (ret < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("660: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 660b - ., 3; .popsection\n# {0}", const 660 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_device_add_6, sym __slate_asm_string_pci_device_add_7, const 2770 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("661: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 661b - ., 4; .popsection\n# {0}", const 661 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
// Establish pdev->tsm for newly added (e.g. new SR-IOV VFs)
pci_tsm_init(dev);
pci_npem_create(dev);
pci_doe_sysfs_init(dev);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_scan_single_device")]
extern "C" fn pci_scan_single_device(bus: *mut pci_bus, devfn: i32) -> *mut pci_dev {
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
dev = unsafe { pci_get_slot(bus, devfn as u32) };
if !dev.is_null() {
unsafe { pci_dev_put(dev) };
return dev;
}
dev = pci_scan_device(bus, devfn);
if dev.is_null() {
return core::ptr::null_mut::<pci_dev>();
}
pci_device_add(dev, bus);
dev
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_scan_single_device_662: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus, i32) -> *mut pci_dev>, *mut ()>(Some(pci_scan_single_device)) };

fn next_ari_fn(bus: *mut pci_bus, dev: *mut pci_dev, r#fn: i32) -> i32 {
let mut pos: i32 = 0;
let mut cap: u16 = 0;
let mut next_fn: u32 = 0;
if dev.is_null() {
return 19i32.wrapping_neg();
}
pos = (unsafe { pci_find_ext_capability(dev, 14i32) }) as i32;
if pos == 0 {
return 19i32.wrapping_neg();
}
unsafe { pci_read_config_word(dev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(cap)) };
next_fn = PCI_ARI_CAP_NFN!(cap) as u32;
if next_fn <= (r#fn as u32) {
return 19i32.wrapping_neg();
}
// protect against malformed list
next_fn as i32
}

fn next_fn(bus: *mut pci_bus, dev: *mut pci_dev, r#fn: i32) -> i32 {
if pci_ari_enabled(bus) {
return next_ari_fn(bus, dev, r#fn);
}
if r#fn >= 7 {
return 19i32.wrapping_neg();
}
// only multifunction devices may have more functions
if !dev.is_null() && ((unsafe { (*dev).__slate_bits_3.__get_multifunction() }) as i32) == 0 {
return 19i32.wrapping_neg();
}
r#fn.wrapping_add(1)
}

fn only_one_child(bus: *mut pci_bus) -> i32 {
let bridge: *mut pci_dev = unsafe { (*bus).self_ };
// Systems with unusual topologies set PCI_SCAN_ALL_PCIE_DEVS so
// we scan for all possible devices, not just Device 0.
if pci_has_flag(64i32) != 0 {
return 0;
}
// A PCIe Downstream Port normally leads to a Link with only Device
// 0 on it (PCIe spec r3.1, sec 7.3.1).  As an optimization, scan
// only for Device 0 in that situation.
let __v0: bool = !bridge.is_null() && pci_is_pcie(bridge);
let __v1: bool = __v0 && pcie_downstream_port(bridge as *const pci_dev);
if __v1 {
return 1;
}
0
}

/// pci_scan_slot - Scan a PCI slot on a bus for devices
/// @bus: PCI bus to scan
/// @devfn: slot number to scan (must have zero function)
///
/// Scan a PCI slot on the specified PCI bus for devices, adding
/// discovered devices to the @bus->devices list.  New devices
/// will not have is_added set.
///
/// Returns the number of new devices found.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_scan_slot")]
extern "C" fn pci_scan_slot(bus: *mut pci_bus, devfn: i32) -> i32 {
let mut r#fn: i32 = 0;
let mut nr: i32 = 0;
if only_one_child(bus) != 0 && devfn > 0 {
return 0;
}
// Already scanned the entire slot
loop {
let dev: *mut pci_dev = pci_scan_single_device(bus, devfn.wrapping_add(r#fn));
if !dev.is_null() {
if !pci_dev_is_added(dev as *const pci_dev) {
nr = nr.wrapping_add(1);
}
if r#fn > 0 {
unsafe {
(*dev).__slate_bits_3.__set_multifunction(1u32);
}
}
} else {
if r#fn == 0 {
// Function 0 is required unless we are running on
// a hypervisor that passes through individual PCI
// functions.
if !hypervisor_isolated_pci_functions() {
break;
}
}
}
r#fn = next_fn(bus, dev, r#fn);
if r#fn < 0 {
break;
}
}
// Only one slot has PCIe device
if !unsafe { (*bus).self_ }.is_null() && nr != 0 {
unsafe { pcie_aspm_init_link_state((*bus).self_) };
}
nr
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_scan_slot_663: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus, i32) -> i32>, *mut ()>(Some(pci_scan_slot)) };

#[unsafe(link_section = ".text.slate_distinct.probe.pcie_find_smpss")]
extern "C" fn pcie_find_smpss(dev: *mut pci_dev, data: *mut ()) -> i32 {
if !pci_is_pcie(dev) {
return 0;
}
// We don't have a way to change MPS settings on devices that have
// drivers attached.  A hot-added device might support only the minimum
// MPS setting (MPS=128).  Therefore, if the fabric contains a bridge
// where devices may be hot-added, we limit the fabric MPS to 128 so
// hot-added devices will work correctly.
//
// However, if we hot-add a device to a slot directly below a Root
// Port, it's impossible for there to be other existing devices below
// the port.  We don't limit the MPS in this case because we can
// reconfigure MPS on both the Root Port and the hot-added device,
// and there are no other devices involved.
//
// Note that this PCIE_BUS_SAFE path assumes no peer-to-peer DMA.
let __v0: bool = ((unsafe { (*dev).__slate_bits_3.__get_is_hotplug_bridge() }) as i32) != 0 && pci_pcie_type(dev as *const pci_dev) != 4;
if __v0 {
unsafe {
*(data as *mut u8) = 0;
}
}
if ((unsafe { *(data as *mut u8) }) as i32) > ((unsafe { (*dev).__slate_bits_0.__get_pcie_mpss() }) as i32) {
unsafe {
*(data as *mut u8) = ((*dev).__slate_bits_0.__get_pcie_mpss() as i32) as u8;
}
}
0
}

fn pcie_write_mps(dev: *mut pci_dev, mut mps: i32) {
let mut rc: i32 = 0;
if ((unsafe { pcie_bus_config }) as u32) == 3 {
mps = 128i32 << ((unsafe { (*dev).__slate_bits_0.__get_pcie_mpss() }) as i32);
if pci_pcie_type(dev as *const pci_dev) != 4 && !unsafe { (*(*dev).bus).self_ }.is_null() {
let mut __v0: i32;
let __UNIQUE_ID_x__664: i32 = mps;
let __UNIQUE_ID_y__665: i32 = unsafe { pcie_get_mps((*(*dev).bus).self_) };
loop {
if __types_ok__3!(__UNIQUE_ID_x__664, __UNIQUE_ID_y__665) == 0 {
unsafe { __compiletime_assert_666() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__664 < __UNIQUE_ID_y__665 { __UNIQUE_ID_x__664 } else { __UNIQUE_ID_y__665 };
mps = __v0;
}
// For "Performance", the assumption is made that
// downstream communication will never be larger than
// the MRRS.  So, the MPS only needs to be configured
// for the upstream communication.  This being the case,
// walk from the top down and set the MPS of the child
// to that of the parent bus.
//
// Configure the device MPS with the smaller of the
// device MPSS or the bridge MPS (which is assumed to be
// properly configured at this point to the largest
// allowable MPS based on its parent bus).
}
rc = unsafe { pcie_set_mps(dev, mps) };
if rc != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Failed attempting to set the MPS\n".as_ptr() as *const u8) };
}
}

fn pcie_write_mrrs(dev: *mut pci_dev) {
let mut rc: i32 = 0;
let mut mrrs: i32 = 0;
// In the "safe" case, do not configure the MRRS.  There appear to be
// issues with setting MRRS to 0 on a number of devices.
if ((unsafe { pcie_bus_config }) as u32) != 3 {
return;
}
// For max performance, the MRRS must be set to the largest supported
// value.  However, it cannot be configured larger than the MPS the
// device or the bus can support.  This should already be properly
// configured by a prior call to pcie_write_mps().
mrrs = unsafe { pcie_get_mps(dev) };
// MRRS is a R/W register.  Invalid values can be written, but a
// subsequent read will verify if the value is acceptable or not.
// If the MRRS value provided is not acceptable (e.g., too large),
// shrink the value until it is acceptable to the HW.
while mrrs != unsafe { pcie_get_readrq(dev) } && mrrs >= 128 {
rc = unsafe { pcie_set_readrq(dev, mrrs) };
if rc == 0 {
break;
}
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Failed attempting to set the MRRS\n".as_ptr() as *const u8) };
mrrs /= 2;
}
if mrrs < 128 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"MRRS was unable to be configured with a safe value.  If problems are experienced, try running with pci=pcie_bus_safe\n".as_ptr() as *const u8) };
}
}

#[unsafe(link_section = ".text.slate_distinct.probe.pcie_bus_configure_set")]
extern "C" fn pcie_bus_configure_set(dev: *mut pci_dev, data: *mut ()) -> i32 {
let mut mps: i32 = 0;
let mut orig_mps: i32 = 0;
if !pci_is_pcie(dev) {
return 0;
}
if ((unsafe { pcie_bus_config }) as u32) == 0 || ((unsafe { pcie_bus_config }) as u32) == 1 {
return 0;
}
mps = 128i32 << ((unsafe { *(data as *mut u8) }) as i32);
orig_mps = unsafe { pcie_get_mps(dev) };
pcie_write_mps(dev, mps);
pcie_write_mrrs(dev);
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Max Payload Size set to %4d/%4d (was %4d), Max Read Rq %4d\n".as_ptr() as *const u8, pcie_get_mps(dev), 128i32 << ((*dev).__slate_bits_0.__get_pcie_mpss() as i32), orig_mps, pcie_get_readrq(dev)) };
0
}

/// pcie_bus_configure_settings() requires that pci_walk_bus work in a top-down,
/// parents then children fashion.  If this changes, then this code will not
/// work as designed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pcie_bus_configure_settings")]
extern "C" fn pcie_bus_configure_settings(bus: *mut pci_bus) {
let mut smpss: u8 = 0;
if unsafe { (*bus).self_ }.is_null() {
return;
}
if !pci_is_pcie(unsafe { (*bus).self_ }) {
return;
}
// FIXME - Peer to peer DMA is possible, though the endpoint would need
// to be aware of the MPS of the destination.  To work around this,
// simply force the MPS of the entire system to the smallest possible.
if ((unsafe { pcie_bus_config }) as u32) == 4 {
smpss = 0;
}
if ((unsafe { pcie_bus_config }) as u32) == 2 {
smpss = ((unsafe { (*(*bus).self_).__slate_bits_0.__get_pcie_mpss() }) as i32) as u8;
pcie_find_smpss(unsafe { (*bus).self_ }, core::ptr::addr_of_mut!(smpss) as *mut ());
unsafe { pci_walk_bus(bus, Some(pcie_find_smpss), core::ptr::addr_of_mut!(smpss) as *mut ()) };
}
pcie_bus_configure_set(unsafe { (*bus).self_ }, core::ptr::addr_of_mut!(smpss) as *mut ());
unsafe { pci_walk_bus(bus, Some(pcie_bus_configure_set), core::ptr::addr_of_mut!(smpss) as *mut ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_bus_configure_settings_667: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus)>, *mut ()>(Some(pcie_bus_configure_settings)) };

/// Called after each bus is probed, but before its children are examined.  This
/// is marked as __weak because multiple architectures define it.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_fixup_bus(bus: *mut pci_bus) {
// nothing to do, expected to be removed in the future
}

// pci_scan_child_bus_extend() - Scan devices below a bus
// @bus: Bus to scan for devices
// @available_buses: Total number of buses available (%0 does not try to
// 		     extend beyond the minimal)
//
// Scans devices below @bus including subordinate buses. Returns new
// subordinate number including all the found devices. Passing
// @available_buses causes the remaining bus space to be distributed
// equally between hotplug-capable bridges to allow future extension of the
// hierarchy.
/// pci_scan_child_bus() - Scan devices below a bus
/// @bus: Bus to scan for devices
///
/// Scans devices below @bus including subordinate buses. Returns new
/// subordinate number including all the found devices.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_scan_child_bus")]
extern "C" fn pci_scan_child_bus(bus: *mut pci_bus) -> u32 {
pci_scan_child_bus_extend(bus, 0u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_scan_child_bus_674: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus) -> u32>, *mut ()>(Some(pci_scan_child_bus)) };

/// pcibios_root_bridge_prepare - Platform-specific host bridge setup
/// @bridge: Host bridge to set up
///
/// Default empty implementation.  Replace with an architecture-specific setup
/// routine, if necessary.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_root_bridge_prepare(bridge: *mut pci_host_bridge) -> i32 {
0
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_add_bus(bus: *mut pci_bus) {
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_remove_bus(bus: *mut pci_bus) {
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_create_root_bus")]
extern "C" fn pci_create_root_bus(parent: *mut device, bus: i32, ops: *mut pci_ops, sysdata: *mut (), resources: *mut list_head) -> *mut pci_bus {
let mut bridge: core::mem::MaybeUninit<*mut pci_host_bridge> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut pci_host_bridge = core::ptr::addr_of_mut!(bridge) as *mut *mut pci_host_bridge;
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = pci_alloc_host_bridge(0u64);
if (*__slate_slot_0).is_null() {
return core::ptr::null_mut::<pci_bus>();
} else {
(*(*__slate_slot_0)).dev.parent = parent;
list_splice_init(resources, core::ptr::addr_of_mut!((*(*__slate_slot_0)).windows));
(*(*__slate_slot_0)).sysdata = sysdata;
(*(*__slate_slot_0)).busnr = bus;
(*(*__slate_slot_0)).ops = ops;
*__slate_slot_1 = pci_register_host_bridge(*__slate_slot_0);
if *__slate_slot_1 < 0 {
put_device(core::ptr::addr_of_mut!((*(*__slate_slot_0)).dev));
return core::ptr::null_mut::<pci_bus>();
} else {
return (*(*__slate_slot_0)).bus;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_create_root_bus_675: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, i32, *mut pci_ops, *mut (), *mut list_head) -> *mut pci_bus>, *mut ()>(Some(pci_create_root_bus)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_host_probe")]
extern "C" fn pci_host_probe(bridge: *mut pci_host_bridge) -> i32 {
let mut bus: *mut pci_bus = unsafe { core::mem::zeroed() };
let mut child: *mut pci_bus = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
pci_lock_rescan_remove();
ret = pci_scan_root_bus_bridge(bridge);
pci_unlock_rescan_remove();
if ret < 0 {
while false {
}
unsafe { _dev_err((*bridge).dev.parent as *const device, c"Scanning root bridge failed".as_ptr() as *const u8) };
return ret;
}
bus = unsafe { (*bridge).bus };
// If we must preserve the resource configuration, claim now
if ((unsafe { (*bridge).__slate_bits_0.__get_preserve_config() }) as i32) != 0 {
unsafe { pci_bus_claim_resources(bus) };
}
// Assign whatever was left unassigned. If we didn't claim above,
// this will reassign everything.
unsafe { pci_assign_unassigned_root_bus_resources(bus) };
child = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*bus).children) });
while list_entry_is_head!(child, unsafe { core::ptr::addr_of_mut!((*bus).children) }, node) == 0 {
pcie_bus_configure_settings(child);
child = list_next_entry__5!(child, node);
}
pci_lock_rescan_remove();
unsafe { pci_bus_add_devices(bus as *const pci_bus) };
pci_unlock_rescan_remove();
// Ensure pm_runtime_enable() is called for the controller drivers
// before calling pci_host_probe(). The PM framework expects that
// if the parent device supports runtime PM, it will be enabled
// before child runtime PM is enabled.
pm_runtime_set_active(unsafe { core::ptr::addr_of_mut!((*bridge).dev) });
unsafe { pm_runtime_no_callbacks(core::ptr::addr_of_mut!((*bridge).dev)) };
unsafe { devm_pm_runtime_enable(core::ptr::addr_of_mut!((*bridge).dev)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_host_probe_676: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_host_bridge) -> i32>, *mut ()>(Some(pci_host_probe)) };

#[unsafe(no_mangle)]
extern "C" fn pci_bus_insert_busn_res(b: *mut pci_bus, bus: i32, bus_max: i32) -> i32 {
let res: *mut resource = unsafe { core::ptr::addr_of_mut!((*b).busn_res) };
let mut parent_res: *mut resource = unsafe { core::mem::zeroed() };
unsafe {
(*res).start = bus as u64;
}
unsafe {
(*res).end = bus_max as u64;
}
unsafe {
(*res).flags = 4096;
}
if !pci_is_root_bus(b) {
parent_res = unsafe { core::ptr::addr_of_mut!((*(*b).parent).busn_res) };
} else {
parent_res = get_pci_domain_busn_res(pci_domain_nr(b));
unsafe {
(*res).flags = (*res).flags | 16u64;
}
}
let conflict: *mut resource = unsafe { request_resource_conflict(parent_res, res) };
if !conflict.is_null() {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*b).dev) as *const device, c"busn_res: can not insert %pR under %s%pR (conflicts with %s %pR)\n".as_ptr() as *const u8, res, if pci_is_root_bus(b) { c"domain ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, parent_res, (*conflict).name, conflict) };
}
conflict.is_null() as i32
}

#[unsafe(no_mangle)]
extern "C" fn pci_bus_update_busn_res_end(b: *mut pci_bus, bus_max: i32) -> i32 {
let res: *mut resource = unsafe { core::ptr::addr_of_mut!((*b).busn_res) };
let mut old_res: resource = unsafe { *res };
let mut size: u64 = 0;
let mut ret: i32 = 0;
if (unsafe { (*res).start }) > (bus_max as u64) {
return 22i32.wrapping_neg();
}
size = (bus_max as u64).wrapping_sub(unsafe { (*res).start }).wrapping_add(1);
ret = unsafe { adjust_resource(res, (*res).start, size) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*b).dev) as *const device, c"busn_res: %pR end %s updated to %02x\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(old_res), if ret != 0 { c"can not be".as_ptr() as *mut u8 } else { c"is".as_ptr() as *mut u8 }, bus_max) };
if ret == 0 && unsafe { (*res).parent }.is_null() {
pci_bus_insert_busn_res(b, (unsafe { (*res).start }) as i32, (unsafe { (*res).end }) as i32);
}
ret
}

#[unsafe(no_mangle)]
extern "C" fn pci_bus_release_busn_res(b: *mut pci_bus) {
let res: *mut resource = unsafe { core::ptr::addr_of_mut!((*b).busn_res) };
let mut ret: i32 = 0;
if (unsafe { (*res).flags }) == 0 || unsafe { (*res).parent }.is_null() {
return;
}
ret = unsafe { release_resource(res) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*b).dev) as *const device, c"busn_res: %pR %s released\n".as_ptr() as *const u8, res, if ret != 0 { c"can not be".as_ptr() as *mut u8 } else { c"is".as_ptr() as *mut u8 }) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_scan_root_bus_bridge")]
extern "C" fn pci_scan_root_bus_bridge(bridge: *mut pci_host_bridge) -> i32 {
let mut window: *mut resource_entry = unsafe { core::mem::zeroed() };
let mut found: bool = false;
let mut b: *mut pci_bus = unsafe { core::mem::zeroed() };
let mut max: i32 = 0;
let mut bus: i32 = 0;
let mut ret: i32 = 0;
if bridge.is_null() {
return 22i32.wrapping_neg();
}
window = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*bridge).windows) });
while list_entry_is_head!(window, unsafe { core::ptr::addr_of_mut!((*bridge).windows) }, node) == 0 {
if (unsafe { (*(*window).res).flags }) & 4096 != 0 {
unsafe {
(*bridge).busnr = (*(*window).res).start as i32;
}
found = true;
break;
}
window = list_next_entry__3!(window, node);
}
ret = pci_register_host_bridge(bridge);
if ret < 0 {
return ret;
}
b = unsafe { (*bridge).bus };
bus = unsafe { (*bridge).busnr };
if !found {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*b).dev) as *const device, c"No busn resource found for root bus, will use [bus %02x-ff]\n".as_ptr() as *const u8, bus) };
pci_bus_insert_busn_res(b, bus, 255i32);
}
max = pci_scan_child_bus(b) as i32;
if !found {
pci_bus_update_busn_res_end(b, max);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_scan_root_bus_bridge_677: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_host_bridge) -> i32>, *mut ()>(Some(pci_scan_root_bus_bridge)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_scan_root_bus")]
extern "C" fn pci_scan_root_bus(parent: *mut device, bus: i32, ops: *mut pci_ops, sysdata: *mut (), resources: *mut list_head) -> *mut pci_bus {
let mut window: *mut resource_entry = unsafe { core::mem::zeroed() };
let mut found: bool = false;
let mut b: *mut pci_bus = unsafe { core::mem::zeroed() };
let mut max: i32 = 0;
window = list_first_entry__3!(resources);
while list_entry_is_head!(window, resources, node) == 0 {
if (unsafe { (*(*window).res).flags }) & 4096 != 0 {
found = true;
break;
}
window = list_next_entry__3!(window, node);
}
b = pci_create_root_bus(parent, bus, ops, sysdata, resources);
if b.is_null() {
return core::ptr::null_mut::<pci_bus>();
}
if !found {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*b).dev) as *const device, c"No busn resource found for root bus, will use [bus %02x-ff]\n".as_ptr() as *const u8, bus) };
pci_bus_insert_busn_res(b, bus, 255i32);
}
max = pci_scan_child_bus(b) as i32;
if !found {
pci_bus_update_busn_res_end(b, max);
}
b
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_scan_root_bus_678: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, i32, *mut pci_ops, *mut (), *mut list_head) -> *mut pci_bus>, *mut ()>(Some(pci_scan_root_bus)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_scan_bus")]
extern "C" fn pci_scan_bus(bus: i32, ops: *mut pci_ops, sysdata: *mut ()) -> *mut pci_bus {
let mut resources: list_head = unsafe { core::mem::zeroed() };
resources = list_head { next: core::ptr::addr_of_mut!(resources), prev: core::ptr::addr_of_mut!(resources) };
unsafe { pci_add_resource(core::ptr::addr_of_mut!(resources), core::ptr::addr_of_mut!(ioport_resource)) };
unsafe { pci_add_resource(core::ptr::addr_of_mut!(resources), core::ptr::addr_of_mut!(iomem_resource)) };
unsafe { pci_add_resource(core::ptr::addr_of_mut!(resources), core::ptr::addr_of_mut!(busn_resource)) };
let b: *mut pci_bus = pci_create_root_bus(core::ptr::null_mut::<device>(), bus, ops, sysdata, core::ptr::addr_of_mut!(resources));
if !b.is_null() {
pci_scan_child_bus(b);
} else {
unsafe { pci_free_resource_list(core::ptr::addr_of_mut!(resources)) };
}
b
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_scan_bus_679: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut pci_ops, *mut ()) -> *mut pci_bus>, *mut ()>(Some(pci_scan_bus)) };

/// pci_rescan_bus_bridge_resize - Scan a PCI bus for devices
/// @bridge: PCI bridge for the bus to scan
///
/// Scan a PCI bus and child buses for new devices, add them,
/// and enable them, resizing bridge mmio/io resource if necessary
/// and possible.  The caller must ensure the child devices are already
/// removed for resizing to occur.
///
/// Returns the max number of subordinate bus discovered.
#[unsafe(no_mangle)]
extern "C" fn pci_rescan_bus_bridge_resize(bridge: *mut pci_dev) -> u32 {
let mut max: u32 = 0;
let bus: *mut pci_bus = unsafe { (*bridge).subordinate };
max = pci_scan_child_bus(bus);
unsafe { pci_assign_unassigned_bridge_resources(bridge) };
unsafe { pci_bus_add_devices(bus as *const pci_bus) };
max
}

/// pci_rescan_bus - Scan a PCI bus for devices
/// @bus: PCI bus to scan
///
/// Scan a PCI bus and child buses for new devices, add them,
/// and enable them.
///
/// Returns the max number of subordinate bus discovered.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_rescan_bus")]
extern "C" fn pci_rescan_bus(bus: *mut pci_bus) -> u32 {
let max: u32 = pci_scan_child_bus(bus);
unsafe { pci_assign_unassigned_bus_resources(bus) };
unsafe { pci_bus_add_devices(bus as *const pci_bus) };
max
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_rescan_bus_680: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus) -> u32>, *mut ()>(Some(pci_rescan_bus)) };

/// pci_rescan_bus(), pci_rescan_bus_bridge_resize() and PCI device removal
/// routines should always be executed under this mutex.
#[unsafe(no_mangle)]
static mut pci_rescan_remove_lock: mutex = {
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

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_lock_rescan_remove")]
extern "C" fn pci_lock_rescan_remove() {
unsafe { mutex_lock(core::ptr::addr_of_mut!(pci_rescan_remove_lock)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_lock_rescan_remove_681: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn()>, *mut ()>(Some(pci_lock_rescan_remove)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_unlock_rescan_remove")]
extern "C" fn pci_unlock_rescan_remove() {
unsafe { mutex_unlock(core::ptr::addr_of_mut!(pci_rescan_remove_lock)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_unlock_rescan_remove_682: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn()>, *mut ()>(Some(pci_unlock_rescan_remove)) };

#[unsafe(link_section = ".init.text")]
extern "C" fn pci_sort_bf_cmp(d_a: *const device, d_b: *const device) -> i32 {
let a: *const pci_dev = to_pci_dev!(d_a) as *const pci_dev;
let b: *const pci_dev = to_pci_dev!(d_b) as *const pci_dev;
if pci_domain_nr(unsafe { (*a).bus }) < pci_domain_nr(unsafe { (*b).bus }) {
return 1i32.wrapping_neg();
} else {
if pci_domain_nr(unsafe { (*a).bus }) > pci_domain_nr(unsafe { (*b).bus }) {
return 1;
}
}
if ((unsafe { (*(*a).bus).number }) as i32) < ((unsafe { (*(*b).bus).number }) as i32) {
return 1i32.wrapping_neg();
} else {
if ((unsafe { (*(*a).bus).number }) as i32) > ((unsafe { (*(*b).bus).number }) as i32) {
return 1;
}
}
if (unsafe { (*a).devfn }) < unsafe { (*b).devfn } {
return 1i32.wrapping_neg();
} else {
if (unsafe { (*a).devfn }) > unsafe { (*b).devfn } {
return 1;
}
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn pci_sort_breadthfirst() {
unsafe { bus_sort_breadthfirst(core::ptr::addr_of!(pci_bus_type), Some(pci_sort_bf_cmp)) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.probe.pci_hp_add_bridge")]
extern "C" fn pci_hp_add_bridge(dev: *mut pci_dev) -> i32 {
let parent: *mut pci_bus = unsafe { (*dev).bus };
let mut busnr: i32 = 0;
let start: i32 = (unsafe { (*parent).busn_res.start }) as i32;
let mut available_buses: u32 = 0;
let end: i32 = (unsafe { (*parent).busn_res.end }) as i32;
busnr = start;
while busnr <= end {
if unsafe { pci_find_bus(pci_domain_nr(parent), busnr) }.is_null() {
break;
}
busnr = busnr.wrapping_add(1);
}
let __v3: i32 = busnr;
busnr = __v3.wrapping_sub(1);
if __v3 > end {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"No bus number available for hot-added bridge\n".as_ptr() as *const u8) };
return 1i32.wrapping_neg();
}
// Scan bridges that are already configured
busnr = pci_scan_bridge(parent, dev, busnr, 0i32);
// Distribute the available bus numbers between hotplug-capable
// bridges to make extending the chain later possible.
available_buses = end.wrapping_sub(busnr) as u32;
// Scan bridges that need to be reconfigured
pci_scan_bridge_extend(parent, dev, busnr, available_buses, 1i32);
if unsafe { (*dev).subordinate }.is_null() {
return 1i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_hp_add_bridge_683: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_hp_add_bridge)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_dma_set_max_seg_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_dma_set_max_seg_size_2: [u8; 57] = *b"/home/takashi/c-corpus/linux/include/linux/dma-mapping.h\0";

static __slate_asm_string_dma_set_seg_boundary_1: [u8; 1] = *b"\0";

static __slate_asm_string_dma_set_seg_boundary_2: [u8; 57] = *b"/home/takashi/c-corpus/linux/include/linux/dma-mapping.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_pci_alloc_child_bus_16: [u8; 1] = *b"\0";

static __slate_asm_string_pci_alloc_child_bus_17: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/pci/probe.c\0";

static __slate_asm_string_pci_alloc_child_bus_18: [u8; 1] = *b"\0";

static __slate_asm_string_pci_alloc_child_bus_19: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/pci/probe.c\0";

static __slate_asm_string_pci_device_add_1: [u8; 1] = *b"\0";

static __slate_asm_string_pci_device_add_2: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/pci/probe.c\0";

static __slate_asm_string_pci_device_add_6: [u8; 1] = *b"\0";

static __slate_asm_string_pci_device_add_7: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/pci/probe.c\0";

static __slate_asm_string_pci_register_host_bridge_38: [u8; 1] = *b"\0";

static __slate_asm_string_pci_register_host_bridge_39: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/pci/probe.c\0";
