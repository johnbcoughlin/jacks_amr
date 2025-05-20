# Jack's Adaptive Mesh Refinement library

This is a prototype / proof of concept library for adaptive mesh refinement (AMR) in JAX.

Prior art:
- [JAX-AMR](https://github.com/JA4S/JAX-AMR)
- [jamr](https://github.com/leo1200/jamr)
- [A gist by @patrick-kidger](https://gist.github.com/patrick-kidger/19e9c40199e5167cad1dc0e24d00855d)

![docs/images/radial_tanh_amr.png]

## Design

### Levels and Blocks

An AMR code consists of multiple nested "levels", each consisting of a certain number of *active* cells.
Typically, a cell at level `n` is refined into `2 x 2 x ...` cells at level `n+1`.
To avoid the appearance of "ell"-shaped elements, refinements must be strictly nested:
we can't refine 1/4 of a level `n` element into 4 level `n+2` elements, nor can we refine
1/4 of a level `n` element into a single level `n+1` element.

What this shows is that for a typical AMR implementation, active elements at a given level 
must appear at minimum in *blocks* of size `2 x 2`. The block is the minimum number of contiguous
elements that must be active together. However, the block need not be `2 x 2`. If we choose,
we can enforce that elements are always refined together in blocks of larger size like `4 x 4`, 
or even non-square blocks like `4 x 8`. The requirements are:
- A level `n+1` block must exactly cover some contiguous hypercube of elements at level `n`
- A level `n` block must be able to be covered exactly by some collection of level `n+1` blocks.

Because level 0 "starts out" fully active, we say that level 0 has a single block of size `Nx x Ny x ...`.

### Implementing in JAX

The fundamental problem to overcome with AMR in JAX is that we can't dynamically allocate or resize
arrays without re-jitting our code. Re-jitting will throw a wrench into everything, so we'd like to
avoid it at all costs. To do so, we pre-allocate all the block bookkeeping data for the entire grid.
For each level, we define a parameter, `n_blocks`, which limits the number of blocks that can ever
be active in that level.
For the finest grid, a reasonable choice is some fraction of the number of blocks that the whole grid
would require at that resolution, say 5%.
For coarser grids, one can perhaps allocate the same amount of storage, so that the levels are able
to cover the following amounts of the grid:
- At level `N-1`, 5%
- At level `N-2`, 20% (40% in 3D)
- At level `N-3`, 80% (100% in 3D)
- At level `N-4` and coarser, 100% of the grid.
This allocation scheme requires roughly 20% of the total storage that a fully refined level `N-1` grid
would require. The storage savings fraction can be tuned by adjusting parameters.

With this picture of pre-allocated storage in mind, we can describe the key bookkeeping arrays for the
AMR grid.

Let `(Nx, Ny)` be the shape of the level 0 block, and `(sx, sy)` be the size of a level `k` block.
Then at level `k`, there are `(Kx, Ky) = (Nx * 2**k // sx, Ny * 2**k // sy)` total possible blocks,
but recall that only `n_blocks` of them may be active.

At a given level `k`, we have the following data:
- `n_active`: An integer, giving the current number of active blocks.
- `block_index_map`: An indicator array of size `(Kx, Ky)`, whose entries are either `-1` or an integer in the range `[0, n_active)`.
- `block_indices`: A 2-tuple of 1D arrays, each of length `n_blocks`. These arrays give the indices of the upper-left element of each active block.
We also maintain an array `active_blocks` of booleans for convenience, indicating whether there is a block with active index `active_blocks[active_index]`.
