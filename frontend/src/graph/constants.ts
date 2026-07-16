// Baseline card width used by graph layout before label and shape content are measured.
export const nodeWidth = 244

// Fixed card height used for vertical edge anchors, layout spacing, and centering.
export const nodeHeight = 136

// Horizontal distance between graph depth columns.
export const columnGap = 128

// Distance between sibling nodes within the same depth level.
export const rowGap = 130

// Inner margin around the auto-laid-out graph before pan/zoom whiteboard padding is added.
export const padding = 64

// Smallest allowed viewport zoom scale.
export const minScale = 0.25

// Largest allowed viewport zoom scale.
export const maxScale = 2.4

// Extra draggable canvas space around the graph so users can pan past edge nodes.
export const whiteboardPadding = 1800

// ReLU naturally produces many zeros, so only flag unusually extreme sparsity.
export const sparseReluThresholdPercent = 90

// Other operations are flagged when more than this percentage of outputs are zero.
export const sparseDefaultThresholdPercent = 5
