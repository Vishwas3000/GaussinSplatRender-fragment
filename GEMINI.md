# GEMINI.md

## Project Overview

This is a highly optimized 3D rendering application that implements Gaussian Splatting for Augmented Reality (AR). The project is written in Swift and Metal, and it uses a tile-based rendering pipeline to achieve real-time performance with a large number of splats.

The renderer is heavily optimized with a variety of techniques, including:

*   **GPU Frustum Culling:** Splats outside the camera's view are culled on the GPU to reduce the number of splats that need to be processed.
*   **Morton Code-based Spatial Indexing:** Morton codes are used to create a spatial index of the splats, which allows for efficient tile-based rasterization.
*   **Pre-computation of Expensive Calculations:** A compute shader is used to pre-calculate expensive per-pixel work, such as the transformation of splats from world to screen space and the inversion of the 2D covariance matrix.
*   **Tile-based Rasterization:** The screen is divided into tiles, and each tile is processed independently on the GPU. This allows for a high degree of parallelism.
*   **Depth Sorting:** Splats within each tile are sorted by depth to ensure correct alpha blending.
*   **Hybrid Rendering Pipeline:** The project includes an alternative "hybrid" rendering pipeline that uses CPU sorting for better energy efficiency.

The project also includes a sophisticated debug visualization system that can be used to visualize the tiles, heatmaps, workload, and culling statistics.

## Building and Running

The project is an Xcode project. To build and run it, you will need:

*   A Mac with an Apple Silicon chip (M1, M2, etc.)
*   Xcode 14 or later
*   An iOS or macOS device to run the application on

To build and run the project:

1.  Open `GaussianSplat.xcodeproj` in Xcode.
2.  Select the "GaussianSplat" scheme.
3.  Choose a compatible iOS or macOS device as the run destination.
4.  Click the "Run" button.

## Development Conventions

*   **Language:** The project is written in Swift and Metal Shading Language (MSL).
*   **Architecture:** The application follows a typical SwiftUI structure, with a `MetalView` that hosts the `MTKView` for rendering. The rendering logic is encapsulated in the `TiledSplatRenderer` class.
*   **Performance:** The code is heavily optimized for performance, with a focus on GPU-based computation and minimizing CPU-GPU synchronization.
*   **Debugging:** The project includes an extensive debug visualization system, which can be controlled through gestures. A double-tap on the screen cycles through the various debug modes.
*   **Code Style:** The code is well-formatted and follows standard Swift and Metal coding conventions.
