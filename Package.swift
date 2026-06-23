// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "JaxsonKit",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        // The UI-free core. The macOS app (added in v0.1) will depend on this.
        .library(name: "JaxsonCore", targets: ["JaxsonCore"])
    ],
    targets: [
        // Shared, deterministic value types: the agent's state-machine primitives.
        .target(name: "JaxsonCore"),
        .testTarget(
            name: "JaxsonCoreTests",
            dependencies: ["JaxsonCore"]
        )
    ]
)
