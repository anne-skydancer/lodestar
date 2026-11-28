#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

struct GLFWwindow;
namespace lodestar {
struct Mesh;
}

class Renderer {
public:
    Renderer() = default;
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void initialize(GLFWwindow* window);
    void upload(const lodestar::Mesh& mesh);
    void draw(float yaw, float pitch, float distance);

private:
    struct Buffer {
        VkBuffer handle{};
        VkDeviceMemory memory{};
    };

    struct SwapImage {
        VkImageView view{};
        VkFramebuffer framebuffer{};
        VkSemaphore rendered{};
    };

    void createDevice();
    void createSwapchain(int width, int height);
    void createDepthBuffer();
    void createPipeline();
    void destroySwapchain();
    void destroyBuffer(Buffer& buffer);
    void fillBuffer(Buffer& buffer, VkBufferUsageFlags usage, const void* data, VkDeviceSize size);
    uint32_t memoryType(uint32_t allowed, VkMemoryPropertyFlags required) const;

    GLFWwindow* window_{};
    VkInstance instance_{};
    VkSurfaceKHR surface_{};
    VkPhysicalDevice physical_{};
    VkDevice device_{};
    uint32_t graphicsFamily_{};
    uint32_t presentFamily_{};
    VkQueue graphicsQueue_{};
    VkQueue presentQueue_{};

    VkCommandPool commandPool_{};
    VkCommandBuffer commands_{};
    VkFence frameComplete_{};
    VkSemaphore imageAvailable_{};
    VkShaderModule vertexShader_{};
    VkShaderModule fragmentShader_{};
    VkPipelineLayout pipelineLayout_{};

    VkSwapchainKHR swapchain_{};
    std::vector<SwapImage> images_;
    VkFormat colorFormat_{};
    VkFormat depthFormat_{};
    VkExtent2D extent_{};
    VkImage depthImage_{};
    VkDeviceMemory depthMemory_{};
    VkImageView depthView_{};
    VkRenderPass renderPass_{};
    VkPipeline pipeline_{};
    int framebufferWidth_{};
    int framebufferHeight_{};
    bool swapchainDirty_{};

    Buffer vertices_;
    Buffer indices_;
    uint32_t indexCount_{};
};
