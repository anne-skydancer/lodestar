#include "renderer.hpp"

#include "lodestar_shaders.hpp"
#include <lodestar/lodestar.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (Vulkan result " +
                                 std::to_string(result) + ")");
    }
}

struct Transforms {
    glm::mat4 mvp;
    glm::mat4 model;
};

static_assert(sizeof(Transforms) == 128);

template <std::size_t Size>
VkShaderModule createShader(VkDevice device, const std::array<std::uint32_t, Size>& code) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size() * sizeof(code[0]);
    info.pCode = code.data();
    VkShaderModule shader{};
    check(vkCreateShaderModule(device, &info, nullptr, &shader), "Creating a shader module");
    return shader;
}

} // namespace

void Renderer::initialize(GLFWwindow* window) {
    window_ = window;
    uint32_t extensionCount = 0;
    const char** extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    if (!extensions || !extensionCount) {
        throw std::runtime_error("GLFW could not find the Vulkan window extensions");
    }

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "Lodestar";
    application.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &application;
    info.enabledExtensionCount = extensionCount;
    info.ppEnabledExtensionNames = extensions;
    check(vkCreateInstance(&info, nullptr, &instance_), "Creating the Vulkan instance");
    check(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_), "Creating the window surface");
    createDevice();

    VkCommandPoolCreateInfo commandPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    commandPoolInfo.queueFamilyIndex = graphicsFamily_;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(device_, &commandPoolInfo, nullptr, &commandPool_), "Creating the command pool");
    VkCommandBufferAllocateInfo commandsInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandsInfo.commandPool = commandPool_;
    commandsInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandsInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device_, &commandsInfo, &commands_), "Allocating the command buffer");

    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailable_), "Creating the acquire semaphore");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(device_, &fenceInfo, nullptr, &frameComplete_), "Creating the frame fence");

    vertexShader_ = createShader(device_, lodestar::shaders::vertex);
    fragmentShader_ = createShader(device_, lodestar::shaders::fragment);
    VkPushConstantRange transforms{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Transforms)};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &transforms;
    check(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_), "Creating the pipeline layout");
}

void Renderer::createDevice() {
    uint32_t deviceCount = 0;
    check(vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr), "Enumerating Vulkan devices");
    std::vector<VkPhysicalDevice> devices(deviceCount);
    check(vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data()), "Reading Vulkan devices");

    const char* swapchainExtension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    for (auto candidate : devices) {
        uint32_t extensionCount = 0;
        check(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, nullptr),
              "Enumerating device extensions");
        std::vector<VkExtensionProperties> extensions(extensionCount);
        check(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, extensions.data()),
              "Reading device extensions");
        if (std::none_of(extensions.begin(), extensions.end(), [&](const auto& extension) {
                return std::strcmp(extension.extensionName, swapchainExtension) == 0;
            })) {
            continue;
        }

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
        uint32_t graphics = familyCount;
        uint32_t present = familyCount;
        for (uint32_t i = 0; i < familyCount; ++i) {
            VkBool32 canPresent{};
            check(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface_, &canPresent),
                  "Checking window presentation support");
            const bool canDraw = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            if (canDraw && canPresent) {
                graphics = present = i;
                break;
            }
            if (canDraw && graphics == familyCount) graphics = i;
            if (canPresent && present == familyCount) present = i;
        }
        if (graphics == familyCount || present == familyCount) continue;

        uint32_t formatCount = 0;
        uint32_t modeCount = 0;
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &formatCount, nullptr),
              "Checking surface formats");
        check(vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface_, &modeCount, nullptr),
              "Checking presentation modes");
        if (!formatCount || !modeCount) continue;

        for (auto format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM, VK_FORMAT_D24_UNORM_S8_UINT}) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(candidate, format, &properties);
            if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                depthFormat_ = format;
                break;
            }
        }
        if (depthFormat_ == VK_FORMAT_UNDEFINED) continue;
        physical_ = candidate;
        graphicsFamily_ = graphics;
        presentFamily_ = present;
        break;
    }
    if (!physical_) {
        throw std::runtime_error("No Vulkan device can draw and present to this window with depth testing");
    }

    const float priority = 1.0f;
    std::array<VkDeviceQueueCreateInfo, 2> queues{};
    queues[0] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, graphicsFamily_, 1, &priority};
    queues[1] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, presentFamily_, 1, &priority};
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.queueCreateInfoCount = graphicsFamily_ == presentFamily_ ? 1u : 2u;
    info.pQueueCreateInfos = queues.data();
    info.enabledExtensionCount = 1;
    info.ppEnabledExtensionNames = &swapchainExtension;
    check(vkCreateDevice(physical_, &info, nullptr, &device_), "Creating the Vulkan device");
    vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
}

void Renderer::createSwapchain(int width, int height) {
    check(vkDeviceWaitIdle(device_), "Waiting for the graphics device");
    destroySwapchain();

    VkSurfaceCapabilitiesKHR capabilities{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &capabilities), "Reading surface capabilities");
    extent_ = capabilities.currentExtent;
    if (extent_.width == std::numeric_limits<uint32_t>::max()) {
        extent_.width = std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width,
                                   capabilities.maxImageExtent.width);
        extent_.height = std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height,
                                    capabilities.maxImageExtent.height);
    }
    if (!extent_.width || !extent_.height) return;

    uint32_t formatCount = 0;
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &formatCount, nullptr), "Enumerating surface formats");
    if (!formatCount) throw std::runtime_error("The window surface has no Vulkan formats");
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &formatCount, formats.data()), "Reading surface formats");
    auto surfaceFormat = formats.front();
    for (const auto& format : formats) {
        if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
            (format.format == VK_FORMAT_B8G8R8A8_SRGB || format.format == VK_FORMAT_R8G8B8A8_SRGB)) {
            surfaceFormat = format;
            break;
        }
    }
    if (surfaceFormat.format == VK_FORMAT_UNDEFINED) surfaceFormat.format = VK_FORMAT_B8G8R8A8_SRGB;
    colorFormat_ = surfaceFormat.format;

    const std::array queueFamilies{graphicsFamily_, presentFamily_};
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = surface_;
    info.minImageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount) info.minImageCount = std::min(info.minImageCount, capabilities.maxImageCount);
    info.imageFormat = colorFormat_;
    info.imageColorSpace = surfaceFormat.colorSpace;
    info.imageExtent = extent_;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (graphicsFamily_ != presentFamily_) {
        info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size());
        info.pQueueFamilyIndices = queueFamilies.data();
    }
    info.preTransform = capabilities.currentTransform;
    for (auto alpha : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                       VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR}) {
        if (capabilities.supportedCompositeAlpha & alpha) {
            info.compositeAlpha = alpha;
            break;
        }
    }
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    check(vkCreateSwapchainKHR(device_, &info, nullptr, &swapchain_), "Creating the swapchain");

    uint32_t imageCount = 0;
    check(vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr), "Enumerating swapchain images");
    std::vector<VkImage> swapImages(imageCount);
    check(vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapImages.data()), "Reading swapchain images");
    images_.resize(imageCount);
    createDepthBuffer();
    createPipeline();

    for (uint32_t i = 0; i < imageCount; ++i) {
        auto& image = images_[i];
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = swapImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = colorFormat_;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device_, &viewInfo, nullptr, &image.view), "Creating a swapchain image view");

        const std::array attachments{image.view, depthView_};
        VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebufferInfo.renderPass = renderPass_;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = extent_.width;
        framebufferInfo.height = extent_.height;
        framebufferInfo.layers = 1;
        check(vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &image.framebuffer), "Creating a framebuffer");

        // Acquiring this image again guarantees presentation has consumed its semaphore.
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &image.rendered), "Creating a presentation semaphore");
    }
    framebufferWidth_ = width;
    framebufferHeight_ = height;
    swapchainDirty_ = false;
}

void Renderer::createDepthBuffer() {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = depthFormat_;
    info.extent = {extent_.width, extent_.height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateImage(device_, &info, nullptr, &depthImage_), "Creating the depth image");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device_, depthImage_, &requirements);
    VkMemoryAllocateInfo memory{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memory.allocationSize = requirements.size;
    memory.memoryTypeIndex = memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(device_, &memory, nullptr, &depthMemory_), "Allocating depth image memory");
    check(vkBindImageMemory(device_, depthImage_, depthMemory_, 0), "Binding the depth image");

    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = depthImage_;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = depthFormat_;
    view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(device_, &view, nullptr, &depthView_), "Creating the depth image view");
}

void Renderer::createPipeline() {
    VkAttachmentDescription color{};
    color.format = colorFormat_;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth = color;
    depth.format = depthFormat_;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    const std::array attachments{color, depth};
    VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;
    subpass.pDepthStencilAttachment = &depthReference;
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = static_cast<uint32_t>(attachments.size());
    pass.pAttachments = attachments.data();
    pass.subpassCount = 1;
    pass.pSubpasses = &subpass;
    pass.dependencyCount = 1;
    pass.pDependencies = &dependency;
    check(vkCreateRenderPass(device_, &pass, nullptr, &renderPass_), "Creating the render pass");

    VkPipelineShaderStageCreateInfo vertexStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertexStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertexStage.module = vertexShader_;
    vertexStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragmentStage = vertexStage;
    fragmentStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragmentStage.module = fragmentShader_;
    const std::array stages{vertexStage, fragmentStage};

    using lodestar::Vertex;
    VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array attributes{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
        VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
    };
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    input.vertexBindingDescriptionCount = 1;
    input.pVertexBindingDescriptions = &binding;
    input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
    input.pVertexAttributeDescriptions = attributes.data();
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depthTest{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthTest.depthTestEnable = VK_TRUE;
    depthTest.depthWriteEnable = VK_TRUE;
    depthTest.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;
    const std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamic.pDynamicStates = dynamicStates.data();

    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount = static_cast<uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &input;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisampling;
    pipeline.pDepthStencilState = &depthTest;
    pipeline.pColorBlendState = &blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = pipelineLayout_;
    pipeline.renderPass = renderPass_;
    check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_),
          "Creating the graphics pipeline");
}

uint32_t Renderer::memoryType(uint32_t allowed, VkMemoryPropertyFlags required) const {
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical_, &memory);
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((allowed & (1u << i)) && (memory.memoryTypes[i].propertyFlags & required) == required) return i;
    }
    throw std::runtime_error("The graphics device has no compatible memory type");
}

void Renderer::fillBuffer(Buffer& buffer, VkBufferUsageFlags usage, const void* data, VkDeviceSize size) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device_, &info, nullptr, &buffer.handle), "Creating a mesh buffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);
    VkMemoryAllocateInfo memory{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memory.allocationSize = requirements.size;
    memory.memoryTypeIndex = memoryType(requirements.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(device_, &memory, nullptr, &buffer.memory), "Allocating mesh buffer memory");
    check(vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0), "Binding a mesh buffer");
    void* mapped{};
    check(vkMapMemory(device_, buffer.memory, 0, size, 0, &mapped), "Mapping a mesh buffer");
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    vkUnmapMemory(device_, buffer.memory);
}

void Renderer::upload(const lodestar::Mesh& mesh) {
    if (mesh.vertices.empty() || mesh.indices.empty() || mesh.indices.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Mesh cannot be drawn with a single indexed Vulkan draw");
    }

    Buffer nextVertices{};
    Buffer nextIndices{};
    try {
        fillBuffer(nextVertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices.data(),
                   mesh.vertices.size() * sizeof(lodestar::Vertex));
        fillBuffer(nextIndices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices.data(),
                   mesh.indices.size() * sizeof(uint32_t));
        check(vkWaitForFences(device_, 1, &frameComplete_, VK_TRUE, UINT64_MAX), "Waiting to replace mesh buffers");
    } catch (...) {
        destroyBuffer(nextVertices);
        destroyBuffer(nextIndices);
        throw;
    }
    destroyBuffer(vertices_);
    destroyBuffer(indices_);
    vertices_ = nextVertices;
    indices_ = nextIndices;
    indexCount_ = static_cast<uint32_t>(mesh.indices.size());
}

void Renderer::draw(float yaw, float pitch, float distance) {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (!width || !height) return;
    if (!swapchain_ || swapchainDirty_ || width != framebufferWidth_ || height != framebufferHeight_) {
        createSwapchain(width, height);
        if (!swapchain_) return;
    }

    check(vkWaitForFences(device_, 1, &frameComplete_, VK_TRUE, UINT64_MAX), "Waiting for the preceding frame");
    uint32_t imageIndex = 0;
    const auto acquire = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_, VK_NULL_HANDLE, &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchainDirty_ = true;
        return;
    }
    if (acquire != VK_SUBOPTIMAL_KHR) check(acquire, "Acquiring a swapchain image");
    auto& image = images_[imageIndex];

    check(vkResetCommandBuffer(commands_, 0), "Resetting the command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(commands_, &begin), "Recording the frame");

    std::array<VkClearValue, 2> clear{};
    clear[0].color = {{0.014f, 0.022f, 0.037f, 1.0f}};
    clear[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = renderPass_;
    pass.framebuffer = image.framebuffer;
    pass.renderArea.extent = extent_;
    pass.clearValueCount = static_cast<uint32_t>(clear.size());
    pass.pClearValues = clear.data();
    vkCmdBeginRenderPass(commands_, &pass, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commands_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    const float viewportWidth = static_cast<float>(extent_.width);
    const float viewportHeight = static_cast<float>(extent_.height);
    const VkViewport viewport{0, 0, viewportWidth, viewportHeight, 0, 1};
    const VkRect2D scissor{{0, 0}, extent_};
    vkCmdSetViewport(commands_, 0, 1, &viewport);
    vkCmdSetScissor(commands_, 0, 1, &scissor);
    if (indexCount_) {
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(commands_, 0, 1, &vertices_.handle, &offset);
        vkCmdBindIndexBuffer(commands_, indices_.handle, 0, VK_INDEX_TYPE_UINT32);
        const auto tilt = glm::rotate(glm::mat4{1}, pitch, glm::vec3{1, 0, 0});
        const auto model = glm::rotate(tilt, yaw, glm::vec3{0, 1, 0});
        const auto view = glm::lookAtRH(glm::vec3{0, 0, distance}, glm::vec3{0}, glm::vec3{0, 1, 0});
        auto projection = glm::perspectiveRH_ZO(glm::radians(50.0f), viewportWidth / viewportHeight, 0.05f, 100.0f);
        projection[1][1] *= -1;
        const Transforms transforms{projection * view * model, model};
        vkCmdPushConstants(commands_, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Transforms), &transforms);
        vkCmdDrawIndexed(commands_, indexCount_, 1, 0, 0, 0);
    }
    vkCmdEndRenderPass(commands_);
    check(vkEndCommandBuffer(commands_), "Finishing frame recording");

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &imageAvailable_;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands_;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &image.rendered;
    check(vkResetFences(device_, 1, &frameComplete_), "Resetting the frame fence");
    check(vkQueueSubmit(graphicsQueue_, 1, &submit, frameComplete_), "Submitting the frame");

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &image.rendered;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &imageIndex;
    const auto result = vkQueuePresentKHR(presentQueue_, &present);
    if (result != VK_ERROR_OUT_OF_DATE_KHR && result != VK_SUBOPTIMAL_KHR) check(result, "Presenting the frame");
    swapchainDirty_ = result != VK_SUCCESS || acquire == VK_SUBOPTIMAL_KHR;
}

void Renderer::destroyBuffer(Buffer& buffer) {
    if (buffer.handle) vkDestroyBuffer(device_, buffer.handle, nullptr);
    if (buffer.memory) vkFreeMemory(device_, buffer.memory, nullptr);
    buffer = {};
}

void Renderer::destroySwapchain() {
    for (auto& image : images_) {
        if (image.framebuffer) vkDestroyFramebuffer(device_, image.framebuffer, nullptr);
        if (image.view) vkDestroyImageView(device_, image.view, nullptr);
        if (image.rendered) vkDestroySemaphore(device_, image.rendered, nullptr);
    }
    images_.clear();
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (renderPass_) vkDestroyRenderPass(device_, renderPass_, nullptr);
    if (depthView_) vkDestroyImageView(device_, depthView_, nullptr);
    if (depthImage_) vkDestroyImage(device_, depthImage_, nullptr);
    if (depthMemory_) vkFreeMemory(device_, depthMemory_, nullptr);
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    pipeline_ = {};
    renderPass_ = {};
    depthView_ = {};
    depthImage_ = {};
    depthMemory_ = {};
    swapchain_ = {};
}

Renderer::~Renderer() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        destroySwapchain();
        destroyBuffer(indices_);
        destroyBuffer(vertices_);
        if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        if (vertexShader_) vkDestroyShaderModule(device_, vertexShader_, nullptr);
        if (fragmentShader_) vkDestroyShaderModule(device_, fragmentShader_, nullptr);
        if (imageAvailable_) vkDestroySemaphore(device_, imageAvailable_, nullptr);
        if (frameComplete_) vkDestroyFence(device_, frameComplete_, nullptr);
        if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
}
