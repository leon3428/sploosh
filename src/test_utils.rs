use crate::WgpuDevice;

pub fn read_buffer<T: bytemuck::Pod>(wgpu_device: &WgpuDevice, buffer: &wgpu::Buffer) -> Vec<T> {
    let buffer_slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    wgpu_device.device.poll(wgpu::Maintain::Wait);

    rx.recv().unwrap().unwrap();

    let data = buffer_slice.get_mapped_range().to_vec();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

    drop(data);
    buffer.unmap();

    result
}