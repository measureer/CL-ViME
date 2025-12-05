import scapy.all as scapy
from collections import defaultdict
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from mytimer import mytimer
import datetime
import numpy as np
from PIL import Image

def split_hex_to_int(hex_string):
    """更高效的处理16进制字符串转换"""
    #hex_string = hex_string[:24] + hex_string[31:32]+ hex_string[39:]
    #print(hex_string)
    try:
        return [int(hex_string[i:i+2], 16) for i in range(0, len(hex_string), 2)]
    except ValueError:
        # 处理不完整的最后字节
        return [int(hex_string[i:i+2], 16) if i+2 <= len(hex_string) else 0 
                for i in range(0, len(hex_string), 2)]

def normalize_session_key(src_ip, dst_ip, src_port, dst_port, protocol):
    """标准化会话标识符"""
    if (src_ip, src_port) > (dst_ip, dst_port):
        return (dst_ip, src_ip, dst_port, src_port, protocol)
    return (src_ip, dst_ip, src_port, dst_port, protocol)

def process_single_packet(pkt):
    """处理单个包，提取关键信息"""
    # 检查是否为IPv4包（排除IPv6和其他协议）
    if not pkt.haslayer(scapy.IP) or pkt.haslayer(scapy.IPv6) or not (pkt.haslayer(scapy.TCP) or pkt.haslayer(scapy.UDP)):
        return None
    pkt = pkt[scapy.IP]

    src_ip = pkt[scapy.IP].src
    dst_ip = pkt[scapy.IP].dst

    if pkt.haslayer(scapy.TCP):
        src_port = pkt[scapy.TCP].sport
        dst_port = pkt[scapy.TCP].dport
        protocol = "TCP"
        raw=scapy.bytes_hex(pkt)
        #IP
        ip_header = raw[:40]
        ip_header=ip_header[:24] + ip_header[31:32]+ ip_header[39:]
        #TCP
        tcp_header_length = pkt[scapy.TCP].dataofs * 8  # 计算TCP头字节
        tcp_header = raw[40:40+tcp_header_length]
        if(tcp_header_length<80):
            tcp_header=tcp_header+b'0'*(80-tcp_header_length)
        if(tcp_header_length>80):
            tcp_header=tcp_header[:80]
        #Payload
        payload=raw[40+tcp_header_length:]
        packet=ip_header+tcp_header+payload
    elif pkt.haslayer(scapy.UDP):
        src_port = pkt[scapy.UDP].sport
        dst_port = pkt[scapy.UDP].dport
        protocol = "UDP"
        packet=scapy.bytes_hex(pkt)
    else:
        return None
    return {
        'key': normalize_session_key(src_ip, dst_ip, src_port, dst_port, protocol),
        'time': pkt.time,
        'raw': packet
    }



def process_session_packets(packets, flow_length, output_path, base_filename, index):
    """处理单个会话的包数据并分批保存为多张图像"""
    packets = sorted(packets, key=lambda x: x['time'])
    total_packets = len(packets)
    batch_count = max(1, (total_packets // flow_length))#一个会话会被分成几张图 
    for batch_num in range(batch_count):
        row = []
        start_idx = batch_num * flow_length
        end_idx = start_idx + flow_length
        
        for i in range(start_idx, end_idx):
                
            if i < total_packets:
                if(packets[i]['time']-packets[i-1]['time']>60):
                    row=[]
                    break
                try:
                    pa = split_hex_to_int(packets[i]['raw'])
                    pa = pa[:flow_length] + [0] * (flow_length - len(pa)) if len(pa) < flow_length else pa[:flow_length]
                    row.append(pa)
                except Exception:
                    row.append([0] * flow_length)
            else:
                row.append([0] * flow_length)#不满的行用0填充
        
        if not row:
            continue
        
        try:
            image_array = np.array(row, dtype=np.uint8)
            image = Image.fromarray(image_array)
            # 在文件名中加入批次号
            image_path = os.path.join(output_path, f"{base_filename}_{index}_b{batch_num}.png")
            image.save(image_path)
        except Exception as e:
            print(f"Error saving image: {e}")
'''
def process_session_packets(packets, flow_length, output_path, base_filename, index):
    """处理单个会话的包数据并分批保存为多张图像"""
    packets = sorted(packets, key=lambda x: x['time'])
    row = []
    batch_num = 0
    
    for i, pkt in enumerate(packets):
        # 如果是第一个包或者与前一个包时间差超过60秒，则开始新批次
        if i == 0 or (pkt['time'] - packets[i-1]['time']) > 60:
            if row:  # 保存当前批次
                try:
                    image_array = np.array(row, dtype=np.uint8)
                    image = Image.fromarray(image_array)
                    image_path = os.path.join(output_path, f"{base_filename}_{index}_b{batch_num}.png")
                    image.save(image_path)
                    batch_num += 1
                except Exception as e:
                    print(f"Error saving image: {e}")
            row = []  # 开始新批次
        
        try:
            pa = split_hex_to_int(pkt['raw'])
            pa = pa[:flow_length] + [0] * (flow_length - len(pa)) if len(pa) < flow_length else pa[:flow_length]
            row.append(pa)
        except Exception:
            row.append([0] * flow_length)
    
    # 处理最后一批数据
    if row:
        try:
            image_array = np.array(row, dtype=np.uint8)
            image = Image.fromarray(image_array)
            image_path = os.path.join(output_path, f"{base_filename}_{index}_b{batch_num}.png")
            image.save(image_path)
        except Exception as e:
            print(f"Error saving image: {e}")
'''
@mytimer
def extract_flow_features(pcap_file, output, flow_length=None, session_num=0):
    """优化内存使用的流特征提取函数"""
    if not os.path.isfile(pcap_file):
        raise ValueError(f"Input file {pcap_file} does not exist")
    
    os.makedirs(output, exist_ok=True)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing {pcap_file}")
    
    # 使用生成器逐包处理
    session_dict = defaultdict(list)
    try:
        # 使用scapy的PcapReader逐包读取而不是一次性加载
        for pkt in scapy.PcapReader(pcap_file):
            packet_info = process_single_packet(pkt)
            if packet_info:
                session_dict[packet_info['key']].append(packet_info)
    except Exception as e:
        print(f"Error reading {pcap_file}: {e}")
        return
    
    # 创建输出目录
    path = os.path.join(output, os.path.basename(os.path.dirname(pcap_file)))
    os.makedirs(path, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(pcap_file))[0]
    
    # 分批处理会话
    index = 0
    for session_key, packets in session_dict.items():
        if len(packets) <= session_num:
            continue
            
        # 处理并保存会话数据(现在会生成多张图片)
        process_session_packets(packets, flow_length, path, base_filename, index)
        index += 1
        
        # 定期清理内存
        if index % 100 == 0:
            session_dict[session_key] = None  # 释放已处理会话的内存

def process_directory(pcap_folder_path, output_folder_path, flow_length=None, max_workers=4, session_num=0):
    """并行处理所有pcap文件"""
    if not os.path.isdir(pcap_folder_path):
        raise ValueError(f"Input directory {pcap_folder_path} does not exist")
    
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing directory {pcap_folder_path}")
    
    success_count = 0
    error_count = 0
    
    # 收集所有要处理的文件
    file_paths = []
    for category in os.listdir(pcap_folder_path):
        category_path = os.path.join(pcap_folder_path, category)
        if not os.path.isdir(category_path):
            continue
            
        for file in os.listdir(category_path):
            if file.endswith((".pcap", ".pcapng")):
                file_paths.append(os.path.join(category_path, file))
    
    # 分批处理文件以减少内存压力
    batch_size = max_workers * 2
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        
        with ProcessPoolExecutor(max_workers) as executor:
            futures = []
            for file_path in batch:
                try:
                    futures.append(executor.submit(
                        extract_flow_features, 
                        file_path, 
                        output_folder_path, 
                        flow_length, 
                        session_num
                    ))
                except Exception as e:
                    print(f"Error submitting task for {file_path}: {e}")
                    error_count += 1
            
            for future in as_completed(futures):
                try:
                    future.result()
                    success_count += 1
                except Exception as e:
                    print(f"Error processing file: {e}")
                    error_count += 1
    
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing completed")
    print(f"Successfully processed files: {success_count}")
    print(f"Failed files: {error_count}")

if __name__ == "__main__":
    flow_length = 60
    max_workers=20
    pcap_folder_path = "/home/ubuntu/DataSet/PCAP/ISCX-Tor-NonTor-2017/PCAPs/Tor"
    output_folder_path = "/home/ubuntu/DataSet/Pic/Original/Tor_tcp40_ut_60_2"
    process_directory(pcap_folder_path, output_folder_path, 
                     flow_length=flow_length, max_workers=max_workers, session_num=5)
    '''
    pcap_folder_path = "/home/ubuntu/DataSet/PCAP/CICIOT34"
    output_folder_path = "/home/ubuntu/DataSet/Pic/Original/CICIOT34_tcp40_ut_60_2"
    process_directory(pcap_folder_path, output_folder_path, 
                     flow_length=flow_length, max_workers=max_workers, session_num=5)

    pcap_folder_path = "/home/ubuntu/DataSet/PCAP/MAWI/MAWI1_5"
    output_folder_path = "/home/ubuntu/DataSet/Pic/Original/MAWI1_5_tcp40_ut_60_2"
    process_directory(pcap_folder_path, output_folder_path, 
                     flow_length=flow_length, max_workers=max_workers, session_num=5)

    pcap_folder_path = "/home/ubuntu/DataSet/PCAP/ISCX-VPN-NonVPN-2016/PCAPs/VPN"
    output_folder_path = "/home/ubuntu/DataSet/Pic/Original/VPN_tcp40_ut_60_2"
    process_directory(pcap_folder_path, output_folder_path, 
                     flow_length=flow_length, max_workers=max_workers, session_num=5)
    pcap_folder_path = "/home/ubuntu/DataSet/PCAP/CICIOT2023/CICIOT_mini_split"
    output_folder_path = "/home/ubuntu/DataSet/Pic/Original/CICIOT_mini_split_tcp40_ut_60_2"
    process_directory(pcap_folder_path, output_folder_path, 
                     flow_length=flow_length, max_workers=max_workers, session_num=5)
    '''