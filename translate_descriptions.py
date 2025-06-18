from googletrans import Translator
import os
import time

def get_translated_ids(output_file):
    """Lấy danh sách các ID ảnh đã được dịch"""
    translated_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    image_id = line.strip().split('\t')[0]
                    translated_ids.add(image_id)
        except Exception as e:
            print(f"Lỗi khi đọc file kết quả cũ: {str(e)}")
    return translated_ids

def translate_descriptions(input_file, output_file):
    # Kiểm tra file đầu vào
    if not os.path.exists(input_file):
        print(f"Lỗi: File đầu vào {input_file} không tồn tại!")
        return False
    
    print(f"Bắt đầu quá trình dịch từ file: {input_file}")
    print(f"Kết quả sẽ được lưu vào: {output_file}")
    
    # Lấy danh sách ID đã dịch
    translated_ids = get_translated_ids(output_file)
    print(f"Đã tìm thấy {len(translated_ids)} dòng đã được dịch trước đó")
    
    translator = Translator()
    
    try:
        # Đọc file gốc
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"Đã đọc thành công {total_lines} dòng từ file gốc")
        
        # Đọc các dòng đã dịch (nếu có)
        translated_lines = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                translated_lines = f.readlines()
            print(f"Đã đọc {len(translated_lines)} dòng từ file kết quả cũ")
        
        print("Bắt đầu quá trình dịch...")
        processed_count = len(translated_lines)
        
        for i, line in enumerate(lines, 1):
            if i % 10 == 0:
                print(f"Đang xử lý: {i}/{total_lines} ({(i/total_lines*100):.1f}%)")
            
            # Tách ID ảnh và miêu tả
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"Cảnh báo: Dòng {i} không đúng định dạng, bỏ qua")
                continue
            
            image_id, description = parts
            
            # Bỏ qua nếu ID này đã được dịch
            if image_id in translated_ids:
                continue
            
            try:
                # Thêm delay nhỏ để tránh bị chặn
                time.sleep(0.1)
                # Dịch miêu tả sang tiếng Việt
                translated = translator.translate(description, src='en', dest='vi')
                translated_line = f"{image_id}\t{translated.text}\n"
                translated_lines.append(translated_line)
                translated_ids.add(image_id)
                processed_count += 1
                
                # Lưu tạm thường xuyên hơn
                if processed_count % 100 == 0:
                    print(f"Lưu tạm kết quả tại dòng {i}...")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.writelines(translated_lines)
                    
            except Exception as e:
                print(f"Lỗi khi dịch dòng {i}: {str(e)}")
                translated_lines.append(line)  # Giữ nguyên dòng gốc nếu có lỗi
        
        # Lưu kết quả cuối cùng
        print("Đang lưu kết quả cuối cùng...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(translated_lines)
        
        print(f"Hoàn thành! Đã dịch {len(translated_lines)} dòng")
        print(f"Kết quả đã được lưu vào: {output_file}")
        return True
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "data/text/Flickr8k.token.txt"
    output_file = "data/text/Flickr8k.token.vietnamese.txt"
    
    success = translate_descriptions(input_file, output_file)
    if success:
        print("Chương trình kết thúc thành công!")
    else:
        print("Chương trình kết thúc với lỗi!") 