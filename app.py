import streamlit as st
import pystac_client
import planetary_computer
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
import datetime
import plotly.express as px
from skimage import exposure
import json

# ページ設定
st.set_page_config(
    page_title="Sentinel衛星データ取得アプリ",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🛰️ Sentinel衛星データ取得アプリ")
st.markdown("任意の場所と日時でSentinel-1およびSentinel-2の観測画像を取得できます。")

# サイドバー
st.sidebar.header("検索条件の設定")

@st.cache_resource
def get_catalog():
    """Planetary Computer カタログを取得"""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    return catalog

def create_map(lat=35.681236, lon=139.767125, zoom=10):
    """地図を作成"""
    m = folium.Map(location=[lat, lon], zoom_start=zoom)
    return m

def transform_coordinates(coords, in_crs="epsg:4326", out_crs="epsg:32654"):
    """座標変換"""
    transformer = Transformer.from_crs(in_crs, out_crs)
    return [transformer.transform(lat, lon) for lon, lat in coords]

def crop_image_to_aoi(data, area_of_interest, target_crs):
    """画像をAOIに切り取り"""
    in_proj = 'epsg:4326'
    
    # area_of_interestの座標を変換
    transformer = Transformer.from_crs(in_proj, target_crs)
    new_coordinates = [transformer.transform(lat, lon) for lon, lat in area_of_interest['coordinates'][0]]
    new_area_of_interest = {'type': 'Polygon', 'coordinates': [new_coordinates]}
    
    # 各バンドのデータから AOI に対応する部分を切り取る
    cropped_data = []
    transforms = []
    
    for band_data in data:
        out_image_crop, out_transform = mask(band_data, [new_area_of_interest], crop=True)
        cropped_data.append(out_image_crop)
        transforms.append(out_transform)
    
    return cropped_data, transforms

# メイン処理
def main():
    try:
        # カタログの取得
        with st.spinner("カタログを読み込み中..."):
            catalog = get_catalog()
        
        # 衛星データタイプの選択
        satellite_type = st.sidebar.selectbox(
            "衛星データタイプ",
            ["Sentinel-2 (光学)", "Sentinel-1 (SAR)"]
        )
        
        collection_id = "sentinel-2-l2a" if satellite_type == "Sentinel-2 (光学)" else "sentinel-1-grd"
        
        # 日時の選択
        st.sidebar.subheader("📅 日時の範囲")
        
        # デフォルトの日時設定
        default_start = datetime.date.today() - datetime.timedelta(days=30)
        default_end = datetime.date.today()
        
        start_date = st.sidebar.date_input("開始日", value=default_start)
        end_date = st.sidebar.date_input("終了日", value=default_end)
        
        if start_date > end_date:
            st.sidebar.error("開始日は終了日より前に設定してください。")
            return
        
        time_range = f"{start_date.isoformat()}/{end_date.isoformat()}"
        
        # 場所の選択方法
        st.sidebar.subheader("📍 場所の選択")
        location_method = st.sidebar.radio(
            "選択方法",
            ["地図選択", "座標入力"]
        )
        
        area_of_interest = None
        
        # セッションステートで area_of_interest を保存
        if 'area_of_interest' not in st.session_state:
            st.session_state.area_of_interest = None
        
        if location_method == "地図選択":
            st.subheader("🗺️ 地図で関心場所を選択")
            
            # 地図の中心座標をセッションステートで管理
            if 'map_center_lat' not in st.session_state:
                st.session_state.map_center_lat = 35.681236
            if 'map_center_lon' not in st.session_state:
                st.session_state.map_center_lon = 139.767125
            if 'zoom_level' not in st.session_state:
                st.session_state.zoom_level = 10
                
            # 地図の中心座標設定
            col1, col2, col3 = st.sidebar.columns(3)
            with col1:
                new_center_lat = st.number_input("中心緯度", value=st.session_state.map_center_lat, format="%.6f", key="center_lat_input")
                if new_center_lat != st.session_state.map_center_lat:
                    st.session_state.map_center_lat = new_center_lat
            with col2:
                new_center_lon = st.number_input("中心経度", value=st.session_state.map_center_lon, format="%.6f", key="center_lon_input")
                if new_center_lon != st.session_state.map_center_lon:
                    st.session_state.map_center_lon = new_center_lon
            with col3:
                new_zoom_level = st.slider("ズーム", min_value=1, max_value=18, value=st.session_state.zoom_level, key="zoom_slider")
                if new_zoom_level != st.session_state.zoom_level:
                    st.session_state.zoom_level = new_zoom_level
                    
            # 現在の値を取得
            center_lat = st.session_state.map_center_lat
            center_lon = st.session_state.map_center_lon
            zoom_level = st.session_state.zoom_level
            
            # 地図中心更新ボタン
            if st.sidebar.button("🎯 地図中心を現在の選択点に移動"):
                st.session_state.map_center_lat = st.session_state.selected_lat
                st.session_state.map_center_lon = st.session_state.selected_lon
                st.rerun()
            
            # 領域設定方法の選択
            area_method = st.sidebar.radio(
                "領域設定方法",
                ["地図をクリックして設定", "座標を直接入力"]
            )
            
            if area_method == "地図をクリックして設定":
                st.info("""
                📍 **地図操作機能:**
                1. **シングルクリック**: 選択点を設定し、地図中心を移動（赤いマーカーが移動し、地図がその位置を中心に再配置）
                2. **ダブルクリック**: 地図中心のみを移動（選択点は変更せず、地図の表示位置のみ変更）
                3. サイドバーで範囲サイズを調整
                4. 選択領域が赤い矩形で地図上に表示されます
                
                💡 **使い方のコツ:**
                - シングルクリックで選択点を設定すると自動的に地図中心も移動
                - ダブルクリックで選択点を変更せずに地図表示位置のみ変更
                - 手動で座標入力すると自動的に地図中心も移動
                """)
                
                # セッションステートの初期化（選択点を地図中心に設定）
                if 'selected_lat' not in st.session_state:
                    st.session_state.selected_lat = st.session_state.map_center_lat
                if 'selected_lon' not in st.session_state:
                    st.session_state.selected_lon = st.session_state.map_center_lon
                
                # 範囲サイズの設定
                range_size = st.sidebar.slider(
                    "範囲サイズ (度)", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=0.1, 
                    step=0.01,
                    help="選択点から四方向への範囲（度数）"
                )
                
                # 地図作成
                m = folium.Map(
                    location=[center_lat, center_lon], 
                    zoom_start=zoom_level,
                    tiles='OpenStreetMap'
                )
                
                # クリックイベント処理のためのJavaScript
                click_js = """
                function onClick(e) {
                    var lat = e.latlng.lat;
                    var lng = e.latlng.lng;
                    // StreamlitのセッションステートにデータをセットするJavaScript
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: {lat: lat, lng: lng}
                    }, '*');
                }
                """
                
                # 座標設定
                st.write("**座標を調整：**")
                st.caption("💡 地図をクリックして選択点を設定することもできます")
                col1, col2 = st.columns(2)
                with col1:
                    new_lat = st.number_input("選択緯度", value=st.session_state.selected_lat, format="%.6f", key="new_lat")
                    if new_lat != st.session_state.selected_lat:
                        st.session_state.selected_lat = new_lat
                        # 地図中心も更新
                        st.session_state.map_center_lat = new_lat
                with col2:
                    new_lon = st.number_input("選択経度", value=st.session_state.selected_lon, format="%.6f", key="new_lon") 
                    if new_lon != st.session_state.selected_lon:
                        st.session_state.selected_lon = new_lon
                        # 地図中心も更新
                        st.session_state.map_center_lon = new_lon
                
                # 更新された座標を取得
                selected_lat = st.session_state.selected_lat
                selected_lon = st.session_state.selected_lon
                
                # 検索範囲を計算
                lat_min = selected_lat - range_size
                lat_max = selected_lat + range_size
                lon_min = selected_lon - range_size  
                lon_max = selected_lon + range_size
                
                # マーカーと選択領域を地図に追加
                folium.Marker(
                    [selected_lat, selected_lon],
                    popup=f"選択点: ({selected_lat:.6f}, {selected_lon:.6f})",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                
                folium.Rectangle(
                    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                    color='red',
                    fillColor='red',
                    fillOpacity=0.2,
                    popup=f"検索範囲\n緯度: {lat_min:.6f} - {lat_max:.6f}\n経度: {lon_min:.6f} - {lon_max:.6f}"
                ).add_to(m)
                
                # 地図表示（クリックとダブルクリックの両方を監視）
                map_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked", "last_clicked"])
                
                # セッションステートでクリック履歴を管理（ダブルクリック検出用）
                if 'last_click_time' not in st.session_state:
                    st.session_state.last_click_time = 0
                if 'last_click_pos' not in st.session_state:
                    st.session_state.last_click_pos = None
                if 'selected_item_id' not in st.session_state:
                    st.session_state.selected_item_id = None
                if 'search_results' not in st.session_state:
                    st.session_state.search_results = []
                if 'search_df' not in st.session_state:
                    st.session_state.search_df = None
                if 'selected_item' not in st.session_state:
                    st.session_state.selected_item = None
                if 'display_items' not in st.session_state:
                    st.session_state.display_items = []
                
                
                # 地図クリック処理
                try:
                    # 地図クリック処理（選択点の更新とダブルクリック検出を統合）
                    if map_data and 'last_clicked' in map_data and map_data['last_clicked'] is not None:
                        if 'lat' in map_data['last_clicked'] and 'lng' in map_data['last_clicked']:
                            clicked_lat = map_data['last_clicked']['lat']
                            clicked_lon = map_data['last_clicked']['lng']
                            
                            import time
                            current_time = time.time()
                            
                            # ダブルクリックの検出（1秒以内の2回クリック、より近い位置）
                            if (st.session_state.last_click_pos is not None and
                                current_time - st.session_state.last_click_time < 1.0 and
                                abs(clicked_lat - st.session_state.last_click_pos[0]) < 0.01 and
                                abs(clicked_lon - st.session_state.last_click_pos[1]) < 0.01):
                                
                                # ダブルクリック検出 - 地図中心を移動
                                st.session_state.map_center_lat = clicked_lat
                                st.session_state.map_center_lon = clicked_lon
                                # クリック履歴をリセット（連続ダブルクリックを防ぐ）
                                st.session_state.last_click_time = 0
                                st.session_state.last_click_pos = None
                                st.success(f"🎯 ダブルクリック検出！地図中心を移動: {clicked_lat:.6f}, {clicked_lon:.6f}")
                                st.rerun()
                            else:
                                # シングルクリック - 選択点を更新し、地図中心も移動
                                if (abs(clicked_lat - st.session_state.selected_lat) > 0.0001 or 
                                    abs(clicked_lon - st.session_state.selected_lon) > 0.0001):
                                    st.session_state.selected_lat = clicked_lat
                                    st.session_state.selected_lon = clicked_lon
                                    # 地図中心も選択点に移動
                                    st.session_state.map_center_lat = clicked_lat
                                    st.session_state.map_center_lon = clicked_lon
                                    st.info(f"📍 選択点を更新し地図中心を移動: {clicked_lat:.6f}, {clicked_lon:.6f}")
                                    st.rerun()
                                
                                # クリック履歴を更新（ダブルクリック検出用）
                                st.session_state.last_click_time = current_time
                                st.session_state.last_click_pos = (clicked_lat, clicked_lon)
                                
                except (KeyError, TypeError, AttributeError) as e:
                    # 地図クリックデータの処理でエラーが発生した場合は無視
                    pass
                
                # area_of_interestを設定
                area_of_interest = {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon_min, lat_min],
                        [lon_max, lat_min],
                        [lon_max, lat_max],
                        [lon_min, lat_max],
                        [lon_min, lat_min]
                    ]]
                }
                
                # セッションステートに保存
                st.session_state.area_of_interest = area_of_interest
                
                # 現在の選択情報を表示
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🎯 **選択中心点**\n緯度: {selected_lat:.6f}\n経度: {selected_lon:.6f}")
                    # 選択点を地図中心に移動するボタン
                    if st.button("🎯 選択点を地図中心に移動", key="move_to_selected", help="現在の選択点に地図中心を移動"):
                        st.session_state.map_center_lat = selected_lat
                        st.session_state.map_center_lon = selected_lon
                        st.success(f"地図中心を移動: {selected_lat:.6f}, {selected_lon:.6f}")
                        st.rerun()
                        
                with col2:
                    st.success(f"📐 **検索範囲**\n緯度: {lat_min:.6f} - {lat_max:.6f}\n経度: {lon_min:.6f} - {lon_max:.6f}\n範囲サイズ: {range_size:.3f}度")
                
            else:  # 座標を直接入力
                st.write("**座標を直接入力してください：**")
                col1, col2 = st.columns(2)
                with col1:
                    lat_min = st.number_input("緯度(最小)", value=35.60, format="%.6f")
                    lon_min = st.number_input("経度(最小)", value=139.60, format="%.6f")
                with col2:
                    lat_max = st.number_input("緯度(最大)", value=35.75, format="%.6f") 
                    lon_max = st.number_input("経度(最大)", value=139.80, format="%.6f")
                
                # 入力値の検証
                if lat_min >= lat_max or lon_min >= lon_max:
                    st.error("最小値は最大値より小さく設定してください。")
                    st.session_state.area_of_interest = None
                else:
                    # area_of_interestを設定
                    area_of_interest = {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon_min, lat_min],
                            [lon_max, lat_min],
                            [lon_max, lat_max],
                            [lon_min, lat_max],
                            [lon_min, lat_min]
                        ]]
                    }
                    
                    # 入力された領域を地図に表示
                    m_display = folium.Map(
                        location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], 
                        zoom_start=zoom_level
                    )
                    
                    folium.Rectangle(
                        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                        color='blue',
                        fillColor='blue',
                        fillOpacity=0.3,
                        popup=f"検索範囲\n緯度: {lat_min:.6f} - {lat_max:.6f}\n経度: {lon_min:.6f} - {lon_max:.6f}"
                    ).add_to(m_display)
                    
                    st.write("**入力された領域のプレビュー:**")
                    st_folium(m_display, width=700, height=300, returned_objects=[])
        
        elif location_method == "座標入力":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                lat_min = st.number_input("緯度(最小)", value=35.6, format="%.6f")
                lon_min = st.number_input("経度(最小)", value=139.7, format="%.6f")
            with col2:
                lat_max = st.number_input("緯度(最大)", value=35.7, format="%.6f")
                lon_max = st.number_input("経度(最大)", value=139.8, format="%.6f")
            
            # bbox to polygon
            area_of_interest = {
                "type": "Polygon",
                "coordinates": [[
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min]
                ]]
            }
            
            # セッションステートに保存
            st.session_state.area_of_interest = area_of_interest
        
        # クラウドカバーフィルター（Sentinel-2の場合のみ）
        if satellite_type == "Sentinel-2 (光学)":
            max_cloud_cover = st.sidebar.slider(
                "最大雲量(%)",
                min_value=0,
                max_value=100,
                value=30,
                help="設定した値以下の雲量の画像のみ表示されます"
            )
        
        # 検索ボタン
        if st.sidebar.button("🔍 データを検索"):
            if st.session_state.area_of_interest:
                with st.spinner("データを検索中..."):
                    # データ検索
                    search_params = {
                        "collections": [collection_id],
                        "intersects": st.session_state.area_of_interest,
                        "datetime": time_range
                    }
                    
                    if satellite_type == "Sentinel-2 (光学)":
                        search_params["query"] = {"eo:cloud_cover": {"lt": max_cloud_cover}}
                    
                    search = catalog.search(**search_params)
                    items_collection = search.item_collection()
                    items = list(items_collection)
                    
                    if len(items) == 0:
                        st.warning("指定した条件でデータが見つかりませんでした。条件を変更してください。")
                        return
                    
                    st.success(f"✅ {len(items)}個のデータが見つかりました！")
                    
                    # 検索結果の概要を表示
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 検索結果数", len(items))
                    with col2:
                        if satellite_type == "Sentinel-2 (光学)":
                            # 平均雲量を計算（可能な場合）
                            cloud_covers = [item.properties.get("eo:cloud_cover") 
                                          for item in items 
                                          if item.properties.get("eo:cloud_cover") is not None]
                            if cloud_covers:
                                avg_cloud = sum(cloud_covers) / len(cloud_covers)
                                st.metric("☁️ 平均雲量", f"{avg_cloud:.1f}%")
                    
                    # データフレームの作成
                    try:
                        # STACアイテムからGeoDataFrameを安全に作成
                        features = []
                        for item in items:
                            feature = {
                                "type": "Feature",
                                "properties": {
                                    "id": item.id,
                                    "datetime": item.properties.get("datetime"),
                                    "eo:cloud_cover": item.properties.get("eo:cloud_cover"),
                                    "collection": item.collection_id if hasattr(item, 'collection_id') else collection_id
                                },
                                "geometry": item.geometry
                            }
                            features.append(feature)
                        
                        items_dict = {"type": "FeatureCollection", "features": features}
                        df = gpd.GeoDataFrame.from_features(items_dict, crs="epsg:4326")
                        
                        # インデックスをリセット
                        df = df.reset_index(drop=True)
                        
                    except Exception as e:
                        st.error(f"データフレーム作成中にエラー: {str(e)}")
                        # 簡易的なデータフレーム作成
                        data_list = []
                        for item in items:
                            data_list.append({
                                'id': item.id,
                                'datetime': item.properties.get("datetime", ""),
                                'eo:cloud_cover': item.properties.get("eo:cloud_cover", 0),
                                'collection': collection_id
                            })
                        df = pd.DataFrame(data_list)
                    
                    # 結果の表示
                    st.subheader("📊 検索結果")
                    
                    # セッションステートに検索結果を保存
                    st.session_state.search_results = items
                    st.session_state.search_df = df
                    
                    # データ選択
                    st.subheader("🎯 データ選択")
                    
                    try:
                        # 表示用のデータを準備
                        display_items = []
                        
                        if satellite_type == "Sentinel-2 (光学)" and 'eo:cloud_cover' in df.columns:
                            # 雲量で並び替え（安全な方法）
                            try:
                                # pandas 2.x以降
                                df_sorted = df.sort_values("eo:cloud_cover", na_position='last')
                            except TypeError:
                                # pandas 1.x以前への後方互換性
                                df_sorted = df.sort_values("eo:cloud_cover").dropna(subset=["eo:cloud_cover"])
                            st.caption("💡 雲量が少ない順に表示しています")
                            
                            for idx, row in df_sorted.head(20).iterrows():
                                try:
                                    item = next((item for item in items if item.id == row['id']), None)
                                    if item:
                                        date_str = str(row['datetime'])[:10] if pd.notna(row['datetime']) else "不明"
                                        cloud_cover = row.get('eo:cloud_cover', None)
                                        
                                        if pd.notna(cloud_cover):
                                            display_text = f"📅 {date_str} ☁️ 雲量: {cloud_cover:.1f}%"
                                        else:
                                            display_text = f"📅 {date_str} ☁️ 雲量: 不明"
                                        
                                        display_items.append({
                                            'text': display_text,
                                            'item': item,
                                            'id': item.id
                                        })
                                except Exception as e:
                                    st.error(f"アイテム処理エラー: {str(e)}")
                                    
                        else:  # Sentinel-1 or fallback
                            # 日付で並び替え（新しい順）
                            if 'datetime' in df.columns:
                                try:
                                    # pandas 2.x以降
                                    df_sorted = df.sort_values("datetime", ascending=False, na_position='last')
                                except TypeError:
                                    # pandas 1.x以前への後方互換性
                                    df_sorted = df.sort_values("datetime", ascending=False).dropna(subset=["datetime"])
                                st.caption("💡 撮影日が新しい順に表示しています")
                                
                                for idx, row in df_sorted.head(20).iterrows():
                                    try:
                                        item = next((item for item in items if item.id == row['id']), None)
                                        if item:
                                            date_str = str(row['datetime'])[:10] if pd.notna(row['datetime']) else "不明"
                                            display_text = f"📅 {date_str} 🛰️ Sentinel-1"
                                            
                                            display_items.append({
                                                'text': display_text,
                                                'item': item,
                                                'id': item.id
                                            })
                                    except Exception as e:
                                        st.error(f"アイテム処理エラー: {str(e)}")
                            else:
                                # フォールバック：全アイテムを表示
                                for i, item in enumerate(items[:20]):
                                    display_items.append({
                                        'text': f"データ {i+1}: {item.id[:20]}...",
                                        'item': item,
                                        'id': item.id
                                    })
                        
                        # データ選択UI
                        if display_items:
                            option_texts = [item['text'] for item in display_items]
                            
                            # 初期選択のインデックス設定
                            default_idx = 0
                            if 'selected_item_id' in st.session_state and st.session_state.selected_item_id:
                                try:
                                    matching_idx = next((i for i, item in enumerate(display_items) 
                                                       if item['id'] == st.session_state.selected_item_id), 0)
                                    default_idx = matching_idx
                                except:
                                    pass
                            
                            selected_idx = st.selectbox(
                                "🎯 表示するデータを選択してください：",
                                options=range(len(option_texts)),
                                format_func=lambda x: option_texts[x],
                                index=default_idx,
                                help="リストから任意のデータを選択できます"
                            )
                            
                            selected_item = display_items[selected_idx]['item']
                            st.session_state.selected_item_id = selected_item.id
                            st.session_state.selected_item = selected_item  # セッション状態に保存
                            st.session_state.display_items = display_items  # display_itemsも保存
                            
                            # 選択情報を表示
                            st.success(f"✅ 選択済み: {option_texts[selected_idx]}")
                            
                        else:
                            st.warning("⚠️ 表示可能なデータがありません。")
                            selected_item = items[0] if items else None
                            
                    except Exception as e:
                        import traceback
                        st.error(f"❌ データ選択処理でエラーが発生しました:")
                        st.code(f"エラーメッセージ: {str(e)}")
                        st.code(f"詳細:\n{traceback.format_exc()}")
                        st.warning("⚠️ 最初のデータを使用します。")
                        selected_item = items[0] if items else None
                        if selected_item:
                            st.session_state.selected_item = selected_item
                            st.session_state.selected_item_id = selected_item.id
                        
        # データ選択UI（検索結果がある場合は常に表示）
        if st.session_state.search_results and st.session_state.display_items:
            st.subheader("🎯 データ選択")
            option_texts = [item['text'] for item in st.session_state.display_items]
            
            # 現在の選択インデックスを取得
            current_idx = 0
            if st.session_state.selected_item_id:
                try:
                    current_idx = next((i for i, item in enumerate(st.session_state.display_items) 
                                      if item['id'] == st.session_state.selected_item_id), 0)
                except:
                    pass
            
            selected_idx = st.selectbox(
                "🎯 表示するデータを選択してください：",
                options=range(len(option_texts)),
                format_func=lambda x: option_texts[x],
                index=current_idx,
                help="リストから任意のデータを選択できます",
                key="data_selector"
            )
            
            # 選択が変更された場合の処理
            if selected_idx is not None:
                selected_item = st.session_state.display_items[selected_idx]['item']
                st.session_state.selected_item = selected_item
                st.session_state.selected_item_id = selected_item.id
                st.success(f"✅ 選択済み: {option_texts[selected_idx]}")
        
        # プレビューの表示（セッション状態から）
        if st.session_state.selected_item:
            selected_item = st.session_state.selected_item
            st.subheader("🖼️ 選択されたデータのプレビュー")
            
            # 選択アイテムの要約を表示
            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("📅 撮影日", str(selected_item.properties.get('datetime', 'N/A'))[:10])
            with info_cols[1]:
                if satellite_type == "Sentinel-2 (光学)":
                    cloud_cover = selected_item.properties.get('eo:cloud_cover')
                    if cloud_cover is not None:
                        st.metric("☁️ 雲量", f"{cloud_cover:.1f}%")
                    else:
                        st.metric("☁️ 雲量", "N/A")
                else:
                    st.metric("🛰️ 衛星", "Sentinel-1")
            with info_cols[2]:
                st.metric("🆔 データID", selected_item.id[:15] + "...")
            
            # プレビュー画像
            if "rendered_preview" in selected_item.assets:
                st.image(
                    selected_item.assets["rendered_preview"].href, 
                    caption=f"衛星画像プレビュー - {selected_item.id}", 
                    width=600
                )
            else:
                st.info("プレビュー画像がありません。")
        
            # 詳細情報の表示
            st.subheader("📋 画像詳細情報")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**基本情報**")
                st.write(f"- **ID**: {selected_item.id}")
                st.write(f"- **撮影日時**: {selected_item.properties.get('datetime', 'N/A')}")
                st.write(f"- **コレクション**: {collection_id}")
                
                if satellite_type == "Sentinel-2 (光学)":
                    cloud_cover = selected_item.properties.get('eo:cloud_cover')
                    if cloud_cover is not None:
                        st.write(f"- **雲量**: {cloud_cover:.1f}%")
                    else:
                        st.write(f"- **雲量**: N/A")
            
            with col2:
                st.write("**利用可能なバンド/アセット**")
                asset_count = 0
                for asset_key, asset in selected_item.assets.items():
                    if asset.title:
                        st.write(f"- **{asset_key}**: {asset.title}")
                        asset_count += 1
                    elif asset_key not in ['rendered_preview', 'thumbnail']:
                        st.write(f"- **{asset_key}**: データファイル")
                        asset_count += 1
                
                if asset_count == 0:
                    st.write("利用可能なアセット情報がありません")
            
            # 追加プロパティの表示
            if len(selected_item.properties) > 3:
                with st.expander("🔍 詳細プロパティを表示"):
                    for key, value in selected_item.properties.items():
                        if key not in ['datetime']:
                            st.write(f"- **{key}**: {value}")
            
            # データダウンロード
            st.subheader("⬇️ データダウンロード")
            
            if satellite_type == "Sentinel-2 (光学)":
                st.info("Sentinel-2の詳細データ処理とダウンロード機能は今後のアップデートで実装予定です。")
                
                # RGB画像生成の例
                if st.button("RGB画像を生成"):
                    with st.spinner("画像を生成中..."):
                        try:
                            # RGB バンドを取得
                            bands = ["B04", "B03", "B02"]  # Red, Green, Blue
                            urls = [selected_item.assets[band].href for band in bands if band in selected_item.assets]
                            
                            if len(urls) == 3:
                                data = [rasterio.open(BytesIO(requests.get(url).content)) for url in urls]
                                
                                # 画像のメタデータを取得
                                crs = str(data[0].crs)
                                
                                # AOIに切り取り
                                cropped_data, _ = crop_image_to_aoi(data, st.session_state.area_of_interest, crs)
                                
                                # RGB画像として結合
                                rgb_image = np.concatenate((
                                    cropped_data[0],  # Red
                                    cropped_data[1],  # Green
                                    cropped_data[2]   # Blue
                                ), axis=0)
                                
                                # 表示用に調整
                                rgb_display = np.rollaxis(rgb_image, 0, 3)
                                rgb_normalized = rgb_display / 5000
                                
                                # 画像表示
                                fig, ax = plt.subplots(figsize=(10, 8))
                                ax.imshow(np.clip(rgb_normalized, 0, 1))
                                ax.set_title("RGB画像 (指定範囲)")
                                ax.axis('off')
                                st.pyplot(fig)
                                
                                # データのクリーンアップ
                                for d in data:
                                    d.close()
                                    
                            else:
                                st.error("RGB バンドが揃いません。")
                                
                        except Exception as e:
                            st.error(f"画像生成中にエラーが発生しました: {str(e)}")
                
            else:  # Sentinel-1
                st.info("Sentinel-1の詳細データ処理とダウンロード機能は今後のアップデートで実装予定です。")
                
                # VHバンドの例
                if "vh" in selected_item.assets and st.button("VH画像を生成"):
                    with st.spinner("画像を生成中..."):
                        try:
                            url = selected_item.assets["vh"].href
                            with rasterio.open(BytesIO(requests.get(url).content)) as src:
                                data = src.read(1)
                                
                                # 画像表示
                                fig, ax = plt.subplots(figsize=(10, 8))
                                im = ax.imshow(data, cmap='gray')
                                ax.set_title("VH画像")
                                ax.axis('off')
                                plt.colorbar(im, ax=ax, shrink=0.8)
                                st.pyplot(fig)
                                
                        except Exception as e:
                            st.error(f"画像生成中にエラーが発生しました: {str(e)}")
        
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.code(f"エラーの詳細:\n{type(e).__name__}: {str(e)}")
        with st.expander("詳細なエラー情報"):
            import traceback
            error_trace = traceback.format_exc()
            st.code(error_trace)
            
            # より詳細なデバッグ情報
            if hasattr(e, '__cause__') and e.__cause__:
                st.write("原因:", str(e.__cause__))
        
        # 解決策の提案
        st.info("💡 **解決策:**")
        st.write("- ブラウザのページを再読み込み（F5）してください")
        st.write("- 異なる場所や日時で再度検索してみてください")
        st.write("- 問題が続く場合は、範囲サイズを小さくしてみてください")

if __name__ == "__main__":
    main() 