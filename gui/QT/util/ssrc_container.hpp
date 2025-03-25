#ifndef SSRC_CONTAINER_HPP_1b2b8951b25f
#define SSRC_CONTAINER_HPP_1b2b8951b25f

#include <vector>

template<typename Key_type, typename Val_type, typename Ts_type>
class SSRC_container{
public:
	struct Holder{
		Key_type key;
		Val_type item;
		Ts_type timestamp;
	};

	void insert(const Key_type& key, const Val_type& item, Ts_type timestamp);
	const std::vector<Holder>& get() const { return items; }
	void remove_timed_out(Ts_type timeout, Ts_type now);
	void clear() { items.clear(); }

private:
	std::vector<Holder> items;
};

template<typename Key_type, typename Val_type, typename Ts_type>
inline void SSRC_container<Key_type, Val_type, Ts_type>::insert(const Key_type& key, const Val_type& item, Ts_type timestamp){
	Holder *h = nullptr;
	for(auto& i : items){
		if(i.key == key){
			h = &i;
		}
	}

	if(!h){
		Holder newItem = {};
		newItem.key = key;
		items.push_back(newItem);
		h = &items.back();
	}

	h->item = item;
	h->timestamp = timestamp;
}

template<typename Key_type, typename Val_type, typename Ts_type>
inline void SSRC_container<Key_type, Val_type, Ts_type>::remove_timed_out(Ts_type timeout, Ts_type now){
	auto endIt = std::remove_if(items.begin(), items.end(),
			[now, timeout](const Holder& h){ return now - h.timestamp > timeout; });

	items.erase(endIt, items.end());
}

#endif //SSRC_CONTAINER_HPP_1b2b8951b25f
