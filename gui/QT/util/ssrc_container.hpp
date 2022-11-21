#ifndef SSRC_CONTAINER_HPP_1b2b8951b25f
#define SSRC_CONTAINER_HPP_1b2b8951b25f

#include <vector>

template<typename T, typename Ts_type>
class SSRC_container{
public:
	struct Holder{
		T item;
		uint32_t ssrc;
		Ts_type timestamp;
	};

	void insert(uint32_t ssrc, const T& item, Ts_type timestamp);
	const std::vector<Holder>& get() const { return items; }
	void remove_timed_out(Ts_type timeout, Ts_type now);
	void clear() { items.clear(); }

private:
	std::vector<Holder> items;
};

template<typename T, typename Ts_type>
inline void SSRC_container<T, Ts_type>::insert(uint32_t ssrc, const T& item, Ts_type timestamp){
	Holder *h = nullptr;
	for(auto& i : items){
		if(i.ssrc == ssrc){
			h = &i;
		}
	}

	if(!h){
		Holder newItem = {};
		newItem.ssrc = ssrc;
		items.push_back(newItem);
		h = &items.back();
	}

	h->item = item;
	h->timestamp = timestamp;
}

template<typename T, typename Ts_type>
inline void SSRC_container<T, Ts_type>::remove_timed_out(Ts_type timeout, Ts_type now){
	auto endIt = std::remove_if(items.begin(), items.end(),
			[now, timeout](const Holder& h){ return now - h.timestamp > timeout; });

	items.erase(endIt, items.end());
}

#endif //SSRC_CONTAINER_HPP_1b2b8951b25f
